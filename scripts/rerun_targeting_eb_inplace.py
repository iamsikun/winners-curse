"""Refit targeting models and replace only empirical Bayes results in existing runs."""

import os

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/winners-curse-matplotlib')

import argparse
import copy
import hashlib
import itertools
import json
import pickle
import shutil
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]

import numpy as np
import yaml
from joblib import Parallel, delayed

from targeting_experiment import create_targeting_config_loader
from winners_curse.bayes_methods import eb_function_dict
from winners_curse.dgp import Targeting
from winners_curse.experiments import (
    create_result_key, get_parameter_lists, identify_varying_params, prepare_experiment_params,
)
from winners_curse.targeting import obj_func, optimize, replace_outliers

RUNS = {
    'targeting_correct_snr': 'targeting_correct_snr_20260920_032752',
    'targeting_depth_compare': 'targeting_depth_compare_20260920_033004',
    'targeting_forest_snr': 'targeting_forest_snr_20260923_160111',
    'targeting_forest_functional_form': 'targeting_forest_functional_form_20260923_131000',
    'targeting_cf_bernoulli': 'targeting_cf_bernoulli_20260919_214621',
}
OLD_PREFIX = 'eb_normal'
NEW_PREFIX = 'eb_tweedies'


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def atomic_pickle(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp, path)


def non_eb_digest(results):
    """Hash every stored value outside the two empirical Bayes namespaces."""
    digest = hashlib.sha256()
    for key, values in results.items():
        digest.update(pickle.dumps(key, protocol=5))
        for field, value in values.items():
            if field.startswith((OLD_PREFIX + '_', NEW_PREFIX + '_')):
                continue
            digest.update(pickle.dumps((field, value), protocol=5))
    return digest.hexdigest()


def selection_digest(selection):
    return hashlib.sha256(np.asarray(selection, dtype=np.int64).tobytes()).hexdigest()


def replace_eb_arrays(results, records):
    """Preserve repeat alignment in diagnostics and existing filtering in summary arrays."""
    updated = copy.deepcopy(results)
    for key, rows in records.items():
        values = updated[key]
        truth = np.asarray(values['nc_val_true_arr'])
        rows = [rows[i] for i in range(len(truth))]
        estimates = np.asarray([row['value'] for row in rows], dtype=float)
        errors = estimates - truth
        valid = np.isfinite(errors) & (errors > -5.) & (errors < 5.)
        for field in list(values):
            if field.startswith((OLD_PREFIX + '_', NEW_PREFIX + '_')):
                del values[field]
        values.update({
            NEW_PREFIX + '_val_est_arr': estimates[valid],
            NEW_PREFIX + '_wc_arr': errors[valid],
            NEW_PREFIX + '_val_est_full_arr': estimates,
            NEW_PREFIX + '_wc_full_arr': errors,
            NEW_PREFIX + '_valid_mask_arr': valid,
            NEW_PREFIX + '_fit_failed_arr': np.asarray([row['failure'] is not None for row in rows]),
        })
    if non_eb_digest(updated) != non_eb_digest(results):
        raise AssertionError('A non-EB result changed')
    return updated


def update_config_text(text):
    """Change the EB selector while retaining comments and unrelated config formatting."""
    lines = text.splitlines(keepends=True)
    inside = False
    for i, line in enumerate(lines):
        if line.startswith('  eb_normal:') or line.startswith('  eb_tweedies:'):
            inside = True
            lines[i] = line.replace('eb_normal:', 'eb_tweedies:')
        elif inside and line.strip() and not line.lstrip().startswith('#'):
            if len(line) - len(line.lstrip()) <= 2:
                inside = False
            elif line.strip() == 'prior: normal':
                lines[i] = line.replace('prior: normal', 'prior: tweedies')
    result = ''.join(lines)
    parsed = yaml.safe_load(result)
    assert parsed['estimators_dict'][NEW_PREFIX]['params']['prior'] == 'tweedies'
    return result


def fit_repeat(repeat, dgp_params, data_params, experiment_params, spline_params):
    """Reproduce the original model fit once and evaluate both EB rules on its effects."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dgp = Targeting(**dgp_params)
        x, treatment, outcome = dgp.sample(data_params['sample_size'], seed=repeat)
        target_x = dgp.sample_individuals(data_params['targ_sample_size'], seed=repeat + 10000)
        model_config = experiment_params['estimator']
        model = model_config['estimator'](**model_config['params']).fit(X=x, Y=outcome, T=treatment)
        effects, variances = model.predict_incremental_effect(target_x)
        effects = replace_outliers(effects, threshold=experiment_params.get('outlier_threshold', 20.))
        selection, naive = optimize(None, target_x, effects)
        truth = obj_func(selection, target_x, demand_model=dgp)
        normal = np.column_stack([
            eb_function_dict['normal'](effects[:, arm], variances[:, arm])
            for arm in range(effects.shape[1])
        ])
        normal_value = obj_func(selection, target_x, cust_treatment_effects=normal)
        failure = None
        try:
            posterior = np.column_stack([
                eb_function_dict['tweedies'](effects[:, arm], variances[:, arm], **spline_params)
                for arm in range(effects.shape[1])
            ])
            if not np.isfinite(posterior).all():
                raise ValueError('Non-finite posterior effects')
            value = obj_func(selection, target_x, cust_treatment_effects=posterior)
        except Exception as exc:
            value = float('nan')
            failure = f'{type(exc).__name__}: {exc}'
    return {
        'repeat': repeat, 'naive': naive, 'truth': truth, 'normal': normal_value,
        'selection_sha256': selection_digest(selection), 'value': value, 'failure': failure,
    }


def validate_records(saved, records, check_selections=False):
    maxima = {'naive': 0., 'truth': 0., 'normal': 0.}
    for key, rows in records.items():
        n = len(saved[key]['nc_wc_arr'])
        assert set(rows) == set(range(n)), f'Incomplete repetitions for {key}'
        for metric, field in [('naive', 'nc_val_est_arr'), ('truth', 'nc_val_true_arr'), ('normal', OLD_PREFIX + '_val_est_arr')]:
            # Existing normal summary arrays can exclude outlier repetitions.
            if field not in saved[key] or len(saved[key][field]) != n:
                continue
            actual = np.asarray([rows[i][metric] for i in range(n)])
            expected = np.asarray(saved[key][field])
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
            maxima[metric] = max(maxima[metric], float(np.max(np.abs(actual - expected))))
        if check_selections and 'nc_selection_arr' in saved[key]:
            for i, selection in enumerate(saved[key]['nc_selection_arr']):
                assert rows[i]['selection_sha256'] == selection_digest(selection), f'Selection mismatch {key}, {i}'
    return maxima


def make_plan(config, saved):
    taus, depths, sizes, noises, functions = get_parameter_lists(config)
    varying = identify_varying_params(taus, depths, sizes, noises, functions)
    plan = []
    for depth, size, function, tau in itertools.product(depths, sizes, functions, taus):
        key = create_result_key(depth, size, tau, None, function, varying)
        if key in saved:
            params = prepare_experiment_params(config, tau, depth, size, char_func=function)
            plan.append((key, params))
    assert {key for key, _ in plan} == set(saved), 'Configuration does not cover all saved settings'
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=16)
    parser.add_argument('--backup-root', type=Path, required=True)
    parser.add_argument('--runs', nargs='+', choices=list(RUNS), default=list(RUNS))
    args = parser.parse_args()
    status_path = ROOT / 'results/targeting_eb_rerun_status.json'
    started = time.time()
    state = {'pid': os.getpid(), 'phase': 'running', 'started_at': datetime.now(timezone.utc).isoformat(), 'jobs': args.jobs, 'completed_runs': [], 'completed_repetitions': 0, 'planned_repetitions': 0, 'failures': 0, 'summaries': []}
    prepared = []
    args.backup_root.mkdir(parents=True, exist_ok=True)
    for name in args.runs:
        run_dir = ROOT / 'results' / RUNS[name]
        source_config = ROOT / 'configs' / (name + '.yaml')
        config = create_targeting_config_loader()(str(source_config))
        with (run_dir / 'results_slim.pkl').open('rb') as handle:
            saved = pickle.load(handle)
        plan = make_plan(config, saved)
        state['planned_repetitions'] += sum(len(values['nc_wc_arr']) for values in saved.values())
        prepared.append((name, run_dir, source_config, config, saved, plan))
        backup = args.backup_root / run_dir.name
        backup.mkdir(exist_ok=True)
        for path in [source_config, run_dir / 'config.yaml', run_dir / 'config.json', *run_dir.glob('results*.pkl')]:
            destination = backup / ('source_config.yaml' if path == source_config else path.name)
            if not destination.exists():
                shutil.copy2(path, destination)
    atomic_json(status_path, state)
    try:
        for name, run_dir, source_config, config, saved, plan in prepared:
            state['current_run'] = name
            checkpoint_path = run_dir / 'eb_tweedies_checkpoint.pkl'
            if checkpoint_path.exists():
                with checkpoint_path.open('rb') as handle:
                    records = pickle.load(handle)
            else:
                records = {}
            method_config = config['estimators_dict'].get(OLD_PREFIX, config['estimators_dict'].get(NEW_PREFIX))
            spline_params = {key: method_config['params'][key] for key in ('dof', 'bin_width')}
            for key, (dp, da, ep) in plan:
                rows = records.setdefault(key, {})
                n = len(saved[key]['nc_wc_arr'])
                state.update(current_setting=repr(key), setting_completed=len(rows), setting_total=n)
                state['completed_repetitions'] += len(rows)
                state['failures'] += sum(row['failure'] is not None for row in rows.values())
                atomic_json(status_path, state)
                remaining = [i for i in range(n) if i not in rows]
                with Parallel(n_jobs=args.jobs, return_as='generator_unordered', batch_size=1) as parallel:
                    for row in parallel(delayed(fit_repeat)(i, dp, da, ep, spline_params) for i in remaining):
                        i = row['repeat']
                        # Reject mismatched model reconstruction before replacing any run file.
                        np.testing.assert_allclose([row['naive'], row['truth']], [saved[key]['nc_val_est_arr'][i], saved[key]['nc_val_true_arr'][i]], rtol=1e-10, atol=1e-10)
                        rows[i] = row
                        state['completed_repetitions'] += 1
                        state['failures'] += row['failure'] is not None
                        state['setting_completed'] = len(rows)
                        state['elapsed_seconds'] = round(time.time() - started, 1)
                        if len(rows) % 50 == 0 or len(rows) == n:
                            atomic_pickle(checkpoint_path, records)
                            atomic_json(status_path, state)
                            print(f'{name} {key}: {len(rows)}/{n}; total {state["completed_repetitions"]}/{state["planned_repetitions"]}; spline failures {state["failures"]}', flush=True)
            validation = validate_records(saved, records)
            provenance = {'method': 'tweedies', 'params': spline_params, 'completed_at': datetime.now(timezone.utc).isoformat(), 'backup_directory': str(args.backup_root / run_dir.name), 'reconstruction_max_abs_difference': validation, 'settings': [], 'files': {}}
            for key, rows in records.items():
                truth = saved[key]['nc_val_true_arr']
                errors = np.array([rows[i]['value'] - truth[i] for i in range(len(truth))])
                valid = np.isfinite(errors) & (errors > -5.) & (errors < 5.)
                failures = {str(i): row['failure'] for i, row in rows.items() if row['failure'] is not None}
                provenance['settings'].append({'key': repr(key), 'attempted': len(rows), 'fit_failures': len(failures), 'finite_outliers': int(np.sum(np.isfinite(errors) & ~valid)), 'retained': int(valid.sum()), 'mean_wc': float(errors[valid].mean()) if valid.any() else None, 'failure_types': dict(Counter(failures.values())), 'failed_repetitions': failures})
            for path in sorted(run_dir.glob('results*.pkl')):
                with path.open('rb') as handle:
                    original = pickle.load(handle)
                validate_records(original, records, check_selections=True)
                before = non_eb_digest(original)
                original_hash = digest_file(path)
                updated = replace_eb_arrays(original, records)
                atomic_pickle(path, updated)
                del original, updated
                with path.open('rb') as handle:
                    verified = pickle.load(handle)
                assert non_eb_digest(verified) == before
                provenance['files'][path.name] = {'before_sha256': original_hash, 'after_sha256': digest_file(path), 'non_eb_sha256': before}
                del verified
            for path in [source_config, run_dir / 'config.yaml']:
                temp = path.with_name(path.name + '.tmp')
                temp.write_text(update_config_text(path.read_text()))
                os.replace(temp, path)
            json_path = run_dir / 'config.json'
            json_config = json.loads(json_path.read_text())
            old = json_config['estimators_dict']
            json_config['estimators_dict'] = {NEW_PREFIX if key == OLD_PREFIX else key: value for key, value in old.items()}
            json_config['estimators_dict'][NEW_PREFIX]['params']['prior'] = 'tweedies'
            json_config['empirical_bayes_rerun'] = {'method': 'tweedies', 'diagnostics_file': 'eb_tweedies_rerun.json', 'repeat_alignment': 'full arrays and valid mask saved; summary arrays retain finite errors in (-5, 5)'}
            atomic_json(json_path, json_config)
            atomic_json(run_dir / 'eb_tweedies_rerun.json', provenance)
            state['completed_runs'].append(name)
            state['summaries'].append({'run': name, 'settings': [{key: value for key, value in setting.items() if key not in ('failed_repetitions', 'failure_types')} for setting in provenance['settings']]})
            atomic_json(status_path, state)
        state['phase'] = 'complete'
        state['elapsed_seconds'] = round(time.time() - started, 1)
        atomic_json(status_path, state)
    except BaseException as exc:
        state['phase'] = 'failed'
        state['error'] = f'{type(exc).__name__}: {exc}'
        atomic_json(status_path, state)
        raise


if __name__ == '__main__':
    main()
