import copy
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts import ab_test_experiment, targeting_experiment
from winners_curse.experiments import make_config_serializable


ROOT = Path(__file__).resolve().parents[1]


def load_targeting_sweep():
    return targeting_experiment.create_targeting_config_loader()(
        str(ROOT / 'configs/targeting_forest_moon_power.yaml')
    )


def test_named_model_sweep_routes_models_and_resumes_each_combo(monkeypatch, tmp_path, caplog):
    config = load_targeting_sweep()
    calls = []
    caplog.set_level(logging.INFO, logger=__name__)

    def repeated_experiment(**kwargs):
        estimator = kwargs['experiment_params']['estimator']
        calls.append((
            estimator['estimator'].__name__,
            estimator['params'].get('max_depth'),
            tuple(var.value for var in kwargs['dgp_params']['base_effect_vars']),
        ))
        return [{'nc_wc_arr': np.array([0.01])}]

    monkeypatch.setattr(targeting_experiment, 'repeated_experiment', repeated_experiment)
    monkeypatch.setattr(
        targeting_experiment, 'calculate_winners_curse_measures',
        lambda result_records, **kwargs: result_records[0],
    )
    completed_key = ('causal_forest_depth_5', (1, 1.01))
    completed_value = {'nc_wc_arr': np.array([0.123])}
    results = targeting_experiment.run_experiment(
        config, logging.getLogger(__name__), output_dir=tmp_path,
        resume_results={completed_key: completed_value},
    )

    assert len(results) == 12
    assert len(calls) == 11
    assert ('CausalForestDML', 5, (1, 1.01)) not in calls
    assert {call[:2] for call in calls} == {
        ('KnownFunctionalForm', None), ('CausalForestDML', 2),
        ('CausalForestDML', 5), ('CausalForestDML', 10),
    }
    assert results[completed_key] is completed_value
    with (tmp_path / 'results.pkl').open('rb') as handle:
        assert pickle.load(handle).keys() == results.keys()
    assert '[model 4/4]' in caplog.text
    assert '[combination 12/12] 100.0% complete' in caplog.text
    assert 'already saved; skipped' in caplog.text
    assert 'experiment elapsed' in caplog.text


def test_model_sweep_enforces_two_dimensions_and_unique_names():
    config = load_targeting_sweep()
    config['comparative_statics']['sample_size_list'] = [100, 200]
    with pytest.raises(ValueError, match='more than 2'):
        targeting_experiment.run_experiment(config, logging.getLogger(__name__))

    config = load_targeting_sweep()
    models = config['comparative_statics']['model_list']
    models[1]['name'] = models[0]['name']
    with pytest.raises(ValueError, match='unique'):
        targeting_experiment.run_experiment(config, logging.getLogger(__name__))


def test_model_sweep_metadata_serializes_model_classes_and_nuisance_models():
    config = make_config_serializable(load_targeting_sweep())
    serialized = json.loads(json.dumps(config))
    models = serialized['comparative_statics']['model_list']
    assert models[0]['estimator'] == 'KnownFunctionalForm'
    assert models[1]['estimator'] == 'CausalForestDML'
    assert models[1]['params']['model_t']['class'] == 'LogisticRegression'
    assert models[1]['params']['model_y']['class'] == 'Lasso'
    assert serialized['data_params']['targ_sample_size'] == 10000


def test_imbalanced_sweep_contains_no_correction_and_correction_at_every_allocation():
    config = ab_test_experiment.create_ab_test_config_loader()(
        str(ROOT / 'configs/ab_test_imbalanced_snr.yaml')
    )
    config = copy.deepcopy(config)
    config['experiment_params']['n_repeats'] = 2
    config['parallel']['max_jobs'] = 1
    config['estimators_dict'] = {'moon_bootstrap': config['estimators_dict']['moon_bootstrap']}
    config['estimators_dict']['moon_bootstrap']['params']['n_bootstraps'] = 2
    results = ab_test_experiment.run_experiment(config, logging.getLogger(__name__))
    assert len(results) == 27
    for allocation in config['comparative_statics']['sample_size_list']:
        for tau in config['comparative_statics']['tau_list']:
            result = results[(tuple(allocation), tuple(tau))]
            assert result['nc_wc_arr'].shape == (2,)
            assert result['moon_bootstrap_wc_arr'].shape == (2,)
            assert np.isfinite(result['moon_bootstrap_wc_arr']).all()


def test_all_simulation_configs_load_with_supported_bootstrap_methods():
    loaders = {
        'ab_test': ab_test_experiment.create_ab_test_config_loader(),
        'targeting': targeting_experiment.create_targeting_config_loader(),
    }
    for prefix, loader in loaders.items():
        for path in (ROOT / 'configs').glob(f'{prefix}_*.yaml'):
            config = loader(str(path))
            json.dumps(make_config_serializable(config))
            for estimator in config['estimators_dict'].values():
                method = estimator.get('params', {}).get('bootstrap_method', 'standard')
                assert method in {'standard', 'moon', 'conditional'}, path.name

    config = yaml.safe_load((ROOT / 'configs/targeting_forest_moon_power.yaml').read_text())
    powers = [entry['params']['power'] for entry in config['estimators_dict'].values()
              if entry['params']['bootstrap_method'] == 'moon']
    assert powers == [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
