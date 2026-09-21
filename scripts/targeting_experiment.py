import os

# Fix for Windows multiprocessing + OpenBLAS issues
# CRITICAL: Must be set BEFORE numpy/scipy is imported
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import pickle
import argparse
import copy
import multiprocessing
from pathlib import Path

# Ensure the package is importable (workaround for uv run .pth file issue)
# Add src/ to Python path if not already there
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
_src_path = _project_root / 'src'
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# Import reusable experiment utilities
from winners_curse.experiments import (
    ExperimentProgress,
    load_config,
    create_output_directory,
    setup_logging,
    save_config,
    save_results,
    log_key_parameters,
    get_max_jobs,
    process_noise_vars,
    get_config_path,
    get_parameter_lists,
    identify_varying_params,
    create_result_key,
    log_sweep_configuration,
    prepare_experiment_params,
)
from joblib.externals.loky import get_reusable_executor

import warnings
# Suppress LightGBM warnings about feature names (benign in this context)
warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBM")

from sklearn.linear_model import LogisticRegression, Lasso

from winners_curse.targeting import (
    optimize,
    CausalForestDML,
    KnownFunctionalForm,
    repeated_experiment,
    calculate_winners_curse_measures,
    bootstrap_correction_estimate,
    sample_splitting_estimate,
    empirical_bayes_estimate,
    selective_inference_estimate,
)
from winners_curse.variables import PointMass


def create_targeting_config_loader():
    """
    Create a config loader with targeting-specific mappings.
    
    Returns:
        Function that loads config with targeting-specific parsers
    """
    optimizer_map = {
        'optimize': optimize,
    }
    
    estimator_class_map = {
        'CausalForestDML': CausalForestDML,
        'KnownFunctionalForm': KnownFunctionalForm,
    }
    
    estimator_function_map = {
        'bootstrap_correction_estimate': bootstrap_correction_estimate,
        'sample_splitting_estimate': sample_splitting_estimate,
        'empirical_bayes_estimate': empirical_bayes_estimate,
        'selective_inference_estimate': selective_inference_estimate,
    }
    
    model_classes = {
        'LogisticRegression': LogisticRegression,
        'Lasso': Lasso,
    }
    
    def load_targeting_config(config_path: str):
        return load_config(
            config_path,
            optimizer_map=optimizer_map,
            estimator_class_map=estimator_class_map,
            estimator_function_map=estimator_function_map,
            model_classes=model_classes
        )
    
    return load_targeting_config


def run_experiment(config: dict, logger, output_dir=None, resume_results: dict = None,
                   model_name: str = None, progress: ExperimentProgress = None) -> dict:
    """
    Run the experiment with the given configuration.

    Supports multi-parameter sweeps with smart result key structure:
    - 1 param varies: keys are just that parameter (e.g., tau tuples only)
    - 2 params vary: keys are (param1, param2) tuples
    - 3+ params vary: ERROR (maximum 2 parameters can be tested simultaneously)

    Args:
        config: Configuration dictionary
        logger: Logger instance
        output_dir: If provided, the accumulating results dict is checkpointed to
            ``output_dir/results.pkl`` after every combo (restart-resilient).
        resume_results: If provided, combos whose result_key is already present are
            skipped, and new results are appended (resume after an interruption).
        model_name: Internal prefix for results from a named model sweep.
        progress: Shared progress counter across models in the same config.

    Returns:
        Dictionary of results with smart keys based on varying parameters
    """
    # A named model (including its depth) is one comparative-statics dimension.
    # Reuse the regular runner so checkpoint/resume still works after every combo.
    comp_statics = config.get('comparative_statics', {})
    if 'model_list' in comp_statics:
        models = comp_statics['model_list']
        names = [model['name'] for model in models]
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError('model_list must contain nonempty model names')
        if len(set(names)) != len(names):
            raise ValueError('model_list names must be unique')
        if 'depth_list' in comp_statics:
            raise ValueError('Set max_depth inside each model_list entry, not depth_list')
        parameter_lists = get_parameter_lists(config)
        varying_params = identify_varying_params(*parameter_lists)
        if len(varying_params) + (len(models) > 1) > 2:
            raise ValueError('Cannot test more than 2 parameters simultaneously, including model_list')

        tau_list, _, sample_sizes, _, char_funcs = parameter_lists
        total_combos = len(models) * len(tau_list) * len(sample_sizes) * len(char_funcs)
        progress = ExperimentProgress(logger, total_combos)
        results = dict(resume_results) if resume_results else {}
        for model_index, model in enumerate(models, 1):
            model_config = copy.deepcopy(config)
            del model_config['comparative_statics']['model_list']
            model_config['experiment_params']['estimator'] = {
                key: copy.deepcopy(value) for key, value in model.items() if key != 'name'
            }
            logger.info(f"[model {model_index}/{len(models)}] Running model: {model['name']}")
            results = run_experiment(
                model_config, logger, output_dir=output_dir,
                resume_results=results, model_name=model['name'], progress=progress,
            )
        return results

    # Extract configuration sections
    optimization_params = config['optimization_params']
    estimators_dict = config['estimators_dict']
    max_jobs = get_max_jobs(config)
    
    # Get parameter lists
    tau_list, depth_list, sample_size_list, noise_vars_list, char_func_list = get_parameter_lists(config)
    
    # Ignore noise_vars_list for targeting experiments (only used in ab_test)
    _ = noise_vars_list
    
    # Identify varying parameters and validate
    varying_params = identify_varying_params(tau_list, depth_list, sample_size_list, noise_vars_list, char_func_list)
    if len(varying_params) > 2:
        raise ValueError(
            f"Cannot test more than 2 parameters simultaneously. "
            f"Currently varying: {varying_params} "
            f"(tau: {len(tau_list)}, depth: {len(depth_list)}, sample_size: {len(sample_size_list)}, char_func: {len(char_func_list)}). "
            f"\nPlease reduce to at most 2 parameter lists with multiple values."
        )
    
    # Log configuration
    total_combos = len(depth_list) * len(sample_size_list) * len(tau_list) * len(char_func_list)
    if progress is None:
        progress = ExperimentProgress(logger, total_combos)
    log_sweep_configuration(logger, tau_list, depth_list, sample_size_list, noise_vars_list, char_func_list,
                            varying_params, total_combos, max_jobs)
    
    # Run experiments for all parameter combinations
    results = dict(resume_results) if resume_results else {}
    if results:
        logger.info(f"[resume] Starting with {len(results)} pre-computed combo(s); they will be skipped.")
    
    for depth in depth_list:
        for sample_size in sample_size_list:
            for char_func in char_func_list:
                for tau in tau_list:

                    # Compute the smart key up front so completed combos can be skipped on resume
                    result_key = create_result_key(depth, sample_size, tau, None, char_func, varying_params)
                    if model_name is not None:
                        result_key = (model_name, result_key)
                    if result_key in results:
                        progress.finish(result_key, skipped=True)
                        continue
                    progress.start(result_key)

                    # Log current combination
                    logger.info("=" * 80)
                    logger.info("Combination parameters:")
                    logger.info(f"  max_depth = {depth}")
                    logger.info(f"  sample_size = {sample_size}")
                    logger.info(f"  char_func = {char_func}")
                    logger.info(f"  tau = {tau}")
                    logger.info("=" * 80)

                    # Prepare parameters for this combination
                    dgp_params, data_params, experiment_params = prepare_experiment_params(
                        config, tau, depth, sample_size, char_func=char_func, rename_noise_var=True
                    )

                    # joblib keeps its worker pool alive between Parallel calls, so the
                    # previous combo's workers are still resident here and would make this
                    # combo look short on memory. Retire them before sizing this one.
                    get_reusable_executor().shutdown(wait=True)

                    # Large samples hold the whole dataset in each worker, so the
                    # job count is re-derived per combo from the memory tiers.
                    combo_jobs = get_max_jobs(config, sample_size=sample_size, logger=logger)

                    # Run experiment
                    logger.info(f"Running {experiment_params['n_repeats']} repetitions (n_jobs={combo_jobs})")
                    result_records = repeated_experiment(
                        optimization_params=optimization_params,
                        dgp_params=dgp_params,
                        data_params=data_params,
                        experiment_params=experiment_params,
                        estimators_dict=estimators_dict,
                        n_jobs=combo_jobs,
                        verbose=True,
                        logger=logger
                    )

                    # Store results with smart key
                    results[result_key] = calculate_winners_curse_measures(
                        result_records=result_records,
                        optimization_params=optimization_params,
                        estimators_dict=estimators_dict,
                        data_params=data_params
                    )

                    # Checkpoint after each combo so an interruption (sleep/restart) loses at most one combo
                    if output_dir is not None:
                        _ckpt = os.path.join(output_dir, 'results.pkl')
                        with open(_ckpt, 'wb') as _f:
                            pickle.dump(results, _f)
                        logger.info(f"[checkpoint] Saved {len(results)} combo(s) -> {_ckpt}")
                    progress.finish(result_key)
    
    logger.info("=" * 80)
    logger.info(f"All {total_combos} parameter combinations completed successfully")
    logger.info("=" * 80)
    
    return results


def main():
    """Main function to run the experiment."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run targeting causal forest experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Use default config (targeting_correct_snr.yaml)
        python scripts/targeting_experiment.py
        
        # Use a specific config file from configs/ directory
        python scripts/targeting_experiment.py --config targeting_forest_moon_power.yaml
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='targeting_correct_snr.yaml',
        help='Path to YAML config file (default: targeting_correct_snr.yaml). '
            'Can be a filename in configs/ directory, relative path, or absolute path.'
    )
    parser.add_argument(
        '--estimators',
        type=str,
        nargs='+',
        default=None,
        help='If set, run only these estimator keys from estimators_dict (e.g. '
             '--estimators mn_bootstrap). Output directory is tagged with '
             '_estsubset so results can be merged into a full run later.'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to a prior output directory whose results.pkl holds completed '
             'parameter combos. Those combos are skipped and remaining ones are '
             'appended into a fresh output dir (restart/sleep resilient).'
    )

    args = parser.parse_args()
    
    # Determine config path
    try:
        config_path = get_config_path(__file__, args.config)
    except FileNotFoundError:
        # Fallback to parsing as absolute/relative path if not found in configs/ via helper
        candidate_path = Path(args.config)
        if candidate_path.exists():
            config_path = candidate_path.resolve()
        else:
            # If get_config_path failed and it's not a direct path, re-raise
            raise
    
    # (1) Load config
    load_targeting_config = create_targeting_config_loader()
    config = load_targeting_config(str(config_path))

    # (1b) Optionally subset estimators_dict for partial reruns
    if args.estimators:
        missing = [k for k in args.estimators if k not in config['estimators_dict']]
        if missing:
            raise SystemExit(
                f"--estimators names not found in config's estimators_dict: {missing}. "
                f"Available: {list(config['estimators_dict'].keys())}"
            )
        config['estimators_dict'] = {
            k: config['estimators_dict'][k] for k in args.estimators
        }

    # (1c) Optionally resume from a prior (partial) run by skipping completed combos
    resume_results = None
    if args.resume:
        resume_pkl = Path(args.resume) / 'results.pkl'
        if resume_pkl.exists():
            with open(resume_pkl, 'rb') as f:
                resume_results = pickle.load(f)
            print(f"[resume] Loaded {len(resume_results)} completed combo(s) from {resume_pkl}")
        else:
            print(f"[resume] No results.pkl found in {args.resume}; starting fresh")

    # (2) Generate timestamped output folder
    output_base = config.get('output', {}).get('results_dir', 'results')
    # Derive experiment name from config filename (without extension)
    experiment_name = config_path.stem
    if args.estimators:
        experiment_name = f'{experiment_name}_estsubset'
    output_dir = create_output_directory(output_base, experiment_name=experiment_name)
    
    # (3) Log parameters
    logger = setup_logging(output_dir, logger_name=f'{experiment_name}_experiment')
    logger.info("=" * 80)
    logger.info("Starting Targeting Experiment")
    logger.info("=" * 80)
    logger.info(f"Config file: {config_path}")

    save_config(config, output_dir, logger)
    log_key_parameters(config, logger)
    
    # (4) Run the experiment
    logger.info("=" * 80)
    logger.info("Running experiment...")
    logger.info("=" * 80)
    
    try:
        tau_result_dict = run_experiment(config, logger, output_dir=output_dir, resume_results=resume_results)
        
        # Note: Result keys are now "smart" based on which parameters vary:
        # - Only tau varies: keys are tau tuples, e.g., (1, 1.01) [BACKWARD COMPATIBLE]
        # - depth + tau vary: keys are (depth, tau), e.g., (5, (1, 1.01))
        # - sample_size + tau vary: keys are (sample_size, tau), e.g., (2500, (1, 1.01))
        # - depth + sample_size vary: keys are (depth, sample_size)
        # - 3+ parameters vary: ERROR (max 2 parameters allowed)
        
        # (5) Save artifacts/metrics/results
        logger.info("=" * 80)
        logger.info("Saving results...")
        logger.info("=" * 80)
        
        save_results(tau_result_dict, output_dir, config, logger)
        
        logger.info("=" * 80)
        logger.info("Experiment completed successfully!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}", exc_info=True)
        raise

    logger.info(f"Results saved to: {output_dir}")

if __name__ == '__main__':
    main()
