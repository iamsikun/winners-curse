import os

# Fix for Windows multiprocessing + OpenBLAS issues
# CRITICAL: Must be set BEFORE numpy/scipy is imported
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import argparse
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
from winners_curse.ab_test import (
    select_higher_effect,
    repeated_experiment,
    calculate_winners_curse_measures,
    bootstrap_correction_estimate,
    empirical_bayes_estimate,
    bayes_estimate,
    selective_inference_estimate,
    sample_splitting_estimate,
    jackknife_estimate,
    plugin_correction_estimate,
    kfold_cv_estimate,
)
from winners_curse.variables import PointMass


def create_ab_test_config_loader():
    """
    Create a config loader with A/B test-specific mappings.
    
    Returns:
        Function that loads config with A/B test-specific parsers
    """
    optimizer_map = {
        'select_higher_effect': select_higher_effect,
    }
    
    estimator_function_map = {
        'bootstrap_correction_estimate': bootstrap_correction_estimate,
        'empirical_bayes_estimate': empirical_bayes_estimate,
        'bayes_estimate': bayes_estimate,
        'selective_inference_estimate': selective_inference_estimate,
        'sample_splitting_estimate': sample_splitting_estimate,
        'jackknife_estimate': jackknife_estimate,
        'plugin_correction_estimate': plugin_correction_estimate,
        'kfold_cv_estimate': kfold_cv_estimate,
    }
    
    def load_ab_test_config(config_path: str):
        return load_config(
            config_path,
            optimizer_map=optimizer_map,
            estimator_function_map=estimator_function_map,
        )
    
    return load_ab_test_config


def run_experiment(config: dict, logger) -> dict:
    """
    Run the experiment with the given configuration.
    
    Supports multi-parameter sweeps with smart result key structure:
    - 1 param varies: keys are just that parameter (e.g., tau tuples only)
    - 2 params vary: keys are (param1, param2) tuples
    - 3+ params vary: ERROR (maximum 2 parameters can be tested simultaneously)
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Dictionary of results with smart keys based on varying parameters
    """
    # Extract configuration sections
    optimization_params = config['optimization_params']
    estimators_dict = config['estimators_dict']
    max_jobs = get_max_jobs(config)
    
    # Get parameter lists
    tau_list, depth_list, sample_size_list, noise_vars_list, char_func_list = get_parameter_lists(config)

    # Ignore char_func_list for ab_test experiments (only used in targeting)
    _ = char_func_list

    # Identify varying parameters and validate
    varying_params = identify_varying_params(tau_list, depth_list, sample_size_list, noise_vars_list, char_func_list)
    if len(varying_params) > 2:
        raise ValueError(
            f"Cannot test more than 2 parameters simultaneously. "
            f"Currently varying: {varying_params} "
            f"(tau: {len(tau_list)}, depth: {len(depth_list)}, sample_size: {len(sample_size_list)}, noise_vars: {len(noise_vars_list)}). "
            f"\nPlease reduce to at most 2 parameter lists with multiple values."
        )

    # Log configuration
    total_combos = len(depth_list) * len(sample_size_list) * len(tau_list) * len(noise_vars_list)
    log_sweep_configuration(logger, tau_list, depth_list, sample_size_list, noise_vars_list, char_func_list,
                            varying_params, total_combos, max_jobs)
    
    # Run experiments for all parameter combinations
    results = {}
    current_combo = 0
    
    for depth in depth_list:
        for sample_size in sample_size_list:
            for noise_vars in noise_vars_list:
                for tau in tau_list:
                    current_combo += 1
                    
                    # Log current combination
                    logger.info("=" * 80)
                    logger.info(f"Combination {current_combo}/{total_combos}:")
                    logger.info(f"  max_depth = {depth}")
                    if isinstance(sample_size, list):
                        logger.info(f"  sample_size = {sample_size} (per treatment)")
                    else:
                        logger.info(f"  sample_size = {sample_size}")
                    logger.info(f"  tau = {tau}")
                    if noise_vars is not None:
                         logger.info(f"  noise_vars = {noise_vars}")
                    logger.info("=" * 80)
                    
                    # Prepare parameters for this combination
                    dgp_params, data_params, experiment_params = prepare_experiment_params(
                        config, tau, depth, sample_size, noise_vars=noise_vars, rename_noise_var=False
                    )
                    
                    # Run experiment
                    result_records = repeated_experiment(
                        optimization_params=optimization_params,
                        dgp_params=dgp_params,
                        data_params=data_params,
                        experiment_params=experiment_params,
                        estimators_dict=estimators_dict,
                        verbose=True,
                        n_jobs=max_jobs,
                        logger=logger
                    )
                    
                    # Store results with smart key
                    result_key = create_result_key(depth, sample_size, tau, noise_vars, None, varying_params)
                    results[result_key] = calculate_winners_curse_measures(
                        result_records=result_records,
                        optimization_params=optimization_params,
                        estimators_dict=estimators_dict,
                    )
                    
                    logger.info(f"[OK] Completed combination {current_combo}/{total_combos} (key={result_key})")
    
    logger.info("=" * 80)
    logger.info(f"All {total_combos} parameter combinations completed successfully")
    logger.info("=" * 80)
    
    return results


def main():
    """Main function to run the experiment."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run A/B test experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Use default config (ab_test_snr.yaml)
        python scripts/ab_test_experiment.py
        
        # Use a specific config file from configs/ directory
        python scripts/ab_test_experiment.py --config ab_test_snr.yaml
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='ab_test_snr.yaml',
        help='Path to YAML config file (default: ab_test_snr.yaml). '
            'Can be a filename in configs/ directory, relative path, or absolute path.'
    )
    parser.add_argument(
        '--estimators',
        type=str,
        nargs='+',
        default=None,
        help='If set, run only these estimator keys from estimators_dict (e.g. '
             '--estimators moon_bootstrap). Output directory is tagged with '
             '_estsubset so results can be merged into a full run later.'
    )

    args = parser.parse_args()
    
    # Determine config path
    try:
        config_path = get_config_path(__file__, args.config)
    except FileNotFoundError:
        # Fallback to parsing as absolute/relative path if not found in configs/ via helper
        # The helper assumes filename is in configs/, but let's support direct paths too
        # Actually, let's just use a more robust logic here or update the helper.
        # For now, let's stick to what the helper does or implement the robust logic here if needed.
        # The previous implementation had a robust parse_config_path. 
        # Let's re-implement a simple robust check here since we removed parse_config_path
        
        candidate_path = Path(args.config)
        if candidate_path.exists():
            config_path = candidate_path.resolve()
        else:
            # If get_config_path failed and it's not a direct path, re-raise
            raise
    
    # (1) Load config
    load_ab_test_config = create_ab_test_config_loader()
    config = load_ab_test_config(str(config_path))

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
    logger.info("Starting A/B Test Experiment")
    logger.info("=" * 80)
    logger.info(f"Config file: {config_path}")

    save_config(config, output_dir, logger)
    log_key_parameters(config, logger)
    
    # (4) Run the experiment
    logger.info("=" * 80)
    logger.info("Running experiment...")
    logger.info("=" * 80)
    
    try:
        tau_result_dict = run_experiment(config, logger)
        
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
