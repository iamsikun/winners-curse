import os
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
    setup_multiprocessing_env,
    load_config,
    create_output_directory,
    setup_logging,
    save_config,
    save_results,
    log_key_parameters,
    get_max_jobs,
    process_noise_vars,
    get_config_path,
)

# Fix for Windows multiprocessing + OpenBLAS issues
setup_multiprocessing_env()

from winners_curse.ab_test import (
    select_higher_effect,
    repeated_experiment,
    calculate_winners_curse_measures,
    bootstrap_correction_estimate,
    empirical_bayes_estimate,
    selective_inference_estimate,
    sample_splitting_estimate,
    jackknife_estimate,
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
        'selective_inference_estimate': selective_inference_estimate,
        'sample_splitting_estimate': sample_splitting_estimate,
        'jackknife_estimate': jackknife_estimate,
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
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Dictionary of results keyed by tau tuples
    """
    optimization_params = config['optimization_params']
    data_params = config['data_params']
    experiment_params = config['experiment_params']
    estimators_dict = config['estimators_dict']
    tau_list = [tuple(tau) for tau in config['tau_list']]
    
    tau_result_dict = {tau: None for tau in tau_list}
    
    max_jobs = get_max_jobs(config)
    
    for te_tuple in tau_list:
        logger.info(f'tau tuple = {te_tuple}')
        
        dgp_params = config['dgp_params'].copy()
        
        # Update base_effects for this tau
        dgp_params['base_effects'] = [PointMass(t) for t in te_tuple]
        
        # Process noise variables using helper
        n_treatments = len(dgp_params['base_effects'])
        process_noise_vars(dgp_params, n_treatments)
        
        result_records = repeated_experiment(
            optimization_params=optimization_params,
            dgp_params=dgp_params,
            data_params=data_params,
            experiment_params=experiment_params,
            estimators_dict=estimators_dict,
            verbose=True,
            n_jobs=max_jobs
        )
        
        tau_result_dict[te_tuple] = calculate_winners_curse_measures(
            result_records=result_records,
            optimization_params=optimization_params,
            estimators_dict=estimators_dict,
        )
        
        logger.info(f"Completed experiment for tau tuple = {te_tuple}")
    
    return tau_result_dict


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
    
    # (2) Generate timestamped output folder
    output_base = config.get('output', {}).get('results_dir', 'results')
    # Derive experiment name from config filename (without extension)
    experiment_name = config_path.stem
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
