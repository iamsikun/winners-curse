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
)

# Fix for Windows multiprocessing + OpenBLAS issues
setup_multiprocessing_env()

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
    dgp_params = config['dgp_params'].copy()  # Make a copy to avoid modifying original
    data_params = config['data_params']
    experiment_params = config['experiment_params']
    estimators_dict = config['estimators_dict']
    tau_list = [tuple(tau) for tau in config['tau_list']]
    
    # Update n_treatments in experiment_params based on base_effect_vars length
    n_treatments = len(dgp_params['base_effect_vars'])
    experiment_params['estimator']['params']['n_treatments'] = n_treatments
    
    tau_result_dict = {tau: None for tau in tau_list}
    
    max_jobs = get_max_jobs(config)
    
    for te_tuple in tau_list:
        logger.info(f'(tau_1, tau_2) = {te_tuple}')
        
        # Update base_effect_vars for this tau
        dgp_params['base_effect_vars'] = [
            PointMass(te_tuple[0]), 
            PointMass(te_tuple[1])
        ]
        
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
            data_params=data_params
        )
        
        logger.info(f"Completed experiment for (tau_1, tau_2) = {te_tuple}")
    
    return tau_result_dict


def parse_config_path(config_arg: str) -> Path:
    """
    Parse config path from command line argument.
    
    Supports:
    - Relative paths (assumed to be in configs/ directory)
    - Absolute paths
    - Filenames (assumed to be in configs/ directory)
    
    Args:
        config_arg: Config file path or filename from command line
        
    Returns:
        Path to config file
    """
    config_path = Path(config_arg)
    
    # If it's an absolute path, use it directly
    if config_path.is_absolute():
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return config_path
    
    # If it's a relative path that exists, use it
    if config_path.exists():
        return config_path.resolve()
    
    # Otherwise, assume it's a filename in the configs/ directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    configs_dir = project_root / 'configs'
    config_path = configs_dir / config_arg
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Tried: {config_arg} (as absolute path)\n"
            f"Tried: {Path(config_arg).resolve()} (as relative path)\n"
            f"Tried: {config_path} (in configs/ directory)"
        )
    
    return config_path


def main():
    """Main function to run the experiment."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run targeting causal forest experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Use default config (targeting_cff_delta_tau.yaml)
        python scripts/run_targeting_cff_delta_tau.py
        
        # Use a specific config file from configs/ directory
        python scripts/run_targeting_cff_delta_tau.py --config targeting_cff_delta_tau.yaml
        
        # Use a config file with relative path
        python scripts/run_targeting_cf.py --config configs/targeting_cff_delta_tau.yaml
        
        # Use a config file with absolute path
        python scripts/run_targeting_cff_delta_tau.py --config /path/to/config.yaml
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='targeting_cff_delta_tau.yaml',
        help='Path to YAML config file (default: targeting_cff_delta_tau.yaml). '
            'Can be a filename in configs/ directory, relative path, or absolute path.'
    )
    
    args = parser.parse_args()
    
    # Determine config path
    config_path = parse_config_path(args.config)
    
    # (1) Load config
    load_targeting_config = create_targeting_config_loader()
    config = load_targeting_config(str(config_path))
    
    # (2) Generate timestamped output folder
    output_base = config.get('output', {}).get('results_dir', 'results')
    # Derive experiment name from config filename (without extension)
    experiment_name = config_path.stem
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
