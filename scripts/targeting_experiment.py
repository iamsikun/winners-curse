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


def _extract_tau_from_base_effect_vars(base_effect_vars: list) -> tuple:
    """Extract tau values from base_effect_vars (PointMass objects or dicts)."""
    return tuple(
        var['value'] if isinstance(var, dict) else var.value 
        for var in base_effect_vars
    )


def _get_parameter_lists(config: dict) -> tuple:
    """
    Extract parameter lists from config with intelligent defaults.
    
    Returns:
        (tau_list, depth_list, sample_size_list)
    """
    comp_statics = config.get('comparative_statics', {})
    dgp_params = config['dgp_params']
    experiment_params = config['experiment_params']
    data_params = config['data_params']
    
    # Get baseline values
    baseline_base_effect_vars = dgp_params.get('base_effect_vars', [])
    default_depth = experiment_params['estimator']['params'].get('max_depth', 5)
    default_sample_size = data_params.get('sample_size', 2500)
    
    # Extract tau_list
    tau_list_raw = comp_statics.get('tau_list')  # if tau_list does not exist, it will be None
    if tau_list_raw is not None:
        tau_list = [tuple(tau) for tau in tau_list_raw]
    elif baseline_base_effect_vars:
        tau_list = [_extract_tau_from_base_effect_vars(baseline_base_effect_vars)]
    else:
        raise ValueError(
            "No tau values specified. Either provide comparative_statics.tau_list "
            "or dgp_params.base_effect_vars in the config."
        )
    
    # Extract other parameter lists
    depth_list = comp_statics.get('depth_list', [default_depth])
    sample_size_list = comp_statics.get('sample_size_list', [default_sample_size])
    
    return tau_list, depth_list, sample_size_list


def _identify_varying_params(tau_list: list, depth_list: list, sample_size_list: list) -> list:
    """Identify which parameters have multiple values (are varying)."""
    varying_params = []
    if len(tau_list) > 1:
        varying_params.append('tau')
    if len(depth_list) > 1:
        varying_params.append('depth')
    if len(sample_size_list) > 1:
        varying_params.append('sample_size')
    return varying_params


def _create_result_key(depth: int, sample_size: int, tau: tuple, varying_params: list) -> any:
    """Create smart result key based on which parameters vary."""
    if len(varying_params) == 0 or (len(varying_params) == 1 and 'tau' in varying_params):
        return tau
    elif len(varying_params) == 1:
        if 'depth' in varying_params:
            return depth
        elif 'sample_size' in varying_params:
            return sample_size
    else:
        # Two parameters vary
        if 'depth' in varying_params and 'tau' in varying_params:
            return (depth, tau)
        elif 'sample_size' in varying_params and 'tau' in varying_params:
            return (sample_size, tau)
        elif 'depth' in varying_params and 'sample_size' in varying_params:
            return (depth, sample_size)
    
    # Fallback
    return (depth, sample_size, tau)


def _log_sweep_configuration(logger, tau_list: list, depth_list: list, sample_size_list: list, 
                             varying_params: list, total_combos: int, max_jobs: int):
    """Log the parameter sweep configuration."""
    logger.info("=" * 80)
    logger.info("Parameter sweep configuration:")
    logger.info(f"  Depths: {depth_list}")
    logger.info(f"  Sample sizes: {sample_size_list}")
    logger.info(f"  Treatment effects: {tau_list}")
    logger.info(f"  Varying parameters: {varying_params if varying_params else ['none (single experiment)']}")
    logger.info(f"  Total combinations: {total_combos}")
    logger.info(f"  Parallel jobs: {max_jobs}")
    
    # Describe result key format
    if not varying_params or (len(varying_params) == 1 and 'tau' in varying_params):
        logger.info("  Result key format: tau tuples (e.g., (1.0, 1.01))")
    elif len(varying_params) == 1:
        param = varying_params[0]
        example = depth_list[0] if param == 'depth' else sample_size_list[0]
        logger.info(f"  Result key format: {param} values (e.g., {example})")
    else:
        logger.info(f"  Result key format: ({varying_params[0]}, {varying_params[1]}) tuples")
    
    logger.info("=" * 80)


def _prepare_experiment_params(config: dict, tau: tuple, depth: int, sample_size: int) -> tuple:
    """Prepare parameter copies for a single experiment run."""
    dgp_params_copy = config['dgp_params'].copy()
    data_params_copy = config['data_params'].copy()
    experiment_params_copy = config['experiment_params'].copy()
    experiment_params_copy['estimator'] = config['experiment_params']['estimator'].copy()
    experiment_params_copy['estimator']['params'] = config['experiment_params']['estimator']['params'].copy()
    
    # Update base_effect_vars
    dgp_params_copy['base_effect_vars'] = [PointMass(t) for t in tau]
    n_treatments = len(tau)
    
    # Process noise variables
    process_noise_vars(dgp_params_copy, n_treatments)
    if 'noise_vars' in dgp_params_copy and dgp_params_copy['noise_vars']:
        dgp_params_copy['noise_var'] = dgp_params_copy['noise_vars'][0]
        del dgp_params_copy['noise_vars']
    
    # Update other parameters
    data_params_copy['sample_size'] = sample_size
    
    # Only set max_depth if the estimator uses it (e.g., CausalForestDML)
    if 'max_depth' in experiment_params_copy['estimator']['params']:
        experiment_params_copy['estimator']['params']['max_depth'] = depth
    
    experiment_params_copy['estimator']['params']['n_treatments'] = n_treatments
    
    return dgp_params_copy, data_params_copy, experiment_params_copy


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
    tau_list, depth_list, sample_size_list = _get_parameter_lists(config)
    
    # Identify varying parameters and validate
    varying_params = _identify_varying_params(tau_list, depth_list, sample_size_list)
    if len(varying_params) > 2:
        raise ValueError(
            f"Cannot test more than 2 parameters simultaneously. "
            f"Currently varying: {varying_params} "
            f"(tau: {len(tau_list)}, depth: {len(depth_list)}, sample_size: {len(sample_size_list)}). "
            f"\nPlease reduce to at most 2 parameter lists with multiple values."
        )
    
    # Log configuration
    total_combos = len(depth_list) * len(sample_size_list) * len(tau_list)
    _log_sweep_configuration(logger, tau_list, depth_list, sample_size_list, 
                            varying_params, total_combos, max_jobs)
    
    # Run experiments for all parameter combinations
    results = {}
    current_combo = 0
    
    for depth in depth_list:
        for sample_size in sample_size_list:
            for tau in tau_list:
                current_combo += 1
                
                # Log current combination
                logger.info("=" * 80)
                logger.info(f"Combination {current_combo}/{total_combos}:")
                logger.info(f"  max_depth = {depth}")
                logger.info(f"  sample_size = {sample_size}")
                logger.info(f"  tau = {tau}")
                logger.info("=" * 80)
                
                # Prepare parameters for this combination
                dgp_params, data_params, experiment_params = _prepare_experiment_params(
                    config, tau, depth, sample_size
                )
                
                # Run experiment
                result_records = repeated_experiment(
                    optimization_params=optimization_params,
                    dgp_params=dgp_params,
                    data_params=data_params,
                    experiment_params=experiment_params,
                    estimators_dict=estimators_dict,
                    n_jobs=max_jobs,
                    verbose=True, 
                )
                
                # Store results with smart key
                result_key = _create_result_key(depth, sample_size, tau, varying_params)
                results[result_key] = calculate_winners_curse_measures(
                    result_records=result_records,
                    optimization_params=optimization_params,
                    estimators_dict=estimators_dict,
                    data_params=data_params
                )
                
                logger.info(f"✓ Completed combination {current_combo}/{total_combos} (key={result_key})")
    
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
        # Use default config (targeting_cff_delta_tau.yaml)
        python scripts/targeting_experiment.py
        
        # Use a specific config file from configs/ directory
        python scripts/targeting_experiment.py --config targeting_cff_delta_tau.yaml
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='targeting_correct_snr.yaml',
        help='Path to YAML config file (default: targeting_correct_snr.yaml). '
            'Can be a filename in configs/ directory, relative path, or absolute path.'
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
