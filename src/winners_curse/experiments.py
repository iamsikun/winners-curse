"""
Reusable utilities for running experiments.

This module provides common functionality for experiment scripts including:
- Configuration loading and parsing
- Output directory management
- Logging setup
- Result saving
"""

import os
import sys
import pickle
import json
import copy
import math
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import contextmanager
from io import StringIO
import logging

import numpy as np
import yaml

from winners_curse.variables import (
    UnivariateGaussian, PointMass, Uniform, RandomVariable, ContinuousRandomVariable,
    UnivariateExponential, StudentT, Laplace, Logistic, Bernoulli
)


def setup_multiprocessing_env():
    """
    Set up environment variables for multiprocessing.
    
    Fixes for Windows multiprocessing + OpenBLAS issues by setting
    OpenBLAS to use only 1 thread per process to avoid access violations.
    """
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'


def get_max_jobs(config: Dict[str, Any], default: int = 24) -> int:
    """
    Calculate maximum number of parallel jobs from config.
    
    Args:
        config: Configuration dictionary
        default: Default max jobs if not specified
        
    Returns:
        Maximum number of jobs (min of config value and CPU count)
    """
    max_jobs = config.get('parallel', {}).get('max_jobs', default)
    return min(max_jobs, multiprocessing.cpu_count())


@contextmanager
def capture_parallel_output(logger: logging.Logger, level: int = logging.INFO):
    """
    Context manager to capture joblib's parallel output and redirect to logger.
    
    This captures stdout/stderr during parallel execution and logs it line-by-line
    through the logger instead of letting it go directly to the console.
    
    Args:
        logger: Logger instance to send captured output to
        level: Logging level for the captured output (default: INFO)
        
    Yields:
        None
        
    Usage:
        with capture_parallel_output(logger):
            result = Parallel(n_jobs=4, verbose=10)(...)
    """
    # Create string buffers
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    
    # Save original streams
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    try:
        # Replace with our buffers
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer
        
        yield
        
    finally:
        # Restore original streams
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        # Log captured output
        stdout_content = stdout_buffer.getvalue()
        if stdout_content.strip():
            for line in stdout_content.strip().split('\n'):
                logger.log(level, line)
        
        stderr_content = stderr_buffer.getvalue()
        if stderr_content.strip():
            for line in stderr_content.strip().split('\n'):
                logger.log(level, line)



def evaluate_param(value: Any) -> Any:
    """
    Evaluate a parameter value if it's a string expression.
    
    Args:
        value: Parameter value (can be string expression or direct value)
        
    Returns:
        Evaluated value
    """
    if isinstance(value, str):
        # Create a safe context with math and numpy functions
        context = {
            '__builtins__': {},
            'math': math,
            'np': np,
            'pi': math.pi,
            'e': math.e,
            'sqrt': math.sqrt,
            'log': math.log,
            'exp': math.exp,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'abs': abs,
            'pow': pow,
        }
        try:
            return eval(value, context)
        except Exception:
            # If evaluation fails, return original string (might be a categorical value)
            return value
    return value


def parse_variable_spec(var_spec: Dict[str, Any]) -> RandomVariable:
    """
    Parse a variable specification dictionary into a RandomVariable object.
    
    Args:
        var_spec: Dictionary with 'type' and parameters
        
    Returns:
        RandomVariable instance
        
    Raises:
        ValueError: If variable type is not supported
    """
    var_type = var_spec.get('type')
    
    if var_type == 'UnivariateGaussian':
        return UnivariateGaussian(
            evaluate_param(var_spec['mean']),
            evaluate_param(var_spec['std'])
        )
    elif var_type == 'PointMass':
        return PointMass(evaluate_param(var_spec['value']))
    elif var_type == 'Uniform':
        return Uniform(evaluate_param(var_spec['a']), evaluate_param(var_spec['b']))
    elif var_type == 'UnivariateExponential':
        return UnivariateExponential(evaluate_param(var_spec['rate']))
    elif var_type == 'StudentT':
        return StudentT(evaluate_param(var_spec['df']))
    elif var_type == 'Laplace':
        return Laplace(evaluate_param(var_spec['mu']), evaluate_param(var_spec['b']))
    elif var_type == 'Logistic':
        return Logistic(evaluate_param(var_spec['mu']), evaluate_param(var_spec['s']))
    elif var_type == 'Bernoulli':
        return Bernoulli(evaluate_param(var_spec['p']))
    else:
        raise ValueError(f"Unknown variable type: {var_type}")


def parse_lambda_expression(expr_str: str) -> callable:
    """
    Parse a string expression into a lambda function.
    
    Args:
        expr_str: String expression (e.g., "x**2 + x")
        
    Returns:
        Lambda function
        
    Example:
        >>> f = parse_lambda_expression("x**2 + x")
        >>> f(2)
        6
    """
    # Use eval with restricted namespace for safety
    return lambda x: eval(expr_str, {'x': x, '__builtins__': {}})


def parse_sklearn_model(model_spec: Dict[str, Any], model_classes: Dict[str, type]) -> Any:
    """
    Parse a sklearn model specification into a model instance.
    
    Args:
        model_spec: Dictionary with 'class' and 'params'
        model_classes: Dictionary mapping class names to classes
        
    Returns:
        Instantiated sklearn model
    """
    class_name = model_spec['class']
    if class_name not in model_classes:
        raise ValueError(f"Unknown model class: {class_name}")
    
    model_class = model_classes[class_name]
    return model_class(**model_spec.get('params', {}))


def load_config(
    config_path: str,
    optimizer_map: Optional[Dict[str, callable]] = None,
    estimator_class_map: Optional[Dict[str, type]] = None,
    estimator_function_map: Optional[Dict[str, callable]] = None,
    model_classes: Optional[Dict[str, type]] = None
) -> Dict[str, Any]:
    """
    Load configuration from YAML file and convert to Python objects.
    
    This is a generic config loader that can be customized for different
    experiment types by providing appropriate mapping dictionaries.
    
    Args:
        config_path: Path to YAML config file
        optimizer_map: Dictionary mapping optimizer names to functions
        estimator_class_map: Dictionary mapping estimator class names to classes
        estimator_function_map: Dictionary mapping estimator function names to functions
        model_classes: Dictionary mapping sklearn model class names to classes
        
    Returns:
        Dictionary with parsed configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set environment variables
    env = config.get('environment', {})
    for key, value in env.items():
        os.environ[key.upper()] = str(value)
    
    # Parse optimization_params - convert function name strings to actual functions
    # Parse optimization_params - convert function name strings to actual functions
    if 'optimization_params' in config and optimizer_map:
        opt_params = config['optimization_params']
        func_name = opt_params.get('optimizer')
        if func_name and func_name in optimizer_map:
            opt_params['optimizer'] = optimizer_map[func_name]
        elif func_name:
            raise ValueError(f"Unknown optimizer function: {func_name}")
    
    # Parse dgp_params - convert variable specifications to objects
    if 'dgp_params' in config:
        dgp = config['dgp_params']
        
        # Parse variable specifications (generic - works for any variable name)
        for key, value in dgp.items():
            if isinstance(value, dict) and 'type' in value:
                dgp[key] = parse_variable_spec(value)
        
        # Parse char_func - convert string expression to lambda
        if 'char_func' in dgp and isinstance(dgp['char_func'], str):
            dgp['char_func'] = parse_lambda_expression(dgp['char_func'])
    
    # Parse experiment_params - convert estimator class name to class
    if 'experiment_params' in config and estimator_class_map:
        exp_params = config['experiment_params']
        if 'estimator' in exp_params:
            est_config = exp_params['estimator']
            est_name = est_config.get('estimator')
            if est_name and est_name in estimator_class_map:
                est_config['estimator'] = estimator_class_map[est_name]
            elif est_name:
                raise ValueError(f"Unknown estimator class: {est_name}")
            
            # Parse sklearn models in params
            if 'params' in est_config and model_classes:
                for model_key in ['model_t', 'model_y', 'model']:
                    if model_key in est_config['params']:
                        model_spec = est_config['params'][model_key]
                        if isinstance(model_spec, dict):
                            est_config['params'][model_key] = parse_sklearn_model(
                                model_spec, model_classes
                            )
    
    # Parse estimators_dict - convert function name strings to actual functions
    if 'estimators_dict' in config and estimator_function_map:
        estimators = config['estimators_dict']
        for est_name, est_config in estimators.items():
            func_name = est_config.get('estimator')
            if func_name and func_name in estimator_function_map:
                est_config['estimator'] = estimator_function_map[func_name]
            elif func_name:
                raise ValueError(f"Unknown estimator function: {func_name}")
    
    return config


def create_output_directory(base_dir: str = 'results', experiment_name: str = 'experiment') -> Path:
    """
    Create a timestamped output directory.
    
    Args:
        base_dir: Base directory for results
        experiment_name: Name prefix for the output directory
        
    Returns:
        Path to the created directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(base_dir) / f'{experiment_name}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def setup_logging(output_dir: Path, logger_name: str = __name__) -> logging.Logger:
    """
    Set up logging to both file and console.
    
    Args:
        output_dir: Directory to save log file
        logger_name: Name for the logger
        
    Returns:
        Configured logger
    """
    log_file = output_dir / 'experiment.log'
    
    # Remove existing handlers to avoid duplicates
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    
    # Set level
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def make_config_serializable(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert config to a serializable format (replace functions/objects with strings).
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Serializable configuration dictionary
    """
    config_copy = copy.deepcopy(config)
    
    # Convert optimization_params
    # Convert optimization_params
    if 'optimization_params' in config_copy:
        opt_config = config_copy['optimization_params']
        if 'optimizer' in opt_config and callable(opt_config['optimizer']):
            if hasattr(opt_config['optimizer'], '__name__') and opt_config['optimizer'].__name__ != '<lambda>':
                opt_config['optimizer'] = opt_config['optimizer'].__name__
            else:
                # Lambda function or callable without name - convert to placeholder
                opt_config['optimizer'] = "<lambda function>"
    
    # Convert dgp_params
    if 'dgp_params' in config_copy:
        dgp = config_copy['dgp_params']
        for key, value in dgp.items():
            if isinstance(value, RandomVariable):
                # Convert RandomVariable back to dict
                if isinstance(value, UnivariateGaussian):
                    dgp[key] = {
                        'type': 'UnivariateGaussian',
                        'mean': value.mean,
                        'std': value.std
                    }
                elif isinstance(value, PointMass):
                    dgp[key] = {
                        'type': 'PointMass',
                        'value': value.value
                    }
                elif isinstance(value, Uniform):
                    dgp[key] = {
                        'type': 'Uniform',
                        'a': value.a,
                        'b': value.b
                    }
                elif isinstance(value, UnivariateExponential):
                    dgp[key] = {
                        'type': 'UnivariateExponential',
                        'rate': value.rate
                    }
                elif isinstance(value, StudentT):
                    dgp[key] = {
                        'type': 'StudentT',
                        'df': value.df
                    }
                elif isinstance(value, Laplace):
                    dgp[key] = {
                        'type': 'Laplace',
                        'mu': value.mu,
                        'b': value.b
                    }
                elif isinstance(value, Logistic):
                    dgp[key] = {
                        'type': 'Logistic',
                        'mu': value.mu,
                        's': value.s
                    }
                elif isinstance(value, Bernoulli):
                    dgp[key] = {
                        'type': 'Bernoulli',
                        'p': value.p
                    }
            elif callable(value):
                # Handle callable functions (including lambdas)
                if hasattr(value, '__name__') and value.__name__ != '<lambda>':
                    # Regular function - convert to name
                    dgp[key] = value.__name__
                else:
                    # Lambda function - convert to placeholder
                    dgp[key] = "<lambda function>"
    
    # Convert experiment_params
    if 'experiment_params' in config_copy:
        exp = config_copy['experiment_params']
        if 'estimator' in exp:
            est = exp['estimator']
            if 'estimator' in est and not isinstance(est['estimator'], str):
                if hasattr(est['estimator'], '__name__') and est['estimator'].__name__ != '<lambda>':
                    est['estimator'] = est['estimator'].__name__
                elif callable(est['estimator']):
                    # Lambda function or callable without name - convert to placeholder
                    est['estimator'] = "<lambda function>"
            if 'params' in est:
                for model_key in ['model_t', 'model_y', 'model']:
                    if model_key in est['params']:
                        model = est['params'][model_key]
                        if not isinstance(model, dict) and hasattr(model, '__class__'):
                            est['params'][model_key] = {
                                'class': model.__class__.__name__,
                                'params': model.get_params() if hasattr(model, 'get_params') else {}
                            }
    
    # Convert estimators_dict
    if 'estimators_dict' in config_copy:
        for est_name, est_config in config_copy['estimators_dict'].items():
            if 'estimator' in est_config and callable(est_config['estimator']):
                if hasattr(est_config['estimator'], '__name__') and est_config['estimator'].__name__ != '<lambda>':
                    est_config['estimator'] = est_config['estimator'].__name__
                else:
                    # Lambda function or callable without name - convert to placeholder
                    est_config['estimator'] = "<lambda function>"
    
    return config_copy


def save_config(config: Dict[str, Any], output_dir: Path, logger: logging.Logger):
    """
    Save configuration to output directory.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save config
        logger: Logger instance
    """
    # Save as YAML (with serializable version)
    config_serializable = make_config_serializable(config)
    config_file = output_dir / 'config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config_serializable, f, default_flow_style=False, sort_keys=False)
    
    # Also save as JSON for easier reading
    config_json_file = output_dir / 'config.json'
    with open(config_json_file, 'w') as f:
        json.dump(config_serializable, f, indent=2)
    
    logger.info(f"Configuration saved to {config_file} and {config_json_file}")


def save_results(
    results: Any,
    output_dir: Path,
    config: Dict[str, Any],
    logger: logging.Logger,
    filename: Optional[str] = None
):
    """
    Save experiment results to output directory.
    
    Args:
        results: Experiment results (can be dict, list, or any pickleable object)
        output_dir: Directory to save results
        config: Configuration dictionary
        logger: Logger instance
        filename: Optional custom filename (defaults to config output.filename or 'results.pkl')
    """
    # Save pickle file
    if filename is None:
        filename = config.get('output', {}).get('filename', 'results.pkl')
    
    results_file = output_dir / filename
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Results saved to {results_file}")


def log_key_parameters(config: Dict[str, Any], logger: logging.Logger):
    """
    Log key parameters from the configuration.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("Key Parameters:")
    
    if 'data_params' in config:
        data_params = config['data_params']
        if 'sample_size' in data_params:
            logger.info(f"  Sample size: {data_params['sample_size']}")
        if 'targ_sample_size' in data_params:
            logger.info(f"  Target sample size: {data_params['targ_sample_size']}")
    
    if 'experiment_params' in config:
        exp_params = config['experiment_params']
        if 'n_repeats' in exp_params:
            logger.info(f"  Number of repeats: {exp_params['n_repeats']}")

        if 'estimator' in exp_params:
            est_name = exp_params['estimator']['estimator'].__name__
            logger.info(f"  Estimator: {est_name}")
    
    if 'parallel' in config:
        max_jobs = get_max_jobs(config)
        logger.info(f"  Max parallel jobs: {max_jobs}")
    
    # Log experiment-specific parameters
    if 'tau_list' in config:
        logger.info(f"  Tau values: {config['tau_list']}")

    # Log estimators_dict
    if 'estimators_dict' in config:
        estimator_names = [est_dict['display_name'] for est_dict in config['estimators_dict'].values()]
        logger.info(f"  Correction estimators: {estimator_names}")


def process_noise_vars(dgp_params: Dict[str, Any], n_treatments: int) -> list:
    """
    Process noise variables from dgp_params.
    
    Handles:
    - Single 'noise_var' vs list 'noise_vars'
    - Parsing variable specifications
    - Padding or truncating to match n_treatments
    
    Args:
        dgp_params: DGP parameters dictionary (modified in-place to set 'noise_vars')
        n_treatments: Number of treatments to match
        
    Returns:
        List of processed noise variables
    """
    # Set noise vars based on config
    if 'noise_var' in dgp_params:
        noise_vars = [dgp_params.pop('noise_var')]
    elif 'noise_vars' in dgp_params:
        noise_vars = dgp_params['noise_vars']
        if not isinstance(noise_vars, list):
            noise_vars = [noise_vars]
    else:
        raise ValueError("Either 'noise_var' or 'noise_vars' must be specified in dgp_params")

    # Parse noise_vars if they are dicts
    noise_vars = [
        parse_variable_spec(nv) if isinstance(nv, dict) else nv 
        for nv in noise_vars
    ]

    # Pad or truncate noise_vars to match n_treatments
    if len(noise_vars) < n_treatments:
        # Pad with the last element
        noise_vars.extend([noise_vars[-1]] * (n_treatments - len(noise_vars)))
    elif len(noise_vars) > n_treatments:
        # Truncate
        noise_vars = noise_vars[:n_treatments]
        
    dgp_params['noise_vars'] = noise_vars
    return noise_vars


def get_config_path(script_path: str, config_filename: str) -> Path:
    """
    Get the path to a config file relative to the project root.
    
    Args:
        script_path: Path to the experiment script (use __file__)
        config_filename: Name of the config file
        
    Returns:
        Path to the config file
    """
    script_dir = Path(script_path).parent
    project_root = script_dir.parent
    config_path = project_root / 'configs' / config_filename
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    return config_path




def extract_tau_from_base_effect_vars(base_effect_vars: list) -> tuple:
    """Extract tau values from base_effect_vars (PointMass objects or dicts)."""
    return tuple(
        var['value'] if isinstance(var, dict) else var.value 
        for var in base_effect_vars
    )


def get_parameter_lists(config: Dict[str, Any]) -> tuple:
    """
    Extract parameter lists from config with intelligent defaults.
    
    Handles both 'base_effect_vars' (targeting) and 'base_effects' (ab_test).
    
    Returns:
        (tau_list, depth_list, sample_size_list)
    """
    comp_statics = config.get('comparative_statics', {})
    dgp_params = config['dgp_params']
    experiment_params = config['experiment_params']
    data_params = config['data_params']
    
    # Get baseline values
    # Try both naming conventions
    baseline_base_effect_vars = dgp_params.get('base_effect_vars') or dgp_params.get('base_effects', [])
    
    default_depth = experiment_params.get('estimator', {}).get('params', {}).get('max_depth', 5)
    default_sample_size = data_params.get('sample_size', 2500)
    
    # Extract tau_list
    tau_list_raw = comp_statics.get('tau_list')  # if tau_list does not exist, it will be None
    if tau_list_raw is not None:
        tau_list = [tuple(tau) for tau in tau_list_raw]
    elif 'tau_list' in config:
         # Fallback for ab_test if it uses 'tau_list' in root (old style)
         tau_list = [tuple(tau) for tau in config['tau_list']]
    elif baseline_base_effect_vars and all(v is not None for v in baseline_base_effect_vars):
        tau_list = [extract_tau_from_base_effect_vars(baseline_base_effect_vars)]
    else:
        raise ValueError(
            "No tau values specified. Either provide comparative_statics.tau_list, "
            "dgp_params.base_effect_vars (or base_effects), or tau_list in the config."
        )
    
    # Extract other parameter lists
    depth_list = comp_statics.get('depth_list', [default_depth])
    sample_size_list = comp_statics.get('sample_size_list', [default_sample_size])
    
    # Extract noise_vars_list
    noise_vars_list = comp_statics.get('noise_vars_list', [None])
    
    return tau_list, depth_list, sample_size_list, noise_vars_list


def identify_varying_params(tau_list: list, depth_list: list, sample_size_list: list, noise_vars_list: list) -> list:
    """Identify which parameters have multiple values (are varying)."""
    varying_params = []
    if len(tau_list) > 1:
        varying_params.append('tau')
    if len(depth_list) > 1:
        varying_params.append('depth')
    if len(sample_size_list) > 1:
        varying_params.append('sample_size')
    if len(noise_vars_list) > 1:
        varying_params.append('noise_vars')
    return varying_params


def create_result_key(depth: int, sample_size: int, tau: tuple, noise_vars: Any, varying_params: list) -> Any:
    """Create smart result key based on which parameters vary."""
    
    # Helper to format noise_vars for key
    def format_noise_vars(nv):
        if isinstance(nv, list):
            return tuple(str(v) for v in nv)
        return str(nv)

    if len(varying_params) == 0 or (len(varying_params) == 1 and 'tau' in varying_params):
        return tau
    elif len(varying_params) == 1:
        if 'depth' in varying_params:
            return depth
        elif 'sample_size' in varying_params:
            return sample_size
        elif 'noise_vars' in varying_params:
            return format_noise_vars(noise_vars)
    else:
        # Two parameters vary
        key_parts = []
        if 'depth' in varying_params:
            key_parts.append(depth)
        if 'sample_size' in varying_params:
            key_parts.append(sample_size)
        if 'tau' in varying_params:
            key_parts.append(tau)
        if 'noise_vars' in varying_params:
            key_parts.append(format_noise_vars(noise_vars))
            
        return tuple(key_parts)
    
    # Fallback
    return (depth, sample_size, tau, format_noise_vars(noise_vars))


def log_sweep_configuration(logger: logging.Logger, tau_list: list, depth_list: list, sample_size_list: list, noise_vars_list: list,
                            varying_params: list, total_combos: int, max_jobs: int):
    """Log the parameter sweep configuration."""
    logger.info("=" * 80)
    logger.info("Parameter sweep configuration:")
    logger.info(f"  Depths: {depth_list}")
    logger.info(f"  Sample sizes: {sample_size_list}")
    logger.info(f"  Treatment effects: {tau_list}")
    logger.info(f"  Noise vars list length: {len(noise_vars_list)}")
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


def prepare_experiment_params(config: Dict[str, Any], tau: tuple, depth: int, sample_size: int, noise_vars: Optional[list] = None, rename_noise_var: bool = True) -> tuple:
    """
    Prepare parameter copies for a single experiment run.
    
    Args:
        config: Configuration dictionary
        tau: Tuple of treatment effects
        depth: Tree depth
        sample_size: Sample size
        rename_noise_var: Whether to rename 'noise_vars' to 'noise_var' (True for targeting, False for ab_test)
        
    Returns:
        (dgp_params, data_params, experiment_params)
    """
    dgp_params_copy = config['dgp_params'].copy()
    data_params_copy = config['data_params'].copy()
    experiment_params_copy = config['experiment_params'].copy()
    
    # Handle nested dictionaries carefully
    if 'estimator' in experiment_params_copy:
         experiment_params_copy['estimator'] = config['experiment_params']['estimator'].copy()
         if 'params' in experiment_params_copy['estimator']:
             experiment_params_copy['estimator']['params'] = config['experiment_params']['estimator']['params'].copy()
    
    # Update base_effects/base_effect_vars
    # Use whichever key is present in the original config
    if 'base_effect_vars' in dgp_params_copy:
        dgp_params_copy['base_effect_vars'] = [PointMass(t) for t in tau]
    elif 'base_effects' in dgp_params_copy:
        dgp_params_copy['base_effects'] = [PointMass(t) for t in tau]
    else:
        # Default to base_effect_vars if neither exists (shouldn't happen given get_parameter_lists checks)
        dgp_params_copy['base_effect_vars'] = [PointMass(t) for t in tau]
        
    n_treatments = len(tau)
    
    # Update noise_vars if provided (from comparative statics)
    if noise_vars is not None:
        dgp_params_copy['noise_vars'] = noise_vars

    # Process noise variables
    process_noise_vars(dgp_params_copy, n_treatments)
    
    if rename_noise_var:
        if 'noise_vars' in dgp_params_copy and dgp_params_copy['noise_vars']:
            dgp_params_copy['noise_var'] = dgp_params_copy['noise_vars'][0]
            del dgp_params_copy['noise_vars']
    
    # Update other parameters
    data_params_copy['sample_size'] = sample_size
    
    # Only set max_depth if the estimator uses it
    if 'estimator' in experiment_params_copy and 'params' in experiment_params_copy['estimator'] and 'max_depth' in experiment_params_copy['estimator']['params']:
        experiment_params_copy['estimator']['params']['max_depth'] = depth
    
    # Update n_treatments if applicable
    if 'estimator' in experiment_params_copy and 'params' in experiment_params_copy['estimator']:
         experiment_params_copy['estimator']['params']['n_treatments'] = n_treatments
    
    return dgp_params_copy, data_params_copy, experiment_params_copy
