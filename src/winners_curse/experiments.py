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
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import logging

import numpy as np
import yaml

from winners_curse.variables import (
    UnivariateGaussian, PointMass, Uniform, RandomVariable, ContinuousRandomVariable
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
            var_spec['mean'],
            var_spec['std']
        )
    elif var_type == 'PointMass':
        return PointMass(var_spec['value'])
    elif var_type == 'Uniform':
        return Uniform(var_spec['a'], var_spec['b'])
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
    if 'optimization_params' in config and optimizer_map:
        opt_params = config['optimization_params']
        for opt_name, opt_config in opt_params.items():
            func_name = opt_config.get('optimizer')
            if func_name and func_name in optimizer_map:
                opt_config['optimizer'] = optimizer_map[func_name]
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
    if 'optimization_params' in config_copy:
        for opt_name, opt_config in config_copy['optimization_params'].items():
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
    
    # Also save a summary as JSON (for quick inspection)
    # Only works if results is a dict
    if isinstance(results, dict):
        summary = {}
        for key, value in results.items():
            if value is not None:
                # Convert non-JSON-serializable keys (like tuples) to strings
                json_key = str(key) if not isinstance(key, (str, int, float, bool, type(None))) else key
                try:
                    if isinstance(value, dict):
                        # Recursively process nested dicts, converting keys to strings if needed
                        summary[json_key] = {
                            str(k) if not isinstance(k, (str, int, float, bool, type(None))) else k: 
                            float(v) if isinstance(v, (np.integer, np.floating)) else str(v)
                            for k, v in value.items()
                            if not isinstance(v, np.ndarray) or v.size < 100  # Skip large arrays
                        }
                    elif isinstance(value, (np.integer, np.floating)):
                        summary[json_key] = float(value)
                    elif isinstance(value, np.ndarray) and value.size < 100:
                        summary[json_key] = value.tolist()
                    else:
                        summary[json_key] = str(value)
                except Exception:
                    summary[json_key] = str(value)
        
        summary_file = output_dir / 'results_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results summary saved to {summary_file}")


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

