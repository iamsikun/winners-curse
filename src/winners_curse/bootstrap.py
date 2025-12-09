import numpy as np
from joblib import Parallel, delayed
from typing import Any, Callable, Tuple

def get_bootstrap_distribution(
    data: Any, 
    estimator_func: Callable[[Any], Any],
    optimizer_func: Callable[[Any], Any],
    evaluator_func: Callable[[Any, Any], float],
    bootstrap_sampler_func: Callable[[Any, int], Any],
    wc_func: Callable[[Any, Any, Any, Callable], float] = None,
    n_bootstraps: int = 1000,
    n_jobs: int = 1,
    verbose: bool = False,
    seed: int = None,
) -> np.ndarray:
    """
    Get the bootstrap distribution of the Winner's Curse.

    This function implements the general bootstrap correction logic:
    1. Compute empirical estimates and selection on the original data.
    2. Compute the "naive" value of the empirical selection.
    3. Generate bootstrap samples.
    4. For each bootstrap sample, compute estimates, selection, and the "Winner's Curse" (bias).
    5. Return the bootstrap distribution of the Winner's Curse.

    Params:
    -------
    data: Any
        The original dataset (e.g., list of arrays, or tuple of arrays).
    estimator_func: Callable[[Any], Any]
        Function to compute estimates from data. Returns `estimates`.
        Signature: `estimator_func(data, **kwargs) -> estimates`
    optimizer_func: Callable[[Any], Any]
        Function to select the best option based on estimates. Returns `selection`.
        Signature: `optimizer_func(estimates, **kwargs) -> selection`
    evaluator_func: Callable[[Any, Any], float]
        Function to evaluate a selection given estimates. Returns a float value.
        Signature: `evaluator_func(selection, estimates, **kwargs) -> float`
    bootstrap_sampler_func: Callable[[Any, int], Any]
        Function to generate a bootstrap sample from data. Returns `boot_data`.
        Signature: `bootstrap_sampler_func(data, seed, **kwargs) -> boot_data`
    wc_func: Callable[[Any, Any, Any, Callable], float], optional
        Function to compute the Winner's Curse for a single bootstrap iteration.
        If None, defaults to: `evaluator_func(boot_sel, boot_est) - evaluator_func(boot_sel, emp_est)`
        Signature: `wc_func(boot_sel, boot_est, emp_est, evaluator_func, **kwargs) -> float`
    n_bootstraps: int
        Number of bootstrap samples.
    n_jobs: int
        Number of parallel jobs.
    verbose: bool
        Whether to show progress.
    seed: int
        Random seed.

    Returns:
    --------
    np.ndarray: Bootstrap distribution of the Winner's Curse.
    """

    if seed is not None:
        np.random.seed(seed)

    # 1. Compute empirical estimates
    emp_est = estimator_func(data)

    # Default Winner's Curse function (standard bootstrap)
    if wc_func is None:
        def default_wc_func(boot_sel, boot_est, emp_est, evaluator_func):
            boot_val = evaluator_func(boot_sel, boot_est)
            cross_val = evaluator_func(boot_sel, emp_est)

            return boot_val - cross_val
        wc_func = default_wc_func

    # Helper for parallel execution
    def compute_single_bootstrap(boot_id) -> float:
        # Reseed for parallelism safety if needed, though joblib handles this usually.
        # We use the boot_id to derive a seed if a master seed was provided.
        current_seed = seed + boot_id if seed is not None else None
        
        # Sample
        boot_data = bootstrap_sampler_func(data, seed=current_seed)
        
        # Estimate
        boot_est = estimator_func(boot_data)
        
        # Optimize
        boot_sel = optimizer_func(boot_est)
        
        # Compute WC
        wc = wc_func(boot_sel, boot_est, emp_est, evaluator_func)
        
        return wc

    # 4. Run bootstrap loop
    if n_jobs == 1:
        # Optimization: Bypass joblib overhead for sequential execution
        # This avoids pickling/backend initialization overhead which can be significant
        # when running inside another parallel loop
        wc_list = [compute_single_bootstrap(boot_id) for boot_id in range(n_bootstraps)]
    else:
        wc_list = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(compute_single_bootstrap)(boot_id) for boot_id in range(n_bootstraps)
        )
    
    return np.array(wc_list)

def bootstrap_correction_estimator(
    data: Any,
    estimator_func: Callable[[Any], Any],
    optimizer_func: Callable[[Any], Any],
    evaluator_func: Callable[[Any, Any], float],
    bootstrap_sampler_func: Callable[[Any, int], Any],
    empirical_estimates: tuple = None,
    wc_func: Callable[[Any, Any, Any, Callable], float] = None,
    n_bootstraps: int = 1000,
    n_jobs: int = 1,
    verbose: bool = False,
    seed: int = None,
) -> Tuple[Any, float]:
    """
    Generic bootstrap correction estimator.

    This function implements the general bootstrap correction logic:
    1. Compute empirical estimates and selection on the original data.
    2. Compute the "naive" value of the empirical selection.
    3. Generate bootstrap samples.
    4. For each bootstrap sample, compute estimates, selection, and the "Winner's Curse" (bias).
    5. Average the Winner's Curse over bootstrap samples to get the bias correction.
    6. Subtract the bias from the naive value to get the corrected estimate.

    Params:
    -------
    data: Any
        The original dataset (e.g., list of arrays, or tuple of arrays).
    estimator_func: Callable[[Any], Any]
        Function to compute estimates from data. Returns `estimates`.
        Signature: `estimator_func(data, **kwargs) -> estimates`
    optimizer_func: Callable[[Any], Any]
        Function to select the best option based on estimates. Returns `selection`.
        Signature: `optimizer_func(estimates, **kwargs) -> selection`
    evaluator_func: Callable[[Any, Any], float]
        Function to evaluate a selection given estimates. Returns a float value.
        Signature: `evaluator_func(selection, estimates, **kwargs) -> float`
    bootstrap_sampler_func: Callable[[Any, int], Any]
        Function to generate a bootstrap sample from data. Returns `boot_data`.
        Signature: `bootstrap_sampler_func(data, seed, **kwargs) -> boot_data`
    empirical_estimates: tuple = None,
    wc_func: Callable[[Any, Any, Any, Callable], float], optional
        Function to compute the Winner's Curse for a single bootstrap iteration.
        If None, defaults to: `evaluator_func(boot_sel, boot_est) - evaluator_func(boot_sel, emp_est)`
        Signature: `wc_func(boot_sel, boot_est, emp_est, evaluator_func, **kwargs) -> float`
    n_bootstraps: int
        Number of bootstrap samples.
    n_jobs: int
        Number of parallel jobs.
    verbose: bool
        Whether to show progress.
    seed: int
        Random seed.

    Returns:
    --------
    tuple: (empirical_selection, corrected_value)
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. Compute empirical estimates
    if empirical_estimates is None:
        emp_est = estimator_func(data)
    else:
        emp_est = empirical_estimates

    # 2. Compute empirical selection
    emp_sel = optimizer_func(emp_est)

    # 3. Compute naive value
    naive_val = evaluator_func(emp_sel, emp_est)

    # 4. Compute bootstrap distribution
    wc_arr = get_bootstrap_distribution(
        data, estimator_func, optimizer_func, evaluator_func, bootstrap_sampler_func, wc_func, n_bootstraps, n_jobs, verbose, seed
    )

    # 5. Compute bias
    bias = np.mean(wc_arr)

    # 6. Return corrected estimate
    corrected_val = naive_val - bias

    return emp_sel, corrected_val
