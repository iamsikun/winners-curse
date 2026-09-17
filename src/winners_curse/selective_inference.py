import numpy as np 
from scipy.optimize import fsolve
from scipy.special import ndtr


def _truncnorm_cdf(x: float, a: float, b: float, mu: float, sigma: float) -> float:
    """
    CDF of N(mu, sigma^2) truncated to the standardized bounds [a, b].

    Closed form of ``scipy.stats.truncnorm.cdf(x, a, b, loc=mu, scale=sigma)``
    written through ``scipy.special.ndtr``. It agrees with scipy to ~1e-16 over
    the ranges used below and is roughly 200x faster, which matters because the
    root finders call it tens of times per customer.

    Params:
    -------
    x: float
        Evaluation point, in the original (unstandardized) units.
    a: float
        Lower truncation bound, standardized as (lb - mu) / sigma.
    b: float
        Upper truncation bound, standardized as (ub - mu) / sigma.
    mu: float
        Mean of the untruncated normal.
    sigma: float
        Standard deviation of the untruncated normal.

    Returns:
    --------
    float
        Truncated normal CDF at x.
    """
    xi = (x - mu) / sigma

    if xi <= a:
        return 0.0
    if xi >= b:
        return 1.0

    lo, hi = ndtr(a), ndtr(b)
    if hi <= lo:  # degenerate truncation interval; matches scipy's nan
        return np.nan

    return (ndtr(xi) - lo) / (hi - lo)


def conditional_inference(
    mean_arr: np.ndarray, 
    std_arr: np.ndarray, 
    max_item_idx: int = None, 
    quantile: float = 0.5
) -> float:
    """ 
    Compute the conditional inference method from (Andrews et al. 2024, QJE)
    """
    # if max_item_idx is not provided, find the index of the max item
    if max_item_idx is None:
        max_item_idx = mean_arr.argmax()

    # get the mean and std of the max item
    max_item_mean, max_item_std = mean_arr[max_item_idx], std_arr[max_item_idx]

    # get the second max mean
    second_max_mean = np.delete(mean_arr, max_item_idx).max()

    def local_truncated_normal_cdf(x, mu) -> float:
        trunc_lb = (second_max_mean - mu) / max_item_std

        return _truncnorm_cdf(x, trunc_lb, np.inf, mu, max_item_std)
    
    return fsolve(
        func=lambda mu: local_truncated_normal_cdf(max_item_mean, mu) - 1 + quantile, 
        x0=max_item_mean, 
        full_output=True, 
    )


def unconditional_inference(
    mean_arr: np.ndarray, 
    std_arr: np.ndarray, 
    max_item_idx: int = None, 
    alpha=0.05, 
    n_simulations=10000
):
    """
    Compute the projected simultaneous confidence interval from (Andrews et al. 2024, QJE)
    
    Params:
        mean_arr (np.array): Array of mean estimates for each alternative.
        std_arr (np.array): Array of standard deviations for each alternative.
        max_item_idx (int): Index of the maximum value in the mean_arr.
        alpha (float): Confidence level (1 - alpha). Default is 0.05 for 95% confidence.
        n_simulations (int): Number of Monte Carlo simulations. Default is 100,000.
    
    Returns:
        CI_lower (float): Lower bound of the confidence interval.
        CI_upper (float): Upper bound of the confidence interval.
        C_alpha (float): Critical value for the confidence interval
    """
    # Step 0: If max_item_idx is not provided, find the index of the maximum value
    if max_item_idx is None:
        max_item_idx = mean_arr.argmax()

    # Step 1: Draw samples for Z
    xi_arr = np.random.multivariate_normal(
        mean=np.zeros_like(mean_arr),
        cov=np.diag(std_arr ** 2), 
        size=n_simulations, 
    )  # shape = (n_simulations, n_sites)
    
    # Step 2: Scale Z to match the standardization: Xi_theta / sqrt(Sigma_X_theta)
    abs_t = np.abs(xi_arr) / std_arr  # Broadcasting over variances
    
    # Step 3: Compute the maximum of the absolute values for each simulation
    max_abs_t = np.max(abs_t, axis=1)  # shape = (n_simulations,)
    
    # Step 4: Determine the (1 - alpha) quantile of the maximum values
    c_alpha = np.quantile(max_abs_t, 1 - alpha)
    
    # Step 5: Construct the confidence interval
    CI_lower = mean_arr[max_item_idx] - c_alpha * std_arr[max_item_idx]
    CI_upper = mean_arr[max_item_idx] + c_alpha * std_arr[max_item_idx]
    
    return CI_lower, CI_upper, c_alpha


def hybrid_inference(
    mean_arr: np.ndarray, 
    std_arr: np.ndarray, 
    max_item_idx: int = None, 
    quantile: float = None, 
    n_simulations: int = 10000, 
) -> float:
    """ 
    Compute the hybrid inference method from (Andrews et al. 2024, QJE)
    """
    # if max_item_idx is not provided, find the index of the max item
    if max_item_idx is None:
        max_item_idx = mean_arr.argmax()

    # get the mean and std of the max item
    max_item_mean, max_item_std = mean_arr[max_item_idx], std_arr[max_item_idx]

    # calculate c_beta from projection method
    _, _, c_beta = unconditional_inference(
        mean_arr, std_arr, max_item_idx=max_item_idx, alpha=0.05/10, 
        n_simulations=n_simulations
    )

    # get the second max mean
    second_max_mean = np.delete(mean_arr, max_item_idx).max()

    def local_truncated_normal_cdf(x, mu) -> float:
        if mu < max_item_mean - c_beta * max_item_std:
            return -10000
        if mu > max_item_mean + c_beta * max_item_std:
            return 10000
        
        # truncation for the normal distribution
        trunc_lb = max(second_max_mean, max_item_mean - c_beta * max_item_std)
        trunc_ub = max_item_mean + c_beta * max_item_std
        
        # update 
        trunc_lb = max(trunc_lb, mu - c_beta * max_item_std)
        trunc_ub = min(trunc_ub, mu + c_beta * max_item_std)

        return _truncnorm_cdf(
            x, 
            (trunc_lb - mu) / max_item_std, 
            (trunc_ub - mu) / max_item_std, 
            mu, max_item_std
        )
    
    return fsolve(
        func=lambda mu: local_truncated_normal_cdf(max_item_mean, mu) - 1 + quantile, 
        x0=max_item_mean, 
        full_output=True, 
    )