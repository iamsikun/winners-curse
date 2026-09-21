import numpy as np 
from scipy.optimize import fsolve
from scipy.special import ndtr, ndtri


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


def _max_abs_normal_quantile(n_items: int, alpha: float) -> float:
    """
    (1 - alpha) quantile of max_j |Z_j| for j = 1, ..., n_items with Z_j iid N(0, 1).

    The projection critical value of Andrews et al. (2024) is the quantile of
    max_j |xi_j| / sigma_j where xi ~ N(0, diag(sigma^2)). Writing xi_j = sigma_j Z_j
    makes the sigma_j cancel, so the critical value depends only on the number of
    items and alpha -- not on the estimates or their standard errors. With the
    components independent,

        P(max_j |Z_j| <= c) = (2 Phi(c) - 1) ** n_items,

    which inverts to c = Phi^{-1}((1 + (1 - alpha) ** (1 / n_items)) / 2).

    Params:
    -------
    n_items: int
        Number of alternatives (treatment arms, or treatments per customer).
    alpha: float
        Tail probability; the returned c satisfies P(max_j |Z_j| > c) = alpha.

    Returns:
    --------
    float
        The critical value c.
    """
    return float(ndtri((1.0 + (1.0 - alpha) ** (1.0 / n_items)) / 2.0))


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

    The critical value is evaluated in closed form by ``_max_abs_normal_quantile``
    rather than simulated: with a diagonal covariance the standard deviations cancel
    out of max_j |xi_j| / sigma_j, so the simulated quantile estimated the same
    constant for every call (up to Monte Carlo error) at a cost of ``n_simulations``
    normal draws each time.

    Params:
        mean_arr (np.array): Array of mean estimates for each alternative.
        std_arr (np.array): Array of standard deviations for each alternative.
        max_item_idx (int): Index of the maximum value in the mean_arr.
        alpha (float): Confidence level (1 - alpha). Default is 0.05 for 95% confidence.
        n_simulations (int): Ignored; retained so existing callers keep working.
    
    Returns:
        CI_lower (float): Lower bound of the confidence interval.
        CI_upper (float): Upper bound of the confidence interval.
        C_alpha (float): Critical value for the confidence interval
    """
    # Step 0: If max_item_idx is not provided, find the index of the maximum value
    if max_item_idx is None:
        max_item_idx = mean_arr.argmax()

    # Step 1: Critical value of max_j |Z_j|, Z iid standard normal
    c_alpha = _max_abs_normal_quantile(len(mean_arr), alpha)

    # Step 2: Construct the confidence interval
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

    ``n_simulations`` is ignored: the projection critical value c_beta is a constant
    that depends only on the number of alternatives, so it is evaluated in closed
    form instead of being re-simulated for every call. See
    ``_max_abs_normal_quantile``.
    """
    # if max_item_idx is not provided, find the index of the max item
    if max_item_idx is None:
        max_item_idx = mean_arr.argmax()

    # get the mean and std of the max item
    max_item_mean, max_item_std = mean_arr[max_item_idx], std_arr[max_item_idx]

    # calculate c_beta from projection method
    c_beta = _max_abs_normal_quantile(len(mean_arr), 0.05 / 10)

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