import os 
import sys 
sys.path.insert(0, os.path.abspath('.'))
import numpy as np 
import statsmodels.api as sm
from scipy.interpolate import CubicSpline
from patsy import dmatrix

from scipy.stats import norm
from scipy.integrate import quad

from joblib import Parallel, delayed

class EmpiricalBayes(object):
    def __init__(self, dof: int = 5, bin_width: float = 0.2, sigma: float = None):
        self.dof = dof
        self.bin_width = bin_width

        # initialize attributes
        self.spline = None
        self.std_est = sigma

    def fit(self, x: np.ndarray):
        # Step 0: estimate sigma
        self.std_est = np.std(x) if self.std_est is None else self.std_est

        # Step 1: Bin data for Poisson regression
        bins = np.arange(min(x), max(x) + self.bin_width, self.bin_width)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        counts, _ = np.histogram(x, bins=bins)

        # Step 2: Poisson Regression for Log-density Estimation
        exog = dmatrix(f"cr(x, df={self.dof})", {'x': bin_centers}, return_type='dataframe')
        model = sm.GLM(counts, exog, family=sm.families.Poisson()).fit()
        log_counts_est = np.log(model.predict(exog))
        # Step 3:
        self.spline = CubicSpline(bin_centers, log_counts_est, bc_type='natural')

        return self

    def predict(self, x: np.ndarray):
        # Apply Tweedie's Formula
        log_density_derivative = self.spline.derivative()(x)
        bayes_correction = self.std_est**2 * log_density_derivative

        return x + bayes_correction


def bayes_normal(
    mle_treatment_effects: np.ndarray,
    sampling_vars: np.ndarray,
    prior_mean: float = 0.0,
    prior_std: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """ 
    Bayesian estimator with Normal prior for density estimation.

    Params:
    -------
    mle_treatment_effects: np.ndarray
        maximum likelihood estimates of treatment effects
    sampling_vars: np.ndarray
        variance of the likelihood model for each experiment
    prior_mean: float
        prior mean for normal distribution
    prior_std: float
        prior standard deviation for normal distribution

    Returns:
    --------
    np.ndarray
        posterior mean estimates
    """
    # calculate shrinkage factor
    shrink_factor = prior_std ** 2 / (prior_std ** 2 + sampling_vars)

    # calculate posterior mean
    posterior_mean = shrink_factor * mle_treatment_effects + (1 - shrink_factor) * prior_mean

    return posterior_mean


def empirical_bayes_normal(
    mle_treatment_effects: np.ndarray, 
    sampling_vars: np.ndarray,
    dof: int = 3, 
    bin_width: float = 0.2, 
    **kwargs, 
) -> np.ndarray:
    """ 
    Empirical Bayes with Normal prior for density estimation. 
    Algorithm follows (Efron 2011, JASA)'s Tweedie formula approach. 

    Params:
    -------
    mle_treatment_effects: np.ndarray
        maximum likelihood estimates of treatment effects
    sampling_vars: np.ndarray
        variance of the likelihood model for each experiment
    prior_mean: float
        prior mean for normal distribution
    prior_std: float
        prior standard deviation for normal distribution
    dof: int
        degree of freedom for cubic spline
    bin_width: float
        bin width for histogram
    """
    # Step 1: Bin data for Poisson regression
    bins = np.arange(min(mle_treatment_effects), max(mle_treatment_effects) + bin_width, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    counts, _ = np.histogram(mle_treatment_effects, bins=bins)

    # Step 2: Poisson Regression for Log-density Estimation
    exog = dmatrix(f"cr(x, df={dof})", {'x': bin_centers}, return_type='dataframe')
    model = sm.GLM(counts, exog, family=sm.families.Poisson()).fit()
    log_counts_est = np.log(model.predict(exog))
    # Step 3:
    spline = CubicSpline(bin_centers, log_counts_est, bc_type='natural')

    # Apply Tweedie's Formula
    log_density_derivative = spline.derivative()(mle_treatment_effects)
    bayes_correction = sampling_vars * log_density_derivative

    return mle_treatment_effects + bayes_correction


def empirical_bayes_spike_slab(
    mle_treatment_effects: np.ndarray, 
    sampling_vars: np.ndarray, 
    pi: float = 0.5, 
    max_iter=100, tol=1e-6, 
    return_all=False, 
    **kwargs
) -> np.ndarray:
    """ 
    Empirical Bayes with Spike-and-Slab prior for density estimation, 
    where spike is a point mass at zero and slab is a normal distribution.

    Params:
    -------
    sampling_vars: np.ndarray
        variance of the likelihood model for each experiment
    pi: float
        prior probability for spike component
    max_iter: int
        maximum number of iterations for EM algorithm
    tol: float
        tolerance for convergence
    return_all: bool
        whether to return all intermediate parameters

    Returns:
    --------
    np.ndarray
        posterior mean estimates
    """
    # extract parameters
    # n_experiments = mle_treatment_effects.shape[0]
    
    # Initialize parameters
    mu0 = mle_treatment_effects.mean()   # Initial guess for the slab mean
    tau2 = mle_treatment_effects.var()   # Initial guess for the slab variance (excess over prior_std^2)
    
    # Store previous parameter values for convergence checking
    pi_old, mu0_old, tau2_old = pi, mu0, tau2
    
    for iteration in range(max_iter):
        # E-step: compute posterior probability (responsibilities) r_i for the slab component (z_i = 1)
        # f_spike = N(y; 0, sigma^2), shape = (n_experiments, )
        f_spike = norm.pdf(mle_treatment_effects, loc=0, scale=sampling_vars**0.5)
        # f_slab = N(y; mu0, sigma^2 + tau2), shape = (n_experiments, )
        f_slab = norm.pdf(mle_treatment_effects, loc=mu0, scale=np.sqrt(sampling_vars + tau2))
        
        # shape = (n_experiments, )
        r = ((1 - pi) * f_slab) / (pi * f_spike + (1 - pi) * f_slab)
        
        # M-step: update parameters
        pi = 1 - np.mean(r)  # proportion of observations in the spike
        # Update mu0 using weighted average (only for observations in the slab)
        if np.sum(r) > 0:
            mu0 = np.sum(r * mle_treatment_effects) / np.sum(r)
        else:
            mu0 = 0
        
        # Update tau2: note that tau2 represents the extra variance in the slab
        tau2_num = np.sum( r * ((mle_treatment_effects - mu0)**2 - sampling_vars**2) )
        tau2_den = np.sum(r)
        tau2 = max(1e-6, tau2_num / tau2_den)  # enforce non-negativity
        
        # Check convergence of parameters
        if (abs(pi - pi_old) < tol and
            abs(mu0 - mu0_old) < tol and
            abs(tau2 - tau2_old) < tol):
            break
        
        pi_old, mu0_old, tau2_old = pi, mu0, tau2
        
    # Compute posterior estimates for mu_i
    # For the slab, the posterior mean given y is:
    slab_posterior_mean = (tau2 / (sampling_vars + tau2)) * mle_treatment_effects + (sampling_vars / (sampling_vars + tau2)) * mu0
    # Final estimate: weighted by posterior probability of slab membership
    mu_hat = r * slab_posterior_mean
    
    if return_all:
        return mu_hat, pi, mu0, tau2
    else:
        return mu_hat


# def bayes_normal_selection_adjusted(
#     mle_estimates: np.ndarray, 
#     selection: np.ndarray,
#     sample_vars: np.ndarray,
#     threshold: float = 0.0,
#     prior_mean: float = 0.0, prior_std: float = 1.0,
# ):
#     """ 
#     Estimate the posterior mean with a normal prior and adjust for selection bias.

#     Params:`
#     -------
#     mle_estimates: np.ndarray, shape = (n_experiments,)
#         The maximum likelihood estimates of the treatment effects.
#     sample_vars: np.ndarray, shape = (n_experiments,)
#         The sample variance of the treatment effects.
#     selection: np.ndarray, shape = (n_experiments,)
#         The selection event for each experiment.
#     threshold: float, default=0.0
#         The threshold for the selection event.
#     prior_mean: float, default=0.0
#         The mean of the normal prior.
#     prior_std: float, default=1.0
#         The standard deviation of the normal prior.
    
#     Returns:
#     --------
#     np.ndarray, shape = (n_experiments,)
#         The posterior mean estimates adjusted for selection bias.
#     """
#     post_mean_arr = np.zeros_like(mle_estimates)

#     def selection_adjusted_posterior_mean(y: float, sigma: float) -> float:
#         """
#         Computes the selection-adjusted posterior mean for a test with observation y.
        
#         Model: 
#             - Likelihood: N(y; mu, sigma^2)
#             - Prior: N(prior_mean, prior_std^2)
#             - Selection event: y > 0, with probability P(y > 0 | mu) = Phi(mu/sigma)

#         The adjusted posterior density is proportional to:
#             f(y|mu) * prior(mu) / Phi(mu/sigma)
        
#         Params:
#         --------
#         - y: observed value (must be > 0 for selected tests)
#         - sigma: standard deviation of the observation noise
        
#         Returns:
#         - selection-adjusted posterior mean for mu.
#         """
        
#         # Define the likelihood: f(y|mu)
#         def likelihood(mu):
#             return norm.pdf(y, loc=mu, scale=sigma)
        
#         # Define the prior: pi(mu)
#         def prior(mu):
#             return norm.pdf(mu, loc=prior_mean, scale=prior_std)
        
#         # Define the adjustment factor: P(S | mu) = Phi(mu/sigma)
#         def selection_prob(mu):
#             # To avoid division by zero, ensure a lower bound for the CDF.
#             return np.maximum(norm.cdf((mu - threshold) / sigma), 1e-12)
        
#         # The integrand for the denominator:
#         def integrand(mu):
#             return likelihood(mu) * prior(mu) / selection_prob(mu)
        
#         # The integrand for the numerator:
#         def integrand_mu(mu):
#             return mu * likelihood(mu) * prior(mu) / selection_prob(mu)
        
#         # Perform numerical integration over a reasonable range.
#         # Using limits (-np.inf, np.inf) for completeness.
#         num, err_num = quad(integrand_mu, -np.inf, np.inf, epsabs=1e-6)
#         den, err_den = quad(integrand, -np.inf, np.inf, epsabs=1e-6)
        
#         if den == 0:
#             raise ValueError("Denominator of the posterior mean integration is zero.")
        
#         return num / den
    
#     for i, (mle, var, select) in enumerate(zip(mle_estimates, sample_vars, selection)):
#         if select:
#             post_mean_arr[i] = selection_adjusted_posterior_mean(mle, np.sqrt(var))
#             # print(mle, post_mean_arr[i])

#     return post_mean_arr


def empirical_bayes_spike_slab_selection_adjusted(
    mle_estimates: np.ndarray, 
    selection: np.ndarray,
    sample_vars: np.ndarray,
    pi: float = 0.5, 
    max_iter=100, tol=1e-6, 
    epsilon=1e-6, 
    n_jobs=-1, verbose=False, 
) -> np.ndarray:
    """ 
    Computes the selection-adjusted posterior mean for a given observed effect y,
    using a spike-and-slab prior with empirical Bayes hyperparameters.

    Params:`
    -------
    mle_estimates: np.ndarray, shape = (n_arms, n_experiments)
        The maximum likelihood estimates of the treatment effects for each arm.
    sample_vars: np.ndarray, shape = (n_experiments,)
        The sample variance of the treatment effects.
    selection: np.ndarray, shape = (n_experiments,)
        The selection event for each experiment.
    threshold: float, default=0.0
        The threshold for the selection event.
    pi: float
        prior probability for spike component
    max_iter: int
        maximum number of iterations for EM algorithm
    tol: float
        tolerance for convergence
    epsilon: float, default=1e-6
        Small variance for approximating the spike
    n_jobs: int
        number of parallel jobs to run
    verbose: bool
        whether to display progress information
        
    Returns:
    --------
    np.ndarray, shape = (n_experiments,)
        The posterior mean estimates adjusted for selection bias.
    """
    post_mean_arr = np.zeros_like(mle_estimates)  # shape = (n_arms, n_experiments)
    n_arms = mle_estimates.shape[0]
    n_experiments = mle_estimates.shape[1]

    # initialize empirical bayes results
    pi_arr = np.zeros(n_arms)
    mu0_arr = np.zeros(n_arms)
    tau2_arr = np.zeros(n_arms)

    # run the empirical Bayes algorithm to estimate the hyperparameters
    for arm_id in range(n_arms):
        _, pi_arr[arm_id], mu0_arr[arm_id], tau2_arr[arm_id] = empirical_bayes_spike_slab(
            mle_estimates[arm_id, :], 
            sample_vars[arm_id, :], 
            pi=pi, max_iter=max_iter, tol=tol, return_all=True
        )

    def selection_adjusted_posterior_mean(
        selected_effect: float, 
        unselected_effect: float, 
        selected_sigma: float, 
        unselected_sigma: float, 
        mu0: float, tau2: float, pi: float, 
    ) -> float:
        """        
        Computes the selection-adjusted posterior mean for an arm in an A/B test, 
        given that the provided arm was selected (i.e., selected_effect > unselected_effect).

        Let arm a1 denote the selected arm. 

        Model:
            - Likelihood: y1 ~ N(mu1, sigma1^2)
            - Prior:    mu1 ~ pi1 * N(0, epsilon1^2) + (1-pi1) * N(mu01, tau1^2)
            - Selection: We select a1 if y1 \geq y2. Under independent normal noise with variance 
                            sigma1^2 and sigma2^2, the selection probability is:
                            P(S1|mu1) = P(S1|mu1, mu2_star) 
                                = Phi((mu1 - mu2_star) / sqrt(sigma1^2 + sigma2^2))
                    
        The unnormalized posterior density is:
            g(mu) = [N(y; mu, sigma^2) * (pi * N(mu; 0, epsilon) + (1-pi) * N(mu; mu0, tau2))]
                    / Phi((mu1 - mu2_star) / sqrt(sigma1^2 + sigma2^2))
        
        The selection-adjusted posterior mean is:
            E(mu|y,S) = (∫ mu * g(mu) dmu) / (∫ g(mu) dmu)
                    
        The adjusted posterior density is proportional to:
            f(y|mu) * prior(mu) / Phi(mu/sigma)
        
        Returns:
        --------
        - selection-adjusted posterior mean for mu.
        """
        
        # Define the likelihood: f(y|mu)
        def likelihood(mu):
            return norm.pdf(selected_effect, loc=mu, scale=selected_sigma)
        
        # Mixture prior: spike-and-slab
        def mixture_prior(mu):
            spike_density = norm.pdf(mu, loc=0, scale=np.sqrt(epsilon))
            slab_density  = norm.pdf(mu, loc=mu0, scale=np.sqrt(tau2))
            return pi * spike_density + (1 - pi) * slab_density
        
        # Define the adjustment factor: P(S | mu) = Phi(mu/sigma)
        def selection_prob(mu):
            # To avoid division by zero, ensure a lower bound for the CDF.
            return np.maximum(norm.cdf((mu - unselected_effect) / np.sqrt(selected_sigma**2 + unselected_sigma**2)), 1e-12)
        
        # The integrand for the denominator:
        def integrand(mu):
            return likelihood(mu) * mixture_prior(mu) / selection_prob(mu)
        
        # The integrand for the numerator:
        def integrand_mu(mu):
            return mu * likelihood(mu) * mixture_prior(mu) / selection_prob(mu)
        
        # Perform numerical integration over a reasonable range.
        # Using limits (-np.inf, np.inf) for completeness.
        num, err_num = quad(integrand_mu, -np.inf, np.inf, epsabs=1e-3)
        den, err_den = quad(integrand, -np.inf, np.inf, epsabs=1e-3)
                    
        if den == 0:
            raise ValueError("Denominator of the posterior mean integration is zero.")
        
        return num / den
    
    def adjust_single_experiment(experiment_id):
        # Extract the selected and unselected effects and variances
        selected_arm = selection[experiment_id]
        unselected_arm = 1 - selected_arm
        selected_effect = mle_estimates[selected_arm, experiment_id]
        unselected_effect = mle_estimates[unselected_arm, experiment_id]
        selected_sigma = np.sqrt(sample_vars[selected_arm, experiment_id])
        unselected_sigma = np.sqrt(sample_vars[unselected_arm, experiment_id])
        pi = pi_arr[selected_arm]
        mu0 = mu0_arr[selected_arm]
        tau2 = tau2_arr[selected_arm]

        # Compute the selection-adjusted posterior mean
        return selected_arm, selection_adjusted_posterior_mean(
            selected_effect, unselected_effect, selected_sigma, unselected_sigma, mu0, tau2, pi
        )
    
    adjust_results_list = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(adjust_single_experiment)(experiment_id) for experiment_id in range(n_experiments)
    )

    for idx, (selected_arm, post_mean) in enumerate(adjust_results_list):
        post_mean_arr[selected_arm, idx] = post_mean

    return post_mean_arr