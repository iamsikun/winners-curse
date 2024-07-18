import os 
import sys 
sys.path.insert(0, os.path.abspath('.'))
from typing import Iterable
from joblib import Parallel, delayed

import numpy as np 

from scipy.stats import ks_2samp
from core.base import BaseEstimator


def calculate_treatment_effect_with_dim(outcomes: np.ndarray, treatments: np.ndarray) -> float:
    """ 
    This function calculates the treatment effect using difference in means. 

    Params:
    ------
    outcomes: np.ndarray
        The observed outcomes
    treatments: np.ndarray
        The treatment assignment
    
    Returns:
    -------
    treatment_effect: float
        The estimated treatment effect
    """
    # check if either the treated or control group is empty
    if (treatments == 0).sum() == 0 or (treatments == 1).sum() == 0:
        return np.nan 

    # calculate the treatment effect
    treatment_effect = outcomes[treatments == 1].mean() - outcomes[treatments == 0].mean()

    return treatment_effect


def segment_assignment(customers: np.ndarray, segment_func: callable) -> np.ndarray:
    """
    Assign customers to segments.

    Params:
    ------
    test_customers: np.ndarray, shape = (sample_size, n_features)
        The observed covariates of the test customers
    segment_func: callable
        The function that assigns customers to segments

    Returns:
    -------
    customer_segment_arr: np.ndarray, shape = (sample_size, )
        The segment assignment for each customer
    """
    return np.array([segment_func(customer) for customer in customers])


def estimate_segment_treatment_effects(
    covariates: np.ndarray, 
    treatments: np.ndarray, 
    outcomes: np.ndarray, 
    segment_func: callable, 
    n_segments: int, 
) -> np.ndarray:
    """
    Estimate the treatment effects for each segment of customers

    Params:
    ------
    covariates: np.ndarray, shape = (sample_size, n_features)
        The observed covariates
    treatments: np.ndarray, shape = (sample_size, 1)
        The treatment assignment
    outcomes: np.ndarray, shape = (sample_size, 1)
        The observed outcomes
    segment_func: callable
        The function that assigns customers to segments
    n_segments: int
        The number of segments

    Returns:
    -------
    segment_te_arr: np.ndarray, shape = (n_segments, )
        The estimated treatment effects for each segment
    """
    # assign each customer to a segment
    customer_segment_arr = segment_assignment(covariates, segment_func)  # shape = (sample_size, )

    # fit OLS estimator for each segment
    segment_te_arr = np.zeros((n_segments, ))

    for i in range(n_segments):
        segment_mask = (customer_segment_arr == i)
        segment_te_arr[i] = calculate_treatment_effect_with_dim(
            outcomes[segment_mask], 
            treatments[segment_mask]
        )

    return segment_te_arr


def estimate_segment_treatment_effects_with_bootstrap(
    covariates: np.ndarray, 
    treatments: np.ndarray, 
    outcomes: np.ndarray, 
    segment_func: callable, 
    n_segments: int, 
    n_bootstraps: int, 
    bootstrap_sample_size: int = None, 
    n_jobs: int = -1  # use all available cores by default
) -> np.ndarray: 
    """
    Estimate the treatment effects for each segment of customers

    Params:
    ------
    covariates: np.ndarray, shape = (n_obs, n_features)
        The observed covariates
    treatments: np.ndarray, shape = (n_obs, 1)
        The treatment assignment
    outcomes: np.ndarray, shape = (n_obs, 1)
        The observed outcomes
    segment_func: callable
        The function that assigns customers to segments
    n_segments: int
        The number of segments
    n_bootstraps: int
        The number of bootstrap samples to use
    bootstrap_sample_size: int
        The size of the bootstrap sample, default is the same as the original sample size
    n_jobs: int
        The number of jobs to use for parallelization

    Returns:
    -------
    boot_segment_te_arr: np.ndarray, shape = (n_bootstraps, n_segments)
        The estimated treatment effects for each segment for each bootstrap sample
    """
    # parameters
    sample_size = covariates.shape[0]
    bootstrap_sample_size = sample_size if bootstrap_sample_size is None else bootstrap_sample_size

    # sample indices with standard bootstrap, shape (n_bootstraps, sample_size)
    boot_index_arr = np.random.choice(
        np.arange(sample_size),  # index of the original data
        size=(n_bootstraps, bootstrap_sample_size), 
        replace=True
    )

    def process_bootstrap(i: int) -> np.ndarray:
        # get the bootstrap sample
        boot_covariates = covariates[boot_index_arr[i]]  # (sample_size, n_features)
        boot_treatments = treatments[boot_index_arr[i]]  # (sample_size, 1)
        boot_outcomes = outcomes[boot_index_arr[i]]  # (sample_size, 1)

        return estimate_segment_treatment_effects(
            covariates=boot_covariates,
            treatments=boot_treatments,
            outcomes=boot_outcomes,
            segment_func=segment_func,
            n_segments=n_segments
        )
    
    boot_segment_te_arr =  Parallel(n_jobs=n_jobs)(delayed(process_bootstrap)(i) for i in range(n_bootstraps))
    
    return np.array(boot_segment_te_arr)


def segment_targeting_obj_func(
    test_customers: np.ndarray, targ_decision: np.ndarray, params: np.ndarray, 
    segment_func: callable, 
) -> float: 
    """
    The objective function for the segment targeting problem

    Params:
    ------
    test_customers: np.ndarray, shape = (sample_size, n_features)
        The observed covariates of the test customers
    targ_decision: np.ndarray[Binary], shape = (sample_size, )
        The targeting decision
    params: np.ndarray, shape = (n_segments, )
        The estimated treatment effects for each segment
    segment_func: callable
        The function that assigns customers to segments
    
    Returns:
    -------
    obj_val: float
        The objective value
    """
    # parameters check
    assert test_customers.shape[0] == targ_decision.shape[0], "Input shapes do not match"

    # assign treatment effects to each customer based on their segments
    customer_segment_arr = segment_assignment(test_customers, segment_func)  # shape = (sample_size, )

    # assign the treatment effects to the test customers
    customer_te_arr = params[customer_segment_arr]  # shape = (sample_size, )

    # calculate the objective value
    obj_val = customer_te_arr.T @ targ_decision 

    return obj_val

def segment_targeting_optimize(
    test_customers: np.ndarray, params: np.ndarray, segment_func: callable, budget: int, 
) -> tuple:
    """ 
    Optimize the segment targeting problem. 

    Params:
    ------
    test_customers: np.ndarray, shape = (sample_size, n_features)
        The observed covariates of the test customers
    params: np.ndarray, shape = (n_segments, )
        The estimated treatment effects for each segment
    segment_func: callable
        The function that assigns customers to segments
    budget: int
        The number of customers to select

    Returns:
    -------
    opt_targ_decision: np.ndarray[Binary], shape = (n_obs, )
        The optimal targeting decision
    opt_targ_est: float
        The estimated treatment effect
    """
    # initialize placeholders
    opt_targ_decision = np.zeros((test_customers.shape[0], ))  # shape = (sample_size, )

    # assign treatment effects to each customer based on their segments
    customer_segment_arr = segment_assignment(test_customers, segment_func)  # shape = (sample_size, )

    # assign the treatment effects to the test customers
    customer_te_arr = params[customer_segment_arr]  # shape = (sample_size, )

    # sort the customers by their estimated treatment effects
    sorted_customers_indices = np.argsort(-customer_te_arr)  # sort in descending order, shape = (sample_size, )

    # select the top-k customers
    opt_targ_decision[sorted_customers_indices[:budget]] = 1
    opt_targ_est = customer_te_arr[sorted_customers_indices[:budget]].sum()

    return opt_targ_decision, opt_targ_est


def segment_targeting_optimize_with_bootstrap(
    test_customers: np.ndarray, params: np.ndarray, segment_func: callable, budget: int,
) -> tuple:
    """ 
    Optimzie the segment targeting problem in batch mode, where the parameters come from multiple 
    bootstrap samples.

    Params:
    ------
    test_customers: np.ndarray, shape = (sample_size, n_features)
        The observed covariates of the test customers
    params: np.ndarray, shape = (n_bootstrap, n_segments)
        The estimated treatment effects for each segment for each bootstrap sample
    segment_func: callable
        The function that assigns customers to segments
    budget: int
        The number of customers to select

    Returns:
    -------
    boot_opt_targ_decisions: np.ndarray[Binary], shape = (n_bootstrap, sample_size)
        The optimal targeting decision for each bootstrap sample
    boot_targ_ests: np.ndarray, shape = (n_bootstrap, )
        The estimated treatment effect for each bootstrap sample
    """
    # extract parameters
    n_bootstraps = params.shape[0]

    # assign treatment effects to each customer based on their segments
    customer_segment_arr = segment_assignment(test_customers, segment_func)  # shape = (sample_size, 

    # assign the treatment effects to the test customers
    boot_customer_te_arr = params[:, customer_segment_arr]  # shape = (n_bootstrap, sample_size)

    # optimize targeting for each bootstrap sample
    # the matrix stores the indices of the top-k customers for each bootstrap sample
    boot_opt_targ_decisions_with_index = np.argsort(-boot_customer_te_arr, axis=1)[:, :budget]  # (n_bootstrap, budget)

    # turn boot_opt_targ_decisions to binary masks with shape (n_bootstrap, sample_size)
    # with 1 indicating the selected customers, and 0 otherwise
    boot_opt_targ_decisions = np.zeros_like(boot_customer_te_arr)
    boot_opt_targ_decisions[np.repeat(np.arange(n_bootstraps)[:, None], budget, axis=1), boot_opt_targ_decisions_with_index] = 1

    # calculate the value of the targeting policy for each bootstrap sample using the bootstrap estimates
    boot_targ_ests = (boot_customer_te_arr * boot_opt_targ_decisions).sum(axis=1)  # (n_bootstrap, )

    return boot_opt_targ_decisions, boot_targ_ests
    

def choose_best_m(distribution_list: Iterable[np.ndarray]) -> np.ndarray:
    """ 
    Given a list of bootstrap distributions coming from different bootstrap sample sizes (m), 
    choose the best m based on the discrepancy between the distributions. 

    This function follows the rule in Bickel and Sakov (2008, Statistica Sinica)

    Params:
    -------
    distribution_list: Iterable[np.ndarray]
        a list of bootstrap distributions, each element is an array of shape (n_bootstraps, )

    Returns:
    --------
    np.ndarray
        an element in the distribution_list 
    """
    # calculate pairwise discrepancies
    discp_list = [None] * (len(distribution_list) - 1)  # list of pairwise discrepancies
    for idx, (prev_dstn, current_dstn) in enumerate(zip(distribution_list[:-1], distribution_list[1:])):
        ks_stat, _ = ks_2samp(prev_dstn, current_dstn)
        discp_list[idx] = ks_stat

    # choose the m with the smallest discrepancy
    min_discp_idx = np.argmin(discp_list)

    return distribution_list[min_discp_idx]


class SegmentTargetingPlugin(BaseEstimator):
    """ 
    A two-stage estimator that solves a plugin problem first, and then evaluate the plugin policy
    with doubly robust estimator (inverse probability weighting)
    """
    def __init__(self, segment_func: callable, n_segments: int):
        # attributes
        self.segment_func = segment_func  
        self.n_segments = n_segments

        # placeholders
        self.segment_te_arr = None  # (n_segments, )

        self.opt_targ_decision = None 
        self.opt_targ_est = np.nan

    @BaseEstimator.check_input_decorator
    def fit(self, covariates: np.ndarray, treatments: np.ndarray, outcomes: np.ndarray) -> object:
        """
        Estimate the treatment effects

        Params:
        ------
        covariates: np.ndarray, shape = (n_obs, n_features)
            The observed covariates
        treatments: np.ndarray, shape = (n_obs, 1)
            The treatment assignment
        outcomes: np.ndarray, shape = (n_obs, 1)
            The observed outcomes
        """
        self.segment_te_arr = estimate_segment_treatment_effects(
            covariates=covariates, 
            treatments=treatments, 
            outcomes=outcomes, 
            segment_func=self.segment_func, 
            n_segments=self.n_segments
        )  # (n_segments, )

        return self

    def optimize(self, test_customers: np.ndarray, budget: int = 1) -> tuple:
        """
        Optimize the plugin policy using the estimated treatment effects

        Params:
        ------
        test_customers: np.ndarray, shape = (n_obs, n_features)
            The observed covariates of the test customers
        budget: int
            The number of customers to select
        """
        self.opt_targ_decision, self.opt_targ_est = segment_targeting_optimize(
            test_customers=test_customers,
            params=self.segment_te_arr,
            segment_func=self.segment_func,
            budget=budget
        )

        return self.opt_targ_decision, self.opt_targ_est


class SegmentTargetingValueCorrection(BaseEstimator):
    """ 
    Correction estimator for targeting value on the value function level

    The empirical distribution of the value function is constructed via vannila nonparametric bootstrap.
    """
    def __init__(self, segment_func: callable, n_segments: int):
        # attributes
        self.segment_func = segment_func 
        self.n_segments = n_segments

        # placeholders
        self.n_bootstraps = None  # number of bootstrap samples
        self.boot_segment_te_arr = None  # (n_bootstrap, n_segments)

    @BaseEstimator.check_input_decorator
    def fit(
        self, covariates: np.ndarray, treatments: np.ndarray, outcomes: np.ndarray, 
        n_bootstraps: int = 1000, n_jobs: int = -1
    ):
        """ 
        Estimate the treatment effect for each segment using bootstrap samples.

        Params:
        ------
        covariates: np.ndarray, shape = (n_obs, n_features)
            The observed covariates
        treatments: np.ndarray, shape = (n_obs, 1)
            The treatment assignment
        outcomes: np.ndarray, shape = (n_obs, 1)
            The observed outcomes
        n_bootstraps: int
            The number of bootstrap samples to use
        n_jobs: int
            The number of jobs to use for parallelization
        """
        # fill in placeholders
        self.n_bootstraps = n_bootstraps
        
        self.boot_segment_te_arr = estimate_segment_treatment_effects_with_bootstrap(
            covariates=covariates, 
            treatments=treatments, 
            outcomes=outcomes, 
            segment_func=self.segment_func, 
            n_segments=self.n_segments, 
            n_bootstraps=self.n_bootstraps, 
            n_jobs=n_jobs
        )

        return self
    
    def estimate(
        self, test_customers: np.ndarray, budget: int, plugin_estmr: SegmentTargetingPlugin, 
    ) -> float:
        """   

        Params:
        -------
        test_customers: np.ndarray, shape = (sample_size, n_features)
            The observed covariates
        budget: int
            The number of customers to target
        plugin_estmr: SegmentTargetingPlugin
            The plugin estimator for the targeting value

        Returns:
        -------
        corrected_targ_est: float
            The corrected targeting value estimate
        """
        # make sure the plugin estimator is fitted
        assert plugin_estmr.segment_te_arr is not None, "Plugin estimator must be fitted first."

        # solve for the targeting decisions and value estimates for each bootstrap
        # boot_opt_targ_decisions has shape (n_bootstraps, sample_size)
        # boot_targ_ests has shape (n_bootstraps, )
        boot_opt_targ_decisions, boot_targ_ests = segment_targeting_optimize_with_bootstrap(
            test_customers=test_customers, 
            params=self.boot_segment_te_arr, 
            segment_func=self.segment_func, 
            budget=budget, 
        )
        
        # evaluate bootstrap policy using plugin estimates
        boot_targ_est_with_emp = np.array([
            segment_targeting_obj_func(
            test_customers=test_customers, 
            targ_decision=boot_opt_targ_decisions[i], 
            params=plugin_estmr.segment_te_arr, 
            segment_func=self.segment_func
        ) for i in range(self.n_bootstraps)])

        # calculate correction term
        correction = np.nanmean(boot_targ_ests) - np.nanmean(boot_targ_est_with_emp)

        # calculate the plugin targeting value estimate
        _, plugin_est = plugin_estmr.optimize(
            test_customers=test_customers, budget=budget
        )

        # calculate the corrected treatment effect estimate
        return plugin_est - correction
        

class SegmentTargetingMNBValueCorrection(BaseEstimator):
    """ 
    Correction estimator for targeting value on the value function level

    The empirical distribution of the value function is constructed via 
    m-out-of-n bootstrap. 

    Rule for selecting the best m follows (Bickle and Sakov 2008, Statistica Sinica)
    """
    def __init__(self, segment_func: callable, n_segments: int):
        self.segment_func = segment_func 
        self.n_segments = n_segments

        # placeholders
        self.n_bootstraps = None  # number of bootstrap samples
        self.m_list = []
        # a list of bootstrap treatment effect estimates for each m, 
        # each element is an array of shape (n_bootstraps, n_segments)
        self.m_boot_segment_te_list = []  

    @BaseEstimator.check_input_decorator
    def fit(
        self, covariates: np.ndarray, treatments: np.ndarray, outcomes: np.ndarray, 
        n_bootstraps: int = 1000, q: float = 0.9, max_j: int = 20, n_jobs: int = -1
    ):
        """ 
        Estimate the treatment effect for each segment using m-out-of-n bootstrap

        Params:
        -------
        covariates: np.ndarray, shape (n_obs, n_features)
            the covariates of each customer in the training set
        treatments: np.ndarray, shape (n_obs, 1)
            the treatment assignments of each customer in the training set
        outcomes: np.ndarray, shape (n_obs, 1)
            the observed outcomes of each customer in the training set
        n_bootstraps: int, default 1000
            the number of bootstrap samples
        q: float, default 0.9
            parameter of m-out-of-n bootstrap
        max_j: int, default 20
            the maximum power of q to calculate the number of bootstrap samples
        n_jobs: int, default -1
            the number of jobs to use for parallelization

        """
        # fill in placeholders
        self.n_bootstraps = n_bootstraps
        sample_size = covariates.shape[0] 
        self.m_list = np.array([int(q**j * sample_size) for j in range(max_j)])
        self.m_boot_segment_te_list = [None] * len(self.m_list)

        # create bootstraps for each m
        for m_idx, m in enumerate(self.m_list):
            self.m_boot_segment_te_list[m_idx] = estimate_segment_treatment_effects_with_bootstrap(
                covariates=covariates, 
                treatments=treatments, 
                outcomes=outcomes, 
                segment_func=self.segment_func, 
                n_segments=self.n_segments, 
                n_bootstraps=self.n_bootstraps, 
                bootstrap_sample_size=m, 
                n_jobs=n_jobs
            )

        return self
    
    def estimate(
        self, test_customers: np.ndarray, budget: int, plugin_estmr: SegmentTargetingPlugin, 
    ) -> float:
        """   
        Correction estimator for the actual value of the plugin estimator

        Params:
        -------
        test_customers: np.ndarray, shape (n_obs, n_features)
            the covariates of each customer to be targeted
        budget: int
            the number of customers to target
        plugin_estmr: SegmentTargetingPlugin
            the plugin estimator for the targeting value function

        Returns:
        --------
        float
            the corrected estimate of the targeting value function
        """
        # make sure the plugin estimator is fitted
        assert plugin_estmr.segment_te_arr is not None, "Plugin estimator must be fitted first."

        # initialize the bootstrap distribution of the correction term for each m
        correction_dstn_list = [None] * len(self.m_list)

        for m_idx in range(len(self.m_list)):
            # solve for the targeting decisions and value estimates for each bootstrap
            # boot_opt_targ_decisions has shape (n_bootstraps, n_obs)
            # boot_targ_ests has shape (n_bootstraps, )
            boot_opt_targ_decisions, boot_targ_ests = segment_targeting_optimize_with_bootstrap(
                test_customers=test_customers, 
                params=self.m_boot_segment_te_list[m_idx], 
                segment_func=self.segment_func, 
                budget=budget, 
            )
            
            # evaluate bootstrap policy using plugin estimates
            boot_targ_est_with_emp = np.array([
                segment_targeting_obj_func(
                test_customers=test_customers, 
                targ_decision=boot_opt_targ_decisions[i], 
                params=plugin_estmr.segment_te_arr, 
                segment_func=self.segment_func
            ) for i in range(len(boot_opt_targ_decisions))])  # shape (n_bootstraps, )

            # calculate correction term and store in the distribution list
            correction_dstn_list[m_idx] = boot_targ_ests - boot_targ_est_with_emp  # shape (n_bootstraps, )

        # choose the best m based on the discrepancy between the distributions
        best_correction_dstn = choose_best_m(correction_dstn_list)  # (n_bootstraps, )

        # calculate the plugin targeting value estimate
        _, plugin_est = plugin_estmr.optimize(
            test_customers=test_customers, budget=budget
        )

        # calculate the corrected treatment effect estimate
        return plugin_est - np.nanmean(best_correction_dstn)


class SegmentTargetingNBValueCorrection(BaseEstimator):
    """ 
    Correction estimator for targeting value on the value function level

    The empirical distribution of the value function is constructed via numerical bootstrap. 
    In this implementation, instead of perturbing the data itself, we perturb the CATE estiamtes. 
    """
    def __init__(self, segment_func: callable, n_segments: int):
        self.segment_func = segment_func 
        self.n_segments = n_segments

        # place holders
        self.n_bootstraps = None  # number of bootstrap samples
        self.perturbation_multiplier = None  # perturbation multiplier
        self.boot_segment_te_arr = None  # (n_bootstraps, n_segments)

    @BaseEstimator.check_input_decorator
    def fit(
        self, covariates: np.ndarray, treatments: np.ndarray, outcomes: np.ndarray, 
        n_bootstraps: int = 100, epsilon_n_pow: float = -0.45, n_jobs: int = -1, 
    ):
        """ 
        Estimate the treatment effect for each segment using bootstrap samples.

        Params:
        ------
        covariates: np.ndarray, shape = (n_obs, n_features)
            The observed covariates
        treatments: np.ndarray, shape = (n_obs, 1)
            The treatment assignment
        outcomes: np.ndarray, shape = (n_obs, 1)
            The observed outcomes
        n_bootstraps: int
            The number of bootstrap samples to use
        epsilon_n_pow: float
            The power of the sample size to calculate the perturbation multiplier
        n_jobs: int
            The number of jobs to use for parallelization
        """
        # fill in placeholders
        self.n_bootstraps = n_bootstraps
        sample_size = covariates.shape[0]
        # get perturbed treatment effect estimates
        self.perturbation_multiplier = (sample_size ** (epsilon_n_pow)) * np.sqrt(sample_size)

        self.boot_segment_te_arr = estimate_segment_treatment_effects_with_bootstrap(
            covariates=covariates, 
            treatments=treatments, 
            outcomes=outcomes, 
            segment_func=self.segment_func, 
            n_segments=self.n_segments, 
            n_bootstraps=self.n_bootstraps, 
            n_jobs=n_jobs
        )

        return self
    
    def estimate(
        self, test_customers: np.ndarray, budget: int, plugin_estmr: SegmentTargetingPlugin, 
    ) -> float:
        # make sure the plugin estimator is fitted
        assert plugin_estmr.segment_te_arr is not None, "Plugin estimator must be fitted first."

        # get perturbed treatment effect estimates
        plugin_segment_te_est = plugin_estmr.segment_te_arr.reshape(1, -1)  # (1, n_segments)
        # (n_bootstraps, n_segments)
        perturbed_segment_te_arr = plugin_segment_te_est + self.perturbation_multiplier * (
            plugin_segment_te_est - self.boot_segment_te_arr
        )

        # solve for the targeting decisions and value estimates for each bootstrap
        # pertb_opt_targ_decisions has shape (n_bootstraps, n_obs)
        # pertb_targ_ests has shape (n_bootstraps, )
        pertb_opt_targ_decisions, pertb_targ_ests = segment_targeting_optimize_with_bootstrap(
            test_customers=test_customers, 
            params=perturbed_segment_te_arr,  # use the perturbed treatment effect estimates
            segment_func=self.segment_func, 
            budget=budget, 
        )
        
        # evaluate bootstrap policy using empirical treatment effect estimates
        pertb_targ_est_with_emp = np.array([
            segment_targeting_obj_func(
            test_customers=test_customers, 
            targ_decision=pertb_opt_targ_decisions[i], 
            params=plugin_estmr.segment_te_arr,  # use the empirical treatment effect estimates
            segment_func=self.segment_func
        ) for i in range(self.n_bootstraps)])

        # calculate correction term
        correction = np.nanmean(pertb_targ_ests) - np.nanmean(pertb_targ_est_with_emp)

        # calculate the plugin targeting value estimate
        _, plugin_est = plugin_estmr.optimize(
            test_customers=test_customers, budget=budget
        )

        # calculate the corrected treatment effect estimate
        return plugin_est - correction


class SegmentTargetingErrorCorrection(SegmentTargetingValueCorrection):
    """
    Correction estimator for targeting value on the prediction error level. 
    This estimator requires the analytical formulation for the winner's curse (value function bias)
    as a function of the estimation errors. 

    The empirical distribution of estimation errors is constructed via vannila nonparametric bootstrap, 
    where the bootstrap sample size is equal to the sample size of the original data. 
    """
    
    def estimate(
        self, test_customers: np.ndarray, budget: int, plugin_estmr: SegmentTargetingPlugin, 
    ) -> float:
        """   
        Params:
        -------
        test_customers: np.ndarray, shape = (sample_size, n_features)
            The observed covariates
        budget: int
            The number of customers to target
        plugin_estmr: SegmentTargetingPlugin
            The plugin estimator for the targeting value

        Returns:
        -------
        corrected_targ_est: float
            The corrected targeting value estimate
        """
        # make sure the plugin estimator is fitted
        assert plugin_estmr.segment_te_arr is not None, "Plugin estimator must be fitted first."

        # assign treatment effects to each customer based on their segments
        customer_segment_arr = segment_assignment(test_customers, self.segment_func)

        # assign the bootstrap treatment effects to the test customers
        boot_customer_te_arr = self.boot_segment_te_arr[:, customer_segment_arr]  # shape = (n_bootstrap, sample_size)

        # assign the empirical treatment effects to the test customers
        emp_customer_te_arr = plugin_estmr.segment_te_arr.reshape(1, -1)[:, customer_segment_arr]
        
        # calculate the estimation error for each bootstrap
        xi_arr = boot_customer_te_arr - emp_customer_te_arr

        # solve for the targeting decisions and value estimates for each bootstrap
        # boot_opt_targ_decisions has shape (n_bootstraps, sample_size)
        boot_opt_targ_decisions, _ = segment_targeting_optimize_with_bootstrap(
            test_customers=test_customers, 
            params=self.boot_segment_te_arr, 
            segment_func=self.segment_func, 
            budget=budget, 
        )

        # calculate the correction term
        correction = np.mean(np.sum(boot_opt_targ_decisions * xi_arr, axis=1))

        # calculate plugin estimate
        # calculate the plugin targeting value estimate
        _, plugin_est = plugin_estmr.optimize(
            test_customers=test_customers, budget=budget
        )

        # calculate the corrected treatment effect estimate
        return plugin_est - correction
    

class SegmentTargetingMNBErrorCorrection(SegmentTargetingMNBValueCorrection):
    """ 
    Correction estimator for targeting value on the error level

    The empirical distribution of the value function is constructed via 
    m-out-of-n bootstrap. 
    """
    def estimate(
        self, test_customers: np.ndarray, budget: int, plugin_estmr: SegmentTargetingPlugin, 
    ) -> float:
        """   
        Correction estimator for the actual value of the plugin estimator

        Params:
        -------
        test_customers: np.ndarray, shape (n_obs, n_features)
            the covariates of each customer to be targeted
        budget: int
            the number of customers to target
        plugin_estmr: SegmentTargetingPlugin
            the plugin estimator for the targeting value function

        Returns:
        --------
        float
            the corrected estimate of the targeting value function
        """
        # make sure the plugin estimator is fitted
        assert plugin_estmr.segment_te_arr is not None, "Plugin estimator must be fitted first."

        # initialize the bootstrap distribution of the correction term for each m
        correction_dstn_list = [None] * len(self.m_list)

        # assign treatment effects to each customer based on their segments
        customer_segment_arr = segment_assignment(test_customers, self.segment_func)

        # assign the empirical treatment effects to the test customers
        emp_customer_te_arr = plugin_estmr.segment_te_arr.reshape(1, -1)[:, customer_segment_arr]

        for m_idx in range(len(self.m_list)):
            # assign the bootstrap treatment effects to the test customers
            boot_customer_te_arr = self.m_boot_segment_te_list[m_idx][:, customer_segment_arr]  # shape = (n_bootstrap, sample_size)
            
            # calculate the estimation error for each bootstrap
            xi_arr = boot_customer_te_arr - emp_customer_te_arr  # shape = (n_bootstrap, sample_size)

            # solve for the targeting decisions and value estimates for each bootstrap
            # boot_opt_targ_decisions has shape (n_bootstraps, sample_size)
            boot_opt_targ_decisions, _ = segment_targeting_optimize_with_bootstrap(
                test_customers=test_customers, 
                params=self.m_boot_segment_te_list[m_idx], 
                segment_func=self.segment_func, 
                budget=budget, 
            )

            # calculate the correction term
            correction_dstn_list[m_idx] = np.sum(boot_opt_targ_decisions * xi_arr, axis=1)  # shape = (n_bootstrap,)

        # choose the best m based on the discrepancy between the distributions
        best_correction_dstn = choose_best_m(correction_dstn_list)  # (n_bootstraps, )

        # calculate the plugin targeting value estimate
        _, plugin_est = plugin_estmr.optimize(
            test_customers=test_customers, budget=budget
        )

        # calculate the corrected treatment effect estimate
        return plugin_est - np.nanmean(best_correction_dstn)


class SegmentTargetingNBErrorCorrection(SegmentTargetingNBValueCorrection):
    """
    Correction estimator for targeting value on the prediction error level. 
    This estimator requires the analytical formulation for the winner's curse (value function bias)
    as a function of the estimation errors. 

    The empirical distribution of estimation errors is constructed via numerical bootstrap, 
    """
    def estimate(
        self, test_customers: np.ndarray, budget: int, plugin_estmr: SegmentTargetingPlugin, 
    ) -> float:
        # make sure the plugin estimator is fitted
        assert plugin_estmr.segment_te_arr is not None, "Plugin estimator must be fitted first."

        # get perturbed treatment effect estimates
        plugin_segment_te_est = plugin_estmr.segment_te_arr.reshape(1, -1)  # (1, n_segments)
        # (n_bootstraps, n_segments)
        pertb_segment_te_arr = plugin_segment_te_est + self.perturbation_multiplier * (
            plugin_segment_te_est - self.boot_segment_te_arr
        )

        # assign treatment effects to each customer based on their segments
        customer_segment_arr = segment_assignment(test_customers, self.segment_func)

        # assign the bootstrap treatment effects to the test customers
        pertb_customer_te_arr = pertb_segment_te_arr[:, customer_segment_arr]  # shape = (n_bootstrap, sample_size)

        # assign the empirical treatment effects to the test customers
        emp_customer_te_arr = plugin_estmr.segment_te_arr.reshape(1, -1)[:, customer_segment_arr]
        
        # calculate the estimation error for each bootstrap
        xi_arr = pertb_customer_te_arr - emp_customer_te_arr

        # solve for the targeting decisions and value estimates for each bootstrap
        # pertb_opt_targ_decisions has shape (n_bootstraps, sample_size)
        pertb_opt_targ_decisions, _ = segment_targeting_optimize_with_bootstrap(
            test_customers=test_customers, 
            params=pertb_segment_te_arr, 
            segment_func=self.segment_func, 
            budget=budget, 
        )

        # calculate the correction term
        correction = np.mean(np.sum(pertb_opt_targ_decisions * xi_arr, axis=1))

        # calculate plugin estimate
        # calculate the plugin targeting value estimate
        _, plugin_est = plugin_estmr.optimize(
            test_customers=test_customers, budget=budget
        )

        # calculate the corrected treatment effect estimate
        return plugin_est - correction

# class BinaryParamErrorCorrection(BinaryErrorCorrection):
#     """ 
#     Correction estimator for targeting value on the error level. 
#     This estimator uses parametric bootstrap to estimate the empirical distribution of the errors
#     """
#     def __init__(self, group_func: callable, n_groups: int):
#         """  

#         Params:
#         -------
#         group_func: callable, a function that assigns each customer to a group
#         n_groups: int, the number of groups

#         """
#         # attributes
#         self.group_func = group_func
#         self.n_groups = n_groups

#         # placeholders
#         self.te_var_list = [None for _ in range(n_groups)]  # (n_groups, )

#     def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
#         """  
#         Fit the OLS estimator for each group.

#         Params:
#         -------
#         X: np.ndarray, shape (n_obs, d), the covariates
#         T: np.ndarray, shape (n_obs, 1), the treatment assignment
#         Y: np.ndarray, shape (n_obs, 1), the outcome
#         """
#         # check input shapes
#         self.check_input(X, T, Y)

#         # get the group assignment for each customer
#         group_arr = np.array([self.group_func(x) for x in X])

#         # fit the OLS estimator for each group
#         for i in range(self.n_groups):
#             idx = np.where(group_arr == i)[0]  # (n_obs_in_group_i, )
#             ols = OLS(Y[idx], T[idx]).fit()  # fit the OLS estimator
#             self.te_var_list[i] = UnivariateGaussian(mean=ols.params[0], std=ols.bse[0])

#         return self 
    
#     def estimate_targeting_value(self, X: np.ndarray, budget: int = 1, n_bootstraps: int = 100) -> float:
#         # assign each customer to a group
#         group_arr = np.array([self.group_func(x) for x in X])  # (n_obs, )

#         # sample the treatment effect for each customer
#         boot_est_arr = np.concatenate([te_var.sample(n_bootstraps).reshape(-1, 1) for te_var in self.te_var_list], axis=1)[:, group_arr]

#         # calculate empirical estimation error
#         xi_arr = boot_est_arr - np.array([te_var.mean for te_var in self.te_var_list])[group_arr]

#         # create a mask to identify the treated individuals within each bootstrap sample
#         # if the treatment effect estimate is among the largests, the mask is 1, otherwise 0
#         # if ties, select the ones with smaller indexes
#         mask_arr = np.zeros_like(xi_arr)
#         # sort the treatment effect estimates in descending order for each bootstrap sample
#         sorted_columns = np.argsort(-boot_est_arr, axis=1)[:, :budget]

#         # set the mask to 1 for the treated individuals
#         rows = np.repeat(np.arange(boot_est_arr.shape[0])[:, None], budget, axis=1)
#         mask_arr[rows, sorted_columns] = 1

#         del sorted_columns, rows

#         # calculate the correction term
#         correction = np.mean(np.sum(mask_arr * xi_arr, axis=1))

#         # calculate plugin estimate
#         mean_te_arr = np.array([te_var.mean for te_var in self.te_var_list])[group_arr]
#         plugin_est = -np.sort(-mean_te_arr)[:budget].sum()

#         # calculate the corrected treatment effect estimate
#         return plugin_est - correction


# class BinaryBayesErrorCorrection(BinaryErrorCorrection):
#     """ 
#     Correction estimator for targeting value on the error level.
#     This estimator uses Bayesian bootstrap to estimate the empirical distribution of the errors
#     """
#     def __init__(self, group_func: callable, n_groups: int):
#         self.group_func = group_func 
#         self.n_groups = n_groups

#         # place holders
#         self.n_bootstraps = None  # number of bootstrap samples
#         self.boot_te_arr = None  # (n_bootstrap, n_groups)
#         self.plugin_estimator = BinaryPlugIn(group_func=group_func, n_groups=n_groups)

#     def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray, n_bootstraps: int = 100):
#         # fill in placeholders
#         self.n_bootstraps = n_bootstraps

#         # assign each customer to a group
#         group_arr = np.array([self.group_func(x) for x in X])  # (n_obs, )

#         self.boot_te_arr = np.array([
#             self.te_with_weighted_dim(
#                 group=group_arr, t=T, y=Y, n_groups=self.n_groups
#             ) for _ in range(n_bootstraps)
#         ])

#         self.plugin_estimator.fit(X, T, Y)

#         return self
    
#     @staticmethod
#     def te_with_weighted_dim(
#         group: np.ndarray, t: np.ndarray, y: np.ndarray, n_groups: int
#     ) -> list: 
#         """ 
#         This function calculates the treatment effect per group with binary treatments
#         using the difference in weighted means, where the weights are generated with 
#         Dirichlet distribution.

#         params:
#         ------
#         group: np.ndarray (None, 1)
#             The group assignment of each customer.
#         t: np.ndarray (None, 1)
#             The treatment assignment of each customer.
#         y: np.ndarray (None, 1)
#             The outcome of each customer.
#         weights: np.ndarray (None, 1)
#             The weights for each customer.
#         """
#         if check_empty_treated_control(
#             group=group, t=t, y=y, n_groups=n_groups
#         ):
#             return [np.nan] * n_groups

#         # calculate the treatment effect per group
#         te_list = [None] * n_groups
#         for group_id in range(n_groups):
#             # sample Dirichlet weights
#             treated_group_weights = np.random.dirichlet(np.ones(len(
#                 y[group == group_id][t[group == group_id] == 1]
#             )))
#             control_group_weights = np.random.dirichlet(np.ones(len(
#                 y[group == group_id][t[group == group_id] == 0]
#             )))

#             # Difference in Weighted Means
#             te_list[group_id] = np.average(
#                 y[group == group_id][t[group == group_id] == 1], weights=treated_group_weights
#             ) - np.average(
#                 y[group == group_id][t[group == group_id] == 0], weights=control_group_weights
#             )

#         return te_list


# class BinaryBootstrap(object):
#     def __init__(self, group_func: callable, n_groups: int):
#         # attributes
#         self.group_func = group_func 
#         self.n_groups = n_groups

#         # placeholder for bootstrap records
#         self.boot_records = None
    
#     def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray, boot_size: int = 10, n_bootstrap: int = 10000):
#         # assign each customer to a group
#         group_id_arr = np.array([self.group_func(x) for x in X])

#         # initialize bootstrap records
#         self.boot_records = np.zeros(shape=(n_bootstrap, self.n_groups))

#         # create bootstrap samples and estimate the treatment effect for each group
#         for group_id in range(self.n_groups):
#             # get the index of customers in group: group_id
#             group_idx_arr = np.where(group_id_arr == group_id)[0]
            
#             # create a list of bootstrap samples
#             group_boot_list = [np.random.choice(group_idx_arr, size=boot_size) for _ in range(n_bootstrap)]

#             self.boot_records[:, group_id] = np.array([
#                 OLS(Y[boot_idx], T[boot_idx]).fit().params[0] for boot_idx in group_boot_list
#             ])

#         return self

#     def estimate_targeting_value(self, X: np.ndarray, budget: int = 1) -> float: 
#         # extract bootstrapped treatment effect estimates for each group
#         boot_te_arr = np.array([
#             self.boot_records[:, self.group_func(x)].flatten() for x in X
#         ])  # (n_obs, n_bootstrap)

#         # sort the treatment effect estimates for each customer
#         sorted_boot_te_arr = -np.sort(-boot_te_arr, axis=0).mean(axis=1)

#         # take the top budget treatment effect estimates
#         top_budget_te_arr = sorted_boot_te_arr[:budget]

#         # calculate the expected maximum
#         expected_max = top_budget_te_arr.sum()

#         return expected_max


# class BinaryAdjustedCorrection(BinaryCorrection):
#     def __init__(self, group_func: callable, n_groups: int):
#         # attributes
#         self.group_func = group_func 
#         self.n_groups = n_groups

#         # place holders
#         self.n_bootstraps = None
#         self.boot_te_arr = None  # (n_bootstrap, n_groups)
#         self.plugin_estimator = BinaryPlugIn(group_func=group_func, n_groups=n_groups)
#         self.plugin_target_arr = None 

#         self.train_x, self.train_t, self.train_y = None, None, None

#         #
#         self.compat_count_list = []

#     def fit(self, X: np.ndarray = None, T: np.ndarray = None, Y: np.ndarray = None, n_bootstraps: int = 100):
#         # fill in placeholders
#         self.n_bootstraps = n_bootstraps

#         if self.train_x is None:
#             self.train_x, self.train_t, self.train_y = X, T, Y

#         # assign each customer to a group
#         group_arr = np.array([self.group_func(x) for x in self.train_x])  # (n_obs, )

#         # bootstrap sample indices, shape (num_bootstraps, m)
#         boot_index_arr = np.random.choice(np.arange(self.train_x.shape[0]), size=(self.n_bootstraps, self.train_x.shape[0]), replace=True)  
        
#         boot_y_arr = self.train_y.flatten()[boot_index_arr]
#         boot_t_arr = self.train_t.flatten()[boot_index_arr]
#         boot_group_arr = group_arr.flatten()[boot_index_arr]

#         self.boot_te_arr = np.array([calculate_binary_treatment_effect_per_group(
#             group=boot_group_arr[boot_id, :], t=boot_t_arr[boot_id, :], y=boot_y_arr[boot_id, :], n_groups=2
#         ) for boot_id in range(self.n_bootstraps)])

#         self.plugin_estimator.fit(X, T, Y)


#         return self
    
#     def estimate_targeting_value(self, X: np.ndarray, budget: int = 1) -> np.ndarray:
#         """   
#         Given a set of covariates, estimate the treatment effect for each covariate.

#         Params:
#         -------
#         X: np.ndarray, shape (n_obs, 1), the covariates
#         budget: int, the number of customers to target

#         Returns:
#         -------
#         np.ndarray, shape (n_bootstraps, n_obs)
#         """
#         # assign each customer to a group
#         group_arr = np.array([self.group_func(x) for x in X])  # (n_obs, )

#         # create an array to store the plugin targeting decision
#         plugin_target_arr = np.repeat(
#             self.plugin_estimator.get_targeting_decision(X, budget).reshape(1, -1), self.n_bootstraps, axis=0
#         )

#         # initialize place holders
#         flag = True
#         corr_arr_list = []
#         self.compat_count_list = []

#         while flag:
#             # create an array to store the treatment effect estimates for each bootstrap sample
#             boot_est_arr = self.boot_te_arr[:, group_arr]  # (n_bootstraps, n_obs)

#             # calculate empirical estimation error
#             xi_arr = self.calculate_empirical_error(boot_est_arr, resid_method=2)

#             # create a mask to identify the treated individuals within each bootstrap sample
#             # if the treatment effect estimate is among the largests, the mask is 1, otherwise 0
#             # if ties, select the ones with smaller indexes
#             mask_arr = np.zeros_like(xi_arr)
#             # sort the treatment effect estimates in descending order for each bootstrap sample
#             sorted_columns = np.argsort(-boot_est_arr, axis=1)[:, :budget]

#             # set the mask to 1 for the treated individuals
#             rows = np.repeat(np.arange(boot_est_arr.shape[0])[:, None], budget, axis=1)
#             mask_arr[rows, sorted_columns] = 1

#             # choose the rows where the plugin estimator and the corrected estimator agree
#             row_id_arr = np.where((mask_arr - plugin_target_arr != 0).sum(axis=1) == 0)[0]
#             self.compat_count_list.append(row_id_arr.shape[0])

#             # calculate the correction term
#             corr_arr = (mask_arr * xi_arr)[row_id_arr, :]  # (row_id_arr.shape[0], n_obs)
#             corr_arr_list.append(corr_arr)

#             # update flags
#             if np.sum([corr_arr.shape[0] for corr_arr in corr_arr_list]) >= self.n_bootstraps:
#                 flag = False

#             # refit bootstraps
#             self.fit(self.train_x, self.train_t, self.train_y, n_bootstraps=self.n_bootstraps)
            
#         # concatenate
#         corr_arr = np.concatenate(corr_arr_list, axis=0)
#         correction = corr_arr[:self.n_bootstraps].sum(axis=1).mean()

#         # calculate plugin estimate
#         plugin_estimate = self.plugin_estimator.estimate_targeting_value(X, budget=budget)

#         # calculate the corrected treatment effect estimate
#         return plugin_estimate - correction


# class BinaryDoubleCorrection(BinaryCorrection):
#     def __init__(self, group_func: callable, n_groups: int):
#         self.group_func = group_func 
#         self.n_groups = n_groups
#         self.plugin_estimator = BinaryPlugIn(group_func=group_func, n_groups=n_groups)

#         # place holders
#         self.n_fl_bootstrap = None  # number of first level bootstraps
#         self.fl_boot_est_arr = None  # (n_fl_bootstrap, n_groups)
#         self.fl_boot_se_arr = None  # (n_fl_bootstrap, n_groups)

#         self.n_sl_bootstrap = None  # number of second level bootstraps
#         self.sl_boot_est_arr = None  # (n_fl_bootstrap, n_sl_bootstrap, n_groups)
#         self.sl_boot_se_arr = None  # (n_fl_bootstrap, n_sl_bootstrap, n_groups)

#     def fit(
#         self, X: np.ndarray, T: np.ndarray, Y: np.ndarray, n_first_bootstrap: int = 100, 
#         n_second_bootstrap: int = 100
#     ):
#         # fill in placeholders
#         self.n_fl_bootstrap = n_first_bootstrap
#         self.n_sl_bootstrap = n_second_bootstrap

#         # first level bootstrap results: (n_fl_bootstrap, n_groups)
#         self.fl_boot_est_arr = np.zeros(shape=(self.n_fl_bootstrap, self.n_groups))
#         self.fl_boot_se_arr = np.zeros(shape=(self.n_fl_bootstrap, self.n_groups))

#         # second level bootstrap results: (n_fl_bootstrap, n_second_bootstrap, n_groups)
#         self.sl_boot_est_arr = np.zeros(shape=(self.n_fl_bootstrap, self.n_sl_bootstrap, self.n_groups)) 
#         self.sl_boot_se_arr = np.zeros(shape=(self.n_fl_bootstrap, self.n_sl_bootstrap, self.n_groups))

#         # assign each customer to a group
#         group_arr = np.array([self.group_func(x) for x in X])  # (n_obs, )

#         # fit OLS models for each group for each bootstrap sample, both first and second level bootstraps
#         for k in range(self.n_groups):
#             group_idx = np.where(group_arr == k)[0]
#             fl_boot_idx = np.random.choice(group_idx, size=group_idx.shape[0], replace=True)
#             for b in range(self.n_fl_bootstrap):
#                 ols = OLS(Y[fl_boot_idx], T[fl_boot_idx]).fit()
#                 self.fl_boot_est_arr[b, k], self.fl_boot_se_arr[b, k] = ols.params[0], ols.bse[0]
#                 for r in range(self.n_sl_bootstrap):
#                     sl_boot_idx = np.random.choice(fl_boot_idx, size=fl_boot_idx.shape[0], replace=True)
#                     ols = OLS(Y[sl_boot_idx], T[sl_boot_idx]).fit()
#                     self.sl_boot_est_arr[b, r, k], self.sl_boot_se_arr[b, r, k] = ols.params[0], ols.bse[0]
#         self.plugin_estimator.fit(X, T, Y)

#         return self

#     def estimate_targeting_value(
#         self, X: np.ndarray, budget: int = 1, resid_method: float = 2
#     ) -> np.ndarray:
#         """   
#         Given a set of covariates, estimate the treatment effect for each covariate.

#         Params:
#         -------
#         X: np.ndarray, shape (n_oobs, 1), the covariates
#         budget: int, the number of customers to target
#         resid_method: int, the method to calculate the empirical estimation error

#         Returns:
#         -------
#         np.ndarray, shape (n_bootstraps, n_oobs)
#         """
#         # 0. preparation
#         # assign each customer to a group
#         group_arr = np.array([self.group_func(x) for x in X])  # (n_obs, )

#         # 1. calculation for first level bootstraps
#         fl_correction = self.calculate_correction(
#             self.fl_boot_est_arr, budget=budget, resid_method=resid_method
#         )

#         # 2. calculation for second level bootstraps
#         # create an array to store the treatment effect estimates for each bootstrap sample
#         sl_boot_est_arr = self.sl_boot_est_arr[:, :, group_arr]  # (n_first_bootstrap, n_second_bootstrap, n_obs)
#         # sl_boot_se_arr = self.sl_boot_se_arr[:, :, group_arr]  # (n_first_bootstrap, n_second_bootstrap, n_obs)
#         # flatten the first two dimensions
#         sl_boot_est_arr = sl_boot_est_arr.reshape(-1, sl_boot_est_arr.shape[2])  # (n_first_bootstrap * n_second_bootstrap, n_obs)

#         # TODO 

#         # # calculate the corrected treatment effect estimate
#         # sl_boot_est = -np.sort(-sl_boot_est_arr, axis=2)[:, :, :budget].sum(axis=2).mean()

#         # # 3. calculate plugin estimate
#         # plugin_est = self.plugin_estimator.estimate_targeting_value(X, budget=budget)

#         # # calculate the biases
#         # correction = fl_boot_est - plugin_est 
#         # correction_bias = sl_boot_est - fl_boot_est
#         # final_correction = correction - correction_bias  # 2 * fl_boot_est - sl_boot_est + plugin_est

#         # # calculate the corrected treatment effect estimate
#         # return plugin_est - final_correction
    
#     def calculate_correction(
#         self, boot_est: np.ndarray, budget: int, resid_method=2, **kwargs
#     ) -> np.ndarray:
#         # calculate the empirical estimation error
#         xi_arr = self.calculate_empirical_error(boot_est, resid_method=2)

#         # create a mask to identify the treated individuals within each bootstrap sample
#         # if the treatment effect estimate is among the largests, the mask is 1, otherwise 0
#         # if ties, select the ones with smaller indexes
#         mask_arr = np.zeros_like(xi_arr)
#         # sort the treatment effect estimates in descending order for each bootstrap sample
#         sorted_columns = np.argsort(-boot_est, axis=1)[:, :budget]

#         # set the mask to 1 for the treated individuals
#         rows = np.repeat(np.arange(boot_est.shape[0])[:, None], budget, axis=1)
#         mask_arr[rows, sorted_columns] = 1

#         del sorted_columns, rows

#         # calculate the correction term
#         correction = np.mean(np.sum(mask_arr * xi_arr, axis=1))

#         return correction

# class BinaryParametric(object):
#     def __init__(self, group_func: callable, n_groups: int, budget: int = 1):
#         # attributes
#         self.group_func = group_func 
#         self.n_groups = n_groups
#         self.budget = budget 

#         # initialize
#         self.group_ols_list = [None for _ in range(self.n_groups)]

#     def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
#         group_arr = np.array([self.group_func(x) for x in X]).reshape(-1, 1)

#         for group_id in range(self.n_groups):
#             group_idx = np.where(group_arr == group_id)[0]
#             self.group_ols_list[group_id] = OLS(Y[group_idx], T[group_idx]).fit()

#         return self
    
#     def estimate_targeting_value(self, X: np.ndarray, budget: int = 1, sample_size: int = 2000):
#         # # create treatment effect variable for each customer
#         # group_arr = np.array([self.group_func(x) for x in X])
#         # var_list = [
#         #     UnivariateGaussian(mean=self.group_ols_list[group_id].params[0], std=self.group_ols_list[group_id].bse[0])
#         #     for group_id in group_arr
#         # ]

#         # # create bootstraps
#         # boot_records = [
#         #     var.sample(sample_size) for var in var_list 
#         # ]  # a list with length n_obs, each element is a np.ndarray of shape (n_bootstrap, )
#         # boot_records = np.array(boot_records)  # (n_obs, n_bootstrap)

#         # # sort each bootstraps
#         # sorted_boot_records = -np.sort(-boot_records, axis=0)  # (n_obs, n_bootstrap)

#         # # take the top budget treatment effect estimates
#         # top_budget_boot_records = sorted_boot_records[:budget, :]  # (budget, n_bootstrap)

#         # # calculate the expected maximum
#         # expected_max = np.mean(np.sum(top_budget_boot_records, axis=0))

#         # initialize treatment effect sample array
#         group_te_arr = np.zeros(shape=(self.n_groups, sample_size))  # (n_groups, sample_size)

#         # estimate order statistics
#         for group_id in range(self.n_groups):
#             # get the OLS model for the group this customer belongs to
#             ols = self.group_ols_list[group_id]  

#             # marginal posterior of sigma^2
#             group_te_arr[group_id, :] = UnivariateGaussian(
#                 mean=ols.params[0], std=ols.bse[0]
#             ).sample(size=sample_size)

#         # assign treatment effect samples to each individual
#         ind_te_arr = np.array([group_te_arr[self.group_func(x), :] for x in X])

#         # sort the treatment effect samples
#         sorted_ind_te_arr = -np.sort(-ind_te_arr, axis=0).mean(axis=1)

#         # take the top budget treatment effect estimates
#         top_budget_te_arr = sorted_ind_te_arr[:budget]

#         # calculate the expected maximum
#         expected_max = top_budget_te_arr.sum()

#         return expected_max
    

# class BinaryBayesParametric(BaseEstimator):
#     def __init__(self, group_func: callable, n_groups: int, n_bootstrap: int = 100, budget: int = 1):
#         # attributes
#         self.group_func = group_func 
#         self.n_groups = n_groups
#         self.n_bootstrap = n_bootstrap
#         self.budget = budget 

#         self.precision = None 

#         # initialize
#         self.group_ols_list = [None for _ in range(self.n_groups)]

#     def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
#         group_arr = np.array([self.group_func(x) for x in X]).reshape(-1, 1)

#         for group_id in range(self.n_groups):
#             group_idx = np.where(group_arr == group_id)[0]
#             self.group_ols_list[group_id] = OLS(Y[group_idx], T[group_idx]).fit()

#         # calculate the precision
#         self.precision = 1 / (X.T @ X)[0, 0] 
#         return self
    
#     def estimate_targeting_value(self, X: np.ndarray, budget: int = 1, sample_size: int = 10000):
#         """ 
#         Estimate the targeting value for an unseen group of customers X given budget

#         Params:
#         -------
#         X: np.ndarray, shape (n_obs, d), the covariates
#         budget: int, the number of customers to target
#         sample_size: int, the number of samples to draw from the posterior distribution
#         """

#         # initialize treatment effect sample array
#         group_te_arr = np.zeros(shape=(self.n_groups, sample_size))  # (n_groups, sample_size)

#         # estimate order statistics
#         for group_id in range(self.n_groups):
#             # get the OLS model for the group this customer belongs to
#             ols = self.group_ols_list[group_id]  

#             # marginal posterior of sigma^2
#             shape = 0.5 * ols.df_resid
#             s2 = (ols.resid @ ols.resid) / ols.df_resid
#             scale = 0.5 * ols.df_resid * s2  # df_resid cancels out

#             # sample from the marginal posterior of sigma^2
#             sigma_sample = invgamma(a=shape, scale=scale).rvs(size=sample_size)

#             # sample from the marginal posterior of beta (treatment effect)
#             for i in range(sample_size):
#                 group_te_arr[group_id, i] = UnivariateGaussian(
#                     mean=ols.params[0], 
#                     std=self.precision * sigma_sample[i]
#                 ).sample(size=1)[0]

#         # assign treatment effect samples to each individual
#         ind_te_arr = np.array([group_te_arr[self.group_func(x), :] for x in X])

#         # sort the treatment effect samples
#         sorted_ind_te_arr = -np.sort(-ind_te_arr, axis=0).mean(axis=1)

#         # take the top budget treatment effect estimates
#         top_budget_te_arr = sorted_ind_te_arr[:budget]

#         # calculate the expected maximum
#         expected_max = top_budget_te_arr.sum()

#         return expected_max
