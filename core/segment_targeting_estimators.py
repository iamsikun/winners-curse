import os 
import sys 
sys.path.insert(0, os.path.abspath('.'))

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

    Returns:
    -------
    segment_te_arr: np.ndarray, shape = (n_segments, )
        The estimated treatment effects for each segment
    """
    # assign each customer to a segment
    customer_segment_arr = np.array([segment_func(x) for x in covariates])  # shape = (n_obs, )

    # fit OLS estimator for each segment
    segment_te_arr = np.zeros((n_segments, ))

    for i in range(n_segments):
        segment_te_arr[i] = calculate_treatment_effect_with_dim(
            outcomes[customer_segment_arr == i], 
            treatments[customer_segment_arr == i]
        )

    return segment_te_arr


def segment_targeting_obj_func(
    test_customers: np.ndarray, targ_decision: np.ndarray, params: np.ndarray, 
    segment_func: callable, 
) -> float: 
    """
    The objective function for the segment targeting problem

    Params:
    ------
    test_customers: np.ndarray, shape = (n_obs, n_features)
        The observed covariates of the test customers
    targ_decision: np.ndarray[Binary], shape = (n_obs, )
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
    customer_segment_arr = np.array([segment_func(x) for x in test_customers])  # shape = (n_obs, )

    # assign the treatment effects to the test customers
    customer_te_arr = params[customer_segment_arr]  # shape = (n_obs, )

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
    test_customers: np.ndarray, shape = (n_obs, n_features)
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
    opt_targ_decision = np.zeros((test_customers.shape[0], ))  # shape = (n_obs, )

    # assign treatment effects to each customer based on their segments
    customer_segment_arr = np.array([segment_func(x) for x in test_customers])  # shape = (n_obs, )

    # assign the treatment effects to the test customers
    customer_te_arr = params[customer_segment_arr]  # shape = (n_obs, )

    # sort the customers by their estimated treatment effects
    sorted_customers_indices = np.argsort(-customer_te_arr)  # sort in descending order, shape = (n_obs, )

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
    
    # initialize placeholders

    # assign treatment effects to each customer based on their segments
    customer_segment_arr = np.array([segment_func(x) for x in test_customers])  # shape = (sample_size, )

    # assign the treatment effects to the test customers
    boot_customer_te_arr = params[:, customer_segment_arr]  # shape = (n_bootstrap, sample_size)

    # optimize targeting for each bootstrap sample
    boot_opt_targ_decisions = np.argsort(-boot_customer_te_arr, axis=1)[:, :budget]  # (n_bootstrap, budget)

    # calculate the value of the targeting policy for each bootstrap sample using the bootstrap estimates
    boot_targ_ests = boot_customer_te_arr[np.arange(n_bootstraps)[:, None], boot_opt_targ_decisions].sum(axis=1)  # (n_bootstrap, )

    return boot_opt_targ_decisions, boot_targ_ests
    

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


class BinaryValueCorrection(object):
    """ 
    Correction estimator for targeting value on the value function level

    The empirical distribution of the value function is constructed via vannila nonparametric bootstrap.
    """
    def __init__(self, group_func: callable, n_groups: int):
        self.group_func = group_func 
        self.n_groups = n_groups

        # place holders
        self.n_bootstraps = None  # number of bootstrap samples
        self.boot_te_arr = None  # (n_bootstrap, n_groups)
        self.plugin_estimator = BinaryPlugIn(group_func=group_func, n_groups=n_groups)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray, n_bootstraps: int = 100):
        # fill in placeholders
        self.n_bootstraps = n_bootstraps

        # assign each customer to a group
        group_arr = np.array([self.group_func(x) for x in X])  # (n_obs, )

        # bootstrap sample indices, shape (n_bootstraps, n_obs)
        boot_index_arr = np.random.choice(np.arange(X.shape[0]), size=(self.n_bootstraps, X.shape[0]), replace=True)  
        
        # create bootstrap samples
        boot_y_arr = Y.flatten()[boot_index_arr]
        boot_t_arr = T.flatten()[boot_index_arr]
        boot_group_arr = group_arr.flatten()[boot_index_arr]

        # calculate the treatment effects for each group for each bootstrap sample
        self.boot_te_arr = np.array([te_with_dim(
            group=boot_group_arr[boot_id, :], t=boot_t_arr[boot_id, :], y=boot_y_arr[boot_id, :], n_groups=2
        ) for boot_id in range(self.n_bootstraps)])  # (n_bootstraps, n_groups)

        self.plugin_estimator.fit(X, T, Y)

        return self
    
    def estimate_targeting_value(
        self, X: np.ndarray, budget: int = 1
    ) -> np.ndarray:
        """   
        Given a set of covariates, estimate the treatment effect for each covariate.

        Params:
        -------
        X: np.ndarray, shape (M, 1), the covariates
        budget: int, the number of customers to target
        resid_method: int, the method to calculate the empirical estimation error

        Returns:
        -------
        np.ndarray, shape (n_bootstraps, n_oobs)
        """
        # assign each customer to a group
        test_group_arr = np.array([self.group_func(x) for x in X])  # (n_obs, )

        # create an array to store the treatment effect estimates for each bootstrap sample
        boot_est_arr = self.boot_te_arr[:, test_group_arr]  # (n_bootstraps, n_obs)

        # optimize targeting for each bootstrap sample
        # which customers to target for each bootstrap sample
        boot_targ_decision_arr = np.argsort(-boot_est_arr, axis=1)[:, :budget]  # (n_bootstraps, budget)

        # calculate the value of the targeting policy for each bootstrap sample using the bootstrap estimates
        boot_targ_val_arr = boot_est_arr[np.arange(boot_est_arr.shape[0])[:, None], boot_targ_decision_arr].sum(axis=1)  # (n_bootstraps, )

        # calculate plugin estimate
        plugin_est = self.plugin_estimator.estimate_targeting_value(X, budget=budget)

        # evaluate bootstrap policy using plugin estimates
        boot_targ_emp_val_arr = self.plugin_estimator.te_est_arr[test_group_arr][boot_targ_decision_arr].flatten()  # (n_bootstraps, )

        # calculate the corrected treatment effect estimate
        return plugin_est - boot_targ_val_arr.mean() + boot_targ_emp_val_arr.mean()
        

class BinaryMNBValueCorrection(BinaryValueCorrection):
    """ 
    Correction estimator for targeting value on the value function level

    The empirical distribution of the value function is constructed via 
    m-out-of-n bootstrap. 

    Rule for selecting the best m follows (Bickle and Sakov 2008, Statistica Sinica)
    """
    def __init__(self, group_func: callable, n_groups: int):
        self.group_func = group_func 
        self.n_groups = n_groups

        # place holders
        self.n_bootstraps = None  # number of bootstrap samples
        # self.boot_te_arr = None  # (n_bootstrap, n_groups)
        self.plugin_estimator = BinaryPlugIn(group_func=group_func, n_groups=n_groups)
        self.m_list = []
        self.m_boot_te_list = []

    def fit(
        self, X: np.ndarray, T: np.ndarray, Y: np.ndarray, n_bootstraps: int = 100, q: float = 0.9, max_j: int = 20
    ):
        # fill in placeholders
        self.n_bootstraps = n_bootstraps

        # calculate the sample sizes 
        sample_size = X.shape[0] 
        self.m_list = np.array([int(q**j * sample_size) for j in range(max_j)])

        # assign each customer to a group
        group_arr = np.array([self.group_func(x) for x in X])  # (n_obs, )

        # create bootstraps
        for m in self.m_list:
            # bootstrap sample indices, shape (n_bootstraps, m)
            boot_index_arr = np.random.choice(np.arange(sample_size), size=(self.n_bootstraps, m), replace=True)  
            
            # create bootstrap samples
            boot_y_arr = Y.flatten()[boot_index_arr]
            boot_t_arr = T.flatten()[boot_index_arr]
            boot_group_arr = group_arr.flatten()[boot_index_arr]

            # calculate the treatment effect for each group for each bootstrap sample
            boot_te_arr = np.array([te_with_dim(
                group=boot_group_arr[boot_id, :], t=boot_t_arr[boot_id, :], y=boot_y_arr[boot_id, :], n_groups=2
            ) for boot_id in range(self.n_bootstraps)])

            # keep only the rows with no NaN values
            boot_te_arr = boot_te_arr[~np.isnan(boot_te_arr).any(axis=1)]

            # update attributes
            self.m_boot_te_list.append(boot_te_arr)

        self.plugin_estimator.fit(X, T, Y)

        return self
    
    def estimate_targeting_value(
        self, X: np.ndarray, budget: int = 1
    ) -> np.ndarray:
        """   
        Given a set of covariates, estimate the treatment effect for each covariate.

        Params:
        -------
        X: np.ndarray, shape (M, 1), the covariates
        budget: int, the number of customers to target
        resid_method: int, the method to calculate the empirical estimation error

        Returns:
        -------
        np.ndarray, shape (n_bootstraps, n_oobs)
        """
        test_group_arr = np.array([self.group_func(x) for x in X])  # (n_obs, )
        boot_dstn_list = [None] * len(self.m_list)  # store the bootstrap distributions for each m

        for i in range(len(self.m_list)):
            boot_te_arr = self.m_boot_te_list[i]  # (n_bootstraps, n_groups)

            # create an array to store the treatment effect estimates for each bootstrap sample
            boot_est_arr = boot_te_arr[:, test_group_arr]  # (n_bootstraps, n_obs)

            # optimize targeting for each bootstrap sample
            # which customers to target for each bootstrap sample
            boot_targ_decision_arr = np.argsort(-boot_est_arr, axis=1)[:, :budget]  # (n_bootstraps, budget)

            # calculate the value of the targeting policy for each bootstrap sample using the bootstrap estimates
            boot_targ_val_arr = boot_est_arr[np.arange(boot_est_arr.shape[0])[:, None], boot_targ_decision_arr].sum(axis=1)  # (n_bootstraps, )

            # evaluate bootstrap policy using plugin estimates
            boot_targ_emp_val_arr = self.plugin_estimator.te_est_arr[test_group_arr][boot_targ_decision_arr].flatten()  # (n_bootstraps, )

            # store the bootstrap distribution of bias estimates
            boot_dstn_list[i] = boot_targ_val_arr - boot_targ_emp_val_arr

        # choose the best m
        discp_list = [None] * (len(self.m_list) - 1)
        for idx, (prev_dstn, current_dstn) in enumerate(zip(boot_dstn_list[:-1], boot_dstn_list[1:])):
            ks_stat, _ = ks_2samp(prev_dstn, current_dstn)
            discp_list[idx] = ks_stat
        min_discp_idx = np.argmin(discp_list)
        correction = boot_dstn_list[min_discp_idx].mean()

        # calculate plugin estimate
        plugin_est = self.plugin_estimator.estimate_targeting_value(X, budget=budget)

        # calculate the corrected treatment effect estimate
        return plugin_est - correction


class BinaryNBValueCorrection(object):
    """ 
    Correction estimator for targeting value on the value function level

    The empirical distribution of the value function is constructed via numerical bootstrap. 
    In this implementation, instead of perturbing the data itself, we perturb the CATE estiamtes. 
    """
    def __init__(self, group_func: callable, n_groups: int):
        self.group_func = group_func 
        self.n_groups = n_groups

        # place holders
        self.n_bootstraps = None  # number of bootstrap samples
        self.perturbed_te_arr = None  # (n_bootstrap, n_groups)
        self.plugin_estimator = BinaryPlugIn(group_func=group_func, n_groups=n_groups)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray, n_bootstraps: int = 100, epsilon_n_pow: float = -0.45):
        # fill in placeholders
        self.n_bootstraps = n_bootstraps

        # assign each customer to a group
        group_arr = np.array([self.group_func(x) for x in X])  # (n_obs, )

        # bootstrap sample indices, shape (n_bootstraps, n_obs)
        boot_index_arr = np.random.choice(np.arange(X.shape[0]), size=(self.n_bootstraps, X.shape[0]), replace=True)  
        
        # create bootstrap samples
        boot_y_arr = Y.flatten()[boot_index_arr]
        boot_t_arr = T.flatten()[boot_index_arr]
        boot_group_arr = group_arr.flatten()[boot_index_arr]

        # calculate the treatment effects for each group for each bootstrap sample
        boot_te_arr = np.array([te_with_dim(
            group=boot_group_arr[boot_id, :], t=boot_t_arr[boot_id, :], y=boot_y_arr[boot_id, :], n_groups=2
        ) for boot_id in range(self.n_bootstraps)])  # (n_bootstraps, n_groups)

        self.plugin_estimator.fit(X, T, Y)

        # get perturbed treatment effect estimates
        perturbation = (X.shape[0] ** (epsilon_n_pow)) * np.sqrt(X.shape[0]) * (self.plugin_estimator.te_est_arr.reshape(1, -1) - boot_te_arr)  # (n_bootstraps, n_groups)
        self.perturbed_te_arr = self.plugin_estimator.te_est_arr.reshape(1, -1) + perturbation  # (n_bootstraps, n_groups)

        return self
    
    def estimate_targeting_value(
        self, X: np.ndarray, budget: int = 1, resid_method: float = 2
    ) -> np.ndarray:
        """   
        Given a set of covariates, estimate the treatment effect for each covariate.

        Params:
        -------
        X: np.ndarray, shape (M, 1), the covariates
        budget: int, the number of customers to target
        resid_method: int, the method to calculate the empirical estimation error

        Returns:
        -------
        np.ndarray, shape (n_bootstraps, n_oobs)
        """
        # assign each customer to a group
        test_group_arr = np.array([self.group_func(x) for x in X])  # (n_obs, )

        # create an array to store the treatment effect estimates for each bootstrap sample
        boot_est_arr = self.perturbed_te_arr[:, test_group_arr]  # (n_bootstraps, n_obs)

        # optimize targeting for each bootstrap sample
        # which customers to target for each bootstrap sample
        boot_targ_decision_arr = np.argsort(-boot_est_arr, axis=1)[:, :budget]  # (n_bootstraps, budget)

        # calculate the value of the targeting policy for each bootstrap sample using the bootstrap estimates
        boot_targ_val_arr = boot_est_arr[np.arange(boot_est_arr.shape[0])[:, None], boot_targ_decision_arr].sum(axis=1)  # (n_bootstraps, )

        # calculate plugin estimate
        plugin_est = self.plugin_estimator.estimate_targeting_value(X, budget=budget)

        # evaluate bootstrap policy using plugin estimates
        boot_targ_emp_val_arr = self.plugin_estimator.te_est_arr[test_group_arr][boot_targ_decision_arr].flatten()  # (n_bootstraps, )

        # calculate the corrected treatment effect estimate
        return plugin_est - boot_targ_val_arr.mean() + boot_targ_emp_val_arr.mean()


class BinaryErrorCorrection(object):
    """
    Correction estimator for targeting value on the prediction error level. 
    This estimator requires the analytical formulation for the winner's curse (value function bias)
    as a function of the estimation errors. 

    The empirical distribution of estimation errors is constructed via vannila nonparametric bootstrap, 
    where the bootstrap sample size is equal to the sample size of the original data. 
    """
    def __init__(self, group_func: callable, n_groups: int):
        self.group_func = group_func 
        self.n_groups = n_groups

        # place holders
        self.n_bootstraps = None  # number of bootstrap samples
        self.boot_te_arr = None  # (n_bootstrap, n_groups)
        self.plugin_estimator = BinaryPlugIn(group_func=group_func, n_groups=n_groups)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray, n_bootstraps: int = 100):
        # fill in placeholders
        self.n_bootstraps = n_bootstraps

        # assign each customer to a group
        group_arr = np.array([self.group_func(x) for x in X])  # (sample_size, )

        # bootstrap sample indices, shape (n_bootstraps, sample_size)
        boot_index_arr = np.random.choice(np.arange(X.shape[0]), size=(self.n_bootstraps, X.shape[0]), replace=True)  
        
        boot_y_arr = Y.flatten()[boot_index_arr]  # (n_bootstraps, sample_size)
        boot_t_arr = T.flatten()[boot_index_arr]  # (n_bootstraps, sample_size)
        boot_group_arr = group_arr.flatten()[boot_index_arr]  # (n_bootstraps, sample_size)

        self.boot_te_arr = np.array([te_with_dim(
            group=boot_group_arr[boot_id, :], t=boot_t_arr[boot_id, :], y=boot_y_arr[boot_id, :], n_groups=self.n_groups
        ) for boot_id in range(self.n_bootstraps)])  # (n_bootstraps, n_groups)

        self.plugin_estimator.fit(X, T, Y)

        return self
    
    def estimate_targeting_value(
        self, X: np.ndarray, budget: int = 1, resid_method: float = 3
    ) -> np.ndarray:
        """   
        Given a set of covariates, estimate the treatment effect for each covariate.

        Params:
        -------
        X: np.ndarray, shape (n_obs, 1), the covariates
        budget: int, the number of customers to target
        resid_method: int, the method to calculate the empirical estimation error

        Returns:
        -------
        np.ndarray, shape (n_bootstraps, n_obs)
        """
        # assign each customer to a group
        group_arr = np.array([self.group_func(x) for x in X])  # (n_obs, )

        # create an array to store the treatment effect estimates for each bootstrap sample
        boot_est_arr = self.boot_te_arr[:, group_arr]  # (n_bootstraps, n_obs)

        # calculate empirical estimation error
        xi_arr = self.calculate_empirical_error(boot_est_arr, resid_method=resid_method, X=X)

        # create a mask to identify the treated individuals within each bootstrap sample
        # if the treatment effect estimate is among the largests, the mask is 1, otherwise 0
        # if ties, select the ones with smaller indexes
        mask_arr = np.zeros_like(xi_arr)
        # sort the treatment effect estimates in descending order for each bootstrap sample
        sorted_columns = np.argsort(-boot_est_arr, axis=1)[:, :budget]

        # set the mask to 1 for the treated individuals
        rows = np.repeat(np.arange(boot_est_arr.shape[0])[:, None], budget, axis=1)
        mask_arr[rows, sorted_columns] = 1

        del sorted_columns, rows

        # calculate the correction term
        correction = np.mean(np.sum(mask_arr * xi_arr, axis=1))

        # calculate plugin estimate
        plugin_estimate = self.plugin_estimator.estimate_targeting_value(X, budget=budget)

        # calculate the corrected treatment effect estimate
        return plugin_estimate - correction
    
    def calculate_empirical_error(self, boot_est: np.ndarray, resid_method=2, **kwargs) -> np.ndarray:
        if resid_method == 1:
            # calculate empirical estimation error: method 1
            return boot_est - boot_est.mean(axis=0)  # (n_bootstraps, n_obs)
        elif resid_method == 2:
            # calculate empirical estimation error: method 2
            # Calculate the sum of all elements along axis 0 (column-wise sum)
            column_sums = np.sum(boot_est, axis=0)
            # Create an adjusted sum by subtracting each row from the column sums
            adjusted_sums = column_sums - boot_est
            # Compute the leave-one-out mean for each row
            leave_one_out_means = adjusted_sums / (boot_est.shape[0] - 1)
            # Calculate the result
            xi_arr = boot_est - leave_one_out_means
            del column_sums, adjusted_sums, leave_one_out_means
            return xi_arr
        else:
            # calculate empirical estimation error: method 3
            return boot_est - self.plugin_estimator.estimate_treatment_effect(kwargs['X'])
    

class BinaryMNBErrorCorrection(BinaryErrorCorrection):
    """ 
    Correction estimator for targeting value on the error level

    The empirical distribution of the value function is constructed via 
    m-out-of-n bootstrap. 
    """
    def __init__(self, group_func: callable, n_groups: int):
        self.group_func = group_func 
        self.n_groups = n_groups

        # place holders
        self.n_bootstraps = None  # number of bootstrap samples
        self.boot_te_arr = None  # (n_bootstrap, n_groups)
        self.plugin_estimator = BinaryPlugIn(group_func=group_func, n_groups=n_groups)
        self.m_list = []
        self.m_boot_te_list = []

    def fit(
        self, X: np.ndarray, T: np.ndarray, Y: np.ndarray, n_bootstraps: int = 100, q: float = 0.9, max_j: int = 20
    ):
        # fill in placeholders
        self.n_bootstraps = n_bootstraps

        # calculate the sample sizes 
        sample_size = X.shape[0] 
        self.m_list = np.array([int(q**j * sample_size) for j in range(max_j)])

        # assign each customer to a group
        group_arr = np.array([self.group_func(x) for x in X])  # (n_obs, )

        # create bootstraps
        for i, m in enumerate(self.m_list):
            # bootstrap sample indices, shape (num_bootstraps, m)
            boot_index_arr = np.random.choice(np.arange(sample_size), size=(self.n_bootstraps, m), replace=True)  
            
            boot_y_arr = Y.flatten()[boot_index_arr]
            boot_t_arr = T.flatten()[boot_index_arr]
            boot_group_arr = group_arr.flatten()[boot_index_arr]

            # calculate the treatment effect per group for each bootstrap sample
            boot_te_arr = np.array([te_with_dim(
                group=boot_group_arr[boot_id, :], t=boot_t_arr[boot_id, :], y=boot_y_arr[boot_id, :], n_groups=2
            ) for boot_id in range(self.n_bootstraps)])

            # keep only the rows with no NaN values
            boot_te_arr = boot_te_arr[~np.isnan(boot_te_arr).any(axis=1)]

            # update attributes
            self.m_boot_te_list.append(boot_te_arr)

        self.plugin_estimator.fit(X, T, Y)

        return self
    
    def estimate_targeting_value(
        self, X: np.ndarray, budget: int = 1, resid_method: float = 2
    ) -> np.ndarray:
        """   
        Given a set of covariates, estimate the treatment effect for each covariate.

        Params:
        -------
        X: np.ndarray, shape (M, 1), the covariates
        budget: int, the number of customers to target
        resid_method: int, the method to calculate the empirical estimation error

        Returns:
        -------
        np.ndarray, shape (n_bootstraps, n_oobs)
        """
        test_group_arr = np.array([self.group_func(x) for x in X])
        corr_dstn_list = [None] * len(self.m_list)

        for i in range(len(self.m_list)):
            boot_te_arr = self.m_boot_te_list[i]

            # create an array to store the treatment effect estimates for each bootstrap sample
            boot_est_arr = boot_te_arr[:, test_group_arr]  # (n_bootstraps, n_obs)

            # calculate empirical estimation error
            xi_arr = self.calculate_empirical_error(boot_est_arr, resid_method=resid_method, X=X)

            # create a mask to identify the treated individuals within each bootstrap sample
            # if the treatment effect estimate is among the largests, the mask is 1, otherwise 0
            # if ties, select the ones with smaller indexes
            mask_arr = np.zeros_like(xi_arr)
            # sort the treatment effect estimates in descending order for each bootstrap sample
            sorted_columns = np.argsort(-boot_est_arr, axis=1)[:, :budget]

            # set the mask to 1 for the treated individuals
            rows = np.repeat(np.arange(boot_est_arr.shape[0])[:, None], budget, axis=1)
            mask_arr[rows, sorted_columns] = 1

            del sorted_columns, rows

            # optimize targeting for each bootstrap sample
            corr_dstn_list[i] = np.sum(mask_arr * xi_arr, axis=1)  # (n_bootstraps, )

        # choose the best m
        discp_list = [None] * (len(self.m_list) - 1)
        for idx, (prev_dstn, current_dstn) in enumerate(zip(corr_dstn_list[:-1], corr_dstn_list[1:])):
            ks_stat, _ = ks_2samp(prev_dstn, current_dstn)
            discp_list[idx] = ks_stat

        min_discp_idx = np.argmin(discp_list)
        boot_corr_avg = corr_dstn_list[min_discp_idx].mean()

        # calculate plugin estimate
        plugin_estimate = self.plugin_estimator.estimate_targeting_value(X, budget=budget)

        # calculate the corrected treatment effect estimate
        return plugin_estimate - boot_corr_avg


class BinaryNBErrorCorrection(object):
    """
    Correction estimator for targeting value on the prediction error level. 
    This estimator requires the analytical formulation for the winner's curse (value function bias)
    as a function of the estimation errors. 

    The empirical distribution of estimation errors is constructed via numerical bootstrap, 
    """
    def __init__(self, group_func: callable, n_groups: int):
        self.group_func = group_func 
        self.n_groups = n_groups

        # place holders
        self.n_bootstraps = None  # number of bootstrap samples
        self.perturbed_te_arr = None  # (n_bootstrap, n_groups)
        self.plugin_estimator = BinaryPlugIn(group_func=group_func, n_groups=n_groups)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray, n_bootstraps: int = 100, epsilon_n_pow: float = -0.45):
        # fill in placeholders
        self.n_bootstraps = n_bootstraps

        # assign each customer to a group
        group_arr = np.array([self.group_func(x) for x in X])  # (n_obs, )

        # bootstrap sample indices, shape (n_bootstraps, n_obs)
        boot_index_arr = np.random.choice(np.arange(X.shape[0]), size=(self.n_bootstraps, X.shape[0]), replace=True)  
        
        # create bootstrap samples
        boot_y_arr = Y.flatten()[boot_index_arr]
        boot_t_arr = T.flatten()[boot_index_arr]
        boot_group_arr = group_arr.flatten()[boot_index_arr]

        # calculate the treatment effects for each group for each bootstrap sample
        boot_te_arr = np.array([te_with_dim(
            group=boot_group_arr[boot_id, :], t=boot_t_arr[boot_id, :], y=boot_y_arr[boot_id, :], n_groups=2
        ) for boot_id in range(self.n_bootstraps)])  # (n_bootstraps, n_groups)

        self.plugin_estimator.fit(X, T, Y)

        # get perturbed treatment effect estimates
        perturbation = (X.shape[0] ** (epsilon_n_pow)) * np.sqrt(X.shape[0]) * (self.plugin_estimator.te_est_arr.reshape(1, -1) - boot_te_arr)  # (n_bootstraps, n_groups)
        self.perturbed_te_arr = self.plugin_estimator.te_est_arr.reshape(1, -1) + perturbation  # (n_bootstraps, n_groups)

        return self
    
    def estimate_targeting_value(
        self, X: np.ndarray, budget: int = 1, resid_method: float = 3
    ) -> np.ndarray:
        """   
        Given a set of covariates, estimate the treatment effect for each covariate.

        Params:
        -------
        X: np.ndarray, shape (n_obs, 1), the covariates
        budget: int, the number of customers to target
        resid_method: int, the method to calculate the empirical estimation error

        Returns:
        -------
        np.ndarray, shape (n_bootstraps, n_obs)
        """
        # assign each customer to a group
        group_arr = np.array([self.group_func(x) for x in X])  # (n_obs, )

        # create an array to store the treatment effect estimates for each bootstrap sample
        boot_est_arr = self.perturbed_te_arr[:, group_arr]  # (n_bootstraps, n_obs)

        # calculate empirical estimation error
        xi_arr = self.calculate_empirical_error(boot_est_arr, resid_method=resid_method, X=X)

        # create a mask to identify the treated individuals within each bootstrap sample
        # if the treatment effect estimate is among the largests, the mask is 1, otherwise 0
        # if ties, select the ones with smaller indexes
        mask_arr = np.zeros_like(xi_arr)
        # sort the treatment effect estimates in descending order for each bootstrap sample
        sorted_columns = np.argsort(-boot_est_arr, axis=1)[:, :budget]

        # set the mask to 1 for the treated individuals
        rows = np.repeat(np.arange(boot_est_arr.shape[0])[:, None], budget, axis=1)
        mask_arr[rows, sorted_columns] = 1

        del sorted_columns, rows

        # calculate the correction term
        correction = np.mean(np.sum(mask_arr * xi_arr, axis=1))

        # calculate plugin estimate
        plugin_estimate = self.plugin_estimator.estimate_targeting_value(X, budget=budget)

        # calculate the corrected treatment effect estimate
        return plugin_estimate - correction
    
    def calculate_empirical_error(self, boot_est: np.ndarray, resid_method=2, **kwargs) -> np.ndarray:
        if resid_method == 1:
            # calculate empirical estimation error: method 1
            return boot_est - boot_est.mean(axis=0)  # (n_bootstraps, n_obs)
        elif resid_method == 2:
            # calculate empirical estimation error: method 2
            # Calculate the sum of all elements along axis 0 (column-wise sum)
            column_sums = np.sum(boot_est, axis=0)
            # Create an adjusted sum by subtracting each row from the column sums
            adjusted_sums = column_sums - boot_est
            # Compute the leave-one-out mean for each row
            leave_one_out_means = adjusted_sums / (boot_est.shape[0] - 1)
            # Calculate the result
            xi_arr = boot_est - leave_one_out_means
            del column_sums, adjusted_sums, leave_one_out_means
            return xi_arr
        else:
            # calculate empirical estimation error: method 3
            return boot_est - self.plugin_estimator.estimate_treatment_effect(kwargs['X'])


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
