import os 
import sys 
sys.path.insert(0, os.path.abspath('.'))

import numpy as np 
from statsmodels.regression.linear_model import OLS
from scipy.stats import norm, invgamma

from core.variables import UnivariateGaussian

class BaseEstimator(object):
    def check_input(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> None:
        assert X.shape[0] == T.shape[0] == Y.shape[0], "Input shapes do not match"
        assert T.shape[1] == 1, "T should be a column vector"
        assert Y.shape[1] == 1, "Y should be a column vector"

class BinaryPlugIn(BaseEstimator):
    def __init__(self, group_func: callable, n_groups: int):
        """  
        A plug-in estimator that estimates the effect of binary treatments for each group.

        Params:
        -------
        group_func: callable, a function that assigns each customer to a group
        n_groups: int, the number of groups

        """
        # attributes
        self.group_func = group_func

        # initialize the list to store the OLS estimators for each group
        self.group_ols_list = [None for _ in range(n_groups)]

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """  
        Fit the OLS estimator for each group.

        Params:
        -------
        X: np.ndarray, shape (n_obs, d), the covariates
        T: np.ndarray, shape (n_obs, 1), the treatment assignment
        Y: np.ndarray, shape (n_obs, 1), the outcome
        """
        # check input shapes
        self.check_input(X, T, Y)

        # get the group assignment for each customer
        group_arr = np.array([self.group_func(x) for x in X])

        # fit the OLS estimator for each group
        for i in range(len(self.group_ols_list)):
            idx = np.where(group_arr == i)[0]  # (n_obs_in_group_i, )
            self.group_ols_list[i] = OLS(Y[idx], T[idx]).fit()  # fit the OLS estimator

        return self 

    def estimate_treatment_effect(self, X: np.ndarray) -> np.ndarray:
        group_arr = np.array([self.group_func(x) for x in X])  # (n_obs_, )

        te_est = np.zeros(X.shape[0])  # (n_obs, )

        for i in range(len(self.group_ols_list)):
            idx = np.where(group_arr == i)[0]  # (n_obs_in_group_i, )
            te_est[idx] = self.group_ols_list[i].params[0]

        return te_est
    
    def get_targeting_decision(self, X: np.ndarray, budget: int = 1) -> np.ndarray:
        te_est = self.estimate_treatment_effect(X)
        targeting_decision_arr = np.zeros(X.shape[0])

        # sort treatment effects
        sorted_idx = np.argsort(-te_est)

        # select the top budget customers
        targeting_decision_arr[sorted_idx[:budget]] = 1

        return targeting_decision_arr
    
    def estimate_targeting_value(self, X: np.ndarray, budget: int = 1) -> float:
        te_est = self.estimate_treatment_effect(X)
        return -np.sort(-te_est)[:budget].sum()


class BinaryBootstrap(object):
    def __init__(self, group_func: callable, n_groups: int):
        # attributes
        self.group_func = group_func 
        self.n_groups = n_groups

        # placeholder for bootstrap records
        self.boot_records = None
    
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray, boot_size: int = 10, n_bootstrap: int = 10000):
        # assign each customer to a group
        group_id_arr = np.array([self.group_func(x) for x in X])

        # initialize bootstrap records
        self.boot_records = np.zeros(shape=(n_bootstrap, self.n_groups))

        # create bootstrap samples and estimate the treatment effect for each group
        for group_id in range(self.n_groups):
            # get the index of customers in group: group_id
            group_idx_arr = np.where(group_id_arr == group_id)[0]
            
            # create a list of bootstrap samples
            group_boot_list = [np.random.choice(group_idx_arr, size=boot_size) for _ in range(n_bootstrap)]

            self.boot_records[:, group_id] = np.array([
                OLS(Y[boot_idx], T[boot_idx]).fit().params[0] for boot_idx in group_boot_list
            ])

        return self

    def estimate_targeting_value(self, X: np.ndarray, budget: int = 1) -> float: 
        # extract bootstrapped treatment effect estimates for each group
        boot_te_arr = np.array([
            self.boot_records[:, self.group_func(x)].flatten() for x in X
        ])  # (n_obs, n_bootstrap)

        # sort the treatment effect estimates for each customer
        sorted_boot_te_arr = -np.sort(-boot_te_arr, axis=0).mean(axis=1)

        # take the top budget treatment effect estimates
        top_budget_te_arr = sorted_boot_te_arr[:budget]

        # calculate the expected maximum
        expected_max = top_budget_te_arr.sum()

        return expected_max


class BinaryCorrection(object):
    def __init__(self, group_func: callable, n_groups: int):
        self.group_func = group_func 
        self.n_groups = n_groups
        self.plugin_estimator = BinaryPlugIn(group_func=group_func, n_groups=n_groups)

        # place holders
        self.n_bootstrap = None  # number of bootstrap samples
        self.boot_est_arr = None  # (n_bootstrap, n_groups)
        self.boot_se_arr = None  # (n_bootstrap, n_groups)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray, n_bootstrap: int = 100):
        # fill in placeholders
        self.n_bootstrap = n_bootstrap
        self.boot_est_arr = np.zeros(shape=(self.n_bootstrap, self.n_groups))  # (n_bootstrap, n_groups)
        self.boot_se_arr = np.zeros(shape=(self.n_bootstrap, self.n_groups))  # (n_bootstrap, n_groups)

        # assign each customer to a group
        group_arr = np.array([self.group_func(x) for x in X])  # (n_obs, )

        for b in range(self.n_bootstrap):
            for i in range(self.n_groups):
                # sample with replacement from each group
                group_idx = np.where(group_arr == i)[0]
                boot_idx = np.random.choice(group_idx, size=group_idx.shape[0], replace=True)

                # fit OLS
                ols = OLS(Y[boot_idx], T[boot_idx]).fit()
                self.boot_est_arr[b, i] = ols.params[0]
                self.boot_se_arr[b, i] = ols.bse[0]

        self.plugin_estimator.fit(X, T, Y)

        return self
    
    def estimate_targeting_value(self, X: np.ndarray, budget: int = 1) -> np.ndarray:
        """   
        Given a set of covariates, estimate the treatment effect for each covariate.

        Params:
        -------
        X: np.ndarray, shape (M, 1), the covariates
        budget: int, the number of customers to target

        Returns:
        -------
        np.ndarray, shape (n_bootstraps, n_oobs)
        """
        # assign each customer to a group
        group_arr = np.array([self.group_func(x) for x in X])  # (n_obs, )

        # create an array to store the treatment effect estimates for each bootstrap sample
        boot_est_arr = self.boot_est_arr[:, group_arr]  # (n_bootstraps, n_obs)
        boot_se_arr = self.boot_se_arr[:, group_arr]  # (n_bootstraps, n_obs)

        # calculate empirical estimation error
        xi_arr = boot_est_arr - boot_est_arr.mean(axis=0)  # (n_bootstraps, n_obs)

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
    

class BinaryAdjustedCorrection(object):
    def __init__(self, group_func: callable, n_groups: int):
        # attributes
        self.group_func = group_func 
        self.n_groups = n_groups

        # place holders
        self.n_bootstrap = None
        self.boot_ols_list = None
        self.plugin_estimator = BinaryPlugIn(group_func=group_func, n_groups=n_groups)
        self.plugin_target_arr = None 

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray, n_bootstrap: int = 100):
        # fill in placeholders
        self.n_bootstrap = n_bootstrap
        self.boot_ols_list = [[None] * self.n_groups for _ in range(self.n_bootstrap)]  # (n_bootstrap, n_groups)

        # assign each customer to a group
        group_arr = np.array([self.group_func(x) for x in X])  # (n_obs, )

        for b in range(self.n_bootstrap):
            for i in range(self.n_groups):
                group_idx = np.where(group_arr == i)[0]
                boot_idx = np.random.choice(group_idx, size=group_idx.shape[0], replace=True)

                self.boot_ols_list[b][i] = OLS(Y[boot_idx], T[boot_idx]).fit()

        self.plugin_estimator.fit(X, T, Y)

        return self
    
    def estimate_targeting_value(self, X: np.ndarray, budget: int = 1) -> np.ndarray:
        """   
        Given a set of covariates, estimate the treatment effect for each covariate.

        Params:
        -------
        X: np.ndarray, shape (n_obs, 1), the covariates
        budget: int, the number of customers to target

        Returns:
        -------
        np.ndarray, shape (n_bootstraps, n_obs)
        """
        # assign each customer to a group
        group_arr = np.array([self.group_func(x) for x in X])  # (n_obs, )

        # create an array to store the treatment effect estimates for each bootstrap sample
        boot_params_arr = np.array([
            [est.params[0] for est in boot_est_list] for boot_est_list in self.boot_ols_list
        ])  # (n_bootstrap, n_groups)

        # retrieve the treatment effect estimates for each individuals 
        te_est = boot_params_arr[:, group_arr] # (n_bootstraps, n_obs)

        del group_arr, boot_params_arr

        # calculate empirical estimation error
        xi_arr = te_est - te_est.mean(axis=0)  # (n_bootstraps, n_obs)

        # create a mask to identify the treated individuals within each bootstrap sample
        # if the treatment effect estimate is among the largests, the mask is 1, otherwise 0
        # if ties, select the ones with smaller indexes
        mask_arr = np.zeros_like(xi_arr)
        # sort the treatment effect estimates in descending order for each bootstrap sample
        sorted_columns = np.argsort(-te_est, axis=1)[:, :budget]

        # set the mask to 1 for the treated individuals
        rows = np.repeat(np.arange(te_est.shape[0])[:, None], budget, axis=1)
        mask_arr[rows, sorted_columns] = 1

        del sorted_columns, rows

        # create an array to store the plugin targeting decision
        plugin_target_arr = np.repeat(
            self.plugin_estimator.get_targeting_decision(X, budget).reshape(1, -1), self.n_bootstrap, axis=0
        )

        # choose the rows where the plugin estimator and the corrected estimator agree
        row_id_arr = np.where((mask_arr - plugin_target_arr != 0).sum(axis=1) == 0)[0]

        # calculate the correction term
        correction = (mask_arr * xi_arr)[row_id_arr, :].sum(axis=1).mean()

        # calculate plugin estimate
        plugin_estimate = self.plugin_estimator.estimate_targeting_value(X, budget=budget)

        # calculate the corrected treatment effect estimate
        return plugin_estimate - correction


class BinaryDoubleBootCorrection(object):
    def __init__(self, group_func: callable, n_groups: int):
        self.group_func = group_func 
        self.n_groups = n_groups
        self.plugin_estimator = BinaryPlugIn(group_func=group_func, n_groups=n_groups)

        # place holders
        self.n_fl_bootstrap = None  # number of first level bootstraps
        self.fl_boot_est_arr = None  # (n_fl_bootstrap, n_groups)
        self.fl_boot_se_arr = None  # (n_filbootstrap, n_groups)

        self.n_sl_bootstrap = None  # number of second level bootstraps
        self.sl_boot_est_arr = None  # (n_fl_bootstrap, n_sl_bootstrap, n_groups)
        self.sl_boot_se_arr = None  # (n_fl_bootstrap, n_sl_bootstrap, n_groups)

    def fit(
        self, X: np.ndarray, T: np.ndarray, Y: np.ndarray, n_first_bootstrap: int = 100, 
        n_second_bootstrap: int = 100
    ):
        # fill in placeholders
        self.n_fl_bootstrap = n_first_bootstrap
        self.n_sl_bootstrap = n_second_bootstrap

        # first level bootstrap results: (n_fl_bootstrap, n_groups)
        self.fl_boot_est_arr = np.zeros(shape=(self.n_fl_bootstrap, self.n_groups))
        self.fl_boot_se_arr = np.zeros(shape=(self.n_fl_bootstrap, self.n_groups))

        # second level bootstrap results: (n_fl_bootstrap, n_second_bootstrap, n_groups)
        self.sl_boot_est_arr = np.zeros(shape=(self.n_fl_bootstrap, self.n_sl_bootstrap, self.n_groups)) 
        self.sl_boot_se_arr = np.zeros(shape=(self.n_fl_bootstrap, self.n_sl_bootstrap, self.n_groups))

        # assign each customer to a group
        group_arr = np.array([self.group_func(x) for x in X])  # (n_obs, )

        # fit OLS models for each group for each bootstrap sample, both first and second level bootstraps
        for k in range(self.n_groups):
            group_idx = np.where(group_arr == k)[0]
            fl_boot_idx = np.random.choice(group_idx, size=group_idx.shape[0], replace=True)
            for b in range(self.n_fl_bootstrap):
                ols = OLS(Y[fl_boot_idx], T[fl_boot_idx]).fit()
                self.fl_boot_est_arr[b, k], self.fl_boot_se_arr[b, k] = ols.params[0], ols.bse[0]
                for r in range(self.n_sl_bootstrap):
                    sl_boot_idx = np.random.choice(fl_boot_idx, size=fl_boot_idx.shape[0], replace=True)
                    ols = OLS(Y[sl_boot_idx], T[sl_boot_idx]).fit()
                    self.sl_boot_est_arr[b, r, k], self.sl_boot_se_arr[b, r, k] = ols.params[0], ols.bse[0]
        self.plugin_estimator.fit(X, T, Y)

        return self

    def estimate_targeting_value(self, X: np.ndarray, budget: int = 1) -> np.ndarray:
        """   
        Given a set of covariates, estimate the treatment effect for each covariate.

        Params:
        -------
        X: np.ndarray, shape (n_oobs, 1), the covariates
        budget: int, the number of customers to target

        Returns:
        -------
        np.ndarray, shape (n_bootstraps, n_oobs)
        """
        # 0. preparation
        # assign each customer to a group
        group_arr = np.array([self.group_func(x) for x in X])  # (n_obs, )

        # 1. calculation for first level bootstraps
        # retrieve the treatment effect estimates for each individuals 
        fl_boot_est_arr = self.fl_boot_est_arr[:, group_arr] # (n_fl_bootstraps, n_obs)
        fl_boot_se_arr = self.fl_boot_se_arr[:, group_arr]  # (n_fl_bootstraps, n_obs)

        # average the targeting value estimates over first level bootstraps
        fl_boot_est = -np.sort(-fl_boot_est_arr, axis=1)[:, :budget].sum(axis=1).mean()

        # 2. calculation for second level bootstraps
        # create an array to store the treatment effect estimates for each bootstrap sample
        sl_boot_est_arr = self.sl_boot_est_arr[:, :, group_arr]  # (n_first_bootstrap, n_second_bootstrap, n_obs)
        sl_boot_se_arr = self.sl_boot_se_arr[:, :, group_arr]  # (n_first_bootstrap, n_second_bootstrap, n_obs)

        # calculate the corrected treatment effect estimate
        sl_boot_est = -np.sort(-sl_boot_est_arr, axis=2)[:, :, :budget].sum(axis=2).mean()

        # 3. calculate plugin estimate
        plugin_est = self.plugin_estimator.estimate_targeting_value(X, budget=budget)

        # calculate the biases
        correction = fl_boot_est - plugin_est 
        correction_bias = sl_boot_est - fl_boot_est
        final_correction = correction - correction_bias  # 2 * fl_boot_est - sl_boot_est + plugin_est

        # calculate the corrected treatment effect estimate
        return plugin_est - final_correction


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
    
#     def estimate_targeting_value(self, X: np.ndarray, budget: int = 1, sample_size: int = 10000):
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
