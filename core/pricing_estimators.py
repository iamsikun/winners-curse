import os 
import sys 
sys.path.insert(0, os.path.abspath('.'))

import numpy as np 

from scipy.stats import ks_2samp
from scipy.optimize import minimize, OptimizeResult
from sklearn.linear_model import LogisticRegression


from core.estimators import BaseEstimator


class PricingPlugIn(BaseEstimator):
    def __init__(self, cov_dim: int):
        # attributes
        self.cov_dim = cov_dim

        # placeholders
        self.util_params = np.zeros((2 * self.cov_dim + 1, 1))  # (2 * cov_dim + 1, 1)
        self.opt_result = None

    def fit(self, covariates: np.ndarray, prices: np.ndarray, outcomes: np.ndarray):
        # fit the utility function
        model = self.logistic_regression(covariates, prices, outcomes)

        self.util_params[0] = model.intercept_  # (1, )
        self.util_params[1:] = model.coef_.T  # (2 * cov_dim, 1)

        return self
    
    @staticmethod
    def logistic_regression(
        covariates: np.ndarray, prices: np.ndarray, outcomes: np.ndarray
    ) -> LogisticRegression:
        exog = np.concatenate([covariates, prices * covariates], axis=1)
        model = LogisticRegression().fit(exog, outcomes.flatten())

        return model

    def purchase_prob(
            self, covariates: np.ndarray, price: float, 
            util_params: np.ndarray = None
    ) -> np.ndarray:
        """ 
        Params:
        -------
        covariates: np.ndarray, (n_obs, cov_dim)
        price: float, price
        """
        if util_params is None:
            util_params = self.util_params
        # variables: (sample_size, 2 * cov_dim + 1)
        var_arr = np.concatenate([np.ones((covariates.shape[0], 1)), covariates, price * covariates], axis=1)
        logit = np.exp(var_arr @ util_params)
        return logit / (1 + logit)
    
    def objective_func(
        self, covariates: np.ndarray, price: float, delta: float = 0.99, 
        util_params: np.ndarray = None
    ) -> float:
        purchase_prob = self.purchase_prob(
            covariates=covariates, price=price, util_params=util_params
        )
        return np.mean(price * purchase_prob / (1 - delta * purchase_prob))
    
    def optimize(
        self, covariates: np.ndarray, delta: float = 0.99, 
        util_params: np.ndarray = None
    ) -> OptimizeResult:
        """ 
        
        Params: 
        -------
        covariates: np.ndarray, (n_obs, cov_dim)
        delta: float, discount factor

        Returns:
        -------
        opt_result: scipy.optimize.OptimizeResult
        """
        self.opt_result = minimize(
            lambda x: -self.objective_func(
                covariates=covariates, price=x, delta=delta, 
                util_params=util_params
            ), 
            x0=1, method='L-BFGS-B',
        )
        return self.opt_result
    
    def estimate_targeting_value(self, covariates: np.ndarray, delta: float = 0.99) -> float:
        """ 
        Estimate the targeting value

        Params:
        -------
        covariates: np.ndarray, (n_obs, cov_dim)
        delta: float, discount factor

        Returns:
        -------
        targeting_value: float
        """
        if self.opt_result is None:
            _ = self.optimize(covariates, delta)

        return - self.opt_result.fun
    
    def get_targeting_policy(self, covariates: np.ndarray, delta: float = 0.99) -> float:
        """ 
        Get the targeting policy

        Params:
        -------
        covariates: np.ndarray, (n_obs, cov_dim)
        delta: float, discount factor

        Returns:
        -------
        targeting_policy: np.ndarray, (n_obs, )
        """
        if self.opt_result is None:
            _ = self.optimize(covariates, delta)
        
        return self.opt_result.x[0]
    

class PricingValueCorrection(PricingPlugIn):
    def __init__(self, cov_dim: int, plugin_estmr: PricingPlugIn):
        # attributes
        self.cov_dim = cov_dim

        # place holders
        self.n_bootstraps = None  # number of bootstrap samples
        self.boot_util_params = None  # (n_bootstraps, 1 + 2 *cov_dim)
        self.plugin_estmr = plugin_estmr  # plugin estimator
        self.boot_targ_val_list = []

    def fit(self, covariates: np.ndarray, prices: np.ndarray, outcomes: np.ndarray, n_bootstraps: int = 100):
        # fill in placeholders
        self.n_bootstraps = n_bootstraps
        self.boot_util_params = np.zeros((n_bootstraps, 1 + 2 * self.cov_dim))

        # bootstrap sample indices, shape (num_bootstraps, m)
        boot_index_arr = np.random.choice(
            np.arange(covariates.shape[0]), size=(self.n_bootstraps, covariates.shape[0]), replace=True
        )  
        
        boot_cov_arr = covariates[boot_index_arr]  # (n_bootstraps, sample_size, cov_dim)
        boot_outcome_arr = outcomes[boot_index_arr]  # (n_bootstraps, sample_size)
        boot_price_arr = prices[boot_index_arr]  # (n_bootstraps, sample_size)

        for boot_id in range(self.n_bootstraps):
            model = self.logistic_regression(
                covariates=boot_cov_arr[boot_id], 
                prices=boot_price_arr[boot_id], 
                outcomes=boot_outcome_arr[boot_id]
            )
            self.boot_util_params[boot_id, 0] = model.intercept_  # (1, )
            self.boot_util_params[boot_id, 1:] = model.coef_.flatten()  # (2 * cov_dim, 1)

        # drop zero rows
        self.boot_util_params = self.boot_util_params[~np.all(self.boot_util_params == 0, axis=1)]

        return self
    
    def estimate_targeting_value(
        self, covariates: np.ndarray, delta: float = 0.99
    ) -> np.ndarray:
        # optimize targeting for each bootstrap sample
        self.boot_targ_val_list = []
        for boot_id in range(self.boot_util_params.shape[0]):
            opt_result = self.optimize(
                covariates=covariates, delta=delta, 
                util_params=self.boot_util_params[boot_id]
            )
            if opt_result.success:
                self.boot_targ_val_list.append(-opt_result.fun)

        # calculate plugin estimate
        plugin_estimate = self.plugin_estmr.estimate_targeting_value(
            covariates=covariates, delta=delta
        )

        # calculate the corrected treatment effect estimate
        return 2 * plugin_estimate - np.nanmean(self.boot_targ_val_list)
