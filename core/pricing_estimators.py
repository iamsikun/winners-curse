import os 
import sys 
sys.path.insert(0, os.path.abspath('.'))

import numpy as np 

from scipy.stats import ks_2samp
from scipy.optimize import minimize, OptimizeResult
from statsmodels.discrete.discrete_model import Logit


from core.estimators import BaseEstimator


class PricingPlugIn(BaseEstimator):
    def __init__(self, cov_dim: int):
        # attributes
        self.cov_dim = cov_dim

        # placeholders
        self.util_const_map = np.zeros((self.cov_dim, 1))  # (cov_dim, 1)
        self.util_price_map = np.zeros((self.cov_dim, 1))  # (cov_dim, 1)
        self.opt_result = None

    def fit(self, covariates: np.ndarray, prices: np.ndarray, outcomes: np.ndarray):
        # fit the utility function
        exog = np.concatenate([covariates, prices * covariates], axis=1)
        model = Logit(outcomes, exog)
        result = model.fit()

        self.util_const_map = result.params[:self.cov_dim].reshape(-1, 1)  # (cov_dim, 1)
        self.util_price_map = result.params[self.cov_dim:].reshape(-1, 1)  # (cov_dim, 1)

        return self

    def purchase_prob(self, covariates: np.ndarray, price: float) -> np.ndarray:
        """ 
        Params:
        -------
        covariates: np.ndarray, (n_obs, cov_dim)
        price: float, price
        """
        logit = np.exp(covariates @ self.util_const_map + (price * covariates) @ self.util_price_map)
        return logit / (1 + logit)
    
    def objective_func(self, covariates: np.ndarray, price: float, delta: float = 0.99) -> float:
        purchase_prob = self.purchase_prob(covariates, price)
        return np.mean(price * purchase_prob / (1 - delta * purchase_prob))
    
    def optimize(self, covariates: np.ndarray, delta: float = 0.99) -> OptimizeResult:
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
            lambda x: -self.objective_func(covariates, x, delta), x0=1, method='L-BFGS-B',
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
    
