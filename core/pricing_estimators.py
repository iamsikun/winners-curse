import os 
import sys 
sys.path.insert(0, os.path.abspath('.'))

import numpy as np 

from scipy.stats import ks_2samp
from scipy.optimize import minimize, OptimizeResult
from statsmodels.discrete.discrete_model import Logit, BinaryResultsWrapper


from core.estimators import BaseEstimator


def calculate_purchase_prob(variables: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ 
    Compute the purchase probability for a given price and covariates. 
    The purchase probability is given by the logit model:
    P(purchase | variables) = exp(variables @ params) / (1 + exp(variables @ params))

    Params: 
    -------
    variables: np.ndarray, shape (sample_size, n_variables)
        The covariates of the customers. 
    params: np.ndarray, shape (n_variables, 1)
        The parameters of the logit model.

    Returns:
    --------
    purchase_prob: np.ndarray, shape (sample_size, 1)
        The purchase probability for each customer.
    """
    logit = np.exp(variables @ params)
    return logit / (1 + logit)

def obj_func(
    covariates: np.ndarray, price: float, params: np.ndarray, add_const: bool = True
) -> float:
    """ 
    Compute the revenue for a given price and a customer, characterized by their covariates.
    
    Params:
    -------
    covariates: np.ndarray, shape (1, n_covariates)
        The covariates of the customer to target.
    price: float
        The price of the product.
    params: np.ndarray, shape (n_covariates,)
        The parameters of the logit model.
    add_const: bool, default=True
        Whether to add a constant to the covariates.
    """
    if add_const:
        variables = np.concatenate([
            np.ones((covariates.shape[0], 1)), covariates, price * covariates
        ], axis=1)
    else:
        variables = np.concatenate([covariates, price * covariates], axis=1)

    purchase_prob = calculate_purchase_prob(variables, params)[0, 0]

    return price * purchase_prob

def optimize_targeting(
    covariates: np.ndarray, params: np.ndarray, add_const: bool = True, 
    price_lb: float = 0, price_ub: float = 100,
) -> OptimizeResult:
    opt_result = minimize(
        lambda x: - obj_func(
            covariates=covariates, price=x, params=params, add_const=add_const
        ), 
        x0=1, method='L-BFGS-B', bounds=[(price_lb, price_ub)]
    )

    return opt_result


class PricingPlugIn(BaseEstimator):
    def __init__(self, cov_dim: int):
        # attributes
        self.cov_dim = cov_dim

        # placeholders
        self.util_params = np.zeros((2 * self.cov_dim + 1, 1))  # (2 * cov_dim + 1, 1)
        self.model = None  # fitted model
        self.opt_result = None

    def fit(self, covariates: np.ndarray, prices: np.ndarray, outcomes: np.ndarray):
        # fit the utility function
        self.model = self.logistic_regression(covariates, prices, outcomes)

        # assign utility parameters
        self.util_params = self.model.params.reshape(-1, 1)  # (2 * cov_dim + 1, 1)

        return self
    
    @staticmethod
    def logistic_regression(
        covariates: np.ndarray, prices: np.ndarray, outcomes: np.ndarray
    ) -> BinaryResultsWrapper:
        # prepare exogenous variables
        covariates_w_const = np.concatenate([np.ones((covariates.shape[0], 1)), covariates], axis=1)
        exog = np.concatenate([
            covariates_w_const, prices * covariates
        ], axis=1)

        # fit model
        model = Logit(endog=outcomes, exog=exog).fit()

        return model
    
    def evaluate(self, covariates: np.ndarray, price: float) -> float:
        """ 
        Evaluate the revenue for a given price and covariates

        Params:
        -------
        covariates: np.ndarray, (1, cov_dim)
        price: float

        Returns:
        -------
        revenue: float
        """
        return obj_func(covariates, price, self.util_params, add_const=True)
    
    def optimize(
        self, covariates: np.ndarray, price_lb: float = 0, price_ub: float = 100,
    ) -> tuple:
        """ 
        Get the targeting policy

        Params:
        -------
        covariates: np.ndarray, (1, cov_dim)
        price_lb: float, lower bound of the price
        price_ub: float, upper bound of the price

        Returns:
        -------
        tuple: 
            - price: float, optimal price
            - revenue: float, revenue at the optimal price
        """
        self.opt_result = optimize_targeting(
            covariates=covariates, params=self.util_params, price_lb=price_lb, price_ub=price_ub, 
            add_const=True
        )
        return self.opt_result.x[0], -self.opt_result.fun
    

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
