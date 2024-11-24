import os 
import sys 
sys.path.insert(0, os.path.abspath('.'))

import numpy as np 

from scipy.stats import ks_2samp
from scipy.optimize import minimize, OptimizeResult
from statsmodels.discrete.discrete_model import Logit, BinaryResultsWrapper
from sklearn.linear_model import LogisticRegression


from archive.segment_targeting_estimators import BaseEstimator


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


def logistic_regression(
    covariates: np.ndarray, prices: np.ndarray, outcomes: np.ndarray
):
    """ 
    Fit a logistic regression model to the data

    Params:
    -------
    covariates: np.ndarray, (n_samples, cov_dim)
        The covariates of the customers
    prices: np.ndarray, (n_samples, 1)
        The price of the product
    outcomes: np.ndarray, (n_samples, 1)
        The purchase outcomes

    Returns:
    --------
    model
    """
    # prepare exogenous variables
    variables = np.concatenate([covariates, prices * covariates], axis=1)

    # fit model
    model = LogisticRegression().fit(variables, outcomes)

    return model


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
        self.model = logistic_regression(covariates, prices, outcomes)

        # assign utility parameters
        self.util_params = np.concatenate([
            self.model.intercept_, self.model.coef_.flatten()
        ]).reshape(-1, 1)  # (2 * cov_dim + 1, 1)

        return self
    
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
    
    def optimize(self, covariates: np.ndarray, price_lb: float = 0, price_ub: float = 100) -> tuple:
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
            covariates=covariates, params=self.util_params, 
            price_lb=price_lb, price_ub=price_ub, 
            add_const=True
        )
        return self.opt_result.x[0], -self.opt_result.fun
    
    def estimate_targeting_value(self, covariates: np.ndarray, price_lb: float = 0, price_ub: float = 100) -> float:
        """ 
        Estimate the targeting value for a given covariate

        Params:
        -------
        covariates: np.ndarray, (n_samples, cov_dim)

        Returns:
        -------
        float: the estimated targeting value
        """
        _, targ_est = self.optimize(covariates, price_lb=price_lb, price_ub=price_ub)

        return targ_est
    

class PricingValueCorrection(PricingPlugIn):
    """ 
    Correction estimator for the targeting value on the value function level

    The empirical distribution of the value function is constructred via standard nonparametric bootstrap
    """
    def __init__(self, cov_dim: int):
        # attributes
        self.cov_dim = cov_dim

        # place holders
        self.n_bootstraps = None  # number of bootstrap samples
        self.boot_util_params = None  # (n_bootstraps, 1 + 2 *cov_dim)
        self.boot_targ_est_list = []
        self.boot_targ_emp_est_list = []

        # initialize plugin estimator
        self.plugin_estmr = PricingPlugIn(cov_dim=cov_dim)

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
            model = logistic_regression(
                covariates=boot_cov_arr[boot_id], 
                prices=boot_price_arr[boot_id], 
                outcomes=boot_outcome_arr[boot_id]
            )
            self.boot_util_params[boot_id] = np.concatenate([
                model.intercept_, model.coef_.flatten()
            ])  # (1 + 2 * cov_dim, )

        # fit plugin estimator
        self.plugin_estmr = self.plugin_estmr.fit(covariates=covariates, prices=prices, outcomes=outcomes)

        # # drop zero rows
        # self.boot_util_params = self.boot_util_params[~np.all(self.boot_util_params == 0, axis=1)]

        return self
    
    def estimate_targeting_value(self, covariates: np.ndarray, price_lb: float = 0, price_ub: float = 100) -> np.ndarray:
        # optimize targeting for each bootstrap sample
        for boot_id in range(self.boot_util_params.shape[0]):
            opt_result = optimize_targeting(
                covariates=covariates, 
                params=self.boot_util_params[boot_id].reshape(-1, 1), 
                price_lb=price_lb, price_ub=price_ub, 
                add_const=True
            )
            if opt_result.success:
                self.boot_targ_est_list.append(-opt_result.fun)
                
                # evaluate bootstrap poligy using plugin estimates
                self.boot_targ_emp_est_list.append(obj_func(
                    covariates=covariates, price=opt_result.x[0], 
                    params=self.plugin_estmr.util_params, add_const=True
                ))

        # calculate plugin estimate
        plugin_est = self.plugin_estmr.estimate_targeting_value(covariates=covariates)


        # calculate the corrected treatment effect estimate
        return plugin_est - np.nanmean(self.boot_targ_est_list) + np.nanmean(self.boot_targ_emp_est_list)
    

class PricingMNBValueCorrection(PricingPlugIn):
    """ 
    Correction estimator for the targeting value on the value function level

    The empirical distribution of the value function is constructred via m-out-of-n bootstrap
    
    Rule for selecting the best m follows (Bickle and Sakov 2008, Statistica Sinica)
    """
    def __init__(self, cov_dim: int):
        # attributes
        self.cov_dim = cov_dim

        # place holders
        self.n_bootstraps = None  # number of bootstrap samples
        self.m_list = []
        self.m_boot_util_params_list = []  # list of params with shape (n_bootstraps, 1 + 2 *cov_dim)

        # initialize plugin estimator
        self.plugin_estmr = PricingPlugIn(cov_dim=cov_dim)

    def fit(
        self, covariates: np.ndarray, prices: np.ndarray, outcomes: np.ndarray, n_bootstraps: int = 100, 
        q: float = 0.9, max_j: int = 20
    ):
        # fill in placeholders
        self.n_bootstraps = n_bootstraps

        # calculate the bootstrap sample sizes
        sample_size = covariates.shape[0]
        self.m_list = np.array([int(q**j * sample_size) for j in range(max_j)])

        # create bootstraps
        for m in self.m_list:
            # initialzie 
            boot_util_params = np.zeros((n_bootstraps, 1 + 2 * self.cov_dim))

            # bootstrap sample indices, shape (num_bootstraps, m)
            boot_index_arr = np.random.choice(
                np.arange(sample_size), size=(self.n_bootstraps, m), replace=True
            )  
        
            # create bootstrap samples
            boot_cov_arr = covariates[boot_index_arr]  # (n_bootstraps, sample_size, cov_dim)
            boot_outcome_arr = outcomes[boot_index_arr]  # (n_bootstraps, sample_size)
            boot_price_arr = prices[boot_index_arr]  # (n_bootstraps, sample_size)

            for boot_id in range(self.n_bootstraps):
                model = logistic_regression(
                    covariates=boot_cov_arr[boot_id], 
                    prices=boot_price_arr[boot_id], 
                    outcomes=boot_outcome_arr[boot_id]
                )
                boot_util_params[boot_id] = np.concatenate([
                    model.intercept_, model.coef_.flatten()
                ])  # (1 + 2 * cov_dim, )

            # update attributes
            self.m_boot_util_params_list.append(boot_util_params)

        # fit plugin estimator
        self.plugin_estmr = self.plugin_estmr.fit(covariates=covariates, prices=prices, outcomes=outcomes)

        # # drop zero rows
        # self.boot_util_params = self.boot_util_params[~np.all(self.boot_util_params == 0, axis=1)]

        return self
    
    def estimate_targeting_value(self, covariates: np.ndarray, price_lb: float = 0, price_ub: float = 100) -> np.ndarray:
        boot_dstn_list = [None] * len(self.m_list)  # store the bootstrap distributions for each m
        
        for i in range(len(self.m_list)):
            boot_targ_est_list = []
            boot_targ_emp_est_list = []

            # optimize targeting for each bootstrap sample
            for boot_id in range(self.n_bootstraps):
                opt_result = optimize_targeting(
                    covariates=covariates, 
                    params=self.m_boot_util_params_list[i][boot_id].reshape(-1, 1), 
                    price_lb=price_lb, price_ub=price_ub, 
                    add_const=True
                )
                if opt_result.success:
                    boot_targ_est_list.append(-opt_result.fun)
                    # evaluate bootstrap poligy using plugin estimates
                    boot_targ_emp_est_list.append(obj_func(
                        covariates=covariates, price=opt_result.x[0], 
                        params=self.plugin_estmr.util_params, add_const=True
                    ))

            boot_dstn_list[i] = np.array(boot_targ_est_list) - np.array(boot_targ_emp_est_list)

        # choose the best m
        discp_list = [None] * (len(self.m_list) - 1)
        for idx, (prev_dstn, current_dstn) in enumerate(zip(boot_dstn_list[:-1], boot_dstn_list[1:])):
            ks_stat, _ = ks_2samp(prev_dstn, current_dstn)
            discp_list[idx] = ks_stat
        min_discp_idx = np.argmin(discp_list)
        correction = np.nanmean(boot_dstn_list[min_discp_idx])

        # calculate plugin estimate
        plugin_est = self.plugin_estmr.estimate_targeting_value(covariates=covariates)

        # calculate the corrected treatment effect estimate
        return plugin_est - correction
