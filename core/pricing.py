
import os 
import sys 
sys.path.insert(0, os.path.abspath('.'))
from tqdm import tqdm

import pandas as pd 
import numpy as np

from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar

import core.dgp as dgp

class DemandModel(object):
    def fit(self):
        raise NotImplementedError
    
    def predict_proba(self) -> np.ndarray:
        raise NotImplementedError

    def predict_choice(self) -> np.ndarray:
        raise NotImplementedError

class MLELogit(DemandModel):
    def __init__(self):
        self.model = None 

    def fit(self, data: pd.DataFrame):
        # extract and transform data
        price_arr = data['price'].values.reshape(-1, 1)  # shape = (sample_size, 1)
        covariates_arr = data[[col for col in data.columns if col.startswith('cov_')]].values  # shape = (sample_size, num_covariates)

        # exogenous variables
        variables = np.concatenate([covariates_arr, price_arr, covariates_arr * price_arr], axis=1)

        # fit model
        self.model = LogisticRegression(
            penalty=None, 
            solver='lbfgs',
            fit_intercept=False, 
            max_iter=1000
        ).fit(X=variables, y=data['choice'].values)

        return self
    
    def predict_proba(self, customers: np.ndarray, prices: np.ndarray) -> np.ndarray:
        """ 
        Predict the purchasing probability for a given set of customers.

        Params:
        -------
        customers: np.ndarray, shape (n_customers, n_features)
            Customer data with covariates.

        prices: np.ndarray, shape (n_customers,)
            Prices of the products.

        Returns:
        --------
        proba: np.ndarray, shape (n_customers, 2)
        """
        assert prices.shape[0] == customers.shape[0], f"Number of prices should be equal to the number of customers."
        assert len(prices.shape) == 1, f"Prices should be a 1D array."

        # exogenous variables
        variables = np.concatenate([customers, prices.reshape(-1, 1), customers * prices.reshape(-1, 1)], axis=1)

        # compute purchasing probability
        purchase_proba = expit(variables @ self.coef_).reshape(-1, 1)
        
        return np.hstack([1 - purchase_proba, purchase_proba])
    
    def predict_choice(self, consumers: np.ndarray, prices: np.ndarray) -> np.ndarray:
        """ 
        Predict the purchasing choice for a given set of customers.

        Params:
        -------
        customers: np.ndarray, shape (n_customers, n_features)
            Customer data with covariates.

        prices: np.ndarray, shape (n_customers,)
            Prices of the products.

        Returns:
        --------
        choice: np.ndarray, shape (n_customers,)
            Purchasing choice for each customer.
        """
        return self.predict_proba(consumers, prices).argmax(axis=1)
    
    @property
    def coef_(self) -> np.ndarray:
        """ 
        Returns the estimated coefficients of the model.

        Returns:
        --------
        coef_: np.ndarray, shape (n_features * 2 + 1,)
            Estimated coefficients of the model.
        """
        if self.model is None:
            raise ValueError("Model not fitted yet.")
        return self.model.coef_.flatten()  # shape = (n_features * 2 + 1,)
    
    def evaluate(self, customers: np.ndarray, prices: np.ndarray, dgp: dgp.Pricing) -> float:
        prob_pred = self.predict_proba(customers, prices)[:, 1]
        prob_true = dgp.predict_proba(customers, prices)[:, 1]

        choice_pred = (prob_pred > 0.5).astype(int)
        choice_true = (prob_true > 0.5).astype(int)

        return {
            'accuracy': (choice_pred == choice_true).mean(),
            'rmse': np.sqrt(((prob_pred - prob_true) ** 2).mean())
        }
    

class LassoLogit(MLELogit):
    def __init__(self):
        self.model = None

    def fit(self, data: pd.DataFrame, seed: int =None):
        # set seed
        if seed is not None:
            np.random.seed(seed)

        # extract and transform data
        price_arr = data['price'].values.reshape(-1, 1)  # shape = (sample_size, 1)
        covariates_arr = data[[col for col in data.columns if col.startswith('cov_')]].values  # shape = (sample_size, num_covariates)

        # exogenous variables
        variables = np.concatenate([covariates_arr, price_arr, covariates_arr * price_arr], axis=1)

        self.model = LogisticRegression(
            penalty='l1',
            C=0.3, 
            solver='liblinear',
            fit_intercept=False,
            max_iter=1000
        ).fit(X=variables, y=data['choice'].values)

        return self
    

class WLBMLELogit(DemandModel):
    def __init__(self, n_bootstraps: int = 100):
        # attributes 
        self.n_bootstraps = n_bootstraps

        # placeholders
        self.model_list = [None] * n_bootstraps
        self.coef_arr = None  # shape = (n_bootstraps, n_features * 2 + 1)

    def fit(self, data: pd.DataFrame, seed: int = None, verbose: bool = False):
        # set seed
        if seed is not None:
            np.random.seed(seed)

        # extract and transform data
        price_arr = data['price'].values.reshape(-1, 1)
        covariates_arr = data[[col for col in data.columns if col.startswith('cov_')]].values
        sample_size = data.shape[0]

        # exogenous variables
        variables = np.concatenate([covariates_arr, price_arr, covariates_arr * price_arr], axis=1)

        # endogenous variables
        outcomes = data['choice'].values

        # update placeholder
        self.coef_arr = np.zeros((self.n_bootstraps, variables.shape[1]))  # shape = (n_bootstraps, n_features * 2 + 1)

        # fit model for each bootstrap
        iterator = range(self.n_bootstraps) if not verbose else tqdm(range(self.n_bootstraps))
        for boot_id in iterator:
            # calculate the weights 
            weights = np.random.exponential(scale=1, size=(sample_size,)) ** 1.8  # 1.8 is the prior from the paper
            boot_indices = np.random.choice(
                np.arange(sample_size), 
                size=sample_size, 
                replace=True, 
                p=weights / weights.sum()
            )

            self.model_list[boot_id] = LogisticRegression(
                penalty=None, 
                solver='lbfgs',
                fit_intercept=False, 
                max_iter=1000
            ).fit(X=variables[boot_indices], y=outcomes[boot_indices])

            self.coef_arr[boot_id] = self.model_list[boot_id].coef_.flatten()

        return self 
    
    @property
    def coef_(self) -> np.ndarray:
        """ 
        Returns the estimated coefficients of the model.

        Returns:
        --------
        coef_: np.ndarray, shape (n_bootstraps, n_features * 2 + 1)
            Estimated coefficients of the model.
        """
        return self.coef_arr
    
    def predict_proba(self, customers: np.ndarray, prices: np.ndarray) -> np.ndarray:
        """ 
        Returns the purchasing probability for a given set of customers.

        Params:
        -------
        customers: np.ndarray, shape (n_customers, n_features)
            Customer data with covariates.

        prices: np.ndarray, shape (n_customers,)
            Prices of the products.

        Returns:
        --------
        proba: np.ndarray, shape (n_customers, n_bootstraps)
            Purchasing probability for each customer.
        """
        assert prices.shape[0] == customers.shape[0], f"Number of prices should be equal to the number of customers."
        assert prices.ndim == 1, f"Prices should be a 1D array."

        # endogenous variables
        variables = np.concatenate([customers, prices.reshape(-1, 1), customers * prices.reshape(-1, 1)], axis=1)

        # compute purchasing probability
        purchase_proba = expit(self.coef_ @ variables.T).T  # shape = (n_customers, n_bootstraps)

        return purchase_proba
    
    def predict_choice(self, customers: np.ndarray, prices: np.ndarray) -> np.ndarray:
        """
        Returns:
        --------
        choice: np.ndarray, shape (n_customers, n_bootstraps)
            Purchasing choice for each customer.
        """
        return (self.predict_proba(customers, prices) > 0.5).astype(int)
    
    def evaluate(self, customers: np.ndarray, prices: np.ndarray, dgp: dgp.Pricing) -> float:
        prob_pred = self.predict_proba(customers, prices)  # shape = (n_customers, n_bootstraps)
        prob_true = dgp.predict_proba(customers, prices)[:, 1].reshape(-1, 1)  # shape = (n_customers, 1)

        choice_pred = (prob_pred > 0.5).astype(int)
        choice_true = (prob_true > 0.5).astype(int)

        return {
            'accuracy': (choice_pred == choice_true).mean(axis=0),  # shape = (n_bootstraps,)
            'rmse': np.sqrt(((prob_pred - prob_true)**2).mean(axis=0))  # shape = (n_bootstraps,)
        }


class WLBLassoLogit(DemandModel):
    def __init__(self, n_bootstraps: int = 100):
        # attributes 
        self.n_bootstraps = n_bootstraps

        # placeholders
        self.model_list = [None] * n_bootstraps
        self.coef_arr = None  # shape = (n_bootstraps, n_features * 2 + 1)

    def fit(self, data: pd.DataFrame, seed: int = None, verbose: bool = False):
        # set seed
        if seed is not None:
            np.random.seed(seed)

        # extract and transform data
        price_arr = data['price'].values.reshape(-1, 1)
        covariates_arr = data[[col for col in data.columns if col.startswith('cov_')]].values
        sample_size = data.shape[0]

        # exogenous variables
        variables = np.concatenate([covariates_arr, price_arr, covariates_arr * price_arr], axis=1)

        # endogenous variables
        outcomes = data['choice'].values

        # update placeholder
        self.coef_arr = np.zeros((self.n_bootstraps, variables.shape[1]))  # shape = (n_bootstraps, n_features * 2 + 1)

        # fit model for each bootstrap
        iterator = range(self.n_bootstraps) if not verbose else tqdm(range(self.n_bootstraps))
        for boot_id in iterator:
            # calculate the weights 
            weights = np.random.exponential(scale=1, size=(sample_size,)) ** 1.8  # 1.8 is the prior from the paper
            boot_indices = np.random.choice(
                np.arange(sample_size), 
                size=sample_size, 
                replace=True, 
                p=weights / weights.sum()
            )

            self.model_list[boot_id] = LogisticRegression(
                penalty='l1',
                C=0.3, 
                solver='liblinear',
                fit_intercept=False,
                max_iter=1000
            ).fit(X=variables[boot_indices], y=outcomes[boot_indices])

            self.coef_arr[boot_id] = self.model_list[boot_id].coef_.flatten()

        return self 
    
    @property
    def coef_(self) -> np.ndarray:
        """ 
        Returns the estimated coefficients of the model.

        Returns:
        --------
        coef_: np.ndarray, shape (n_bootstraps, n_features * 2 + 1)
            Estimated coefficients of the model.
        """
        return self.coef_arr
    
    def predict_proba(self, customers: np.ndarray, prices: np.ndarray) -> np.ndarray:
        """ 
        Returns the purchasing probability for a given set of customers.

        Params:
        -------
        customers: np.ndarray, shape (n_customers, n_features)
            Customer data with covariates.

        prices: np.ndarray, shape (n_customers,)
            Prices of the products.

        Returns:
        --------
        proba: np.ndarray, shape (n_customers, n_bootstraps)
            Purchasing probability for each customer.
        """
        assert prices.shape[0] == customers.shape[0], f"Number of prices should be equal to the number of customers."
        assert prices.ndim == 1, f"Prices should be a 1D array."

        # endogenous variables
        variables = np.concatenate([customers, prices.reshape(-1, 1), customers * prices.reshape(-1, 1)], axis=1)

        # compute purchasing probability
        purchase_proba = expit(self.coef_ @ variables.T).T  # shape = (n_customers, n_bootstraps)

        return purchase_proba
    
    def predict_choice(self, customers: np.ndarray, prices: np.ndarray) -> np.ndarray:
        """
        Returns:
        --------
        choice: np.ndarray, shape (n_customers, n_bootstraps)
            Purchasing choice for each customer.
        """
        return (self.predict_proba(customers, prices) > 0.5).astype(int)
    
    def evaluate(self, customers: np.ndarray, prices: np.ndarray, dgp: dgp.Pricing) -> float:
        prob_pred = self.predict_proba(customers, prices)  # shape = (n_customers, n_bootstraps)
        prob_true = dgp.predict_proba(customers, prices)[:, 1].reshape(-1, 1)  # shape = (n_customers, 1)

        choice_pred = (prob_pred > 0.5).astype(int)
        choice_true = (prob_true > 0.5).astype(int)

        return {
            'accuracy': (choice_pred == choice_true).mean(axis=0),  # shape = (n_bootstraps,)
            'rmse': np.sqrt(((prob_pred - prob_true)**2).mean(axis=0))  # shape = (n_bootstraps,)
        }


demand_model_dict = {
    'mle': MLELogit, 'lasso': LassoLogit, 'wlb': WLBLassoLogit
}


def obj_func_with_purchase_proba(
    prices: np.ndarray, purchase_proba: np.ndarray
) -> float:
    """
    Objective function to maximize. 
    
    Params:
    -------
    prices: np.ndarray, shape (n_customers,)
        Prices.

    purchase_proba: np.ndarray, shape = (n_customers, 2) or (n_customers, n_bootstraps)

    Returns:
    --------
    obj_val: float
        Objective value.
    """
    if purchase_proba.shape[1] == 2:
        purchase_proba = purchase_proba[:, 1]
        return np.dot(prices, purchase_proba) / prices.shape[0]
    else:  # prob_arr has shape (n_customers, n_bootstraps)
        return np.mean(purchase_proba.T @ prices / prices.shape[0])



def obj_func(
    customers: np.ndarray, prices: np.ndarray, demand_model: DemandModel, 
) -> float:
    """
    Objective function to maximize. 
    
    Params:
    -------
    customers: np.ndarray, shape (n_customers, n_features)
        Customer features.

    prices: np.ndarray, shape (n_customers,)
        Prices.

    demand_model: DemandModel
        Demand model.

    Returns:
    --------
    obj_val: float
        Objective value.
    """
    prob_arr = demand_model.predict_proba(customers=customers, prices=prices)

    return obj_func_with_purchase_proba(
        prices=prices, purchase_proba=prob_arr
    )
    

def optimize(
    customers: np.ndarray, purchase_proba: np.ndarray
) -> tuple: 
    """ 
    Optimize prices for each customer given demand model

    Params:
    -------
    customers: np.ndarray, shape (n_customers, n_features)
        Customer features.

    purchase_proba: np.ndarray, shape = (n_customers, 2) or (n_customers, n_bootstraps)

    Returns:
    --------
    opt_prices: np.ndarray, shape (n_customers,)
        Optimal prices for each customer.

    opt_obj_val: np.ndarray, shape (n_customers,)
        Average value of the objective function.
    """
    n_customers = customers.shape[0]
    opt_result_list = [None] * n_customers

    for cust_id in range(n_customers):
        opt_result_list[cust_id] = minimize_scalar(
            fun=lambda x: -obj_func(customers[cust_id].reshape(1, -1), np.array([x]), purchase_proba), 
            bounds=(0, 10)
        )

    opt_prices = np.array([result.x for result in opt_result_list if result.success])
    opt_obj_val = np.mean([-result.fun for result in opt_result_list if result.success])

    return opt_prices, opt_obj_val


def optimize(
    customers: np.ndarray, demand_model: DemandModel, 
) -> tuple:
    """ 
    Optimize prices for each customer given demand model

    Params:
    -------
    customers: np.ndarray, shape (n_customers, n_features)
        Customer features.

    demand_model: DemandModel
        Demand model.

    Returns:
    --------
    opt_prices: np.ndarray, shape (n_customers,)
        Optimal prices for each customer.

    opt_obj_val: np.ndarray, shape (n_customers,)
        Average value of the objective function.
    """
    n_customers = customers.shape[0]
    opt_result_list = [None] * n_customers

    for cust_id in range(n_customers):
        opt_result_list[cust_id] = minimize_scalar(
            fun=lambda x: -obj_func(customers[cust_id].reshape(1, -1), np.array([x]), demand_model), 
            bounds=(0, 10)
        )

    opt_prices = np.array([result.x for result in opt_result_list if result.success])
    opt_obj_val = np.mean([-result.fun for result in opt_result_list if result.success])

    return opt_prices, opt_obj_val

