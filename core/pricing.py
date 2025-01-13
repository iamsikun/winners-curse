import os 
import sys 
sys.path.insert(0, os.path.abspath('.'))
from tqdm import tqdm
from joblib import Parallel, delayed

import pandas as pd 
import numpy as np

from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar

from core.dgp import Pricing

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
    
    def evaluate(self, customers: np.ndarray, prices: np.ndarray, dgp: Pricing) -> float:
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

    def fit(
        self, data: pd.DataFrame, seed: int = None, verbose: bool = False, 
        n_jobs: int = -1,
    ):
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

        def fit_single_bootstrap():
            # calculate the weights 
            weights = np.random.exponential(scale=1, size=(sample_size,)) ** 1.8  # 1.8 is the prior from the paper
            boot_indices = np.random.choice(
                np.arange(sample_size), 
                size=sample_size, 
                replace=True, 
                p=weights / weights.sum()
            )

            return LogisticRegression(
                penalty=None, 
                solver='lbfgs',
                fit_intercept=False, 
                max_iter=1000
            ).fit(X=variables[boot_indices], y=outcomes[boot_indices])

        self.model_list = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(fit_single_bootstrap)() for _ in range(self.n_bootstraps)
        )
        self.coef_arr = np.array([model.coef_.flatten() for model in self.model_list])

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
    
    def evaluate(self, customers: np.ndarray, prices: np.ndarray, dgp: Pricing) -> float:
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

    def fit(
        self, data: pd.DataFrame, seed: int = None, verbose: bool = False, 
        n_jobs: int = -1, 
    ):
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

        def fit_single_model():
            # calculate the weights 
            weights = np.random.exponential(scale=1, size=(sample_size,)) ** 1.8  # 1.8 is the prior from the paper
            boot_indices = np.random.choice(
                np.arange(sample_size), 
                size=sample_size, 
                replace=True, 
                p=weights / weights.sum()
            )

            return LogisticRegression(
                penalty='l1',
                C=0.3, 
                solver='liblinear',
                fit_intercept=False,
                max_iter=1000
            ).fit(X=variables[boot_indices], y=outcomes[boot_indices])

        # fit model for each bootstrap
        self.model_list = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(fit_single_model)() for _ in range(self.n_bootstraps)
        )

        # extract coefficients
        self.coef_arr = np.array([model.coef_.flatten() for model in self.model_list])

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
    
    def evaluate(self, customers: np.ndarray, prices: np.ndarray, dgp: Pricing) -> float:
        prob_pred = self.predict_proba(customers, prices)  # shape = (n_customers, n_bootstraps)
        prob_true = dgp.predict_proba(customers, prices)[:, 1].reshape(-1, 1)  # shape = (n_customers, 1)

        choice_pred = (prob_pred > 0.5).astype(int)
        choice_true = (prob_true > 0.5).astype(int)

        return {
            'accuracy': (choice_pred == choice_true).mean(axis=0),  # shape = (n_bootstraps,)
            'rmse': np.sqrt(((prob_pred - prob_true)**2).mean(axis=0))  # shape = (n_bootstraps,)
        }


demand_model_dict = {
    'mle': MLELogit, 'lasso': LassoLogit, 
    'wlb-lasso': WLBLassoLogit, 'wlb-mle': WLBMLELogit, 
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
    

# def optimize(
#     customers: np.ndarray, purchase_proba: np.ndarray
# ) -> tuple: 
#     """ 
#     Optimize prices for each customer given demand model

#     Params:
#     -------
#     customers: np.ndarray, shape (n_customers, n_features)
#         Customer features.

#     purchase_proba: np.ndarray, shape = (n_customers, 2) or (n_customers, n_bootstraps)

#     Returns:
#     --------
#     opt_prices: np.ndarray, shape (n_customers,)
#         Optimal prices for each customer.

#     opt_obj_val: np.ndarray, shape (n_customers,)
#         Average value of the objective function.
#     """
#     n_customers = customers.shape[0]
#     opt_result_list = [None] * n_customers

#     for cust_id in range(n_customers):
#         opt_result_list[cust_id] = minimize_scalar(
#             fun=lambda x: -obj_func(customers[cust_id].reshape(1, -1), np.array([x]), purchase_proba), 
#             bounds=(0, 10)
#         )

#     opt_prices = np.array([result.x for result in opt_result_list if result.success])
#     opt_obj_val = np.mean([-result.fun for result in opt_result_list if result.success])

#     return opt_prices, opt_obj_val


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


def repeated_experiments(
    data_params: dict, 
    experiment_params: dict, 
    estimators_dict: dict, 
    verbose: bool = False, 
    n_jobs: int = -1,
) -> list:
    # extract parameters
    n_experiments = experiment_params['n_experiments']
    is_in_sample = experiment_params['in_sample']
    demand_model = experiment_params['demand_model']
    sample_size = data_params['sample_size']
    n_customers = data_params['n_customers']

    # create data generation process
    dgp = Pricing(seed=0)

    # placeholder for results
    result_list = [None] * n_experiments

    def run_single_experiment(experiment_id):
        # generate data
        data = dgp.sample(sample_size, seed=experiment_id)

        if is_in_sample:
            target_customers = data[[col for col in data.columns if col.startswith('cov_')]].values
        else:
            target_customers = dgp.sample_individuals(n_customers, seed=10 * experiment_id)

        # estimate demand model
        dm = demand_model_dict[demand_model]().fit(data)

        # optimize prices
        plugin_prices, pricing_val_est = optimize(
            customers=target_customers, demand_model=dm
        )
        
        # calculate the actual value of the plugin targeting policy
        true_pricing_val = obj_func(
            customers=target_customers, 
            prices=plugin_prices, 
            demand_model=dgp
        )

        # bookkeeping
        result = {
            'plugin_pricing_val_est': pricing_val_est, 
            'true_plugin_pricing_val': true_pricing_val,
            'plugin_wc': pricing_val_est - true_pricing_val, 
            'plugin_wc_pct': (pricing_val_est - true_pricing_val) / true_pricing_val
        }

        # run all remaining estimators
        for name in estimators_dict.keys():
            temp_targ_val_est = estimators_dict[name]['estimator'](
                data=data, 
                target_customers=target_customers, 
                demand_model=demand_model, 
                **estimators_dict[name]['params']
            )

            # bookkeeping 
            result.update({
                f'{name}_targ_val_est': temp_targ_val_est, 
                f'{name}_wc': temp_targ_val_est - true_pricing_val, 
                f'{name}_wc_pct': (temp_targ_val_est - true_pricing_val) / true_pricing_val, 
            })

        return result

    result_list = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(run_single_experiment)(i) for i in range(n_experiments)
    )

    return result_list


def get_wc_boot_dstn(
    data: pd.DataFrame, target_customers: np.ndarray,
    demand_model: str, n_bootstraps: int = 500, 
    verbose: bool = False, n_jobs: int = -1, **kwargs
) -> np.ndarray:
    # attributes
    sample_size = data.shape[0]

    # fit demand model with all data
    emp_dm = demand_model_dict[demand_model]().fit(data)

    def fit_single_bootstrap():
        # bootstrap data
        boot_data = data.sample(sample_size, replace=True)

        # estimate the treatment effect table from table
        boot_dm = demand_model_dict[demand_model]().fit(boot_data)

        # optimize
        boot_prices, boot_pricing_val_est = optimize(
            customers=target_customers, demand_model=boot_dm
        )

        # evaluate boot_targ_decision with empirical treatment effects
        emp_boot_pricing_val_est = obj_func(
            customers=target_customers, 
            prices=boot_prices, 
            demand_model=emp_dm
        )

        return boot_pricing_val_est - emp_boot_pricing_val_est

    # create bootstrap distribution
    # iterator = tqdm(range(n_bootstraps)) if verbose else range(n_bootstraps)
    boot_wc_dstn_arr = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(fit_single_bootstrap)() for _ in range(n_bootstraps)
    )
    boot_wc_dstn_arr = np.array(boot_wc_dstn_arr)

    return boot_wc_dstn_arr


def get_wc_m_out_of_n_boot_dstn(
    data: pd.DataFrame, target_customers: np.ndarray,
    demand_model: str, 
    n_bootstraps: int = 500, power: float = 0.90, 
    verbose: bool = False, seed: int = None, 
    n_jobs: int = -1, 
    **kwargs
) -> np.ndarray:
    # parameter checks
    assert 0 < power < 1, 'Power should be between 0 and 1.' 

    # set seed
    if seed is not None:
        np.random.seed(seed)

    # attributes
    sample_size = data.shape[0]
    m = int(sample_size ** power)  # bootstrap sample size

    # fit demand model with all data
    emp_dm = demand_model_dict[demand_model]().fit(data)

    def fit_single_bootstrap():
        # bootstrap data
        boot_data = data.sample(m, replace=True)

        # estimate the treatment effect table from table
        boot_dm = demand_model_dict[demand_model]().fit(boot_data)

        # optimize
        boot_prices, boot_pricing_val_est = optimize(
            customers=target_customers, demand_model=boot_dm
        )

        # evaluate boot_targ_decision with empirical treatment effects
        emp_boot_pricing_val_est = obj_func(
            customers=target_customers, 
            prices=boot_prices, 
            demand_model=emp_dm
        )

        return boot_pricing_val_est - emp_boot_pricing_val_est

    # create bootstrap distribution
    boot_wc_dstn_arr = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(fit_single_bootstrap)() for _ in  range(n_bootstraps)
    )
    boot_wc_dstn_arr = np.array(boot_wc_dstn_arr)

    return boot_wc_dstn_arr[~np.isnan(boot_wc_dstn_arr)]


def get_wc_num_boot_dstn(
    data: pd.DataFrame, target_customers: np.ndarray, 
    demand_model: str, 
    n_bootstraps: int = 500, power: float = -0.45, 
    verbose: bool = False, seed: int = None, n_jobs: int = -1, 
    **kwargs
):
    # parameter checks
    assert 0 > power > -0.5, "Power must be between 0 and -0.5"

    # set seed
    if seed is not None:
        np.random.seed(seed)

    # attributes
    sample_size = data.shape[0]
    epsilon_n = sample_size ** power  # epsilon_n

    # placeholder for the bootstrap distribution of winner's curse
    boot_wc_dstn_arr = np.zeros(shape=(n_bootstraps, ))

    # fit demand model with all data
    emp_dm = demand_model_dict[demand_model]().fit(data)

    # create bootstrap distribution
    def fit_single_bootstrap():
        # bootstrap data
        boot_data = data.sample(sample_size, replace=True)

        # estimate the treatment effect table from table
        boot_dm = demand_model_dict[demand_model]().fit(boot_data)

        # optimize
        boot_prices, _ = optimize(
            customers=target_customers, demand_model=boot_dm
        )

        # predict purchase probabilities under empirical model and bootstrapped model
        emp_purchase_prob = emp_dm.predict_proba(target_customers, boot_prices)
        boot_purchase_prob = boot_dm.predict_proba(target_customers, boot_prices)

        # error
        norm_error = np.sqrt(sample_size) * (boot_purchase_prob - emp_purchase_prob)

        # calculate perturbation
        perturb_purchase_prob = emp_purchase_prob + epsilon_n * norm_error

        # evaluate boot_price with perturbed purchase probabilities
        perturb_val_est = obj_func_with_purchase_proba(
            prices=boot_prices, purchase_proba=perturb_purchase_prob
        )

        # evaluate boot_price with empirical purchase probabilities
        emp_val_est = obj_func_with_purchase_proba(
            prices=boot_prices, purchase_proba=emp_purchase_prob
        )

        # calculate the winner's curse
        return perturb_val_est - emp_val_est

    boot_wc_dstn_arr = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(fit_single_bootstrap)() for _ in range(n_bootstraps)
    )
    boot_wc_dstn_arr = np.array(boot_wc_dstn_arr)

    return boot_wc_dstn_arr[~np.isnan(boot_wc_dstn_arr)]


def bootstrap_correction_estimate(
    data: pd.DataFrame, target_customers: np.ndarray,
    demand_model: str, 
    bootstrap_method: str = 'standard', n_bootstraps: int = 500, 
    verbose: bool = False,
    **kwargs
) -> np.ndarray:
    # estimation
    emp_dm = demand_model_dict[demand_model]().fit(data)

    # solve the plugin problem
    _, plugin_pricing_val_est = optimize(
        customers=target_customers, demand_model=emp_dm
    )

    # dictionary of available bootstrap methods
    boot_methods_dict = {
        'standard': get_wc_boot_dstn, 
        'm_out_of_n': get_wc_m_out_of_n_boot_dstn, 
        'numerical': get_wc_num_boot_dstn, 
    }

    # get the bootstrap distribution of winner' curse
    boot_wc_dstn_arr = boot_methods_dict[bootstrap_method](
        data=data, target_customers=target_customers, 
        demand_model=demand_model, 
        n_bootstraps=n_bootstraps, verbose=False, 
        **kwargs
    )

    return plugin_pricing_val_est - boot_wc_dstn_arr.mean()

