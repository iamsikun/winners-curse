import os 
import sys 
sys.path.insert(0, os.path.abspath('.'))
from tqdm import tqdm 

import numpy as np
import pandas as pd

from statsmodels.regression.linear_model import OLS

from core.dgp import SingleSegment


treatment_space = np.array([0, 1])


class PotentialOutcomeModel(object):
    """ 
    Demand model for binary treatments and single segment built on potential outcomes. 
    """
    def __init__(self):        
        # placeholder
        self.ols = None

    def fit(self, data: pd.DataFrame):
        """ 
        Run OLS to estimate the treatment effect. The estimated treatment effect is 
        the same as difference-in-mean estimator (DiM). We use OLS because it's 
        faster than DiM. 

        * Need to add constants to the exogenous variables before running OLS!
        """
        # add constants to exogenous variables
        exdog = np.concatenate([
            data['treatment'].values.reshape(-1, 1), 
            np.ones_like(data['treatment']).reshape(-1, 1)
        ], axis=1)

        # run OLS
        self.ols = OLS(endog=data['outcome'], exog=exdog).fit()
        
        return self
    
    @property
    def te(self) -> float:
        assert self.ols is not None, 'Model not fitted yet'

        return self.ols.params['x1']
    
    @property
    def te_se(self) -> float:
        assert self.ols is not None, 'Model not fitted yet'

        return self.ols.bse['x1']

def optimize(
    te: float, price: float, cost: float
) -> tuple:
    """ 
    Solve the optimization problem: max(price * te - cost, 0)

    Params:
    -------
    te: float
        Treatment effect
    price: float
        Unit gain from customers
    cost: float
        Cost of applying treatments

    Returns:
    --------
    opt_targ_decision: bool
        Whether the optimal treatment is to target
    opt_targ_val_est: float
        Estimated value of targeting
    """
    opt_targ_decision = price * te > cost 

    opt_targ_val_est = opt_targ_decision * (price * te - cost)

    return opt_targ_decision, opt_targ_val_est 


def objective_func(
    targ_decision: bool, te: float, price: float, cost: float
) -> float:
    """ 
    Evaluate a targeting decision with the given treatment effect, price, and cost. 
    The objective function is (price * te - cost) if the decision is to target, and 0 otherwise.

    Params: 
    -------
    targ_decision: bool
        Whether to target
    te: float
        Treatment effect
    price: float
        Unit gain from customers
    cost: float
        Cost of applying treatments
    """
    return float(targ_decision * (price * te - cost))


def repeated_experiments(
    operations_params: dict, 
    dgp_params: dict, 
    data_params: dict, 
    experiment_params: dict, 
    estimators_dict: dict, 
    verbose: bool = False, 
) -> list:
    # extract attributes
    n_experiments = experiment_params['n_experiments']
    sample_size = data_params['sample_size']

    # create data generation process
    dgp = SingleSegment(**dgp_params)

    # initialize result
    result_list = [None] * n_experiments

    # if verbose is true, use tqdm for for loop
    iterator = tqdm(range(n_experiments)) if verbose else range(n_experiments)
    for experiment_id in iterator:
        # generate data
        data = dgp.sample(sample_size=sample_size, seed=experiment_id)

        # fit potential outcome model
        pom = PotentialOutcomeModel().fit(data)
        
        # solve plugin optimization
        plugin_decision, plugin_targ_val_est = optimize(
            te=pom.te, **operations_params
        )

        # calculate actual targeting value of the plugin targeting policy
        true_plugin_targ_val = objective_func(
            targ_decision=plugin_decision, te=dgp.te, 
            **operations_params
        )
        
        # bookkeeping
        result_list[experiment_id] = {
            'plugin_targ_val_est': plugin_targ_val_est, 
            'act_plugin_targ_val': true_plugin_targ_val, 
            'plugin_wc': plugin_targ_val_est - true_plugin_targ_val, 
        }

        for name in estimators_dict.keys():
            temp_targ_val_est = estimators_dict[name]['estimator'](
                data=data, 
                **operations_params, 
                **estimators_dict[name]['params']
            )

            result_list[experiment_id].update({
                f'{name}_target_val_est': temp_targ_val_est, 
                f'{name}_wc': temp_targ_val_est - true_plugin_targ_val
            })
    
    return result_list


def stratified_bootstrap(data: pd.DataFrame, boot_sample_size: int) -> pd.DataFrame:
    """ 
    Stratified bootstrap sampling for binary treatments. 

    Params:
    -------
    data: pd.DataFrame
        Data to sample from, contains 'treatment' and 'outcome' columns
    boot_sample_size: int
        Size of the bootstrapped sample
    """
    # Group by treatment only once
    treated_group = data[data['treatment'] == 1]
    control_group = data[data['treatment'] == 0]

    # Calculate the base group size and the remainder
    group_base_size = boot_sample_size // 2
    remainder = boot_sample_size % 2

    # Perform stratified bootstrap sampling
    treated_indices = np.random.choice(treated_group.index, group_base_size + remainder, replace=True)
    control_indices = np.random.choice(control_group.index, group_base_size, replace=True)

    # Return the bootstrapped sample using hstack for efficiency
    return data.loc[np.hstack((treated_indices, control_indices))]


def get_wc_boot_dstn(
    data: pd.DataFrame, 
    price: float, cost: float, 
    n_bootstraps: int = 500, **kwargs, 
) -> np.ndarray:
    # attributes
    sample_size = data.shape[0]

    # initialize array for the bootstrap distribution of winner's curse
    boot_wc_dstn_arr = np.zeros(shape=(n_bootstraps, ))  # shape = (n_bootstraps)

    # get treatment effect estimate using all data (empirical estimate)
    emp_pom = PotentialOutcomeModel().fit(data)
    
    # create bootstrap distribution
    for boot_id in range(n_bootstraps):
        # bootstrap data
        boot_data = stratified_bootstrap(data, boot_sample_size=sample_size)

        # estimate treatment effect using bootstrap data
        boot_pom = PotentialOutcomeModel().fit(boot_data)

        # optimize
        boot_decision, boot_targ_val_est = optimize(
            te=boot_pom.te, price=price, cost=cost
        )
        
        # evaluate bootstrap targeting policy using empirical CATE estimates
        emp_boot_targ_val_est = objective_func(
            targ_decision=boot_decision,  # decision to be evaluated: bootstrapped decision
            te=emp_pom.te,  # evaluation criterion: empirical treatment effect
            price=price, cost=cost
        )

        boot_wc_dstn_arr[boot_id] = boot_targ_val_est - emp_boot_targ_val_est

    return boot_wc_dstn_arr


def get_wc_m_out_of_n_boot_dstn(
    data: pd.DataFrame, 
    price: float, cost: float, 
    power: float = 0.95, n_bootstraps: int = 500, 
    **kwargs
) -> np.ndarray:
    # parameter checks
    assert 0 < power < 1, 'Power should be between 0 and 1.' 

    # attributes
    sample_size = data.shape[0]
    m = int(sample_size ** power)

    # initialize array for the bootstrap distribution of winner's curse
    boot_wc_dstn_arr = np.zeros(shape=(n_bootstraps, ))  # shape = (n_bootstraps)

    # get treatment effect estimate using all data (empirical estimate)
    emp_pom = PotentialOutcomeModel().fit(data)

    for boot_id in range(n_bootstraps):
        # bootstrap data
        boot_data = stratified_bootstrap(data, boot_sample_size=m)

        # estimate treatment effect using bootstrap data
        boot_pom = PotentialOutcomeModel().fit(boot_data)

        # optimize 
        boot_decision, boot_targ_val_est = optimize(
            te=boot_pom.te, price=price, cost=cost
        )
        
        # evaluate bootstrap targeting policy using empirical CATE estimates
        emp_boot_targ_val_est = objective_func(
            targ_decision=boot_decision,  # decision to be evaluated: bootstrapped decision
            te=emp_pom.te,  # evaluation criterion: empirical treatment effect
            price=price, cost=cost
        )

        boot_wc_dstn_arr[boot_id] = boot_targ_val_est - emp_boot_targ_val_est

    return boot_wc_dstn_arr


def get_wc_num_boot_dstn(
    data: pd.DataFrame,
    price: float, cost: float,
    n_bootstraps: int = 500, power: int = -0.45, 
    **kwargs,
) -> np.ndarray:
    # parameter checks
    assert 0 > power > -0.5, "Power must be between 0 and -0.5"

    # attributes
    sample_size = data.shape[0]

    # initialize array for the bootstrap distribution of winner's curse
    boot_tau_est_arr = np.zeros(shape=(n_bootstraps, ))  # shape = (n_bootstraps)

    # get treatment effect estimate using all data (empirical estimate)
    emp_pom = PotentialOutcomeModel().fit(data)

    for boot_id in range(n_bootstraps):
        # stratified bootstrap
        boot_data = stratified_bootstrap(data, boot_sample_size=sample_size)

        # estimate treatment effect with difference in mean estimator
        boot_pom = PotentialOutcomeModel().fit(boot_data)
        boot_tau_est_arr[boot_id] = boot_pom.te

    boot_tau_err_arr = np.sqrt(sample_size) * (boot_tau_est_arr - emp_pom.te)

    return price * (sample_size ** power) * boot_tau_err_arr * (price * boot_tau_est_arr > cost)


def bootstrap_correction_estimate(
    data: pd.DataFrame, 
    price: float, cost: float, 
    bootstrap_method: str = 'standard', **kwargs, 
) -> np.ndarray:
    # fit the potential outcome model
    pom = PotentialOutcomeModel().fit(data)

    # solve plugin problem
    _, plugin_target_val_est = optimize(
        te=pom.te, price=price, cost=cost
    )

    # dictionary of available bootstrap methods
    boot_methods_dict = {
        'standard': get_wc_boot_dstn, 
        'm_out_of_n': get_wc_m_out_of_n_boot_dstn, 
        'numerical': get_wc_num_boot_dstn, 
    }

    # get the bootstrap distribution of winner's curse
    boot_wc_dstn_arr = boot_methods_dict[bootstrap_method](
        data=data, 
        price=price, cost=cost, **kwargs, 
    )

    return plugin_target_val_est - boot_wc_dstn_arr.mean()


# def old_get_wc_m_out_of_n_boot_dstn(
#     treatments: np.ndarray, outcomes: np.ndarray, 
#     price: float, cost: float, 
#     q: float = 0.9, max_j: int = 20, min_m: int = 10, 
#     pow: float = 1 / 3, n_bootstraps: int = 500, 
#     m_method: str = 'Bickle-Sakov', 
#     **kwargs
# ) -> np.ndarray:
#     # attributes
#     sample_size = treatments.shape[0]

#     # initialize list of arrays for the bootstrap distribution of winner's curse
#     m_arr = np.array([int(q**j * sample_size) for j in range(max_j)])
#     m_arr = m_arr[m_arr >= min_m]
#     # list of numpy arrays with shape (n_bootstraps, )
#     m_boot_wc_dstn_list = [np.zeros(shape=(n_bootstraps, ))] * m_arr.shape[0]

#     # get treatment effect estimate using all data (empirical estimate)
#     emp_tau_est = difference_in_mean(treatments=treatments, outcomes=outcomes)
    
#     # create bootstrap distribution
#     if m_method == 'Bickle-Sakov':
#         for m_id, m in enumerate(m_arr):
#             boot_wc_dstn_arr = np.zeros(shape=(n_bootstraps, ))
#             for boot_id in range(n_bootstraps):
#                 boot_treatment_arr, boot_outcome_arr = stratified_bootstrap(
#                     treatments=treatments, outcomes=outcomes, boot_sample_size=m
#                 )

#                 boot_is_target, boot_target_val_est = plugin_optimize(
#                     treatments=boot_treatment_arr, outcomes=boot_outcome_arr, 
#                     price=price, cost=cost, 
#                 )
                
#                 # evaluate bootstrap targeting policy using empirical CATE estimates
#                 emp_target_val_est = obj_func(
#                     price=price, cost=cost, 
#                     tau=emp_tau_est, is_target=boot_is_target 
#                 )

#                 boot_wc_dstn_arr[boot_id] = boot_target_val_est - emp_target_val_est 

#             m_boot_wc_dstn_list[m_id] = boot_wc_dstn_arr

#         return choose_best_m(m_boot_wc_dstn_list)

#     elif m_method == 'power':
#         m = int(sample_size ** pow)
#         boot_wc_dstn_arr = np.zeros(shape=(n_bootstraps, ))
#         for boot_id in range(n_bootstraps):
#                 boot_treatment_arr, boot_outcome_arr = stratified_bootstrap(
#                     treatments=treatments, outcomes=outcomes, boot_sample_size=m
#                 )

#                 boot_is_target, boot_target_val_est = plugin_optimize(
#                     treatments=boot_treatment_arr, outcomes=boot_outcome_arr, 
#                     price=price, cost=cost, 
#                 )
                
#                 # evaluate bootstrap targeting policy using empirical CATE estimates
#                 emp_target_val_est = obj_func(
#                     price=price, cost=cost, 
#                     tau=emp_tau_est, is_target=boot_is_target 
#                 )

#                 boot_wc_dstn_arr[boot_id] = boot_target_val_est - emp_target_val_est 


#         return boot_wc_dstn_arr
#     else:
#         raise ValueError(f'Invalid m_method: {m_method}')