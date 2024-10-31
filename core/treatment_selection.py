import os 
import sys 
import pickle 
sys.path.insert(0, os.path.abspath('.'))
from tqdm import tqdm 
from joblib import Parallel, delayed
from typing import Iterable
from datetime import datetime

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS

from core.dgp import SingleSegmentWithoutControl


def ols(treatments: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """ 
    Estimate the treatment effect using OLS

    Params:
    -------
    treatments: np.ndarray
        Array of treatment values
    outcomes: np.ndarray
        Array of outcome values

    Returns:
    --------
    estimated treatment effect: np.ndarray
    """
    n_unique_treatments = len(np.unique(treatments))

    # change treatment array to onehot
    exog_arr = np.eye(n_unique_treatments)[treatments]

    # estimate the treatment effect using OLS
    ols_ = OLS(endog=outcomes, exog=exog_arr).fit()

    return ols_.params


def optimize(te_arr: np.ndarray) -> tuple:
    """ 
    Solve the optimization problem: choose the best treatment to maximize profit

    Params:
    -------
    te_arr: np.ndarray
        Array of treatment effects

    Returns:
    --------
    opt_decision: int
        Optimal treatment decision
    opt_val: float
        Optimal value
    """
    opt_decision = np.argmax(te_arr)
    opt_val = te_arr[opt_decision]

    return opt_decision, opt_val


def obj_func(
    targ_decision: int, te_arr: np.ndarray
) -> float:
    """ 
    Objective function: profit

    Params:
    -------
    targ_decision: int
        Decision
    te_arr: np.ndarray
        Array of treatment effects

    Returns:
    --------
    profit: float
    """
    return te_arr[targ_decision]

def repeated_experiments(
    operations_params: dict, 
    dgp_params: dict, 
    data_params: dict, 
    experiment_params: dict, 
    estimators_dict: dict, 
    verbose: bool = False, 
    n_jobs: int = -1, 
) -> list:
    # extract attributes
    n_experiments = experiment_params['n_experiments']
    sample_size = data_params['sample_size']
    stats = experiment_params['stats']

    # create data generation process
    dgp = SingleSegmentWithoutControl(**dgp_params)

    def run_single_experiment(experiment_id: int) -> dict:
        # generate data
        treatment_arr, outcome_arr = dgp.sample(sample_size=sample_size, seed=experiment_id)

        # fit potential outcome model
        emp_te = ols(treatments=treatment_arr, outcomes=outcome_arr)
        
        # solve plugin optimization
        plugin_decision, plugin_val_est = optimize(
            te_arr=emp_te, **operations_params
        )

        # solve clairvoyant optimization
        _, clairvoyant_val = optimize(te_arr=dgp.te_arr, **operations_params
        )

        # calculate actual targeting value of the plugin targeting policy
        true_plugin_val = obj_func(
            targ_decision=plugin_decision, te_arr=dgp.te_arr, 
            **operations_params
        )
        
        # bookkeeping
        result = {
            'plugin_decision': plugin_decision, 
            'clairvoyant_val': clairvoyant_val,
            'plugin_val_est': plugin_val_est, 
            'true_plugin_val': true_plugin_val, 
            'plugin_wc': plugin_val_est - true_plugin_val, 
            'plugin_wc_pct': (plugin_val_est - true_plugin_val) / true_plugin_val,
        }

        for name in estimators_dict.keys():
            temp_val_est = estimators_dict[name]['estimator'](
                treatments=treatment_arr, 
                outcomes=outcome_arr,
                stats=stats,
                **operations_params, 
                **estimators_dict[name]['params']
            )
            result.update({
                f'{name}_val_est': temp_val_est, 
                f'{name}_wc': temp_val_est - true_plugin_val, 
                f'{name}_wc_pct': (temp_val_est - true_plugin_val) / true_plugin_val,
            })
        
        return result
    
    if verbose:
        print(f'Running {n_experiments} experiments...')
    result_list = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(run_single_experiment)(experiment_id) 
        for experiment_id in range(n_experiments)
    )
    
    return [result for result in result_list if result is not None]


def stratified_bootstrap(
    treatments: np.ndarray, outcomes: np.ndarray, boot_sample_size: int 
) -> tuple:
    """
    Perform stratified bootstrap sampling.

    Params:
    -------
    treatments: np.ndarray
        The treatment values.
    outcomes: np.ndarray
        The outcomes.
    boot_sample_size: int
        The size of the bootstrap sample

    Returns:
    --------
    (treatment_sample, outcome_sample): tuple
        - treatment_sample: np.ndarray
            The sampled treatment values.
        - outcome_sample: np.ndarray
            The sampled outcomes.
    """
    # placeholder for the sampled data
    treatment_sample = []
    outcome_sample = []

    # stratified sampling
    for treatment_val in np.unique(treatments):
        # select the indices of the treatment values
        treatment_indices = np.where(treatments == treatment_val)[0]

        # calculate sample size, porportionate to the original data
        sample_size = int(treatment_indices.shape[0] / treatments.shape[0] * boot_sample_size)

        # sample the indices
        sample_indices = np.random.choice(
            treatment_indices, sample_size, replace=True
        )
        
        # append the sampled data
        treatment_sample.extend(treatments[sample_indices])
        outcome_sample.extend(outcomes[sample_indices])

    treatment_sample = np.array(treatment_sample)
    outcome_sample = np.array(outcome_sample)

    # check if the sample size is correct
    if treatment_sample.shape[0] < boot_sample_size:
        remainder = boot_sample_size - treatment_sample.shape[0]
        sample_indices = np.random.choice(
            np.arange(treatments.shape[0]), remainder, replace=True
        )
        
        treatment_sample = np.concatenate([treatment_sample, treatments[sample_indices]])
        outcome_sample = np.concatenate([outcome_sample, outcomes[sample_indices]])

    return treatment_sample, outcome_sample


def get_wc_boot_dstn(
    treatments: np.ndarray, outcomes: np.ndarray,
    n_bootstraps: int = 500,
    **kwargs, 
) -> np.ndarray:
    # initialize array for the bootstrap distribution of winner's curse
    boot_wc_dstn_arr = np.zeros(shape=(n_bootstraps, ))  # shape = (n_bootstraps)
    boot_decision_arr = np.zeros(shape=(n_bootstraps, ), dtype=bool)  # shape = (n_bootstraps)

    # get treatment effect estimate using all data (empirical estimate)
    emp_te_arr = ols(treatments=treatments, outcomes=outcomes)
    
    # create bootstrap distribution
    for boot_id in range(n_bootstraps):
        # bootstrap data
        boot_treatment_arr, boot_outcome_arr = stratified_bootstrap(
            treatments=treatments, outcomes=outcomes, boot_sample_size=treatments.shape[0]
        )

        # estimate treatment effect using bootstrap data
        boot_te_arr = ols(treatments=boot_treatment_arr, outcomes=boot_outcome_arr)

        # optimize
        boot_decision, boot_val_est = optimize(te_arr=boot_te_arr)
        
        # evaluate bootstrap targeting policy using empirical CATE estimates
        emp_boot_val_est = obj_func(targ_decision=boot_decision, te_arr=emp_te_arr)
        
        boot_decision_arr[boot_id] = boot_decision
        boot_wc_dstn_arr[boot_id] = boot_val_est - emp_boot_val_est

    return boot_wc_dstn_arr


def get_wc_m_out_of_n_boot_dstn(
    treatments: np.ndarray, outcomes: np.ndarray,
    power: float = 0.95, n_bootstraps: int = 500,
    **kwargs, 
) -> np.ndarray:
    # parameter checks
    assert 0 < power < 1, 'Power should be between 0 and 1.' 

    # attributes
    sample_size = treatments.shape[0]
    m = int(sample_size ** power)

    # initialize array for the bootstrap distribution of winner's curse
    boot_wc_dstn_arr = np.zeros(shape=(n_bootstraps, ))  # shape = (n_bootstraps)
    boot_decision_arr = np.zeros(shape=(n_bootstraps, ), dtype=bool)  # shape = (n_bootstraps)

    # get treatment effect estimate using all data (empirical estimate)
    emp_te_arr = ols(treatments=treatments, outcomes=outcomes)
    
    # create bootstrap distribution
    for boot_id in range(n_bootstraps):
        # bootstrap data
        boot_treatment_arr, boot_outcome_arr = stratified_bootstrap(
            treatments=treatments, outcomes=outcomes, boot_sample_size=m
        )

        # estimate treatment effect using bootstrap data
        boot_te_arr = ols(treatments=boot_treatment_arr, outcomes=boot_outcome_arr)

        # optimize
        boot_decision, boot_val_est = optimize(te_arr=boot_te_arr)
        
        # evaluate bootstrap targeting policy using empirical CATE estimates
        emp_boot_val_est = obj_func(targ_decision=boot_decision, te_arr=emp_te_arr)
        
        boot_decision_arr[boot_id] = boot_decision
        boot_wc_dstn_arr[boot_id] = boot_val_est - emp_boot_val_est

    return boot_wc_dstn_arr


def get_wc_num_boot_dstn(
    treatments: np.ndarray, outcomes: np.ndarray,
    power: float = -0.45, n_bootstraps: int = 500,
    **kwargs, 
) -> np.ndarray:
    # parameter checks
    assert 0 > power > -0.5, "Power must be between 0 and -0.5"

    # attributes
    sample_size = treatments.shape[0]
    epsilon_n = sample_size ** power

    # initialize array for the bootstrap distribution of winner's curse
    boot_wc_dstn_arr = np.zeros(shape=(n_bootstraps, ))  # shape = (n_bootstraps)
    boot_decision_arr = np.zeros(shape=(n_bootstraps, ), dtype=bool)  # shape = (n_bootstraps)

    # get treatment effect estimate using all data (empirical estimate)
    emp_te_arr = ols(treatments=treatments, outcomes=outcomes)
    
    # create bootstrap distribution
    for boot_id in range(n_bootstraps):
        # bootstrap data
        boot_treatment_arr, boot_outcome_arr = stratified_bootstrap(
            treatments=treatments, outcomes=outcomes, boot_sample_size=sample_size
        )

        # estimate treatment effect using bootstrap data
        boot_te_arr = ols(treatments=boot_treatment_arr, outcomes=boot_outcome_arr)

        # optimize
        boot_decision, _ = optimize(te_arr=boot_te_arr)

        # calcualte error
        norm_err_arr = np.sqrt(sample_size) * (boot_te_arr - emp_te_arr)

        # calculate perturbed treatment effect
        pert_te_arr = emp_te_arr + epsilon_n * norm_err_arr

        # evaluate boot_decision with perturbed treatment effect
        perturb_val_est = obj_func(targ_decision=boot_decision, te_arr=pert_te_arr)
        
        # evaluate bootstrap targeting policy using empirical CATE estimates
        emp_val_est = obj_func(targ_decision=boot_decision, te_arr=emp_te_arr)
        
        boot_decision_arr[boot_id] = boot_decision
        boot_wc_dstn_arr[boot_id] = perturb_val_est - emp_val_est

    return boot_wc_dstn_arr


def bootstrap_correction_estimate(
    treatments: np.ndarray, outcomes: np.ndarray,
    bootstrap_method: str = 'standard', stats: str = 'mean',
    **kwargs, 
) -> float:
    # fit the potential outcome model
    emp_te_arr = ols(treatments=treatments, outcomes=outcomes)

    # solve plugin problem
    _, plugin_val_est = optimize(te_arr=emp_te_arr)

    # dictionary of available bootstrap methods
    boot_methods_dict = {
        'standard': get_wc_boot_dstn, 
        'm_out_of_n': get_wc_m_out_of_n_boot_dstn, 
        'numerical': get_wc_num_boot_dstn, 
    }

    # get the bootstrap distribution of winner's curse
    boot_wc_dstn_arr = boot_methods_dict[bootstrap_method](
        treatments=treatments, outcomes=outcomes, **kwargs, 
    )

    if stats == 'mean':
        correction = boot_wc_dstn_arr.mean()
    elif stats == 'median':
        correction = np.median(boot_wc_dstn_arr)
    else:
        raise ValueError(f'Invalid stats: {stats}')

    return plugin_val_est - correction


def sample_size_test(
    sample_sizes: Iterable[int],
    operations_params: dict,
    dgp_params: dict,
    data_params: dict,
    experiment_params: dict,
    estimators_dict: dict,
    n_jobs: int = -1, verbose: bool = False,
    save_path: str = None, 
) -> dict:
    # placeholder for results
    result_dict = {sample_size: {} for sample_size in sample_sizes}

    # attributes 
    estimators_list = ['plugin'] + list(estimators_dict.keys())

    for sample_size in sample_sizes:
        if verbose:
            print(f'Running sample size {sample_size}...')
            start = datetime.now()
        
        # update sample size in data params
        data_params['sample_size'] = sample_size

        result_records = repeated_experiments(
            operations_params=operations_params, 
            dgp_params=dgp_params, 
            data_params=data_params, 
            experiment_params=experiment_params, 
            estimators_dict=estimators_dict, 
            n_jobs=n_jobs, verbose=verbose, 
        )

        # extract parameters
        for estimator in estimators_list:
            wc_arr = np.array([record[f'{estimator}_wc'] for record in result_records])
            wc_pct_arr = np.array([record[f'{estimator}_wc_pct'] for record in result_records])

            result_dict[sample_size].update({
                f'{estimator}_wc_mean': wc_arr.mean(),
                f'{estimator}_wc_se': wc_arr.std() / np.sqrt(sample_size),
                f'{estimator}_wc_pct_mean': wc_pct_arr.mean(),
                f'{estimator}_wc_pct_se': wc_pct_arr.std() / np.sqrt(sample_size),
            })

        if verbose:
            print(f'Finished sample size {sample_size} in {datetime.now() - start}')

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(result_dict, f)

    return result_dict
