import os 
import sys 
sys.path.insert(0, os.path.abspath('.'))
import pickle 

from tqdm import tqdm 
from joblib import Parallel, delayed
from typing import Iterable
from datetime import datetime

import numpy as np
import pandas as pd


from core.dgp import SingleSegmentWithControl

def difference_in_mean(
        treatments: np.ndarray, outcomes: np.ndarray, 
        treatment_vals: Iterable, 
) -> np.ndarray:
    """ 
    Calculate lift for each treatment value compared to the control group 
    using the difference in mean.

    Params:
    -------
    treatments: np.ndarray
        The treatment values.
    outcomes: np.ndarray
        The outcomes.
    treatment_vals: Iterable
        All treatment levels

    Returns:
    --------
    np.ndarray
        The lift for each treatment value compared to the control group. 
    """
    assert treatment_vals[0] == 0, "The first treatment value should be the control group."

    # placeholder for the treatment effect estimates
    te_arr = np.zeros(treatment_vals.shape[0])

    # mask for control group
    control_mask = treatments == 0

    # iterate over the treatment values
    for idx, treatment_val in enumerate(treatment_vals):
        # mask for the treatment value
        treatment_mask = treatments == treatment_val
        
        # calculate the difference in mean
        te_arr[idx] = outcomes[treatment_mask].mean() - outcomes[control_mask].mean()

    return te_arr


def optimize(
    customer_tes: np.ndarray, treatment_space: np.ndarray,
    price: float, cost: float, 
) -> tuple:
    """
    Optimize the treatment assignment for a given price and cost.

    Params:
    -------
    customer_tes: np.ndarray, shape = (n_treatments - 1, )
        Lift under each treatment level except control
    treatment_space: np.ndarray, shape = (n_treatments, )
        The treatment space.
    price: float
        The price.
    cost: float
        The cost.

    Returns:
    --------
    (opt_decision, opt_obj): tuple
        - opt_decision: int
            The optimal treatment decision.
        - opt_obj: float
            The optimal objective value.
    """
    assert treatment_space[0] == 0, "The first treatment value should be the control group."    
    
    # calculate the profit lift for each customer and each treatment
    profit_lift_arr = customer_tes * price - treatment_space * cost 

    # optimize
    opt_decision = treatment_space[profit_lift_arr.argmax()]
    opt_obj = profit_lift_arr.max().sum()

    return opt_decision, opt_obj


def obj_func(
    customer_tes: np.ndarray, targ_decision: int, 
    price: float, cost: float, 
) -> float:
    """
    Objective function for the optimization problem.

    Params:
    -------
    customer_tes: np.ndarray, shape = (n_treatments - 1, )
        Lift under each treatment level except control
    targ_decision: int
        The treatment decision.
    price: float
        The price.
    cost: float
        The cost.

    Returns:
    --------
    float
        The objective value.
    """
    return customer_tes[targ_decision] * price - targ_decision * cost


def repeated_experiments(
    operations_params: dict, dgp_params: dict, data_params: dict, 
    experiment_params: dict, estimators_dict: dict, 
    n_jobs: int = -1, verbose: bool = False, 
) -> list:
    # extract parameters
    n_experiments = experiment_params['n_experiments']
    sample_size = data_params['sample_size']
    conditional_on_treated = experiment_params['conditional_on_treated']
    stats = experiment_params['stats']

    # create data generation process
    dgp = SingleSegmentWithControl(**dgp_params)

    def run_single_experiment(experiment_id: int) -> dict:
        # sample data
        treatment_arr, outcome_arr = dgp.sample(sample_size=sample_size, seed=experiment_id)

        # estimate the treatment effects
        emp_te_arr = difference_in_mean(
            treatments=treatment_arr, outcomes=outcome_arr,
            treatment_vals=dgp_params['treatment_space']
        )  # shape = (n_treatments, )

        # solve plugin optimization
        plugin_decision, plugin_val_est = optimize(
            customer_tes=emp_te_arr, 
            treatment_space=dgp_params['treatment_space'],
            **operations_params
        )

        # solve clairvoyant optimization
        _, clairvoyant_val = optimize(
            customer_tes=dgp.lift_arr, 
            treatment_space=dgp_params['treatment_space'],
            **operations_params
        )

        if conditional_on_treated and plugin_decision == 0:
            return None  # if conditional_on_treated, we do not consider the control group

        # calculate the true plugin value
        true_plugin_val = obj_func(
            customer_tes=dgp.lift_arr,  # true treatment effects, shape = (n_treatments, )
            targ_decision=plugin_decision,
            **operations_params
        )

        # bookkeeping
        # don't need to check for zero since we already exclude the case where targ_decision = 0
        plugin_wc_pct = (plugin_val_est - true_plugin_val) / true_plugin_val  
        total_cost = operations_params['cost'] * plugin_decision
        result_dict = {
            'clairvoyant_val': clairvoyant_val,
            'plugin_val_est': plugin_val_est,
            'true_plugin_val': true_plugin_val,
            'plugin_wc': plugin_val_est - true_plugin_val,
            'plugin_wc_pct': plugin_wc_pct,
            'plugin_roi_wc': (plugin_val_est - true_plugin_val) / total_cost
        }

        # run all remaining estimators:
        for name in estimators_dict.keys():
            targ_val_est = estimators_dict[name]['estimator'](
                treatments=treatment_arr, outcomes=outcome_arr,
                treatment_space=dgp_params['treatment_space'],
                **operations_params, 
                **estimators_dict[name]['params']
            )

            result_dict.update({
                f'{name}_val_est': targ_val_est,
                f'{name}_wc': targ_val_est - true_plugin_val,
                f'{name}_wc_pct': (targ_val_est - true_plugin_val) / true_plugin_val,
                f'{name}_roi_wc': (targ_val_est - true_plugin_val) / total_cost
            })

        return result_dict

    result_list = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(run_single_experiment)(i) for i in range(n_experiments))

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
    treatment_space: np.ndarray, price: float, cost: float,
    n_bootstraps: int = 1000, conditional_on_treated: bool = False,
    n_jobs: int = -1, verbose: bool = False, 
) -> np.ndarray:
    # initialize placeholders for the bootstrap distribution of winner's curse
    boot_wc_dstn_arr = np.zeros(shape=(n_bootstraps, ))  # shape = (n_bootstraps)
    boot_decision_arr = np.zeros(shape=(n_bootstraps, ))  # shape = (n_bootstraps)

    # get treatment efffect estimate using all data (empirical estimate)
    emp_te_arr = difference_in_mean(
        treatments=treatments, outcomes=outcomes,
        treatment_vals=treatment_space
    )  # shape = (n_treatments, )

    def fig_single_bootstrap():
        # bootstrap data
        boot_treatment_arr, boot_outcome_arr = stratified_bootstrap(
            treatments=treatments, outcomes=outcomes, boot_sample_size=treatments.shape[0]
        )

        # estimate the treatment effects
        boot_te_arr = difference_in_mean(
            treatments=boot_treatment_arr, outcomes=boot_outcome_arr,
            treatment_vals=treatment_space
        )  # shape = (n_treatments, )

        # solve plugin optimization
        boot_decision, boot_val_est = optimize(
            customer_tes=boot_te_arr, 
            treatment_space=treatment_space,
            price=price, cost=cost
        )

        # evaluate bootstrap targeting policy using empirical treatment effect estiamtes
        emp_boot_val_est = obj_func(
            customer_tes=emp_te_arr, 
            targ_decision=boot_decision,
            price=price, cost=cost
        )

        return boot_decision, boot_val_est - emp_boot_val_est
    
    # run bootstrap
    boot_results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(fig_single_bootstrap)() for _ in range(n_bootstraps)
    )

    for i, (boot_decision, boot_wc) in enumerate(boot_results):
        boot_decision_arr[i] = boot_decision
        boot_wc_dstn_arr[i] = boot_wc

    if conditional_on_treated:
        return boot_wc_dstn_arr[boot_decision_arr != 0]
    else:
        return boot_wc_dstn_arr


def get_wc_m_out_of_n_boot_dstn(
    treatments: np.ndarray, outcomes: np.ndarray,
    treatment_space: np.ndarray, price: float, cost: float,
    n_bootstraps: int = 1000, power: float = 0.9, 
    conditional_on_treated: bool = False,
    n_jobs: int = -1, verbose: bool = False, 
) -> np.ndarray:
    # parameter checks
    assert 0 < power < 1, 'Power should be between 0 and 1.' 

    # initialize placeholders for the bootstrap distribution of winner's curse
    boot_wc_dstn_arr = np.zeros(shape=(n_bootstraps, ))  # shape = (n_bootstraps)
    boot_decision_arr = np.zeros(shape=(n_bootstraps, ), dtype=bool)  # shape = (n_bootstraps)
    # attributes
    m = int(treatments.shape[0] ** power)

    # get treatment efffect estimate using all data (empirical estimate)
    emp_te_arr = difference_in_mean(
        treatments=treatments, outcomes=outcomes,
        treatment_vals=treatment_space
    )  # shape = (n_treatments, )

    def fig_single_bootstrap():
        # bootstrap data
        boot_treatment_arr, boot_outcome_arr = stratified_bootstrap(
            treatments=treatments, outcomes=outcomes, boot_sample_size=m
        )

        # estimate the treatment effects
        boot_te_arr = difference_in_mean(
            treatments=boot_treatment_arr, outcomes=boot_outcome_arr,
            treatment_vals=treatment_space
        )  # shape = (n_treatments, )

        # solve plugin optimization
        boot_decision, boot_val_est = optimize(
            customer_tes=boot_te_arr, 
            treatment_space=treatment_space,
            price=price, cost=cost
        )

        # evaluate bootstrap targeting policy using empirical treatment effect estiamtes
        emp_boot_val_est = obj_func(
            customer_tes=emp_te_arr, 
            targ_decision=boot_decision,
            price=price, cost=cost
        )

        return boot_decision, boot_val_est - emp_boot_val_est
    
    # run bootstrap
    boot_results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(fig_single_bootstrap)() for _ in range(n_bootstraps)
    )

    for i, (boot_decision, boot_wc) in enumerate(boot_results):
        boot_decision_arr[i] = boot_decision
        boot_wc_dstn_arr[i] = boot_wc

    if conditional_on_treated:
        return boot_wc_dstn_arr[boot_decision_arr != 0]
    else:
        return boot_wc_dstn_arr



def get_wc_num_boot_dstn(
    treatments: np.ndarray, outcomes: np.ndarray,
    treatment_space: np.ndarray, price: float, cost: float,
    n_bootstraps: int = 1000, power: float = -0.45, 
    conditional_on_treated: bool = False,
    n_jobs: int = -1, verbose: bool = False, 
) -> np.ndarray:
    # parameter checks
    assert 0 > power > -0.5, "Power must be between 0 and -0.5"

    # attributes
    sample_size = treatments.shape[0]
    epsilon_n = sample_size ** power

    # initialize placeholders for the bootstrap distribution of winner's curse
    boot_wc_dstn_arr = np.zeros(shape=(n_bootstraps, ))  # shape = (n_bootstraps)
    boot_decision_arr = np.zeros(shape=(n_bootstraps, ), dtype=bool)  # shape = (n_bootstraps)
    # get treatment efffect estimate using all data (empirical estimate)
    emp_te_arr = difference_in_mean(
        treatments=treatments, outcomes=outcomes,
        treatment_vals=treatment_space
    )  # shape = (n_treatments, )

    def fig_single_bootstrap():
        # bootstrap data
        boot_treatment_arr, boot_outcome_arr = stratified_bootstrap(
            treatments=treatments, outcomes=outcomes, boot_sample_size=sample_size
        )

        # estimate the treatment effects
        boot_te_arr = difference_in_mean(
            treatments=boot_treatment_arr, outcomes=boot_outcome_arr,
            treatment_vals=treatment_space
        )  # shape = (n_treatments, )

        # optimize
        boot_decision, _ = optimize(
            customer_tes=boot_te_arr, 
            treatment_space=treatment_space,
            price=price, cost=cost
        )

        # error
        norm_error = np.sqrt(sample_size) * (boot_te_arr - emp_te_arr)

        # calculate perturbation
        perturb_te_arr = emp_te_arr + epsilon_n * norm_error

        # evaluate bootstrap policy using perturbed treatment effect estimates
        perturb_val_est = obj_func(
            customer_tes=perturb_te_arr, 
            targ_decision=boot_decision,
            price=price, cost=cost
        )

        # evaluate bootstrap policy using empirical treatment effect estiamtes
        emp_boot_val_est = obj_func(
            customer_tes=emp_te_arr, 
            targ_decision=boot_decision,
            price=price, cost=cost
        )

        return boot_decision, perturb_val_est - emp_boot_val_est
    
    # run bootstrap
    boot_results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(fig_single_bootstrap)() for _ in range(n_bootstraps)
    )

    for i, (boot_decision, boot_wc) in enumerate(boot_results):
        boot_decision_arr[i] = boot_decision
        boot_wc_dstn_arr[i] = boot_wc

    if conditional_on_treated:
        return boot_wc_dstn_arr[boot_decision_arr != 0]
    else:
        return boot_wc_dstn_arr


def bootstrap_correction_estimate(
    treatments: np.ndarray, outcomes: np.ndarray,
    treatment_space: np.ndarray, price: float, cost: float,
    bootstrap_method: str = 'standard', conditional_on_treated: bool = False,
    stats: str = 'mean',
    n_jobs: int = -1, verbose: bool = False, **kwargs
) -> float:
    # estiamte empirical treatment effects
    emp_te_arr = difference_in_mean(
        treatments=treatments, outcomes=outcomes,
        treatment_vals=treatment_space
    )  # shape = (n_treatments, )

    # get the empirical targeting policy
    plugin_decision, plugin_val_est = optimize(
        customer_tes=emp_te_arr, 
        treatment_space=treatment_space,
        price=price, cost=cost
    )

    # dictionary of available bootstrap methods
    boot_methods_dict = {
        'standard': get_wc_boot_dstn, 
        'm_out_of_n': get_wc_m_out_of_n_boot_dstn, 
        'numerical': get_wc_num_boot_dstn, 
    }

    # get the bootstrap distribution of winner's curse
    boot_wc_dstn_arr = boot_methods_dict[bootstrap_method](
        treatments=treatments, outcomes=outcomes,
        treatment_space=treatment_space,
        price=price, cost=cost,
        conditional_on_treated=conditional_on_treated,
        n_jobs=n_jobs, verbose=verbose, **kwargs
    )

    if stats == 'mean':
        return plugin_val_est - np.nanmean(boot_wc_dstn_arr)
    elif stats == 'median':
        return plugin_val_est - np.nanmedian(boot_wc_dstn_arr)
    else:
        raise ValueError(f'Invalid stats: {stats}')


def sample_size_test(
    sample_sizes: Iterable[int],
    operations_params: dict,
    dgp_params: dict,
    data_params: dict,
    experiment_params: dict,
    estimators_dict: dict,
    n_jobs: int = -1, verbose: bool = False, 
    save_path: str = None
) -> dict:
    # placeholder for results
    result_dict = {sample_size: {} for sample_size in sample_sizes}

    # attributes
    estimators_list = ['plugin'] + list(estimators_dict.keys())

    for idx, sample_size in enumerate(sample_sizes):
        if verbose:
            print(f'\nRunning sample size {sample_size} ({idx + 1}/{len(sample_sizes)})')
            start = datetime.now()
        
        # update sample size in data params
        data_params['sample_size'] = sample_size

        result_records = repeated_experiments(
            operations_params=operations_params, 
            dgp_params=dgp_params, 
            data_params=data_params, 
            experiment_params=experiment_params, 
            estimators_dict=estimators_dict, 
            n_jobs=n_jobs, verbose=verbose
        )

        # extract parameters
        for estimator in estimators_list:
            wc_arr = np.array([record[f'{estimator}_wc'] for record in result_records])
            wc_pct_arr = np.array([record[f'{estimator}_wc'] / record['clairvoyant_val'] for record in result_records])
            
            # remove nan and inf from wc_pct_arr
            wc_pct_arr = wc_pct_arr[~np.isnan(wc_pct_arr)]
            wc_pct_arr = wc_pct_arr[~np.isinf(wc_pct_arr)]

            result_dict[sample_size].update({
                f'{estimator}_wc_mean': wc_arr.mean(),
                f'{estimator}_wc_se': wc_arr.std() / np.sqrt(data_params['sample_size']),
                f'{estimator}_wc_pct_mean': wc_pct_arr.mean(),
                f'{estimator}_wc_pct_se': wc_pct_arr.std() / np.sqrt(data_params['sample_size'])
            })

        if verbose:
            print(f'Finished sample size {sample_size} in {datetime.now() - start}')

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(result_dict, f)

    return result_dict


def noise_level_test(
    noise_stds: Iterable[int],
    operations_params: dict,
    dgp_params: dict,
    data_params: dict,
    experiment_params: dict,
    estimators_dict: dict,
    n_jobs: int = -1, verbose: bool = False, 
    save_path: str = None
) -> dict:
    # placeholder for results
    result_dict = {noise_std: {} for noise_std in noise_stds}

    # attributes
    estimators_list = ['plugin'] + list(estimators_dict.keys())

    for idx, noise_std in enumerate(noise_stds):
        if verbose:
            print(f'\nRunning nnoise_level {noise_std} ({idx + 1}/{len(noise_stds)})')
            start = datetime.now()
        
        # update noise std in data params
        data_params['noise_std'] = noise_std

        result_records = repeated_experiments(
            operations_params=operations_params, 
            dgp_params=dgp_params, 
            data_params=data_params, 
            experiment_params=experiment_params, 
            estimators_dict=estimators_dict, 
            n_jobs=n_jobs, verbose=verbose
        )

        # extract parameters
        for estimator in estimators_list:
            wc_arr = np.array([record[f'{estimator}_wc'] for record in result_records])
            wc_pct_arr = np.array([record[f'{estimator}_wc'] / record['clairvoyant_val'] for record in result_records])
            
            # remove nan and inf from wc_pct_arr
            wc_pct_arr = wc_pct_arr[~np.isnan(wc_pct_arr)]
            wc_pct_arr = wc_pct_arr[~np.isinf(wc_pct_arr)]

            result_dict[noise_std].update({
                f'{estimator}_wc_mean': wc_arr.mean(),
                f'{estimator}_wc_se': wc_arr.std() / np.sqrt(data_params['sample_size']),
                f'{estimator}_wc_pct_mean': wc_pct_arr.mean(),
                f'{estimator}_wc_pct_se': wc_pct_arr.std() / np.sqrt(data_params['sample_size'])
            })

        if verbose:
            print(f'Finished noise_level {noise_std} in {datetime.now() - start}')

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(result_dict, f)

    return result_dict
