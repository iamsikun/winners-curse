import os 
import sys 
sys.path.insert(0, os.path.abspath('.'))
import pickle
from joblib import Parallel, delayed
from typing import Iterable
from datetime import datetime

import numpy as np
import pandas as pd

from core.dgp import MultipleSegments


def segment_difference_in_mean(
    segments: np.ndarray, treatments: np.ndarray, outcomes: np.ndarray
) -> np.ndarray:
    """ 
    Estimate treatment effect for each segment using difference in mean

    Params:
    -------
    segments: np.ndarray
        Array of segments. 
    treatments: np.ndarray
        Array of treatment assignments
    outcomes: np.ndarray
        Array of outcomes

    Returns:
    --------
    te_est_arr: np.ndarray
        Array of estimated treatment effects for each segment
    """
    unique_segments = np.unique(segments)
    te_est_arr = np.zeros(len(unique_segments))
    for i, segment in enumerate(unique_segments):
        mask = segments == segment
        te_est_arr[i] = np.mean(outcomes[mask][treatments[mask] == 1]) - np.mean(outcomes[mask][treatments[mask] == 0])
    return te_est_arr


def optimize(
    targ_customers: np.ndarray, te_arr: np.ndarray, 
    budget: int, 
): 
    """
    Optimize the targeting strategy for a given budget. 

    Params:
    -------
    targ_customers: np.ndarray, shape = (n_customers, )
        Customers to target, each customer is represented by their segment. 
    te_arr: np.ndarray, shape = (n_segments, )
        Treatment effects for each segment
    budget: int
        Number of customers to target.

    Returns:
    --------
    (opt_decision, opt_value): tuple
        Optimal decision and value of the objective function.
        - opt_decision: np.ndarray, shape = (n_customers, )
            Boolean array indicating whether to target each customer.
        - opt_value: float
            Value of the objective function.
    """
    # parameter check
    assert budget <= len(targ_customers), f'budget {budget} must be less than or equal to the number of customers {len(targ_customers)}'

    # assign treatment effects to each customer
    customer_te_arr = te_arr[targ_customers]  # shape = (n_customers, )

    # get the customers with the largest treatment effects
    opt_decision = np.zeros(len(targ_customers), dtype=bool)
    opt_decision[np.argsort(customer_te_arr)[-budget:]] = True

    # calculate the value of the objective function
    opt_value = np.sum(customer_te_arr[opt_decision])

    return opt_decision, opt_value

def obj_func(
    targ_customers: np.ndarray, te_arr: np.ndarray, targ_decision: np.ndarray, 
):
    """
    Objective function for the optimization problem. 

    Params:
    -------
    targ_customers: np.ndarray, shape = (n_customers, )
        Customers to target, each customer is represented by their segment. 
    te_arr: np.ndarray, shape = (n_segments, )
        Treatment effects for each segment
    targ_decision: np.ndarray, shape = (n_customers, )
        Boolean array indicating whether to target each customer.

    Returns:
    --------
    value: float
        Value of the objective function.
    """
    return np.sum(te_arr[targ_customers][targ_decision])


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

    # create data generation process
    dgp = MultipleSegments(**dgp_params)

    # initialize placeholder for results
    result_list = [None] * n_experiments

    def run_single_experiment(experiment_id: int) -> dict:
        # generate data
        segment_arr, treatment_arr, outcome_arr = dgp.sample(
            sample_size, seed=None
        )
        targ_customers_arr = dgp.sample_individuals(
            data_params['n_customers'], seed=None
        )

        # estimate treatment effects
        emp_te_arr = segment_difference_in_mean(segment_arr, treatment_arr, outcome_arr)

        # solve plugin optimization
        plugin_decision, plugin_val_est = optimize(
            targ_customers=targ_customers_arr, 
            te_arr=emp_te_arr, 
            **operations_params
        )

        # solve clairvoyant optimization
        _, clairvoyant_val = optimize(
            targ_customers=targ_customers_arr, 
            te_arr=dgp.segment_te_arr, 
            **operations_params
        )

        # calculate the true value of the plugin policy
        true_plugin_val = obj_func(
            targ_customers=targ_customers_arr, 
            te_arr=dgp.segment_te_arr,  # * evaluation uses true treatment effects
            targ_decision=plugin_decision
        )

        # bookkeeping
        result_dict = {
            'plugin_val_est': plugin_val_est,
            'true_plugin_val': true_plugin_val,
            'plugin_wc': plugin_val_est - true_plugin_val, 
            'clairvoyant_val': clairvoyant_val,
        }

        for name in estimators_dict.keys():
            targ_val_est = estimators_dict[name]['estimator'](
                segments=segment_arr,
                treatments=treatment_arr,
                outcomes=outcome_arr,
                targ_customers=targ_customers_arr,
                **operations_params, 
                **estimators_dict[name]['params']
            )
            result_dict.update({
                f'{name}_val_est': targ_val_est,
                f'{name}_wc': targ_val_est - true_plugin_val
            })        

        return result_dict
    
    # run experiments
    if verbose:
        print(f'Running {n_experiments} experiments...')

    result_list = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(run_single_experiment)(experiment_id) for experiment_id in range(n_experiments)
    )

    return result_list


def stratified_bootstrap(
    segments: np.ndarray, treatments: np.ndarray, outcomes: np.ndarray,
    boot_sample_size: int
) -> tuple:
    """ 
    Stratified bootstrap: sample with replacement from each segment and each 
    each treatment level 
    """
    unique_segment_arr = np.unique(segments)
    unique_treatment_arr = np.unique(treatments) 
    unique_combination_arr = np.array([
        (segment, treatment) for segment in unique_segment_arr for treatment in unique_treatment_arr
    ])

    # placeholders
    segment_sample = []
    treatment_sample = []
    outcome_sample = []
    count = 0  # count the bootstrap sample size
    # 
    for segment, treatment in unique_combination_arr:
        mask = (segments == segment) & (treatments == treatment)
        freq = mask.sum() / segments.shape[0]
        
        boot_sample_index = np.random.choice(
            np.where(mask)[0], 
            size=int(boot_sample_size * freq), 
            replace=True
        )

        count += boot_sample_index.shape[0]

        segment_sample.extend(segments[boot_sample_index])
        treatment_sample.extend(treatments[boot_sample_index])
        outcome_sample.extend(outcomes[boot_sample_index])

    # check if the sample size is correct
    if count < boot_sample_size:
        remainder = boot_sample_size - count
        boot_sample_index = np.random.choice(
            np.arange(segments.shape[0]), 
            size=remainder, 
            replace=True
        )

        segment_sample.extend(segments[boot_sample_index])
        treatment_sample.extend(treatments[boot_sample_index])
        outcome_sample.extend(outcomes[boot_sample_index])

    return np.array(segment_sample), np.array(treatment_sample), np.array(outcome_sample)


def get_wc_boot_dstn(
    segments: np.ndarray, treatments: np.ndarray, outcomes: np.ndarray,
    targ_customers: np.ndarray, 
    budget: int, n_bootstraps: int = 500, 
    n_jobs: int = -1, verbose: bool = False, **kwargs, 
):
    # attributes
    sample_size = segments.shape[0]

    # initialize placeholder for bootstrap distribution
    boot_wc_dstn_arr = np.zeros(shape=(n_bootstraps, ))  # shape = (n_bootstraps)

    # estimate treatment effects using all data
    emp_segment_te_arr = segment_difference_in_mean(
        segments, treatments, outcomes
    )  # shape = (n_segments, )

    # function for fitting a single bootstrap 
    def fit_bootstrap() -> float:
        # bootstrap sample
        boot_segment_arr, boot_treatment_arr, boot_outcome_arr = stratified_bootstrap(
            segments, treatments, outcomes, 
            boot_sample_size=sample_size
        )

        # estimate treatment effects
        boot_segment_te_arr = segment_difference_in_mean(
            boot_segment_arr, boot_treatment_arr, boot_outcome_arr
        )  # shape = (n_segments, )

        # solve plugin optimization
        boot_decision, boot_val_est = optimize(
            targ_customers=targ_customers, 
            te_arr=boot_segment_te_arr, 
            budget=budget
        )

        # evaluate bootstrap decision using empirical estimates
        emp_boot_val_est = obj_func(
            targ_customers=targ_customers, 
            te_arr=emp_segment_te_arr, 
            targ_decision=boot_decision
        )

        return boot_val_est - emp_boot_val_est
    
    # run bootstraps
    boot_wc_dstn_arr = np.array(Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(fit_bootstrap)() for _ in range(n_bootstraps)
    ))

    return boot_wc_dstn_arr


def get_wc_m_out_of_n_boot_dstn(
    segments: np.ndarray, treatments: np.ndarray, outcomes: np.ndarray,
    targ_customers: np.ndarray, 
    budget: int, n_bootstraps: int = 500, power: float = 0.9, 
    n_jobs: int = -1, verbose: bool = False, **kwargs, 
):
    # parameter checks
    assert 0 < power < 1, 'Power should be between 0 and 1.' 

    # attributes
    sample_size = segments.shape[0]
    m = int(sample_size ** power)

    # initialize placeholder for bootstrap distribution
    boot_wc_dstn_arr = np.zeros(shape=(n_bootstraps, ))  # shape = (n_bootstraps)

    # estimate treatment effects using all data
    emp_segment_te_arr = segment_difference_in_mean(
        segments, treatments, outcomes
    )  # shape = (n_segments, )

    # function for fitting a single bootstrap 
    def fit_bootstrap() -> float:
        # bootstrap sample
        boot_segment_arr, boot_treatment_arr, boot_outcome_arr = stratified_bootstrap(
            segments, treatments, outcomes, 
            boot_sample_size=m
        )

        # estimate treatment effects
        boot_segment_te_arr = segment_difference_in_mean(
            boot_segment_arr, boot_treatment_arr, boot_outcome_arr
        )  # shape = (n_segments, )

        # solve plugin optimization
        boot_decision, boot_val_est = optimize(
            targ_customers=targ_customers, 
            te_arr=boot_segment_te_arr, 
            budget=budget
        )

        # evaluate bootstrap decision using empirical estimates
        emp_boot_val_est = obj_func(
            targ_customers=targ_customers, 
            te_arr=emp_segment_te_arr, 
            targ_decision=boot_decision
        )

        return boot_val_est - emp_boot_val_est
    
    # run bootstraps
    boot_wc_dstn_arr = np.array(Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(fit_bootstrap)() for _ in range(n_bootstraps)
    ))

    return boot_wc_dstn_arr


def get_wc_num_boot_dstn(
    segments: np.ndarray, treatments: np.ndarray, outcomes: np.ndarray,
    targ_customers: np.ndarray, 
    budget: int, n_bootstraps: int = 500, power: float = -0.45, 
    n_jobs: int = -1, verbose: bool = False, **kwargs, 
):
    # parameter checks
    assert 0 > power > -0.5, "Power must be between 0 and -0.5"

    # attributes
    sample_size = segments.shape[0]
    epsilon_n = sample_size ** power 

    # initialize placeholder for bootstrap distribution
    boot_wc_dstn_arr = np.zeros(shape=(n_bootstraps, ))  # shape = (n_bootstraps)

    # estimate treatment effects using all data
    emp_segment_te_arr = segment_difference_in_mean(
        segments, treatments, outcomes
    )  # shape = (n_segments, )

    # function for fitting a single bootstrap 
    def fit_bootstrap() -> float:
        # bootstrap sample
        boot_segment_arr, boot_treatment_arr, boot_outcome_arr = stratified_bootstrap(
            segments, treatments, outcomes, 
            boot_sample_size=sample_size
        )

        # estimate treatment effects
        boot_segment_te_arr = segment_difference_in_mean(
            boot_segment_arr, boot_treatment_arr, boot_outcome_arr
        )  # shape = (n_segments, )

        # solve plugin optimization
        boot_decision, _ = optimize(
            targ_customers=targ_customers, 
            te_arr=boot_segment_te_arr, 
            budget=budget
        )

        # calculate estimation error
        norm_error = np.sqrt(sample_size) * (boot_segment_te_arr - emp_segment_te_arr)

        # calculate  perturbation
        perturb_segment_te_arr = emp_segment_te_arr + epsilon_n * norm_error

        # evaluate bootstrap policy using perturbed estimates
        perturb_val_est = obj_func(
            targ_customers=targ_customers, 
            te_arr=perturb_segment_te_arr, 
            targ_decision=boot_decision
        )

        # evaluate bootstrap decision using empirical estimates
        emp_boot_val_est = obj_func(
            targ_customers=targ_customers, 
            te_arr=emp_segment_te_arr, 
            targ_decision=boot_decision
        )

        return perturb_val_est - emp_boot_val_est
    
    # run bootstraps
    boot_wc_dstn_arr = np.array(Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(fit_bootstrap)() for _ in range(n_bootstraps)
    ))

    return boot_wc_dstn_arr


def bootstrap_correction_estimate(
    segments: np.ndarray, treatments: np.ndarray, outcomes: np.ndarray,
    targ_customers: np.ndarray, budget: int, 
    bootstrap_method: str = 'standard',
    n_jobs: int = -1, verbose: bool = False, **kwargs, 
) -> float:
    # estimate empirical treatment effects
    emp_segment_te_arr = segment_difference_in_mean(
        segments, treatments, outcomes
    )  # shape = (n_segments, )

    # get empirical value of the plugin policy
    _, plugin_val_est = optimize(
        targ_customers=targ_customers, 
        te_arr=emp_segment_te_arr, 
        budget=budget
    )

    # dictionary of bootstrap methods
    bootstrap_methods = {
        'standard': get_wc_boot_dstn,
        'm_out_of_n': get_wc_m_out_of_n_boot_dstn,
        'numerical': get_wc_num_boot_dstn
    }

    # get the bootstrap distribution of winner's curse
    boot_wc_dstn_arr = bootstrap_methods[bootstrap_method](
        segments, treatments, outcomes, 
        targ_customers, budget, 
        n_jobs=n_jobs, verbose=verbose, **kwargs
    )

    return plugin_val_est - np.nanmean(boot_wc_dstn_arr)


def sample_size_test(
    sample_sizes: Iterable[int],
    operations_params: dict,
    dgp_params: dict,
    data_params: dict,
    experiment_params: dict,
    estimators_dict: dict,
    n_jobs: int = -1, verbose: bool = False,
    base_seed: int = 0, 
    save_path: str = None, 
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
            n_jobs=n_jobs, verbose=verbose, 
            base_seed=base_seed, 
        )

        # extract parameters
        for estimator in estimators_list:
            wc_arr = np.array([record[f'{estimator}_wc'] for record in result_records])
            wc_pct_arr = np.array([record[f'{estimator}_wc'] / record['clairvoyant_val'] for record in result_records])

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
