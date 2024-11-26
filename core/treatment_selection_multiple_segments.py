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

from core.dgp import MultipleSegmentsTreatmentSelection


import matplotlib.pyplot as plt
tick_label_size = 12
legend_label_size = 12
axis_label_size = 14
title_size = 18
plt.rcParams['font.family'] = 'serif'


def estimate_te(
    segment_arr: np.ndarray, treatment_arr: np.ndarray, outcome_arr: np.ndarray, 
    n_segments: int, n_treatments: int
) -> np.ndarray:
    # initialize 
    est_te_arr = np.zeros((n_segments, n_treatments))

    # estimate treatment effect for each segment and treatment
    for segment_idx in range(n_segments):
        for treatment_idx in range(n_treatments):
            est_te_arr[segment_idx, treatment_idx] = np.mean(outcome_arr[(segment_arr == segment_idx) & (treatment_arr == treatment_idx)])

    return est_te_arr


def optimize(te_arr: np.ndarray, budget: int) -> tuple:
    """ 
    Solve the optimization problem: choose the best treatment to maximize profit

    Params:
    -------
    te_arr: np.ndarray, shape = (n_segments, n_treatments)
        Treatment effect array
    budget: int
        Budget

    Returns:
    --------
    opt_decision: np.ndarray[binary int], shape = (n_segments, n_treatments)
        Optimal treatment decision for each segment
    opt_val: float
        Optimal value
    """
    # for each segment (row), 

    opt_val = np.sum(opt_decision * te_arr)

    return opt_decision, opt_val


def obj_func(te_arr: np.ndarray, decision: np.ndarray) -> float:
    """ 
    Objective function: profit
    """
    return np.sum(decision * te_arr)


def repeated_experiments(
    operations_params: dict, 
    dgp_params: dict, 
    data_params: dict, 
    experiment_params: dict, 
    estimators_dict: dict, 
    verbose: bool = False, 
    n_jobs: int = -1
) -> list:
    # extract attributes
    n_experiments = experiment_params['n_experiments']
    sample_size = data_params['sample_size']
    stats = experiment_params['stats']

    # create data generation process
    dgp = MultipleSegmentsTreatmentSelection(**dgp_params)

    def run_single_experiment(experiment_id: int) -> dict:
        # generate data
        segment_arr, treatment_arr, outcome_arr = dgp.sample_treatment(sample_size, seed=experiment_id)

        # estimate treatment effect
        emp_te_arr = estimate_te(segment_arr, treatment_arr, outcome_arr, dgp.n_segments, dgp.n_treatments)

        # solve optimization
        plugin_decision, plugin_val_est = optimize(te_arr=emp_te_arr, budget=operations_params['budget'])

        # calculate true targeting value
        true_plugin_val = obj_func(te_arr=dgp.te_arr, decision=plugin_decision)

        # bookkeeping
        result = {
            'plugin_decision': plugin_decision, 
            'plugin_val_est': plugin_val_est, 
            'true_plugin_val': true_plugin_val, 
            'plugin_wc': plugin_val_est - true_plugin_val, 
            'plugin_wc_pct': (plugin_val_est - true_plugin_val) / true_plugin_val,
        }

        for name in estimators_dict.keys():
            temp_val_est = estimators_dict[name]['estimator'](
                segments=segment_arr, treatments=treatment_arr, outcomes=outcome_arr, 
                n_segments=dgp.n_segments, n_treatments=dgp.n_treatments, 
                stats=stats, **operations_params, **estimators_dict[name]['params']
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
    segment_arr: np.ndarray, treatment_arr: np.ndarray, outcome_arr: np.ndarray, 
    boot_sample_size: int, 
) -> tuple:
    """ 
    Stratified Bootstrap 

    Params:
    -------
    segment_arr: np.ndarray, shape = (n_samples, )
        Segment array
    treatment_arr: np.ndarray, shape = (n_samples, )
        Treatment array
    outcome_arr: np.ndarray, shape = (n_samples, )
        Outcome array
    boot_sample_size: int
        Bootstrap sample size

    Returns:
    --------
    segment_sample: np.ndarray, shape = (boot_sample_size, )
        Segment sample
    treatment_sample: np.ndarray, shape = (boot_sample_size, )
        Treatment sample
    outcome_sample: np.ndarray, shape = (boot_sample_size, )
        Outcome sample
    """
    # placeholder for the sampled data
    segment_sample = []
    treatment_sample = []
    outcome_sample = []

    # 
    unique_segments_arr = np.unique(segment_arr)
    unique_treatments_arr = np.unique(treatment_arr)

    n_combinations = unique_segments_arr.shape[0] * unique_treatments_arr.shape[0]

    for segment_val in unique_segments_arr:
        segment_indices = np.where(segment_arr == segment_val)[0]
        for treatment_val in unique_treatments_arr:            
            treatment_indices = np.where(treatment_arr == treatment_val)[0]

            sample_indices = np.intersect1d(segment_indices, treatment_indices)

            # sample the indices
            sample_indices = np.random.choice(
                sample_indices, size=int(boot_sample_size / n_combinations), replace=True
            )

            # append the sampled data
            segment_sample.extend(segment_arr[sample_indices])
            treatment_sample.extend(treatment_arr[sample_indices])
            outcome_sample.extend(outcome_arr[sample_indices])

    
    segment_sample = np.array(segment_sample)
    treatment_sample = np.array(treatment_sample)
    outcome_sample = np.array(outcome_sample)

    # check if the sample size is correct
    if segment_sample.shape[0] < boot_sample_size:
        remainder = boot_sample_size - segment_sample.shape[0]
        sample_indices = np.random.choice(
            np.arange(segment_arr.shape[0]), remainder, replace=True
        )

        segment_sample = np.concatenate([segment_sample, segment_arr[sample_indices]])
        treatment_sample = np.concatenate([treatment_sample, treatment_arr[sample_indices]])
        outcome_sample = np.concatenate([outcome_sample, outcome_arr[sample_indices]])

    return segment_sample, treatment_sample, outcome_sample


def get_wc_boot_dstn(
    segments: np.ndarray, treatments: np.ndarray, outcomes: np.ndarray, 
    n_segments: int, n_treatments: int, budget: int,
    n_bootstraps: int = 500, 
    n_jobs: int = -1, verbose: bool = False,
    **kwargs, 
) -> np.ndarray:
    # initialize array for the bootstrap distribution of winner's curse
    boot_wc_dstn_arr = np.zeros(shape=(n_bootstraps, ))  # shape = (n_bootstraps)

    # get treatment effect estimate using all data (empirical estimate)
    emp_te_arr = estimate_te(
        segment_arr=segments, treatment_arr=treatments, outcome_arr=outcomes, 
        n_segments=n_segments, n_treatments=n_treatments
    )
    
    def run_single_bootstrap() -> float:
        # bootstrap data
        boot_segment_arr, boot_treatment_arr, boot_outcome_arr = stratified_bootstrap(
            segment_arr=segments, treatment_arr=treatments, outcome_arr=outcomes, 
            boot_sample_size=treatments.shape[0]
        )

        # estimate treatment effect using bootstrap data
        boot_te_arr = estimate_te(
            segment_arr=boot_segment_arr, treatment_arr=boot_treatment_arr, outcome_arr=boot_outcome_arr, 
            n_segments=n_segments, n_treatments=n_treatments
        )

        # optimize
        boot_decision, boot_val_est = optimize(te_arr=boot_te_arr, budget=budget)
        
        # evaluate bootstrap targeting policy using empirical CATE estimates
        emp_boot_val_est = obj_func(te_arr=emp_te_arr, decision=boot_decision)
        
        return boot_val_est - emp_boot_val_est

    boot_wc_dstn_arr = np.array(Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(run_single_bootstrap)() for _ in range(n_bootstraps)
    ))

    return boot_wc_dstn_arr


def get_wc_m_out_of_n_boot_dstn(
    segments: np.ndarray, treatments: np.ndarray, outcomes: np.ndarray, 
    n_segments: int, n_treatments: int, budget: int,
    power: float = 0.95, n_bootstraps: int = 500,
    n_jobs: int = -1, verbose: bool = False,
    **kwargs, 
) -> np.ndarray:
    # parameter checks
    assert 0 < power < 1, 'Power should be between 0 and 1.'

    # attributes
    sample_size = treatments.shape[0]
    m = int(sample_size ** power)

    # initialize array for the bootstrap distribution of winner's curse
    boot_wc_dstn_arr = np.zeros(shape=(n_bootstraps, ))  # shape = (n_bootstraps)

    # get treatment effect estimate using all data (empirical estimate)
    emp_te_arr = estimate_te(
        segment_arr=segments, treatment_arr=treatments, outcome_arr=outcomes, 
        n_segments=n_segments, n_treatments=n_treatments
    )

    def run_single_bootstrap(boot_id: int) -> float:
        # bootstrap data
        boot_segment_arr, boot_treatment_arr, boot_outcome_arr = stratified_bootstrap(
            segment_arr=segments, treatment_arr=treatments, outcome_arr=outcomes, 
            boot_sample_size=m
        )

        # estimate treatment effect using bootstrap data
        boot_te_arr = estimate_te(
            segment_arr=boot_segment_arr, treatment_arr=boot_treatment_arr, outcome_arr=boot_outcome_arr, 
            n_segments=n_segments, n_treatments=n_treatments
        )

        # optimize
        boot_decision, boot_val_est = optimize(te_arr=boot_te_arr, budget=budget)

        # evaluate bootstrap targeting policy using empirical CATE estimates
        emp_boot_val_est = obj_func(te_arr=emp_te_arr, decision=boot_decision)

        return boot_val_est - emp_boot_val_est

    boot_wc_dstn_arr = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(run_single_bootstrap)(boot_id) for boot_id in range(n_bootstraps)
    )

    return np.array(boot_wc_dstn_arr)


def get_wc_num_boot_dstn(
    segments: np.ndarray, treatments: np.ndarray, outcomes: np.ndarray, 
    n_segments: int, n_treatments: int, budget: int,
    power: float = -0.45, n_bootstraps: int = 500,
    n_jobs: int = -1, verbose: bool = False,
    **kwargs, 
) -> np.ndarray:
    # parameter checks
    assert 0 > power > -0.5, "Power must be between 0 and -0.5"

    # attributes
    sample_size = treatments.shape[0]
    epsilon_n = sample_size ** power

    # initialize array for the bootstrap distribution of winner's curse
    boot_wc_dstn_arr = np.zeros(shape=(n_bootstraps, ))  # shape = (n_bootstraps)

    # get treatment effect estimate using all data (empirical estimate)
    emp_te_arr = estimate_te(segments, treatments, outcomes, n_segments, n_treatments)

    def run_single_bootstrap() -> float:
        # bootstrap data
        boot_segment_arr, boot_treatment_arr, boot_outcome_arr = stratified_bootstrap(
            segment_arr=segments, treatment_arr=treatments, outcome_arr=outcomes, 
            boot_sample_size=sample_size
        )

        # estimate treatment effect using bootstrap data
        boot_te_arr = estimate_te(
            segment_arr=boot_segment_arr, treatment_arr=boot_treatment_arr, outcome_arr=boot_outcome_arr, 
            n_segments=n_segments, n_treatments=n_treatments
        )

        # optimize
        boot_decision, _ = optimize(te_arr=boot_te_arr, budget=budget)

        # calculate treatment effect estimation error 
        norm_err_arr = np.sqrt(sample_size) * (boot_te_arr - emp_te_arr)

        # calculate perturbed treatment effect
        pert_te_arr = emp_te_arr + epsilon_n * norm_err_arr 

        # evaluate boot_decision with perturbed treatment effect
        perturb_val_est = obj_func(te_arr=pert_te_arr, decision=boot_decision)

        # evaluate bootstrap targeting policy using empirical CATE estimates
        emp_val_est = obj_func(te_arr=emp_te_arr, decision=boot_decision)

        return perturb_val_est - emp_val_est
    
    boot_wc_dstn_arr = np.array(Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(run_single_bootstrap)() for _ in range(n_bootstraps)
    ))

    return boot_wc_dstn_arr


def bootstrap_correction_estimate(
    segments: np.ndarray, treatments: np.ndarray, outcomes: np.ndarray, 
    n_segments: int, n_treatments: int, budget: int,
    bootstrap_method: str = 'standard', stats: str = 'mean',
    **kwargs, 
) -> float:
    # estimate treatment effect
    emp_te_arr = estimate_te(
        segment_arr=segments, treatment_arr=treatments, outcome_arr=outcomes, 
        n_segments=n_segments, n_treatments=n_treatments
    )

    # solve plugin problem
    _, plugin_val_est = optimize(te_arr=emp_te_arr, budget=budget)

    # dictionary of available bootstrap methods
    boot_methods_dict = {
        'standard': get_wc_boot_dstn, 
        'm_out_of_n': get_wc_m_out_of_n_boot_dstn, 
        'numerical': get_wc_num_boot_dstn, 
    }

    # get the bootstrap distribution of winner's curse
    boot_wc_dstn_arr = boot_methods_dict[bootstrap_method](
        segments=segments, treatments=treatments, outcomes=outcomes, 
        n_segments=n_segments, n_treatments=n_treatments, budget=budget,
        **kwargs, 
    )

    if stats == 'mean':
        correction = boot_wc_dstn_arr.mean()
    elif stats == 'median':
        correction = np.median(boot_wc_dstn_arr)
    else:
        raise ValueError(f'Invalid stats: {stats}')

    return plugin_val_est - correction


