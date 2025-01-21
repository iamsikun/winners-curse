import os 
import sys 
import pickle 
sys.path.insert(0, os.path.abspath('.'))
import warnings 
from joblib import Parallel, delayed
from typing import Iterable
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from scipy.optimize import fsolve
from scipy.stats import truncnorm
from sklearn.linear_model import LogisticRegression

from core.dgp import SingleSegmentTreatmentSelection
from core.bayes_methods import EmpiricalBayes

import matplotlib.pyplot as plt
tick_label_size = 12
legend_label_size = 12
axis_label_size = 14
title_size = 18
plt.rcParams['font.family'] = 'serif'


def estimate_te(
    treatments: np.ndarray, outcomes: np.ndarray, response_type: str, 
) -> np.ndarray:
    if response_type in ['continuous', 'bernoulli']:
        unique_treatments = np.sort(np.unique(treatments))
        te_est_arr = np.zeros_like(unique_treatments, dtype=float)
        for treatment in unique_treatments:
            te_est_arr[treatment] = outcomes[treatments == treatment].mean()
    elif response_type == 'logit':
        exog_var = np.zeros((treatments.size, 2))
        exog_var[np.arange(treatments.size), treatments.astype(int)] = 1

        logit_model = LogisticRegression(
            penalty=None, fit_intercept=False, max_iter=1000
        ).fit(X=exog_var, y=outcomes)

        te_est_arr = logit_model.coef_[0]
    else:
        raise ValueError(f'Invalid response type: {response_type}')

    return te_est_arr


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


def calculate_winners_curse_measures(
    result_records: list, estimators_dict: dict, data_params: dict
) -> dict:
    # initialize dictionary
    wc_measure_dict = {}

    est_val_arr = np.array([record['plugin_val_est'] for record in result_records])
    true_val_arr = np.array([record['true_plugin_val'] for record in result_records])
    correct_decision_arr = np.array([record['plugin_decision'] == record['clairvoyant_decision'] for record in result_records])

    wc_measure_dict.update({
        'nc_val_est_avg': np.mean(est_val_arr), 'nc_val_est_se': np.std(est_val_arr) / np.sqrt(data_params['sample_size']), 
        'nc_val_true_avg': np.mean(true_val_arr), 'nc_val_true_se': np.std(true_val_arr) / np.sqrt(data_params['sample_size']),
        'nc_correct_decision_rate': np.mean(correct_decision_arr),
    })

    # no correction
    nc_wc_arr = est_val_arr - true_val_arr  # winner's curse
    nc_wc_pct_arr = nc_wc_arr / np.abs(true_val_arr)  # winner's curse percentage of true value

    wc_measure_dict.update({
        'nc_wc_arr': nc_wc_arr, 'nc_wc_pct_arr': nc_wc_pct_arr,
        'nc_wc_avg': np.mean(nc_wc_arr), 'nc_wc_se': np.std(nc_wc_arr) / np.sqrt(data_params['sample_size']),
        'nc_wc_pct_avg': np.mean(nc_wc_arr / np.abs(true_val_arr)), 'nc_wc_pct_se': np.std(nc_wc_arr / np.abs(true_val_arr)) / np.sqrt(data_params['sample_size']),
        'nc_wc_rmse': np.sqrt(np.mean(np.square(nc_wc_arr))),
    })

    # for each estimator
    for estimator in estimators_dict.keys():
        est_val_arr = np.array([record[f'{estimator}_val_est'] for record in result_records])
        true_val_arr = np.array([record[f'{estimator}_val_true'] for record in result_records])

        correct_decision_arr = np.array([record[f'{estimator}_decision'] == record['clairvoyant_decision'] for record in result_records])

        wc_arr = est_val_arr - true_val_arr
        wc_pct_arr = wc_arr / np.abs(true_val_arr)  # winner's curse percentage of true value

        wc_measure_dict.update({
            f'{estimator}_correct_decision_rate': np.mean(correct_decision_arr),
            f'{estimator}_val_est_avg': np.nanmean(est_val_arr), f'{estimator}_val_est_se': np.nanstd(est_val_arr) / np.sqrt(data_params['sample_size']),
            f'{estimator}_val_true_avg': np.mean(true_val_arr), f'{estimator}_val_true_se': np.std(true_val_arr) / np.sqrt(data_params['sample_size']),
            f'{estimator}_wc_arr': wc_arr, f'{estimator}_wc_pct_arr': wc_pct_arr,
            f'{estimator}_wc_avg': np.nanmean(wc_arr), f'{estimator}_wc_se': np.nanstd(wc_arr) / np.sqrt(data_params['sample_size']),
            f'{estimator}_wc_pct_avg': np.nanmean(wc_pct_arr), f'{estimator}_wc_pct_se': np.nanstd(wc_pct_arr) / np.sqrt(data_params['sample_size']),
            f'{estimator}_wc_rmse': np.sqrt(np.nanmean(np.square(wc_arr))),
        })

    return wc_measure_dict


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
    dgp = SingleSegmentTreatmentSelection(**dgp_params)

    # solve clairvoyant optimization
    clairvoyant_decision, clairvoyant_val = optimize(te_arr=dgp.te_arr, **operations_params)

    def run_single_experiment(experiment_id: int) -> dict:
        # generate data
        treatment_arr, outcome_arr = dgp.sample(sample_size=sample_size, seed=experiment_id)

        # fit potential outcome model
        emp_te = estimate_te(treatments=treatment_arr, outcomes=outcome_arr, response_type=dgp_params['response_type'])
        
        # solve plugin optimization
        plugin_decision, plugin_val_est = optimize(
            te_arr=emp_te, **operations_params
        )

        # calculate actual targeting value of the plugin targeting policy
        true_plugin_val = obj_func(
            targ_decision=plugin_decision, te_arr=dgp.te_arr, 
            **operations_params
        )
        
        # bookkeeping
        result = {
            'plugin_decision': plugin_decision, 
            'clairvoyant_decision': clairvoyant_decision,
            'clairvoyant_val': clairvoyant_val,
            'plugin_val_est': plugin_val_est, 
            'true_plugin_val': true_plugin_val, 
            'plugin_wc': plugin_val_est - true_plugin_val, 
        }

        for name in estimators_dict.keys():
            temp_decision, temp_val_est = estimators_dict[name]['estimator'](
                treatments=treatment_arr, 
                outcomes=outcome_arr,
                response_type=dgp_params['response_type'],
                stats=stats,
                **operations_params, 
                **estimators_dict[name]['params']
            )
            temp_decision = temp_decision if temp_decision is not None else plugin_decision
            temp_val_true = obj_func(
                targ_decision=temp_decision, te_arr=dgp.te_arr,
            )
            result.update({
                f'{name}_val_true': temp_val_true,
                f'{name}_decision': temp_decision,
                f'{name}_val_est': temp_val_est, 
                f'{name}_wc': temp_val_est - true_plugin_val, 
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
    response_type: str, n_bootstraps: int = 500, 
    **kwargs, 
) -> np.ndarray:
    # initialize array for the bootstrap distribution of winner's curse
    boot_wc_dstn_arr = np.zeros(shape=(n_bootstraps, ))  # shape = (n_bootstraps)
    boot_decision_arr = np.zeros(shape=(n_bootstraps, ), dtype=bool)  # shape = (n_bootstraps)

    # get treatment effect estimate using all data (empirical estimate)
    emp_te_arr = estimate_te(treatments=treatments, outcomes=outcomes, response_type=response_type)
    
    # create bootstrap distribution
    for boot_id in range(n_bootstraps):
        # bootstrap data
        boot_treatment_arr, boot_outcome_arr = stratified_bootstrap(
            treatments=treatments, outcomes=outcomes, boot_sample_size=treatments.shape[0]
        )

        # estimate treatment effect using bootstrap data
        boot_te_arr = estimate_te(treatments=boot_treatment_arr, outcomes=boot_outcome_arr, response_type=response_type)

        # optimize
        boot_decision, boot_val_est = optimize(te_arr=boot_te_arr)
        
        # evaluate bootstrap targeting policy using empirical CATE estimates
        emp_boot_val_est = obj_func(targ_decision=boot_decision, te_arr=emp_te_arr)
        
        boot_decision_arr[boot_id] = boot_decision
        boot_wc_dstn_arr[boot_id] = boot_val_est - emp_boot_val_est

    return boot_wc_dstn_arr


def get_wc_m_out_of_n_boot_dstn(
    treatments: np.ndarray, outcomes: np.ndarray,
    response_type: str,
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
    emp_te_arr = estimate_te(treatments=treatments, outcomes=outcomes, response_type=response_type)
    
    # create bootstrap distribution
    for boot_id in range(n_bootstraps):
        # bootstrap data
        boot_treatment_arr, boot_outcome_arr = stratified_bootstrap(
            treatments=treatments, outcomes=outcomes, boot_sample_size=m
        )

        # estimate treatment effect using bootstrap data
        boot_te_arr = estimate_te(treatments=boot_treatment_arr, outcomes=boot_outcome_arr, response_type=response_type)

        # optimize
        boot_decision, boot_val_est = optimize(te_arr=boot_te_arr)
        
        # evaluate bootstrap targeting policy using empirical CATE estimates
        emp_boot_val_est = obj_func(targ_decision=boot_decision, te_arr=emp_te_arr)
        
        boot_decision_arr[boot_id] = boot_decision
        boot_wc_dstn_arr[boot_id] = boot_val_est - emp_boot_val_est

    return boot_wc_dstn_arr


def get_wc_num_boot_dstn(
    treatments: np.ndarray, outcomes: np.ndarray,
    response_type: str, 
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
    emp_te_arr = estimate_te(treatments=treatments, outcomes=outcomes, response_type=response_type)
    
    # create bootstrap distribution
    for boot_id in range(n_bootstraps):
        # bootstrap data
        boot_treatment_arr, boot_outcome_arr = stratified_bootstrap(
            treatments=treatments, outcomes=outcomes, boot_sample_size=sample_size
        )

        # estimate treatment effect using bootstrap data
        boot_te_arr = estimate_te(treatments=boot_treatment_arr, outcomes=boot_outcome_arr, response_type=response_type)

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
    response_type: str, 
    bootstrap_method: str = 'standard', stats: str = 'mean',
    **kwargs, 
) -> tuple:
    # fit the potential outcome model
    emp_te_arr = estimate_te(treatments=treatments, outcomes=outcomes, response_type=response_type)

    # solve plugin problem
    plugin_decision, plugin_val_est = optimize(te_arr=emp_te_arr)

    # dictionary of available bootstrap methods
    boot_methods_dict = {
        'standard': get_wc_boot_dstn, 
        'm_out_of_n': get_wc_m_out_of_n_boot_dstn, 
        'numerical': get_wc_num_boot_dstn, 
    }

    # get the bootstrap distribution of winner's curse
    boot_wc_dstn_arr = boot_methods_dict[bootstrap_method](
        treatments=treatments, outcomes=outcomes, response_type=response_type, **kwargs, 
    )

    if stats == 'mean':
        correction = boot_wc_dstn_arr.mean()
    elif stats == 'median':
        correction = np.median(boot_wc_dstn_arr)
    else:
        raise ValueError(f'Invalid stats: {stats}')

    return plugin_decision, plugin_val_est - correction


def sample_splitting_estimate(
    treatments: np.ndarray, outcomes: np.ndarray, 
    response_type: str, **kwargs, 
) -> tuple: 
    # split the sample in to training and estimation
    train_treatments, est_treatments, train_outcomes, est_outcomes = train_test_split(
        treatments, outcomes, test_size=0.5
    )

    # estimate the treatment effect on the training sample
    emp_te_arr = estimate_te(treatments=train_treatments, outcomes=train_outcomes, response_type=response_type)

    # estimate the treatment effect on the estimation sample
    est_te_arr = estimate_te(treatments=est_treatments, outcomes=est_outcomes, response_type=response_type)

    # optimize based on the training set
    train_decision, _ = optimize(te_arr=emp_te_arr)

    # evaluate the decision on the estimation set
    val_est = obj_func(targ_decision=train_decision, te_arr=est_te_arr)

    return train_decision, val_est


def cross_validation_estimate(
    treatments: np.ndarray, outcomes: np.ndarray, 
    response_type: str,
    n_splits: int = 10, seed: int = None, **kwargs
) -> tuple:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    policy_values = []
    train_decisions = []

    for train_index, val_index in kf.split(treatments):
        train_treatments, val_treatments = treatments[train_index], treatments[val_index]
        train_outcomes, val_outcomes = outcomes[train_index], outcomes[val_index]

        # Estimate treatment effect on the training set
        train_te_arr = estimate_te(treatments=train_treatments, outcomes=train_outcomes, response_type=response_type)

        # Optimize based on the training set
        train_decision, _ = optimize(te_arr=train_te_arr)

        # Evaluate the decision on the validation set
        val_te_arr = estimate_te(treatments=val_treatments, outcomes=val_outcomes, response_type=response_type)
        val_est = obj_func(targ_decision=train_decision, te_arr=val_te_arr)

        policy_values.append(val_est)
        train_decisions.append(train_decision)

    avg_policy_value = np.mean(policy_values)
    # pick the most frequent decision
    train_decision = np.argmax(np.bincount(train_decisions))

    return train_decision, avg_policy_value


def empirical_bayes_estimate(
    treatments: np.ndarray, outcomes: np.ndarray, 
    response_type: str, 
    dof: int = 5, bin_width: float = 0.2, 
    **kwargs,
) -> tuple:
    # estimate targeting policy
    emp_te_arr = estimate_te(treatments=treatments, outcomes=outcomes, response_type=response_type)
    plugin_decision, _ = optimize(te_arr=emp_te_arr)

    # estimate the posterior of the treatment effect for each customer
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            emp_bayes = EmpiricalBayes(
                dof=dof, bin_width=bin_width, sigma=np.std(outcomes), 
            ).fit(outcomes)
            post_cust_te_arr = emp_bayes.predict(outcomes)  # shape = (sample_size, )
        except RuntimeWarning:
            return plugin_decision, np.nan
        else:
            pass # no exception
    
    # calculate the posterior treatment effect for each treatment
    post_te_arr = np.array([np.mean(post_cust_te_arr[treatments == idx]) for idx in range(2)])

    # evaluate the plugin policy using the posterior treatment effect
    val_est = obj_func(
        targ_decision=plugin_decision, te_arr=post_te_arr
    )

    return plugin_decision, val_est


def normal_prior_bayes_estimate(
    treatments: np.ndarray, outcomes: np.ndarray,
    response_type: str,
    prior_mean: float, prior_std: float, 
    **kwargs, 
) -> tuple:
    # estimate targeting policy
    emp_te_arr = estimate_te(treatments=treatments, outcomes=outcomes, response_type=response_type)
    plugin_decision, _ = optimize(te_arr=emp_te_arr)


    for idx, treatment in enumerate(np.unique(treatments)):
        # calculate sampling variance
        sampling_var = np.var(outcomes[treatments == treatment]) / np.sum(treatments == treatment)
        
        # calculate posterior effect
        weight = prior_std ** 2 / (prior_std ** 2 + sampling_var)
        posterior_te = weight * emp_te_arr[idx] + (1 - weight) * prior_mean

        emp_te_arr[idx] = posterior_te

    val_est = obj_func(
        targ_decision=plugin_decision, te_arr=emp_te_arr
    )

    return plugin_decision, val_est



def conditional_selective_inference(
    treatments: np.ndarray, outcomes: np.ndarray, 
    response_type: str,
    quantile: float = 0.5, **kwargs
) -> float:
    """ 
    Compute the conditional inference method from (Andrews et al. 2024, QJE)
    """
    # estimate treatment effect and plugin decision
    emp_te_arr = estimate_te(treatments=treatments, outcomes=outcomes, response_type=response_type)  # shape = (n_treatments, )
    plugin_decision, _ = optimize(te_arr=emp_te_arr)

    # get the mean and std of the max item
    max_item_mean = emp_te_arr[plugin_decision]
    max_item_std = np.std(outcomes[treatments == plugin_decision]) / np.sqrt(np.sum(treatments == plugin_decision))

    # get the second max mean
    second_max_mean = np.delete(emp_te_arr, plugin_decision).max()

    def local_truncated_normal_cdf(x, mu) -> float:
        trunc_lb = (second_max_mean - mu) / max_item_std

        return truncnorm.cdf(x, trunc_lb, np.inf, loc=mu, scale=max_item_std)
    
    return plugin_decision, fsolve(
        func=lambda mu: local_truncated_normal_cdf(max_item_mean, mu) - 1 + quantile, x0=max_item_mean
    )[0]


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


def noise_level_test(
    noise_levels: Iterable[float],
    operations_params: dict,
    dgp_params: dict,
    data_params: dict,
    experiment_params: dict,
    estimators_dict: dict,
    n_jobs: int = -1, verbose: bool = False,
    save_path: str = None, 
) -> dict:
    # placeholder for results
    result_dict = {noise_level: {} for noise_level in noise_levels}

    # attributes
    estimators_list = ['plugin'] + list(estimators_dict.keys())
    
    for noise_level in noise_levels:
        if verbose:
            print(f'Running noise level {noise_level}...')
            start = datetime.now()

        # update noise level in dgp params
        dgp_params['noise_std'] = noise_level

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

            result_dict[noise_level].update({
                f'{estimator}_wc_mean': wc_arr.mean(),
                f'{estimator}_wc_se': wc_arr.std() / np.sqrt(data_params['sample_size']),
                f'{estimator}_wc_pct_mean': wc_pct_arr.mean(),
                f'{estimator}_wc_pct_se': wc_pct_arr.std() / np.sqrt(data_params['sample_size']),
            })

        if verbose:
            print(f'Finished noise level {noise_level} in {datetime.now() - start}')

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(result_dict, f)

    return result_dict


def te_diff_test(
    params_records: list,
    operations_params: dict,
    dgp_params: dict,
    data_params: dict,
    experiment_params: dict,
    estimators_dict: dict,
    n_jobs: int = -1, verbose: bool = False,
    save_path: str = None, 
) -> dict:
    # placeholder for results
    result_dict = {record['te_diff']: {} for record in params_records}

    # attributes
    estimators_list = ['plugin'] + list(estimators_dict.keys())

    for record in params_records:
        te_diff = record['te_diff']

        # update te diff in dgp params
        dgp_params['te_arr'][1] = dgp_params['te_arr'][0] + te_diff
        estimators_dict['mn_bootstrap']['params']['power'] = record['mn_power']
        estimators_dict['num_bootstrap']['params']['power'] = record['num_power']

        if verbose:
            print(f'Running with treatment effects = ({dgp_params["te_arr"][0]:.4f}, {dgp_params["te_arr"][1]:.4f})...')
            start = datetime.now()

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

            result_dict[te_diff].update({
                f'{estimator}_wc_mean': wc_arr.mean(),
                f'{estimator}_wc_se': wc_arr.std() / np.sqrt(data_params['sample_size']),
                f'{estimator}_wc_pct_mean': wc_pct_arr.mean(),
                f'{estimator}_wc_pct_se': wc_pct_arr.std() / np.sqrt(data_params['sample_size']),
            })

        if verbose:
            print(f'Finished te diff {te_diff} in {datetime.now() - start}')

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(result_dict, f)

    return result_dict


def ratio_test(
    params_dict: dict,
    operations_params: dict,
    dgp_params: dict,
    data_params: dict,
    experiment_params: dict,
    n_jobs: int = -1, verbose: bool = False,
) -> list:
    # placeholder for results
    result_records = []

    for params in params_dict.values():
        for te_diff, noise_std in zip(params['te_diff'], params['noise_std']):

            dgp_params['te_arr'][1] = dgp_params['te_arr'][0] + te_diff
            dgp_params['noise_std'] = noise_std

            if verbose:
                print(f'Running with te_diff = {te_diff}, noise_std = {noise_std}...')
                start = datetime.now()

            result_record = repeated_experiments(
                operations_params=operations_params, 
                dgp_params=dgp_params, 
                data_params=data_params, 
                experiment_params=experiment_params,  
                estimators_dict={}, # only test no correction estimator 
                n_jobs=n_jobs, verbose=verbose, 
            )

            # extract parameters
            wc_arr = np.array([record['plugin_wc'] for record in result_record])
            wc_pct_arr = np.array([record['plugin_wc_pct'] for record in result_record])

            result_records.append({
                'te_diff': te_diff, 'noise_std': noise_std, 
                'plugin_wc_mean': wc_arr.mean(), 
                'plugin_wc_se': wc_arr.std() / np.sqrt(data_params['sample_size']),
                'plugin_wc_pct_mean': wc_pct_arr.mean(),
                'plugin_wc_pct_se': wc_pct_arr.std() / np.sqrt(data_params['sample_size']),
            })

            if verbose:
                print(f'Finished te_diff = {te_diff}, noise_std = {noise_std} in {datetime.now() - start}')

    return result_records


def visualize_sensitivity_test(
    label: str, 
    estimators_list: list, result_df: pd.DataFrame, 
    metric: str, plot_error_bar: bool = True, 
    title: str = None, model_label_map: dict = None, 
    save_path: str = None, 
):
    assert metric in ['wc', 'wc_pct', 'roi_wc'], "Invalid metric"

    if model_label_map is None:
        model_label_map = {estimator: estimator for estimator in estimators_list}

    pct_multiplier = 100 if metric == 'wc_pct' or metric == 'roi_wc' else 1

    fig, ax = plt.subplots(1, 1, figsize=(10, 6.18))

    for idx, estimator in enumerate(estimators_list):
        ax.plot(
            result_df.index, 
            result_df[f'{estimator}_{metric}_mean'] * pct_multiplier,
            'o-', markersize=5, label=model_label_map[estimator], 
            color=f'C{idx}'
        )
        if plot_error_bar:
            ax.fill_between(
                result_df.index, 
                pct_multiplier * (result_df[f'{estimator}_{metric}_mean'] - 1.96 * result_df[f'{estimator}_{metric}_se']), 
                pct_multiplier * (result_df[f'{estimator}_{metric}_mean'] + 1.96 * result_df[f'{estimator}_{metric}_se']), 
                color=f'C{idx}', alpha=0.3
            )

    ax.set_title(title, fontsize=title_size)
    ax.set_xlabel(label, fontsize=axis_label_size)
    if metric == 'wc':
        ax.set_ylabel("Winner's Curse", fontsize=axis_label_size)
    else:
        ax.set_ylabel("Winner's Curse (%)", fontsize=axis_label_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_size)
    ax.grid()
    ax.legend(loc='best', fontsize=legend_label_size)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()    

