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

from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy.interpolate import UnivariateSpline

from core.dgp import SingleSegment
from core.bayes_methods import EmpiricalBayes


treatment_space = np.array([0, 1])


def difference_in_mean(treatments: np.ndarray, outcomes: np.ndarray) -> float:
    """ 
    Return the difference in mean outcomes between treated and control groups.

    Params:
    -------
    treatments: np.ndarray
        Array of treatments
    outcomes: np.ndarray
        Array of outcomes

    Returns:
    --------
    float
        Difference in mean outcomes between treated and control groups
    """
    return outcomes[treatments == 1].mean() - outcomes[treatments == 0].mean()


def optimize(te: float, price: float, cost: float) -> tuple:
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


def calculate_winners_curse_measures(
    result_records: list, operations_params: dict, estimators_dict: dict, data_params: dict
) -> dict:
    # initialize dictionary
    wc_measure_dict = {}

    decision_arr = np.array([record['plugin_decision'] for record in result_records])
    est_val_arr = np.array([record['plugin_val_est'] for record in result_records])
    true_val_arr = np.array([record['true_plugin_val'] for record in result_records])

    est_roi_arr = est_val_arr / operations_params['cost']
    true_roi_arr = true_val_arr / operations_params['cost']

    wc_measure_dict.update({
        'nc_val_est_avg': np.mean(est_val_arr), 'nc_val_est_se': np.std(est_val_arr) / np.sqrt(data_params['sample_size']), 
        'nc_val_true_avg': np.mean(true_val_arr), 'nc_val_true_se': np.std(true_val_arr) / np.sqrt(data_params['sample_size']),
        'nc_roi_est_avg': np.mean(est_roi_arr), 'nc_roi_est_se': np.std(est_roi_arr) / np.sqrt(data_params['sample_size']), 
        'roi_true_avg': np.mean(true_roi_arr),
    })

    # no correction
    nc_wc_arr = est_val_arr - true_val_arr  # winner's curse
    nc_wc_arr[~decision_arr] = 0  # when decision is not to target, there is no winner's curse
    nc_wc_pct_arr = nc_wc_arr[decision_arr] / np.abs(true_val_arr[decision_arr])  # winner's curse percentage of true value
    nc_over_roi_arr = nc_wc_arr / operations_params['cost']  # overestimated ROI

    wc_measure_dict.update({
        'nc_wc_arr': nc_wc_arr, 'nc_over_roi_arr': nc_over_roi_arr, 'nc_wc_pct_arr': nc_wc_pct_arr,
        'nc_wc_avg': np.mean(nc_wc_arr), 'nc_wc_se': np.std(nc_wc_arr) / np.sqrt(data_params['sample_size']),
        'nc_over_roi_avg': np.mean(nc_over_roi_arr), 'nc_over_roi_se': np.std(nc_over_roi_arr) / np.sqrt(data_params['sample_size']),
        'nc_wc_pct_avg': np.mean(nc_wc_arr / np.abs(true_val_arr)), 'nc_wc_pct_se': np.std(nc_wc_arr / np.abs(true_val_arr)) / np.sqrt(data_params['sample_size']),
        'nc_wc_rmse': np.sqrt(np.mean(np.square(nc_wc_arr))),
    })

    # for each estimator
    for estimator in estimators_dict.keys():
        decision_arr = np.array([record[f'{estimator}_decision'] for record in result_records])
        est_val_arr = np.array([record[f'{estimator}_val_est'] for record in result_records])
        true_val_arr = np.array([record[f'{estimator}_val_true'] for record in result_records])
        est_roi_arr = est_val_arr / operations_params['cost']

        wc_arr = est_val_arr - true_val_arr
        wc_arr[~decision_arr] = 0  # when decision is not to target, there is no winner's curse
        wc_pct_arr = wc_arr[decision_arr] / np.abs(true_val_arr[decision_arr])  # winner's curse percentage of true value
        over_roi_arr = wc_arr / operations_params['cost']

        wc_measure_dict.update({
            f'{estimator}_val_est_avg': np.mean(est_val_arr), f'{estimator}_val_est_se': np.std(est_val_arr) / np.sqrt(data_params['sample_size']),
            f'{estimator}_val_true_avg': np.mean(true_val_arr), f'{estimator}_val_true_se': np.std(true_val_arr) / np.sqrt(data_params['sample_size']),
            f'{estimator}_roi_est_avg': np.mean(est_roi_arr), f'{estimator}_roi_est_se': np.std(est_roi_arr) / np.sqrt(data_params['sample_size']),
            f'{estimator}_wc_arr': wc_arr, f'{estimator}_over_roi_arr': over_roi_arr, f'{estimator}_wc_pct_arr': wc_pct_arr,
            f'{estimator}_wc_avg': np.mean(wc_arr), f'{estimator}_wc_se': np.std(wc_arr) / np.sqrt(data_params['sample_size']),
            f'{estimator}_over_roi_avg': np.mean(over_roi_arr), f'{estimator}_over_roi_se': np.std(over_roi_arr) / np.sqrt(data_params['sample_size']),
            f'{estimator}_wc_pct_avg': np.mean(wc_pct_arr), f'{estimator}_wc_pct_se': np.std(wc_pct_arr) / np.sqrt(data_params['sample_size']),
            f'{estimator}_wc_rmse': np.sqrt(np.mean(np.square(wc_arr))),
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
    conditional_on_treated = experiment_params['conditional_on_treated']
    stats = experiment_params['stats']

    # create data generation process
    dgp = SingleSegment(**dgp_params)

    def run_single_experiment(experiment_id: int) -> dict:
        # generate data
        treatment_arr, outcome_arr = dgp.sample(sample_size=sample_size, seed=experiment_id)

        # fit potential outcome model
        emp_te = difference_in_mean(
            treatments=treatment_arr, outcomes=outcome_arr
        )
        
        # solve plugin optimization
        plugin_decision, plugin_val_est = optimize(
            te=emp_te, **operations_params
        )

        # solve clairvoyant optimization
        _, clairvoyant_val = optimize(
            te=dgp.te, **operations_params
        )

        if conditional_on_treated and plugin_decision == 0:
            return None

        # calculate actual targeting value of the plugin targeting policy
        true_plugin_val = objective_func(
            targ_decision=plugin_decision, te=dgp.te, 
            **operations_params
        )
        
        # bookkeeping
        plugin_wc = plugin_val_est - true_plugin_val if plugin_decision else 0  # if decide not to target, there is no winner's curse
        result = {
            'plugin_decision': plugin_decision, 
            'clairvoyant_val': clairvoyant_val,
            'plugin_val_est': plugin_val_est, 
            'true_plugin_val': true_plugin_val, 
            'plugin_wc': plugin_wc, 
        }

        for name in estimators_dict.keys():
            temp_decision, temp_val_est = estimators_dict[name]['estimator'](
                treatments=treatment_arr, 
                outcomes=outcome_arr,
                conditional_on_treated=conditional_on_treated,
                stats=stats,
                **operations_params, 
                **estimators_dict[name]['params']
            )

            temp_decision = temp_decision if temp_decision is not None else plugin_decision
            temp_val_true = objective_func(
                targ_decision=temp_decision, te=dgp.te, 
                **operations_params
            )

            result.update({
                f'{name}_val_est': temp_val_est, 
                f'{name}_decision': temp_decision,
                f'{name}_val_true': temp_val_true,
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


def calculate_wc_pct(record: dict) -> dict:
    """
    Calculate the percentage of winner's curse for a given record:
    wc_pct = wc / true_plugin_targ_val

    Params:
    -------
    record: dict
        A dictionary containing the following
        - true_plugin_targ_val: float
            True value of targeting from the plugin optimization
        - keys ending with 'wc': float
            Winner's curse value

    Returns:
    --------
    wc_pct: dict
        A dictionary containing the percentage of winner's curse
    """
    return {
        key: record[key] / record['true_plugin_targ_val'] 
        for key in record.keys() if key.endswith('wc') and record['plugin_targ_val_est'] != 0
    }


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
    price: float, cost: float, 
    n_bootstraps: int = 500, conditional_on_treated: bool = False, 
    **kwargs, 
) -> np.ndarray:
    # initialize array for the bootstrap distribution of winner's curse
    boot_wc_dstn_arr = np.zeros(shape=(n_bootstraps, ))  # shape = (n_bootstraps)
    boot_decision_arr = np.zeros(shape=(n_bootstraps, ), dtype=bool)  # shape = (n_bootstraps)

    # get treatment effect estimate using all data (empirical estimate)
    emp_te = difference_in_mean(treatments=treatments, outcomes=outcomes)
    
    # create bootstrap distribution
    for boot_id in range(n_bootstraps):
        # bootstrap data
        boot_treatment_arr, boot_outcome_arr = stratified_bootstrap(
            treatments=treatments, outcomes=outcomes, boot_sample_size=treatments.shape[0]
        )

        # estimate treatment effect using bootstrap data
        boot_te = difference_in_mean(treatments=boot_treatment_arr, outcomes=boot_outcome_arr)

        # optimize
        boot_decision, boot_val_est = optimize(
            te=boot_te, price=price, cost=cost
        )
        
        # evaluate bootstrap targeting policy using empirical CATE estimates
        emp_boot_val_est = objective_func(
            targ_decision=boot_decision,  # decision to be evaluated: bootstrapped decision
            te=emp_te,  # evaluation criterion: empirical treatment effect
            price=price, cost=cost
        )
        
        boot_decision_arr[boot_id] = boot_decision
        boot_wc_dstn_arr[boot_id] = boot_val_est - emp_boot_val_est

    if conditional_on_treated:
        return boot_wc_dstn_arr[boot_decision_arr]
    else:
        return boot_wc_dstn_arr


def get_wc_m_out_of_n_boot_dstn(
    treatments: np.ndarray, outcomes: np.ndarray,
    price: float, cost: float, 
    power: float = 0.95, n_bootstraps: int = 500, 
    conditional_on_treated: bool = False,
    **kwargs
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
    emp_te = difference_in_mean(treatments=treatments, outcomes=outcomes)

    for boot_id in range(n_bootstraps):
        # bootstrap data
        boot_treatment_arr, boot_outcome_arr = stratified_bootstrap(
            treatments=treatments, outcomes=outcomes, boot_sample_size=m
        )

        # estimate treatment effect using bootstrap data
        boot_te = difference_in_mean(treatments=boot_treatment_arr, outcomes=boot_outcome_arr)

        # optimize 
        boot_decision, boot_targ_val_est = optimize(te=boot_te, price=price, cost=cost)
        
        # evaluate bootstrap targeting policy using empirical CATE estimates
        emp_boot_targ_val_est = objective_func(
            targ_decision=boot_decision,  # decision to be evaluated: bootstrapped decision
            te=emp_te,  # evaluation criterion: empirical treatment effect
            price=price, cost=cost
        )

        boot_decision_arr[boot_id] = boot_decision
        boot_wc_dstn_arr[boot_id] = boot_targ_val_est - emp_boot_targ_val_est

    if conditional_on_treated:
        return boot_wc_dstn_arr[boot_decision_arr]
    else:
        return boot_wc_dstn_arr


def get_wc_num_boot_dstn(
    treatments: np.ndarray, outcomes: np.ndarray,
    price: float, cost: float,
    n_bootstraps: int = 500, power: int = -0.45, conditional_on_treated: bool = False,
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
    emp_te = difference_in_mean(treatments=treatments, outcomes=outcomes)

    for boot_id in range(n_bootstraps):
        # stratified bootstrap
        boot_treatment_arr, boot_outcome_arr = stratified_bootstrap(
            treatments=treatments, outcomes=outcomes, boot_sample_size=sample_size
        )

        # estimate treatment effect with difference in mean estimator
        boot_te = difference_in_mean(
            treatments=boot_treatment_arr, outcomes=boot_outcome_arr
        )

        # solve optimization with boot_te as input
        boot_decision, _ = optimize(
            te=boot_te, price=price, cost=cost
        )

        # calculate error
        norm_error = np.sqrt(sample_size) * (boot_te - emp_te)

        # calculate perturbed treatment effect
        perturb_te = emp_te + epsilon_n * norm_error

        # evaluate boot_decision with perturbed treatment effect
        perturb_val_est = objective_func(
            targ_decision=boot_decision, te=perturb_te, 
            price=price, cost=cost
        )

        # evaluate bootstrap targeting policy using empirical CATE estimates
        emp_val_est = objective_func(
            targ_decision=boot_decision, te=emp_te, 
            price=price, cost=cost
        )

        boot_decision_arr[boot_id] = boot_decision
        boot_wc_dstn_arr[boot_id] = perturb_val_est - emp_val_est

    if conditional_on_treated:
        return boot_wc_dstn_arr[boot_decision_arr]
    else:
        return boot_wc_dstn_arr    


def bootstrap_correction_estimate(
    treatments: np.ndarray, outcomes: np.ndarray,
    price: float, cost: float, 
    bootstrap_method: str = 'standard', 
    conditional_on_treated: bool = False, stats: str = 'mean',
    **kwargs, 
) -> tuple:
    # fit the potential outcome model
    emp_te = difference_in_mean(treatments=treatments, outcomes=outcomes)

    # solve plugin problem
    plugin_decision, plugin_val_est = optimize(
        te=emp_te, price=price, cost=cost
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
        price=price, cost=cost, conditional_on_treated=conditional_on_treated, 
        **kwargs, 
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
    price: float, cost: float, **kwargs, 
) -> tuple: 
    # split the sample in to training and estimation
    train_treatments, est_treatments, train_outcomes, est_outcomes = train_test_split(
        treatments, outcomes, test_size=0.5
    )

    # estimate the treatment effect on the training sample
    train_te = difference_in_mean(treatments=train_treatments, outcomes=train_outcomes)

    # estimate the treatment effect on the estimation sample
    est_te = difference_in_mean(treatments=est_treatments, outcomes=est_outcomes)

    # optimize based on the training set
    train_decision, _ = optimize(te=train_te, price=price, cost=cost)

    # evaluate the decision on the estimation set
    val_est = objective_func(targ_decision=train_decision, te=est_te, price=price, cost=cost)

    return train_decision, val_est


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
        )

        # extract parameters
        for estimator in estimators_list:
            wc_arr = np.array([record[f'{estimator}_wc'] for record in result_records])
            wc_pct_arr = np.array([record[f'{estimator}_wc_pct'] for record in result_records])
            roi_wc_arr = np.array([record[f'{estimator}_roi_wc'] for record in result_records])

            result_dict[sample_size].update({
                f'{estimator}_wc_mean': wc_arr.mean(),
                f'{estimator}_wc_se': wc_arr.std() / np.sqrt(data_params['sample_size']),
                f'{estimator}_wc_pct_mean': wc_pct_arr.mean(),
                f'{estimator}_wc_pct_se': wc_pct_arr.std() / np.sqrt(data_params['sample_size']), 
                f'{estimator}_roi_wc_mean': roi_wc_arr.mean(),
                f'{estimator}_roi_wc_se': roi_wc_arr.std() / np.sqrt(data_params['sample_size']), 
            })

        if verbose:
            print(f'Finished sample size {sample_size} in {datetime.now() - start}')

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(result_dict, f)

    return result_dict


def empirical_bayes_estimate(
    treatments: np.ndarray, outcomes: np.ndarray, 
    price: float, cost: float, 
    n_degree: int = 5, n_knots: int = 5, bin_width: float = 0.2, smoothing: float = 0.5, 
    **kwargs
) -> tuple:
    # estimate targeting policy
    emp_te = difference_in_mean(treatments=treatments, outcomes=outcomes)
    plugin_decision, _ = optimize(te=emp_te, price=price, cost=cost)

    # estimate the posterior of the treatment effect and policy value
    target_est = EmpiricalBayes(
        n_degree=n_degree, n_knots=n_knots, bin_width=bin_width, smoothing=smoothing
    ).fit_predict(outcomes[treatments == 1]).mean()

    control_est = EmpiricalBayes(
        n_degree=n_degree, n_knots=n_knots, bin_width=bin_width, smoothing=smoothing
    ).fit_predict(outcomes[treatments == 0]).mean()

    val_est = objective_func(
        targ_decision=plugin_decision, 
        te=target_est - control_est,
        price=price, cost=cost
    )

    return plugin_decision, val_est


def normal_prior_bayes_estimate(
    treatments: np.ndarray, outcomes: np.ndarray,
    price: float, cost: float, 
    prior_mean: float, prior_std: float, 
    **kwargs, 
) -> tuple:
    # estimate targeting policy
    emp_te = difference_in_mean(treatments=treatments, outcomes=outcomes)
    plugin_decision, _ = optimize(te=emp_te, price=price, cost=cost)

    # calculate sampling variance
    sampling_var = np.var(outcomes) / treatments.size

    # calculate posterior effect
    weight = prior_std ** 2 / (prior_std ** 2 + sampling_var)
    posterior_te = weight * emp_te + (1 - weight) * prior_mean

    val_est = objective_func(
        targ_decision=plugin_decision, te=posterior_te, price=price, cost=cost
    )

    return plugin_decision, val_est


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
        )

        # extract parameters
        for estimator in estimators_list:
            wc_arr = np.array([record[f'{estimator}_wc'] for record in result_records])
            wc_pct_arr = np.array([record[f'{estimator}_wc_pct'] for record in result_records])
            roi_wc_arr = np.array([record[f'{estimator}_roi_wc'] for record in result_records])

            result_dict[sample_size].update({
                f'{estimator}_wc_mean': wc_arr.mean(),
                f'{estimator}_wc_se': wc_arr.std() / np.sqrt(data_params['sample_size']),
                f'{estimator}_wc_pct_mean': wc_pct_arr.mean(),
                f'{estimator}_wc_pct_se': wc_pct_arr.std() / np.sqrt(data_params['sample_size']), 
                f'{estimator}_roi_wc_mean': roi_wc_arr.mean(),
                f'{estimator}_roi_wc_se': roi_wc_arr.std() / np.sqrt(data_params['sample_size']), 
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

    for idx, noise_level in enumerate(noise_levels):
        if verbose:
            print(f'\nRunning noise level {noise_level} ({idx + 1}/{len(noise_levels)})')
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
            roi_wc_arr = np.array([record[f'{estimator}_roi_wc'] for record in result_records])

            result_dict[noise_level].update({
                f'{estimator}_wc_mean': wc_arr.mean(),
                f'{estimator}_wc_se': wc_arr.std() / np.sqrt(data_params['sample_size']),
                f'{estimator}_wc_pct_mean': wc_pct_arr.mean(),
                f'{estimator}_wc_pct_se': wc_pct_arr.std() / np.sqrt(data_params['sample_size']), 
                f'{estimator}_roi_wc_mean': roi_wc_arr.mean(),
                f'{estimator}_roi_wc_se': roi_wc_arr.std() / np.sqrt(data_params['sample_size']), 
            })

        if verbose:
            print(f'Finished noise level {noise_level} in {datetime.now() - start}')

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(result_dict, f)

    return result_dict


def sample_size_and_noise_level_test(
    sample_sizes: Iterable[int],
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
    result_dict = {(sample_size, noise_level): {} for sample_size in sample_sizes for noise_level in noise_levels}

    # attributes
    estimators_list = ['plugin'] + list(estimators_dict.keys())

    for sample_size in sample_sizes:
        for noise_level in noise_levels:
            if verbose:
                print(f'Running sample size {sample_size} and noise level {noise_level}')

            # update noise level in dgp params
            dgp_params['noise_std'] = noise_level

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
                roi_wc_arr = np.array([record[f'{estimator}_roi_wc'] for record in result_records])

                result_dict[(sample_size, noise_level)][f'{estimator}_wc_mean'] = wc_arr.mean()
                result_dict[(sample_size, noise_level)][f'{estimator}_wc_se'] = wc_arr.std() / np.sqrt(data_params['sample_size'])
                result_dict[(sample_size, noise_level)][f'{estimator}_wc_pct_mean'] = wc_pct_arr.mean()
                result_dict[(sample_size, noise_level)][f'{estimator}_wc_pct_se'] = wc_pct_arr.std() / np.sqrt(data_params['sample_size'])
                result_dict[(sample_size, noise_level)][f'{estimator}_roi_wc_mean'] = roi_wc_arr.mean()
                result_dict[(sample_size, noise_level)][f'{estimator}_roi_wc_se'] = roi_wc_arr.std() / np.sqrt(data_params['sample_size'])

        if verbose:
            print(f'Finished sample size {sample_size} and noise level {noise_level}')

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(result_dict, f)

    return result_dict

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