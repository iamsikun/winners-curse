import os 
import sys 
import pickle 
sys.path.insert(0, os.path.abspath('.'))

from joblib import Parallel, delayed
from typing import Union

import numpy as np
from scipy.stats import ttest_ind

from core.dgp import RCTs
from core.bayes_methods import *

##########
# Estimation
##########

def difference_in_means(
    treated_sample: np.ndarray, control_sample: np.ndarray,
) -> tuple:
    """
    Estimate the treatment effect from an RCT using Difference-in-Means (DiM) with p-values.

    Params:
    -------
    treated_sample: np.ndarray
        An array of shape (n_samples, n_experiments) representing the treated group.
    control_sample: np.ndarray
        An array of shape (n_samples, n_experiments) representing the control group.
    
    Returns:
    --------
    tuple
        A tuple containing the estimated treatment effect, t-statistic, and p-value.
        - The estimated treatment effect is an array of shape (n_experiments,).
        - The t-statistic is an array of shape (n_experiments,).
        - The p-value is an array of shape (n_experiments,).
    """
    # compute the difference in means
    emp_te_arr = treated_sample.mean(axis=0) - control_sample.mean(axis=0)

    # compute the t statistic and p-value
    t_stat_arr, p_val_arr = ttest_ind(treated_sample, control_sample, axis=0)

    return emp_te_arr, t_stat_arr, p_val_arr

##########
# Selection
##########

def select_significant_experiments(
    treatment_effects: np.ndarray = None, p_vals: np.ndarray = None, 
    alpha: float = 0.05
) -> np.ndarray:
    """
    Select the experiments with positively significant treatment effects.

    Params:
    -------
    treatment_effects: np.ndarray, shape (n_experiments,)
        An array representing the treatment effects for each experiment.
    p_vals: np.ndarray, shape (n_experiments,)
        An array representing the p-values for each experiment.
    alpha: float
        The significance level.
    
    Returns:
    --------
    np.ndarray
        A boolean array of shape (n_experiments,) where True indicates that the experiment
        has a positively significant treatment effect.
    """
    assert p_vals is not None
    assert treatment_effects is not None
    
    is_significant = p_vals < alpha
    is_positive = treatment_effects > 0

    return is_significant & is_positive

def select_above_threshold(
    treatment_effects: np.ndarray = None, p_vals: np.ndarray = None,
    threshold: float = 0.0
) -> np.ndarray:
    """
    Select the experiments with treatment effects above a threshold.

    Params:
    -------
    treatment_effects: np.ndarray, shape (n_experiments,)
        An array representing the treatment effects for each experiment.
    p_vals: np.ndarray, shape (n_experiments,)
        An array representing the p-values for each experiment.
    threshold: float
        The threshold value.
    
    Returns:
    --------
    np.ndarray
        A boolean array of shape (n_experiments,) where True indicates that the experiment
        has a treatment effect above the threshold.
    """
    assert treatment_effects is not None

    return treatment_effects > threshold

def select_largest_effects(
    treatment_effects: np.ndarray = None, p_vals: np.ndarray = None,
    k: Union[float, int] = 0.1, 
) -> np.ndarray: 
    """
    Select the experiments with the largest positive treatment effects.

    Params:
    -------
    treatment_effects: np.ndarray, shape (n_experiments,)
        An array representing the treatment effects for each experiment.
    p_vals: np.ndarray, shape (n_experiments,)
        An array representing the p-values for each experiment.
    k: int or float
        The number of largest treatment effects to select. If k is a float, then it is interpreted
        as a proportion of the total number of experiments.
    
    Returns:
    --------
    np.ndarray
        A boolean array of shape (n_experiments,) where True indicates that the experiment
        has one of the k largest positive treatment effects.
    """
    assert treatment_effects is not None

    if isinstance(k, float):
        assert 0.0 < k < 1.0
        k = int(k * len(treatment_effects))

    largest_indices = np.argsort(treatment_effects)[-k:]

    selected = np.zeros_like(treatment_effects, dtype=bool)
    selected[largest_indices] = True

    is_positive = treatment_effects > 0
    selected = selected & is_positive

    return selected

def select_bh(
    treatment_effects: np.ndarray = None, p_vals: np.ndarray = None,
    fdr: float = 0.05
) -> np.ndarray:
    """
    Select experiments with statistically significant treatment effects using the Benjamini-Hochberg procedure.

    Params:
    -------
    treatment_effects: np.ndarray, shape (n_experiments,)
        An array representing the treatment effects for each experiment.
    p_vals: np.ndarray, shape (n_experiments,)
        An array representing the p-values for each experiment.
    fdr: float
        The false discovery rate.

    Returns:
    --------
    np.ndarray
        A boolean array of shape (n_experiments,) where True indicates that the experiment
        has a statistically significant treatment effect.
    """
    assert treatment_effects is not None
    assert p_vals is not None

    # sort the p-values
    sorted_indices = np.argsort(p_vals)
    sorted_p_vals = p_vals[sorted_indices]

    # compute the Benjamini-Hochberg critical value
    n_experiments = len(p_vals)
    ranks = np.arange(1, n_experiments + 1)
    critical_values = fdr * (ranks / n_experiments)

    # find the largest k such that p_i <= critical value
    largest_k = np.argmax(sorted_p_vals <= critical_values)

    # select the experiments with p_i <= critical value
    selected = np.zeros_like(p_vals, dtype=bool)
    selected[sorted_indices[:largest_k + 1]] = True

    is_positive = treatment_effects > 0
    selected = selected & is_positive

    return selected


def obj_func(
    selection: np.ndarray, treatment_effects: np.ndarray, 
) -> float:
    """
    Objective function of the optimization problem: total treatment effect

    Params:
    -------
    selection: np.ndarray, shape (n_experiments,)
        A boolean array representing the selected experiments.
    treatment_effects: np.ndarray, shape (n_experiments,)
        An array representing the treatment effects for each experiment.

    Returns:
    --------
    float
        The total treatment effect of the selected experiments.
    """
    return treatment_effects[selection].mean()

def repeated_experiment(
    optimization_params: dict, 
    dgp_params: dict, 
    data_params: dict, 
    experiment_params: dict, 
    estimators_dict: dict, 
    verbose: bool = False,
    n_jobs: int = 1,
) -> dict:
    """
    Perform a repeated experiment.
    """ 
    # unpack the parameters
    n_repeats = experiment_params['n_repeats']
    sample_size = data_params['sample_size']
    
    # create data generation process
    dgp = RCTs(**dgp_params)

    def run_single_experiment(experiment_id: int) -> dict:
        """
        Run a single experiment.
        """
        # initialize result dictionary
        result_dict ={}

        # sample data
        treated_sample, control_sample = dgp.sample(sample_size=sample_size, seed=experiment_id)

        # estimate treatment effects
        emp_te_arr, _, p_val_arr = difference_in_means(treated_sample, control_sample)


        # run no correction estimator for each selection methods
        for optimizer_name in optimization_params.keys():
            optimizer = optimization_params[optimizer_name]['optimizer']
            optimize_params = optimization_params[optimizer_name]['params']

            # select experiments
            selection = optimizer(
                treatment_effects=emp_te_arr, p_vals=p_val_arr, **optimize_params
            )

            # evaluate selection
            val_true = obj_func(selection, dgp.treatment_effects)
            val_est = obj_func(selection, emp_te_arr)

            # compute the Wald Closure
            wc = val_est - val_true

            # store results
            result_dict[f'{optimizer_name}_selection'] = selection 
            result_dict[f'{optimizer_name}_val_true'] = val_true
            result_dict[f'{optimizer_name}_val_est'] = val_est
            result_dict[f'{optimizer_name}_wc'] = wc
            
        result_dict['est_treatment_effects'] = emp_te_arr
        result_dict['p_values'] = p_val_arr

        # run all other estimators for each selection methods
        for estimator_name in estimators_dict.keys():
            temp_estimator = estimators_dict[estimator_name]['estimator']
            temp_params = estimators_dict[estimator_name]['params']
            temp_selection_dict, temp_est_dict = temp_estimator(
                treated_sample=treated_sample, control_sample=control_sample,
                optimization_params=optimization_params, **temp_params
            )

            if temp_selection_dict is None:
                for optimizer_name in temp_est_dict.keys():
                    result_dict[f'{optimizer_name}_{estimator_name}_val_est'] = temp_est_dict[optimizer_name]
                    result_dict[f'{optimizer_name}_{estimator_name}_wc'] = temp_est_dict[optimizer_name] - result_dict[f'{optimizer_name}_val_true']
            else:
                for optimizer_name in temp_selection_dict.keys():
                    temp_val_true = obj_func(temp_selection_dict[optimizer_name], dgp.treatment_effects)
                    result_dict[f'{optimizer_name}_{estimator_name}_selection'] = temp_selection_dict[optimizer_name]
                    result_dict[f'{optimizer_name}_{estimator_name}_val_true'] = temp_val_true
                    result_dict[f'{optimizer_name}_{estimator_name}_val_est'] = temp_est_dict[optimizer_name]
                    result_dict[f'{optimizer_name}_{estimator_name}_wc'] = temp_est_dict[optimizer_name] - temp_val_true

        return result_dict
    
    # run experiments
    if verbose:
        print(f'Running {n_repeats} experiments...')
    result_records = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(run_single_experiment)(experiment_id) for experiment_id in range(n_repeats)
    )

    return result_records


def calculate_winners_curse_measures(
    result_records: list, optimization_params: dict, 
    estimators_dict: dict, data_params: dict
) -> dict:
    # unpack parameters
    sample_size = data_params['sample_size']

    # initialize results dict
    wc_measure_dict = {}

    # iterate over optimizers
    for optimizer in optimization_params.keys():
        # calculate average winner's curse for no correction
        nc_wc_arr = np.array([result[f'{optimizer}_wc'] for result in result_records])
        val_true_arr = np.array([result[f'{optimizer}_val_true'] for result in result_records])
        val_est_arr = np.array([result[f'{optimizer}_val_est'] for result in result_records])
        wc_measure_dict.update({
            f'{optimizer}_nc_wc_arr': nc_wc_arr, f'{optimizer}_nc_val_est_arr': val_est_arr, f'{optimizer}_nc_val_true_arr': val_true_arr,
            f'{optimizer}_nc_wc_avg': np.nanmean(nc_wc_arr), f'{optimizer}_nc_wc_se': np.nanstd(nc_wc_arr) / sample_size ** 0.5, 
            f'{optimizer}_nc_val_est_avg': np.nanmean(val_est_arr), f'{optimizer}_nc_val_est_se': np.nanstd(val_est_arr) / sample_size ** 0.5,
            f'{optimizer}_nc_val_true_avg': np.nanmean(val_true_arr), f'{optimizer}_nc_val_true_se': np.nanstd(val_true_arr) / sample_size ** 0.5
        })

        for estimator in estimators_dict.keys():
            temp_wc_arr = np.array([result[f'{optimizer}_{estimator}_wc'] for result in result_records])
            temp_val_est_arr = np.array([result[f'{optimizer}_{estimator}_val_est'] for result in result_records])
            wc_measure_dict.update({
                f'{optimizer}_{estimator}_wc_arr': temp_wc_arr,
                f'{optimizer}_{estimator}_val_est_arr': temp_val_est_arr,
                f'{optimizer}_{estimator}_val_est_avg': np.nanmean(temp_val_est_arr),
                f'{optimizer}_{estimator}_val_est_se': np.nanstd(temp_val_est_arr) / sample_size ** 0.5,
                f'{optimizer}_{estimator}_wc_avg': np.nanmean(temp_wc_arr),
                f'{optimizer}_{estimator}_wc_se': np.nanstd(temp_wc_arr) / sample_size ** 0.5
            })

            if f'{optimizer}_{estimator}_val_true' in result_records[0].keys():
                temp_val_true_arr = np.array([result[f'{optimizer}_{estimator}_val_true'] for result in result_records])
                wc_measure_dict.update({
                    f'{optimizer}_{estimator}_val_true_avg': np.nanmean(temp_val_true_arr),
                    f'{optimizer}_{estimator}_val_true_se': np.nanstd(temp_val_true_arr) / sample_size ** 0.5, 
                })

    return wc_measure_dict


##########
# Bootstrap Correction
##########


def get_wc_boot_dstn(
    treated_sample: np.ndarray, control_sample: np.ndarray,
    optimization_params: dict, 
    n_bootstraps: int = 1000, 
    n_jobs: int = 1, 
    verbose: bool = False,
    seed: int = None, **kwargs
) -> dict:
    """
    Compute the bootstrap distribution of the Winner's Curse.

    Params:
    -------
    treated_sample: np.ndarray, shape (sample_size, n_experiments)
        An array representing the treated group.
    control_sample: np.ndarray, shape (sample_size, n_experiments)
        An array representing the control group.
    optimization_params: dict
        A dictionary containing the optimization methods to evaluate.
    n_bootstraps: int
        The number of bootstrap samples to generate.
    n_jobs: int
        The number of parallel jobs to run.
    verbose: bool
        Whether to display progress
    seed: int
        The random seed.
    
    Returns:
    --------
    dictionary.
        A dictionary containing the bootstrap distribution of the Winner's Curse for each optimization method.
    """
    # set random seed
    if seed is not None:
        np.random.seed(seed)

    # compute empirical treatment effects 
    emp_te_arr, _, _ = difference_in_means(treated_sample, control_sample)

    # create bootstrap samples
    sample_size = treated_sample.shape[0]

    def compute_wc(boot_id) -> float:
        """
        Compute the Winner's Curse for a single bootstrap sample.
        """
        # draw bootstrap samples
        boot_indices = np.random.choice(sample_size, size=sample_size, replace=True)
        boot_treated_sample = treated_sample[boot_indices]  # shape = (boot_sample_size, n_experiments)
        boot_control_sample = control_sample[boot_indices]  # shape = (boot_sample_size, n_experiments)
        
        # selection
        boot_te_arr, _, boot_p_val_arr = difference_in_means(boot_treated_sample, boot_control_sample)

        # optimization
        wc_dict = {}
        for optimizer_name in optimization_params.keys():
            optimizer = optimization_params[optimizer_name]['optimizer']
            optimize_params = optimization_params[optimizer_name]['params']

            # select experiments
            boot_selection = optimizer(
                treatment_effects=boot_te_arr, 
                p_vals=boot_p_val_arr, **optimize_params
            )

            # evaluate selection
            emp_val_est = obj_func(boot_selection, emp_te_arr)
            boot_val_est = obj_func(boot_selection, boot_te_arr)

            # compute the Wald Closure
            wc_dict[optimizer_name] = boot_val_est - emp_val_est

        return wc_dict
    
    # compute the Winner's Curse for each bootstrap sample
    boot_wc_records = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(compute_wc)(boot_id) for boot_id in range(n_bootstraps)
    )

    # compute the average Winner's Curse for each optimization method
    boot_wc_dstn_dict = {
        optimizer_name: np.array([record[optimizer_name] for record in boot_wc_records])
        for optimizer_name in optimization_params.keys()
    }

    return boot_wc_dstn_dict


def get_wc_m_out_of_n_boot_dstn(
    treated_sample: np.ndarray, control_sample: np.ndarray,
    optimization_params: dict, 
    n_bootstraps: int = 1000, power: float = 0.95, 
    n_jobs: int = 1, 
    verbose: bool = False,
    seed: int = None, **kwargs
) -> dict:
    """
    Compute the bootstrap distribution of the Winner's Curse.

    Params:
    -------
    treated_sample: np.ndarray, shape (sample_size, n_experiments)
        An array representing the treated group.
    control_sample: np.ndarray, shape (sample_size, n_experiments)
        An array representing the control group.
    optimization_params: dict
        A dictionary containing the optimization methods to evaluate.
    n_bootstraps: int
        The number of bootstrap samples to generate.
    power: float
        The power of the m-out-of-n bootstrap.
    n_jobs: int
        The number of parallel jobs to run.
    verbose: bool
        Whether to display progress
    seed: int
        The random seed.
    
    Returns:
    --------
    dictionary.
        A dictionary containing the bootstrap distribution of the Winner's Curse for each optimization method.
    """
    # parameter check 
    assert 0.0 < power < 1.0, 'The power must be in the range (0, 1).'

    # set random seed
    if seed is not None:
        np.random.seed(seed)

    # compute empirical treatment effects 
    emp_te_arr, _, _ = difference_in_means(treated_sample, control_sample)

    # create bootstrap samples
    sample_size = treated_sample.shape[0]

    def compute_wc(boot_id) -> float:
        """
        Compute the Winner's Curse for a single bootstrap sample.
        """
        # draw bootstrap samples
        boot_sample_size = int(sample_size ** power)
        boot_indices = np.random.choice(sample_size, size=boot_sample_size, replace=True)
        boot_treated_sample = treated_sample[boot_indices]  # shape = (boot_sample_size, n_experiments)
        boot_control_sample = control_sample[boot_indices]  # shape = (boot_sample_size, n_experiments)
        
        # selection
        boot_te_arr, _, boot_p_val_arr = difference_in_means(boot_treated_sample, boot_control_sample)

        # optimization
        wc_dict = {}
        for optimizer_name in optimization_params.keys():
            optimizer = optimization_params[optimizer_name]['optimizer']
            optimize_params = optimization_params[optimizer_name]['params']

            # select experiments
            boot_selection = optimizer(
                treatment_effects=boot_te_arr, 
                p_vals=boot_p_val_arr, **optimize_params
            )

            # evaluate selection
            emp_val_est = obj_func(boot_selection, emp_te_arr)
            boot_val_est = obj_func(boot_selection, boot_te_arr)

            # compute the Wald Closure
            wc_dict[optimizer_name] = boot_val_est - emp_val_est

        return wc_dict
    
    # compute the Winner's Curse for each bootstrap sample
    boot_wc_records = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(compute_wc)(boot_id) for boot_id in range(n_bootstraps)
    )

    # compute the average Winner's Curse for each optimization method
    boot_wc_dstn_dict = {
        optimizer_name: np.array([record[optimizer_name] for record in boot_wc_records])
        for optimizer_name in optimization_params.keys()
    }

    return boot_wc_dstn_dict


def get_wc_num_boot_dstn(
    treated_sample: np.ndarray, control_sample: np.ndarray,
    optimization_params: dict, 
    n_bootstraps: int = 1000, power: float = -0.45, 
    n_jobs: int = 1, 
    verbose: bool = False,
    seed: int = None, **kwargs
) -> dict:
    """
    Compute the bootstrap distribution of the Winner's Curse.

    Params:
    -------
    treated_sample: np.ndarray, shape (sample_size, n_experiments)
        An array representing the treated group.
    control_sample: np.ndarray, shape (sample_size, n_experiments)
        An array representing the control group.
    optimization_params: dict
        A dictionary containing the optimization methods to evaluate.
    n_bootstraps: int
        The number of bootstrap samples to generate.
    power: float
        The power parameter for the numerical bootstrap.
    n_jobs: int
        The number of parallel jobs to run.
    verbose: bool
        Whether to display progress
    seed: int
        The random seed.
    
    Returns:
    --------
    dictionary.
        A dictionary containing the bootstrap distribution of the Winner's Curse for each optimization method.
    """
    # parameter checks
    assert 0.0 > power > -0.5, "Power must be between 0 and -0.5"

    # set random seed
    if seed is not None:
        np.random.seed(seed)

    # compute empirical treatment effects 
    emp_te_arr, _, _ = difference_in_means(treated_sample, control_sample)

    # create bootstrap samples
    sample_size = treated_sample.shape[0]

    # compute perturbation parameter
    epsilon_n = sample_size ** power

    def compute_wc(boot_id) -> float:
        """
        Compute the Winner's Curse for a single bootstrap sample.
        """
        # draw bootstrap samples
        boot_indices = np.random.choice(sample_size, size=sample_size, replace=True)
        boot_treated_sample = treated_sample[boot_indices]  # shape = (boot_sample_size, n_experiments)
        boot_control_sample = control_sample[boot_indices]  # shape = (boot_sample_size, n_experiments)
        
        # selection
        boot_te_arr, _, boot_p_val_arr = difference_in_means(boot_treated_sample, boot_control_sample)

        # compute perturbed treatment effect estimates
        norm_error = np.sqrt(sample_size) * (boot_te_arr - emp_te_arr)
        perturbed_te_arr = emp_te_arr + epsilon_n * norm_error

        # optimization
        wc_dict = {}
        for optimizer_name in optimization_params.keys():
            optimizer = optimization_params[optimizer_name]['optimizer']
            optimize_params = optimization_params[optimizer_name]['params']

            # select experiments
            boot_selection = optimizer(
                treatment_effects=boot_te_arr, 
                p_vals=boot_p_val_arr, **optimize_params
            )

            # evaluate selection
            emp_val_est = obj_func(boot_selection, emp_te_arr)
            perturbed_val_est = obj_func(boot_selection, perturbed_te_arr)

            # compute the Wald Closure
            wc_dict[optimizer_name] = perturbed_val_est - emp_val_est

        return wc_dict
    
    # compute the Winner's Curse for each bootstrap sample
    boot_wc_records = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(compute_wc)(boot_id) for boot_id in range(n_bootstraps)
    )

    # compute the average Winner's Curse for each optimization method
    boot_wc_dstn_dict = {
        optimizer_name: np.array([record[optimizer_name] for record in boot_wc_records])
        for optimizer_name in optimization_params.keys()
    }

    return boot_wc_dstn_dict


def bootstrap_correction_estimate(
    treated_sample: np.ndarray, control_sample: np.ndarray, 
    optimization_params: dict, 
    bootstrap_method: str = 'standard', 
    **kwargs, 
) -> tuple:
    """
    Estimate the Winner's Curse using bootstrap correction.

    Params:
    -------
    treated_sample: np.ndarray, shape (sample_size, n_experiments)
        An array representing the treated group.
    control_sample: np.ndarray, shape (sample_size, n_experiments)
        An array representing the control group.
    optimization_params: dict
        A dictionary containing the optimization methods to evaluate.
    bootstrap_methods: str
        The bootstrap method to use. Options are 'standard', 'm_out_of_n', and 'numerical'.
    **kwargs
        Additional keyword arguments for the bootstrap method.
    
    Returns:
    --------
    tuple: selection_dict, boot_est_dict
        A tuple containing the selection dictionary and the bootstrap-corrected policy value estimate dictionary.
    """
    # compute empirical treatment effects
    emp_te_arr, _, p_val_arr = difference_in_means(treated_sample, control_sample)

    # optimize selection
    selection_dict = {
        optimizer_name: optimizer_dict['optimizer'](
            treatment_effects=emp_te_arr, p_vals=p_val_arr, **optimizer_dict['params']
        )
        for optimizer_name, optimizer_dict in optimization_params.items()
    }

    # compute estimate without correction
    nc_est_dict = {
        optimizer_name: obj_func(selection, emp_te_arr)
        for optimizer_name, selection in selection_dict.items()
    }

    # get the bootstrap distribution of the Winner's Curse for each selection method
    boot_dstn_dict = {
        'standard': get_wc_boot_dstn,
        'm_out_of_n': get_wc_m_out_of_n_boot_dstn,
        'numerical': get_wc_num_boot_dstn,
    }[bootstrap_method](
        treated_sample, control_sample, optimization_params, **kwargs
    )

    # compute the bootstrap-corrected policy value estimate for each selection method
    boot_est_dict = {
        optimizer_name: nc_est_dict[optimizer_name] - boot_dstn_dict[optimizer_name].mean()
        for optimizer_name in optimization_params.keys()
    }

    return None, boot_est_dict


##########
# Bayesian (Shrinkage) Estimation
##########

def bayes_estimate(
    treated_sample: np.ndarray, control_sample: np.ndarray, 
    optimization_params: dict, 
    prior: str = 'normal', **kwargs, 
) -> dict:
    """ 
    Estimate policy value using normal prior Bayesian estimator.

    Params: 
    """
    # parameter check 
    assert prior in ['normal'], f'The prior must be "normal". {prior} is not supported.'
    
    sample_size = treated_sample.shape[0]

    # point estimate
    emp_te_arr, _, p_val_arr = difference_in_means(treated_sample, control_sample)

    # optimzie selection based on point estimate
    selection_dict = {
        optimizer_name: optimizer_dict['optimizer'](
            treatment_effects=emp_te_arr, p_vals=p_val_arr, **optimizer_dict['params']
        )
        for optimizer_name, optimizer_dict in optimization_params.items()
    }

    # calculate sampling variance 
    sampling_var = (treated_sample.var(axis=0) + control_sample.var(axis=0)) / sample_size
    
    # calculate posterior mean
    posterior_mean = {'normal': bayes_normal}[prior](
        mle_treatment_effects=emp_te_arr, 
        sampling_vars=sampling_var,
        **kwargs
    )

    # evaluate policy value with posterior mean
    val_est_dict = {
        optimizer_name: obj_func(selection_dict[optimizer_name], posterior_mean)
        for optimizer_name in optimization_params.keys()
    }

    return None, val_est_dict


##########
# Bayesian Post-Selection Inference
##########


def bayes_selection_adjusted_estimate(
    treated_sample: np.ndarray, control_sample: np.ndarray,
    optimization_params: dict,
    selection_threshold: float, 
    prior_mean: float = 0.0, prior_std: float = 1.0,
    **kwargs, 
) -> dict:
    """
    Estimate the policy value using a normal prior Bayesian estimator with selection adjustment.

    * This method only works for threshold-based selection. 

    Params:
    -------
    treated_sample: np.ndarray
        An array of shape (n_samples, n_experiments) representing the treated group.
    control_sample: np.ndarray
        An array of shape (n_samples, n_experiments) representing the control group.
    optimization_params: dict
        A dictionary containing the optimization methods to evaluate.
    selection_threshold: float
        The threshold for the selection event.
    prior_mean: float
        The mean of the normal prior.
    prior_std: float
        The standard deviation of the normal prior.

    Returns:
    --------
    dict
        A dictionary containing the selection dictionary and the policy value estimate dictionary.
    """
    assert 'threshold' in optimization_params, 'This method only works for threshold-based selection.'

    sample_size = treated_sample.shape[0]

    # point estimate
    emp_te_arr, _, p_val_arr = difference_in_means(treated_sample, control_sample)

    # optimzie selection based on point estimate
    selection_dict = {
        optimizer_name: optimizer_dict['optimizer'](
            treatment_effects=emp_te_arr, p_vals=p_val_arr, **optimizer_dict['params']
        )
        for optimizer_name, optimizer_dict in optimization_params.items()
    }

    # calculate sampling variance 
    sampling_var = (treated_sample.var(axis=0) + control_sample.var(axis=0)) / sample_size

    # calculate posterior mean 
    post_te_arr = bayes_normal_selection_adjusted(
        mle_estimates=emp_te_arr, 
        selection=selection_dict['threshold'], 
        sample_vars=sampling_var,
        threshold=selection_threshold,
        prior_mean=prior_mean, prior_std=prior_std, 
    )

    # evaluate policy value with posterior mean
    val_est_dict = {}
    for optimizer_key in optimization_params.keys():
        if optimizer_key == 'threshold':
            val_est_dict['threshold'] = obj_func(selection_dict['threshold'], post_te_arr)
        else:
            val_est_dict[optimizer_key] = np.nan

    return None, val_est_dict



def empirical_bayes_estimate(
    treated_sample: np.ndarray, control_sample: np.ndarray,
    optimization_params: dict,
    prior: str = 'normal', **kwargs
) -> tuple:
    # parameter check
    assert prior in ['normal', 'spike_slab'], 'The prior must be either "normal" or "spike_slab".'

    # point estimate
    emp_te_arr, _, p_val_arr = difference_in_means(treated_sample, control_sample)

    # optimzie selection based on point estimate
    selection_dict = {
        optimizer_name: optimizer_dict['optimizer'](
            treatment_effects=emp_te_arr, p_vals=p_val_arr, **optimizer_dict['params']
        )
        for optimizer_name, optimizer_dict in optimization_params.items()
    }

    # calculate posterior mean
    try:
        post_mean_arr = {
            'normal': empirical_bayes_normal,
            'spike_slab': empirical_bayes_spike_slab,
        }[prior](
            mle_treatment_effects=emp_te_arr, 
            sampling_vars=(treated_sample.var(axis=0) + control_sample.var(axis=0)) / treated_sample.shape[0],
            **kwargs
        )  # shape = (n_experiments, )
    except:
        return None, {
            optimizer_name: np.nan for optimizer_name in optimization_params.keys()
        }

    # evaluate policy value with posterior mean
    val_est_dict = {
        optimizer_name: obj_func(selection_dict[optimizer_name], post_mean_arr)
        for optimizer_name in optimization_params.keys()
    }

    return None, val_est_dict


def empirical_bayes_selection_adjusted_estimate(
    treated_sample: np.ndarray, control_sample: np.ndarray,
    optimization_params: dict,
    selection_threshold: float,
    **kwargs, 
) -> tuple:
    """
    Estimate the policy value using an empirical Bayesian estimator with selection adjustment.

    * This method only works for threshold-based selection. 

    Params:
    -------
    treated_sample: np.ndarray
        An array of shape (n_samples, n_experiments) representing the treated group.
    control_sample: np.ndarray
        An array of shape (n_samples, n_experiments) representing the control group.
    optimization_params: dict
        A dictionary containing the optimization methods to evaluate.
    selection_threshold: float
        The threshold for the selection event.

    Returns:
    --------
    dict
        A dictionary containing the selection dictionary and the policy value estimate dictionary.
    """
    assert 'threshold' in optimization_params, 'This method only works for threshold-based selection.'

    sample_size = treated_sample.shape[0]

    # point estimate
    emp_te_arr, _, p_val_arr = difference_in_means(treated_sample, control_sample)

    # optimzie selection based on point estimate
    selection_dict = {
        optimizer_name: optimizer_dict['optimizer'](
            treatment_effects=emp_te_arr, p_vals=p_val_arr, **optimizer_dict['params']
        )
        for optimizer_name, optimizer_dict in optimization_params.items()
    }

    # calculate sampling variance 
    sampling_var = (treated_sample.var(axis=0) + control_sample.var(axis=0)) / sample_size

    # calculate posterior mean 
    post_te_arr = empirical_bayes_spike_slab_selection_adjusted(
        mle_estimates=emp_te_arr, 
        selection=selection_dict['threshold'], 
        sample_vars=sampling_var,
        threshold=selection_threshold,
        **kwargs
    )

    # evaluate policy value with posterior mean
    val_est_dict = {}
    for optimizer_key in optimization_params.keys():
        if optimizer_key == 'threshold':
            val_est_dict['threshold'] = obj_func(selection_dict['threshold'], post_te_arr)
        else:
            val_est_dict[optimizer_key] = np.nan

    return None, val_est_dict


##########
# Sample Splitting
##########

def sample_splitting_estimate(
    treated_sample: np.ndarray, control_sample: np.ndarray,
    optimization_params: dict, 
    estimation_split: float = 0.5,
    **kwargs,  
) -> tuple: 
    """
    Estimate the policy value using sample splitting.

    Params:
    -------
    treated_sample: np.ndarray
        An array of shape (n_samples, n_experiments) representing the treated group.
    control_sample: np.ndarray
        An array of shape (n_samples, n_experiments) representing the control group.
    optimization_params: dict
        A dictionary containing the optimization methods to evaluate.
    estimation_split: float
        The proportion of samples to use for estimation.

    Returns:
    --------
    tuple:
        A tuple containing the selection dictionary and the policy value estimate dictionary.
    """
    # parameter check 
    assert 0.0 < estimation_split < 1.0, 'The estimation split must be in the range (0, 1).'

    # split data into estimation and evaluation sets
    sample_size = treated_sample.shape[0]
    est_size = int(sample_size * estimation_split)
    est_indices = np.random.choice(sample_size, size=est_size, replace=False)
    eval_indices = np.setdiff1d(np.arange(sample_size), est_indices)

    # estimate treatment effects on estimation set
    est_te_arr, _, est_p_val_arr = difference_in_means(
        treated_sample[est_indices], control_sample[est_indices]
    )

    # optimzie selection based on point estimate
    selection_dict = {
        optimizer_name: optimizer_dict['optimizer'](
            treatment_effects=est_te_arr, p_vals=est_p_val_arr, **optimizer_dict['params']
        )
        for optimizer_name, optimizer_dict in optimization_params.items()
    }

    # estimate treatment effects on evaluation set
    eval_te_arr, _, _ = difference_in_means(
        treated_sample[eval_indices], control_sample[eval_indices]
    )
    # evaluate policy value with posterior mean
    val_est_dict = {
        optimizer_name: obj_func(selection_dict[optimizer_name], eval_te_arr)
        for optimizer_name in optimization_params.keys()
    }
    return selection_dict, val_est_dict