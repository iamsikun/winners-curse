import os 
import sys 
import pickle 
sys.path.insert(0, os.path.abspath('.'))

from joblib import Parallel, delayed
from typing import Union

import numpy as np
from scipy.stats import ttest_ind

from scipy.optimize import fsolve 
from scipy.stats import truncnorm
from statsmodels.discrete.discrete_model import Logit

from core.dgp import RCTs
from core.bayes_methods import *
from core.selective_inference import *

##########
# Estimation
##########

# def difference_in_means(
#     treated_sample: np.ndarray, control_sample: np.ndarray,
# ) -> tuple:
#     """
#     Estimate the treatment effect from an RCT using Difference-in-Means (DiM) with p-values.

#     Params:
#     -------
#     treated_sample: np.ndarray
#         An array of shape (sample_size, n_experiments) representing the treated group.
#     control_sample: np.ndarray
#         An array of shape (sample_size, n_experiments) representing the control group.
    
#     Returns:
#     --------
#     tuple
#         A tuple containing the estimated treatment effect, t-statistic, and p-value.
#         - The estimated treatment effect is an array of shape (n_experiments,).
#         - The t-statistic is an array of shape (n_experiments,).
#         - The p-value is an array of shape (n_experiments,).
#     """
#     # compute the difference in means
#     emp_te_arr = treated_sample.mean(axis=0) - control_sample.mean(axis=0)

#     # compute the t statistic and p-value
#     t_stat_arr, p_val_arr = ttest_ind(treated_sample, control_sample, axis=0)

#     return emp_te_arr, t_stat_arr, p_val_arr


def estimate_treatment_effects(
    samples: list[np.ndarray], response_type: str, 
) -> tuple:
    """ 
    Estimate the treatment effects for each arm

    Params:
    -------
    samples: list[np.ndarray]
        List of samples for each arm, where each sample is of shape (sample_size, n_experiments)
    response_type: str
        The type of response variable. Options are 'continuous', 'bernoulli', or 'logit'.
        
    Returns:
    --------
    treatment_effects: np.ndarray, shape = (n_arms, n_experiments)
        Estimated treatment effects for each arm
    sample_vars: np.ndarray, shape = (n_arms, n_experiments)
        Estimated variance of treatment effects for each arm
    """
    # extract attributes 
    n_arms = len(samples)
    # n_experiments = samples[0].shape[1]
    sample_size = samples[0].shape[0]

    if response_type in ['continuous', 'bernoulli']:
        # initialize list to store results
        te_list = [None] * n_arms 
        te_var_list = [None] * n_arms

        # iterate over each arm
        for arm_id, sample in enumerate(samples):
            # estimate treatment effects
            te_list[arm_id] = sample.mean(axis=0)  # shape = (n_experiments, )
            te_var_list[arm_id] = sample.var(axis=0) / sample_size # shape = (n_experiments, )

        # concatenate results to form arrays with shape = (n_arms, n_experiments)
        treatment_effects = np.stack(te_list, axis=0)
        sample_vars = np.stack(te_var_list, axis=0)
    else:  # response_type == 'logit'
        assert samples[0].shape[1] == 1, 'Logit response type only supports one experiment per arm.'

        treatments = [i * np.ones(sample_size) for i in range(n_arms)]
        treatments = np.concatenate(treatments, axis=0)
        outcomes = np.concatenate(samples, axis=0).flatten()

        exog_var = np.zeros((treatments.size, 2))
        exog_var[np.arange(treatments.size), treatments.astype(int)] = 1

        logit_model = Logit(endog=outcomes, exog=exog_var).fit(disp=0)

        # extract the estimated coefficients
        treatment_effects = logit_model.params[:, None]
        sample_vars = (logit_model.bse ** 2)[:, None]

    return treatment_effects, sample_vars



##########
# Selection
##########

def select_higher_effect(
    treatment_effects: np.ndarray = None, treatment_vars: np.ndarray = None,
) -> np.ndarray[int]:
    """ 
    For each experiment, select the treatment effect with the higher magnitude.

    Params:
    -------
    treatment_effects: np.ndarray, shape (n_arms, n_experiments)
        An array representing the treatment effects for each arm.
    treatment_var: np.ndarray, shape (n_arms, n_experiments)
        An array representing the variance of the treatment effects for each arm.

    Returns:
    --------
    np.ndarray[int], shape = (n_experiments,)
        An inteter array of shape (n_experiments,) where the element indicates the selected arm.
    """
    assert treatment_effects is not None

    return np.argmax(treatment_effects, axis=0)


# def select_significant_experiments_with_constant_control(
#     treatment_effects: np.ndarray = None, p_vals: np.ndarray = None, 
#     alpha: float = 0.05
# ) -> np.ndarray:
#     """
#     Select the experiments with positively significant treatment effects.

#     Params:
#     -------
#     treatment_effects: np.ndarray, shape (n_experiments,)
#         An array representing the treatment effects for each experiment.
#     p_vals: np.ndarray, shape (n_experiments,)
#         An array representing the p-values for each experiment.
#     alpha: float
#         The significance level.
    
#     Returns:
#     --------
#     np.ndarray
#         A boolean array of shape (n_experiments,) where True indicates that the experiment
#         has a positively significant treatment effect.
#     """
#     assert p_vals is not None
#     assert treatment_effects is not None
    
#     is_significant = p_vals < alpha
#     is_positive = treatment_effects > 0

#     return is_significant & is_positive

# def select_above_threshold_with_constant_control(
#     treatment_effects: np.ndarray = None, p_vals: np.ndarray = None,
#     threshold: float = 0.0
# ) -> np.ndarray:
#     """
#     Select the experiments with treatment effects above a threshold.

#     Params:
#     -------
#     treatment_effects: np.ndarray, shape (n_experiments,)
#         An array representing the treatment effects for each experiment.
#     p_vals: np.ndarray, shape (n_experiments,)
#         An array representing the p-values for each experiment.
#     threshold: float
#         The threshold value.
    
#     Returns:
#     --------
#     np.ndarray
#         A boolean array of shape (n_experiments,) where True indicates that the experiment
#         has a treatment effect above the threshold.
#     """
#     assert treatment_effects is not None

#     return treatment_effects > threshold

# def select_largest_effects_with_constant_control(
#     treatment_effects: np.ndarray = None, p_vals: np.ndarray = None,
#     k: Union[float, int] = 0.1, 
# ) -> np.ndarray: 
#     """
#     Select the experiments with the largest positive treatment effects.

#     Params:
#     -------
#     treatment_effects: np.ndarray, shape (n_experiments,)
#         An array representing the treatment effects for each experiment.
#     p_vals: np.ndarray, shape (n_experiments,)
#         An array representing the p-values for each experiment.
#     k: int or float
#         The number of largest treatment effects to select. If k is a float, then it is interpreted
#         as a proportion of the total number of experiments.
    
#     Returns:
#     --------
#     np.ndarray
#         A boolean array of shape (n_experiments,) where True indicates that the experiment
#         has one of the k largest positive treatment effects.
#     """
#     assert treatment_effects is not None

#     if isinstance(k, float):
#         assert 0.0 < k < 1.0
#         k = int(k * len(treatment_effects))

#     largest_indices = np.argsort(treatment_effects)[-k:]

#     selected = np.zeros_like(treatment_effects, dtype=bool)
#     selected[largest_indices] = True

#     is_positive = treatment_effects > 0
#     selected = selected & is_positive

#     return selected

# def select_bh_with_constant_control(
#     treatment_effects: np.ndarray = None, p_vals: np.ndarray = None,
#     fdr: float = 0.05
# ) -> np.ndarray:
#     """
#     Select experiments with statistically significant treatment effects using the Benjamini-Hochberg procedure.

#     Params:
#     -------
#     treatment_effects: np.ndarray, shape (n_experiments,)
#         An array representing the treatment effects for each experiment.
#     p_vals: np.ndarray, shape (n_experiments,)
#         An array representing the p-values for each experiment.
#     fdr: float
#         The false discovery rate.

#     Returns:
#     --------
#     np.ndarray
#         A boolean array of shape (n_experiments,) where True indicates that the experiment
#         has a statistically significant treatment effect.
#     """
#     assert treatment_effects is not None
#     assert p_vals is not None

#     # sort the p-values
#     sorted_indices = np.argsort(p_vals)
#     sorted_p_vals = p_vals[sorted_indices]

#     # compute the Benjamini-Hochberg critical value
#     n_experiments = len(p_vals)
#     ranks = np.arange(1, n_experiments + 1)
#     critical_values = fdr * (ranks / n_experiments)

#     # find the largest k such that p_i <= critical value
#     largest_k = np.argmax(sorted_p_vals <= critical_values)

#     # select the experiments with p_i <= critical value
#     selected = np.zeros_like(p_vals, dtype=bool)
#     selected[sorted_indices[:largest_k + 1]] = True

#     is_positive = treatment_effects > 0
#     selected = selected & is_positive

#     return selected


def obj_func(
    selection: np.ndarray, treatment_effects: np.ndarray, stats = 'mean'
) -> float: 
    """
    Objective function of the optimization problem: total treatment effect

    Params:
    -------
    selection: np.ndarray[int], shape (n_experiments,)
        An array representing the index of the selected arm for each experiment.
    treatment_effects: np.ndarray, shape (n_arms, n_experiments)
        An array representing the treatment effects for each arm and experiment.
    stats: str
        The statistics to compute. Options are 'mean' and 'sum'.
    Returns:
    --------
    float
        The avg treatment effect of the selected experiments.
    """
    assert stats in ['mean', 'sum']

    return {'mean': np.mean, 'sum': np.sum}[stats](treatment_effects[selection, np.arange(len(selection))])


def obj_func_with_constant_control(
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

    # fixed vs. random parameter design 
    if 'fixed_params' in experiment_params.keys():
        is_fixed_params = experiment_params['fixed_params']
    else: 
        is_fixed_params = True 
    
    # create data generation process
    fixed_dgp = RCTs(**dgp_params)

    def run_single_experiment(experiment_id: int) -> dict:
        """
        Run a single experiment.
        """
        if not is_fixed_params:
            dgp_params['dgp_seed'] = experiment_id
            dgp = RCTs(**dgp_params)
        else:
            dgp = fixed_dgp

        # initialize result dictionary
        result_dict ={}

        # sample data
        samples = dgp.sample(sample_size=sample_size, seed=experiment_id)

        # estimate treatment effects
        emp_te_arr, emp_var_arr = estimate_treatment_effects(samples, response_type=dgp.response_type)

        # run no correction estimator for each selection methods
        for optimizer_name in optimization_params.keys():
            optimizer = optimization_params[optimizer_name]['optimizer']
            optimize_params = optimization_params[optimizer_name]['params']

            # select experiments
            selection = optimizer(
                treatment_effects=emp_te_arr, 
                treatment_vars=emp_var_arr,
                **optimize_params
            )

            # evaluate selection
            val_true = obj_func(selection=selection, treatment_effects=dgp.treatment_effects)
            val_est = obj_func(selection=selection, treatment_effects=emp_te_arr)

            # compute the Wald Closure
            wc = val_est - val_true

            # store results
            result_dict[f'{optimizer_name}_selection'] = selection 
            result_dict[f'{optimizer_name}_val_true'] = val_true
            result_dict[f'{optimizer_name}_val_est'] = val_est
            result_dict[f'{optimizer_name}_wc'] = wc
            
        result_dict['est_treatment_effects'] = emp_te_arr

        # run all other estimators for each selection methods
        for estimator_name in estimators_dict.keys():
            temp_estimator = estimators_dict[estimator_name]['estimator']
            temp_params = estimators_dict[estimator_name]['params']
            temp_selection_dict, temp_est_dict = temp_estimator(
                samples=samples, 
                emp_treatment_effects=emp_te_arr,
                emp_treatment_vars=emp_var_arr,
                optimization_params=optimization_params, 
                response_type=dgp.response_type,
                # seed=experiment_id,
                **temp_params
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
    estimators_dict: dict, data_params: dict, 
    truncate_outliers: bool = True, truncate_lb: float = -5.0, truncate_ub: float = 5.0,
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

            # remove outliers
            if truncate_outliers:
                valid_idx = np.logical_and(temp_wc_arr > truncate_lb, temp_wc_arr < truncate_ub)
                temp_wc_arr = temp_wc_arr[valid_idx]
                temp_val_est_arr = temp_val_est_arr[valid_idx]                    

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
    samples: list[np.ndarray],
    optimization_params: dict,
    response_type: str,  
    n_bootstraps: int = 1000, 
    emp_treatment_effects: np.ndarray = None, 
    emp_treatment_vars: np.ndarray = None,
    n_jobs: int = 1, 
    verbose: bool = False,
    seed: int = None, **kwargs
) -> dict:
    """
    Compute the bootstrap distribution of the Winner's Curse.

    Params:
    -------
    samples: list[np.ndarray]
        A list of arrays representing experimental outcomes for each arm.
    optimization_params: dict
        A dictionary containing the optimization methods to evaluate.
    response_type: str
        The type of response variable. Options are 'continuous', 'bernoulli', or 'logit'.
    n_bootstraps: int
        The number of bootstrap samples to generate.
    emp_treatment_effects: np.ndarray, shape (n_arms, n_experiments)
        An array representing the empirical treatment effects for each arm.
    emp_treatment_vars: np.ndarray, shape (n_arms, n_experiments)
        An array representing the variance of the empirical treatment effects for each arm.
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
    if emp_treatment_effects is None or emp_treatment_vars is None:
        emp_te_arr, _ = estimate_treatment_effects(samples, response_type=response_type)
    else: 
        emp_te_arr = emp_treatment_effects

    # create bootstrap samples
    sample_size = samples[0].shape[0]

    def compute_wc(boot_id) -> float:
        """
        Compute the Winner's Curse for a single bootstrap sample.
        """
        # draw bootstrap samples
        boot_indices = np.random.choice(sample_size, size=sample_size, replace=True)
        boot_samples = [sample[boot_indices] for sample in samples]
        
        # selection
        boot_te_arr, boot_vars_arr = estimate_treatment_effects(boot_samples, response_type=response_type)

        # optimization
        wc_dict = {}
        for optimizer_name in optimization_params.keys():
            optimizer = optimization_params[optimizer_name]['optimizer']
            optimize_params = optimization_params[optimizer_name]['params']

            # select experiments
            boot_selection = optimizer(
                treatment_effects=boot_te_arr, 
                treatment_vars=boot_vars_arr,
                **optimize_params
            )

            # evaluate selection
            emp_val_est = obj_func(selection=boot_selection, treatment_effects=emp_te_arr)
            boot_val_est = obj_func(selection=boot_selection, treatment_effects=boot_te_arr)

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
    samples: list[np.ndarray],
    optimization_params: dict, 
    response_type: str,
    emp_treatment_effects: np.ndarray = None,
    emp_treatment_vars: np.ndarray = None,
    n_bootstraps: int = 1000, power: float = 0.95, 
    n_jobs: int = 1, 
    verbose: bool = False,
    seed: int = None, **kwargs
) -> dict:
    """
    Compute the bootstrap distribution of the Winner's Curse.

    Params:
    -------
    samples: list[np.ndarray]
        A list of arrays representing experimental outcomes for each arm.
    optimization_params: dict
        A dictionary containing the optimization methods to evaluate.
    response_type: str
        The type of response variable. Options are 'continuous', 'bernoulli', or 'logit'.
    emp_treatment_effects: np.ndarray, shape (n_arms, n_experiments)
        An array representing the empirical treatment effects for each arm.
    emp_treatment_vars: np.ndarray, shape (n_arms, n_experiments)
        An array representing the variance of the empirical treatment effects for each arm.
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
    if emp_treatment_effects is None or emp_treatment_vars is None:
        emp_te_arr, _ = estimate_treatment_effects(samples, response_type=response_type)
    else: 
        emp_te_arr = emp_treatment_effects

    # create bootstrap samples
    sample_size = samples[0].shape[0]

    def compute_wc(boot_id) -> float:
        """
        Compute the Winner's Curse for a single bootstrap sample.
        """
        # draw bootstrap samples
        boot_sample_size = int(sample_size ** power)
        boot_indices = np.random.choice(sample_size, size=boot_sample_size, replace=True)
        boot_samples = [sample[boot_indices] for sample in samples]
        
        # selection
        boot_te_arr, boot_var_arr = estimate_treatment_effects(boot_samples, response_type=response_type)

        # optimization
        wc_dict = {}
        for optimizer_name in optimization_params.keys():
            optimizer = optimization_params[optimizer_name]['optimizer']
            optimize_params = optimization_params[optimizer_name]['params']

            # select experiments
            boot_selection = optimizer(
                treatment_effects=boot_te_arr, 
                treatment_vars=boot_var_arr,
                 **optimize_params
            )

            # evaluate selection
            emp_val_est = obj_func(selection=boot_selection, treatment_effects=emp_te_arr)
            boot_val_est = obj_func(selection=boot_selection, treatment_effects=boot_te_arr)

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
    samples: list[np.ndarray],
    optimization_params: dict, 
    response_type: str,
    emp_treatment_effects: np.ndarray = None,
    emp_treatment_vars: np.ndarray = None,
    n_bootstraps: int = 1000, power: float = -0.45, 
    n_jobs: int = 1, 
    verbose: bool = False,
    seed: int = None, **kwargs
) -> dict:
    """
    Compute the bootstrap distribution of the Winner's Curse.

    Params:
    -------
    samples: list[np.ndarray]
        A list of arrays representing experimental outcomes for each arm.
    optimization_params: dict
        A dictionary containing the optimization methods to evaluate.
    response_type: str
        The type of response variable. Options are 'continuous', 'bernoulli', or 'logit'.
    emp_treatment_effects: np.ndarray, shape (n_arms, n_experiments)
        An array representing the empirical treatment effects for each arm.
    emp_treatment_vars: np.ndarray, shape (n_arms, n_experiments)
        An array representing the variance of the empirical treatment effects for each arm.
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
    if emp_treatment_effects is None or emp_treatment_vars is None:
        emp_te_arr, _= estimate_treatment_effects(samples, response_type=response_type)
    else: 
        emp_te_arr = emp_treatment_effects

    # create bootstrap samples
    sample_size = samples[0].shape[0]

    # compute perturbation parameter
    epsilon_n = sample_size ** power

    def compute_wc(boot_id) -> float:
        """
        Compute the Winner's Curse for a single bootstrap sample.
        """
        # draw bootstrap samples
        boot_indices = np.random.choice(sample_size, size=sample_size, replace=True)
        boot_samples = [sample[boot_indices] for sample in samples]
        
        # selection
        boot_te_arr, boot_var_arr = estimate_treatment_effects(boot_samples, response_type=response_type)

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
                treatment_vars=boot_var_arr,
                **optimize_params
            )

            # evaluate selection
            emp_val_est = obj_func(selection=boot_selection, treatment_effects=emp_te_arr)
            perturbed_val_est = obj_func(selection=boot_selection, treatment_effects=perturbed_te_arr)

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
    samples: list[np.ndarray],
    optimization_params: dict, 
    response_type: str,
    emp_treatment_effects: np.ndarray = None,
    emp_treatment_vars: np.ndarray = None,
    bootstrap_method: str = 'standard', 
    seed: int = None,
    **kwargs, 
) -> tuple:
    """
    Estimate the Winner's Curse using bootstrap correction.

    Params:
    -------
    samples: list[np.ndarray]
        A list of arrays representing experimental outcomes for each arm.
    optimization_params: dict
        A dictionary containing the optimization methods to evaluate.
    response_type: str
        The type of response variable. Options are 'continuous', 'bernoulli', or 'logit'.
    emp_treatment_effects: np.ndarray, shape (n_arms, n_experiments)
        An array representing the empirical treatment effects for each arm.
    emp_treatment_vars: np.ndarray, shape (n_arms, n_experiments)
        An array representing the variance of the empirical treatment effects for each arm.
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
    if emp_treatment_effects is None or emp_treatment_vars is None:
        emp_te_arr, emp_var_arr = estimate_treatment_effects(samples, response_type=response_type)
    else:
        emp_te_arr = emp_treatment_effects
        emp_var_arr = emp_treatment_vars

    # optimize selection
    selection_dict = {
        optimizer_name: optimizer_dict['optimizer'](
            treatment_effects=emp_te_arr, 
            treatment_vars=emp_var_arr,
            **optimizer_dict['params']
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
    }[bootstrap_method](samples, optimization_params, response_type=response_type, seed=seed, **kwargs)

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
    samples: list[np.ndarray],
    optimization_params: dict, 
    response_type: str, 
    emp_treatment_effects: np.ndarray = None,
    emp_treatment_vars: np.ndarray = None,
    prior: str = 'normal', **kwargs, 
) -> tuple:
    """ 
    Estimate policy value using normal prior Bayesian estimator.

    Params: 
    -------
    samples: list[np.ndarray]
        A list of arrays representing experimental outcomes for each arm.
    optimization_params: dict
        A dictionary containing the optimization methods to evaluate.
    response_type: str
        The type of response variable. Options are 'continuous', 'bernoulli', or 'logit'.
    emp_treatment_effects: np.ndarray, shape (n_arms, n_experiments)
        An array representing the empirical treatment effects for each arm.
    emp_treatment_vars: np.ndarray, shape (n_arms, n_experiments)
        An array representing the variance of the empirical treatment effects for each arm.
    prior: str
        The prior distribution. Options are 'normal'.

    Returns:
    --------
    tuple: selection_dict, val_est_dict
        A tuple containing the selection dictionary and the policy value estimate dictionary.
    """
    # parameter check 
    assert prior in ['normal'], f'The prior must be "normal". {prior} is not supported.'
    
    # compute empirical treatment effects 
    if emp_treatment_effects is None or emp_treatment_vars is None:
        emp_te_arr, emp_var_arr = estimate_treatment_effects(samples, response_type=response_type)
    else: 
        emp_te_arr = emp_treatment_effects
        emp_var_arr = emp_treatment_vars

    # create bootstrap samples
    n_arms = len(samples)

    # optimzie selection based on point estimate
    selection_dict = {
        optimizer_name: optimizer_dict['optimizer'](
            treatment_effects=emp_te_arr, 
            treatment_vars=emp_var_arr,
            **optimizer_dict['params']
        )
        for optimizer_name, optimizer_dict in optimization_params.items()
    }
    
    # calculate posterior mean
    bayes_estimate_func = {'normal': bayes_normal}[prior]
    post_mean_arr = np.array([
        bayes_estimate_func(
            mle_treatment_effects=emp_te_arr[arm_id, :], 
            sampling_vars=emp_var_arr[arm_id, :], **kwargs
        )
        for arm_id in range(n_arms)
    ])

    # evaluate policy value with posterior mean
    val_est_dict = {
        optimizer_name: obj_func(selection=selection_dict[optimizer_name], treatment_effects=post_mean_arr)
        for optimizer_name in optimization_params.keys()
    }

    return None, val_est_dict


def empirical_bayes_estimate(
    samples: list[np.ndarray],
    optimization_params: dict,
    response_type: str,
    emp_treatment_effects: np.ndarray = None,
    emp_treatment_vars: np.ndarray = None,
    prior: str = 'normal', **kwargs
) -> tuple:
    """ 
    Estimate policy value using empirical Bayesian estimator.

    Params:
    -------
    samples: list[np.ndarray]
        A list of arrays representing experimental outcomes for each arm.
    optimization_params: dict
        A dictionary containing the optimization methods to evaluate.
    response_type: str
        The type of response variable. Options are 'continuous', 'bernoulli', or 'logit'.
    emp_treatment_effects: np.ndarray, shape (n_arms, n_experiments)
        An array representing the empirical treatment effects for each arm.
    emp_treatment_vars: np.ndarray, shape (n_arms, n_experiments)
        An array representing the variance of the empirical treatment effects for each arm.
    prior: str
        The prior distribution. Options are 'normal' and 'spike_slab'.

    Returns:
    --------
    tuple: selection_dict, val_est_dict
        A tuple containing the selection dictionary and the policy value estimate dictionary.
    """
    # parameter check
    assert prior in ['normal', 'spike_slab'], 'The prior must be either "normal" or "spike_slab".'

    # compute empirical treatment effects 
    if emp_treatment_effects is None or emp_treatment_vars is None:
        emp_te_arr, emp_var_arr = estimate_treatment_effects(samples, response_type=response_type)
    else: 
        emp_te_arr = emp_treatment_effects
        emp_var_arr = emp_treatment_vars

    # create bootstrap samples
    n_arms = len(samples)

    # optimzie selection based on point estimate
    selection_dict = {
        optimizer_name: optimizer_dict['optimizer'](
            treatment_effects=emp_te_arr, 
            treatment_vars=emp_var_arr,
            **optimizer_dict['params']
        )
        for optimizer_name, optimizer_dict in optimization_params.items()
    }

    # calculate posterior mean
    eb_func = {
        'normal': empirical_bayes_normal, 
        'spike_slab': empirical_bayes_spike_slab
    }[prior]
    post_mean_arr = np.array([
        eb_func(
            mle_treatment_effects=emp_te_arr[arm_id, :], 
            sampling_vars=emp_var_arr[arm_id, :], **kwargs
        )  # shape = (n_experiments, )
        for arm_id in range(n_arms)
    ])  # shape = (n_arms, n_experiments)
    # try:
    #     eb_func(
    #         mle_treatment_effects=emp_te_arr, 
    #         sampling_vars=(treated_sample.var(axis=0) + control_sample.var(axis=0)) / treated_sample.shape[0],
    #         **kwargs
    #     )  # shape = (n_experiments, )
    # except:
    #     return None, {
    #         optimizer_name: np.nan for optimizer_name in optimization_params.keys()
    #     }

    # evaluate policy value with posterior mean
    val_est_dict = {
        optimizer_name: obj_func(selection=selection_dict[optimizer_name], treatment_effects=post_mean_arr)
        for optimizer_name in optimization_params.keys()
    }

    return None, val_est_dict

##########
# Selective Inference
##########


# def bayes_selection_adjusted_estimate(
#     samples: list[np.ndarray], 
#     optimization_params: dict,
#     selection_threshold: float, 
#     emp_treatment_effects: np.ndarray = None,
#     emp_treatment_vars: np.ndarray = None,
#     prior_mean: float = 0.0, prior_std: float = 1.0,
#     **kwargs, 
# ) -> dict:
#     """
#     Estimate the policy value using a normal prior Bayesian estimator with selection adjustment.

#     * This method only works for threshold-based selection. 

#     Params:
#     -------
#     samples: list[np.ndarray]
#         A list of arrays representing experimental outcomes for each arm.
#     optimization_params: dict
#         A dictionary containing the optimization methods to evaluate.
#     emp_treatment_effects: np.ndarray, shape (n_arms, n_experiments)
#         An array representing the empirical treatment effects for each arm.
#     emp_treatment_vars: np.ndarray, shape (n_arms, n_experiments)
#         An array representing the variance of the empirical treatment effects for each arm.
#     selection_threshold: float
#         The threshold for the selection event.
#     prior_mean: float
#         The mean of the normal prior.
#     prior_std: float
#         The standard deviation of the normal prior.

#     Returns:
#     --------
#     dict
#         A dictionary containing the selection dictionary and the policy value estimate dictionary.
#     """
#     assert 'threshold' in optimization_params, 'This method only works for threshold-based selection.'

#     # compute empirical treatment effects 
#     if emp_treatment_effects is None or emp_treatment_vars is None:
#         emp_te_arr, emp_var_arr = estimate_treatment_effects(samples)
#     else: 
#         emp_te_arr = emp_treatment_effects
#         emp_var_arr = emp_treatment_vars

#     # create bootstrap samples
#     n_arms = len(samples)

#     # optimzie selection based on point estimate
#     selection_dict = {
#         optimizer_name: optimizer_dict['optimizer'](
#             treatment_effects=emp_te_arr, 
#             treatment_vars=emp_var_arr,
#             **optimizer_dict['params']
#         )
#         for optimizer_name, optimizer_dict in optimization_params.items()
#     }

#     # calculate posterior mean 
#     post_mean_arr = np.array([bayes_normal_selection_adjusted(
#         mle_estimates=emp_te_arr[arm_id, :], 
#         selection=selection_dict['threshold'], 
#         sample_vars=emp_var_arr[arm_id, :],
#         threshold=selection_threshold,
#         prior_mean=prior_mean, prior_std=prior_std, 
#     ) for arm_id in range(n_arms)])  # shape = (n_arms, n_experiments)

#     # evaluate policy value with posterior mean
#     val_est_dict = {}
#     for optimizer_key in optimization_params.keys():
#         if optimizer_key == 'threshold':
#             val_est_dict['threshold'] = obj_func(selection=selection_dict['threshold'], treatment_effects=post_mean_arr)
#         else:
#             val_est_dict[optimizer_key] = np.nan

#     return None, val_est_dict


def empirical_bayes_selection_adjusted_estimate(
    samples: list[np.ndarray], 
    optimization_params: dict,
    response_type: str,
    emp_treatment_effects: np.ndarray = None,
    emp_treatment_vars: np.ndarray = None,
    **kwargs, 
) -> tuple:
    """
    Estimate the policy value using an empirical Bayesian estimator with selection adjustment.

    * This method only works for threshold-based selection. 

    Params:
    -------
    samples: list[np.ndarray]
        A list of arrays representing experimental outcomes for each arm.
    optimization_params: dict
        A dictionary containing the optimization methods to evaluate.
    response_type: str
        The type of response variable. Options are 'continuous', 'bernoulli', or 'logit'.
    emp_treatment_effects: np.ndarray, shape (n_arms, n_experiments)
        An array representing the empirical treatment effects for each arm.
    emp_treatment_vars: np.ndarray, shape (n_arms, n_experiments)
        An array representing the variance of the empirical treatment effects for each arm.

    Returns:
    --------
    dict
        A dictionary containing the selection dictionary and the policy value estimate dictionary.
    """
    assert 'rank_and_select' in optimization_params, 'This method only works for rank-and-select selection.'
    
    # compute empirical treatment effects 
    if emp_treatment_effects is None or emp_treatment_vars is None:
        emp_te_arr, emp_var_arr = estimate_treatment_effects(samples, response_type=response_type)
    else: 
        emp_te_arr = emp_treatment_effects
        emp_var_arr = emp_treatment_vars

    # optimzie selection based on point estimate
    selection_dict = {
        optimizer_name: optimizer_dict['optimizer'](
            treatment_effects=emp_te_arr, 
            treatment_vars=emp_var_arr,
            **optimizer_dict['params']
        )
        for optimizer_name, optimizer_dict in optimization_params.items()
    }

    # calculate posterior mean 
    post_mean_arr = empirical_bayes_spike_slab_selection_adjusted(
        mle_estimates=emp_te_arr, 
        selection=selection_dict['rank_and_select'], 
        sample_vars=emp_var_arr,
        **kwargs
    )  # shape = (n_arms, n_experiments)

    # evaluate policy value with posterior mean
    val_est_dict = {}
    for optimizer_key in optimization_params.keys():
        if optimizer_key == 'rank_and_select':
            val_est_dict['rank_and_select'] = obj_func(selection=selection_dict['rank_and_select'], treatment_effects=post_mean_arr)
        else:
            val_est_dict[optimizer_key] = np.nan

    return None, val_est_dict


def selective_inference_estimate(
    samples: list[np.ndarray], 
    optimization_params: dict,
    response_type: str,
    emp_treatment_effects: np.ndarray = None, 
    emp_treatment_vars: np.ndarray = None,
    method: str = 'conditional',
    quantile: float = 0.5, 
    n_jobs: int = 1,
    verbose: bool = False, 
    **kwargs,
):
    """ 
    Compute the conditional inference method from (Andrews et al. 2024, QJE)
    
    Params:
    -------
    samples: list[np.ndarray]
        A list of arrays representing experimental outcomes for each arm.
    optimization_params: dict
        A dictionary containing the optimization methods to evaluate.
    response_type: str
        The type of response variable. Options are 'continuous', 'bernoulli', or 'logit'
    emp_treatment_effects: np.ndarray, shape (n_arms, n_experiments)
        An array representing the empirical treatment effects for each arm.
    emp_treatment_vars: np.ndarray, shape (n_arms, n_experiments)
        An array representing the variance of the empirical treatment effects for each arm.
    method: str
        The method to use for selective inference. Options are 'conditional' and 'hybrid'.
    quantile: float
        The quantile for the conditional inference. Default is 0.5 for median unbiased estimator.
    n_jobs: int
        The number of parallel jobs to run.
    verbose: bool
        Whether to display progress
        
    Returns:
    --------
    tuple: selection_dict, boot_est_dict
        A tuple containing the selection dictionary and the bootstrap-corrected policy value estimate dictionary.
    """
    # parameter check
    assert method in ['conditional', 'hybrid'], 'The method must be either "conditional" or "hybrid".'
    assert quantile == 0.5, 'The quantile must be 0.5 for the median unbiased estimator.'

    # selective inference methods
    si_func = {
        'conditional': conditional_inference, 'hybrid': hybrid_inference
    }

    # compute empirical treatment effects
    if emp_treatment_effects is None or emp_treatment_vars is None:
        emp_te_arr, emp_var_arr = estimate_treatment_effects(samples, response_type=response_type)
    else:
        emp_te_arr = emp_treatment_effects  # shape = (n_arms, n_experiments)
        emp_var_arr = emp_treatment_vars  # shape = (n_arms, n_experiments)

    # optimize selection
    selection_dict = {
        optimizer_name: optimizer_dict['optimizer'](
            treatment_effects=emp_te_arr, 
            treatment_vars=emp_var_arr,
            **optimizer_dict['params']
        )
        for optimizer_name, optimizer_dict in optimization_params.items() 
        if optimizer_name == 'rank_and_select' 
    }

    # adjust for winner's curse for each experiment
    def adjust_single_experiment(experiment_id):
        # Extract the selected and unselected effects and variances
        selected_arm = selection_dict['rank_and_select'][experiment_id]

        result = si_func[method](
            mean_arr=emp_te_arr[:, experiment_id],
            std_arr=emp_var_arr[:, experiment_id] ** 0.5,
            max_item_idx=selected_arm, 
            quantile=quantile
        )

        return result[0]
    
    adjusted_selected_effect_arr = np.array(Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(adjust_single_experiment)(experiment_id)
        for experiment_id in range(emp_te_arr.shape[1])
    ))  # shape = (n_experiments,)

    # change the selected effect in emp_te_arr to the adjusted selected effect
    adjusted_emp_te_arr = emp_te_arr.copy()

    adjusted_emp_te_arr[
        selection_dict['rank_and_select'], np.arange(emp_te_arr.shape[1])
    ] = adjusted_selected_effect_arr  # shape = (n_arms, n_experiments)

    # evaluate policy value with adjusted effects
    val_est_dict = {}
    for optimizer_key in optimization_params.keys():
        if optimizer_key == 'rank_and_select':           
            # evaluate the policy value
            val_est_dict['rank_and_select'] = obj_func(
                selection=selection_dict['rank_and_select'], 
                treatment_effects=adjusted_emp_te_arr
            )

    return None, val_est_dict

##########
# Sample Splitting
##########

def sample_splitting_estimate(
    samples: list[np.ndarray], 
    optimization_params: dict, 
    response_type: str,
    estimation_split: float = 0.5,
    **kwargs,  
) -> tuple: 
    """
    Estimate the policy value using sample splitting.

    Params:
    -------
    samples: list[np.ndarray]
        A list of arrays representing experimental outcomes for each arm.
    optimization_params: dict
        A dictionary containing the optimization methods to evaluate.
    response_type: str
        The type of response variable. Options are 'continuous', 'bernoulli', or 'logit'.
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
    sample_size = samples[0].shape[0]
    est_size = int(sample_size * estimation_split)
    est_indices = np.random.choice(sample_size, size=est_size, replace=False)
    eval_indices = np.setdiff1d(np.arange(sample_size), est_indices)

    # compute empirical treatment effects 
    est_te_arr, est_var_arr = estimate_treatment_effects([
        sample[est_indices, :] for sample in samples
    ], response_type=response_type)

    # optimzie selection based on point estimate
    selection_dict = {
        optimizer_name: optimizer_dict['optimizer'](
            treatment_effects=est_te_arr, 
            treatment_vars=est_var_arr,
            **optimizer_dict['params']
        )
        for optimizer_name, optimizer_dict in optimization_params.items()
    }

    # estimate treatment effects on evaluation set
    eval_te_arr, _ = estimate_treatment_effects([
        sample[eval_indices, :] for sample in samples
    ], response_type=response_type)

    # evaluate policy value with posterior mean
    val_est_dict = {
        optimizer_name: obj_func(selection=selection_dict[optimizer_name], treatment_effects=eval_te_arr)
        for optimizer_name in optimization_params.keys()
    }
    return selection_dict, val_est_dict