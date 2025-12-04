from joblib import Parallel, delayed
from typing import Union

import numpy as np
from scipy.stats import ttest_ind

from scipy.optimize import fsolve 
from scipy.stats import truncnorm
from statsmodels.discrete.discrete_model import Logit

from winners_curse.dgp import ABTest
from winners_curse.bayes_methods import *
from winners_curse.selective_inference import *
from winners_curse.bootstrap import bootstrap_correction_estimator

##########
# Estimation
##########

def estimate_treatment_effects(
    samples: list[np.ndarray], response_type: str, 
) -> tuple:
    """ 
    Estimate the treatment effects for each arm

    Params:
    -------
    samples: list[np.ndarray]
        List of samples for each arm, where each sample can have different shapes
    response_type: str
        The type of response variable. Options are 'continuous', 'bernoulli', or 'logit'.
        
    Returns:
    --------
    treatment_effects: np.ndarray, shape = (n_arms, )
        Estimated treatment effects for each arm
    treatment_effect_vars: np.ndarray, shape = (n_arms, )
        Estimated variance of treatment effects for each arm
    """
    # extract attributes 
    n_arms = len(samples)
    sample_sizes = [sample.shape[0] for sample in samples]

    if response_type in ['continuous', 'bernoulli']:
        # initialize list to store results
        treatment_effects = np.zeros(n_arms)
        treatment_effect_vars = np.zeros(n_arms)

        # iterate over each arm
        for arm_id, sample in enumerate(samples):
            # estimate treatment effects via sample mean 
            treatment_effects[arm_id] = sample.mean(axis=0)

            # compute variance of sample mean
            treatment_effect_vars[arm_id] = sample.var(axis=0) / sample_sizes[arm_id]
    else:  # response_type == 'logit'
        treatments = [i * np.ones(sample_sizes[i]) for i in range(n_arms)]
        treatments = np.concatenate(treatments, axis=0)
        outcomes = np.concatenate(samples, axis=0).flatten()

        exog_var = np.zeros((treatments.size, 2))
        exog_var[np.arange(treatments.size), treatments.astype(int)] = 1

        logit_model = Logit(endog=outcomes, exog=exog_var).fit(disp=0)

        # extract the estimated coefficients
        treatment_effects = logit_model.params[:, None]
        treatment_effect_vars = (logit_model.bse ** 2)[:, None]

    return treatment_effects, treatment_effect_vars


def _apply_optimizer(
    optimization_params: dict,
    treatment_effects: np.ndarray,
    treatment_effect_vars: np.ndarray,
) -> int:
    """
    Apply optimizer from optimization_params to treatment effects.
    
    Helper function to extract and apply the optimizer function consistently.
    
    Params:
    -------
    optimization_params: dict
        Dictionary containing 'optimizer' (callable) and 'params' (dict)
    treatment_effects: np.ndarray
        Treatment effects for each arm
    treatment_effect_vars: np.ndarray
        Treatment effect variances for each arm
        
    Returns:
    --------
    int 
        The index of the selected arm
    """
    optimizer = optimization_params['optimizer']
    optimize_params = optimization_params['params']
    
    return optimizer(
        treatment_effects=treatment_effects,
        treatment_effect_vars=treatment_effect_vars,
        **optimize_params
    )



##########
# Selection
##########

def select_higher_effect(
    treatment_effects: np.ndarray = None, treatment_effect_vars: np.ndarray = None,
) -> int:
    """ 
    For each experiment, select the treatment effect with the higher magnitude.

    Params:
    -------
    treatment_effects: np.ndarray, shape (n_arms, )
        An array representing the treatment effects for each arm.
    treatment_effect_vars: np.ndarray, shape (n_arms, )
        An array representing the variance of the treatment effects for each arm.

    Returns:
    --------
    int
        The index of the selected arm
    """
    assert treatment_effects is not None

    return np.argmax(treatment_effects)


def obj_func(
    selection: int, treatment_effects: np.ndarray, stats = 'mean', response_type: str = 'continuous',
) -> float: 
    """
    Objective function of the optimization problem: total treatment effect

    Params:
    -------
    selection: int
        The index of the selected arm
    treatment_effects: np.ndarray, shape (n_arms, )
        An array representing the treatment effects for each arm.
    stats: str
        The statistics to compute. Options are 'mean' and 'sum'.
    response_type: str
        The type of response variable. Options are 'continuous', 'bernoulli', or 'logit'.
    Returns:
    --------
    float
        The treatment effect of the selected arm.
    """
    assert stats in ['mean', 'sum']

    agg_func = {'mean': np.mean, 'sum': np.sum}[stats]

    treatment_effects = treatment_effects[selection]

    if response_type == 'logit':
        return agg_func(1 / (1 + np.exp(-treatment_effects)))
    else:  # response_type in ['continuous', 'bernoulli']
        return treatment_effects


def repeated_experiment(
    optimization_params: dict, 
    dgp_params: dict, 
    data_params: dict, 
    experiment_params: dict, 
    estimators_dict: dict, 
    verbose: bool = False,
    n_jobs: int = 1,
    logger = None,
) -> dict:
    """
    Perform a repeated experiment.
    
    Args:
        optimization_params: Optimization parameters
        dgp_params: Data generation process parameters
        data_params: Data parameters
        experiment_params: Experiment parameters
        estimators_dict: Dictionary of estimators to use
        verbose: Whether to show progress (default: False)
        n_jobs: Number of parallel jobs (default: 1)
        logger: Optional logger instance to capture parallel output (default: None)
        
    Returns:
        List of result dictionaries from each experiment run
    """ 
    # unpack the parameters
    n_repeats = experiment_params['n_repeats']
    sample_size = data_params['sample_size']
    # sample_size can be int or list[int] - handle both cases
    if isinstance(sample_size, int):
        # Convert single int to list for consistency
        sample_size_list = [sample_size]
    elif isinstance(sample_size, list):
        sample_size_list = sample_size
    else:
        raise TypeError(f"sample_size must be int or list[int], got {type(sample_size)}")

    # fixed vs. random parameter design 
    if 'fixed_params' in experiment_params.keys():
        is_fixed_params = experiment_params['fixed_params']
    else: 
        is_fixed_params = True 
    
    # create data generation process
    fixed_dgp = ABTest(**dgp_params)

    def run_single_experiment(experiment_id: int) -> dict:
        """
        Run a single experiment.
        """
        if not is_fixed_params:
            dgp_params['dgp_seed'] = experiment_id
            dgp = ABTest(**dgp_params)
        else:
            dgp = fixed_dgp

        # initialize result dictionary
        result_dict ={}

        # sample data - pass sample_size (can be int or list)
        samples = dgp.sample(sample_size=sample_size, seed=experiment_id)

        # estimate treatment effects
        empirical_estimates = estimate_treatment_effects(samples, response_type=dgp.response_type)
        emp_te_arr, emp_var_arr = empirical_estimates

        # run no correction estimator
        optimizer = optimization_params['optimizer']
        optimize_params = optimization_params['params']

        # select arm
        selection = optimizer(
            treatment_effects=emp_te_arr, 
            treatment_effect_vars=emp_var_arr,
            **optimize_params
        )

        # evaluate selection
        val_true = obj_func(selection=selection, treatment_effects=dgp.treatment_effects, response_type=dgp.response_type)
        val_est = obj_func(selection=selection, treatment_effects=emp_te_arr, response_type=dgp.response_type)

        # compute the Wald Closure
        wc = val_est - val_true

        # store results
        result_dict['selection'] = selection 
        result_dict['val_true'] = val_true
        result_dict['val_est'] = val_est
        result_dict['wc'] = wc
            
        result_dict['est_treatment_effects'] = emp_te_arr

        # run all other estimators for each selection methods
        for estimator_name in estimators_dict.keys():
            temp_estimator = estimators_dict[estimator_name]['estimator']
            temp_params = estimators_dict[estimator_name]['params']
            temp_selection, temp_est = temp_estimator(
                samples=samples, 
                empirical_estimates=empirical_estimates,
                optimization_params=optimization_params, 
                response_type=dgp.response_type,
                seed=experiment_id,
                **temp_params
            )

            # bookkeeping 
            result_dict[f'{estimator_name}_val_est'] = temp_est
            if temp_selection is None:
                result_dict[f'{estimator_name}_wc'] = temp_est - result_dict['val_true']
            else:
                temp_val_true = obj_func(temp_selection, dgp.treatment_effects, response_type=dgp.response_type)
                result_dict[f'{estimator_name}_selection'] = temp_selection
                result_dict[f'{estimator_name}_val_true'] = temp_val_true
                result_dict[f'{estimator_name}_val_est'] = temp_est
                result_dict[f'{estimator_name}_wc'] = temp_est - temp_val_true

        return result_dict
    
    # run experiments
    if logger:  # Use context manager if logger is provided
        from winners_curse.experiments import capture_parallel_output
        with capture_parallel_output(logger):
            result_records = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
                delayed(run_single_experiment)(experiment_id) for experiment_id in range(n_repeats)
            )
    else:
        result_records = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(run_single_experiment)(experiment_id) for experiment_id in range(n_repeats)
        )

    return result_records



def calculate_winners_curse_measures(
    result_records: list, 
    optimization_params: dict, 
    estimators_dict: dict, 
    truncate_outliers: bool = True, truncate_lb: float = -5.0, truncate_ub: float = 5.0,
) -> dict:
    # initialize results dict
    wc_measure_dict = {}

    # estimate treatment effects
    wc_measure_dict[f'effect_est_arr'] = np.array([result[f'est_treatment_effects'].flatten() for result in result_records])
    
    # calculate average winner's curse for no correction
    wc_measure_dict.update({
        f'nc_wc_arr': np.array([result[f'wc'] for result in result_records]), 
        f'nc_val_est_arr': np.array([result[f'val_est'] for result in result_records]), 
        f'nc_val_true_arr': np.array([result[f'val_true'] for result in result_records]),
        f'nc_selection_arr': np.array([result[f'selection'] for result in result_records]),
    })

    # iterate over estimators
    for estimator in estimators_dict.keys():
        temp_wc_arr = np.array([result[f'{estimator}_wc'] for result in result_records])
        temp_val_est_arr = np.array([result[f'{estimator}_val_est'] for result in result_records])

        # remove outliers
        if truncate_outliers:
            valid_idx = np.logical_and(temp_wc_arr > truncate_lb, temp_wc_arr < truncate_ub)
            temp_wc_arr = temp_wc_arr[valid_idx]
            temp_val_est_arr = temp_val_est_arr[valid_idx]                    

        wc_measure_dict.update({
            f'{estimator}_wc_arr': temp_wc_arr,
            f'{estimator}_val_est_arr': temp_val_est_arr,
        })

        if f'{estimator}_val_true' in result_records[0].keys():
            wc_measure_dict[f'{estimator}_val_true_arr'] = np.array([result[f'{estimator}_val_true'] for result in result_records])
            wc_measure_dict[f'{estimator}_selection_arr'] = np.array([result[f'{estimator}_selection'] for result in result_records])

    return wc_measure_dict


##########
# Bootstrap Correction
##########

def get_wc_stand_boot(
    samples: list[np.ndarray],
    optimization_params: dict,
    response_type: str,  
    n_bootstraps: int = 1000, 
    empirical_estimates: tuple = None,
    n_jobs: int = 1, 
    verbose: bool = False,
    seed: int = None, 
) -> dict:
    """
    Compute the bootstrap distribution of the Winner's Curse.
    """
    # Define wrappers for generic bootstrap
    def estimator_func(data):
        return estimate_treatment_effects(data, response_type=response_type)

    def optimizer_func(estimates):
        te_arr, var_arr = estimates
        return _apply_optimizer(optimization_params, te_arr, var_arr)

    def evaluator_func(selection, estimates):
        te_arr, _ = estimates
        return obj_func(selection, te_arr, response_type=response_type)

    def bootstrap_sampler_func(data, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return [np.random.choice(sample, size=len(sample), replace=True) for sample in data]

    _, corrected_val = bootstrap_correction_estimator(
        data=samples,
        estimator_func=estimator_func,
        optimizer_func=optimizer_func,
        evaluator_func=evaluator_func,
        bootstrap_sampler_func=bootstrap_sampler_func,
        empirical_estimates=empirical_estimates,
        n_bootstraps=n_bootstraps,
        n_jobs=n_jobs,
        verbose=verbose,
        seed=seed,
    )
    
    return corrected_val


def get_wc_m_out_of_n_boot(
    samples: list[np.ndarray],
    optimization_params: dict, 
    response_type: str,
    empirical_estimates: tuple = None,
    n_bootstraps: int = 1000, power: float = 0.95, 
    n_jobs: int = 1, 
    verbose: bool = False,
    seed: int = None, 
) -> dict:
    """
    Compute the bootstrap distribution of the Winner's Curse.
    """
    # parameter check 
    assert 0.0 < power < 1.0, 'The power must be in the range (0, 1).'

    # Define wrappers for m-out-of-n bootstrap
    def estimator_func(data):
        return estimate_treatment_effects(data, response_type=response_type)

    def optimizer_func(estimates):
        te_arr, var_arr = estimates
        return _apply_optimizer(optimization_params, te_arr, var_arr)

    def evaluator_func(selection, estimates):
        te_arr, _ = estimates
        return obj_func(selection, te_arr, response_type=response_type)

    def bootstrap_sampler_func(data, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # Handle different sample sizes per treatment - use m-out-of-n for each
        return [np.random.choice(sample, size=int(len(sample) ** power), replace=True) for sample in data]

    _, corrected_val = bootstrap_correction_estimator(
        data=samples,
        estimator_func=estimator_func,
        optimizer_func=optimizer_func,
        evaluator_func=evaluator_func,
        bootstrap_sampler_func=bootstrap_sampler_func,
        empirical_estimates=empirical_estimates,
        n_bootstraps=n_bootstraps,
        n_jobs=n_jobs,
        verbose=verbose,
        seed=seed,
    )
    
    return corrected_val


def get_wc_num_boot(
    samples: list[np.ndarray],
    optimization_params: dict, 
    response_type: str,
    empirical_estimates: tuple = None,
    n_bootstraps: int = 1000, power: float = -0.45, 
    n_jobs: int = 1, 
    verbose: bool = False,
    seed: int = None, 
) -> dict:
    """
    Compute the bootstrap distribution of the Winner's Curse.
    """
    # parameter checks
    assert 0.0 > power > -0.5, "Power must be between 0 and -0.5"

    # Define wrappers for numerical bootstrap
    def estimator_func(data):
        return estimate_treatment_effects(data, response_type=response_type)

    def optimizer_func(estimates):
        te_arr, var_arr = estimates
        return _apply_optimizer(optimization_params, te_arr, var_arr)

    def evaluator_func(selection, estimates):
        te_arr, _ = estimates
        return obj_func(selection, te_arr, response_type=response_type)

    def bootstrap_sampler_func(data, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # Handle different sample sizes per treatment
        return [np.random.choice(sample, size=len(sample), replace=True) for sample in data]

    def wc_func(boot_sel, boot_est, emp_est, evaluator_func):
        boot_te_arr, _ = boot_est
        emp_te_arr, _ = emp_est
        # Use average sample size for numerical bootstrap power calculation
        sample_sizes = np.array([sample.shape[0] for sample in samples])
        epsilons = sample_sizes ** power
        
        norm_error = np.sqrt(sample_sizes) * (boot_te_arr - emp_te_arr)
        perturbed_te_arr = emp_te_arr + epsilons * norm_error
        
        emp_val_est = evaluator_func(boot_sel, emp_est)
        
        # For perturbed value, we need to pass perturbed estimates to evaluator
        # But evaluator_func extracts te_arr from estimates tuple
        # So we wrap perturbed_te_arr in a tuple
        perturbed_est = (perturbed_te_arr, None)
        perturbed_val_est = evaluator_func(boot_sel, perturbed_est)
        
        return perturbed_val_est - emp_val_est

    _, corrected_val = bootstrap_correction_estimator(
        data=samples,
        estimator_func=estimator_func,
        optimizer_func=optimizer_func,
        evaluator_func=evaluator_func,
        bootstrap_sampler_func=bootstrap_sampler_func,
        empirical_estimates=empirical_estimates,
        wc_func=wc_func,
        n_bootstraps=n_bootstraps,
        n_jobs=n_jobs,
        verbose=verbose,
        seed=seed,
    )
    
    return corrected_val


def get_wc_double_boot(
    samples: list[np.ndarray],
    optimization_params: dict,
    response_type: str,
    n_outer_bootstraps: int = 100,
    n_inner_bootstraps: int = 100,
    empirical_estimates: tuple = None,
    n_jobs: int = 1,
    verbose: bool = False,
    seed: int = None,
) -> float:
    """
    Compute the double bootstrap corrected estimate.
    
    The double bootstrap includes two nested loops:
    1. Outer loop: Sample with replacement from original samples to create bootstrap samples
    2. Inner loop: For each outer bootstrap sample, apply standard winner's curse correction
    3. Average the winner's curse over all outer loops and subtract from empirical estimate
    
    Params:
    -------
    samples: list[np.ndarray]
        A list of arrays representing experimental outcomes for each arm.
    optimization_params: dict
        A dictionary containing the optimization method configuration.
    response_type: str
        The type of response variable. Options are 'continuous', 'bernoulli', or 'logit'.
    n_outer_bootstraps: int
        Number of outer bootstrap iterations (default: 100)
    n_inner_bootstraps: int
        Number of inner bootstrap iterations per outer loop (default: 100)
    emp_treatment_effects: np.ndarray, shape (n_arms, n_experiments)
        An array representing the empirical treatment effects for each arm.
    emp_treatment_vars: np.ndarray, shape (n_arms, n_experiments)
        An array representing the variance of the empirical treatment effects for each arm.
    n_jobs: int
        Number of parallel jobs to run (parallelizes outer loop)
    verbose: bool
        Whether to print progress information
    seed: int
        Random seed for reproducibility
    
    Returns:
    --------
    float
        Bootstrap-corrected policy value estimate
    """
    # set random seed
    if seed is not None:
        np.random.seed(seed)
    
    # compute empirical treatment effects
    if empirical_estimates is None:
        emp_te_arr, emp_var_arr = estimate_treatment_effects(samples, response_type=response_type)
    else:
        emp_te_arr, emp_var_arr = empirical_estimates
        
    # compute empirical selection and value
    emp_sel = _apply_optimizer(optimization_params, emp_te_arr, emp_var_arr)
    emp_val = obj_func(emp_sel, emp_te_arr, response_type=response_type)
    
    # Store sample sizes for each treatment
    sample_sizes = [len(sample) for sample in samples]
    
    def compute_outer_wc(outer_id, samples, emp_te_arr, emp_var_arr, optimization_params, response_type):
        """
        Compute winner's curse for one outer bootstrap iteration.
        
        This involves:
        1. Creating an outer bootstrap sample from the original data
        2. Running inner bootstrap loop on this outer sample
        3. Returning the average winner's curse from inner loop
        """
        # Draw outer bootstrap sample - resample each treatment independently
        outer_boot_samples = [np.random.choice(sample, size=len(sample), replace=True) for sample in samples]
        
        # Compute treatment effects on outer bootstrap sample
        outer_boot_te_arr, outer_boot_var_arr = estimate_treatment_effects(
            outer_boot_samples, response_type=response_type
        )
        
        # Inner bootstrap loop: compute winner's curse distribution
        inner_wc_list = []
        for inner_id in range(n_inner_bootstraps):
            # Draw inner bootstrap sample from outer bootstrap sample - resample each treatment independently
            inner_boot_samples = [np.random.choice(outer_sample, size=len(outer_sample), replace=True) for outer_sample in outer_boot_samples]
            
            # Compute treatment effects on inner bootstrap sample
            inner_boot_te_arr, inner_boot_var_arr = estimate_treatment_effects(
                inner_boot_samples, response_type=response_type
            )
            
            # Apply optimizer to select arm based on inner bootstrap sample
            inner_boot_selection = _apply_optimizer(optimization_params, inner_boot_te_arr, inner_boot_var_arr)
            
            # Evaluate selection on outer bootstrap sample (empirical) vs inner bootstrap sample
            outer_val_est = obj_func(selection=inner_boot_selection, treatment_effects=outer_boot_te_arr, response_type=response_type)
            inner_val_est = obj_func(selection=inner_boot_selection, treatment_effects=inner_boot_te_arr, response_type=response_type)
            
            # Compute winner's curse for this inner iteration
            inner_wc = inner_val_est - outer_val_est
            inner_wc_list.append(inner_wc)
        
        # Average winner's curse over inner bootstrap iterations
        outer_wc = np.mean(inner_wc_list)
        return outer_wc
    
    # Parallelize over outer bootstrap iterations
    outer_wc_records = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(compute_outer_wc)(outer_id, samples, emp_te_arr, emp_var_arr, optimization_params, response_type)
        for outer_id in range(n_outer_bootstraps)
    )
    
    # Convert to numpy array and compute the mean WC
    outer_wc_dstn = np.array([record for record in outer_wc_records])
    mean_wc = outer_wc_dstn.mean()
    
    # Return corrected value: empirical value - mean winner's curse
    return emp_val - mean_wc


def get_wc_double_boot_hall(
    samples: list[np.ndarray],
    optimization_params: dict,
    response_type: str,
    n_outer_bootstraps: int = 100,
    n_inner_bootstraps: int = 100,
    empirical_estimates: tuple = None,
    n_jobs: int = 1,
    verbose: bool = False,
    seed: int = None,
) -> float:
    """
    Double-bootstrap winner's-curse estimator (Hall-style bias correction).

    Step 1 (base statistic T(D)):
        - On the original data `samples`, run the standard bootstrap WC
          with `n_inner_bootstraps` resamples.
        - Let T_hat = mean of that WC distribution.

    Step 2 (outer bootstrap for bias of T):
        For r = 1,...,n_outer_bootstraps:

            (a) Draw an outer bootstrap dataset D^{*(r)} from `samples`.
            (b) On D^{*(r)}, run the *same* standard bootstrap WC with
                `n_inner_bootstraps` resamples to get T_hat_r.
            (c) Form the double-bootstrap WC estimate for this outer sample:
                    T_DB_r = 2 * T_hat - T_hat_r

    Return:
        float: the double-bootstrap bias-corrected policy value estimate,
               i.e., empirical_value - (2*T_hat - mean(T_hat_r)).
    """
    # global seed (like your other functions)
    if seed is not None:
        np.random.seed(seed)

    # Store sample sizes for each treatment (for reference, though not directly used in Hall method)
    # sample_sizes = [len(sample) for sample in samples]

    # ------------------------------------------------------------
    # Step 1: single-bootstrap WC estimate on the original data
    # ------------------------------------------------------------
    # Compute empirical estimates
    if empirical_estimates is None:
        emp_te_arr, emp_var_arr = estimate_treatment_effects(samples, response_type=response_type)
    else:
        emp_te_arr, emp_var_arr = empirical_estimates
    emp_sel = _apply_optimizer(optimization_params, emp_te_arr, emp_var_arr)
    emp_val = obj_func(emp_sel, emp_te_arr, response_type=response_type)
    
    # Run standard bootstrap to get corrected value
    base_corrected_val = get_wc_stand_boot(
        samples=samples,
        optimization_params=optimization_params,
        response_type=response_type,
        n_bootstraps=n_inner_bootstraps,
        empirical_estimates=empirical_estimates,
        n_jobs=1,
        verbose=False,
        seed=None,
    )
    # T_hat is the WC bias: emp_val - base_corrected_val
    T_hat = emp_val - base_corrected_val

    # ------------------------------------------------------------
    # Step 2: outer bootstrap – estimate bias of T_hat
    # ------------------------------------------------------------
    def compute_outer_T_DB(
        outer_id,
        samples,
        optimization_params,
        response_type,
        T_hat,
        empirical_estimates,
    ) -> float:
        # draw outer bootstrap dataset - resample each treatment independently
        outer_boot_samples = [np.random.choice(sample, size=len(sample), replace=True) for sample in samples]

        # single-bootstrap WC estimate on the outer dataset
        # Compute empirical estimates for outer bootstrap
        outer_emp_te_arr, outer_emp_var_arr = estimate_treatment_effects(
            outer_boot_samples, response_type=response_type
        )
        outer_emp_sel = _apply_optimizer(optimization_params, outer_emp_te_arr, outer_emp_var_arr)
        outer_emp_val = obj_func(outer_emp_sel, outer_emp_te_arr, response_type=response_type)
        
        # Run standard bootstrap on outer sample
        outer_corrected_val = get_wc_stand_boot(
            samples=outer_boot_samples,
            optimization_params=optimization_params,
            response_type=response_type,
            n_bootstraps=n_inner_bootstraps,
            empirical_estimates=empirical_estimates,
            n_jobs=1,
            verbose=False,
            seed=None,
        )
        # T_hat_r is the WC bias for outer sample
        T_hat_r = outer_emp_val - outer_corrected_val

        # Double-bootstrap bias-corrected WC for this outer sample
        T_DB_r = 2.0 * T_hat - T_hat_r
        return T_DB_r

    outer_T_DB_list = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(compute_outer_T_DB)(
            outer_id,
            samples,
            optimization_params,
            response_type,
            T_hat,
            empirical_estimates,
        )
        for outer_id in range(n_outer_bootstraps)
    )

    outer_T_DB_arr = np.asarray(outer_T_DB_list, dtype=float)
    mean_T_DB = outer_T_DB_arr.mean()
    
    # Return corrected value: empirical value - double bootstrap bias correction
    return emp_val - mean_T_DB



def _get_uncorrected_estimates(
    samples: list[np.ndarray],
    optimization_params: dict,
    response_type: str,
    empirical_estimates: tuple = None,
) -> tuple:
    """
    Helper function to compute empirical treatment effects and uncorrected estimates.
    """
    # compute empirical treatment effects
    if empirical_estimates is None:
        emp_te_arr, emp_var_arr = estimate_treatment_effects(samples, response_type=response_type)
    else:
        emp_te_arr, emp_var_arr = empirical_estimates

    # optimize selection
    selection = _apply_optimizer(optimization_params, emp_te_arr, emp_var_arr)

    # compute estimate without correction
    nc_est = obj_func(selection, emp_te_arr, response_type=response_type)
    
    return emp_te_arr, emp_var_arr, nc_est


def plugin_correction_estimate(
    samples: list[np.ndarray],
    optimization_params: dict,
    response_type: str,
    empirical_estimates: tuple = None,
    delta_tau: float = None,
    sigma: float = None,
    **kwargs,
) -> tuple:
    """
    Estimate the Winner's Curse using plugin correction.
    """
    assert len(samples) == 2, 'The number of treatments must be 2 for this method to work.'

    # compute empirical treatment effects
    emp_te_arr, emp_var_arr, nc_est = _get_uncorrected_estimates(
        samples, optimization_params, response_type, empirical_estimates
    )

    # compute the winner's curse
    if delta_tau is None:
        delta_tau = emp_te_arr[0] - emp_te_arr[1]
    if sigma is None:
        sigma = np.concatenate(samples, axis=0).std()
    sampling_vol = sigma * np.sqrt(2 / samples[0].shape[0])
    wc_est = sampling_vol * norm.pdf(delta_tau / sampling_vol)

    # compute the corrected estimate
    pc_est = nc_est - wc_est

    return None, pc_est


def plugin_with_integration_correction_estimate(
    samples: list[np.ndarray],
    optimization_params: dict,
    response_type: str,
    emp_treatment_effects: np.ndarray = None,
    emp_treatment_vars: np.ndarray = None,
    delta_tau: np.ndarray = None,
    sigma: float = None,
    **kwargs,
) -> tuple:
    """
    Estimate the Winner's Curse using plugin correction.
    """
    assert len(samples) == 2, 'The number of treatments must be 2 for this method to work.'

    # compute empirical treatment effects
    emp_te_arr, emp_var_arr, nc_est = _get_uncorrected_estimates(
        samples, optimization_params, response_type, emp_treatment_effects, emp_treatment_vars
    )

    # compute the winner's curse
    if delta_tau is None:
        delta_tau_arr = np.abs(samples[1] - samples[0])  # shape = (sample_size, )
    if sigma is None:
        sigma = np.concatenate(samples, axis=0).std()
    sampling_vol = sigma * np.sqrt(2 / samples[0].shape[0])
    wc_est = sampling_vol * norm.pdf(delta_tau_arr / sampling_vol).mean()

    # compute the corrected estimate
    pc_est = nc_est - wc_est

    return None, pc_est


def bootstrap_correction_estimate(
    samples: list[np.ndarray],
    optimization_params: dict, 
    response_type: str,
    empirical_estimates: tuple = None,
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
    empirical_estimates: tuple
        A tuple containing the empirical treatment effects and treatment effect variances.
    bootstrap_method: str
        The bootstrap method to use. Options are 'standard', 'm_out_of_n', 'numerical', 
        'adjusted', and 'double'.
    seed: int
        Random seed for reproducibility.
    **kwargs
        Additional keyword arguments for the bootstrap method.
    
    Returns:
    --------
    tuple: (None, boot_est)
        A tuple containing None for selection and the bootstrap-corrected policy value estimate.
    """
    # get the bootstrap-corrected estimate (all functions now return scalars)
    boot_est = {
        'standard': get_wc_stand_boot,
        'm_out_of_n': get_wc_m_out_of_n_boot,
        'numerical': get_wc_num_boot,
        'double': get_wc_double_boot_hall,
    }[bootstrap_method](
        samples, optimization_params, 
        response_type=response_type, 
        empirical_estimates=empirical_estimates,
        seed=seed,
    )

    return None, boot_est


##########
# Bayesian (Shrinkage) Estimation
##########

def bayes_estimate(
    samples: list[np.ndarray],
    optimization_params: dict, 
    response_type: str, 
    empirical_estimates: tuple = None,
    prior: str = 'normal', 
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
    empirical_estimates: tuple
        A tuple containing the empirical treatment effects and treatment effect variances.
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
    if empirical_estimates is None:
        emp_te_arr, emp_var_arr = estimate_treatment_effects(
        samples, response_type=response_type)
    else:
        emp_te_arr, emp_var_arr = empirical_estimates

    # create bootstrap samples
    n_arms = len(samples)

    # optimize selection based on point estimate
    selection = _apply_optimizer(optimization_params, emp_te_arr, emp_var_arr)
    
    # calculate posterior mean
    bayes_estimate_func = {'normal': bayes_normal}[prior]
    post_mean_arr = np.array([
        bayes_estimate_func(
            mle_treatment_effects=emp_te_arr[arm_id, :], 
            sampling_vars=emp_var_arr[arm_id, :]
        )
        for arm_id in range(n_arms)
    ])

    # evaluate policy value with posterior mean
    val_est = obj_func(selection=selection, treatment_effects=post_mean_arr, response_type=response_type)

    return None, val_est


def empirical_bayes_estimate(
    samples: list[np.ndarray],
    optimization_params: dict,
    response_type: str,
    empirical_estimates: tuple = None,
    prior: str = 'normal', 
    **kwargs
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
        The prior distribution. Options are 'tweedies', 'normal' and 'spike_slab'.

    Returns:
    --------
    tuple: selection_dict, val_est_dict
        A tuple containing the selection dictionary and the policy value estimate dictionary.
    """
    # parameter check
    assert prior in eb_function_dict.keys(), f'The prior must be {eb_function_dict.keys()}. {prior} is not supported.'

    # compute empirical treatment effects
    if empirical_estimates is None:
        emp_te_arr, emp_var_arr = estimate_treatment_effects(
        samples, response_type=response_type)
    else:
        emp_te_arr, emp_var_arr = empirical_estimates

    # create bootstrap samples
    n_arms = len(samples)

    # optimize selection based on point estimate
    selection = _apply_optimizer(optimization_params, emp_te_arr, emp_var_arr)

    # calculate posterior mean
    eb_func = eb_function_dict[prior]
    post_mean_arr = eb_func(mle_treatment_effects=emp_te_arr, sampling_vars=emp_var_arr)

    # evaluate policy value with posterior mean
    val_est = obj_func(selection=selection, treatment_effects=post_mean_arr, response_type=response_type)

    return None, val_est

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
    empirical_estimates: tuple = None,
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
    if empirical_estimates is None:
        emp_te_arr, emp_var_arr = estimate_treatment_effects(
        samples, response_type=response_type)
    else:
        emp_te_arr, emp_var_arr = empirical_estimates

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
    )  # shape = (n_arms, n_experiments)

    # evaluate policy value with posterior mean
    val_est_dict = {}
    for optimizer_key in optimization_params.keys():
        if optimizer_key == 'rank_and_select':
            val_est_dict['rank_and_select'] = obj_func(selection=selection_dict['rank_and_select'], treatment_effects=post_mean_arr, response_type=response_type)
        else:
            val_est_dict[optimizer_key] = np.nan

    return None, val_est_dict


def selective_inference_estimate(
    samples: list[np.ndarray], 
    optimization_params: dict,
    response_type: str,
    empirical_estimates: tuple = None,
    method: str = 'conditional',
    quantile: float = 0.5, 
    n_jobs: int = 1,
    verbose: bool = False, 
    seed: int = None,
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

    if seed is not None:
        np.random.seed(seed)

    # selective inference methods
    si_func = {
        'conditional': conditional_inference, 'hybrid': hybrid_inference
    }

    # compute empirical treatment effects
    if empirical_estimates is None:
        emp_te_arr, emp_var_arr = estimate_treatment_effects(
        samples, response_type=response_type)
    else:
        emp_te_arr, emp_var_arr = empirical_estimates

    # optimize selection
    selection = _apply_optimizer(optimization_params, emp_te_arr, emp_var_arr)

    # adjust for winner's curse for each experiment
    adjusted_selected_effect = si_func[method](
        mean_arr=emp_te_arr,
        std_arr=emp_var_arr ** 0.5,
        max_item_idx=selection, 
        quantile=quantile
    )[0]

    # change the selected effect in emp_te_arr to the adjusted selected effect
    emp_te_arr[selection] = adjusted_selected_effect  # shape = (n_arms, )

    # evaluate policy value with adjusted effects
    val_est = obj_func(
        selection=selection, 
        treatment_effects=emp_te_arr, 
        response_type=response_type
    )

    return None, val_est

##########
# Sample Splitting
##########

def sample_splitting_estimate(
    samples: list[np.ndarray], 
    optimization_params: dict, 
    response_type: str,
    estimation_split: float = 0.5,
    seed: int = None,
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
    seed: int
        The seed for the sample splitting. 
    Returns:
    --------
    tuple:
        A tuple containing the selection dictionary and the policy value estimate dictionary.
    """
    # parameter check 
    assert 0.0 < estimation_split < 1.0, 'The estimation split must be in the range (0, 1).'

    if seed is not None:
        np.random.seed(seed)

    # split data into estimation and evaluation sets
    sample_sizes = [sample.shape[0] for sample in samples]
    est_sizes = [int(sample_size * estimation_split) for sample_size in sample_sizes]
    est_indices = [np.random.choice(sample_size, size=est_size, replace=False) for sample_size, est_size in zip(sample_sizes, est_sizes)]
    eval_indices = [np.setdiff1d(np.arange(sample_size), est_indices) for sample_size, est_indices in zip(sample_sizes, est_indices)]

    # compute empirical treatment effects 
    est_te_arr, est_var_arr = estimate_treatment_effects([
        sample[est_indices[sample_id]] for sample_id, sample in enumerate(samples)
    ], response_type=response_type)

    # optimize selection based on point estimate
    selection = _apply_optimizer(optimization_params, est_te_arr, est_var_arr)

    # estimate treatment effects on evaluation set
    eval_te_arr, _ = estimate_treatment_effects([
        sample[eval_indices[sample_id]] for sample_id, sample in enumerate(samples)
    ], response_type=response_type)

    # evaluate policy value with posterior mean
    val_est = obj_func(selection=selection, treatment_effects=eval_te_arr, response_type=response_type)
    return selection, val_est


def jackknife_estimate(
    samples: list[np.ndarray], 
    optimization_params: dict, 
    response_type: str,
    n_jobs: int = 1,
    verbose: bool = False,
    **kwargs,
):
    """
    Perform jackknife estimation for treatment effect optimization.
    
    Params:
    -------
    samples: list[np.ndarray]
        A list of arrays representing experimental outcomes for each arm.
    optimization_params: dict
        A dictionary containing the optimization methods to evaluate.
    response_type: str
        The type of response variable. Options are 'continuous', 'bernoulli', or 'logit'.
    n_jobs: int
        The number of parallel jobs to run.
    verbose: bool
        Whether to display progress
        
    Returns:
    --------
    tuple:
        A tuple containing the selection dictionary and the policy value estimate dictionary.
        The selection dictionary is None because the jackknife estimator does not provide a unified consistent selection.
        The policy value estimate dictionary contains the policy value estimate for each optimizer.
    """
    assert all(sample.shape[0] == samples[0].shape[0] for sample in samples), 'The number of samples must be the same for all arms.'

    # jackknife estimation
    def jackknife_single_experiment(experiment_id):
        # create jackknife indices (leave out observation i)
        jackknife_indices = list(range(len(samples[0])))  # shape = (n_experiments,)
        jackknife_indices.pop(experiment_id)  # shape = (n_experiments - 1,)
        
        # compute empirical treatment effects on jackknife sample
        jackknife_te_arr, jackknife_var_arr = estimate_treatment_effects([
            sample[jackknife_indices, :] for sample in samples
        ], response_type=response_type)  # shape = (n_arms, n_experiments - 1)
        
        # optimize selection based on jackknife sample
        optimizer = optimization_params['optimizer']
        optimize_params = optimization_params['params']
        
        jackknife_selection = optimizer(
            treatment_effects=jackknife_te_arr,
            treatment_vars=jackknife_var_arr,
            **optimize_params
        )
        
        # estimate treatment effects on left-out observation
        left_out_te_arr, _ = estimate_treatment_effects([
            sample[[experiment_id], :] for sample in samples
        ], response_type=response_type)  # shape = (n_arms, 1)
        
        # evaluate policy value on left-out observation
        jackknife_val_est = obj_func(
            selection=jackknife_selection, 
            treatment_effects=left_out_te_arr, 
            response_type=response_type
        )

        return jackknife_val_est
    
    # perform jackknife estimation (leave-one-out)
    val_est_list = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(jackknife_single_experiment)(experiment_id)
        for experiment_id in range(len(samples[0]))
    )

    # compute jackknife estimates
    final_val_est = np.mean(val_est_list)
    
    return None, final_val_est


def kfold_cv_estimate(
    samples: list[np.ndarray], 
    optimization_params: dict, 
    response_type: str,
    k: int = 5,
    seed: int = None,
    n_jobs: int = 1,
    verbose: bool = False,
    **kwargs,
):
    """
    Perform k-fold cross-validation for treatment effect optimization.

    Params:
    -------
    samples: list[np.ndarray]
        A list of arrays representing experimental outcomes for each arm.
    optimization_params: dict
        A dictionary containing the optimization methods to evaluate.
    response_type: str
        The type of response variable. Options are 'continuous', 'bernoulli', or 'logit'.
    k: int
        The number of folds.
    seed: int
        The seed for the random number generator.
    n_jobs: int
        The number of parallel jobs to run.
    verbose: bool
        Whether to display progress
    kwargs: dict
        Additional keyword arguments to pass to the optimization methods.
        
    Returns:    
    --------
    tuple:
        A tuple containing the selection dictionary and the policy value estimate dictionary.
        The selection dictionary is None because the k-fold cross-validation estimator does not provide a unified consistent selection.
        The policy value estimate dictionary contains the policy value estimate for each optimizer.
    """
    # parameter check   
    assert k > 1, 'The number of folds must be greater than 1.'
    assert all(sample.shape[0] == samples[0].shape[0] for sample in samples), 'The number of samples must be the same for all arms.'

    if seed is not None:
        np.random.seed(seed)

    # split data into k folds
    sample_size = samples[0].shape[0]
    fold_indices = np.array_split(np.arange(sample_size), k)  # shape = (k, sample_size / k)

    def kfold_cv_single_experiment(fold_id):
        # compute empirical treatment effects on fold
        fold_te_arr, fold_var_arr = estimate_treatment_effects([    
            sample[fold_indices[fold_id], :] for sample in samples
        ], response_type=response_type)

        # optimize selection based on fold
        fold_selection_dict = {
            optimizer_name: optimizer_dict['optimizer'](
                treatment_effects=fold_te_arr,
                treatment_vars=fold_var_arr,
                **optimizer_dict['params']
            )
            for optimizer_name, optimizer_dict in optimization_params.items()
        }

        # estimate treatment effects on left-out fold
        left_out_indices = np.setdiff1d(np.arange(sample_size), fold_indices[fold_id])
        left_out_te_arr, _ = estimate_treatment_effects([
            sample[left_out_indices, :] for sample in samples
        ], response_type=response_type)

        # evaluate policy value on left-out fold
        kfold_cv_val_est_dict = {
            optimizer_name: obj_func(
                selection=fold_selection_dict[optimizer_name], 
                treatment_effects=left_out_te_arr, 
                response_type=response_type
            )
            for optimizer_name in optimization_params.keys()
        }

        return kfold_cv_val_est_dict

    # perform k-fold cross-validation
    val_est_dict_list = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(kfold_cv_single_experiment)(fold_id)
        for fold_id in range(k)
    )

    # compute k-fold cross-validation estimates
    final_val_est_dict = {}
    
    for optimizer_name in optimization_params.keys():
        final_val_est_dict[optimizer_name] = np.mean([item[optimizer_name] for item in val_est_dict_list])
    
    return None, final_val_est_dict