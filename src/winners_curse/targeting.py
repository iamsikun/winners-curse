import os 
import sys 
import pickle 
sys.path.insert(0, os.path.abspath('.'))

from joblib import Parallel, delayed
from typing import Union

# import warnings
# warnings.filterwarnings('error')

import numpy as np
from scipy.stats import ttest_ind
from statsmodels.api import OLS
import econml.grf as grf
import econml.dml as dml


from scipy.optimize import fsolve 
from scipy.stats import truncnorm
from scipy.special import expit


# visualization parameters
import matplotlib.pyplot as plt
import seaborn as sns
tick_label_size = 12
legend_label_size = 12
axis_label_size = 14
title_size = 18
plt.rcParams['font.family'] = 'serif'

from winners_curse.dgp import Targeting
from winners_curse.bayes_methods import *
from winners_curse.selective_inference import * 
from winners_curse.bootstrap import bootstrap_correction_estimator 


##########
# Visualization
##########
def plot_hte(
    dgp: Targeting, 
    model = None, 
    ci: bool = False, 
    x_lb=-1, x_ub=1
):
    # assert dgp.n_treatments == 2, "Currently only supports 2 treatments"

    fig, axes = plt.subplots(1, dgp.n_treatments, figsize=(10 * (dgp.n_treatments), 6.18), sharey=True)
    # plot the true HTE
    x = np.linspace(x_lb, x_ub, 1000).reshape(-1, 1)

    true_inc_te_arr, _ = dgp.predict_incremental_effect(x)
    est_inc_te_arr, _ = model.predict_incremental_effect(x)

    for treatment_id in range(dgp.n_treatments):
        axes[treatment_id].plot(
            x, true_inc_te_arr[:, treatment_id], label='True', color='black', linewidth=2
        )

        # plot the estimated HTE
        axes[treatment_id].plot(
            x, est_inc_te_arr[:, treatment_id], label='Estimated', color='red', linewidth=2
        )

        if ci:
            # plot the confidence intervals
            ci_lb, ci_ub = model.predict_incremental_effect_interval(x)
            axes[treatment_id].fill_between(
                x.flatten(), 
                ci_lb[:, treatment_id], 
                ci_ub[:, treatment_id], 
                alpha=0.2, 
                color='red'
            )

        axes[treatment_id].set_xlabel('Customer Feature', fontsize=axis_label_size)
        axes[treatment_id].set_ylabel('Treatment Effect', fontsize=axis_label_size)
        axes[treatment_id].set_title(f'Hetereogeneous Effects of Treatment {treatment_id + 1}', fontsize=title_size)
        axes[treatment_id].legend(fontsize=legend_label_size)
        axes[treatment_id].tick_params(axis='both', which='major', labelsize=tick_label_size)
        axes[treatment_id].grid()

    plt.tight_layout()
    plt.show()



##########
# Estimation
##########

class CausalForestDML(object):
    def __init__(self, n_treatments, **kwargs):
        self.n_treatments = n_treatments

        self.model = dml.CausalForestDML(**kwargs)
        
    def fit(self, X: np.ndarray, Y: np.ndarray, T: np.ndarray):
        # Ensure Y is the right type for econml
        # If Y appears to be binary (only 0 and 1), ensure it's integer type
        if Y.dtype != np.int64 and len(np.unique(Y)) == 2 and set(np.unique(Y)).issubset({0, 1}):
            Y = Y.astype(int)
        self.model.fit(X=X, Y=Y, T=T)
        return self 
    
    def predict_incremental_effect(self, X: np.ndarray) -> tuple:
        inference_result = self.model.const_marginal_effect_inference(X)

        # point estimate and standard error, both have shape = (sample_size, n_treatments)
        point, se = inference_result.pred, inference_result.pred_stderr
        return point, se ** 2

    def predict_incremental_effect_point(self, X: np.ndarray) -> np.ndarray:
        """Predict effects without computing inference statistics."""
        point = self.model.const_marginal_effect(X)
        return np.asarray(point).reshape(X.shape[0], self.n_treatments)
    
    def predict_incremental_effect_interval(self, X: np.ndarray, alpha: float=0.05):
        return self.model.const_marginal_effect_interval(X, alpha=alpha)


class KnownFunctionalForm(object):
    def __init__(self, n_treatments, fit_intercept: bool = True, **kwargs):
        self.n_treatments = n_treatments
        self.fit_intercept = fit_intercept

        # placeholder for the model
        self.models = [None] * self.n_treatments

    def fit(self, X: np.ndarray, Y: np.ndarray, T: np.ndarray):
        X_transformed = np.hstack([X**2, X])

        for i in range(self.n_treatments):
            endog_vars = Y[T == i+1]
            exog_vars = X_transformed[T == i+1]

            if self.fit_intercept:
                exog_vars = np.hstack([exog_vars, np.ones((exog_vars.shape[0], 1))])

            self.models[i] = OLS(endog=endog_vars, exog=exog_vars).fit()

        return self 

    def predict_incremental_effect(self, X: np.ndarray) -> tuple:
        X_transformed = np.hstack([X**2, X])

        if self.fit_intercept:
            X_transformed = np.hstack([X_transformed, np.ones((X_transformed.shape[0], 1))])

        effects_arr = np.zeros((X.shape[0], self.n_treatments))
        effects_var_arr = np.zeros((X.shape[0], self.n_treatments))
        
        for i in range(self.n_treatments):
            predictions = self.models[i].get_prediction(X_transformed)
            effects_arr[:, i] = predictions.predicted_mean
            effects_var_arr[:, i] = predictions.se_mean ** 2

        return effects_arr, effects_var_arr


def _draw_cv_valid_bootstrap_indices(
    treatments: np.ndarray,
    outcomes: np.ndarray,
    sample_size: int,
    discrete_treatment: bool = False,
    discrete_outcome: bool = False,
    n_splits: int = 2,
    max_attempts: int = 1000,
) -> np.ndarray:
    """Draw bootstrap indices with enough class support for cross-fitting.

    This is ordinary rejection sampling from the nonparametric bootstrap. It is
    only needed for discrete nuisance models: an m-out-of-n draw can otherwise
    contain fewer observations of a class than the cross-fitting procedure has
    folds, making the estimator undefined.
    """
    arrays_to_check = []
    if discrete_treatment:
        arrays_to_check.append(treatments)
    if discrete_outcome:
        arrays_to_check.append(outcomes)

    if arrays_to_check:
        discrete_strata = np.column_stack(arrays_to_check)
        _, stratum_ids = np.unique(
            discrete_strata, axis=0, return_inverse=True
        )
        n_strata = np.unique(stratum_ids).size
    else:
        stratum_ids = None
        n_strata = 0

    for _ in range(max_attempts):
        indices = np.random.choice(
            treatments.shape[0], size=sample_size, replace=True
        )
        counts = (
            np.bincount(stratum_ids[indices], minlength=n_strata)
            if stratum_ids is not None
            else np.array([sample_size])
        )
        if np.all(counts >= n_splits):
            return indices

    checked = "joint discrete treatment/outcome strata"
    raise RuntimeError(
        f"Could not draw a bootstrap sample with at least {n_splits} "
        f"observations per class for {checked} after {max_attempts} attempts."
    )

##########
# Selection
##########

# Optional per-repeat diagnostics an estimator may return as a third element.
# calculate_winners_curse_measures collects these into full-length arrays.
ESTIMATOR_DIAGNOSTIC_FIELDS = ('n_nonconverged', 'n_customers')


def obj_func(
    selection: np.ndarray, 
    cust_feautres: np.ndarray, 
    demand_model = None, 
    cust_treatment_effects: np.ndarray = None
) -> float:
    """
    Objective function to be maximized by the targeting algorithm. 
    This is the expected value of the treatment effect for the selected customers. 
    """
    if cust_treatment_effects is None:
        assert demand_model is not None, "Either treatment effects or demand model must be provided."
        cust_treatment_effects, _ = demand_model.predict_incremental_effect(cust_feautres)  # shape = (sample_size, n_treatments)
    else:
        assert cust_treatment_effects.shape[0] == selection.shape[0], "Treatment effects must be the same length as the selection vector."

    # calculate the expected value of the treatment effect for the selected customers
    return cust_treatment_effects[np.arange(selection.shape[0]), selection].mean()

def optimize(demand_model, cust_features: np.ndarray, cust_treatment_effects: np.ndarray = None) -> tuple:
    """ 
    Optimize targeting decision

    Params:
    -------
    demand_model: object
        demand model

    cust_features: np.ndarray, shape=(n_customers, n_features)
        features of the individuals

    cust_treatment_effects: np.ndarray, shape=(n_customers, n_treatments)
        incremental treatment effects

    Returns:
    --------
    tuple
        optimal decision, optimal value
    """
    if cust_treatment_effects is not None:
        assert cust_features.shape[0] == cust_treatment_effects.shape[0], "Features and treatment effects must have the same length"
    else:  # calculate objective values under each treatment
        cust_treatment_effects, _ = demand_model.predict_incremental_effect(cust_features)  # shape = (sample_size, n_treatments)

    # choose the better treatment for each customer 
    opt_decision = np.argmax(cust_treatment_effects, axis=1)  # shape = (sample_size, )

    # calculate the optimal value
    opt_value = obj_func(
        selection=opt_decision, 
        cust_feautres=cust_features, 
        demand_model=None,
        cust_treatment_effects=cust_treatment_effects
    )

    return opt_decision, opt_value


##########
# Experiments
##########
def replace_outliers(
    treatment_effects: np.ndarray, 
    threshold: float = 20.0, 
) -> np.ndarray:
    """
    Replace outliers from the treatment effects array with row mean. 
    Outliers are defined as values greater than the threshold.

    Params:
    -------
    boot_treatment_effects: np.ndarray, shape = (sample_size, n_treatments)
        Array of treatment effects to be checked for outliers
    threshold: float
        Threshold value to identify outliers. Default is 100.0.
        Values greater than this threshold are considered outliers.
    """
    # Check if the input is a numpy array
    if not isinstance(treatment_effects, np.ndarray):
        raise ValueError("Input must be a numpy array.")

    # Create a boolean mask for outliers
    outlier_mask = np.abs(treatment_effects) > threshold

    # Replace outliers with the mean of the array if specified
    if np.any(outlier_mask):
        # Get the row indices that have outliers
        row_indices = np.unique(np.where(outlier_mask)[0])
        
        # Replace outliers in each affected row with the row's non-outlier mean
        for row_idx in row_indices:
            # Get non-outlier values in this row
            row_values = treatment_effects[row_idx]
            row_outliers = outlier_mask[row_idx]
            non_outlier_values = row_values[~row_outliers]
            
            # If there are non-outlier values, use their mean
            # Otherwise use 0 (or another sensible default)
            if len(non_outlier_values) > 0:
                non_outlier_mean = non_outlier_values.mean()
            else:
                non_outlier_mean = 0
                
            # Replace outliers with the mean
            treatment_effects[row_idx, row_outliers] = non_outlier_mean

    return treatment_effects


def _compute_or_fit_treatment_effects(
    cust_features: np.ndarray,
    treatments: np.ndarray,
    outcomes: np.ndarray,
    targ_cust_features: np.ndarray,
    estimator,
    estimator_params: dict,
    emp_targ_te_arr: np.ndarray = None,
    emp_targ_var_arr: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit model and compute treatment effects or use provided effects.
    
    Helper function to handle the common pattern of fitting an estimator
    and computing treatment effects only when they are not already provided.
    
    Params:
    -------
    cust_features: np.ndarray
        Features of the training customers
    treatments: np.ndarray
        Treatment assignments for training customers
    outcomes: np.ndarray
        Observed outcomes for training customers
    targ_cust_features: np.ndarray
        Features of the target customers for optimization
    estimator: class
        Estimator model class to use
    estimator_params: dict
        Parameters for the estimator
    emp_targ_te_arr: np.ndarray, optional
        Pre-computed treatment effects for target customers
    emp_targ_var_arr: np.ndarray, optional
        Pre-computed treatment effect variances for target customers
        
    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        (treatment_effects, treatment_vars) for target customers
    """
    if emp_targ_te_arr is None:
        emp_model = estimator(**estimator_params).fit(X=cust_features, Y=outcomes, T=treatments)
        emp_targ_te_arr, emp_targ_var_arr = emp_model.predict_incremental_effect(targ_cust_features)
    elif emp_targ_var_arr is None:
        # If only treatment effects provided but not variances, fit model to get variances
        emp_model = estimator(**estimator_params).fit(X=cust_features, Y=outcomes, T=treatments)
        _, emp_targ_var_arr = emp_model.predict_incremental_effect(targ_cust_features)
    
    return emp_targ_te_arr, emp_targ_var_arr


def _predict_point_effects(model, targ_cust_features: np.ndarray) -> np.ndarray:
    """
    Predict incremental treatment effects without computing inference statistics.

    Use this wherever only point estimates are consumed. For causal forests the
    variance computation (``const_marginal_effect_inference``) costs several
    times a point prediction, so skipping it is free speed. Estimators without a
    point-only predictor fall back to the full inference path unchanged.

    Params:
    -------
    model: object
        Fitted estimator exposing ``predict_incremental_effect`` and optionally
        ``predict_incremental_effect_point``.
    targ_cust_features: np.ndarray
        Features of the target customers.

    Returns:
    --------
    np.ndarray
        Point estimates of the incremental treatment effects,
        shape = (n_target_customers, n_treatments).
    """
    if hasattr(model, 'predict_incremental_effect_point'):
        return model.predict_incremental_effect_point(targ_cust_features)

    targ_te_arr, _ = model.predict_incremental_effect(targ_cust_features)
    return targ_te_arr


def _apply_optimizer_targeting(
    optimization_params: dict,
    demand_model,
    cust_features: np.ndarray,
    cust_treatment_effects: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Apply optimizer from optimization_params to targeting problem.
    
    Helper function to extract and apply the optimizer function consistently.
    
    Params:
    -------
    optimization_params: dict
        Dictionary containing 'optimizer' (callable) and 'params' (dict)
    demand_model: object
        Demand model for prediction
    cust_features: np.ndarray
        Customer features
    cust_treatment_effects: np.ndarray
        Treatment effects for each customer
        
    Returns:
    --------
    tuple[np.ndarray, float]
        (selection, estimated_value) tuple
    """
    optimizer = optimization_params['optimizer']
    optimize_params = optimization_params['params']
    
    return optimizer(
        demand_model=demand_model,
        cust_features=cust_features,
        cust_treatment_effects=cust_treatment_effects,
        **optimize_params
    )



def repeated_experiment(
    optimization_params: dict, 
    dgp_params: dict, 
    data_params: dict, 
    experiment_params: dict, 
    estimators_dict: dict, 
    outlier_threshold: float = 20.0, 
    n_jobs: int = 1,
    verbose: int = 0,
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
        outlier_threshold: Threshold for outlier replacement (default: 20.0)
        n_jobs: Number of parallel jobs (default: 1)
        verbose: Verbosity level (default: 0)
        logger: Optional logger instance to capture parallel output (default: None)
        
    Returns:
        List of result dictionaries from each experiment run
    """ 
    # unpack the parameters
    n_repeats = experiment_params['n_repeats']
    sample_size = data_params['sample_size']
    targ_sample_size = data_params['targ_sample_size']
    estimator = experiment_params['estimator']['estimator']
    estimator_params = experiment_params['estimator']['params']

    if 'outlier_threshold' in experiment_params.keys():
        outlier_threshold = experiment_params['outlier_threshold']
    
    # create data generation process
    dgp = Targeting(**dgp_params)

    def run_single_experiment(experiment_id: int) -> dict:
        """
        Run a single experiment.
        """
        # initialize result dictionary
        result_dict ={}

        # sample training and out-of-sample data
        cust_features, treatments, outcomes = dgp.sample(sample_size=sample_size, seed=experiment_id)
        targ_cust_features = dgp.sample_individuals(sample_size=targ_sample_size, seed=experiment_id + 10000)

        # estimate treatment effects
        emp_estimator = estimator(**estimator_params).fit(
            X=cust_features, Y=outcomes, T=treatments
        )
        emp_targ_te_arr, emp_targ_var_arr = emp_estimator.predict_incremental_effect(targ_cust_features)  # shape = (sample_size, n_treatments)
        # replace outliers in the treatment effects array
        emp_targ_te_arr = replace_outliers(emp_targ_te_arr, threshold=outlier_threshold)

        # run no correction estimator for optimization method
        selection, val_est = _apply_optimizer_targeting(
            optimization_params, emp_estimator, targ_cust_features, emp_targ_te_arr
        )  # shape = (sample_size,)

        # evaluate selection
        val_true = obj_func(
            selection=selection, cust_feautres=targ_cust_features, demand_model=dgp
        )

        # compute the Wald Closure
        wc = val_est - val_true

        # store results
        result_dict['selection'] = selection 
        result_dict['val_true'] = val_true
        result_dict['val_est'] = val_est
        result_dict['wc'] = wc
            
        result_dict['est_treatment_effects'] = emp_targ_te_arr

        # run all other estimators for each selection methods
        for estimator_name in estimators_dict.keys():
            temp_estimator = estimators_dict[estimator_name]['estimator']
            temp_params = estimators_dict[estimator_name]['params']
            temp_result = temp_estimator(
                cust_features=cust_features, 
                treatments=treatments,
                outcomes=outcomes,
                optimization_params=optimization_params, 
                estimator=estimator,
                estimator_params=estimator_params,
                targ_cust_features=targ_cust_features,
                emp_targ_te_arr=emp_targ_te_arr,
                emp_targ_var_arr = emp_targ_var_arr,
                **temp_params
            )

            # Estimators return (selection, estimate) and may append a dict of
            # per-repeat diagnostics as a third element.
            temp_selection_dict, temp_est = temp_result[0], temp_result[1]
            temp_diagnostics = temp_result[2] if len(temp_result) > 2 else None

            if temp_diagnostics is not None:
                for diag_name, diag_value in temp_diagnostics.items():
                    result_dict[f'{estimator_name}_{diag_name}'] = diag_value

            if temp_selection_dict is None:
                result_dict[f'{estimator_name}_val_est'] = temp_est
                result_dict[f'{estimator_name}_wc'] = temp_est - result_dict['val_true']
            else:
                temp_val_true = obj_func(
                    selection=temp_selection_dict, 
                    cust_feautres=targ_cust_features, demand_model=dgp, 
                )
                result_dict[f'{estimator_name}_selection'] = temp_selection_dict
                result_dict[f'{estimator_name}_val_true'] = temp_val_true
                result_dict[f'{estimator_name}_val_est'] = temp_est
                result_dict[f'{estimator_name}_wc'] = temp_est - result_dict['val_true']

        return result_dict
    
    # Use context manager if logger is provided
    if logger:
        from winners_curse.experiments import capture_parallel_output
        with capture_parallel_output(logger):
            result_records = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
                delayed(run_single_experiment)(experiment_id) for experiment_id in range(n_repeats)
            )
    else:
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

    # calculate average winner's curse for no correction
    nc_wc_arr = np.array([result['wc'] for result in result_records])
    val_true_arr = np.array([result['val_true'] for result in result_records])
    val_est_arr = np.array([result['val_est'] for result in result_records])
    wc_measure_dict.update({
        'nc_wc_arr': nc_wc_arr, 'nc_val_est_arr': val_est_arr, 'nc_val_true_arr': val_true_arr,
        'nc_selection_arr': np.array([result['selection'] for result in result_records])
    })

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

        # Diagnostics stay at full length: they describe every repeat, including
        # the ones the outlier filter above drops from the wc arrays.
        for diag_name in ESTIMATOR_DIAGNOSTIC_FIELDS:
            record_key = f'{estimator}_{diag_name}'
            if record_key in result_records[0]:
                wc_measure_dict[f'{record_key}_arr'] = np.array(
                    [result[record_key] for result in result_records]
                )

        if f'{estimator}_val_true' in result_records[0].keys():
            temp_val_true_arr = np.array([result[f'{estimator}_val_true'] for result in result_records])
            wc_measure_dict.update({
                f'{estimator}_val_true_arr': temp_val_true_arr,
                f'{estimator}_selection_arr': np.array([result[f'{estimator}_selection'] for result in result_records])
            })

            # Overwrite winner's curse
            wc_measure_dict[f'{estimator}_wc_arr'] = wc_measure_dict[f'{estimator}_val_est_arr'] - wc_measure_dict[f'{estimator}_val_true_arr']

    return wc_measure_dict


##########
# Estimators
##########

def get_wc_boot_dstn(
    cust_features: np.ndarray,
    treatments: np.ndarray,
    outcomes: np.ndarray,
    targ_cust_features: np.ndarray,
    optimization_params: dict,
    estimator,
    estimator_params: dict,
    n_bootstraps: int = 1000,
    emp_targ_te_arr: np.ndarray = None,
    emp_targ_var_arr: np.ndarray = None,
    outlier_threshold: float = 20.0, 
    n_jobs: int = 1,
    verbose: bool = False,
    seed: int = None,
    **kwargs
) -> dict:
    """
    Compute the bootstrap distribution of the Winner's Curse for targeting applications.

    Params:
    -------
    cust_features: np.ndarray
        Features of the training customers
    treatments: np.ndarray
        Treatment assignments for training customers
    outcomes: np.ndarray
        Observed outcomes for training customers
    targ_cust_features: np.ndarray
        Features of the target customers for optimization
    optimization_params: dict
        Dictionary containing optimization methods to evaluate
    estimator: class
        Estimator model class to use (e.g., CausalForest)
    estimator_params: dict
        Parameters for the estimator
    n_bootstraps: int
        Number of bootstrap samples to generate
    emp_targ_te_arr: np.ndarray
        Pre-computed empirical treatment effects for target customers
    emp_targ_var_arr: np.ndarray
        Pre-computed empirical treatment effect variances for target customers
    outlier_threshold: float
        Threshold for outlier detection in treatment effects
    n_jobs: int
        Number of parallel jobs to run
    verbose: bool
        Whether to display progress
    seed: int
        Random seed
    
    Returns:
    --------
    dict
        Dictionary containing the bootstrap distribution of Winner's Curse for each optimization method
    """
    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    # Fit empirical model if not provided
    emp_targ_te_arr, emp_targ_var_arr = _compute_or_fit_treatment_effects(
        cust_features, treatments, outcomes, targ_cust_features,
        estimator, estimator_params, emp_targ_te_arr, emp_targ_var_arr
    )  # shape = (sample_size, n_treatments)
    
    # Sample size
    sample_size = cust_features.shape[0]

    # Define wrappers for generic bootstrap
    def estimator_func(data, **kwargs):
        cust_features, treatments, outcomes, targ_cust_features = data
        model = estimator(**estimator_params).fit(X=cust_features, Y=outcomes, T=treatments)
        # Only the point estimates are read downstream (optimizer_func and
        # evaluator_func), so skip the variance computation entirely.
        targ_te_arr = _predict_point_effects(model, targ_cust_features)
        targ_te_arr = replace_outliers(targ_te_arr, threshold=outlier_threshold)
        return (model, targ_te_arr, None)

    def optimizer_func(estimates, **kwargs):
        model, targ_te_arr, targ_var_arr = estimates
        selection, _ = _apply_optimizer_targeting(optimization_params, model, targ_cust_features, targ_te_arr)
        return selection

    def evaluator_func(selection, estimates, **kwargs):
        model, targ_te_arr, targ_var_arr = estimates
        return obj_func(selection=selection, cust_feautres=targ_cust_features, demand_model=None, cust_treatment_effects=targ_te_arr)

    def bootstrap_sampler_func(data, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        cust_features, treatments, outcomes, targ_cust_features = data
        sample_size = cust_features.shape[0]
        boot_indices = np.random.choice(sample_size, size=sample_size, replace=True)
        return (cust_features[boot_indices], treatments[boot_indices], outcomes[boot_indices], targ_cust_features)

    _, corrected_val = bootstrap_correction_estimator(
        data=(cust_features, treatments, outcomes, targ_cust_features),
        estimator_func=estimator_func,
        optimizer_func=optimizer_func,
        evaluator_func=evaluator_func,
        bootstrap_sampler_func=bootstrap_sampler_func,
        n_bootstraps=n_bootstraps,
        n_jobs=n_jobs,
        verbose=verbose,
        seed=seed,
        **kwargs
    )
    
    return corrected_val


def get_wc_moon_boot_dstn(
    cust_features: np.ndarray,
    treatments: np.ndarray,
    outcomes: np.ndarray,
    targ_cust_features: np.ndarray,
    optimization_params: dict,
    estimator,
    estimator_params: dict,
    n_bootstraps: int = 1000,
    emp_targ_te_arr: np.ndarray = None,
    emp_targ_var_arr: np.ndarray = None,
    outlier_threshold: float = 20.0,
    power: float = 0.95,
    n_jobs: int = 1,
    verbose: bool = False,
    seed: int = None,
    **kwargs
) -> dict:
    """
    Compute the scaled m-out-of-n bootstrap correction for targeting.

    Each bootstrap WC draw is multiplied by sqrt(m/N) to rescale from the
    m-observation noise level back to the N-observation noise level.
    """
    # parameter check
    assert 0.0 < power < 1.0, 'The power must be in the range (0, 1).'

    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    # Fit empirical model if not provided
    emp_targ_te_arr, emp_targ_var_arr = _compute_or_fit_treatment_effects(
        cust_features, treatments, outcomes, targ_cust_features,
        estimator, estimator_params, emp_targ_te_arr, emp_targ_var_arr
    )

    # Sample size
    sample_size = cust_features.shape[0]
    boot_sample_size = int(sample_size ** power)

    # Define wrappers for m-out-of-n bootstrap
    def estimator_func(data, **kwargs):
        cust_features, treatments, outcomes, targ_cust_features = data
        model = estimator(**estimator_params).fit(X=cust_features, Y=outcomes, T=treatments)
        if hasattr(model, 'predict_incremental_effect_point'):
            targ_te_arr = model.predict_incremental_effect_point(targ_cust_features)
            targ_var_arr = None
        else:
            targ_te_arr, targ_var_arr = model.predict_incremental_effect(targ_cust_features)
        targ_te_arr = replace_outliers(targ_te_arr, threshold=outlier_threshold)
        return (model, targ_te_arr, targ_var_arr)

    def optimizer_func(estimates, **kwargs):
        model, targ_te_arr, targ_var_arr = estimates
        selection, _ = _apply_optimizer_targeting(optimization_params, model, targ_cust_features, targ_te_arr)
        return selection

    def evaluator_func(selection, estimates, **kwargs):
        model, targ_te_arr, targ_var_arr = estimates
        return obj_func(selection=selection, cust_feautres=targ_cust_features, demand_model=None, cust_treatment_effects=targ_te_arr)

    def bootstrap_sampler_func(data, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        cust_features, treatments, outcomes, targ_cust_features = data
        sample_size = cust_features.shape[0]
        boot_sample_size = int(sample_size ** power)
        cv = estimator_params.get('cv', 2)
        n_splits = cv if isinstance(cv, int) and cv > 1 else 1
        boot_indices = _draw_cv_valid_bootstrap_indices(
            treatments=treatments,
            outcomes=outcomes,
            sample_size=boot_sample_size,
            discrete_treatment=estimator_params.get('discrete_treatment', False),
            discrete_outcome=estimator_params.get('discrete_outcome', False),
            n_splits=n_splits,
        )
        return (cust_features[boot_indices], treatments[boot_indices], outcomes[boot_indices], targ_cust_features)

    scale_factor = np.sqrt(boot_sample_size / sample_size)

    def wc_func(boot_sel, boot_est, emp_est, evaluator_func, **kwargs):
        boot_val = evaluator_func(boot_sel, boot_est)
        cross_val = evaluator_func(boot_sel, emp_est)
        return (boot_val - cross_val) * scale_factor

    _, corrected_val = bootstrap_correction_estimator(
        data=(cust_features, treatments, outcomes, targ_cust_features),
        estimator_func=estimator_func,
        optimizer_func=optimizer_func,
        evaluator_func=evaluator_func,
        bootstrap_sampler_func=bootstrap_sampler_func,
        wc_func=wc_func,
        n_bootstraps=n_bootstraps,
        n_jobs=n_jobs,
        verbose=verbose,
        seed=seed,
        **kwargs
    )

    return corrected_val


def get_wc_num_boot_dstn(
    cust_features: np.ndarray,
    treatments: np.ndarray,
    outcomes: np.ndarray,
    targ_cust_features: np.ndarray,
    optimization_params: dict,
    estimator,
    estimator_params: dict,
    n_bootstraps: int = 1000,
    emp_targ_te_arr: np.ndarray = None,
    emp_targ_var_arr: np.ndarray = None,
    outlier_threshold: float = 20.0,
    power: float = -0.45,
    n_jobs: int = 1,
    verbose: bool = False,
    seed: int = None,
    **kwargs
) -> dict:
    """
    Compute the numerical bootstrap distribution of the Winner's Curse for targeting applications.

    Params:
    -------
    cust_features: np.ndarray
        Features of the training customers
    treatments: np.ndarray
        Treatment assignments for training customers
    outcomes: np.ndarray
        Observed outcomes for training customers
    targ_cust_features: np.ndarray
        Features of the target customers for optimization
    optimization_params: dict
        Dictionary containing optimization methods to evaluate
    estimator: class
        Estimator model class to use (e.g., CausalForest)
    estimator_params: dict
        Parameters for the estimator
    n_bootstraps: int
        Number of bootstrap samples to generate
    emp_targ_te_arr: np.ndarray
        Pre-computed empirical treatment effects for target customers
    emp_targ_var_arr: np.ndarray
        Pre-computed empirical treatment effect variances for target customers
    outlier_threshold: float
        Threshold for outlier detection in treatment effects
    power: float
        The power parameter for the numerical bootstrap
    n_jobs: int
        Number of parallel jobs to run
    verbose: bool
        Whether to display progress
    seed: int
        Random seed
    
    Returns:
    --------
    dict
        Dictionary containing the bootstrap distribution of Winner's Curse for each optimization method
    """
    # parameter checks
    assert -0.5 < power < 0.0, "Power must be between -0.5 and 0.0"

    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    # Fit empirical model if not provided
    emp_targ_te_arr, emp_targ_var_arr = _compute_or_fit_treatment_effects(
        cust_features, treatments, outcomes, targ_cust_features,
        estimator, estimator_params, emp_targ_te_arr, emp_targ_var_arr
    )
    
    # Sample size
    sample_size = cust_features.shape[0]
    
    # Compute perturbation parameter
    epsilon_n = sample_size ** power

    # Define wrappers for numerical bootstrap
    def estimator_func(data, **kwargs):
        cust_features, treatments, outcomes, targ_cust_features = data
        model = estimator(**estimator_params).fit(X=cust_features, Y=outcomes, T=treatments)
        # Only the point estimates are read downstream (optimizer_func and
        # evaluator_func), so skip the variance computation entirely.
        targ_te_arr = _predict_point_effects(model, targ_cust_features)
        targ_te_arr = replace_outliers(targ_te_arr, threshold=outlier_threshold)
        return (model, targ_te_arr, None)

    def optimizer_func(estimates, **kwargs):
        model, targ_te_arr, targ_var_arr = estimates
        selection, _ = _apply_optimizer_targeting(optimization_params, model, targ_cust_features, targ_te_arr)
        return selection

    def evaluator_func(selection, estimates, **kwargs):
        model, targ_te_arr, targ_var_arr = estimates
        return obj_func(selection=selection, cust_feautres=targ_cust_features, demand_model=None, cust_treatment_effects=targ_te_arr)

    def bootstrap_sampler_func(data, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        cust_features, treatments, outcomes, targ_cust_features = data
        sample_size = cust_features.shape[0]
        boot_indices = np.random.choice(sample_size, size=sample_size, replace=True)
        return (cust_features[boot_indices], treatments[boot_indices], outcomes[boot_indices], targ_cust_features)

    def wc_func(boot_sel, boot_est, emp_est, evaluator_func, **kwargs):
        boot_model, boot_targ_te_arr, _ = boot_est
        emp_model, emp_targ_te_arr, _ = emp_est
        
        norm_error = np.sqrt(sample_size) * (boot_targ_te_arr - emp_targ_te_arr)
        perturbed_te_arr = emp_targ_te_arr + epsilon_n * norm_error
        
        emp_val_est = evaluator_func(boot_sel, emp_est)
        
        # For perturbed value, we need to pass perturbed estimates to evaluator
        perturbed_est = (emp_model, perturbed_te_arr, None)
        perturbed_val_est = evaluator_func(boot_sel, perturbed_est)
        
        return perturbed_val_est - emp_val_est

    _, corrected_val = bootstrap_correction_estimator(
        data=(cust_features, treatments, outcomes, targ_cust_features),
        estimator_func=estimator_func,
        optimizer_func=optimizer_func,
        evaluator_func=evaluator_func,
        bootstrap_sampler_func=bootstrap_sampler_func,
        wc_func=wc_func,
        n_bootstraps=n_bootstraps,
        n_jobs=n_jobs,
        verbose=verbose,
        seed=seed,
        **kwargs
    )
    
    return corrected_val



def bootstrap_correction_estimate(
    cust_features: np.ndarray,
    treatments: np.ndarray,
    outcomes: np.ndarray,
    optimization_params: dict,
    estimator,
    estimator_params: dict,
    targ_cust_features: np.ndarray,
    emp_targ_te_arr: np.ndarray,
    emp_targ_var_arr: np.ndarray,
    bootstrap_method: str = 'standard', 
    n_bootstraps: int = 1000,
    n_jobs: int = 1,
    verbose: bool = False,
    seed: int = None,
    **kwargs
) -> tuple:
    """
    Estimate the Winner's Curse using bootstrap correction for targeting applications.

    Params:
    -------
    cust_features: np.ndarray
        Features of the training customers
    treatments: np.ndarray
        Treatment assignments for training customers
    outcomes: np.ndarray
        Observed outcomes for training customers
    optimization_params: dict
        Dictionary containing optimization methods to evaluate
    estimator: class
        Estimator model class to use (e.g., CausalForest)
    estimator_params: dict
        Parameters for the estimator
    targ_cust_features: np.ndarray
        Features of the target customers for optimization
    emp_targ_te_arr: np.ndarray
        Pre-computed empirical treatment effects for target customers
    emp_targ_var_arr: np.ndarray
        Pre-computed empirical treatment effect variances for target customers
    bootstrap_method: str
        The bootstrap method to use. Options are 'standard', 'moon',
        and 'numerical'. 'moon' uses sqrt(m/N) scaling.
    n_bootstraps: int
        Number of bootstrap samples to generate
    n_jobs: int
        Number of parallel jobs to run
    verbose: bool
        Whether to display progress
    seed: int
        Random seed
    **kwargs
        Additional keyword arguments for the bootstrap method
    
    Returns:
    --------
    tuple: None, boot_est
        A tuple containing None (we don't change selections) and the bootstrap-corrected 
        policy value estimate (float)
    """

    # Get the corrected estimate or bootstrap distribution of the Winner's Curse
    result = {
        'standard': get_wc_boot_dstn,
        'moon': get_wc_moon_boot_dstn,
        'numerical': get_wc_num_boot_dstn,
    }[bootstrap_method](
        cust_features=cust_features,
        treatments=treatments,
        outcomes=outcomes,
        targ_cust_features=targ_cust_features,
        optimization_params=optimization_params,
        estimator=estimator,
        estimator_params=estimator_params,
        n_bootstraps=n_bootstraps,
        emp_targ_te_arr=emp_targ_te_arr,
        emp_targ_var_arr=emp_targ_var_arr, 
        n_jobs=n_jobs,
        verbose=verbose,
        seed=seed,
        **kwargs
    )

    # Compute the bootstrap-corrected policy value estimate
    if isinstance(result, (float, np.float64)):
        boot_est = result
    else:
        # result is the bootstrap distribution of the Winner's Curse
        # Compute base estimates (without correction)
        optimizer = optimization_params['optimizer']
        optimize_params = optimization_params['params']
        
        # Get the original selection and estimated value
        _, val_est = optimizer(
            demand_model=None,
            cust_features=targ_cust_features,
            cust_treatment_effects=emp_targ_te_arr,
            **optimize_params
        )
        
        # Compute the bootstrap-corrected policy value estimate
        boot_est = val_est - result.mean()

    # Return None for selection since we don't change selections
    return None, boot_est

##########
# Bayesian Estimation
##########

def bayes_estimate(
    cust_features: np.ndarray,
    treatments: np.ndarray,
    outcomes: np.ndarray,
    optimization_params: dict,
    estimator,
    estimator_params: dict,
    targ_cust_features: np.ndarray,
    emp_targ_te_arr: np.ndarray = None,
    emp_targ_var_arr: np.ndarray = None,
    prior: str = 'normal',
    **kwargs
) -> tuple:
    """
    Estimate the policy value using Bayesian shrinkage for targeting applications.

    Params:
    -------
    cust_features: np.ndarray
        Features of the training customers
    treatments: np.ndarray
        Treatment assignments for training customers
    outcomes: np.ndarray
        Observed outcomes for training customers
    optimization_params: dict
        Dictionary containing optimization methods to evaluate
    estimator: class
        Estimator model class to use (e.g., CausalForest)
    estimator_params: dict
        Parameters for the estimator
    targ_cust_features: np.ndarray
        Features of the target customers for optimization
    emp_targ_te_arr: np.ndarray, shape = (sample_size, n_treatments)
        Pre-computed empirical treatment effects for target customers
    emp_targ_var_arr: np.ndarray, shape = (sample_size, n_treatments)
        Pre-computed empirical treatment effect variances for target customers
    prior: str
        The prior distribution to use. Options are 'normal'.
    shrinkage_intensity: float
        Controls the strength of the shrinkage (higher values shrink more)
    
    Returns:
    --------
    tuple: None, val_est
        A tuple containing None (we don't change selections) and the adjusted policy value estimate (float)
    """
    # Parameter check 
    assert prior in ['normal'], f'The prior must be "normal". {prior} is not supported.'
    
    # Fit empirical model if not provided
    if emp_targ_te_arr is None or emp_targ_var_arr is None:
        emp_model = estimator(**estimator_params).fit(X=cust_features, Y=outcomes, T=treatments)
        emp_targ_te_arr, emp_targ_var_arr = emp_model.predict_incremental_effect(targ_cust_features)
    
    # Extract optimizer (single optimizer format expected)
    optimizer = optimization_params['optimizer']
    optimize_params = optimization_params['params']
    
    # Make targeting decisions using empirical treatment effects
    selection, _ = optimizer(
        demand_model=None,
        cust_features=targ_cust_features,
        cust_treatment_effects=emp_targ_te_arr,
        **optimize_params
    )
    
    # Apply Bayesian shrinkage to treatment effects
    n_treatments = emp_targ_te_arr.shape[1]
    
    # calculate posterior mean 
    bayes_estimate_func = {'normal': bayes_normal}[prior]
    post_mean_arr = np.array([
        bayes_estimate_func(
            mle_treatment_effects=emp_targ_te_arr[:, i],
            sampling_vars=emp_targ_var_arr[:, i], 
            **kwargs
        ) for i in range(n_treatments)
    ]).T  # shape = (sample_size, n_treatments)
    
    # Evaluate policy value with shrunken effects
    val_est = obj_func(
        selection=selection,
        cust_feautres=targ_cust_features,
        demand_model=None,
        cust_treatment_effects=post_mean_arr
    )
    
    return None, val_est


def empirical_bayes_estimate(
    cust_features: np.ndarray,
    treatments: np.ndarray,
    outcomes: np.ndarray,
    optimization_params: dict,
    estimator,
    estimator_params: dict,
    targ_cust_features: np.ndarray,
    emp_targ_te_arr: np.ndarray = None,
    emp_targ_var_arr: np.ndarray = None,
    prior: str = 'normal',
    **kwargs
) -> tuple:
    """ 
    Estimate the policy value using empirical Bayesian methods for targeting applications.

    Params:
    -------
    cust_features: np.ndarray
        Features of the training customers
    treatments: np.ndarray
        Treatment assignments for training customers
    outcomes: np.ndarray
        Observed outcomes for training customers
    optimization_params: dict
        Dictionary containing optimization methods to evaluate
    estimator: class
        Estimator model class to use (e.g., CausalForest)
    estimator_params: dict
        Parameters for the estimator
    targ_cust_features: np.ndarray
        Features of the target customers for optimization
    emp_targ_te_arr: np.ndarray, shape = (sample_size, n_treatments)
        Pre-computed empirical treatment effects for target customers
    emp_targ_var_arr: np.ndarray, shape = (sample_size, n_treatments)
        Pre-computed empirical treatment effect variances for target customers
    prior: str
        The prior distribution. Options are 'tweedies', 'normal' and 'spike_slab'.
    
    Returns:
    --------
    tuple: None, val_est
        A tuple containing None (we don't change selections) and the adjusted policy value estimate (float)
    """
    # parameter check
    assert prior in eb_function_dict.keys(), f'The prior must be {eb_function_dict.keys()}. {prior} is not supported.'
    
    # Extract optimizer (single optimizer format expected)
    optimizer = optimization_params['optimizer']
    optimize_params = optimization_params['params']
    
    # Fit empirical model if not provided
    if emp_targ_te_arr is None or emp_targ_var_arr is None:
        emp_model = estimator(**estimator_params).fit(X=cust_features, Y=outcomes, T=treatments)
        emp_targ_te_arr, emp_targ_var_arr = emp_model.predict_incremental_effect(targ_cust_features)
    
    # Make targeting decisions using empirical treatment effects
    selection, _ = optimizer(
        demand_model=None,
        cust_features=targ_cust_features,
        cust_treatment_effects=emp_targ_te_arr,
        **optimize_params
    )
    
    # Apply empirical Bayesian shrinkage to treatment effects
    n_treatments = emp_targ_te_arr.shape[1]
    
    # Calculate posterior mean using empirical Bayes methods
    eb_func = eb_function_dict[prior]
    try: 
        post_mean_arr = np.array([
            eb_func(
                mle_treatment_effects=emp_targ_te_arr[:, i],
                sampling_vars=emp_targ_var_arr[:, i], 
                **kwargs
            ) for i in range(n_treatments)
        ]).T  # shape = (sample_size, n_treatments)
    except:
        return None, np.nan
    
    # Evaluate policy value with shrunken effects
    val_est = obj_func(
        selection=selection,
        cust_feautres=targ_cust_features,
        demand_model=None,
        cust_treatment_effects=post_mean_arr
    )
    
    return None, val_est


##########
# Sample Splitting
##########

def sample_splitting_estimate(
    cust_features: np.ndarray,
    treatments: np.ndarray,
    outcomes: np.ndarray,
    optimization_params: dict,
    estimator,
    estimator_params: dict,
    targ_cust_features: np.ndarray,
    estimation_split: float = 0.5,
    seed: int = None,
    **kwargs
) -> tuple:
    """
    Estimate the policy value using sample splitting for targeting applications.

    Params:
    -------
    cust_features: np.ndarray
        Features of the training customers
    treatments: np.ndarray
        Treatment assignments for training customers
    outcomes: np.ndarray
        Observed outcomes for training customers
    optimization_params: dict
        Dictionary containing optimization methods to evaluate
    estimator: class
        Estimator model class to use (e.g., CausalForest)
    estimator_params: dict
        Parameters for the estimator
    targ_cust_features: np.ndarray
        Features of the target customers for optimization
    estimation_split: float
        The proportion of samples to use for estimation
    seed: int (optional)
        Random seed for reproducibility
        
    Returns:
    --------
    tuple:
        A tuple containing the selection array and the policy value estimate (float)
    """
    # parameter check 
    assert 0.0 < estimation_split < 1.0, 'The estimation split must be in the range (0, 1).'
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Split data into estimation and evaluation sets
    sample_size = cust_features.shape[0]
    est_size = int(sample_size * estimation_split)
    est_indices = np.random.choice(sample_size, size=est_size, replace=False)
    eval_indices = np.setdiff1d(np.arange(sample_size), est_indices)
    
    # Training data for estimation and evaluation
    est_features = cust_features[est_indices]
    est_treatments = treatments[est_indices]
    est_outcomes = outcomes[est_indices]
    
    eval_features = cust_features[eval_indices]
    eval_treatments = treatments[eval_indices]
    eval_outcomes = outcomes[eval_indices]
    
    # Train model on estimation set
    est_model = estimator(**estimator_params).fit(X=est_features, Y=est_outcomes, T=est_treatments)
    
    # Predict treatment effects for target customers using estimation model
    est_targ_te_arr = _predict_point_effects(est_model, targ_cust_features)
    
    # Extract optimizer (single optimizer format expected)
    optimizer = optimization_params['optimizer']
    optimize_params = optimization_params['params']
    
    # Select treatments using estimation set model
    selection, _ = optimizer(
        demand_model=est_model,
        cust_features=targ_cust_features,
        cust_treatment_effects=est_targ_te_arr,
        **optimize_params
    )
    
    # Train evaluation model on evaluation set
    eval_model = estimator(**estimator_params).fit(X=eval_features, Y=eval_outcomes, T=eval_treatments)
    
    # Predict treatment effects for target customers using evaluation model
    eval_targ_te_arr = _predict_point_effects(eval_model, targ_cust_features)
    
    # Calculate policy value using evaluation set model
    val_est = obj_func(
        selection=selection,
        cust_feautres=targ_cust_features,
        demand_model=None,
        cust_treatment_effects=eval_targ_te_arr
    )
    
    return selection, val_est


##########
# Selective Inference
##########


def selective_inference_estimate(
    cust_features: np.ndarray,
    treatments: np.ndarray,
    outcomes: np.ndarray,
    optimization_params: dict,
    estimator: callable,
    estimator_params: dict,
    targ_cust_features: np.ndarray,
    emp_targ_te_arr: np.ndarray = None,
    emp_targ_var_arr: np.ndarray = None,
    method: str = 'conditional', 
    quantile: float = 0.5, 
    n_jobs: int = 1,
    verbose: bool = False, 
    **kwargs
) -> tuple:
    """ 
    Compute the conditional inference method from (Andrews et al. 2024, QJE)
    adapted for targeting applications.
    
    Params:
    -------
    cust_features: np.ndarray
        Features of the training customers
    treatments: np.ndarray
        Treatment assignments for training customers
    outcomes: np.ndarray
        Observed outcomes for training customers
    optimization_params: dict
        Dictionary containing optimization methods to evaluate
    estimator: class
        Estimator model class to use (e.g., CausalForest)
    estimator_params: dict
        Parameters for the estimator
    targ_cust_features: np.ndarray
        Features of the target customers for optimization
    emp_targ_te_arr: np.ndarray
        Pre-computed empirical treatment effects for target customers
    emp_targ_var_arr: np.ndarray
        Pre-computed empirical treatment effect variances for target customers
    method: str
        The method to use for selective inference. Options are 'conditional' and 'hybrid'.
    quantile: float
        The quantile to use for adjustment (default: 0.5 for median unbiased estimator)
    n_jobs: int
        Number of parallel jobs to run
    verbose: bool
        Whether to display progress
    
    Returns:
    --------
    tuple: None, val_est
        A tuple containing None (we don't change selections) and the adjusted policy value estimate (float)
    """
    # parameter check
    assert method in ['conditional', 'hybrid'], 'The method must be either "conditional" or "hybrid".'
    assert quantile == 0.5, 'The quantile must be 0.5 for the median unbiased estimator.'

    # Extract optimizer (single optimizer format expected)
    optimizer = optimization_params['optimizer']
    optimize_params = optimization_params['params']

    # selective inference methods
    si_func = {
        'conditional': conditional_inference, 'hybrid': hybrid_inference
    }

    # Fit empirical model if not provided
    if emp_targ_te_arr is None:
        emp_model = estimator(**estimator_params).fit(X=cust_features, Y=outcomes, T=treatments)
        emp_targ_te_arr, emp_targ_var_arr = emp_model.predict_incremental_effect(targ_cust_features)

    # Make targeting decisions using empirical treatment effects
    selection, _ = optimizer(
        demand_model=None,
        cust_features=targ_cust_features,
        cust_treatment_effects=emp_targ_te_arr,
        **optimize_params
    )

    def adjust_single_customer(customer_id):
        # Get the selected treatment for this customer
        selected = selection[customer_id]
        
        result = si_func[method](
            mean_arr=emp_targ_te_arr[customer_id, :],  # shape = (n_treatments, )
            std_arr=emp_targ_var_arr[customer_id, :] ** 0.5,  # shape = (n_treatments, )
            max_item_idx=selected,
            quantile=quantile,
        )
        
        # si_func returns fsolve(..., full_output=True) == (x, infodict, ier, mesg).
        # ier == 1 means converged; any other value means the solver gave up,
        # which happens when the winner and runner-up are close enough that the
        # truncation window collapses and the objective goes flat. The returned
        # iterate is kept (unchanged behaviour), but the flag is counted so the
        # rate is visible downstream rather than silent.
        return selected, result[0][0], result[2]
    
    customer_adjustments = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(adjust_single_customer)(customer_id) 
        for customer_id in range(emp_targ_te_arr.shape[0])
    )
    
    # Create adjusted treatment effects array
    adjusted_te_arr = emp_targ_te_arr.copy()
    n_nonconverged = 0
    
    for customer_id, (selected, adjusted_effect, ier) in enumerate(customer_adjustments):
        adjusted_te_arr[customer_id, selected] = adjusted_effect
        if ier != 1:
            n_nonconverged += 1
    
    # Evaluate policy value with adjusted effects
    val_est = obj_func(
        selection=selection,
        cust_feautres=targ_cust_features,
        demand_model=None,
        cust_treatment_effects=adjusted_te_arr
    )
    
    return None, val_est, {
        'n_nonconverged': n_nonconverged,
        'n_customers': emp_targ_te_arr.shape[0],
    }


# def no_correction_with_known_functional_form(
#     cust_features: np.ndarray,
#     treatments: np.ndarray,
#     outcomes: np.ndarray,
#     optimization_params: dict,
#     targ_cust_features: np.ndarray,
#     **kwargs
# ) -> tuple:
#     """
#     Estimate the policy value using no correction with known functional form for targeting applications.

#     Params:
#     -------
#     cust_features: np.ndarray
#         Features of the training customers
#     treatments: np.ndarray
#         Treatment assignments for training customers
#     outcomes: np.ndarray
#         Observed outcomes for training customers
#     optimization_params: dict
#         Dictionary containing optimization methods to evaluate
#     targ_cust_features: np.ndarray  
#     """

#     # fit the model
#     correct_fmodel = EstimateWithKnownFunctionalForm(n_treatments=2, fit_intercept=False).fit(
#         X=cust_features, Y=outcomes, T=treatments
#     )
    
#     # make targeting decisions
#     selection_dict = {}
#     val_est_dict = {}
#     for optimizer_name in optimization_params.keys():
#         optimizer = optimization_params[optimizer_name]['optimizer']
#         optimize_params = optimization_params[optimizer_name]['params']

#         targ_te_arr, _ = correct_fmodel.predict_incremental_effect(targ_cust_features)
        
#         # Make targeting decisions using empirical treatment effects
#         selection, val_est = optimizer(
#             demand_model=correct_fmodel,
#             cust_features=targ_cust_features,
#             cust_treatment_effects=targ_te_arr,
#             **optimize_params
#         )
        
#         selection_dict[optimizer_name] = selection
#         val_est_dict[optimizer_name] = val_est
    
#     return selection_dict, val_est_dict
