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
        self.model.fit(X=X, Y=Y, T=T)
        return self 
    
    def predict_incremental_effect(self, X: np.ndarray) -> tuple:
        inference_result = self.model.const_marginal_effect_inference(X)

        # point estimate and standard error, both have shape = (sample_size, n_treatments)
        point, se = inference_result.pred, inference_result.pred_stderr
        return point, se ** 2
    
    def predict_incremental_effect_interval(self, X: np.ndarray, alpha: float=0.05):
        return self.model.const_marginal_effect_interval(X, alpha=alpha)


class EstimateWithKnownFunctionalForm(object):
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

##########
# Selection
##########

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


def repeated_experiment(
    optimization_params: dict, 
    dgp_params: dict, 
    data_params: dict, 
    experiment_params: dict, 
    estimators_dict: dict, 
    outlier_threshold: float = 20.0, 
    verbose: bool = False,
    n_jobs: int = 1,
) -> dict:
    """
    Perform a repeated experiment.
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

        # run no correction estimator for each selection methods
        for optimizer_name in optimization_params.keys():
            optimizer = optimization_params[optimizer_name]['optimizer']
            optimize_params = optimization_params[optimizer_name]['params']

            # select experiments
            selection, val_est = optimizer(
                demand_model=emp_estimator, 
                cust_features=targ_cust_features, 
                cust_treatment_effects=emp_targ_te_arr, 
                **optimize_params
            )  # shape = (sample_size,)

            # evaluate selection
            val_true = obj_func(
                selection=selection, cust_feautres=targ_cust_features, demand_model=dgp
            )

            # compute the Wald Closure
            wc = val_est - val_true

            # store results
            result_dict[f'{optimizer_name}_selection'] = selection 
            result_dict[f'{optimizer_name}_val_true'] = val_true
            result_dict[f'{optimizer_name}_val_est'] = val_est
            result_dict[f'{optimizer_name}_wc'] = wc
            
        result_dict['est_treatment_effects'] = emp_targ_te_arr

        # run all other estimators for each selection methods
        for estimator_name in estimators_dict.keys():
            temp_estimator = estimators_dict[estimator_name]['estimator']
            temp_params = estimators_dict[estimator_name]['params']
            temp_selection_dict, temp_est_dict = temp_estimator(
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

            if temp_selection_dict is None:
                for optimizer_name in temp_est_dict.keys():
                    result_dict[f'{optimizer_name}_{estimator_name}_val_est'] = temp_est_dict[optimizer_name]
                    result_dict[f'{optimizer_name}_{estimator_name}_wc'] = temp_est_dict[optimizer_name] - result_dict[f'{optimizer_name}_val_true']
            else:
                for optimizer_name in temp_selection_dict.keys():
                    temp_val_true = obj_func(
                        selection=temp_selection_dict[optimizer_name], 
                        cust_feautres=targ_cust_features, demand_model=dgp, 
                    )
                    result_dict[f'{optimizer_name}_{estimator_name}_selection'] = temp_selection_dict[optimizer_name]
                    result_dict[f'{optimizer_name}_{estimator_name}_val_true'] = temp_val_true
                    result_dict[f'{optimizer_name}_{estimator_name}_val_est'] = temp_est_dict[optimizer_name]
                    result_dict[f'{optimizer_name}_{estimator_name}_wc'] = temp_est_dict[optimizer_name] - result_dict[f'{optimizer_name}_val_true']

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
            f'{optimizer}_nc_val_true_avg': np.nanmean(val_true_arr), f'{optimizer}_nc_val_true_se': np.nanstd(val_true_arr) / sample_size ** 0.5, 
            f'{optimizer}_nc_selection_arr': np.array([result[f'{optimizer}_selection'] for result in result_records])
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
                    f'{optimizer}_{estimator}_val_true_arr': temp_val_true_arr,
                    f'{optimizer}_{estimator}_val_true_avg': np.nanmean(temp_val_true_arr),
                    f'{optimizer}_{estimator}_val_true_se': np.nanstd(temp_val_true_arr) / sample_size ** 0.5, 
                    f'{optimizer}_{estimator}_selection_arr': np.array([result[f'{optimizer}_{estimator}_selection'] for result in result_records])
                })

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
    if emp_targ_te_arr is None:
        emp_model = estimator(**estimator_params).fit(X=cust_features, Y=outcomes, T=treatments)
        emp_targ_te_arr, _ = emp_model.predict_incremental_effect(targ_cust_features)  # shape = (sample_size, n_treatments)
    
    # Sample size
    sample_size = cust_features.shape[0]

    def compute_wc(boot_id) -> dict:
        """
        Compute Winner's Curse for a single bootstrap sample.
        """
        # Draw bootstrap sample
        boot_indices = np.random.choice(sample_size, size=sample_size, replace=True)
        boot_cust_features = cust_features[boot_indices]
        boot_treatments = treatments[boot_indices]
        boot_outcomes = outcomes[boot_indices]
        
        # Fit model on bootstrap sample
        boot_model = estimator(**estimator_params).fit(
            X=boot_cust_features, Y=boot_outcomes, T=boot_treatments
        )
        boot_targ_te_arr, _ = boot_model.predict_incremental_effect(targ_cust_features)

        # replace outliers in the treatment effects array
        boot_targ_te_arr = replace_outliers(boot_targ_te_arr, threshold=outlier_threshold)

        # Compute winner's curse for each optimizer
        wc_dict = {}
        for optimizer_name in optimization_params.keys():
            optimizer = optimization_params[optimizer_name]['optimizer']
            optimize_params = optimization_params[optimizer_name]['params']
            
            # Make targeting decisions using bootstrap model
            boot_selection, boot_val_est = optimizer(
                demand_model=boot_model,
                cust_features=targ_cust_features,
                cust_treatment_effects=boot_targ_te_arr,
                **optimize_params
            )
            
            # Evaluate decision using empirical treatment effects
            emp_val_est = obj_func(
                selection=boot_selection,
                cust_feautres=targ_cust_features,
                demand_model=None,
                cust_treatment_effects=emp_targ_te_arr
            )
            
            # Compute winner's curse
            wc_dict[optimizer_name] = boot_val_est - emp_val_est
            
        return wc_dict
    
    # Compute winner's curse for each bootstrap sample
    boot_wc_records = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(compute_wc)(boot_id) for boot_id in range(n_bootstraps)
    )
    
    # Organize results by optimizer
    boot_wc_dstn_dict = {
        optimizer_name: np.array([record[optimizer_name] for record in boot_wc_records])
        for optimizer_name in optimization_params.keys()
    }
    
    return boot_wc_dstn_dict


def get_wc_m_out_of_n_boot_dstn(
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
    Compute the m-out-of-n bootstrap distribution of the Winner's Curse for targeting applications.

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
        The power of the m-out-of-n bootstrap (controls subsample size)
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
    # parameter check 
    assert 0.0 < power < 1.0, 'The power must be in the range (0, 1).'

    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    # Fit empirical model if not provided
    if emp_targ_te_arr is None:
        emp_model = estimator(**estimator_params).fit(X=cust_features, Y=outcomes, T=treatments)
        emp_targ_te_arr, _ = emp_model.predict_incremental_effect(targ_cust_features)
    
    # Sample size
    sample_size = cust_features.shape[0]

    def compute_wc(boot_id) -> dict:
        """
        Compute Winner's Curse for a single bootstrap sample using m-out-of-n bootstrap.
        """
        # Draw smaller bootstrap sample (m < n)
        boot_sample_size = int(sample_size ** power)
        boot_indices = np.random.choice(sample_size, size=boot_sample_size, replace=True)
        boot_cust_features = cust_features[boot_indices]
        boot_treatments = treatments[boot_indices]
        boot_outcomes = outcomes[boot_indices]
        
        # Fit model on bootstrap sample
        boot_model = estimator(**estimator_params).fit(
            X=boot_cust_features, Y=boot_outcomes, T=boot_treatments
        )
        boot_targ_te_arr, _ = boot_model.predict_incremental_effect(targ_cust_features)

        # replace outliers in the treatment effects array
        boot_targ_te_arr = replace_outliers(boot_targ_te_arr, threshold=outlier_threshold)
        
        # Compute winner's curse for each optimizer
        wc_dict = {}
        for optimizer_name in optimization_params.keys():
            optimizer = optimization_params[optimizer_name]['optimizer']
            optimize_params = optimization_params[optimizer_name]['params']
            
            # Make targeting decisions using bootstrap model
            boot_selection, boot_val_est = optimizer(
                demand_model=boot_model,
                cust_features=targ_cust_features,
                cust_treatment_effects=boot_targ_te_arr,
                **optimize_params
            )
            
            # Evaluate decision using empirical treatment effects
            emp_val_est = obj_func(
                selection=boot_selection,
                cust_feautres=targ_cust_features,
                demand_model=None,
                cust_treatment_effects=emp_targ_te_arr
            )
            
            # Compute winner's curse
            wc_dict[optimizer_name] = boot_val_est - emp_val_est
            
        return wc_dict
    
    # Compute winner's curse for each bootstrap sample
    boot_wc_records = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(compute_wc)(boot_id) for boot_id in range(n_bootstraps)
    )
    
    # Organize results by optimizer
    boot_wc_dstn_dict = {
        optimizer_name: np.array([record[optimizer_name] for record in boot_wc_records])
        for optimizer_name in optimization_params.keys()
    }
    
    return boot_wc_dstn_dict


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
    if emp_targ_te_arr is None:
        emp_model = estimator(**estimator_params).fit(X=cust_features, Y=outcomes, T=treatments)
        emp_targ_te_arr, _ = emp_model.predict_incremental_effect(targ_cust_features)
    
    # Sample size
    sample_size = cust_features.shape[0]
    
    # Compute perturbation parameter
    epsilon_n = sample_size ** power

    def compute_wc(boot_id) -> dict:
        """
        Compute Winner's Curse for a single numerical bootstrap sample.
        """
        # Draw regular bootstrap sample
        boot_indices = np.random.choice(sample_size, size=sample_size, replace=True)
        boot_cust_features = cust_features[boot_indices]
        boot_treatments = treatments[boot_indices]
        boot_outcomes = outcomes[boot_indices]
        
        # Fit model on bootstrap sample
        boot_model = estimator(**estimator_params).fit(
            X=boot_cust_features, Y=boot_outcomes, T=boot_treatments
        )
        boot_targ_te_arr, _ = boot_model.predict_incremental_effect(targ_cust_features)

        # replace outliers in the treatment effects array
        boot_targ_te_arr = replace_outliers(boot_targ_te_arr, threshold=outlier_threshold)
        
        # Compute perturbed treatment effect estimates
        norm_error = np.sqrt(sample_size) * (boot_targ_te_arr - emp_targ_te_arr)
        perturbed_te_arr = emp_targ_te_arr + epsilon_n * norm_error
        
        # Compute winner's curse for each optimizer
        wc_dict = {}
        for optimizer_name in optimization_params.keys():
            optimizer = optimization_params[optimizer_name]['optimizer']
            optimize_params = optimization_params[optimizer_name]['params']
            
            # Make targeting decisions using bootstrap model
            boot_selection, _ = optimizer(
                demand_model=boot_model,
                cust_features=targ_cust_features,
                cust_treatment_effects=boot_targ_te_arr,
                **optimize_params
            )
            
            # Evaluate decisions using empirical and perturbed treatment effects
            emp_val_est = obj_func(
                selection=boot_selection,
                cust_feautres=targ_cust_features,
                demand_model=None,
                cust_treatment_effects=emp_targ_te_arr
            )
            
            perturbed_val_est = obj_func(
                selection=boot_selection,
                cust_feautres=targ_cust_features,
                demand_model=None,
                cust_treatment_effects=perturbed_te_arr
            )
            
            # Compute winner's curse
            wc_dict[optimizer_name] = perturbed_val_est - emp_val_est
            
        return wc_dict
    
    # Compute winner's curse for each bootstrap sample
    boot_wc_records = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(compute_wc)(boot_id) for boot_id in range(n_bootstraps)
    )
    
    # Organize results by optimizer
    boot_wc_dstn_dict = {
        optimizer_name: np.array([record[optimizer_name] for record in boot_wc_records])
        for optimizer_name in optimization_params.keys()
    }
    
    return boot_wc_dstn_dict


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
        The bootstrap method to use. Options are 'standard', 'm_out_of_n', and 'numerical'
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
    tuple: None, boot_est_dict
        A tuple containing None (we don't change selections) and the bootstrap-corrected 
        policy value estimate dictionary
    """

    # Get the bootstrap distribution of the Winner's Curse for each selection method
    boot_dstn_dict = {
        'standard': get_wc_boot_dstn,
        'm_out_of_n': get_wc_m_out_of_n_boot_dstn,
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

    # Compute base estimates (without correction) for each optimizer
    base_est_dict = {}
    for optimizer_name in optimization_params.keys():
        optimizer = optimization_params[optimizer_name]['optimizer']
        optimize_params = optimization_params[optimizer_name]['params']
        
        # Get the original selection and estimated value
        _, val_est = optimizer(
            demand_model=None,
            cust_features=targ_cust_features,
            cust_treatment_effects=emp_targ_te_arr,
            **optimize_params
        )
        
        base_est_dict[optimizer_name] = val_est
    
    # Compute the bootstrap-corrected policy value estimate for each selection method
    boot_est_dict = {
        optimizer_name: base_est_dict[optimizer_name] - boot_dstn_dict[optimizer_name].mean()
        for optimizer_name in optimization_params.keys()
    }

    # Return None for selection_dict since we don't change selections
    return None, boot_est_dict

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
    tuple: None, val_est_dict
        A tuple containing None (we don't change selections) and the adjusted policy value estimate dictionary
    """
    # Parameter check 
    assert prior in ['normal'], f'The prior must be "normal". {prior} is not supported.'
    
    # Fit empirical model if not provided
    if emp_targ_te_arr is None or emp_targ_var_arr is None:
        emp_model = estimator(**estimator_params).fit(X=cust_features, Y=outcomes, T=treatments)
        emp_targ_te_arr, emp_targ_var_arr = emp_model.predict_incremental_effect(targ_cust_features)
    
    # Get original selections for each optimization method
    selection_dict = {}
    for optimizer_name, optimizer_dict in optimization_params.items():
        optimizer = optimizer_dict['optimizer']
        optimize_params = optimizer_dict['params']
        
        # Make targeting decisions using empirical treatment effects
        selection, _ = optimizer(
            demand_model=None,
            cust_features=targ_cust_features,
            cust_treatment_effects=emp_targ_te_arr,
            **optimize_params
        )
        
        selection_dict[optimizer_name] = selection
    
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
    val_est_dict = {}
    for optimizer_name in optimization_params.keys():
        val_est_dict[optimizer_name] = obj_func(
            selection=selection_dict[optimizer_name],
            cust_feautres=targ_cust_features,
            demand_model=None,
            cust_treatment_effects=post_mean_arr
        )
    
    return None, val_est_dict


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
    tuple: None, val_est_dict
        A tuple containing None (we don't change selections) and the adjusted policy value estimate dictionary
    """
    # parameter check
    assert prior in eb_function_dict.keys(), f'The prior must be {eb_function_dict.keys()}. {prior} is not supported.'
    
    # Fit empirical model if not provided
    if emp_targ_te_arr is None or emp_targ_var_arr is None:
        emp_model = estimator(**estimator_params).fit(X=cust_features, Y=outcomes, T=treatments)
        emp_targ_te_arr, emp_targ_var_arr = emp_model.predict_incremental_effect(targ_cust_features)
    
    # Get original selections for each optimization method
    selection_dict = {}
    for optimizer_name, optimizer_dict in optimization_params.items():
        optimizer = optimizer_dict['optimizer']
        optimize_params = optimizer_dict['params']
        
        # Make targeting decisions using empirical treatment effects
        selection, _ = optimizer(
            demand_model=None,
            cust_features=targ_cust_features,
            cust_treatment_effects=emp_targ_te_arr,
            **optimize_params
        )
        
        selection_dict[optimizer_name] = selection
    
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
        return None, {optimizer_name: np.nan for optimizer_name in optimization_params.keys()}
    
    # Evaluate policy value with shrunken effects
    val_est_dict = {}
    for optimizer_name in optimization_params.keys():
        val_est_dict[optimizer_name] = obj_func(
            selection=selection_dict[optimizer_name],
            cust_feautres=targ_cust_features,
            demand_model=None,
            cust_treatment_effects=post_mean_arr
        )
    
    return None, val_est_dict


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
        A tuple containing the selection dictionary and the policy value estimate dictionary
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
    est_targ_te_arr, _ = est_model.predict_incremental_effect(targ_cust_features)
    
    # Make targeting decisions and evaluate them
    selection_dict = {}
    val_est_dict = {}
    
    for optimizer_name in optimization_params.keys():
        optimizer = optimization_params[optimizer_name]['optimizer']
        optimize_params = optimization_params[optimizer_name]['params']
        
        # Select treatments using estimation set model
        selection, _ = optimizer(
            demand_model=est_model,
            cust_features=targ_cust_features,
            cust_treatment_effects=est_targ_te_arr,
            **optimize_params
        )
        
        selection_dict[optimizer_name] = selection
        
        # Train evaluation model on evaluation set
        eval_model = estimator(**estimator_params).fit(X=eval_features, Y=eval_outcomes, T=eval_treatments)
        
        # Predict treatment effects for target customers using evaluation model
        eval_targ_te_arr, _ = eval_model.predict_incremental_effect(targ_cust_features)
        
        # Calculate policy value using evaluation set model
        val_est = obj_func(
            selection=selection,
            cust_feautres=targ_cust_features,
            demand_model=None,
            cust_treatment_effects=eval_targ_te_arr
        )
        
        val_est_dict[optimizer_name] = val_est
    
    return selection_dict, val_est_dict


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
    tuple: None, est_dict
        A tuple containing None (we don't change selections) and the adjusted policy value estimate dictionary
    """
    # parameter check
    assert method in ['conditional', 'hybrid'], 'The method must be either "conditional" or "hybrid".'
    assert quantile == 0.5, 'The quantile must be 0.5 for the median unbiased estimator.'

    # selective inference methods
    si_func = {
        'conditional': conditional_inference, 'hybrid': hybrid_inference
    }

    # Fit empirical model if not provided
    if emp_targ_te_arr is None:
        emp_model = estimator(**estimator_params).fit(X=cust_features, Y=outcomes, T=treatments)
        emp_targ_te_arr, emp_targ_var_arr = emp_model.predict_incremental_effect(targ_cust_features)

    # Get original selections for each optimization method
    selection_dict = {}
    for optimizer_name, optimizer_dict in optimization_params.items():
        optimizer = optimizer_dict['optimizer']
        optimize_params = optimizer_dict['params']
        
        # Make targeting decisions using empirical treatment effects
        selection, _ = optimizer(
            demand_model=None,
            cust_features=targ_cust_features,
            cust_treatment_effects=emp_targ_te_arr,
            **optimize_params
        )
        
        selection_dict[optimizer_name] = selection

    def adjust_single_customer(customer_id):
        # Get the selected treatment for this customer
        selected_treatment = {}
        for optimizer_name in optimization_params.keys():
            selected_treatment[optimizer_name] = selection_dict[optimizer_name][customer_id]
        
        adjusted_effects = {}
        
        for optimizer_name in optimization_params.keys():
            # Extract the selected and unselected effects
            selected = selected_treatment[optimizer_name]
        
            result = si_func[method](
                mean_arr=emp_targ_te_arr[customer_id, :],  # shape = (n_treatments, )
                std_arr=emp_targ_var_arr[customer_id, :] ** 0.5,  # shape = (n_treatments, )
                max_item_idx=selected,
                quantile=quantile,
            )
            adjusted_effects[optimizer_name] = selected, result[0]

        return adjusted_effects
    
    customer_adjustments = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(adjust_single_customer)(customer_id) 
        for customer_id in range(emp_targ_te_arr.shape[0])
    )
    
    # Create adjusted treatment effects array for each optimizer
    adjusted_te_dict = {optimizer_name: emp_targ_te_arr.copy() for optimizer_name in optimization_params.keys()}
    
    for customer_id, adjustments in enumerate(customer_adjustments):
        for optimizer_name, (selected, adjusted_effect) in adjustments.items():
            adjusted_te_dict[optimizer_name][customer_id, selected] = adjusted_effect
    
    # Evaluate policy value with adjusted effects
    val_est_dict = {}
    for optimizer_name in optimization_params.keys():
        val_est_dict[optimizer_name] = obj_func(
            selection=selection_dict[optimizer_name],
            cust_feautres=targ_cust_features,
            demand_model=None,
            cust_treatment_effects=adjusted_te_dict[optimizer_name]
        )
    
    return None, val_est_dict


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