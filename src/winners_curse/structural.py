import os 
import sys 
import pickle 
sys.path.insert(0, os.path.abspath('.'))

from joblib import Parallel, delayed
from typing import Union
from itertools import product

import warnings 

import numpy as np

from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning

import jax 
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS

from winners_curse.bayes_methods import *
from winners_curse.mnl import compute_purchase_probabilities
from winners_curse.dgp import MNLAssortment
    

def negative_log_likelihood(
    params: np.ndarray, counts: np.ndarray, sample_size: int
) -> float:
    """
    Compute the negative log-likelihood of the data given the parameters of the MNL model.
    For identification, we fix product 1's fixed effect to 0.

    Log-likelihood of the MNL model:
    l(\beta) = \sum_{i=1}^{N} \log(p_{i})
        - p_{i} = \frac{e^{x_{i} \beta}}{1 + \sum_{j=1}^{J} e^{x_{j}\beta}}
        - x_{i} is product i's fixed effects
        - \beta is the parameter vector
        - N is the sample size
        - J is the number of products

    Params:
    -------
    params: np.ndarray, shape = (n_products, )
        The parameters of the MNL model.
    counts: np.ndarray, shape=(n_products, )
        The counts of each product in the data.
    sample_size: int
        The number of purchase records in the data.

    Returns:
    --------
    float
        The negative log-likelihood of the data given the parameters.
    """
    assert len(params) == len(counts), "Number of parameters must match number of products"

    # Compute denominator: 1 (for the outside option) + sum_j exp(v_j)
    denom = 1 + np.sum(np.exp(params))

    # Log-likelihood: sum_i count_i*v_i - total_customers*log(denom)
    ll = np.sum(counts * params) - sample_size * np.log(denom)

    return -ll  # negative log-likelihood for minimization


def estimate_fixed_effects(purchase_records: np.ndarray, n_products: int):
    """
    Estimate the fixed effects of the customers based on their purchase records.
    The outside option has utility 0.

    Params:
    -------
    purchase_records: np.ndarray
        The purchase records of the customers, where each row corresponds to a customer and 
        each entry corresponds to the product index that the customer chose.
    n_products: int
        The number of products in the assortment.

    Returns:
    --------
    np.ndarray
        The estimated fixed effects of the customers, shape = (n_products, ).
    """
    # Compute the counts of each product excluding the outside option
    counts = np.zeros(n_products)
    outside_count = 0
    for choice in purchase_records:
        if choice == "outside":
            outside_count += 1
        else:
            counts[int(choice)] += 1

    # Initialize the parameters
    init_params = np.zeros(n_products)  # shape = (n_products, )

    # Estimate the fixed effects
    result = minimize(
        negative_log_likelihood, init_params, args=(counts, len(purchase_records)), method="L-BFGS-B"
    )

    if not result.success:
        warnings.warn("Estimation did not converge", ConvergenceWarning)
        return init_params 

    return result.x

def estimate_fixed_effects_bayes(
    purchase_records: list, n_products: int, 
    prior_mean: int = 0, prior_std: int = 10, 
    n_warmup: int = 500, post_sample_size: int = 1000, 
    **kwargs
) -> np.ndarray:
    """ 
    Estimate the product fixed effects using a Bayesian approach.

    Params: 
    -------
    purchase_records: list
        The purchase records to use for estimation.
    n_products: int
        The number of products in the assortment.
    prior_mean: int
        The mean of the prior distribution for the fixed effects.
    prior_std: int
        The standard deviation of the prior distribution for the fixed effects.
    n_warmup: int
        The number of warmup steps to use in the MCMC sampler.
    post_sample_size: int
        The number of posterior samples

    Returns:
    --------
    np.ndarray, shape = (n_obs, n_products)
        Posterior samples of the fixed effects.
    """
    # calculate observed product counts
    prod_counts = {str(i): 0 for i in range(n_products)}
    prod_counts['outside'] = 0
    obs_products, obs_counts = np.unique(purchase_records, return_counts=True)
    for i, count in zip(obs_products, obs_counts):
        prod_counts[str(i)] = count
    observed_counts = list(prod_counts.values())
    
    # shift observed counts to align with product array
    observed_counts = np.concatenate([[observed_counts[-1]], observed_counts[:-1]])

    # count total number of observations
    total_customers = len(purchase_records)

    def mnl_model():
        # Prior for each product fixed effect: v_j ~ N(0, 1)
        v = numpyro.sample("v", dist.Normal(prior_mean * jnp.ones(n_products), prior_std * jnp.ones(n_products)))
        
        # Compute exponentiated fixed effects
        exp_v = jnp.exp(v)
        # Denom: outside option plus sum of exponentiated product utilities
        denom = 1.0 + jnp.sum(exp_v)
        # Probabilities for outside option and products
        p0 = 1.0 / denom
        p_products = exp_v / denom
        # Concatenate to get the full probability vector
        p_all = jnp.concatenate([jnp.array([p0]), p_products])
        
        # Likelihood: observed counts come from a multinomial distribution
        numpyro.sample("obs", dist.Multinomial(total_customers, p_all), obs=observed_counts)

    # run MCMC sampler using NumPyro's NUTS kernel
    rng_key = jax.random.PRNGKey(0)
    nuts_kernel = NUTS(mnl_model)
    mcmc = MCMC(nuts_kernel, num_warmup=n_warmup, num_samples=post_sample_size, progress_bar=False)
    mcmc.run(rng_key)

    return np.array(mcmc.get_samples()['v'])  # shape = (post_sample_size, n_products)


def obj_func(
    fixed_effects: np.ndarray, 
    assortment: np.ndarray,
) -> float: 
    """ 
    Calculate the expected value of the objective function

    Params:
    -------
    fixed_effects: np.ndarray, shape = (n_products, )
        Fixed effects for each product
    assortment: np.ndarray, shape = (n_products, )
        Assortment of products

    Returns:
    --------
    float
    """
    # check that assortment is a binary matrix 
    assert np.all(np.isin(assortment, [0, 1]))

    purchase_probs = compute_purchase_probabilities(
        fixed_effects=fixed_effects, assortment=assortment
    )[:-1]  # shape = (n_products, )

    return np.ones(len(assortment)) @ purchase_probs

def obj_func_batch(
    fixed_effects: np.ndarray,
    assortments: np.ndarray
) -> np.ndarray: 
    """
    Calculate the expected value of the objective function for a batch of assortments

    Params:
    -------
    fixed_effects: np.ndarray, shape = (n_products, )
        Fixed effects for each product
    assortments: np.ndarray, shape = (n_assortments, n_products)
        Assortments of products
    """
    # check the shape of the fixed effects
    assert fixed_effects.shape[0] == assortments.shape[1]

    # check that assortment is a binary matrix 
    assert np.all(np.isin(assortments, [0, 1]))

    # extract parameters
    n_products = fixed_effects.shape[0]

    # replace zeros in the assortment with negative infinity
    # to ensure that the product is not chosen
    assortments = assortments.astype(float)
    neg_inf_mask = assortments == 0  # shape = (n_assortments, n_products)

    # compute the utilities for each assortment
    exponent = fixed_effects[None, :] * assortments
    exponent[neg_inf_mask] = -np.inf
    exp_utilities = np.exp(exponent)

    # compute the purchase probabilities for each assortment
    purchase_probs = exp_utilities / (1 + exp_utilities.sum(axis=1)[:, None])

    return purchase_probs @ np.ones(n_products)
    

def obj_func_bayes(
    post_sample: np.ndarray,
    assortments: np.ndarray
) -> np.ndarray: 
    """
    Calculate the expected value of the objective function for a batch of assortments

    Params:
    -------
    post_sample: np.ndarray, shape = (sample_size, n_products)
        Posterior samples of fixed effects
    assortments: np.ndarray, shape = (n_assortments, n_products)
        Assortments of products

    Returns:
    --------
    obj_values: np.ndarray, shape = (sample_size, n_assortments)
        Objective function for each assortment in each posterior sample
    """
    # check the shape of the fixed effects
    assert post_sample.shape[1] == assortments.shape[1]

    # check that assortment is a binary matrix 
    assert np.all(np.isin(assortments, [0, 1]))

    # extract parameters
    n_products = post_sample.shape[1]

    # replace zeros in the assortment with negative infinity
    # to ensure that the product is not chosen
    assortments = assortments.astype(float)
    neg_inf_mask = assortments == 0  # shape = (n_assortments, n_products)

    # compute the utilities for each assortment
    exponent = post_sample[:, None, :] * assortments  # shape = (sample_size, n_assortments, n_products)
    exponent[:, neg_inf_mask] = -np.inf
    exp_utilities = np.exp(exponent)

    # compute the purchase probabilities for each assortment
    purchase_probs = exp_utilities / (1 + exp_utilities.sum(axis=2)[:, :, None])

    return purchase_probs @ np.ones(n_products)


def optimize(
    fixed_effects: np.ndarray,
    capacity: int,
) -> tuple:
    """ 
    Optimize the objective function to find the optimal assortment

    Params:
    -------
    fixed_effects: np.ndarray, shape = (n_products, )
        Fixed effects for each product
    capacity: int
        Capacity of the assortment

    Returns:
    --------
    """
    # extract parameters 
    n_products = fixed_effects.shape[0]

    # create all possible assortments
    all_assortments = np.array(list(product([0, 1], repeat=n_products)))

    # find all assortments within the capacity limit
    feasible_assortments = all_assortments[all_assortments.sum(axis=1) <= capacity]

    # calculate the objective function for each assortment
    obj_values = obj_func_batch(
        fixed_effects=fixed_effects, assortments=feasible_assortments
    )

    # find the assortment that maximizes the objective function
    max_idx = np.argmax(obj_values)

    return feasible_assortments[max_idx], obj_values[max_idx]


def optimize_bayes(
    post_sample: np.ndarray, 
    capacity: int,
) -> tuple:
    """ 
    Optimize the objective function to find the optimal assortment

    Params:
    -------
    post_sample: np.ndarray, shape = (sample_size, n_products)
        Posterior samples of fixed effects for each product
    capacity: int
        Capacity of the assortment

    Returns:
    --------
    
    """
    # extract parameters 
    n_products = post_sample.shape[1]

    # create all possible assortments
    all_assortments = np.array(list(product([0, 1], repeat=n_products)))

    # find all assortments within the capacity limit
    feasible_assortments = all_assortments[all_assortments.sum(axis=1) <= capacity]

    # calculate the objective function for each assortment
    post_obj_values = obj_func_bayes(
        post_sample=post_sample, assortments=feasible_assortments
    )  # shape = (sample_size, n_assortments)

    # calculate the posterior average objective value for each assortment
    post_mean_obj_values = post_obj_values.mean(axis=0)  # shape = (n_assortments, )

    # find the assortment with the maximum posterior average objective value
    max_idx = np.argmax(post_mean_obj_values)

    return feasible_assortments[max_idx], post_mean_obj_values[max_idx]


def repeated_experiment(
    capacity: int,
    dgp_params: dict, 
    data_params: dict, 
    experiment_params: dict,
    estimators_dict: dict = None,
    verbose: bool = False,
    n_jobs: int = 1,
) -> dict:
    """
    Perform a repeated experiment with the MNL model.
    
    Params:
    -------
    capacity: int
        Capacity constraint for the assortment
    dgp_params: dict
        Parameters for the data generation process
    data_params: dict
        Parameters related to the data (e.g., sample_size)
    experiment_params: dict
        Parameters related to the experiment (e.g., n_repeats)
    estimators_dict: dict, optional
        Dictionary of additional estimators to evaluate
    verbose: bool, default=False
        Whether to print progress information
    n_jobs: int, default=1
        Number of parallel jobs to run
        
    Returns:
    --------
    list
        List of dictionaries containing results for each experiment
    """
    # Unpack the parameters
    n_repeats = experiment_params['n_repeats']
    sample_size = data_params['sample_size']
    
    # Create data generation process
    dgp = MNLAssortment(**dgp_params)
    true_fixed_effects = dgp.fixed_effects if hasattr(dgp, 'fixed_effects') else None

    def run_single_experiment(experiment_id: int) -> dict:
        """
        Run a single experiment.
        """
        # Initialize result dictionary
        result_dict = {}

        # Sample data
        purchase_records = dgp.sample(sample_size=sample_size, seed=experiment_id)
        
        # Estimate fixed effects
        estimated_fixed_effects = estimate_fixed_effects(purchase_records, dgp.n_products)
        result_dict['estimated_fixed_effects'] = estimated_fixed_effects
        
        # Optimize assortment using estimated fixed effects
        optimal_assortment, estimated_value = optimize(
            fixed_effects=estimated_fixed_effects,
            capacity=capacity
        )
        
        # Evaluate performance using true fixed effects
        true_value = obj_func(fixed_effects=true_fixed_effects, assortment=optimal_assortment) \
            if true_fixed_effects is not None else None
        
        # Store results
        result_dict['optimal_assortment'] = optimal_assortment
        result_dict['estimated_value'] = estimated_value
        result_dict['true_value'] = true_value
        result_dict['optimism'] = estimated_value - true_value if true_value is not None else None
        
        # Run additional estimators if provided
        if estimators_dict:
            for estimator_name, estimator_info in estimators_dict.items():
                estimator_func = estimator_info['estimator']
                estimator_params = estimator_info['params']
                
                # Apply the estimator
                corrected_fixed_effects = estimator_func(
                    purchase_records=purchase_records, 
                    estimated_fixed_effects=estimated_fixed_effects,
                    **estimator_params
                )
                
                # Optimize with corrected estimates
                corrected_assortment, corrected_estimated_value = optimize(
                    fixed_effects=corrected_fixed_effects,
                    capacity=capacity
                )
                
                # Evaluate with true fixed effects
                corrected_true_value = obj_func(
                    fixed_effects=true_fixed_effects, 
                    assortment=corrected_assortment
                ) if true_fixed_effects is not None else None
                
                # Store results
                result_dict[f'{estimator_name}_assortment'] = corrected_assortment
                result_dict[f'{estimator_name}_estimated_value'] = corrected_estimated_value
                result_dict[f'{estimator_name}_true_value'] = corrected_true_value
                result_dict[f'{estimator_name}_optimism'] = \
                    corrected_estimated_value - corrected_true_value if corrected_true_value is not None else None
        
        return result_dict
    
    # Run experiments
    if verbose:
        print(f'Running {n_repeats} experiments...')
    
    result_records = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(run_single_experiment)(experiment_id) for experiment_id in range(n_repeats)
    )

    return result_records


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
    dgp = MNLAssortment(**dgp_params)

    def run_single_experiment(experiment_id: int) -> dict:
        """
        Run a single experiment.
        """
        # initialize result dictionary
        result_dict ={}

        # sample data
        purchase_records = dgp.sample(sample_size=sample_size, seed=experiment_id)

        # estimate treatment effects
        emp_effects_arr = estimate_fixed_effects(
            purchase_records=purchase_records, n_products=dgp.n_products
        )

        # run no correction estimator for each optimizer
        for optimizer_name in optimization_params.keys():
            optimizer = optimization_params[optimizer_name]['optimizer']
            optimize_params = optimization_params[optimizer_name]['params']

            # select experiments
            assortment, val_est = optimizer(
                fixed_effects=emp_effects_arr, **optimize_params
            )

            # evaluate assortment
            val_true = obj_func(dgp.prod_fixed_effects, assortment)

            # compute the Wald Closure
            wc = val_est - val_true

            # store results
            result_dict[f'{optimizer_name}_assortment'] = assortment
            result_dict[f'{optimizer_name}_val_true'] = val_true
            result_dict[f'{optimizer_name}_val_est'] = val_est
            result_dict[f'{optimizer_name}_wc'] = wc
            
        result_dict['est_fixed_effects'] = emp_effects_arr

        # run all other estimators for each selection methods
        for estimator_name in estimators_dict.keys():
            temp_estimator = estimators_dict[estimator_name]['estimator']
            temp_params = estimators_dict[estimator_name]['params']

            try:
                temp_selection_dict, temp_est_dict = temp_estimator(
                    purchase_records=purchase_records, 
                    emp_fixed_effects=emp_effects_arr,
                    optimization_params=optimization_params, 
                    n_products=dgp.n_products,
                    **temp_params
                )
            except ValueError:
                print(experiment_id)

            if temp_selection_dict is None:
                for optimizer_name in temp_est_dict.keys():
                    result_dict[f'{optimizer_name}_{estimator_name}_val_est'] = temp_est_dict[optimizer_name]
                    result_dict[f'{optimizer_name}_{estimator_name}_wc'] = temp_est_dict[optimizer_name] - result_dict[f'{optimizer_name}_val_true']
            else:
                for optimizer_name in temp_selection_dict.keys():
                    temp_val_true = obj_func(dgp.prod_fixed_effects, temp_selection_dict[optimizer_name])
                    result_dict[f'{optimizer_name}_{estimator_name}_assortment'] = temp_selection_dict[optimizer_name]
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
                    f'{optimizer}_{estimator}_val_true_arr': temp_val_true_arr,
                    f'{optimizer}_{estimator}_val_true_avg': np.nanmean(temp_val_true_arr),
                    f'{optimizer}_{estimator}_val_true_se': np.nanstd(temp_val_true_arr) / sample_size ** 0.5, 
                })

    return wc_measure_dict



##########
# Bootstrap Correction
##########


def get_wc_boot_dstn(
    purchase_records: np.ndarray,
    optimization_params: dict, 
    n_bootstraps: int = 1000, 
    emp_fixed_effects: np.ndarray = None, 
    n_products: int = None,
    n_jobs: int = 1, 
    verbose: bool = False,
    seed: int = None, **kwargs
) -> dict:
    """
    Compute the bootstrap distribution of the Winner's Curse for MNL assortment optimization.

    Params:
    -------
    purchase_records: np.ndarray
        The purchase records of customers, where each entry corresponds to the product index chosen.
    optimization_params: dict
        A dictionary containing the optimization methods to evaluate.
    n_bootstraps: int
        The number of bootstrap samples to generate.
    emp_fixed_effects: np.ndarray
        An array representing the empirical fixed effects for each product.
    n_products: int
        Number of products in the assortment.
    n_jobs: int
        The number of parallel jobs to run.
    verbose: bool
        Whether to display progress
    seed: int
        The random seed.
    
    Returns:
    --------
    dictionary
        A dictionary containing the bootstrap distribution of the Winner's Curse for each optimization method.
    """
    # set random seed
    if seed is not None:
        np.random.seed(seed)

    # compute empirical fixed effects if not provided
    if emp_fixed_effects is None:
        if n_products is None:
            raise ValueError("Must provide either emp_fixed_effects or n_products")
        emp_fixed_effects = estimate_fixed_effects(purchase_records, n_products)

    # get sample size
    sample_size = len(purchase_records)

    def compute_wc(boot_id) -> float:
        """
        Compute the Winner's Curse for a single bootstrap sample.
        """
        # draw bootstrap samples
        boot_indices = np.random.choice(sample_size, size=sample_size, replace=True)
        boot_records = purchase_records[boot_indices]
        
        # estimate fixed effects from bootstrap sample
        boot_fixed_effects = estimate_fixed_effects(boot_records, len(emp_fixed_effects))

        # optimization
        wc_dict = {}
        for optimizer_name in optimization_params.keys():
            optimizer = optimization_params[optimizer_name]['optimizer']
            optimize_params = optimization_params[optimizer_name]['params']

            # select assortment using bootstrap estimates
            boot_assortment, boot_val_est = optimizer(
                fixed_effects=boot_fixed_effects,
                **optimize_params
            )

            # evaluate assortment using empirical estimates
            emp_val_est = obj_func(fixed_effects=emp_fixed_effects, assortment=boot_assortment)

            # compute the Winner's Curse
            wc_dict[optimizer_name] = boot_val_est - emp_val_est

        return wc_dict
    
    # compute the Winner's Curse for each bootstrap sample
    boot_wc_records = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(compute_wc)(boot_id) for boot_id in range(n_bootstraps)
    )

    # extract the Winner's Curse distribution for each optimization method
    boot_wc_dstn_dict = {
        optimizer_name: np.array([record[optimizer_name] for record in boot_wc_records])
        for optimizer_name in optimization_params.keys()
    }

    return boot_wc_dstn_dict


def get_wc_m_out_of_n_boot_dstn(
    purchase_records: np.ndarray,
    optimization_params: dict, 
    emp_fixed_effects: np.ndarray = None,
    n_products: int = None,
    n_bootstraps: int = 1000, 
    power: float = 0.95, 
    n_jobs: int = 1, 
    verbose: bool = False,
    seed: int = None, **kwargs
) -> dict:
    """
    Compute the m-out-of-n bootstrap distribution of the Winner's Curse for MNL assortment optimization.

    Params:
    -------
    purchase_records: np.ndarray
        The purchase records of customers, where each entry corresponds to the product index chosen.
    optimization_params: dict
        A dictionary containing the optimization methods to evaluate.
    emp_fixed_effects: np.ndarray
        An array representing the empirical fixed effects for each product.
    n_products: int
        Number of products in the assortment.
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
    dictionary
        A dictionary containing the bootstrap distribution of the Winner's Curse for each optimization method.
    """
    # parameter check 
    assert 0.0 < power < 1.0, 'The power must be in the range (0, 1).'

    # set random seed
    if seed is not None:
        np.random.seed(seed)

    # compute empirical fixed effects if not provided
    if emp_fixed_effects is None:
        if n_products is None:
            raise ValueError("Must provide either emp_fixed_effects or n_products")
        emp_fixed_effects = estimate_fixed_effects(purchase_records, n_products)

    # get sample size
    sample_size = len(purchase_records)

    def compute_wc(boot_id) -> float:
        """
        Compute the Winner's Curse for a single bootstrap sample.
        """
        # draw bootstrap samples
        boot_sample_size = int(sample_size ** power)
        boot_indices = np.random.choice(sample_size, size=boot_sample_size, replace=True)
        boot_records = purchase_records[boot_indices]
        
        # estimate fixed effects from bootstrap sample
        boot_fixed_effects = estimate_fixed_effects(boot_records, len(emp_fixed_effects))

        # optimization
        wc_dict = {}
        for optimizer_name in optimization_params.keys():
            optimizer = optimization_params[optimizer_name]['optimizer']
            optimize_params = optimization_params[optimizer_name]['params']

            # select assortment using bootstrap estimates
            boot_assortment, boot_val_est = optimizer(
                fixed_effects=boot_fixed_effects,
                **optimize_params
            )

            # evaluate assortment using empirical estimates
            emp_val_est = obj_func(fixed_effects=emp_fixed_effects, assortment=boot_assortment)

            # compute the Winner's Curse
            wc_dict[optimizer_name] = boot_val_est - emp_val_est

        return wc_dict
    
    # compute the Winner's Curse for each bootstrap sample
    boot_wc_records = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(compute_wc)(boot_id) for boot_id in range(n_bootstraps)
    )

    # extract the Winner's Curse distribution for each optimization method
    boot_wc_dstn_dict = {
        optimizer_name: np.array([record[optimizer_name] for record in boot_wc_records])
        for optimizer_name in optimization_params.keys()
    }

    return boot_wc_dstn_dict


def get_wc_num_boot_dstn(
    purchase_records: np.ndarray,
    optimization_params: dict, 
    emp_fixed_effects: np.ndarray = None,
    n_products: int = None,
    n_bootstraps: int = 1000, 
    power: float = -0.45, 
    n_jobs: int = 1, 
    verbose: bool = False,
    seed: int = None, **kwargs
) -> dict:
    """
    Compute the numerical bootstrap distribution of the Winner's Curse for MNL assortment optimization.

    Params:
    -------
    purchase_records: np.ndarray
        The purchase records of customers, where each entry corresponds to the product index chosen.
    optimization_params: dict
        A dictionary containing the optimization methods to evaluate.
    emp_fixed_effects: np.ndarray
        An array representing the empirical fixed effects for each product.
    n_products: int
        Number of products in the assortment.
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
    dictionary
        A dictionary containing the bootstrap distribution of the Winner's Curse for each optimization method.
    """
    # parameter checks
    assert 0.0 > power > -0.5, "Power must be between 0 and -0.5"

    # set random seed
    if seed is not None:
        np.random.seed(seed)

    # compute empirical fixed effects if not provided
    if emp_fixed_effects is None:
        if n_products is None:
            raise ValueError("Must provide either emp_fixed_effects or n_products")
        emp_fixed_effects = estimate_fixed_effects(purchase_records, n_products)

    # get sample size
    sample_size = len(purchase_records)

    # compute perturbation parameter
    epsilon_n = sample_size ** power

    def compute_wc(boot_id) -> float:
        """
        Compute the Winner's Curse for a single bootstrap sample.
        """
        # draw bootstrap samples
        boot_indices = np.random.choice(sample_size, size=sample_size, replace=True)
        boot_records = purchase_records[boot_indices]
        
        # estimate fixed effects from bootstrap sample
        boot_fixed_effects = estimate_fixed_effects(boot_records, len(emp_fixed_effects))

        # compute perturbed fixed effect estimates
        norm_error = np.sqrt(sample_size) * (boot_fixed_effects - emp_fixed_effects)
        perturbed_fixed_effects = emp_fixed_effects + epsilon_n * norm_error

        # optimization
        wc_dict = {}
        for optimizer_name in optimization_params.keys():
            optimizer = optimization_params[optimizer_name]['optimizer']
            optimize_params = optimization_params[optimizer_name]['params']

            # select assortment using bootstrap estimates
            boot_assortment, boot_val_est = optimizer(
                fixed_effects=boot_fixed_effects,
                **optimize_params
            )

            # evaluate assortment using empirical and perturbed estimates
            emp_val_est = obj_func(fixed_effects=emp_fixed_effects, assortment=boot_assortment)
            perturbed_val_est = obj_func(fixed_effects=perturbed_fixed_effects, assortment=boot_assortment)

            # compute the Winner's Curse
            wc_dict[optimizer_name] = perturbed_val_est - emp_val_est

        return wc_dict
    
    # compute the Winner's Curse for each bootstrap sample
    boot_wc_records = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(compute_wc)(boot_id) for boot_id in range(n_bootstraps)
    )

    # extract the Winner's Curse distribution for each optimization method
    boot_wc_dstn_dict = {
        optimizer_name: np.array([record[optimizer_name] for record in boot_wc_records])
        for optimizer_name in optimization_params.keys()
    }

    return boot_wc_dstn_dict


def bootstrap_correction_estimate(
    purchase_records: np.ndarray,
    optimization_params: dict, 
    emp_fixed_effects: np.ndarray = None,
    n_products: int = None,
    bootstrap_method: str = 'standard', 
    **kwargs, 
) -> tuple:
    """
    Estimate the Winner's Curse using bootstrap correction for MNL assortment optimization.

    Params:
    -------
    purchase_records: np.ndarray
        The purchase records of customers, where each entry corresponds to the product index chosen.
    optimization_params: dict
        A dictionary containing the optimization methods to evaluate.
    emp_fixed_effects: np.ndarray
        An array representing the empirical fixed effects for each product.
    n_products: int
        Number of products in the assortment.
    bootstrap_method: str
        The bootstrap method to use. Options are 'standard', 'm_out_of_n', and 'numerical'.
    **kwargs
        Additional keyword arguments for the bootstrap method.
    
    Returns:
    --------
    tuple: selection_dict, boot_est_dict
        A tuple containing the selection dictionary and the bootstrap-corrected policy value estimate dictionary.
    """
    # compute empirical fixed effects if not provided
    if emp_fixed_effects is None:
        if n_products is None:
            raise ValueError("Must provide either emp_fixed_effects or n_products")
        emp_fixed_effects = estimate_fixed_effects(purchase_records, n_products)

    # optimize selection
    selection_dict = {}
    nc_est_dict = {}
    for optimizer_name, optimizer_dict in optimization_params.items():
        optimizer = optimizer_dict['optimizer']
        optimize_params = optimizer_dict['params']
        
        # Get optimal assortment and estimated value
        assortment, val_est = optimizer(
            fixed_effects=emp_fixed_effects,
            **optimize_params
        )
        
        selection_dict[optimizer_name] = assortment
        nc_est_dict[optimizer_name] = val_est

    # get the bootstrap distribution of the Winner's Curse for each selection method
    boot_dstn_dict = {
        'standard': get_wc_boot_dstn,
        'm_out_of_n': get_wc_m_out_of_n_boot_dstn,
        'numerical': get_wc_num_boot_dstn,
    }[bootstrap_method](
        purchase_records=purchase_records, 
        optimization_params=optimization_params, 
        emp_fixed_effects=emp_fixed_effects,
        n_products=n_products,
        **kwargs
    )

    # compute the bootstrap-corrected policy value estimate for each selection method
    boot_est_dict = {
        optimizer_name: nc_est_dict[optimizer_name] - boot_dstn_dict[optimizer_name].mean()
        for optimizer_name in optimization_params.keys()
    }

    return None, boot_est_dict


##########
# Sample Splitting
##########

def sample_splitting_estimate(
    purchase_records: np.ndarray,
    optimization_params: dict, 
    emp_fixed_effects: np.ndarray = None,
    n_products: int = None,
    estimation_split: float = 0.5,
    seed: int = None,
    **kwargs,  
) -> tuple: 
    """
    Estimate the policy value using sample splitting for MNL assortment optimization.

    Params:
    -------
    purchase_records: np.ndarray
        The purchase records of customers, where each entry corresponds to the product index chosen.
    optimization_params: dict
        A dictionary containing the optimization methods to evaluate.
    emp_fixed_effects: np.ndarray, optional
        Precomputed empirical fixed effects (not used in sample splitting but kept for API consistency)
    n_products: int, optional
        Number of products in the assortment. Required if emp_fixed_effects is None.
    estimation_split: float
        The proportion of samples to use for estimation.
    seed: int, optional
        Random seed for reproducibility.
        
    Returns:
    --------
    tuple: selection_dict, val_est_dict
        A tuple containing the selection dictionary and the policy value estimate dictionary.
    """
    # parameter check 
    assert 0.0 < estimation_split < 1.0, 'The estimation split must be in the range (0, 1).'

    # set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # split data into estimation and evaluation sets
    sample_size = len(purchase_records)
    est_size = int(sample_size * estimation_split)
    est_indices = np.random.choice(sample_size, size=est_size, replace=False)
    eval_indices = np.setdiff1d(np.arange(sample_size), est_indices)
    
    est_records = purchase_records[est_indices]
    eval_records = purchase_records[eval_indices]

    # estimate fixed effects on estimation set
    est_fixed_effects = estimate_fixed_effects(est_records, n_products)

    # optimize selection based on estimation set
    selection_dict = {}
    for optimizer_name, optimizer_dict in optimization_params.items():
        optimizer = optimizer_dict['optimizer']
        optimize_params = optimizer_dict['params']
        
        # Get optimal assortment and estimated value
        assortment, _ = optimizer(
            fixed_effects=est_fixed_effects,
            **optimize_params
        )
        
        selection_dict[optimizer_name] = assortment

    # estimate fixed effects on evaluation set
    eval_fixed_effects = estimate_fixed_effects(eval_records, n_products)

    # evaluate selected assortments using evaluation set fixed effects
    val_est_dict = {
        optimizer_name: obj_func(
            fixed_effects=eval_fixed_effects, 
            assortment=selection_dict[optimizer_name]
        )
        for optimizer_name in optimization_params.keys()
    }
    
    return selection_dict, val_est_dict


##########
# Bayesian MNL
##########

def bayesian_mnl_estimate(
    purchase_records: np.ndarray,
    optimization_params: dict, 
    n_products: int = None,
    post_sample_size: int = 1000,
    n_warmup: int = 500,
    seed: int = None,
    **kwargs,
) -> tuple:
    """
    Estimate the policy value using Bayesian MNL for assortment optimization.

    Params:
    -------
    purchase_records: np.ndarray
        The purchase records of customers, where each entry corresponds to the product index chosen.
    optimization_params: dict
        A dictionary containing the optimization methods to evaluate.
    emp_fixed_effects: np.ndarray, optional
        Precomputed empirical fixed effects (not used in Bayesian MNL but kept for API consistency)
    n_products: int, optional
        Number of products in the assortment. Required if emp_fixed_effects is None.
    post_sample_size: int
        The number of posterior samples to draw.
    n_warmup: int
        The number of warmup steps for the MCMC sampler.
    seed: int, optional
        Random seed for reproducibility.
        
    Returns:
    --------
    tuple: selection_dict, val_est_dict
        A tuple containing the selection dictionary and the policy value estimate dictionary.
    """
    # set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # estimate fixed effects
    post_sample = estimate_fixed_effects_bayes(
        purchase_records=purchase_records,
        n_products=n_products,
        n_warmup=n_warmup,
        post_sample_size=post_sample_size,
        **kwargs
    )  # shape = (post_sample_size, n_products)

    # optimize selection based on posterior samples
    selection_dict = {}
    val_est_dict = {}
    for optimizer_name, optimizer_dict in optimization_params.items():
        optimize_params = optimizer_dict['params']
        
        # Get optimal assortment and estimated value
        bayes_assortment, bayes_val_est = optimize_bayes(
            post_sample=post_sample,
            **optimize_params
        )
        
        selection_dict[optimizer_name] = bayes_assortment
        val_est_dict[optimizer_name] = bayes_val_est

    return selection_dict, val_est_dict
