import os 
import sys 
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
from typing import Iterable
from datetime import datetime

from core.dgp import PersonalizedPricingDGP
from core.pricing_estimators import obj_func


def single_pricing_experiment(
    dgp: PersonalizedPricingDGP,
    estimators_dict: dict, 
    train_covariates: np.ndarray,
    train_prices: np.ndarray,
    train_outcomes: np.ndarray,
    customer_to_target: np.ndarray, 
    save_estimator: bool = False, 
) -> dict:
    """  
    Run a single experiment

    Params:
    -------
    dgp: PersonalizedPricingDGP 
        The data generating process
    estimators_dict: dict
        The dictionary of estimators
    customer_to_target: np.ndarray
        The characteristic of the customer to target
    save_estimator: bool
        Whether to save the estimator
    
    Returns:
    --------
    dict: 
    """
    # initialize result dictionary
    result_dict = {name: None for name in estimators_dict.keys()}

    # define and train estimators
    for estimator_name, estimator_dict in estimators_dict.items():
        # initialize and fit estimator
        estimator = estimator_dict['estimator'](
            cov_dim=dgp.cov_dim
        ).fit(
            covariates=train_covariates, prices=train_prices, 
            outcomes=train_outcomes, **estimator_dict['train_params']
        )

        # estimate targeting value
        if estimator_name == 'plugin':
            plugin_opt_price, targ_est = estimator.optimize(
                covariates=customer_to_target, **estimator_dict['targeting_params']
            )
        else:
            targ_est = estimator.estimate_targeting_value(covariates=customer_to_target, **estimator_dict['targeting_params'])
            
        if save_estimator:
            result_dict[estimator_name] = {
                'estimator': estimator, 'targ_est': targ_est
            }
        else:
            result_dict[estimator_name] = {'targ_est': targ_est}

        if estimator_name == 'plugin':
            result_dict[estimator_name]['act_targ_val'] = obj_func(
                covariates=customer_to_target, price=plugin_opt_price, params=dgp.true_util_params, add_const=False
            )
            result_dict[estimator_name]['plugin_opt_price'] = plugin_opt_price

    return result_dict


def grid_pricing_experiment(
    sample_sizes: Iterable[int],
    num_bootstraps: Iterable[int],
    dgp_params: dict, 
    estimators_dict: dict, 
    num_training_samples: int, 
    num_test_customers_per_train_sample: int, 
    verbose: bool = False, 
    save_dir: str = None, 
) -> list:
    """ 
    Run a grid of pricing experiments

    Params:
    -------
    sample_sizes: Iterable[int]
        The sample sizes to use
    num_bootstraps: Iterable[int]
        The number of bootstraps to use
    dgp_params: dict
        The parameters of the data generating process
    estimators_dict: dict
        The dictionary of estimators
    num_training_samples: int
        The number of training samples
    num_test_customers_per_train_sample: int
        The number of test customers per training sample
    verbose: bool, default to False
        Whether to print the results
    save_dir: str, default to None
        The directory to save the results

    Returns:
    --------
    list: 
        list of dictionaries containing the results
    """
    # initialize the results list
    results_list = []

    # create a dictionary to store the grid of experiment parameters
    knob_dict= {
        'sample_size': sample_sizes,
        'num_bootstraps': num_bootstraps
    }
    knob_val_grids = np.array([
        [sample_size, num_bootstrap]
        for sample_size in knob_dict['sample_size']
        for num_bootstrap in knob_dict['num_bootstraps']
    ])  # * Must be in the same order as knob_keys

    # check if any of the kbos have a single value
    fixed_knob_keys = [
        knob_key for knob_key, knob_val_list in knob_dict.items() if len(knob_val_list) == 1
    ]

    # print knob keys and values for the ones that have a single value
    if len(fixed_knob_keys) > 0:
        if verbose:
            print(f"Fixed knobs: {fixed_knob_keys}")
            for knob_key in fixed_knob_keys:
                print(f"{knob_key}: {knob_dict[knob_key]}")

    # start grid experiments
    for knob_arr in knob_val_grids:
        sample_size, num_bootstrap = knob_arr 
        start = datetime.now()
        
        # initialize data generating process
        dgp = PersonalizedPricingDGP(**dgp_params)

        # generate num_training_samples training data
        for train_sample_id in range(num_training_samples):
            # generate training data 
            cov_arr, price_arr, outcome_arr = dgp.generate_training_data(sample_size=sample_size)

            for test_customer_id in range(num_test_customers_per_train_sample):
                flag = True  # flag to check if the plugin optimal price is on the boundary
                while flag:
                # sample a test customer
                    test_customer = dgp.sample_individuals(sample_size=1)

                    # run a single pricing experiment
                    try:
                        result_dict = single_pricing_experiment(
                            dgp=dgp, 
                            train_covariates=cov_arr, train_prices=price_arr, train_outcomes=outcome_arr,
                            estimators_dict=estimators_dict, 
                            customer_to_target=test_customer
                        )
                    except ValueError:
                        continue

                    # update flag
                    flag = np.isclose(result_dict['plugin']['plugin_opt_price'], dgp.price_ub, rtol=1e-3)

                if result_dict is None:
                    continue

                # add the sample and customer ids to the result dictionary
                result_dict.update({'train_sample_id': train_sample_id, 'test_customer_id': test_customer_id})

                # add the parameter values to the result dictionary
                for knob_key, knob_val in zip(knob_dict.keys(), knob_arr):
                    result_dict[knob_key] = knob_val

                # save the results to list
                results_list.append(result_dict)

        # print the time taken
        if verbose: 
            for knob_key, knob_val in zip(knob_dict.keys(), knob_arr):
                if knob_key not in fixed_knob_keys:
                    print(f"{knob_key}: {knob_val}", end=' ')
            print(f"Time taken: {datetime.now() - start}")

    # save the results
    results_df = pd.DataFrame.from_records([
        {
            **{knob_key: result_dict[knob_key] for knob_key in knob_dict.keys()},
            **{
                f"{estimator_name}_est": result_dict[estimator_name]['targ_est'] 
                for estimator_name in estimators_dict.keys()
            }, 
            'train_sample_id': result_dict['train_sample_id'],
            'test_customer_id': result_dict['test_customer_id'],
            'act_plugin_val': result_dict['plugin']['act_targ_val'], 
        }
        for result_dict in results_list
    ])

    
    if save_dir is not None:
        results_df.to_csv(save_dir, index=False)

    return results_df 