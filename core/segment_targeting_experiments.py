import os 
import sys 
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
from typing import Iterable
from tqdm import tqdm 
from datetime import datetime

from core.dgp import SegmentTargetingDGP
from core.segment_targeting_estimators import segment_targeting_obj_func


def single_segment_targeting_experiment(
    dgp: SegmentTargetingDGP,
    train_covariates: np.ndarray,
    train_treatments: np.ndarray,
    train_outcomes: np.ndarray, 
    test_customers: np.ndarray, 
    targeting_params: dict, 
    estimators_dict: dict, 
    save_estimator: bool = False, 
) -> dict:
    """  
    Run a single experiment for segment targeting

    Params:
    -------
    dgp: SegmentTargetingDGP
        Data generating process for segment targeting
    train_covariates: np.ndarray
        Covariates of customer characteristics in the training set
    train_treatments: np.ndarray
        Treatments of customers in the training set
    train_outcomes: np.ndarray
        Outcomes of customers in the training set
    test_customers: np.ndarray
        Covariates of customers to be targeted
    targeting_params: dict
        Parameters for targeting
    estimators_dict: dict
        Dictionary of estimators to be evaluated
    save_estimator: bool
        Whether to save the estimator object in the result dictionary

    Returns:
    --------
    result_dict: dict
        Dictionary of results for each estimator

    """
    # estimators_dict must contain plugin estimator
    assert 'plugin' in estimators_dict.keys()

    # initialize result dictionary
    result_dict = {name: None for name in estimators_dict.keys()}

    # fit and optimzie plugin estimator
    plugin_estmr = estimators_dict['plugin']['estimator'].fit(
        covariates=train_covariates, 
        treatments=train_treatments,
        outcomes=train_outcomes,
        **estimators_dict['plugin']['train_params']
    )
    plugin_decision, plugin_targ_est = plugin_estmr.optimize(test_customers=test_customers, **targeting_params)
    true_plugin_targ_val = segment_targeting_obj_func(
        test_customers=test_customers, 
        targ_decision=plugin_decision, 
        params=dgp.segment_te_arr, 
        segment_func=dgp.segment_func
    )

    # update result dictionary
    result_dict['plugin'] = {
        'targ_est': plugin_targ_est, 'act_targ_val': true_plugin_targ_val, 
        'wc': plugin_targ_est - true_plugin_targ_val
    }
    if save_estimator: 
        result_dict['plugin']['estimator'] = plugin_estmr

    # define and train estimators
    for estimator_name, estimator_dict in estimators_dict.items():
        if estimator_name == 'plugin':
            continue
        # fit estimators
        estimator = estimator_dict['estimator'].fit(
            covariates=train_covariates, 
            treatments=train_treatments,
            outcomes=train_outcomes,
            **estimator_dict['train_params']
        )

        targ_est = estimator.estimate(
            test_customers=test_customers,
            plugin_estmr=plugin_estmr, 
            **estimator_dict['targeting_params'], 
            **targeting_params
        )

        result_dict[estimator_name] = {'targ_est': targ_est, 'wc': targ_est - true_plugin_targ_val}
        if save_estimator:
            result_dict[estimator_name]['estimator'] = estimator

    return result_dict


def grid_experiment(
    sample_sizes: Iterable[int], 
    n_bootstraps: Iterable[int],
    noise_stds: Iterable[float],
    te_diffs: Iterable[float],
    dgp_params: dict,
    targeting_params: dict,
    estimators_dict: dict,
    num_train_samples: int,
    num_test_customers_per_train_sample: int,
    save_dir: str = None, 
    verbose: bool = False
) -> list:
    """  
    Run a grid of experiments

    Params:
    -------
    sample_sizes: Iterable[int], the sample sizes to consider
    n_bootstraps: Iterable[int], the number of bootstrap samples to consider
    noise_stds: Iterable[float], the standard deviations of the noise to consider
    te_diffs: Iterable[float], the treatment effect differences to consider
    dgp_params: dict, parameters for the data generating process
    targeting_params: dict, parameters for the targeting policy
    estimators_dict: dict, dictionary containing the estimators and their parameters
    save_dir: str, the directory to save the results
    num_train_samples: int, the number of training samples to generate
    num_test_customers_per_train_sample: int, the number of test customers to generate per training sample
    verbose: bool, whether to print progress
    
    Returns:
    --------
    list: List of dictionaries containing the results of the experiments
    """
    results = []

    # create an array to store the grid of experiment parameters
    knob_dict = {
        'sample_size': sample_sizes, 'n_bootstrap': n_bootstraps, 
        'noise_std': noise_stds, 'te_diff': te_diffs
    }
    knob_val_grids = np.array([
        [sample_size, n_bootstrap, noise_std, te_diff]  # * Must be in the same order as knob_keys
        for sample_size in sample_sizes
        for n_bootstrap in n_bootstraps
        for noise_std in noise_stds
        for te_diff in te_diffs
    ])

    # check if any of the knobs have a single value
    fixed_knob_keys = [
        knob_key for knob_key, knob_val_list in knob_dict.items() if len(knob_val_list) == 1
    ]

    # print knob keys and values for the ones that have a single value
    if len(fixed_knob_keys) > 0:
        if verbose: 
            print("Fixed knob values: ")
            for knob_key in fixed_knob_keys:
                print(f"{knob_key}={knob_dict[knob_key][0]}")

    for knob_arr in knob_val_grids:
        sample_size, n_bootstrap, noise_std, te_diff = knob_arr
        start = datetime.now()
        dgp_params['train_size'] = int(sample_size)
        dgp_params['noise_std'] = noise_std
        dgp_params['te_diff'] = te_diff

        # initialize the data generating process
        dgp = SegmentTargetingDGP(**dgp_params)

        # generate testing data
        for train_sample_id in tqdm(range(num_train_samples)):
            # generate training data
            train_covariates, train_treatments, train_outcomes = dgp.generate_training_data(int(sample_size))

            # fit estimators: estimate treatment effects
            for name, estimator_dict in estimators_dict.items():
                if 'n_bootstrap' in estimator_dict.keys():
                    estimator_dict['train_params']['n_bootstrap'] = n_bootstrap

                estimator_dict['instance'] = estimator_dict['estimator'](
                    **estimator_dict['init_params']
                ).fit(
                    covariates=train_covariates, 
                    treatments=train_treatments,
                    outcomes=train_outcomes,
                    **estimator_dict['train_params']
                )

            for test_customer_id in range(num_test_customers_per_train_sample):
                # initialize result dictionary
                result_dict = {name: None for name in estimators_dict.keys()}

                # generate test customers
                test_customers = dgp.sample_individuals(int(targeting_params['size']))

                # optimize plugin estimator
                plugin_decision, plugin_targ_est = estimators_dict['plugin']['instance'].optimize(
                    test_customers=test_customers, **{k: v for k, v in targeting_params.items() if k != 'size'}
                )
                true_plugin_targ_val = segment_targeting_obj_func(
                    test_customers=test_customers, 
                    targ_decision=plugin_decision, 
                    params=dgp.segment_te_arr, 
                    segment_func=dgp.segment_func
                )
                result_dict['plugin'] = {
                    'targ_est': plugin_targ_est, 'act_targ_val': true_plugin_targ_val, 
                    'wc': plugin_targ_est - true_plugin_targ_val
                }

                # estimate targeting values for other estimators
                for name, estimator_dict in estimators_dict.items():
                    if name == 'plugin':
                        continue

                    targ_est = estimator_dict['instance'].estimate(
                        test_customers=test_customers,
                        plugin_estmr=estimators_dict['plugin']['instance'], 
                        **estimator_dict['targeting_params'], 
                        **{k: v for k, v in targeting_params.items() if k != 'size'}
                    )

                    result_dict[name] = {'targ_est': targ_est, 'wc': targ_est - true_plugin_targ_val}

                # add the sample and customer ids to the result dictionary
                result_dict.update({'train_sample_id': train_sample_id, 'test_customer_id': test_customer_id})

                # add the parameter values to the result dictionary
                for knob_key, knob_val in zip(knob_dict.keys(), knob_arr):
                    result_dict[knob_key] = knob_val

                results.append(result_dict)

        if verbose:
            # print parameter values except for the fixed ones
            for knob_key, knob_val in zip(knob_dict.keys(), knob_arr):
                if knob_key not in fixed_knob_keys:
                    print(f"{knob_key}={knob_val},", end=' '), 
            print(f"Time taken: {datetime.now() - start}")

    # organize results into a dataframe
    result_df = pd.DataFrame.from_records([
        {
            **{knob_key: result_dict[knob_key] for knob_key in knob_dict.keys()},
            **{
                f"{estimator_name}_est": result_dict[estimator_name]['targ_est'] 
                for estimator_name in estimators_dict.keys()
            }, 
            **{
                f"{estimator_name}_wc": result_dict[estimator_name]['wc'] 
                for estimator_name in estimators_dict.keys()
            }, 
            'train_sample_id': result_dict['train_sample_id'],
            'test_customer_id': result_dict['test_customer_id'],
            'act_plugin_val': result_dict['plugin']['act_targ_val'], 
        }
        for result_dict in results
    ])

    if save_dir is not None:
        result_df.to_csv(save_dir, index=False)

    return result_df
