import os 
import sys 
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
from typing import Iterable
from datetime import datetime

from core.dgp import DGP1

def get_best_targeting_val(X: np.ndarray, dgp: DGP1, budget: int = 1) -> float:
    true_est = np.zeros(X.shape)
    true_group = np.array([dgp.group_func(x) for x in X])
    
    for i in range(len(dgp.te_list)):
        true_est[true_group == i] = dgp.te_list[i]

    true_val = -np.sort(-true_est)[:budget].sum()

    return true_val

def evaluate_targeting_policy(X: np.ndarray, targeting_arr: np.ndarray, dgp: DGP1, budget: int = 1) -> float:
    """  
    Calculate the actual targeting value of a given targeting decision

    params:
    -------
    X: np.ndarray, shape (n_samples, d), the covariates
    targeting_arr: np.ndarray, shape (n_samples, ), the targeting decision
    dgp: DGP1, the data generating process
    budget: int, the number of customers to target
    """
    assert X.shape[0] == targeting_arr.shape[0], 'X and targeting_arr must have the same number of samples'
    assert targeting_arr.sum() <= budget, 'The sum of the targeting array must be within the budget'

    true_est = np.zeros(X.shape)
    true_group = np.array([dgp.group_func(x) for x in X])

    for i in range(len(dgp.te_list)):
        true_est[true_group == i] = dgp.te_list[i]

    true_val = true_est[targeting_arr == 1].sum()

    return true_val

def single_experiment(
    dgp: DGP1,
    sample_size: int, 
    targeting_params: dict, 
    estimators_dict: dict, 
    test_data: np.ndarray, 
    save_estimator: bool = False, 
    seed=None
) -> dict:
    """  
    Run a single experiment

    Params:
    -------
    dgp: DGP1, the data generating process
    sample_size: int, the number of training samples
    estimators_dict: dict, dictionary containing the estimators and their parameters
    test_data: np.ndarray, the targeting individuals
    save_estimator: bool, whether to save the estimator
    seed: int, random seed
    
    Returns:
    --------
    dict: Dictionary containing the following keys:
        - best_targ_val: float, the best targeting value
        - plugin_targ_est: float, the plugin targeting estimate
        - true_plugin_targ_val: float, the actual value of the plugin targeting decision
        - param_targ_est: float, the parametric targeting estimate
        - bayes_param_targ_est: float, the bayesian parametric targeting estimate
        - boot_correction_target_est: float, the bootstrap correction targeting estimate
    """
    # set seed
    seed = seed if seed is not None else np.random.randint(0, 1e6)

    # initialize result dictionary
    result_dict = {name: None for name in estimators_dict.keys()}

    # generate training data and targeting individuals
    train_x, train_t, train_y = dgp.generate_training_data(sample_size, seed=seed)  # generate training data
    # if test_data is None:
    #     test_data = dgp.generate_testing_data(targeting_params['size'], seed=seed)  # generate targeting individuals

    # define and train estimators
    for estimator_name, estimator_dict in estimators_dict.items():
        estimator = estimator_dict['estimator'](
            group_func=dgp.group_func, n_groups=len(dgp.te_list)
        ).fit(X=train_x, T=train_t, Y=train_y, **estimator_dict['train_params'])
        targ_est = estimator.estimate_targeting_value(
            X=test_data, budget=targeting_params['budget'], 
            **estimator_dict['targeting_params']
        )
        if save_estimator:
            result_dict[estimator_name] = {
                'estimator': estimator, 'targ_est': targ_est
            }
        else:
            result_dict[estimator_name] = {'targ_est': targ_est}

        if estimator_name == 'plugin':
            result_dict[estimator_name]['act_targ_val'] = evaluate_targeting_policy(
                X=test_data, targeting_arr=estimator.get_targeting_decision(test_data, targeting_params['budget']), 
                dgp=dgp, budget=targeting_params['budget']
            )

    # result_dict['clairvoyant'] = get_best_targeting_val(test_data, dgp, budget=targeting_params['budget'])

    return result_dict


def grid_experiment(
    sample_sizes: Iterable[int], 
    num_bootstraps: Iterable[int],
    noise_stds: Iterable[float],
    te_diffs: Iterable[float],
    dgp_params: dict,
    targeting_params: dict,
    estimators_dict: dict,
    save_dir: str, 
    num_test_groups: int,
    num_repeats_per_test_group: int,
    verbose: bool = False
) -> list:
    """  
    Run a grid of experiments

    Params:
    -------
    sample_sizes: Iterable[int], the sample sizes to consider
    num_bootstraps: Iterable[int], the number of bootstrap samples to consider
    noise_stds: Iterable[float], the standard deviations of the noise to consider
    te_diffs: Iterable[float], the treatment effect differences to consider
    dgp_params: dict, parameters for the data generating process
    targeting_params: dict, parameters for the targeting policy
    estimators_dict: dict, dictionary containing the estimators and their parameters
    save_dir: str, the directory to save the results
    num_test_groups: int, the number of test groups
    num_repeats_per_test_group: int, the number of times to repeat the experiment for each test group
    verbose: bool, whether to print progress
    
    Returns:
    --------
    list: List of dictionaries containing the results of the experiments
    """
    results = [
        [None for _ in range(num_repeats_per_test_group)]
        for _ in range(num_test_groups)
    ]  # shape: (num_test_groups, num_repeats_per_test_group)

    # create an array to store the grid of experiment parameters
    knob_dict = {
        'sample_size': sample_sizes, 'num_bootstrap': num_bootstraps, 
        'noise_std': noise_stds, 'te_diff': te_diffs
    }
    knob_val_grids = np.array([
        [sample_size, num_bootstrap, noise_std, te_diff]  # * Must be in the same order as knob_keys
        for sample_size in sample_sizes
        for num_bootstrap in num_bootstraps
        for noise_std in noise_stds
        for te_diff in te_diffs
    ])

    # check if any of the knobs have a single value
    fixed_knob_keys = [
        knob_key for knob_key, knob_val_list in knob_dict.items() if len(knob_val_list) == 1
    ]

    # print knob keys and values for the ones that have a single value
    if len(fixed_knob_keys) > 0:
        print("Fixed knob values: ")
        for knob_key in fixed_knob_keys:
            print(f"{knob_key}={knob_dict[knob_key][0]}")

    for knob_arr in knob_val_grids:
        sample_size, num_bootstrap, noise_std, te_diff = knob_arr
        start = datetime.now()
        dgp_params['train_size'] = int(sample_size)
        dgp_params['noise_std'] = noise_std
        dgp_params['te_diff'] = te_diff

        # initialize the data generating process
        dgp = DGP1(**dgp_params)

        # generate testing data
        for test_group_id in range(num_test_groups):
            # * test_data is generated outside the repeat loop to ensure that the same test individuals are used for all replicates
            test_data = dgp.generate_testing_data(targeting_params['size'])  # generate targeting individuals
            
            for repeat_id in range(num_repeats_per_test_group):
                # change n_bootstrap parameters for the correction estimators
                for estimator_name in estimators_dict.keys():
                    if 'n_bootstrap' in estimators_dict[estimator_name]['train_params'].keys():
                        estimators_dict[estimator_name]['train_params']['n_bootstrap'] = int(num_bootstrap)
                
                # run experiments for the current parameter values
                result_dict = single_experiment(
                    dgp=dgp, 
                    sample_size=int(sample_size),
                    targeting_params=targeting_params, 
                    estimators_dict=estimators_dict, 
                    test_data=test_data, 
                    save_estimator=False, 
                    seed=repeat_id
                )

                # add the parameter values to the result dictionary
                for knob_key, knob_val in zip(knob_dict.keys(), knob_arr):
                    result_dict[knob_key] = knob_val
                results[test_group_id][repeat_id] = result_dict

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
            'act_plugin_val': result_dict['plugin']['act_targ_val'], 
        }
        for result_dict in results
    ])

    result_df.to_csv(save_dir, index=False)

    return result_df
