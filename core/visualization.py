import os 
import sys 
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

from scipy.stats import t

def plot_winners_curse_hist(result_dir: str, estimators: list = None, **kwargs):
    # read data
    result_df = pd.read_csv(result_dir)

    # get estimators 
    full_estimators = [col[:-4] for col in result_df.columns if '_est' in col]
    if estimators is None:
        estimators = full_estimators


    # fixed knobs keys 
    target_columns = [f'{estimator}_est' for estimator in full_estimators] + ['act_plugin_val', 'test_group_id', 'repeat_id']
    fixed_knob_keys = [col for col in result_df.columns if col not in target_columns]
    fixed_knob_vals = [result_df[key].iloc[0] if key not in kwargs.keys() else kwargs[key] for key in fixed_knob_keys]

    # print the fixed knobs
    print('Fixed Knobs: ', end=' ')
    for key, val in zip(fixed_knob_keys, fixed_knob_vals):
        print(f"{key}: {val},", end=' ')

    # extract the data for the fixed knobs
    fixed_knob_mask_list = np.array([
        (result_df[key] == val).values for key, val in zip(fixed_knob_keys, fixed_knob_vals)
    ])
    result_df = result_df[pd.Series(np.all(fixed_knob_mask_list, axis=0), index=result_df.index)]

    # calculate winner's curse
    for estimator in estimators:
        result_df[f'{estimator}_wc'] = result_df[f"{estimator}_est"] - result_df['act_plugin_val']

    # clean unnecessary columns
    result_df = result_df[[f'{estimator}_wc' for estimator in estimators]]


    fig, ax = plt.subplots(1, 1, figsize=(10, 6.18))

    for i, col in enumerate(result_df.columns):
        ax.hist(result_df[col], bins=20, alpha=0.6, color=f'C{i}', label=col, density=True)
        ax.axvline(result_df[col].mean(), linestyle='--', linewidth=2.5, color=f'C{i}', label=f'{col} Mean')

    ax.set_xlabel('Winner\'s Curse')
    ax.set_ylabel('Frequency')
    ax.set_title('Winner\'s Curse of Different Estimators')
    ax.legend()

    plt.tight_layout()
    plt.grid()
    plt.show()


def plot_winners_curse_violin(result_dir: str, estimators: list = None, **kwargs):
    # read data
    result_df = pd.read_csv(result_dir)

    # get estimators 
    full_estimators = [col[:-4] for col in result_df.columns if '_est' in col]
    if estimators is None:
        estimators = full_estimators


    # fixed knobs keys 
    target_columns = [f'{estimator}_est' for estimator in full_estimators] + ['act_plugin_val', 'test_group_id', 'repeat_id']
    fixed_knob_keys = [col for col in result_df.columns if col not in target_columns]
    fixed_knob_vals = [result_df[key].iloc[0] if key not in kwargs.keys() else kwargs[key] for key in fixed_knob_keys]

    # print the fixed knobs
    print('Fixed Knobs: ', end=' ')
    for key, val in zip(fixed_knob_keys, fixed_knob_vals):
        print(f"{key}: {val},", end=' ')

    # extract the data for the fixed knobs
    fixed_knob_mask_list = np.array([
        (result_df[key] == val).values for key, val in zip(fixed_knob_keys, fixed_knob_vals)
    ])
    result_df = result_df[pd.Series(np.all(fixed_knob_mask_list, axis=0), index=result_df.index)]

    # calculate winner's curse
    for estimator in estimators:
        result_df[f'{estimator}_wc'] = result_df[f"{estimator}_est"] - result_df['act_plugin_val']

    # clean unnecessary columns
    result_df = result_df[[f'{estimator}_wc' for estimator in estimators]]


    fig, ax = plt.subplots(1, 1, figsize=(10, 6.18))

    for i, col in enumerate(result_df.columns):
        ax.violinplot(result_df[col], positions=[i], showmeans=True, showextrema=False)
    
    # rotate the x-axis labels 
    ax.set_xticks(range(len(result_df.columns)))
    ax.set_xticklabels(result_df.columns, rotation=45)
    ax.set_xlabel('Estimator')
    ax.set_ylabel('Winner\'s Curse')
    ax.set_title('Winner\'s Curse of Different Estimators')

    plt.tight_layout()
    plt.grid()
    plt.show()


def plot_winners_curse_2d(result_dir: str, x: str, confidence_level: float = 0.95, estimators: list = None, **kwargs):
    # read data
    result_df = pd.read_csv(result_dir)

    # get estimators 
    full_estimators = [col[:-4] for col in result_df.columns if '_est' in col]
    if estimators is None:
        estimators = full_estimators

    # check if x and y are columns in the dataframe
    assert x in result_df.columns, f'{x} is not a column in the dataframe'

    # fixed knobs keys 
    target_columns = [x] + [f'{estimator}_est' for estimator in full_estimators] + ['act_plugin_val', 'test_group_id', 'repeat_id']
    fixed_knob_keys = [col for col in result_df.columns if col not in target_columns]
    fixed_knob_vals = [result_df[key].iloc[0] if key not in kwargs.keys() else kwargs[key] for key in fixed_knob_keys]

    # print the fixed knobs
    print('Fixed Knobs: ', end=' ')
    for key, val in zip(fixed_knob_keys, fixed_knob_vals):
        print(f"{key}: {val},", end=' ')

    # extract the data for the fixed knobs
    fixed_knob_mask_list = np.array([
        (result_df[key] == val).values for key, val in zip(fixed_knob_keys, fixed_knob_vals)
    ])
    result_df = result_df[pd.Series(np.all(fixed_knob_mask_list, axis=0), index=result_df.index)]

    # calculate winner's curse
    for estimator in estimators:
        result_df[f'{estimator}_wc'] = result_df[f"{estimator}_est"] - result_df['act_plugin_val']

    # clean unnecessary columns
    wc_columns = [f'{estimator}_wc' for estimator in estimators]
    result_df = result_df[[x] + wc_columns + ['test_group_id', 'repeat_id']]

    # calcualte the average and standard deviation of the winner's curse across repeated trials
    mean_df = result_df.groupby([x])[wc_columns].mean().reset_index()
    se_df = result_df.groupby([x])[wc_columns].sem().reset_index()

    # calculate the confidence interval
    z = t.ppf(1 - (1 - confidence_level) / 2, result_df.groupby([x])[wc_columns].count().iloc[0, 0] - 1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6.18))

    for i, estimator_name in enumerate(wc_columns):
        ax.plot(mean_df[x], mean_df[estimator_name], 'o-', label=estimator_name, color=f'C{i}')
        ax.fill_between(
            mean_df[x], 
            mean_df[estimator_name] - z * se_df[estimator_name], 
            mean_df[estimator_name] + z * se_df[estimator_name], 
            alpha=0.2, color=f'C{i}'
        )

    ax.set_xlabel(x.replace('_', ' ').capitalize())
    ax.set_ylabel('Winner\'s Curse')
    ax.set_title(f'Winner\'s Curse of Different Estimators')
    ax.grid()
    ax.legend()

    plt.show()

def plot_winners_curse_2d_modified(result_dir: str, confidence_level: float = 0.95, estimators: list = None, **kwargs):
    x = 'sample_size'
    # read data
    result_df = pd.read_csv(result_dir)

    # get estimators 
    full_estimators = [col[:-4] for col in result_df.columns if '_est' in col]
    if estimators is None:
        estimators = full_estimators

    # check if x and y are columns in the dataframe
    assert x in result_df.columns, f'{x} is not a column in the dataframe'

    # fixed knobs keys 
    target_columns = [x] + [f'{estimator}_est' for estimator in full_estimators] + ['act_plugin_val', 'test_group_id', 'repeat_id']
    fixed_knob_keys = [col for col in result_df.columns if col not in target_columns]
    fixed_knob_vals = [result_df[key].iloc[0] if key not in kwargs.keys() else kwargs[key] for key in fixed_knob_keys]

    # print the fixed knobs
    print('Fixed Knobs: ', end=' ')
    for key, val in zip(fixed_knob_keys, fixed_knob_vals):
        print(f"{key}: {val},", end=' ')

    # extract the data for the fixed knobs
    fixed_knob_mask_list = np.array([
        (result_df[key] == val).values for key, val in zip(fixed_knob_keys, fixed_knob_vals)
    ])
    result_df = result_df[pd.Series(np.all(fixed_knob_mask_list, axis=0), index=result_df.index)]

    # calculate winner's curse
    for estimator in estimators:
        result_df[f'{estimator}_wc'] = result_df[f"{estimator}_est"] - result_df['act_plugin_val']

    # clean unnecessary columns
    wc_columns = [f'{estimator}_wc' for estimator in estimators]
    result_df = result_df[[x] + wc_columns + ['test_group_id', 'repeat_id']]

    # calcualte the average and standard deviation of the winner's curse across repeated trials
    mean_df = result_df.groupby([x])[wc_columns].mean().reset_index()
    se_df = result_df.groupby([x])[wc_columns].sem().reset_index()

    # calculate the confidence interval
    z = t.ppf(1 - (1 - confidence_level) / 2, result_df.groupby([x])[wc_columns].count().iloc[0, 0] - 1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6.18))

    for i, estimator_name in enumerate(wc_columns):
        ax.plot(1 / (mean_df[x] ** 0.5), mean_df[estimator_name], 'o-', label=estimator_name, color=f'C{i}')
        ax.fill_between(
            1 / (mean_df[x] ** 0.5), 
            mean_df[estimator_name] - z * se_df[estimator_name], 
            mean_df[estimator_name] + z * se_df[estimator_name], 
            alpha=0.2, color=f'C{i}'
        )

    ax.set_xlabel('1 / sqrt(N)')
    ax.set_ylabel('Winner\'s Curse')
    ax.set_title(f'Winner\'s Curse of Different Estimators')
    ax.grid()
    ax.legend()

    plt.show()

def plot_winners_curse_3d(result_dir: str, x: str, y: str, **kwargs): 
    # read data
    result_df = pd.read_csv(result_dir)

    # get estimators 
    estimators = [col[:-4] for col in result_df.columns if '_est' in col]

    # check if x and y are columns in the dataframe
    assert x in result_df.columns, f'{x} is not a column in the dataframe'
    assert y in result_df.columns, f'{y} is not a column in the dataframe'

    # fixed knobs keys 
    target_columns = [x, y] + [f'{estimator}_est' for estimator in estimators] + ['act_plugin_val']
    fixed_knob_keys = [col for col in result_df.columns if col not in target_columns]
    fixed_knob_vals = [result_df[key].iloc[0] if key not in kwargs.keys() else kwargs[key] for key in fixed_knob_keys]

    # print the fixed knobs
    print('Fixed Knobs: ', end=' ')
    for key, val in zip(fixed_knob_keys, fixed_knob_vals):
        print(f"{key}: {val},", end=' ')

    # extract the data for the fixed knobs
    fixed_knob_mask_list = np.array([
        (result_df[key] == val).values for key, val in zip(fixed_knob_keys, fixed_knob_vals)
    ])
    result_df = result_df[pd.Series(np.all(fixed_knob_mask_list, axis=0), index=result_df.index)]

    # calculate winner's curse
    for estimator in estimators:
        result_df[f'{estimator}_wc'] = result_df[f"{estimator}_est"] - result_df['act_plugin_val']

    # clean unnecessary columns
    result_df = result_df[[x, y] + [f'{estimator}_wc' for estimator in estimators]]

    # calcualte the average and standard deviation of the winner's curse across repeated trials
    avg_result_df = result_df.groupby([x, y]).mean().reset_index()

    # visualization params
    vmin = -0.05
    vmax = 1.1 * round(np.max(avg_result_df[[f'{estimator}_wc' for estimator in estimators]].max()), 4)

    # make a 3D plot, sample size vs num bootstrap vs winner's curse
    fig, axes = plt.subplots(1, len(estimators), figsize=(5 * len(estimators), 6.18), subplot_kw={'projection': '3d'})

    for estimator, ax in zip([f'{estimator}_wc' for estimator in estimators], axes):
        surf = ax.plot_trisurf(
            avg_result_df[x], avg_result_df[y], avg_result_df[estimator], 
            cmap=cm.jet, alpha=0.6, label=estimator.replace('_', ' ').capitalize(), vmin=vmin, vmax=vmax
        )
        ax.set_title(f"Winner\'s Curse of {estimator.replace('_', ' ').capitalize()}")
        ax.set_xlabel(x.replace('_', ' ').capitalize())
        ax.set_ylabel(y.replace('_', ' ').capitalize())
        ax.set_zlabel('Winner\'s Curse')
        ax.set_zlim(vmin, vmax)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(surf, cax=cbar_ax)

    plt.show()