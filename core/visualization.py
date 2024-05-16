import os 
import sys 
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

def plot_winners_curse_hist(result_dir: str, **kwargs):
    # read data
    result_df = pd.read_csv(result_dir)

    # get estimators 
    estimators = [col[:-4] for col in result_df.columns if '_est' in col]

    # fixed knobs keys 
    target_columns = [f'{estimator}_est' for estimator in estimators] + ['act_plugin_val']
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


    fig, axes = plt.subplots(1, 2, figsize=(20, 6.18))

    for i, col in enumerate(result_df.columns):
        axes[0].hist(result_df[col], bins=20, alpha=0.6, color=f'C{i}', label=col, density=True)
        axes[0].axvline(result_df[col].mean(), linestyle='--', linewidth=2.5, color=f'C{i}', label=f'{col} Mean')

    axes[0].set_xlabel('Winner\'s Curse')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Winner\'s Curse of Different Estimators')
    axes[0].legend()

    for i, col in enumerate(result_df.columns):
        axes[1].violinplot(result_df[col], positions=[i], showmeans=True, showextrema=False)
        
    axes[1].set_xticks(range(len(result_df.columns)))
    axes[1].set_xticklabels(result_df.columns)
    axes[1].set_xlabel('Estimator')
    axes[1].set_ylabel('Winner\'s Curse')
    axes[1].set_title('Winner\'s Curse of Different Estimators')

    for ax in axes:
        ax.grid()

    plt.show()

def plot_winners_curse_2d(result_dir: str, x: str, **kwargs):
    # read data
    result_df = pd.read_csv(result_dir)

    # get estimators 
    estimators = [col[:-4] for col in result_df.columns if '_est' in col]

    # check if x and y are columns in the dataframe
    assert x in result_df.columns, f'{x} is not a column in the dataframe'

    # fixed knobs keys 
    target_columns = [x] + [f'{estimator}_est' for estimator in estimators] + ['act_plugin_val']
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
    result_df = result_df[[x] + [f'{estimator}_wc' for estimator in estimators]]

    # calcualte the average and standard deviation of the winner's curse across repeated trials
    avg_result_df = result_df.groupby([x]).mean().reset_index()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6.18))

    for i, estimator_name in enumerate([f'{estimator}_wc' for estimator in estimators]):
        ax.plot(avg_result_df[x], avg_result_df[estimator_name], 'o-', label=estimator_name, color=f'C{i}')

    ax.set_xlabel(x.replace('_', ' ').capitalize())
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