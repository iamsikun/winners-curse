from __future__ import annotations

import warnings 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

stats_fn_map: Dict[str, callable] = {
    'mean': np.mean,
    'median': np.median,
    'std': np.std,
}

@dataclass
class PlotConfig:
    tick_label_size: int = 12
    legend_label_size: int = 12
    axis_label_size: int = 14
    title_size: int = 18
    figsize: tuple[float, float] = (10, 6.18)


def get_noise_std(config: Dict) -> float:
    """
    Robustly extract noise standard deviation from config.
    
    Handles both formats:
    - noise_vars: list of noise variable dicts (new format)
    - noise_var: single noise variable dict (old format)
    
    Args:
        config: Configuration dictionary with dgp_params
        
    Returns:
        Standard deviation of the first noise variable
    """
    dgp_params = config['dgp_params']
    
    # Try noise_vars (list format) first
    if 'noise_vars' in dgp_params:
        noise_vars = dgp_params['noise_vars']
        if isinstance(noise_vars, list) and len(noise_vars) > 0:
            first_noise = noise_vars[0]
            return first_noise.get('std', 1.0) if isinstance(first_noise, dict) else 1.0
    
    # Fall back to noise_var (singular) for backward compatibility
    if 'noise_var' in dgp_params:
        noise_var = dgp_params['noise_var']
        return noise_var.get('std', 1.0) if isinstance(noise_var, dict) else 1.0
    
    # Default if neither format found
    warnings.warn("Noise standard deviation not found in config. Defaulting to 1.0.")
    return 1.0


def compute_snr_sum_stats(
    snr_results: Dict,
    snr_config: Dict, 
    stats_list: List[str], 
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Calculate summary statistics for the SNR experiment.
    
    Args:
        snr_results: Dictionary of results keyed by tau tuple
        snr_config: Configuration dictionary
        stats_list: List of statistics to compute (e.g., ['mean', 'std'])
        normalize: Whether to normalize by delta_tau (percentage)
        
    Returns:
        MultiIndex DataFrame with statistics
    """
    optimizer_key = list(snr_config['optimization_params'].keys())[0]
    estimators_dict = snr_config['estimators_dict']

    estimator_names = ['No Correction'] + [
        est_config['display_name'] for est_config in estimators_dict.values()
    ]
    estimator_keys = ['nc'] + list(estimators_dict.keys())

    # Prepare data for DataFrame construction
    data_records = []
    
    # Sort keys to ensure consistent order
    tau_tuples = sorted(snr_results.keys())
    
    # Get noise standard deviation robustly
    noise_std = get_noise_std(snr_config)
    
    for tau_tup in tau_tuples:
        delta_tau = tau_tup[1] - tau_tup[0]
        norm_factor = 100 / delta_tau if normalize else 1
        
        snr = delta_tau / noise_std
        col_name = '{:.1f}%'.format(snr * 100)

        for est_key, est_name in zip(estimator_keys, estimator_names):
            # Extract the winner's curse array
            wc_arr = snr_results[tau_tup][f'{optimizer_key}_{est_key}_wc_arr']
            
            for stat in stats_list:
                val = stats_fn_map[stat](wc_arr) * norm_factor
                data_records.append({
                    r'$\Delta\tau$': col_name,
                    'Estimator': est_name,
                    'Statistic': stat,
                    'Value': val
                })

    # Create DataFrame from records
    df_long = pd.DataFrame(data_records)
    
    # Pivot to get the desired structure: Index=(Estimator, Statistic), Columns=SNR
    stats_df = df_long.pivot(
        index=['Estimator', 'Statistic'], 
        columns=r'$\Delta\tau$', 
        values='Value'
    )
    
    # Reorder index to match input estimator order
    stats_df = stats_df.reindex(estimator_names, level='Estimator')
    stats_df = stats_df.reindex(stats_list, level='Statistic')
    
    # Reorder columns to match SNR order
    snr_cols = ['{:.1f}%'.format(((t[1]-t[0])/noise_std) * 100) for t in tau_tuples]
    stats_df = stats_df[snr_cols]
    
    return stats_df


def generate_snr_sum_stats_latex_table(
    snr_results: Dict,
    snr_config: Dict,
    normalize: bool = True,
    stats: str = 'mean',
    decimals: int = 2,
) -> str:
    """
    Generate a LaTeX table for SNR summary statistics.
    """
    assert stats in ['mean', 'median'], 'stats must be either "mean" or "median"'

    stats_df = compute_snr_sum_stats(snr_results, snr_config, [stats, 'std'], normalize)
    
    # Create a new DataFrame for the LaTeX table
    latex_data = {}
    
    estimators = stats_df.index.get_level_values('Estimator').unique()
    columns = stats_df.columns
    
    for col in columns:
        col_data = {}
        for est in estimators:
            val = stats_df.loc[(est, stats), col]
            std = stats_df.loc[(est, 'std'), col]
            
            if stats == 'mean':
                cell = f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{val:.{decimals}f}\\%\\\\ ({std:.{decimals}f}\\%)\\end{{tabular}}"
            else:
                cell = f"{val:.{decimals}f}\\%"
            col_data[est] = cell
        latex_data[col] = col_data
        
    latex_df = pd.DataFrame(latex_data)
    latex_df.columns.name = r'$\Delta\tau/\sigma$'
    
    return latex_df.style.to_latex(
        column_format='c' * (len(columns) + 1), 
        hrules=True
    )


def plot_snr_wc_comparison(
    snr_results: Dict,
    snr_config: Dict,
    normalize: bool = True,
    plot_config: PlotConfig = PlotConfig(),
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Axes]:
    """
    Plot Winner's Curse comparison across SNR levels.
    """
    stats_df = compute_snr_sum_stats(snr_results, snr_config, ['mean'], normalize)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=plot_config.figsize)

    estimators = stats_df.index.get_level_values('Estimator').unique()
    n_estimators = len(estimators)
    n_snr_levels = len(stats_df.columns)
    
    bar_width = 0.8 / n_estimators
    indices = np.arange(n_snr_levels)

    for i, est in enumerate(estimators):
        # Calculate offset for grouped bars
        offset = (i - n_estimators / 2 + 0.5) * bar_width
        
        values = stats_df.loc[(est, 'mean')]
        
        ax.bar(
            indices + offset,
            values,
            width=bar_width,
            label=est,
            alpha=0.7,
            capsize=5
        )

    ax.set_xlabel(r'Signal-to-Noise Ratio $\Delta\tau / \sigma$', fontsize=plot_config.axis_label_size)

    if normalize: 
        ax.set_ylabel('Winner\'s Curse (%)', fontsize=plot_config.axis_label_size)
    else: 
        ax.set_ylabel('Winner\'s Curse', fontsize=plot_config.axis_label_size)

    # Extract SNR values for x-tick labels using robust helper
    noise_std = get_noise_std(snr_config)
    snr_labels = ['{:.1%}'.format((t[1]-t[0])/noise_std) for t in snr_results.keys()]
    ax.set_xticks(indices)
    ax.set_xticklabels(snr_labels)
    
    ax.tick_params(axis='both', labelsize=plot_config.tick_label_size)
    ax.legend(fontsize=plot_config.legend_label_size)
    ax.grid(axis='y', alpha=0.3)

    if ax is None:
        plt.tight_layout()
        plt.show()
        return None
    
    return ax

def compute_n_treatments_sum_stats(
    results: Dict,
    config: Dict, 
    stats_list: List[str], 
    normalize: bool = False,
) -> pd.DataFrame:
    """
    Calculate summary statistics for the N-treatments experiment.
    
    Args:
        results: Dictionary of results keyed by tau tuple
        config: Configuration dictionary
        stats_list: List of statistics to compute (e.g., ['mean', 'std'])
        normalize: Whether to normalize by delta_tau (percentage)
        
    Returns:
        MultiIndex DataFrame with statistics
    """
    optimizer_key = list(config['optimization_params'].keys())[0]
    estimators_dict = config['estimators_dict']

    estimator_names = ['No Correction'] + [
        est_config['display_name'] for est_config in estimators_dict.values()
    ]
    estimator_keys = ['nc'] + list(estimators_dict.keys())

    # Prepare data for DataFrame construction
    data_records = []
    
    # Sort keys to ensure consistent order
    tau_tuples = sorted(results.keys(), key=len)
    
    for tau_tup in tau_tuples:
        n_treatments = len(tau_tup)
        sorted_tau = sorted(tau_tup)
        if sorted_tau[-1] == sorted_tau[0]:
            # All treatments have the same effect
            normalize = False
            delta_tau = 1.0
        else:
            # Use average delta_tau between treatments: (max - min) / (n_treatments - 1)
            delta_tau = (sorted_tau[-1] - sorted_tau[0]) / (n_treatments - 1)
        
        norm_factor = 100 / delta_tau if normalize else 1
        
        col_name = f'{n_treatments}'

        for est_key, est_name in zip(estimator_keys, estimator_names):
            # Extract the winner's curse array
            wc_arr = results[tau_tup][f'{optimizer_key}_{est_key}_wc_arr']
            
            for stat in stats_list:
                val = stats_fn_map[stat](wc_arr) * norm_factor
                data_records.append({
                    'N Treatments': col_name,
                    'Estimator': est_name,
                    'Statistic': stat,
                    'Value': val
                })

    # Create DataFrame from records
    df_long = pd.DataFrame(data_records)
    
    # Pivot to get the desired structure: Index=(Estimator, Statistic), Columns=N Treatments
    stats_df = df_long.pivot(
        index=['Estimator', 'Statistic'], 
        columns='N Treatments', 
        values='Value'
    )
    
    # Reorder index to match input estimator order
    stats_df = stats_df.reindex(estimator_names, level='Estimator')
    stats_df = stats_df.reindex(stats_list, level='Statistic')
    
    # Reorder columns to match N Treatments order
    n_treat_cols = [f'{len(t)}' for t in tau_tuples]
    # Remove duplicates while preserving order
    n_treat_cols = list(dict.fromkeys(n_treat_cols))
    stats_df = stats_df[n_treat_cols]
    
    return stats_df


def generate_n_treatments_sum_stats_latex_table(
    results: Dict,
    config: Dict,
    normalize: bool = False,
    stats: str = 'mean',
    decimals: int = 2,
) -> str:
    """
    Generate a LaTeX table for N-treatments summary statistics.
    """
    assert stats in ['mean', 'median'], 'stats must be either "mean" or "median"'

    stats_df = compute_n_treatments_sum_stats(results, config, [stats, 'std'], normalize)
    
    # Create a new DataFrame for the LaTeX table
    latex_data = {}
    
    estimators = stats_df.index.get_level_values('Estimator').unique()
    columns = stats_df.columns
    
    for col in columns:
        col_data = {}
        for est in estimators:
            val = stats_df.loc[(est, stats), col]
            std = stats_df.loc[(est, 'std'), col]
            
            if stats == 'mean':
                if normalize: 
                    cell = f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{val:.{decimals}f}\\%\\\\ ({std:.{decimals}f}\\%)\\end{{tabular}}"
                else: 
                    cell = f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{val:.{decimals}f}\\\\ ({std:.{decimals}f})\\end{{tabular}}"
            else:
                if normalize: 
                    cell = f"{val:.{decimals}f}\\%"
                else: 
                    cell = f"{val:.{decimals}f}"
            col_data[est] = cell
        latex_data[col] = col_data
        
    latex_df = pd.DataFrame(latex_data)
    
    return latex_df.style.to_latex(
        column_format='c' * (len(columns) + 1), 
        hrules=True
    )


def plot_n_treatments_wc_comparison(
    results: Dict,
    config: Dict,
    normalize: bool = False,
    plot_config: PlotConfig = PlotConfig(),
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Axes]:
    """
    Plot Winner's Curse comparison across Number of Treatments.
    """
    stats_df = compute_n_treatments_sum_stats(results, config, ['mean'], normalize)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=plot_config.figsize)

    estimators = stats_df.index.get_level_values('Estimator').unique()
    n_estimators = len(estimators)
    n_levels = len(stats_df.columns)
    
    bar_width = 0.8 / n_estimators
    indices = np.arange(n_levels)

    for i, est in enumerate(estimators):
        # Calculate offset for grouped bars
        offset = (i - n_estimators / 2 + 0.5) * bar_width
        
        values = stats_df.loc[(est, 'mean')]
        
        ax.bar(
            indices + offset,
            values,
            width=bar_width,
            label=est,
            alpha=0.7,
            capsize=5
        )

    ax.set_xlabel('Number of Treatments', fontsize=plot_config.axis_label_size)

    if normalize: 
        ax.set_ylabel('Winner\'s Curse (%)', fontsize=plot_config.axis_label_size)
    else: 
        ax.set_ylabel('Winner\'s Curse', fontsize=plot_config.axis_label_size)

    # Extract N Treatments values for x-tick labels
    # We need to ensure we use the same unique sorted labels as in compute_n_treatments_sum_stats
    tau_tuples = sorted(results.keys(), key=len)
    n_treat_labels = [f'{len(t)}' for t in tau_tuples]
    n_treat_labels = list(dict.fromkeys(n_treat_labels))
    
    ax.set_xticks(indices)
    ax.set_xticklabels(n_treat_labels)
    
    ax.tick_params(axis='both', labelsize=plot_config.tick_label_size)
    ax.legend(fontsize=plot_config.legend_label_size)
    ax.grid(axis='y', alpha=0.3)

    if ax is None:
        plt.tight_layout()
        plt.show()
        return None
    

    return ax


def compute_bernoulli_sum_stats(
    results: Dict,
    config: Dict, 
    stats_list: List[str], 
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Calculate summary statistics for the Bernoulli experiment.
    
    Args:
        results: Dictionary of results keyed by tau tuple
        config: Configuration dictionary
        stats_list: List of statistics to compute (e.g., ['mean', 'std'])
        normalize: Whether to normalize by delta_tau (percentage)
        
    Returns:
        MultiIndex DataFrame with hierarchical columns (Delta Tau, Effect Size) and index (Estimator, Statistic)
    """
    optimizer_key = list(config['optimization_params'].keys())[0]
    estimators_dict = config['estimators_dict']

    estimator_names = ['No Correction'] + [
        est_config['display_name'] for est_config in estimators_dict.values()
    ]
    estimator_keys = ['nc'] + list(estimators_dict.keys())

    # Prepare data for DataFrame construction
    data_records = []
        
    for tau_tup in results.keys():
        delta_tau = tau_tup[1] - tau_tup[0]
        norm_factor = 100 / delta_tau if normalize else 1
        
        for est_key, est_name in zip(estimator_keys, estimator_names):
            # Extract the winner's curse array
            wc_arr = results[tau_tup][f'{optimizer_key}_{est_key}_wc_arr']
            
            for stat in stats_list:
                val = stats_fn_map[stat](wc_arr) * norm_factor
                data_records.append({
                    r'$\Delta\tau$': delta_tau,
                    r'$(\tau_0, \tau_1)$': tau_tup,
                    'Estimator': est_name,
                    'Statistic': stat,
                    'Value': val
                })

    # Create DataFrame from records
    df_long = pd.DataFrame(data_records)
    
    # Pivot to get the desired structure: Index=(Estimator, Statistic), Columns=(Delta Tau, Effect Size)
    stats_df = df_long.pivot(
        index=['Estimator', 'Statistic'], 
        columns=[r'$\Delta\tau$', r'$(\tau_0, \tau_1)$'], 
        values='Value'
    )
    
    # Reorder index to match input estimator order
    stats_df = stats_df.reindex(estimator_names, level='Estimator')
    stats_df = stats_df.reindex(stats_list, level='Statistic')
    
    # Reorder columns to match tau tuple order (delta_tau, effect_size pairs)
    column_order = [(t[1] - t[0], t) for t in results.keys()]
    stats_df = stats_df[column_order]
    
    return stats_df


def generate_bernoulli_sum_stats_latex_table(
    results: Dict,
    config: Dict,
    normalize: bool = True,
    stats: str = 'mean',
    decimals: int = 2,
) -> str:
    """
    Generate a LaTeX table for Bernoulli summary statistics with hierarchical columns.
    """
    assert stats in ['mean', 'median'], 'stats must be either "mean" or "median"'

    stats_df = compute_bernoulli_sum_stats(results, config, [stats, 'std'], normalize)
    
    # Create a new DataFrame preserving the MultiIndex column structure
    # Build data for each estimator combining mean and std
    latex_data = []
    
    estimators = stats_df.index.get_level_values('Estimator').unique()
    
    for est in estimators:
        row_data = {}
        for col_tuple in stats_df.columns:
            val = stats_df.loc[(est, stats), col_tuple].values[0]
            std = stats_df.loc[(est, 'std'), col_tuple].values[0]
            
            if stats == 'mean':
                if normalize:
                    cell = f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{val:.{decimals}f}\\%\\\\ ({std:.{decimals}f}\\%)\\end{{tabular}}"
                else:
                    cell = f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{val:.{decimals}f}\\\\ ({std:.{decimals}f})\\end{{tabular}}"
            else:
                if normalize:
                    cell = f"{val:.{decimals}f}\\%"
                else:
                    cell = f"{val:.{decimals}f}"
            
            # Keep the MultiIndex column tuple
            row_data[col_tuple] = cell
        
        latex_data.append(row_data)
    
    # Create DataFrame with MultiIndex columns preserved
    latex_df = pd.DataFrame(latex_data, index=estimators)
    latex_df.columns = stats_df.columns  # Preserve the MultiIndex
    
    return latex_df.to_latex(
        escape=False,
        column_format='l' + 'c' * len(latex_df.columns),
        multicolumn=True,
        multicolumn_format='c',
        index_names=False
    )


def plot_bernoulli_wc_comparison(
    results: Dict,
    config: Dict,
    normalize: bool = True,
    plot_config: PlotConfig = PlotConfig(),
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Axes]:
    """
    Plot Winner's Curse comparison across Bernoulli effect sizes.
    """
    stats_df = compute_bernoulli_sum_stats(results, config, ['mean'], normalize)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=plot_config.figsize)

    estimators = stats_df.index.get_level_values('Estimator').unique()
    n_estimators = len(estimators)
    n_levels = len(stats_df.columns)
    
    bar_width = 0.8 / n_estimators
    indices = np.arange(n_levels)

    for i, est in enumerate(estimators):
        # Calculate offset for grouped bars
        offset = (i - n_estimators / 2 + 0.5) * bar_width
        
        values = stats_df.loc[(est, 'mean')]
        
        ax.bar(
            indices + offset,
            values,
            width=bar_width,
            label=est,
            alpha=0.7,
            capsize=5
        )

    ax.set_xlabel(r'$(\tau_1, \tau_2)$', fontsize=plot_config.axis_label_size)

    if normalize: 
        ax.set_ylabel('Winner\'s Curse (%)', fontsize=plot_config.axis_label_size)
    else: 
        ax.set_ylabel('Winner\'s Curse', fontsize=plot_config.axis_label_size)

    # Extract effect size labels
    tau_tuples = sorted(results.keys())
    
    ax.set_xticks(indices)
    ax.set_xticklabels(tau_tuples, rotation=45, ha='right')
    
    ax.tick_params(axis='both', labelsize=plot_config.tick_label_size)
    ax.legend(fontsize=plot_config.legend_label_size)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()
