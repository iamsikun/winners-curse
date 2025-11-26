"""
Analysis utilities for Winner's Curse experiments.

This module provides functions for computing summary statistics, generating LaTeX tables,
and creating visualizations for different experiment types:
- SNR (Signal-to-Noise Ratio) experiments
- N-treatments experiments
- Bernoulli A/B test experiments
- Targeting experiments

Each experiment type has three main functions:
1. compute_*_sum_stats: Calculate summary statistics
2. generate_*_sum_stats_latex_table: Generate LaTeX tables
3. plot_*_wc_comparison: Create visualizations
"""

from __future__ import annotations

import warnings 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

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


# ===========================
# Private Helper Functions
# ===========================

def _extract_estimator_info(config: Dict) -> Tuple[List[str], List[str]]:
    """
    Extract estimator names and keys from config.
    
    Args:
        config: Configuration dictionary with optimization_params and estimators_dict
        
    Returns:
        Tuple of (estimator_names, estimator_keys)
    """
    estimators_dict = config['estimators_dict']
    
    estimator_names = ['No Correction'] + [
        est_config['display_name'] for est_config in estimators_dict.values()
    ]
    estimator_keys = ['nc'] + list(estimators_dict.keys())
    
    return estimator_names, estimator_keys


def _get_wc_array(results: Dict, tau_tup: tuple, est_key: str) -> np.ndarray:
    """
    Extract winner's curse array from results dictionary.
    
    Args:
        results: Results dictionary keyed by tau tuples
        tau_tup: Tuple of tau values (treatment effect parameters)
        est_key: Key for the estimator
        
    Returns:
        Array of winner's curse values
    """
    return results[tau_tup][f'{est_key}_wc_arr']


def _compute_stat_value(wc_arr: np.ndarray, stat: str, norm_factor: float) -> float:
    """
    Compute statistic value with normalization.
    
    Args:
        wc_arr: Array of winner's curse values
        stat: Statistic name ('mean', 'median', 'std')
        norm_factor: Normalization factor to apply
        
    Returns:
        Computed statistic value after normalization
    """
    return stats_fn_map[stat](wc_arr) * norm_factor


def _build_latex_cell(val: float, std: float, stats: str, normalize: bool, decimals: int) -> str:
    """
    Generate LaTeX table cell with consistent formatting.
    
    Args:
        val: Main value to display
        std: Standard deviation value
        stats: Statistic type ('mean' or 'median')
        normalize: Whether values are normalized (adds % symbol)
        decimals: Number of decimal places
        
    Returns:
        LaTeX-formatted cell string
    """
    if stats == 'mean':
        if normalize:
            return f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{val:.{decimals}f}\\%\\\\ ({std:.{decimals}f}\\%)\\end{{tabular}}"
        else:
            return f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{val:.{decimals}f}\\\\ ({std:.{decimals}f})\\end{{tabular}}"
    else:
        if normalize:
            return f"{val:.{decimals}f}\\%"
        else:
            return f"{val:.{decimals}f}"


def _create_grouped_bar_plot(
    stats_df: pd.DataFrame,
    ax: plt.Axes,
    plot_config: PlotConfig,
    normalize: bool
) -> None:
    """
    Create grouped bar plot from statistics DataFrame.
    
    Args:
        stats_df: DataFrame with MultiIndex (Estimator, Statistic) and columns for different levels
        ax: Matplotlib axes to plot on
        plot_config: Plot configuration settings
        normalize: Whether to use percentage label on y-axis
    """
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
    
    # Set y-axis label
    if normalize:
        ax.set_ylabel('Winner\'s Curse (%)', fontsize=plot_config.axis_label_size)
    else:
        ax.set_ylabel('Winner\'s Curse', fontsize=plot_config.axis_label_size)
    
    # Styling
    ax.tick_params(axis='both', labelsize=plot_config.tick_label_size)
    ax.legend(fontsize=plot_config.legend_label_size)
    ax.grid(axis='y', alpha=0.3)


# ===========================
# Public Utility Functions
# ===========================



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
    estimator_names, estimator_keys = _extract_estimator_info(snr_config)

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
            wc_arr = _get_wc_array(snr_results, tau_tup, est_key)
            
            for stat in stats_list:
                val = _compute_stat_value(wc_arr, stat, norm_factor)
                data_records.append({
                    r'$\Delta\tau/\sigma$': col_name,
                    'Estimator': est_name,
                    'Statistic': stat,
                    'Value': val
                })

    # Create DataFrame from records
    df_long = pd.DataFrame(data_records)
    
    # Pivot to get the desired structure: Index=(Estimator, Statistic), Columns=SNR
    stats_df = df_long.pivot(
        index=['Estimator', 'Statistic'], 
        columns=r'$\Delta\tau/\sigma$', 
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
            col_data[est] = _build_latex_cell(val, std, stats, normalize, decimals)
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
    save_path: Optional[str] = None,
) -> Optional[plt.Axes]:
    """
    Plot Winner's Curse comparison across SNR levels.
    
    Args:
        snr_results: Dictionary of results keyed by tau tuple
        snr_config: Configuration dictionary
        normalize: Whether to normalize by delta_tau (percentage)
        plot_config: Configuration for plot styling
        ax: Optional existing axes to plot on
        save_path: Optional path to save the figure (e.g., 'figures/snr_comparison.png')
        
    Returns:
        Axes object if ax was provided, None otherwise
    """
    stats_df = compute_snr_sum_stats(snr_results, snr_config, ['mean'], normalize)

    # Track if ax was originally None
    ax_was_none = ax is None
    
    if ax_was_none:
        fig, ax = plt.subplots(1, 1, figsize=plot_config.figsize)

    # Create grouped bar plot using helper
    _create_grouped_bar_plot(stats_df, ax, plot_config, normalize)
    
    # Set x-axis label and ticks (custom for SNR)
    ax.set_xlabel(r'Signal-to-Noise Ratio $\Delta\tau / \sigma$', fontsize=plot_config.axis_label_size)
    
    # Extract SNR values for x-tick labels
    noise_std = get_noise_std(snr_config)
    snr_labels = ['{:.1%}'.format((t[1]-t[0])/noise_std) for t in snr_results.keys()]
    indices = np.arange(len(stats_df.columns))
    ax.set_xticks(indices)
    ax.set_xticklabels(snr_labels)

    if ax_was_none:
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight')
        plt.show()
        return None
    
    # If ax was provided, save the figure from the axes
    if save_path is not None:
        fig = ax.get_figure()
        fig.savefig(save_path, bbox_inches='tight')
    
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
    estimator_names, estimator_keys = _extract_estimator_info(config)

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
            wc_arr = _get_wc_array(results, tau_tup, est_key)
            
            for stat in stats_list:
                val = _compute_stat_value(wc_arr, stat, norm_factor)
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
            col_data[est] = _build_latex_cell(val, std, stats, normalize, decimals)
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
    save_path: Optional[str] = None,
) -> Optional[plt.Axes]:
    """
    Plot Winner's Curse comparison across Number of Treatments.
    
    Args:
        results: Dictionary of results keyed by tau tuple
        config: Configuration dictionary
        normalize: Whether to normalize by delta_tau (percentage)
        plot_config: Configuration for plot styling
        ax: Optional existing axes to plot on
        save_path: Optional path to save the figure (e.g., 'figures/n_treatments_comparison.png')
        
    Returns:
        Axes object if ax was provided, None otherwise
    """
    stats_df = compute_n_treatments_sum_stats(results, config, ['mean'], normalize)

    # Track if ax was originally None
    ax_was_none = ax is None
    
    if ax_was_none:
        fig, ax = plt.subplots(1, 1, figsize=plot_config.figsize)

    # Create grouped bar plot using helper
    _create_grouped_bar_plot(stats_df, ax, plot_config, normalize)
    
    # Set x-axis label and ticks (custom for N-treatments)
    ax.set_xlabel('Number of Treatments', fontsize=plot_config.axis_label_size)
    
    # Extract N Treatments values for x-tick labels
    tau_tuples = sorted(results.keys(), key=len)
    n_treat_labels = [f'{len(t)}' for t in tau_tuples]
    n_treat_labels = list(dict.fromkeys(n_treat_labels))
    
    indices = np.arange(len(n_treat_labels))
    ax.set_xticks(indices)
    ax.set_xticklabels(n_treat_labels)

    if ax_was_none:
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight')
        plt.show()
        return None
    
    # If ax was provided, save the figure from the axes
    if save_path is not None:
        fig = ax.get_figure()
        fig.savefig(save_path, bbox_inches='tight')
    
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
    estimator_names, estimator_keys = _extract_estimator_info(config)

    # Prepare data for DataFrame construction
    data_records = []
        
    for tau_tup in results.keys():
        delta_tau = tau_tup[1] - tau_tup[0]
        norm_factor = 100 / delta_tau if normalize else 1
        
        for est_key, est_name in zip(estimator_keys, estimator_names):
            wc_arr = _get_wc_array(results, tau_tup, est_key)
            
            for stat in stats_list:
                val = _compute_stat_value(wc_arr, stat, norm_factor)
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
    latex_data = []
    
    estimators = stats_df.index.get_level_values('Estimator').unique()
    
    for est in estimators:
        row_data = {}
        for col_tuple in stats_df.columns:
            val = stats_df.loc[(est, stats), col_tuple].values[0]
            std = stats_df.loc[(est, 'std'), col_tuple].values[0]
            
            # Keep the MultiIndex column tuple
            row_data[col_tuple] = _build_latex_cell(val, std, stats, normalize, decimals)
        
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
    save_path: Optional[str] = None,
) -> Optional[plt.Axes]:
    """
    Plot Winner's Curse comparison across Bernoulli effect sizes.
    
    Args:
        results: Dictionary of results keyed by tau tuple
        config: Configuration dictionary
        normalize: Whether to normalize by delta_tau (percentage)
        plot_config: Configuration for plot styling
        ax: Optional existing axes to plot on
        save_path: Optional path to save the figure (e.g., 'figures/bernoulli_comparison.png')
        
    Returns:
        Axes object if ax was provided, None otherwise
    """
    stats_df = compute_bernoulli_sum_stats(results, config, ['mean'], normalize)

    # Track if ax was originally None
    ax_was_none = ax is None
    
    if ax_was_none:
        fig, ax = plt.subplots(1, 1, figsize=plot_config.figsize)

    # Create grouped bar plot using helper
    _create_grouped_bar_plot(stats_df, ax, plot_config, normalize)
    
    # Set x-axis label and ticks (custom for Bernoulli)
    ax.set_xlabel(r'$(\tau_1, \tau_2)$', fontsize=plot_config.axis_label_size)
    
    # Extract effect size labels
    tau_tuples = sorted(results.keys())
    indices = np.arange(len(stats_df.columns))
    ax.set_xticks(indices)
    ax.set_xticklabels(tau_tuples, rotation=45, ha='right')

    plt.tight_layout()
    
    if ax_was_none:
        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight')
        plt.show()
        return None
    
    # If ax was provided, save the figure from the axes
    if save_path is not None:
        fig = ax.get_figure()
        fig.savefig(save_path, bbox_inches='tight')
    
    return ax


def plot_targeting_wc_depth_sample_size(
    results: Dict,
    config: Dict,
    normalize: bool = True,
    viz_sample_sizes: list | None = None,
    plot_config: PlotConfig = PlotConfig(),
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> Optional[plt.Axes]:
    """
    Plot Winner's Curse vs Sample Size for different max_depth settings.
    
    Args:
        results: Dictionary of results keyed by (depth, sample_size) tuples
        config: Configuration dictionary
        normalize: Whether to normalize by delta_tau (percentage)
        viz_sample_sizes: List of sample sizes to plot
        plot_config: Configuration for plot styling
        ax: Optional existing axes to plot on
        save_path: Optional path to save the figure (e.g., 'figures/targeting_depth_sample_size.png')
        
    Returns:
        Axes object if ax was provided, None otherwise
    """
    # Extract depth_list and sample_size_list from config
    depth_list = config['comparative_statics']['depth_list']
    sample_size_list = config['comparative_statics']['sample_size_list']
    
    # Calculate normalization factor
    delta_tau = config['dgp_params']['base_effect_vars'][1]['value'] - config['dgp_params']['base_effect_vars'][0]['value']
    norm_factor = 100 / delta_tau if normalize else 1
    
    # Determine which estimator to use
    # optimizer_key = list(config['optimization_params'].keys())[0]
    
    # Track if ax was originally None
    ax_was_none = ax is None
    
    if ax_was_none:
        fig, ax = plt.subplots(1, 1, figsize=plot_config.figsize)
    
    # Plot markers for different depths
    markers = ['o', 's', 'D', '^', 'x', 'v', '<', '>', 'p']
    
    for i, depth in enumerate(depth_list):
        wc_avg_list = [
            norm_factor * np.mean(results[(depth, sample_size)][f'nc_wc_arr'])
            for sample_size in sample_size_list
        ]
        
        marker = markers[i % len(markers)]
        ax.plot(
            sample_size_list, wc_avg_list,
            label=f'Max Depth = {depth}',
            marker=marker,
            markersize=8
        )
    
    # Axis labels
    ax.set_xlabel('Sample Size Per Treatment', fontsize=plot_config.axis_label_size)
    
    if normalize:
        ax.set_ylabel("Winner's Curse (%)", fontsize=plot_config.axis_label_size)
    else:
        ax.set_ylabel("Winner's Curse", fontsize=plot_config.axis_label_size)
    
    # Automatically select which labels to show
    if viz_sample_sizes is None:
        xtick_indices = sample_size_list
    else: 
        xtick_indices = [s for s in viz_sample_sizes if s in sample_size_list]
    xtick_labels = [f'{s:,}' for s in xtick_indices]
    
    # Set x-ticks with selected labels
    ax.set_xticks(xtick_indices, labels=xtick_labels, rotation=30)
    
    # Styling
    ax.tick_params(axis='both', labelsize=plot_config.tick_label_size)
    ax.legend(fontsize=plot_config.legend_label_size)
    ax.grid()
    
    if ax_was_none:
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight')
        plt.show()
        return None
    
    # If ax was provided, save the figure from the axes
    if save_path is not None:
        fig = ax.get_figure()
        fig.savefig(save_path, bbox_inches='tight')
    
    return ax


def compute_noise_dist_sum_stats(
    results: Dict,
    config: Dict, 
    stats_list: List[str], 
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Calculate summary statistics for the Noise Distribution experiment.
    
    Args:
        results: Dictionary of results keyed by noise_vars tuple (strings)
        config: Configuration dictionary
        stats_list: List of statistics to compute (e.g., ['mean', 'std'])
        normalize: Whether to normalize by delta_tau (percentage)
        
    Returns:
        MultiIndex DataFrame with statistics
    """
    optimizer_key, estimator_names, estimator_keys = _extract_estimator_info(config)

    # Prepare data for DataFrame construction
    data_records = []
    
    # Calculate delta_tau for normalization
    # Assuming delta_tau is constant across noise distributions (which it should be for this experiment)
    # We take the first result to calculate it, or use dgp_params
    dgp_params = config['dgp_params']
    if 'base_effect_vars' in dgp_params:
        base_effects = dgp_params['base_effect_vars']
        delta_tau = base_effects[1]['value'] - base_effects[0]['value']
    elif 'base_effects' in dgp_params:
        base_effects = dgp_params['base_effects']
        delta_tau = base_effects[1]['value'] - base_effects[0]['value']
    else:
        # Fallback if we can't find it easily (e.g. if it's not PointMass)
        delta_tau = 1.0
        
    norm_factor = 100 / delta_tau if normalize else 1
    
    # Sort keys to ensure consistent order
    # Keys are tuples of strings representing noise vars
    noise_keys = sorted(results.keys())
    
    for noise_key in noise_keys:
        # Generate a readable label for the noise distribution
        # noise_key is a tuple of strings, e.g. ("{'type': 'Gaussian', ...}", ...)
        # We take the first one (assuming symmetric noise across treatments for labeling)
        try:
            # Parse the string representation of the dictionary
            # We use a safe eval or just string manipulation
            # Since we trust our own output, eval is okay-ish, but let's try to be safer if possible
            # But these are just strings produced by our own code.
            first_noise_str = noise_key[0]
            # Use the evaluate_param context from experiments.py if we could import it, 
            # but here we'll just use a simple eval since we know the format
            noise_dict = eval(first_noise_str)
            dist_type = noise_dict.get('type', 'Unknown')
            
            # Add parameters to label
            params = []
            for k, v in noise_dict.items():
                if k != 'type':
                    # Format float values nicely
                    if isinstance(v, (float, int)):
                        val_str = f"{v:.2f}".rstrip('0').rstrip('.')
                    else:
                        val_str = str(v)
                    params.append(f"{k}={val_str}")
            
            if params:
                col_name = f"{dist_type}\n({', '.join(params)})"
            else:
                col_name = dist_type
                
        except Exception:
            col_name = str(noise_key)

        for est_key, est_name in zip(estimator_keys, estimator_names):
            wc_arr = _get_wc_array(results, noise_key, optimizer_key, est_key)
            
            for stat in stats_list:
                val = _compute_stat_value(wc_arr, stat, norm_factor)
                data_records.append({
                    'Noise Distribution': col_name,
                    'Estimator': est_name,
                    'Statistic': stat,
                    'Value': val
                })

    # Create DataFrame from records
    df_long = pd.DataFrame(data_records)
    
    # Pivot to get the desired structure: Index=(Estimator, Statistic), Columns=Noise Distribution
    stats_df = df_long.pivot(
        index=['Estimator', 'Statistic'], 
        columns='Noise Distribution', 
        values='Value'
    )
    
    # Reorder index to match input estimator order
    stats_df = stats_df.reindex(estimator_names, level='Estimator')
    stats_df = stats_df.reindex(stats_list, level='Statistic')
    
    return stats_df


def generate_noise_dist_sum_stats_latex_table(
    results: Dict,
    config: Dict,
    normalize: bool = True,
    stats: str = 'mean',
    decimals: int = 2,
) -> str:
    """
    Generate a LaTeX table for Noise Distribution summary statistics.
    """
    assert stats in ['mean', 'median'], 'stats must be either "mean" or "median"'

    stats_df = compute_noise_dist_sum_stats(results, config, [stats, 'std'], normalize)
    
    # Create a new DataFrame for the LaTeX table
    latex_data = {}
    
    estimators = stats_df.index.get_level_values('Estimator').unique()
    columns = stats_df.columns
    
    for col in columns:
        col_data = {}
        for est in estimators:
            val = stats_df.loc[(est, stats), col]
            std = stats_df.loc[(est, 'std'), col]
            col_data[est] = _build_latex_cell(val, std, stats, normalize, decimals)
        latex_data[col] = col_data
        
    latex_df = pd.DataFrame(latex_data)
    
    return latex_df.style.to_latex(
        column_format='c' * (len(columns) + 1), 
        hrules=True
    )


def plot_noise_dist_wc_comparison(
    results: Dict,
    config: Dict,
    normalize: bool = True,
    plot_config: PlotConfig = PlotConfig(),
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> Optional[plt.Axes]:
    """
    Plot Winner's Curse comparison across Noise Distributions.
    
    Args:
        results: Dictionary of results keyed by noise_vars tuple (strings)
        config: Configuration dictionary
        normalize: Whether to normalize by delta_tau (percentage)
        plot_config: Configuration for plot styling
        ax: Optional existing axes to plot on
        save_path: Optional path to save the figure (e.g., 'figures/noise_dist_comparison.png')
        
    Returns:
        Axes object if ax was provided, None otherwise
    """
    stats_df = compute_noise_dist_sum_stats(results, config, ['mean'], normalize)

    # Track if ax was originally None
    ax_was_none = ax is None
    
    if ax_was_none:
        fig, ax = plt.subplots(1, 1, figsize=plot_config.figsize)

    # Create grouped bar plot using helper
    _create_grouped_bar_plot(stats_df, ax, plot_config, normalize)
    
    # Set x-axis label
    ax.set_xlabel('Noise Distribution', fontsize=plot_config.axis_label_size)
    
    # Set x-ticks
    indices = np.arange(len(stats_df.columns))
    ax.set_xticks(indices)
    ax.set_xticklabels(stats_df.columns, rotation=45, ha='right')

    if ax_was_none:
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight')
        plt.show()
        return None
    
    # If ax was provided, save the figure from the axes
    if save_path is not None:
        fig = ax.get_figure()
        fig.savefig(save_path, bbox_inches='tight')
    
    return ax
