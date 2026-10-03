import json
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from winners_curse.analysis import (
    compute_targeting_depth_stats,
    compute_targeting_policy_comparison,
    plot_bernoulli_wc_comparison,
    plot_snr_wc_comparison,
    plot_targeting_wc_depth_sample_size,
)
from winners_curse.result_io import find_latest_result_dir, load_saved_results


def write_run(root, name, filename='results_slim.pkl', values=None):
    directory = root / name
    directory.mkdir()
    (directory / 'config.json').write_text(json.dumps({'estimators_dict': {}}))
    if filename:
        (directory / filename).write_bytes(pickle.dumps(values or {'source': filename}))
    return directory


def test_discovery_skips_missing_artifacts_and_estimator_subsets(tmp_path):
    write_run(tmp_path, 'targeting_forest_depth_20260920_062447')
    newest = write_run(tmp_path, 'targeting_forest_depth_20260921_002545')
    write_run(tmp_path, 'targeting_forest_depth_20260922_000000', filename=None)
    write_run(tmp_path, 'targeting_forest_depth_estsubset_20260923_000000')
    write_run(tmp_path, 'targeting_forest_depth_n2_20260924_000000')
    assert find_latest_result_dir(tmp_path, 'targeting_forest_depth') == newest
    with pytest.raises(FileNotFoundError, match='No saved run'):
        find_latest_result_dir(tmp_path, 'targeting_correct_snr')


def test_discovery_compares_alias_timestamps_not_prefixes(tmp_path):
    write_run(tmp_path, 'targeting_forest_bernoulli_20251212_105431')
    newest = write_run(tmp_path, 'targeting_cf_bernoulli_20260919_214621')
    assert find_latest_result_dir(
        tmp_path, ('targeting_cf_bernoulli', 'targeting_forest_bernoulli')
    ) == newest


@pytest.mark.parametrize('preferred', [
    'results_slim.pkl', 'results_with_new_moon.pkl', 'results.pkl',
])
def test_loader_precedence_and_legacy_fallback(tmp_path, preferred):
    directory = write_run(tmp_path, 'run', filename=preferred)
    if preferred != 'results.pkl':
        # This must never be read when a higher-priority artifact exists.
        (directory / 'results.pkl').write_bytes(b'not a pickle')
    if preferred == 'results_slim.pkl':
        (directory / 'results_with_new_moon.pkl').write_bytes(b'not a pickle')
    values, config, path = load_saved_results(directory)
    assert values == {'source': preferred}
    assert config == {'estimators_dict': {}}
    assert path.name == preferred


def test_depth_statistics_use_actual_repeats_and_absolute_errors():
    results = {
        (5, 2500): {'nc_wc_arr': np.array([-1., 3., 2.])},
        (5, 5000000): {'nc_wc_arr': np.array([-2., 4.])},
        (15, 5000000): {'nc_wc_arr': np.array([2.])},
    }
    stats = compute_targeting_depth_stats(results)
    row = stats.loc[(5, 5000000)]
    assert row['Repeats'] == 2
    assert row['Mean WC'] == 1
    assert row['MAE'] == 3
    assert row['MC SE'] == pytest.approx(3)
    assert np.isnan(stats.loc[(15, 5000000), 'MC SE'])


def test_depth_plot_handles_partial_grid_and_includes_largest_size():
    results = {
        (5, 2500): {'nc_wc_arr': np.array([0.01, 0.03])},
        (5, 5000000): {'nc_wc_arr': np.array([0.001, 0.003])},
        (15, 2500): {'nc_wc_arr': np.array([0.1, 0.12])},
    }
    config = {
        'comparative_statics': {
            'depth_list': [5, 10, 15], 'sample_size_list': [2500, 10000, 5000000],
        },
        'dgp_params': {'base_effect_vars': [{'value': 1}, {'value': 1.01}]},
    }
    fig, ax = plt.subplots()
    try:
        plot_targeting_wc_depth_sample_size(
            results, config, ax=ax, xscale='log', show_mc_interval=True,
        )
        assert ax.get_xscale() == 'log'
        assert len(ax.lines) == 2
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [2500, 5000000])
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [200, 20])
        assert len(ax.collections) == 2
    finally:
        plt.close(fig)


def test_slim_policy_comparison_preserves_paired_values_without_selections():
    results = {(1, 1.01): {
        'nc_val_true_arr': np.array([1., 4., 7.]),
        'sample_splitting_val_true_arr': np.array([0., 3., 6.]),
    }}
    row = compute_targeting_policy_comparison(results).iloc[0]
    assert row['Paired value difference'] == 1
    # Marginal values vary, but their paired difference is constant.
    assert row['Difference MC SE'] == 0
    assert np.isnan(row['Selection agreement'])
    assert np.isnan(row['Agreement MC SE'])


def test_selection_agreement_se_uses_repeat_level_rates():
    values = {
        'nc_val_true_arr': np.array([1., 2.]),
        'sample_splitting_val_true_arr': np.array([0., 1.]),
        'nc_selection_arr': np.array([[0, 0, 0], [1, 1, 1]]),
        'sample_splitting_selection_arr': np.array([[0, 0, 0], [0, 0, 0]]),
    }
    row = compute_targeting_policy_comparison({2: values}).iloc[0]
    assert row['Selection agreement'] == 0.5
    assert row['Agreement MC SE'] == pytest.approx(0.5)


def test_compact_agreement_rates_reproduce_full_selection_statistics():
    values = {
        'nc_val_true_arr': np.array([1., 2., 3.]),
        'sample_splitting_val_true_arr': np.array([0., 1., 2.]),
        'nc_selection_arr': np.array([[0, 1, 0, 1], [0, 0, 0, 0], [1, 1, 1, 1]]),
        'sample_splitting_selection_arr': np.array([[0, 1, 0, 1], [1, 0, 0, 0], [0, 0, 0, 0]]),
    }
    expected = compute_targeting_policy_comparison({2: values})
    compact = {key: value for key, value in values.items() if 'selection' not in key}
    compact['sample_splitting_agreement_arr'] = (
        values['nc_selection_arr'] == values['sample_splitting_selection_arr']
    ).mean(axis=1)
    actual = compute_targeting_policy_comparison({2: compact})
    np.testing.assert_allclose(
        actual.iloc[0, 1:].to_numpy(dtype=float),
        expected.iloc[0, 1:].to_numpy(dtype=float),
    )


@pytest.mark.parametrize('rates', [[0.5], [0.5, 1.1], [0.5, np.nan]])
def test_compact_agreement_rates_validate_repeat_count_and_range(rates):
    values = {
        'nc_val_true_arr': np.array([1., 2.]),
        'sample_splitting_val_true_arr': np.array([0., 1.]),
        'sample_splitting_agreement_arr': np.asarray(rates),
    }
    with pytest.raises(ValueError, match='one rate in'):
        compute_targeting_policy_comparison({2: values})


def test_bernoulli_labels_follow_bars_in_run_order():
    # New runs group by gap before base rate; lexicographic sorting mislabels bars.
    effects = [(0.1, 0.101), (0.2, 0.201), (0.1, 0.105)]
    results = {tau: {'nc_wc_arr': np.array([value])}
               for tau, value in zip(effects, [1, 2, 3])}
    fig, ax = plt.subplots()
    try:
        plot_bernoulli_wc_comparison(
            results, {'estimators_dict': {}}, normalize=False, ax=ax,
        )
        assert [label.get_text() for label in ax.get_xticklabels()] == list(map(str, effects))
        assert [bar.get_height() for bar in ax.patches] == [1, 2, 3]
    finally:
        plt.close(fig)


def test_snr_labels_follow_sorted_summary_columns():
    results = {
        (1, 1.02): {'nc_wc_arr': np.array([2.])},
        (1, 1.01): {'nc_wc_arr': np.array([1.])},
    }
    config = {'estimators_dict': {}, 'dgp_params': {'noise_vars': [{'std': 1.}]}}
    fig, ax = plt.subplots()
    try:
        plot_snr_wc_comparison(results, config, normalize=False, ax=ax)
        assert [label.get_text() for label in ax.get_xticklabels()] == ['1.0%', '2.0%']
        assert [bar.get_height() for bar in ax.patches] == [1, 2]
    finally:
        plt.close(fig)
