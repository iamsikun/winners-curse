import numpy as np

from winners_curse.analysis import (
    compute_bernoulli_sum_stats,
    compute_snr_sum_stats,
    generate_bernoulli_sum_stats_latex_table,
    generate_snr_sum_stats_latex_table,
    generate_targeting_model_comparison_latex_table,
)


def base_config():
    return {
        "dgp_params": {
            "noise_vars": [{"std": 1.0}],
            "base_effects": [
                {"value": 1.0},
                {"value": 1.1},
            ],
        },
        "estimators_dict": {
            "sample_splitting": {"display_name": "Sample Splitting"},
            "standard_bootstrap": {"display_name": "Standard Bootstrap"},
        },
    }


def test_snr_mae_uses_absolute_errors_before_averaging():
    results = {
        (1.0, 1.1): {
            "nc_wc_arr": np.array([-0.02, 0.04]),
            "sample_splitting_wc_arr": np.array([-0.01, 0.03]),
            "standard_bootstrap_wc_arr": np.array([-0.005, 0.015]),
        }
    }

    signed = compute_snr_sum_stats(
        results, base_config(), ["mean"], normalize=True, metric="wc"
    )
    mae = compute_snr_sum_stats(
        results, base_config(), ["mean"], normalize=False, metric="mae"
    )

    assert np.isclose(signed.loc[("No Correction", "mean"), "10.0%"], 10.0)
    assert np.isclose(mae.loc[("No Correction", "mean"), "10.0%"], 0.03)
    assert np.isclose(mae.loc[("Sample Splitting", "mean"), "10.0%"], 0.02)


def test_snr_mae_preserves_estimator_order():
    results = {
        (1.0, 1.1): {
            "nc_wc_arr": np.array([0.01]),
            "sample_splitting_wc_arr": np.array([0.02]),
            "standard_bootstrap_wc_arr": np.array([0.03]),
        }
    }

    stats = compute_snr_sum_stats(
        results, base_config(), ["mean"], normalize=False, metric="mae"
    )

    assert list(stats.index.get_level_values("Estimator").unique()) == [
        "No Correction",
        "Sample Splitting",
        "Standard Bootstrap",
    ]


def test_snr_mae_latex_can_omit_standard_deviation():
    results = {
        (1.0, 1.1): {
            "nc_wc_arr": np.array([-0.02, 0.04]),
            "sample_splitting_wc_arr": np.array([-0.01, 0.03]),
            "standard_bootstrap_wc_arr": np.array([-0.005, 0.015]),
        }
    }

    with_std = generate_snr_sum_stats_latex_table(
        results,
        base_config(),
        normalize=False,
        decimals=4,
        metric="mae",
    )
    without_std = generate_snr_sum_stats_latex_table(
        results,
        base_config(),
        normalize=False,
        decimals=4,
        metric="mae",
        include_std=False,
    )

    assert r"\begin{tabular}[c]" in with_std
    assert r"\begin{tabular}[c]" not in without_std
    assert "0.0300" in without_std


def test_sample_splitting_mae_uses_stored_estimator_error():
    results = {
        (1.0, 1.1): {
            "nc_wc_arr": np.array([1.0, -1.0]),
            "nc_val_true_arr": np.array([8.0, 8.0]),
            "nc_val_est_arr": np.array([9.0, 7.0]),
            "sample_splitting_wc_arr": np.array([0.02, -0.04]),
            "sample_splitting_val_true_arr": np.array([9.98, 10.04]),
            "sample_splitting_val_est_arr": np.array([10.0, 10.0]),
            "standard_bootstrap_wc_arr": np.array([0.01, -0.01]),
        }
    }

    stats = compute_snr_sum_stats(
        results, base_config(), ["mean"], normalize=False, metric="mae"
    )

    assert np.isclose(stats.loc[("Sample Splitting", "mean"), "10.0%"], 0.03)


def test_bernoulli_mae_multiindex_columns_and_latex():
    config = base_config()
    results = {
        (0.1, 0.101): {
            "nc_wc_arr": np.array([0.001, -0.002]),
            "sample_splitting_wc_arr": np.array([0.0005, -0.0015]),
            "standard_bootstrap_wc_arr": np.array([0.00025, -0.00075]),
        },
        (0.1, 0.105): {
            "nc_wc_arr": np.array([0.005, -0.010]),
            "sample_splitting_wc_arr": np.array([0.001, -0.002]),
            "standard_bootstrap_wc_arr": np.array([0.0005, -0.0010]),
        },
    }

    stats = compute_bernoulli_sum_stats(
        results, config, ["mean"], normalize=False, metric="mae"
    )
    latex = generate_bernoulli_sum_stats_latex_table(
        results, config, normalize=False, metric="mae", decimals=4
    )

    assert list(stats.columns) == [
        (0.0010000000000000009, (0.1, 0.101)),
        (0.0049999999999999906, (0.1, 0.105)),
    ]
    assert np.isclose(
        stats.loc[("No Correction", "mean"), stats.columns[0]].values[0],
        0.0015,
    )
    assert "No Correction" in latex
    assert "0.0015" in latex
    assert "0.0015\\%" not in latex


def test_targeting_model_comparison_supports_mae_without_std():
    correct_results = {
        (1.0, 1.1): {
            "nc_wc_arr": np.array([-0.02, 0.04]),
            "sample_splitting_wc_arr": np.array([-0.01, 0.03]),
            "standard_bootstrap_wc_arr": np.array([-0.005, 0.015]),
        }
    }
    forest_results = {
        2: {
            "nc_wc_arr": np.array([-0.10, 0.20]),
            "sample_splitting_wc_arr": np.array([-0.05, 0.15]),
            "standard_bootstrap_wc_arr": np.array([-0.025, 0.075]),
        }
    }

    latex = generate_targeting_model_comparison_latex_table(
        forest_results,
        correct_results,
        base_config(),
        normalize=False,
        stats="mean",
        decimals=4,
        metric="mae",
        include_std=False,
    )

    assert "Correct" in latex
    assert "Functional Form" in latex
    assert "Causal Forest" in latex
    assert "0.0300" in latex
    assert "0.1500" in latex
    assert r"\begin{tabular}[c]{@{}c@{}}0.0300" not in latex
