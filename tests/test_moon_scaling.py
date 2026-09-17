import numpy as np
import pytest

from winners_curse import ab_test, structural, targeting
from winners_curse.analysis import tabulate_sample_size_moon_wc


def test_ab_moon_subtracts_bias_at_original_sample_size():
    samples = [np.arange(16, dtype=float), np.arange(16, dtype=float) + 0.2]
    optimization = {'optimizer': ab_test.select_higher_effect, 'params': {}}
    seed = 12
    raw_bias = []
    for draw in range(5):
        rng = np.random.RandomState(seed + draw)
        means = np.array([rng.choice(arm, size=4, replace=True).mean() for arm in samples])
        winner = means.argmax()
        raw_bias.append(means[winner] - samples[winner].mean())
    expected = max(arm.mean() for arm in samples) - np.mean(raw_bias) * 0.5

    _, actual = ab_test.bootstrap_correction_estimate(
        samples, optimization, response_type='continuous', bootstrap_method='moon',
        power=0.5, n_bootstraps=5, seed=seed,
    )
    assert actual == pytest.approx(expected)


def test_targeting_moon_always_scales_bias(monkeypatch):
    def correction(**kwargs):
        def evaluator(selection, estimate):
            return estimate
        measured_bias = kwargs['wc_func'](None, 10.0, 2.0, evaluator)
        return None, measured_bias

    monkeypatch.setattr(targeting, 'bootstrap_correction_estimator', correction)
    value = targeting.get_wc_moon_boot_dstn(
        cust_features=np.zeros((100, 1)), treatments=np.zeros(100),
        outcomes=np.zeros(100), targ_cust_features=np.zeros((5, 1)),
        optimization_params={}, estimator=None, estimator_params={},
        emp_targ_te_arr=np.zeros((5, 2)), emp_targ_var_arr=np.zeros((5, 2)),
        power=0.5,
    )
    assert value == pytest.approx(8 * np.sqrt(10 / 100))


def test_structural_moon_always_scales_bias(monkeypatch):
    monkeypatch.setattr(structural, 'estimate_fixed_effects', lambda records, n: np.array([2.0]))
    monkeypatch.setattr(structural, 'obj_func', lambda **kwargs: 2.0)
    results = structural.get_wc_moon_boot_dstn(
        purchase_records=np.zeros(100), emp_fixed_effects=np.array([1.0]),
        optimization_params={
            'pick': {'optimizer': lambda **kwargs: (np.array([0]), 10.0), 'params': {}}
        },
        power=0.5, n_bootstraps=2, n_jobs=1, seed=1,
    )
    np.testing.assert_allclose(results['pick'], 8 * np.sqrt(10 / 100))


def test_sample_size_table_uses_supported_moon_parameters():
    config = {
        'dgp_params': {'base_effects': [{'value': 1.0}, {'value': 1.1}]},
        'estimators_dict': {
            'standard_bootstrap': {'params': {'bootstrap_method': 'standard'}},
            'moon_custom_name': {'params': {'bootstrap_method': 'moon', 'power': 0.4}},
        },
    }
    results = {100: {
        'nc_wc_arr': np.array([1.0, 3.0]),
        'standard_bootstrap_wc_arr': np.array([0.5, 1.5]),
        'moon_custom_name_wc_arr': np.array([0.1, 0.3]),
    }}
    table = tabulate_sample_size_moon_wc(results, config, [100], normalize=False)
    assert list(table.columns) == [
        ('No Correction', ''), ('Standard Bootstrap', ''), ('γ=0.4', 'Scaled'),
    ]
    assert table.loc[100, ('γ=0.4', 'Scaled')] == pytest.approx(0.2)
