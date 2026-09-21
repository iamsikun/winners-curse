import numpy as np

from winners_curse import experiments
from winners_curse.experiments import (
    get_max_jobs,
    prepare_experiment_params,
    select_by_sample_size,
)
from winners_curse.variables import UnivariateGaussian
from winners_curse.selective_inference import (
    _max_abs_normal_quantile,
    unconditional_inference,
)


TIERS = {
    0: {'max_jobs': 32, 'gb_per_job': 0.5},
    500000: {'max_jobs': 12, 'gb_per_job': 2.2},
    5000000: {'max_jobs': 2, 'gb_per_job': 15.0},
}


def tiered_config(**parallel):
    config = {'parallel': {'max_jobs': 32, 'max_jobs_by_sample_size': TIERS}}
    config['parallel'].update(parallel)
    return config


def test_select_by_sample_size_picks_largest_threshold_at_or_below():
    assert select_by_sample_size(TIERS, 2500)['max_jobs'] == 32
    assert select_by_sample_size(TIERS, 499999)['max_jobs'] == 32
    assert select_by_sample_size(TIERS, 500000)['max_jobs'] == 12
    assert select_by_sample_size(TIERS, 1000000)['max_jobs'] == 12
    assert select_by_sample_size(TIERS, 5000000)['max_jobs'] == 2
    # ab_test passes a per-arm list, which is not tiered
    assert select_by_sample_size(TIERS, [2500, 2500]) is None
    assert select_by_sample_size(None, 2500) is None


def test_get_max_jobs_applies_tier_and_memory_budget(monkeypatch):
    monkeypatch.setattr(experiments, 'available_memory_gb', lambda: 40.0)

    # 40 GB * 0.75 = 30 GB budget; the tier ceiling binds for small samples
    assert get_max_jobs(tiered_config(), sample_size=2500) == 32
    # 30 GB / 2.2 GB per job = 13 jobs, so the tier ceiling of 12 still binds
    assert get_max_jobs(tiered_config(), sample_size=500000) == 12
    # 30 GB / 15 GB per job = 2 jobs
    assert get_max_jobs(tiered_config(), sample_size=5000000) == 2

    # A busy machine tightens the cap below the tier ceiling
    monkeypatch.setattr(experiments, 'available_memory_gb', lambda: 12.0)
    assert get_max_jobs(tiered_config(), sample_size=500000) == 4
    assert get_max_jobs(tiered_config(), sample_size=5000000) == 1


def test_get_max_jobs_without_tiers_is_unchanged():
    assert get_max_jobs({'parallel': {'max_jobs': 32}}, sample_size=5000000) == 32
    assert get_max_jobs({}, default=24) == 24


def test_prepare_experiment_params_tiers_n_repeats():
    config = {
        'dgp_params': {
            'base_effect_vars': [],
            'noise_vars': [UnivariateGaussian(mean=0, std=1)],
            'char_func': 'x**2 + x',
        },
        'data_params': {'sample_size': 2500, 'targ_sample_size': 10000},
        'experiment_params': {
            'n_repeats': 500,
            'n_repeats_by_sample_size': {0: 500, 1000000: 100, 5000000: 20},
        },
    }
    for sample_size, expected in ((2500, 500), (1000000, 100), (5000000, 20)):
        _, data_params, experiment_params = prepare_experiment_params(
            config, tau=(1.0, 1.01), depth=5, sample_size=sample_size
        )
        assert experiment_params['n_repeats'] == expected
        assert data_params['sample_size'] == sample_size
    # the config itself is untouched, so later combos still see the baseline
    assert config['experiment_params']['n_repeats'] == 500


def test_projection_critical_value_matches_simulation_and_ignores_scale():
    # max of k iid |N(0, 1)| by simulation, for reference
    rng = np.random.default_rng(0)
    draws = np.abs(rng.standard_normal(size=(400000, 2)))
    simulated = np.quantile(draws.max(axis=1), 1 - 0.05 / 10)
    assert abs(_max_abs_normal_quantile(2, 0.05 / 10) - simulated) < 0.02

    # the critical value does not depend on the estimates or their scale
    means, stds = np.array([1.0, 2.0]), np.array([0.03, 0.11])
    _, _, c_small = unconditional_inference(means, stds, max_item_idx=1, alpha=0.05 / 10)
    _, _, c_large = unconditional_inference(means, stds * 100, max_item_idx=1, alpha=0.05 / 10)
    assert c_small == c_large == _max_abs_normal_quantile(2, 0.05 / 10)


def test_unconditional_interval_brackets_the_selected_estimate():
    means, stds = np.array([1.0, 2.0]), np.array([0.3, 0.4])
    lower, upper, c_alpha = unconditional_inference(means, stds, max_item_idx=1, alpha=0.05)
    assert lower == means[1] - c_alpha * stds[1]
    assert upper == means[1] + c_alpha * stds[1]
    assert lower < means[1] < upper
