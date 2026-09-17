import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from winners_curse.ab_test import estimate_treatment_effects
from winners_curse.empirical_ab import (
    aggregate_treatment_effects,
    filter_eligible_tests,
    make_base_test_record,
    prepare_arm_table,
    sample_splitting_draws,
)


def fixture_arm_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split": "exploratory",
                "source_row_id": 1,
                "clickability_test_id": "single",
                "impressions": 10,
                "clicks": 2,
                "click_rate": 0.2,
                "first_place": True,
                "winner": False,
            },
            {
                "split": "exploratory",
                "source_row_id": 1,
                "clickability_test_id": "tiny",
                "impressions": 1,
                "clicks": 1,
                "click_rate": 1.0,
                "first_place": True,
                "winner": False,
            },
            {
                "split": "exploratory",
                "source_row_id": 2,
                "clickability_test_id": "tiny",
                "impressions": 20,
                "clicks": 1,
                "click_rate": 0.05,
                "first_place": False,
                "winner": False,
            },
            {
                "split": "holdout",
                "source_row_id": 20,
                "clickability_test_id": "tie",
                "impressions": 10,
                "clicks": 5,
                "click_rate": 0.5,
                "first_place": False,
                "winner": False,
            },
            {
                "split": "holdout",
                "source_row_id": 10,
                "clickability_test_id": "tie",
                "impressions": 8,
                "clicks": 4,
                "click_rate": 0.5,
                "first_place": True,
                "winner": True,
            },
            {
                "split": "confirmatory",
                "source_row_id": 1,
                "clickability_test_id": "regular",
                "impressions": 12,
                "clicks": 3,
                "click_rate": 0.25,
                "first_place": False,
                "winner": False,
            },
            {
                "split": "confirmatory",
                "source_row_id": 2,
                "clickability_test_id": "regular",
                "impressions": 12,
                "clicks": 4,
                "click_rate": 1 / 3,
                "first_place": True,
                "winner": False,
            },
        ]
    )


def test_aggregate_effects_match_raw_bernoulli_estimator():
    impressions = np.array([5, 4, 6])
    clicks = np.array([2, 1, 0])
    effects, variances = aggregate_treatment_effects(impressions, clicks)

    samples = [
        np.r_[np.ones(click), np.zeros(impression - click)]
        for impression, click in zip(impressions, clicks)
    ]
    raw_effects, raw_variances = estimate_treatment_effects(samples, "bernoulli")

    np.testing.assert_allclose(effects, raw_effects)
    np.testing.assert_allclose(variances, raw_variances)


def test_eligibility_filters_single_arm_and_unsplittable_arms():
    arms = prepare_arm_table(fixture_arm_table())
    filtered, summary = filter_eligible_tests(arms, min_impressions=2)

    assert summary.total_tests == 4
    assert summary.excluded_single_arm_tests == 1
    assert summary.excluded_min_impressions_tests == 1
    assert summary.eligible_tests == 2
    assert set(filtered["clickability_test_id"]) == {"regular", "tie"}


def test_winner_tie_breaks_by_smallest_source_row_id():
    arms = prepare_arm_table(fixture_arm_table())
    filtered, _summary = filter_eligible_tests(arms, min_impressions=2)
    tie_group = filtered[filtered["clickability_test_id"] == "tie"]

    record = make_base_test_record(tie_group)

    assert record["selected_source_row_id"] == 10
    assert record["max_ctr_tie_count"] == 2
    assert record["naive_selected_ctr"] == 0.5


def test_sample_splitting_hypergeometric_preserves_observed_clicks():
    impressions = np.array([10, 12])
    clicks = np.array([3, 4])
    draws = sample_splitting_draws(
        impressions=impressions,
        clicks=clicks,
        n_splits=25,
        estimation_split=0.5,
        rng=np.random.default_rng(123),
        full_sample_winner_idx=1,
    )

    np.testing.assert_array_equal(
        draws["est_clicks"] + draws["eval_clicks"],
        np.repeat(clicks[:, None], 25, axis=1),
    )
    np.testing.assert_array_equal(
        draws["est_impressions"] + draws["eval_impressions"],
        impressions,
    )
    assert draws["validation_selected_ctr"].shape == (25,)
    assert draws["selected_arm"].shape == (25,)


def run_smoke_cli(input_path: Path, output_dir: Path) -> Path:
    script = Path(__file__).resolve().parents[1] / "scripts" / "upworthy_empirical_ab.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--n-bootstraps",
            "5",
            "--n-sample-splits",
            "5",
            "--hybrid-n-simulations",
            "200",
            "--checkpoint-size",
            "1",
            "--max-jobs",
            "1",
            "--seed",
            "42",
            "--min-impressions",
            "2",
        ],
        check=True,
        env=env,
        cwd=Path(__file__).resolve().parents[1],
    )
    runs = sorted(output_dir.glob("upworthy_empirical_ab_*"))
    assert len(runs) == 1
    return runs[0]


def test_smoke_cli_writes_expected_artifacts_and_is_deterministic(tmp_path):
    input_path = tmp_path / "fixture.csv"
    fixture_arm_table().to_csv(input_path, index=False)

    run_a = run_smoke_cli(input_path, tmp_path / "out_a")
    run_b = run_smoke_cli(input_path, tmp_path / "out_b")

    expected = {
        "arm_level_input.csv.gz",
        "per_test_estimates.csv.gz",
        "estimator_draws.npz",
        "summary.json",
        "summary_tables.csv",
        "run_manifest.json",
        "experiment.log",
    }
    assert expected.issubset({path.name for path in run_a.iterdir()})
    assert (run_a / "checkpoints" / "chunk_000000_000001.pkl").exists()

    per_test_a = pd.read_csv(run_a / "per_test_estimates.csv.gz")
    per_test_b = pd.read_csv(run_b / "per_test_estimates.csv.gz")
    pd.testing.assert_frame_equal(per_test_a, per_test_b)
    assert per_test_a.shape[0] == 2
    assert {
        "standard_bootstrap_corrected_ctr",
        "moon_bootstrap_corrected_ctr",
        "sample_splitting_corrected_ctr",
        "eb_normal_corrected_ctr",
        "hybrid_si_corrected_ctr",
    }.issubset(per_test_a.columns)

    with np.load(run_a / "estimator_draws.npz") as draws:
        assert draws["standard_bootstrap_bias"].shape == (2, 5)
        assert draws["moon_bootstrap_bias"].shape == (2, 5)
        assert draws["sample_splitting_validation_ctr"].shape == (2, 5)
