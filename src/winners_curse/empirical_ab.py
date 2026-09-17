"""Aggregate-count A/B helpers for empirical winner's curse studies.

The functions in this module operate on arm-level binomial summaries
(`clicks`, `impressions`) instead of expanding data to one row per impression.
They mirror the A/B estimators used elsewhere in the package where practical,
but keep draw-level outputs so empirical runs can be audited without rerunning
expensive Monte Carlo estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.stats import truncnorm


ESTIMATOR_KEYS = (
    "standard_bootstrap",
    "moon_bootstrap",
    "sample_splitting",
    "eb_normal",
    "hybrid_si",
)

REQUIRED_ARM_COLUMNS = (
    "split",
    "source_row_id",
    "clickability_test_id",
    "impressions",
    "clicks",
    "click_rate",
    "first_place",
    "winner",
)


@dataclass(frozen=True)
class EligibilitySummary:
    """Counts describing the empirical test filter."""

    total_tests: int
    eligible_tests: int
    excluded_tests: int
    excluded_single_arm_tests: int
    excluded_min_impressions_tests: int
    min_arms: int
    min_impressions: int

    def to_dict(self) -> dict[str, int]:
        return {
            "total_tests": self.total_tests,
            "eligible_tests": self.eligible_tests,
            "excluded_tests": self.excluded_tests,
            "excluded_single_arm_tests": self.excluded_single_arm_tests,
            "excluded_min_impressions_tests": self.excluded_min_impressions_tests,
            "min_arms": self.min_arms,
            "min_impressions": self.min_impressions,
        }


def coerce_bool_series(series: pd.Series) -> pd.Series:
    """Convert archive boolean-like values into strict booleans."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    normalized = series.fillna(False).astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})


def prepare_arm_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize an aggregate arm table.

    Params:
    -------
    raw:
        DataFrame with one row per tested arm.

    Returns:
    --------
    pd.DataFrame
        Sorted copy with computed CTR and deterministic ``arm_ordinal``.
    """

    missing = [col for col in REQUIRED_ARM_COLUMNS if col not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    arms = raw.copy()
    arms["clickability_test_id"] = arms["clickability_test_id"].astype(str)
    arms["split"] = arms["split"].astype(str)
    arms["source_row_id"] = pd.to_numeric(arms["source_row_id"], errors="raise")
    arms["impressions"] = pd.to_numeric(arms["impressions"], errors="raise").astype(np.int64)
    arms["clicks"] = pd.to_numeric(arms["clicks"], errors="raise").astype(np.int64)
    arms["first_place"] = coerce_bool_series(arms["first_place"])
    arms["winner"] = coerce_bool_series(arms["winner"])

    if (arms["impressions"] <= 0).any():
        raise ValueError("All arms must have positive impressions.")
    if ((arms["clicks"] < 0) | (arms["clicks"] > arms["impressions"])).any():
        raise ValueError("All arms must satisfy 0 <= clicks <= impressions.")

    arms["click_rate"] = arms["clicks"] / arms["impressions"]
    arms = arms.sort_values(
        ["clickability_test_id", "source_row_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    arms["arm_ordinal"] = arms.groupby("clickability_test_id", sort=False).cumcount()
    return arms


def filter_eligible_tests(
    arms: pd.DataFrame,
    min_arms: int = 2,
    min_impressions: int = 1000,
    limit_tests: int | None = None,
) -> tuple[pd.DataFrame, EligibilitySummary]:
    """Filter to tests usable by within-test winner selection and splitting.

    The default empirical specification requires at least 1,000 impressions
    in every arm. Callers may lower the threshold for tests and robustness
    analyses, but each arm must remain splittable into two nonempty samples.
    """

    if min_arms < 2:
        raise ValueError("min_arms must be at least 2.")
    if min_impressions < 2:
        raise ValueError("min_impressions must be at least 2.")

    test_stats = arms.groupby("clickability_test_id", sort=True).agg(
        arms=("arm_ordinal", "size"),
        min_impressions=("impressions", "min"),
    )
    single_arm = test_stats["arms"] < min_arms
    low_impressions = (test_stats["arms"] >= min_arms) & (
        test_stats["min_impressions"] < min_impressions
    )
    eligible = ~(single_arm | low_impressions)
    eligible_ids = test_stats.index[eligible]

    if limit_tests is not None:
        if limit_tests <= 0:
            raise ValueError("limit_tests must be positive when provided.")
        eligible_ids = eligible_ids[:limit_tests]

    filtered = arms[arms["clickability_test_id"].isin(eligible_ids)].copy()
    filtered = filtered.sort_values(
        ["clickability_test_id", "source_row_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    filtered["arm_ordinal"] = filtered.groupby("clickability_test_id", sort=False).cumcount()

    summary = EligibilitySummary(
        total_tests=int(test_stats.shape[0]),
        eligible_tests=int(eligible.sum() if limit_tests is None else len(eligible_ids)),
        excluded_tests=int((~eligible).sum()),
        excluded_single_arm_tests=int(single_arm.sum()),
        excluded_min_impressions_tests=int(low_impressions.sum()),
        min_arms=min_arms,
        min_impressions=min_impressions,
    )
    return filtered, summary


def aggregate_treatment_effects(
    impressions: np.ndarray,
    clicks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return arm CTRs and sampling variances matching ``ab_test`` Bernoulli logic."""

    impressions = np.asarray(impressions, dtype=float)
    clicks = np.asarray(clicks, dtype=float)
    effects = clicks / impressions
    variances = effects * (1.0 - effects) / impressions
    return effects, variances


def _rng(seed: int, test_index: int, salt: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(test_index), int(salt)]))


def select_winner_index(
    effects: np.ndarray,
    source_row_ids: np.ndarray,
) -> tuple[int, int]:
    """Select max effect, breaking ties by smallest source row id."""

    effects = np.asarray(effects, dtype=float)
    source_row_ids = np.asarray(source_row_ids)
    max_effect = np.nanmax(effects)
    tied = np.flatnonzero(effects == max_effect)
    tie_count = int(tied.size)
    if tie_count == 1:
        return int(tied[0]), tie_count
    tied_source_ids = source_row_ids[tied]
    return int(tied[np.argmin(tied_source_ids)]), tie_count


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0 or not np.isfinite(denominator):
        return np.nan
    return float(numerator / denominator)


def _bias_stats(draws: np.ndarray) -> dict[str, float]:
    draws = np.asarray(draws, dtype=float)
    if draws.size == 0:
        return {
            "bias_mc_sd": np.nan,
            "bias_mc_se": np.nan,
            "bias_q025": np.nan,
            "bias_q50": np.nan,
            "bias_q975": np.nan,
        }
    sd = float(np.std(draws, ddof=1)) if draws.size > 1 else 0.0
    return {
        "bias_mc_sd": sd,
        "bias_mc_se": float(sd / np.sqrt(draws.size)) if draws.size else np.nan,
        "bias_q025": float(np.quantile(draws, 0.025)),
        "bias_q50": float(np.quantile(draws, 0.5)),
        "bias_q975": float(np.quantile(draws, 0.975)),
    }


def _add_estimator_record(
    record: dict,
    estimator_key: str,
    naive_ctr: float,
    corrected_ctr: float,
    bias_draws: np.ndarray | None = None,
) -> None:
    estimated_bias = naive_ctr - corrected_ctr if np.isfinite(corrected_ctr) else np.nan
    record[f"{estimator_key}_corrected_ctr"] = float(corrected_ctr)
    record[f"{estimator_key}_estimated_bias"] = float(estimated_bias)
    record[f"{estimator_key}_bias_pct_of_naive_ctr"] = _safe_ratio(
        estimated_bias, naive_ctr
    )
    stats = _bias_stats(bias_draws) if bias_draws is not None else _bias_stats(np.array([]))
    for stat_key, value in stats.items():
        record[f"{estimator_key}_{stat_key}"] = value


def standard_bootstrap_aggregate(
    impressions: np.ndarray,
    clicks: np.ndarray,
    n_bootstraps: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | float]:
    """Estimate bootstrap winner's curse draws from aggregate binomial arms."""

    effects, _ = aggregate_treatment_effects(impressions, clicks)
    impressions = np.asarray(impressions, dtype=np.int64)
    boot_clicks = np.vstack(
        [rng.binomial(int(n), float(p), size=n_bootstraps) for n, p in zip(impressions, effects)]
    )
    boot_ctr = boot_clicks / impressions[:, None]
    selected = np.argmax(boot_ctr, axis=0).astype(np.int16)
    draw_idx = np.arange(n_bootstraps)
    bias_draws = boot_ctr[selected, draw_idx] - effects[selected]
    return {
        "bias_draws": bias_draws.astype(np.float64),
        "selected_arm": selected,
        "mean_bias": float(np.mean(bias_draws)),
    }


def moon_bootstrap_aggregate(
    impressions: np.ndarray,
    clicks: np.ndarray,
    n_bootstraps: int,
    power: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | float]:
    """Estimate scaled m-out-of-n bootstrap winner's curse draws."""

    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1).")

    effects, _ = aggregate_treatment_effects(impressions, clicks)
    impressions = np.asarray(impressions, dtype=np.int64)
    m_sizes = np.maximum(1, np.floor(impressions.astype(float) ** power).astype(np.int64))
    boot_clicks = np.vstack(
        [rng.binomial(int(m), float(p), size=n_bootstraps) for m, p in zip(m_sizes, effects)]
    )
    boot_ctr = boot_clicks / m_sizes[:, None]
    selected = np.argmax(boot_ctr, axis=0).astype(np.int16)
    draw_idx = np.arange(n_bootstraps)
    raw_bias = boot_ctr[selected, draw_idx] - effects[selected]

    avg_n = float(np.mean(impressions))
    avg_m = max(1, int(avg_n**power))
    scale = np.sqrt(avg_m / avg_n)
    bias_draws = raw_bias * scale
    return {
        "bias_draws": bias_draws.astype(np.float64),
        "selected_arm": selected,
        "m_sizes": m_sizes,
        "scale": float(scale),
        "mean_bias": float(np.mean(bias_draws)),
    }


def sample_splitting_draws(
    impressions: np.ndarray,
    clicks: np.ndarray,
    n_splits: int,
    estimation_split: float,
    rng: np.random.Generator,
    full_sample_winner_idx: int,
) -> dict[str, np.ndarray | float]:
    """Run aggregate faithful sample splitting with hypergeometric allocations."""

    if not 0.0 < estimation_split < 1.0:
        raise ValueError("estimation_split must be in (0, 1).")

    impressions = np.asarray(impressions, dtype=np.int64)
    clicks = np.asarray(clicks, dtype=np.int64)
    est_impressions = np.floor(impressions * estimation_split).astype(np.int64)
    eval_impressions = impressions - est_impressions
    if np.any(est_impressions <= 0) or np.any(eval_impressions <= 0):
        raise ValueError("Each arm must have positive estimation and evaluation impressions.")

    est_clicks = np.vstack(
        [
            rng.hypergeometric(
                ngood=int(c),
                nbad=int(n - c),
                nsample=int(est_n),
                size=n_splits,
            )
            for n, c, est_n in zip(impressions, clicks, est_impressions)
        ]
    )
    eval_clicks = clicks[:, None] - est_clicks

    est_ctr = est_clicks / est_impressions[:, None]
    eval_ctr = eval_clicks / eval_impressions[:, None]
    selected = np.argmax(est_ctr, axis=0).astype(np.int16)
    draw_idx = np.arange(n_splits)
    validation_selected_ctr = eval_ctr[selected, draw_idx]
    validation_full_winner_ctr = eval_ctr[int(full_sample_winner_idx), draw_idx]
    estimation_selected_ctr = est_ctr[selected, draw_idx]

    return {
        "selected_arm": selected,
        "est_clicks": est_clicks,
        "eval_clicks": eval_clicks,
        "est_impressions": est_impressions,
        "eval_impressions": eval_impressions,
        "validation_selected_ctr": validation_selected_ctr.astype(np.float64),
        "validation_full_winner_ctr": validation_full_winner_ctr.astype(np.float64),
        "estimation_selected_ctr": estimation_selected_ctr.astype(np.float64),
        "full_winner_selection_rate": float(np.mean(selected == int(full_sample_winner_idx))),
        "disagreement_rate": float(np.mean(selected != int(full_sample_winner_idx))),
        "mean_validation_decision_loss_vs_full_winner": float(
            np.mean(validation_full_winner_ctr - validation_selected_ctr)
        ),
    }


def empirical_bayes_normal_aggregate(
    effects: np.ndarray,
    sampling_vars: np.ndarray,
    selected_idx: int,
) -> dict[str, float | np.ndarray]:
    """Repo-compatible empirical Bayes normal-prior shrinkage."""

    effects = np.asarray(effects, dtype=float)
    sampling_vars = np.asarray(sampling_vars, dtype=float)
    prior_mean = float(np.mean(effects))
    prior_var = float(np.var(effects))
    denom = prior_var + sampling_vars
    with np.errstate(divide="ignore", invalid="ignore"):
        shrink = np.divide(
            prior_var,
            denom,
            out=np.zeros_like(sampling_vars, dtype=float),
            where=denom > 0,
        )
    posterior = shrink * effects + (1.0 - shrink) * prior_mean
    return {
        "posterior_means": posterior,
        "corrected_ctr": float(posterior[int(selected_idx)]),
        "prior_mean": prior_mean,
        "prior_std": float(np.sqrt(prior_var)),
    }


def hybrid_selective_inference_aggregate(
    effects: np.ndarray,
    sampling_vars: np.ndarray,
    selected_idx: int,
    rng: np.random.Generator,
    quantile: float = 0.5,
    n_simulations: int = 10_000,
) -> dict[str, float | int | str]:
    """Hybrid selective inference with explicit solver diagnostics."""

    if quantile != 0.5:
        raise ValueError("Only quantile=0.5 is supported for the median-unbiased estimator.")
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive.")

    effects = np.asarray(effects, dtype=float)
    sampling_vars = np.asarray(sampling_vars, dtype=float)
    std = np.sqrt(np.maximum(sampling_vars, 0.0))
    selected_idx = int(selected_idx)
    selected_mean = float(effects[selected_idx])
    selected_std = float(std[selected_idx])

    if effects.size < 2:
        return {
            "corrected_ctr": np.nan,
            "solver_ier": 0,
            "solver_message": "At least two arms are required.",
            "c_beta": np.nan,
        }
    if not np.isfinite(selected_std) or selected_std <= 0:
        return {
            "corrected_ctr": np.nan,
            "solver_ier": 0,
            "solver_message": "Selected arm has non-positive standard error.",
            "c_beta": np.nan,
        }

    max_abs_t = np.max(np.abs(rng.standard_normal(size=(n_simulations, effects.size))), axis=1)
    c_beta = float(np.quantile(max_abs_t, 1.0 - 0.05 / 10.0))
    second_max_mean = float(np.delete(effects, selected_idx).max())
    lower_region = selected_mean - c_beta * selected_std
    upper_region = selected_mean + c_beta * selected_std

    def local_truncated_normal_cdf(x: float, mu: float) -> float:
        if mu < lower_region:
            return -10000.0
        if mu > upper_region:
            return 10000.0

        trunc_lb = max(second_max_mean, lower_region)
        trunc_ub = upper_region
        trunc_lb = max(trunc_lb, mu - c_beta * selected_std)
        trunc_ub = min(trunc_ub, mu + c_beta * selected_std)
        if trunc_lb >= trunc_ub:
            return np.nan
        return float(
            truncnorm.cdf(
                x,
                (trunc_lb - mu) / selected_std,
                (trunc_ub - mu) / selected_std,
                loc=mu,
                scale=selected_std,
            )
        )

    def objective(mu_arr: np.ndarray) -> float:
        cdf = local_truncated_normal_cdf(selected_mean, float(mu_arr[0]))
        if not np.isfinite(cdf):
            return 10000.0
        return cdf - 1.0 + quantile

    root, _info, ier, message = fsolve(
        func=objective,
        x0=np.array([selected_mean]),
        full_output=True,
    )
    corrected = float(root[0]) if ier == 1 and np.isfinite(root[0]) else np.nan
    return {
        "corrected_ctr": corrected,
        "solver_ier": int(ier),
        "solver_message": str(message),
        "c_beta": c_beta,
    }


def make_base_test_record(group: pd.DataFrame) -> dict[str, object]:
    """Create deterministic per-test metadata and naive selected-winner fields."""

    ordered = group.sort_values("arm_ordinal", kind="mergesort")
    impressions = ordered["impressions"].to_numpy(dtype=np.int64)
    clicks = ordered["clicks"].to_numpy(dtype=np.int64)
    effects, sampling_vars = aggregate_treatment_effects(impressions, clicks)
    source_row_ids = ordered["source_row_id"].to_numpy()
    selected_idx, tie_count = select_winner_index(effects, source_row_ids)
    sorted_effects = np.sort(effects)
    runner_up_ctr = float(sorted_effects[-2])
    selected = ordered.iloc[selected_idx]

    return {
        "clickability_test_id": str(selected["clickability_test_id"]),
        "split": str(selected["split"]),
        "n_arms": int(len(ordered)),
        "total_impressions": int(impressions.sum()),
        "total_clicks": int(clicks.sum()),
        "weighted_test_ctr": float(clicks.sum() / impressions.sum()),
        "selected_arm_ordinal": int(selected["arm_ordinal"]),
        "selected_source_row_id": int(selected["source_row_id"]),
        "selected_impressions": int(selected["impressions"]),
        "selected_clicks": int(selected["clicks"]),
        "naive_selected_ctr": float(effects[selected_idx]),
        "runner_up_ctr": runner_up_ctr,
        "winner_runner_up_gap": float(effects[selected_idx] - runner_up_ctr),
        "max_ctr_tie_count": tie_count,
        "selected_first_place": bool(selected["first_place"]),
        "selected_winner": bool(selected["winner"]),
        "_selected_idx": selected_idx,
        "_effects": effects,
        "_sampling_vars": sampling_vars,
        "_impressions": impressions,
        "_clicks": clicks,
    }


def estimate_aggregate_test(
    group: pd.DataFrame,
    test_index: int,
    seed: int,
    estimators: Iterable[str],
    n_bootstraps: int,
    n_sample_splits: int,
    moon_power: float,
    estimation_split: float,
    hybrid_n_simulations: int = 10_000,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Estimate all requested aggregate winner's curse corrections for one test."""

    estimator_set = set(estimators)
    unknown = estimator_set.difference(ESTIMATOR_KEYS)
    if unknown:
        raise ValueError(f"Unknown estimators: {sorted(unknown)}")

    record = make_base_test_record(group)
    draws: dict[str, np.ndarray] = {}
    effects = record.pop("_effects")
    sampling_vars = record.pop("_sampling_vars")
    impressions = record.pop("_impressions")
    clicks = record.pop("_clicks")
    selected_idx = int(record.pop("_selected_idx"))
    naive_ctr = float(record["naive_selected_ctr"])

    if "standard_bootstrap" in estimator_set:
        boot = standard_bootstrap_aggregate(
            impressions=impressions,
            clicks=clicks,
            n_bootstraps=n_bootstraps,
            rng=_rng(seed, test_index, 101),
        )
        _add_estimator_record(
            record,
            "standard_bootstrap",
            naive_ctr,
            naive_ctr - float(boot["mean_bias"]),
            np.asarray(boot["bias_draws"]),
        )
        draws["standard_bootstrap_bias"] = np.asarray(boot["bias_draws"], dtype=np.float32)
        draws["standard_bootstrap_selected_arm"] = np.asarray(
            boot["selected_arm"], dtype=np.int16
        )

    if "moon_bootstrap" in estimator_set:
        moon = moon_bootstrap_aggregate(
            impressions=impressions,
            clicks=clicks,
            n_bootstraps=n_bootstraps,
            power=moon_power,
            rng=_rng(seed, test_index, 202),
        )
        _add_estimator_record(
            record,
            "moon_bootstrap",
            naive_ctr,
            naive_ctr - float(moon["mean_bias"]),
            np.asarray(moon["bias_draws"]),
        )
        record["moon_bootstrap_scale"] = float(moon["scale"])
        draws["moon_bootstrap_bias"] = np.asarray(moon["bias_draws"], dtype=np.float32)
        draws["moon_bootstrap_selected_arm"] = np.asarray(moon["selected_arm"], dtype=np.int16)

    if "sample_splitting" in estimator_set:
        split = sample_splitting_draws(
            impressions=impressions,
            clicks=clicks,
            n_splits=n_sample_splits,
            estimation_split=estimation_split,
            rng=_rng(seed, test_index, 303),
            full_sample_winner_idx=selected_idx,
        )
        validation_ctr = np.asarray(split["validation_selected_ctr"])
        bias_draws = naive_ctr - validation_ctr
        corrected_ctr = float(np.mean(validation_ctr))
        _add_estimator_record(
            record,
            "sample_splitting",
            naive_ctr,
            corrected_ctr,
            bias_draws,
        )
        record["sample_splitting_mean_validation_selected_ctr"] = corrected_ctr
        record["sample_splitting_mean_estimation_selected_ctr"] = float(
            np.mean(split["estimation_selected_ctr"])
        )
        record["sample_splitting_mean_validation_full_winner_ctr"] = float(
            np.mean(split["validation_full_winner_ctr"])
        )
        record["sample_splitting_full_winner_selection_rate"] = float(
            split["full_winner_selection_rate"]
        )
        record["sample_splitting_disagreement_rate"] = float(split["disagreement_rate"])
        record["sample_splitting_mean_validation_decision_loss_vs_full_winner"] = float(
            split["mean_validation_decision_loss_vs_full_winner"]
        )
        draws["sample_splitting_validation_ctr"] = validation_ctr.astype(np.float32)
        draws["sample_splitting_selected_arm"] = np.asarray(split["selected_arm"], dtype=np.int16)

    if "eb_normal" in estimator_set:
        eb = empirical_bayes_normal_aggregate(
            effects=effects,
            sampling_vars=sampling_vars,
            selected_idx=selected_idx,
        )
        _add_estimator_record(
            record,
            "eb_normal",
            naive_ctr,
            float(eb["corrected_ctr"]),
            None,
        )
        record["eb_normal_prior_mean"] = float(eb["prior_mean"])
        record["eb_normal_prior_std"] = float(eb["prior_std"])

    if "hybrid_si" in estimator_set:
        hybrid = hybrid_selective_inference_aggregate(
            effects=effects,
            sampling_vars=sampling_vars,
            selected_idx=selected_idx,
            rng=_rng(seed, test_index, 404),
            quantile=0.5,
            n_simulations=hybrid_n_simulations,
        )
        _add_estimator_record(
            record,
            "hybrid_si",
            naive_ctr,
            float(hybrid["corrected_ctr"]),
            None,
        )
        record["hybrid_si_solver_ier"] = int(hybrid["solver_ier"])
        record["hybrid_si_solver_message"] = str(hybrid["solver_message"])
        record["hybrid_si_c_beta"] = float(hybrid["c_beta"])

    return record, draws


def stack_draws(draw_dicts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack per-test draw arrays, skipping estimators that were not run."""

    if not draw_dicts:
        return {}
    keys = sorted({key for draw_dict in draw_dicts for key in draw_dict})
    stacked: dict[str, np.ndarray] = {}
    for key in keys:
        arrays = [draw_dict[key] for draw_dict in draw_dicts if key in draw_dict]
        if len(arrays) == len(draw_dicts):
            stacked[key] = np.stack(arrays, axis=0)
    return stacked


def summarize_estimates(
    per_test: pd.DataFrame,
    estimators: Iterable[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create headline and subgroup summary tables."""

    df = per_test.copy()
    if df.empty:
        return pd.DataFrame(), {"n_tests": 0}

    df["arm_count_bin"] = pd.cut(
        df["n_arms"],
        bins=[1, 2, 3, 4, 5, 7, np.inf],
        labels=["2", "3", "4", "5", "6-7", "8+"],
        right=True,
    ).astype(str)
    df["impression_bin"] = pd.qcut(
        df["total_impressions"],
        q=4,
        duplicates="drop",
    ).astype(str)

    group_specs = [("overall", pd.Series("all", index=df.index))]
    group_specs.extend(
        [
            ("split", df["split"].astype(str)),
            ("arm_count_bin", df["arm_count_bin"]),
            ("impression_bin", df["impression_bin"]),
            ("selected_winner", df["selected_winner"].astype(str)),
            ("selected_first_place", df["selected_first_place"].astype(str)),
        ]
    )

    rows = []
    for group_name, labels in group_specs:
        for group_value, group_df in df.groupby(labels, dropna=False, sort=True):
            weights = group_df["selected_impressions"].to_numpy(dtype=float)
            for estimator in estimators:
                bias_col = f"{estimator}_estimated_bias"
                corrected_col = f"{estimator}_corrected_ctr"
                pct_col = f"{estimator}_bias_pct_of_naive_ctr"
                if bias_col not in group_df:
                    continue

                valid = group_df[bias_col].notna()
                valid_df = group_df.loc[valid]
                if valid_df.empty:
                    continue

                valid_weights = valid_df["selected_impressions"].to_numpy(dtype=float)
                bias = valid_df[bias_col].to_numpy(dtype=float)
                naive = valid_df["naive_selected_ctr"].to_numpy(dtype=float)
                corrected = valid_df[corrected_col].to_numpy(dtype=float)
                pct = valid_df[pct_col].to_numpy(dtype=float)
                weighted_bias = float(np.average(bias, weights=valid_weights))
                weighted_naive = float(np.average(naive, weights=valid_weights))

                rows.append(
                    {
                        "group": group_name,
                        "group_value": str(group_value),
                        "estimator": estimator,
                        "n_tests": int(valid_df.shape[0]),
                        "test_weighted_mean_bias": float(np.mean(bias)),
                        "test_weighted_median_bias": float(np.median(bias)),
                        "test_weighted_mean_corrected_ctr": float(np.mean(corrected)),
                        "test_weighted_mean_naive_ctr": float(np.mean(naive)),
                        "test_weighted_mean_bias_pct_of_naive_ctr": float(np.nanmean(pct)),
                        "winner_impression_weighted_mean_bias": weighted_bias,
                        "winner_impression_weighted_mean_naive_ctr": weighted_naive,
                        "winner_impression_weighted_mean_bias_pct_of_naive_ctr": float(
                            np.average(pct, weights=valid_weights)
                        ),
                        "aggregate_bias_pct_of_weighted_naive_ctr": _safe_ratio(
                            weighted_bias, weighted_naive
                        ),
                        "selected_impressions": int(valid_df["selected_impressions"].sum()),
                    }
                )

    summary_table = pd.DataFrame(rows)
    headline = {
        "n_tests": int(df.shape[0]),
        "n_arms": int(df["n_arms"].sum()),
        "estimators": list(estimators),
        "overall": summary_table[summary_table["group"] == "overall"].to_dict(
            orient="records"
        ),
    }
    return summary_table, headline
