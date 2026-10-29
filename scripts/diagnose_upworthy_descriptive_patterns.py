"""Reproduce the Upworthy comparative-statics appendix from saved estimates.

Run: uv run python scripts/diagnose_upworthy_descriptive_patterns.py
The regressions describe estimated corrections, not observed true bias.
"""
from __future__ import annotations

import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import patsy
import statsmodels
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "results/empirical_study/upworthy/upworthy_empirical_ab_20261005_143323"
COMPARISON_RUN = ROOT / "results/empirical_study/upworthy/upworthy_empirical_ab_20260927_105022"
ARM_GROUPS = ["2", "3", "4", "5", "6-7", "8+"]
REGRESSIONS = {
    "coauthor": "wc_pp ~ snr_pct + log_J",
    "add_sample_size": "wc_pp ~ snr_pct + log_J + log_n",
    "full": "wc_pp ~ snr_pct + log_J + log_n + log_sigma",
    "add_month": "wc_pp ~ snr_pct + log_J + log_n + log_sigma + C(month)",
    "flexible_controls": "wc_pp ~ snr_pct + C(n_arms) + bs(log_n, df=4) + bs(log_sigma, df=4)",
}
ADJUSTMENTS = {
    "snr_decile": "wc_pp ~ C(snr_decile) + log_J + log_n + log_sigma",
    "total_decile": "wc_pp ~ C(total_decile) + log_J + snr_pct + log_sigma",
    "arm_group": "wc_pp ~ C(arm_group) + snr_pct + log_n + log_sigma",
}
# Continuous adjusted profiles: a cubic B-spline in the focal level, same controls as ADJUSTMENTS.
CURVES = {
    "snr_pct": "wc_pp ~ bs(snr_pct, df=5) + log_J + log_n + log_sigma",
    "total_impressions": "wc_pp ~ bs(log_total, df=5) + log_J + snr_pct + log_sigma",
}
FLEXIBLE_ADJUSTMENTS = {
    "snr_decile": "wc_pp ~ C(snr_decile) + C(n_arms) + bs(log_n, df=4) + bs(log_sigma, df=4)",
    "total_decile": "wc_pp ~ C(total_decile) + C(n_arms) + bs(snr_pct, df=4) + bs(log_sigma, df=4)",
    "arm_group": "wc_pp ~ C(arm_group) + bs(snr_pct, df=4) + bs(log_n, df=4) + bs(log_sigma, df=4)",
}


def derive(run: Path) -> pd.DataFrame:
    """Join the saved corrections to the notebook's empirical SNR and controls."""
    p = pd.read_csv(run / "per_test_estimates.csv.gz")
    a = pd.read_csv(run / "arm_level_input.csv.gz", low_memory=False)
    ranked = a.sort_values(
        ["clickability_test_id", "click_rate", "source_row_id"],
        ascending=[True, False, True], kind="mergesort",
    )
    top = ranked.groupby("clickability_test_id").head(2).copy()
    top["rank"] = top.groupby("clickability_test_id").cumcount()
    v = top.pivot(index="clickability_test_id", columns="rank", values=["click_rate", "click_sd"])
    d = pd.DataFrame(index=v.index)
    d["sigma_top_two"] = np.sqrt((v["click_sd"][0] ** 2 + v["click_sd"][1] ** 2) / 2)
    d["snr_pct"] = 100 * (v["click_rate"][0] - v["click_rate"][1]) / d.sigma_top_two
    d = d.join(a.groupby("clickability_test_id").agg(date=("created_at", "min")))
    p = p.merge(d, on="clickability_test_id", validate="one_to_one")
    p["n_per_arm"] = p.total_impressions / p.n_arms
    p["sigma_all"] = np.sqrt(p.weighted_test_ctr * (1 - p.weighted_test_ctr))
    p["wc_pp"] = 100 * p.moon_bootstrap_estimated_bias
    p["gap_pp"] = 100 * p.winner_runner_up_gap
    p["ctr_pct"] = 100 * p.weighted_test_ctr
    for group, column in [("snr_decile", "snr_pct"), ("total_decile", "total_impressions")]:
        p[group] = pd.qcut(p[column], 10, labels=False) + 1
    p["arm_group"] = pd.cut(p.n_arms, [1, 2, 3, 4, 5, 7, np.inf], labels=ARM_GROUPS)
    dates = pd.to_datetime(p.date, utc=True, format="mixed")
    p["week"] = dates.dt.tz_localize(None).dt.to_period("W").astype(str)
    p["month"] = dates.dt.strftime("%Y-%m")
    # Proxy only: official archive problem flags are absent from this input snapshot.
    p["risk_period_proxy"] = dates.between("2013-06-25", "2014-01-11", inclusive="left")
    for name, column in [("J", "n_arms"), ("n", "n_per_arm"), ("sigma", "sigma_all")]:
        p[f"log_{name}"] = np.log(p[column])
    p["log_total"] = np.log(p.total_impressions)
    assert p.clickability_test_id.is_unique and p.arm_group.notna().all()
    assert np.isfinite(p[["snr_pct", "wc_pp", "log_J", "log_n", "log_sigma"]]).all().all()
    assert np.allclose(p.naive_selected_ctr - p.moon_bootstrap_corrected_ctr,
                       p.moon_bootstrap_estimated_bias, rtol=0, atol=1e-12)
    return p


def fit(formula: str, p: pd.DataFrame, covariance: str = "cluster"):
    """Fit OLS with calendar-week clustering or HC3 heteroskedasticity adjustment."""
    kwargs = {"groups": p.week} if covariance == "cluster" else {}
    return smf.ols(formula, p).fit(cov_type=covariance, cov_kwds=kwargs)


def grouped(p: pd.DataFrame, group: str) -> pd.DataFrame:
    return p.groupby(group, observed=True).agg(
        tests=("wc_pp", "size"), raw_wc_pp=("wc_pp", "mean"),
        snr_pct=("snr_pct", "mean"), arms=("n_arms", "mean"),
        total_impressions=("total_impressions", "mean"),
        impressions_per_arm=("n_per_arm", "mean"), ctr_pct=("ctr_pct", "mean"),
        sigma=("sigma_all", "mean"), observed_gap_pp=("gap_pp", "mean"),
        risk_period_proxy_share=("risk_period_proxy", "mean"),
    )


def adjusted(p: pd.DataFrame, group: str, formula: str) -> pd.DataFrame:
    """Standardize group predictions to the full-sample covariate distribution.

    Returns:
        DataFrame with adjusted means and pointwise normal 95% confidence intervals.
    """
    model = fit(formula, p)
    groups = ARM_GROUPS if group == "arm_group" else list(range(1, 11))
    original_x = np.asarray(model.model.exog)
    full_x_mean = original_x.mean(axis=0)
    rows = []
    for value in groups:
        hypothetical = p.copy()
        hypothetical[group] = value
        design = np.asarray(patsy.build_design_matrices([model.model.data.design_info], hypothetical)[0])
        x = design.mean(axis=0)
        mean = float(x @ model.params)
        se = float(np.sqrt(x @ model.cov_params() @ x))
        # Verify both average prediction and composition-adjustment identities.
        selected = p[group].eq(value).to_numpy()
        adjustment_x = full_x_mean - original_x[selected].mean(axis=0)
        group_columns = [i for i, name in enumerate(model.model.exog_names)
                         if name.startswith(f"C({group})")]
        adjustment_x[group_columns] = 0
        direct = p.loc[selected, "wc_pp"].mean() + adjustment_x @ model.params
        assert np.isclose(mean, np.mean(model.predict(hypothetical)), atol=1e-10)
        assert np.isclose(mean, direct, atol=1e-10)
        assert np.isfinite(se) and se >= 0
        rows.append({group: value, "adjusted_wc_pp": mean, "se_week_cluster": se,
                     "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se})
    result = pd.DataFrame(rows).set_index(group)
    assert np.isclose(np.average(result.adjusted_wc_pp, weights=grouped(p, group).tests),
                      p.wc_pp.mean(), atol=1e-10)
    return result


def curve(p: pd.DataFrame, column: str, formula: str, points: int = 200) -> pd.DataFrame:
    """Standardized predictions over a grid of the focal level (1st to 99th percentile).

    Returns:
        DataFrame indexed by the focal level with adjusted means and pointwise 95% intervals.
    """
    model = fit(formula, p)
    low, high = p[column].quantile([.01, .99])
    grid = np.geomspace(low, high, points) if column == "total_impressions" else np.linspace(low, high, points)
    rows = []
    for value in grid:
        hypothetical = p.copy()
        hypothetical[column] = value
        hypothetical["log_total"] = np.log(hypothetical.total_impressions)
        design = np.asarray(patsy.build_design_matrices([model.model.data.design_info], hypothetical)[0])
        x = design.mean(axis=0)
        mean = float(x @ model.params)
        se = float(np.sqrt(x @ model.cov_params() @ x))
        rows.append({column: value, "adjusted_wc_pp": mean, "se_week_cluster": se,
                     "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se})
    return pd.DataFrame(rows).set_index(column)


def regression_rows(model, sample: str, name: str, data: pd.DataFrame, covariance: str):
    intervals = model.conf_int()
    return [dict(sample=sample, model=name, formula=model.model.formula, covariance=covariance,
                 term=term, coefficient=model.params[term], se=model.bse[term], p=model.pvalues[term],
                 ci_low=intervals.loc[term, 0], ci_high=intervals.loc[term, 1],
                 tests=int(model.nobs), weeks=data.week.nunique(), r_squared=model.rsquared)
            for term in model.params.index]


def headline(p: pd.DataFrame, comparison: pd.DataFrame) -> dict:
    """Compute the bootstrap headline row and comparisons to the retained EB row."""
    assert set(comparison.clickability_test_id) == set(p.clickability_test_id)
    aligned = comparison.set_index("clickability_test_id").loc[p.clickability_test_id]
    assert np.allclose(aligned.naive_selected_ctr.to_numpy(), p.naive_selected_ctr, atol=1e-12, rtol=0)
    bias = p.moon_bootstrap_estimated_bias
    naive = p.naive_selected_ctr
    ratio = bias.mean() / naive.mean()
    ratio_se = (bias - ratio * naive).std(ddof=1) / (np.sqrt(len(p)) * naive.mean())
    return dict(
        tests=len(p), naive_ctr_pct=100 * naive.mean(),
        corrected_ctr_pct=100 * p.moon_bootstrap_corrected_ctr.mean(),
        corrected_ctr_se_pp=100 * p.moon_bootstrap_corrected_ctr.sem(),
        correction_pp=100 * bias.mean(), correction_se_pp=100 * bias.sem(),
        correction_over_mean_ctr_pct=100 * ratio, correction_over_mean_ctr_se_pct=100 * ratio_se,
        tests_correction_over_10pct_share=100 * (bias / naive > .1).mean(),
        correction_over_mean_gap_pct=100 * bias.mean() / p.winner_runner_up_gap.mean(),
        retained_eb_mean_correction_pp=100 * comparison.eb_normal_estimated_bias.mean(),
        retained_eb_over_moon_pct=100 * comparison.eb_normal_estimated_bias.mean() / bias.mean(),
        se_definition="Cross-test SEM; paired delta method for ratio of means, matching the original headline table.",
    )


def plot_panel(ax, table: pd.DataFrame, group: str, title: str | None = None):
    # Continuous levels for SNR and impressions; treatment-count groups for arms.
    x = np.arange(1, len(table) + 1) if group == "arm_group" else table.index.to_numpy()
    style = "o-" if group == "arm_group" else "-"
    ax.plot(x, table.adjusted_wc_pp, style, color="#27648f", linewidth=1.8, markersize=4.5)
    ax.fill_between(x, table.ci_low, table.ci_high, color="#27648f", alpha=.17, linewidth=0)
    xlabel = {"snr_pct": "Empirical SNR (%)",
              "total_impressions": "Total impressions per test (log scale)",
              "arm_group": "Number of candidate treatments"}[group]
    ax.set(xlabel=xlabel, ylabel="Estimated winner's curse (pp)")
    if group == "arm_group":
        ax.set(xticks=x, xticklabels=ARM_GROUPS)
    elif group == "total_impressions":
        ticks = [6_000, 10_000, 20_000, 40_000]
        ax.set_xscale("log")
        ax.set(xticks=ticks, xticklabels=[f"{t // 1000}k" for t in ticks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    if title:
        ax.set_title(title, fontsize=12, pad=10)
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=.045, y=.10)


def save_figures(tables: dict, out: Path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    slugs = {"snr_pct": "snr", "total_impressions": "impressions", "arm_group": "arms"}
    for group, table in tables.items():
        fig, ax = plt.subplots(figsize=(5.4, 3.65), layout="constrained")
        plot_panel(ax, table, group)
        for ext in ["pdf", "png"]:
            fig.savefig(out / f"upworthy-adjusted-{slugs[group]}.{ext}", dpi=200)
        plt.close(fig)
    fig = plt.figure(figsize=(10.8, 7.6), layout="constrained")
    grid = fig.add_gridspec(2, 4)
    axes = [fig.add_subplot(grid[0, :2]), fig.add_subplot(grid[0, 2:]), fig.add_subplot(grid[1, 1:3])]
    for ax, group, title in zip(axes, tables, ["(a) Empirical signal-to-noise ratio", "(b) Total impressions", "(c) Number of treatments"]):
        plot_panel(ax, tables[group], group, title)
    for ext in ["pdf", "png"]:
        fig.savefig(out / f"upworthy-adjusted-patterns.{ext}", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--comparison-run", type=Path, default=COMPARISON_RUN)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "docs/assets/upworthy/descriptive-patterns/gamma-0.5")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.result_dir / "run_manifest.json").read_text())
    gamma = source["parameters"]["moon_power"]
    p = derive(args.result_dir)
    assert len(p) == source["n_tests_run"] == 32389
    if args.result_dir.resolve() == DEFAULT_RUN.resolve():
        assert gamma == .5
    comparison = pd.read_csv(args.comparison_run / "per_test_estimates.csv.gz")
    stats = headline(p, comparison)
    stats.update(gamma=gamma, n_bootstraps=source["parameters"]["n_bootstraps"])
    (out / "headline.json").write_text(json.dumps(stats, indent=2) + "\n")
    p[["clickability_test_id", "wc_pp", "snr_pct", "n_arms", "total_impressions", "n_per_arm",
       "sigma_all", "log_J", "log_n", "log_sigma", "snr_decile", "total_decile", "arm_group",
       "date", "week", "month", "risk_period_proxy"]].to_csv(out / "analysis_input.csv.gz", index=False)
    samples = {"all": p, "exclude_top_snr_decile": p[p.snr_decile < 10],
               "outside_risk_period_proxy": p[~p.risk_period_proxy]}
    rows = []
    for sample, data in samples.items():
        names = list(REGRESSIONS) if sample == "all" else ["full"]
        for name in names:
            for covariance in ["cluster", "HC3"]:
                model = fit(REGRESSIONS[name], data, covariance)
                rows.extend(regression_rows(model, sample, name, data, covariance))
    regressions = pd.DataFrame(rows)
    regressions.to_csv(out / "regressions.csv", index=False)
    tables = {}
    for group, formula in ADJUSTMENTS.items():
        table = grouped(p, group).join(adjusted(p, group, formula))
        table.to_csv(out / f"{group}_adjusted.csv")
        if group == "arm_group":
            tables[group] = table
        flexible = grouped(p, group).join(adjusted(p, group, FLEXIBLE_ADJUSTMENTS[group]))
        flexible.to_csv(out / f"{group}_flexible.csv")
    for column, formula in CURVES.items():
        table = curve(p, column, formula)
        table.to_csv(out / f"{column}_curve.csv")
        tables = {column: table, **tables}
    tables = {k: tables[k] for k in ["snr_pct", "total_impressions", "arm_group"]}
    save_figures(tables, out)
    primary = regressions.query("sample == 'all' and covariance == 'cluster' and model in ['coauthor', 'add_sample_size', 'full']")
    lines = []
    for term, label in [("snr_pct", r"Empirical SNR ($S_i$)"), ("log_J", r"$\log J_i$"),
                        ("log_n", r"$\log n_i$"), ("log_sigma", r"$\log\hat\sigma_i$"), ("Intercept", "Constant")]:
        values = []
        for name in ["coauthor", "add_sample_size", "full"]:
            row = primary[(primary.model == name) & (primary.term == term)]
            values.append("---" if row.empty else rf"\begin{{tabular}}[c]{{@{{}}c@{{}}}}{row.iloc[0].coefficient:.4f}\\ ({row.iloc[0].se:.4f})\end{{tabular}}")
        lines.append(label + " & " + " & ".join(values) + r" \\")
    for name, label in [("tests", "Observations"), ("weeks", "Calendar-week clusters"), ("r_squared", r"$R^2$")]:
        values = [primary[primary.model == m].iloc[0][name] for m in ["coauthor", "add_sample_size", "full"]]
        formatted = [f"{v:.3f}" if name == "r_squared" else f"{int(v):,}" for v in values]
        lines.append(label + " & " + " & ".join(formatted) + r" \\")
    (out / "regression-table-rows.tex").write_text("\n".join(lines) + "\n")
    inputs = [args.result_dir / f for f in ["per_test_estimates.csv.gz", "arm_level_input.csv.gz", "run_manifest.json"]]
    inputs.append(args.comparison_run / "per_test_estimates.csv.gz")
    manifest = dict(tests=len(p), weeks=p.week.nunique(), gamma=gamma, n_bootstraps=source["parameters"]["n_bootstraps"],
        result_dir=str(args.result_dir.resolve()), retained_eb_source=str(args.comparison_run.resolve()),
        input_sha256={str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        packages={"numpy": np.__version__, "pandas": pd.__version__, "statsmodels": statsmodels.__version__, "matplotlib": matplotlib.__version__},
        regression_formulas=REGRESSIONS, adjusted_formulas=ADJUSTMENTS, curve_formulas=CURVES, flexible_adjusted_formulas=FLEXIBLE_ADJUSTMENTS,
        uncertainty="Pointwise normal 95% intervals using finite-sample-adjusted calendar-week cluster covariance; separate HC3 regression checks. No estimator-bias guarantee.",
        sigma_definition="sqrt(pooled test CTR * (1 - pooled test CTR)); SNR uses the notebook top-two outcome SD definition.",
        risk_proxy="Earliest package created_at in [2013-06-25, 2014-01-11) UTC. Official problem flags are absent from the saved input.",
        risk_period_proxy_tests=int(p.risk_period_proxy.sum()),
        external_source="https://www.nature.com/articles/s41597-021-00934-7",
        interpretation="OLS standardization to the full-sample covariate distribution, with potential extrapolation; descriptive, not causal.")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    print(primary[["model", "term", "coefficient", "se", "r_squared"]].to_string(index=False))
    print(regressions.query("covariance == 'cluster' and term == 'snr_pct'")[["sample", "model", "tests", "coefficient", "se", "p"]].to_string(index=False))
    for group, table in tables.items():
        values = table.adjusted_wc_pp if group == "arm_group" else table.adjusted_wc_pp.iloc[[0, -1]]
        print(group, values.round(6).to_list())
    print(f"Validated standardization identities and saved outputs to {out}")


if __name__ == "__main__":
    main()
