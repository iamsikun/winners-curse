# Paper reproduction map

**Current A/B baseline (2026-09-17):** the completed rerun is now authoritative.
See [the pinned run manifest](ab-results-current.json) and
[the numerical paper cross-check](ab-paper-crosscheck/README.md).
Historical A/B run directories have been deleted; the three A/B analysis notebooks
now reference the designated batch. Older-run discussion below is retained only
as provenance, not as instructions to load deleted files. Targeting and Upworthy
provenance is unchanged.

Audited 2026-09-17 against the local Overleaf-linked checkout at
`/Users/iamsikun/research/winners-curse-paper/main.tex`, commit `2e1d119`
(“Update on Overleaf.”, 2026-09-17), and the current working tree of this repository.
This is an inventory and provenance audit, updated with the requested config
changes. The initial audit used reduced runs; the user subsequently completed
the full A/B batch, which is now pinned and cross-checked in the linked report.
Targeting changes have only been validated with reduced runs.

The manuscript contains **21 tables and 14 figures**, including the appendix.
The paper simulation suite is **8 A/B YAMLs + 6 targeting YAMLs**.
A seventh targeting config adds the requested model-by-bootstrap-power sensitivity study.
It also needs the Upworthy empirical script and notebook-only figure generation.
Table 1 is a literature compilation, not a simulation. Mean, median, and MAE
tables generally reuse the same simulation arrays; they are not separate runs.

Appendix numbers below follow the current TeX: tables/figures are reset at the
start of the appendix, but table numbering is not reset between appendix
sections. Thus the Bayesian table is D.1 and the first binary-outcome table is
E.2. TeX labels are included to make the mapping stable if numbering changes.

**A/B simulation checklist**

All eight configurations use [ab_test_experiment.py](../scripts/ab_test_experiment.py).
Run from the repository root with
`uv run python scripts/ab_test_experiment.py --config configs/<filename>.yaml`.
Unless otherwise noted, the design has 2,500 observations per arm, 1,000
repetitions, and 100 bootstrap draws. The paper's default subsample exponent is
`power: 0.6`.

| Config | Experiment | Paper outputs | Analysis location / qualification |
|---|---|---|---|
| [ab_test_snr.yaml](../configs/ab_test_snr.yaml) | Normal outcomes; effects `(1, 1.005)`, `(1, 1.01)`, `(1, 1.02)` | Tables 2, 3, E.12: mean WC, MAE, median WC (`tab: ab test estimates`, `tab: ab test mae`, `tab: ab test estimates median (appendix)`) | `ab_test.ipynb`, Comparative Statics → SNR. One run supplies all three tables. Printed standard-bootstrap values do not all match the saved arrays; see audit notes below. |
| [ab_test_noise_vars.yaml](../configs/ab_test_noise_vars.yaml) | Normal, uniform, Laplace, logistic noise; each has variance 1; effects `(1, 1.01)` | Tables 4, E.13: mean and median WC across distributions (`tab: ab test noise dstn`, `tab: ab test noise dstn median (appendix)`) | `ab_test.ipynb`, Noise distribution. |
| [ab_test_bayesian.yaml](../configs/ab_test_bayesian.yaml) | Positive versus negative effects: `(1, 1.01)` and `(-1.01, -1)`; fixed normal prior | Table D.1 (`tab: ab test bayes`) | `bayesian_methods.ipynb`, Positive vs. Negative Effects. Config now uses the scaled moon algorithm with `power: 0.6`. The historical April results still use 0.8, and the notebook names a missing December result folder. |
| [ab_test_bernoulli.yaml](../configs/ab_test_bernoulli.yaml) | Baseline probabilities 0.1, 0.2, 0.3 crossed with lifts 0.001, 0.005 | Table E.2 (`tab: ab test bernoulli delta tau`) | `ab_test.ipynb`, Bernoulli Response. |
| [ab_test_snr_additional.yaml](../configs/ab_test_snr_additional.yaml) | SNR grid plus conditional SI, plugin, 90/10 splitting, 10-fold CV, jackknife | Table E.5 (`tab: ab test estimates (appendix)`) | `ab_test.ipynb`, All estimators. Only the five additional estimators and automatic no-correction output are needed for this table. The config's 0.8 moon row is not displayed in E.5. |
| [ab_test_n_treatments.yaml](../configs/ab_test_n_treatments.yaml) | 2, 4, 6, 8, 10 equal-effect arms at fixed `N=2500` | Table E.6 (`tab: ab test multiple actions`) | `ab_test.ipynb`, Number of Treatments. **Current config and designated rerun have 1,000 repetitions, matching the paper caption.** This fixed-N run does not produce Figure 2. |
| [ab_test_imbalanced_snr.yaml](../configs/ab_test_imbalanced_snr.yaml) | Nine allocations `(500,4500)` through `(4500,500)` crossed with three SNRs, with all corrections and no correction | Tables E.7 and E.8 (`tab: ab test wc imbalanced`, `tab: ab test estimates imbalanced`) | Results use `(allocation_tuple, tau_tuple)` keys. Use every allocation for E.7 and select `(4500,500)` for E.8. Replaces the deleted `ab_test_imbalanced.yaml`. |
| [ab_test_moon_power.yaml](../configs/ab_test_moon_power.yaml) | Three SNRs crossed with bootstrap exponents 0.1, 0.2, …, 0.9, 0.95 | Table E.9 (`tab: gamma sweep`) | `ab_test.ipynb`, m-out-of-n Power Sweep. This is the hyperparameter table; it is not the smooth-value-function illustration in Figure D.1. |

For Table E.5 alone, avoid repeating the five main estimators:

```bash
uv run python scripts/ab_test_experiment.py \
  --config configs/ab_test_snr_additional.yaml \
  --estimators conditional_si plugin sample_splitting9010 cross_validation jackknife
```

This creates an `_estsubset` directory; no-correction arrays are still included.

**Personalized-targeting simulation checklist**

All seven configurations use [targeting_experiment.py](../scripts/targeting_experiment.py).
Run with
`uv run python scripts/targeting_experiment.py --config configs/<filename>.yaml`.
The usual design is 2,500 training observations per arm, 10,000 evaluation
customers, 1,000 repetitions, 100 bootstrap draws, and `power: 0.6`.

| Config | Experiment | Paper outputs | Analysis location / qualification |
|---|---|---|---|
| [targeting_correct_snr.yaml](../configs/targeting_correct_snr.yaml) | Known functional form `g(x)=x**2+x`, effects `(1,1.01)` | Correct-functional-form column of Tables 5, 6, E.14 (`tab: targeting estimates`, `tab: targeting mae`, `tab: targeting estimates median`) | `targeting.ipynb`, Causal Forest vs. Correct Functional Form. Config now uses `targ_sample_size: 10000` and 1,000 repetitions. Existing paper results combine runs with evaluation sizes 100 and 10,000. |
| [targeting_depth_compare.yaml](../configs/targeting_depth_compare.yaml) | Forest depths 2, 5, 10 at fixed effects `(1,1.01)` | Forest columns of Tables 5, 6, E.14 | Same model-comparison section; combine with the preceding run. This produces performance tables, not the fitted-curve illustration in Figure 3. |
| [targeting_forest_depth.yaml](../configs/targeting_forest_depth.yaml) | Forest depths 5, 10, 15 crossed with sample size | Figure 4, `targeting_depth_sample_size.pdf` (`fig: targeting depth sample size`) | `targeting.ipynb`, Max Depth; `plot_targeting_wc_depth_sample_size`. **Config has 500 repetitions versus the caption's 1,000.** Notebook's saved result directory is missing. |
| [targeting_cf_bernoulli.yaml](../configs/targeting_cf_bernoulli.yaml) | Binary outcomes; six probability pairs; depth 5; uniform features | Tables E.3, E.4: mean WC and MAE (`tab: targeting bernoulli delta tau`, `tab: targeting bernoulli mae`) | `targeting.ipynb`, Bernoulli Response. Root-level `tau_list` is supported by the loader. The historical full-run folder is named `targeting_forest_bernoulli`, not `targeting_cf_bernoulli`. |
| [targeting_forest_snr.yaml](../configs/targeting_forest_snr.yaml) | SNR 0.5%, 1%, 2% at depth 5 | Table E.10 (`tab: targeting estimates (old)`) | `targeting.ipynb`, SNR → Causal Forest. Despite `(old)` in the TeX label, this table is active in the paper. |
| [targeting_forest_functional_form.yaml](../configs/targeting_forest_functional_form.yaml) | `g(x)=x`, `x**2+x`, `abs(x)` at depth 5 | Table E.11 (`tab: targeting func form`) | `targeting.ipynb`, Functional Form. |

| [targeting_forest_moon_power.yaml](../configs/targeting_forest_moon_power.yaml) | Scaled moon powers 0.4, 0.5, …, 0.9, 0.95; known form and forests of depth 2, 5, 10; three SNRs | New sensitivity analysis; no existing paper table | Twelve `(model_name, tau_tuple)` combinations, each with no correction, standard bootstrap, and seven moon estimators. Uses 10,000 evaluation customers and 1,000 repetitions. |

The forest sample-size YAML currently contains
`[2500,5000,10000,15000,20000,30000,40000,50000,100000,500000,1000000]`.
The Figure 4 caption lists only `[2500,10000,50000,100000,500000,1000000]`.
The plotting function plots **every sample-size result**; its
`viz_sample_sizes` argument changes axis ticks only, not the plotted data.
Choose the intended grid explicitly when reconciling the figure and caption.

**Figures that require notebook code rather than an existing YAML**

Cell numbers here are zero-based JSON cell positions, not execution counters.
They help locate code in the audited snapshot; section names and output
filenames are more stable references.

| Paper figure | Source | Parameters / reproduction requirements |
|---|---|---|
| Figure 1: `rct_delta_tau_sample_size.pdf` (`fig: ab test sample size`) | [ab_test.ipynb](../notebooks/ab_test.ipynb), final Sample Size → Signal-to-Noise Ratio, cells 103–106 | Three lifts 0.005, 0.01, 0.02; `N=[2500,5000,10000,50000,100000,250000,500000]`; no correction; notebook specifies **5,000 repetitions**. **No dedicated YAML exists.** The README's `ab_test_sample_size.yaml` is absent. A new config could use the A/B script, the SNR tau grid, this `sample_size_list`, and `estimators_dict: {}`. |
| Figure 2: `rct_n_treatments_sample_size.pdf` (`fig: ab test multiple actions`) | [ab_test.ipynb](../notebooks/ab_test.ipynb), final Sample Size → Number of Treatments, cells 108–112 | Arms 2,4,6,8,10, all with effect 1; `N=[2500,5000,10000,25000,50000,100000,250000,500000]`; no correction. **No dedicated YAML exists.** Extend the multiple-arms config with this sample-size grid and an empty estimator dictionary. Notebook uses 500 repetitions; paper says 1,000. |
| Figure 3: `cf_depth_compare.pdf` (`fig: cf depth compare`) | [targeting.ipynb](../notebooks/targeting.ipynb), DGP/Estimation, especially cell 13 | One simulated dataset; fitted HTE curves at depths 2,5,10. **No YAML for this illustration.** The current cell uses 50 trees; caption says 100. Reconcile the tree-count setting before regenerating. |
| Figure D.1: `moon_vs_standard_bootstrap.pdf` (`fig: $m$-out-of-$n$ vs standard bootstrap`) | [subsampling.ipynb](../notebooks/subsampling.ipynb), cell 2 | `N=2500`, `m=int(N**0.6)`, sigma 1, tau1 1, tau2 0.7–1.3, seed 42; 10,000 simulated datasets/grid point and 1,000 bootstrap draws. Standalone parametric illustration; no YAML. |
| Figure D.2: `sample_splitting_value_loss.pdf` (`fig: sample splitting optimality gap`) | [sample_splitting.ipynb](../notebooks/sample_splitting.ipynb), analytical curve cells 12–13 | Normal-CDF calculation for `N=[500,2500,10000]`, sigma 1, 500 lift values from 0.001 to 0.1. No Monte Carlo run or YAML needed for this figure. **Notebook is invalid JSON at line 30**, containing `{winners_curse.ab_test`; it also imports obsolete `winners_curse.rct`. |
| Figures D.3, D.4, D.5, D.6: `regular_truncated_normal_density.pdf`, `irregular_truncated_normal_density.pdf`, `zero_function_of_truncated_normal.pdf`, `corrected_winner_vs_diff.pdf` | [andrews_et_al_2024_qje.ipynb](../notebooks/andrews_et_al_2024_qje.ipynb), Toy Example | Four numerical/analytical selective-inference illustrations; no YAML. Two-arm example uses means `(1,1.01)`, sigma 1, `N=100`, median quantile 0.5; density examples use seeds 3 and 4. Initial imports include the absent module `winners_curse.treatment_selection_single_segment`; isolate/update the toy-example setup before executing. |

The two A/B sample-size sections also use the obsolete `rct` name and old
optimizer/result-key conventions. Simply running those notebook cells against
the current package will not work. They currently save `.jpg` files under
`figures/`, while the manuscript includes `.pdf` files. Update the execution and
export steps, or implement those two grids through the current A/B script.
Do not run all cells in the exploratory notebooks as a paper reproduction job.

All ten non-Upworthy PDFs actually included by the manuscript are byte-identical
to the corresponding files in this repository's `figures/` directory. That
establishes the existing artifact mapping, but does not establish that the
current notebook source can regenerate those historical PDFs unchanged.

**Upworthy: one script run, one analysis notebook, no YAML**

[upworthy_empirical_ab.py](../scripts/upworthy_empirical_ab.py) supplies Table 7
(`tab:upworthy`) and the estimates used by Figures 5–8. The analysis/export step
is [upworthy.ipynb](../notebooks/upworthy.ipynb).

The saved run manifest records this command, with the following defaults made
explicit here:

```bash
uv run python scripts/upworthy_empirical_ab.py \
  --input dataset/upworthy/derived/deployed-arm-audit.csv \
  --output results/empirical_study/upworthy \
  --n-bootstraps 1000 --n-sample-splits 1000 \
  --moon-power 0.6 --estimation-split 0.5 \
  --seed 20260628 --max-jobs 24 \
  --min-arms 2 --min-impressions 1000 \
  --hybrid-n-simulations 10000
```

The default estimator list includes standard bootstrap, moon bootstrap, sample
splitting, empirical Bayes, and hybrid SI. Do not use `--limit-tests` for the
paper. This command assumes the existing derived input CSV; it does not recreate
the raw-archive preprocessing.

The notebook currently uses
`results/empirical_study/upworthy/upworthy_empirical_ab_20260721_103812/`.
Its manifest and the current input CSV have matching SHA-256 hashes. The run has
32,389 tests; hybrid SI has 32,254 finite results, consistent with the paper's 135
failures. The manuscript says 92 tests were removed, whereas the run manifest
records 98 exclusions from 32,487 inputs; reconcile that descriptive count.

| Paper output | Notebook section / exported PDF |
|---|---|
| Table 7 | Headline Result: corrected CTR, mean estimated bias, ratio of mean bias to mean naive CTR, and their standard errors. Use the notebook's ratio-of-means calculation, not the summary CSV's mean of per-test ratios. |
| Figure 5 (`fig: upworthy impressions`) | Pre-Cutoff Package-Arm Impressions → `impressions-per-arm-histogram.pdf`. Uses the input data before applying the 1,000-impression cutoff, with the notebook's pre-cutoff eligibility filter. |
| Figure 6 (`fig: upworthy top two distributions`) | `winner-runner-up-gap-histogram.pdf` + `snr-distribution-histogram.pdf`. |
| Figure 7 (`fig: upworthy wc snr impr`) | `bias-by-snr.pdf` + `bias-by-total-impressions.pdf`. |
| Figure 8 (`fig: upworthy arm count`) | `bias-by-arm-count.pdf`. |

For a fresh run, set the notebook's `RESULT_DIR` to the new directory. It writes
PDFs/PNGs to `docs/assets/upworthy/`, `paper/journal/figures/upworthy/`, and
`docs/Winner_s_Curse_in_Personalized_Targeting/figures/upworthy/`. It does **not**
write to the sibling Overleaf checkout. Exported PDFs must be transferred to
that checkout's `figures/` directory separately. The six current local Upworthy
PDFs are not byte-identical to the Overleaf copies; PDF metadata can also cause
this, so file hashes alone do not establish a visible difference.

**Designated result folders to inspect**

A/B entries below are the designated reruns and have been compared numerically
against all 12 A/B simulation tables. Other entries retain the earlier audit status. All paths below are relative
to `results/`. A timestamp identifies a run directory, not necessarily the last
time its arrays or metadata were updated.

| Experiment | Full analysis file |
|---|---|
| A/B SNR | `ab_test_snr_20260917_121311/results.pkl` |
| A/B noise distributions | `ab_test_noise_vars_20260917_121208/results.pkl` |
| A/B additional estimators | `ab_test_snr_additional_20260917_121409/results.pkl` |
| A/B Bayesian | `ab_test_bayesian_20260917_114819/results.pkl` |
| A/B binary | `ab_test_bernoulli_20260917_114903/results.pkl` |
| A/B multiple arms | `ab_test_n_treatments_20260917_122850/results.pkl` |
| A/B imbalanced corrections | `ab_test_imbalanced_snr_20260917_115117/results.pkl` |
| A/B exponent sweep | `ab_test_moon_power_20260917_115819/results.pkl` |
| A/B allocation sweep | Shared combined sweep: `ab_test_imbalanced_snr_20260917_115117/results.pkl` |
| Targeting known form | `targeting_correct_snr_20251213_120521/results_with_new_moon.pkl` |
| Targeting depth comparison | `targeting_depth_compare_20251212_000908/results_with_new_moon.pkl` |
| Targeting SNR | `targeting_forest_snr_20251214_020344/results_with_new_moon.pkl` |
| Targeting functional forms | `targeting_forest_functional_form_20251213_135015/results_with_new_moon.pkl` |
| Targeting binary | `targeting_forest_bernoulli_20251212_105431/results_with_new_moon.pkl` |
| Targeting depth × sample size | Missing notebook target: `targeting_forest_depth_20251122_112306/results.pkl` |
| Upworthy | `empirical_study/upworthy/upworthy_empirical_ab_20260721_103812/` |

The five targeting `results_with_new_moon.pkl` files contain moon WC arrays
identical to the following July estimator-only reruns, respectively:

| Full experiment | Matching moon-only run |
|---|---|
| Known functional form | `targeting_correct_snr_estsubset_20260721_113347` |
| Depth comparison | `targeting_depth_compare_estsubset_20260721_113419` |
| SNR | `targeting_forest_snr_estsubset_20260722_110930` |
| Functional forms | `targeting_forest_functional_form_estsubset_20260722_135815` |
| Binary | `targeting_cf_bernoulli_estsubset_20260721_155332` |

An `_estsubset` file is a partial rerun, not a replacement for the whole
experiment. Some such directories have configs/logs but no result pickle.
[merge_results.py](../src/winners_curse/merge_results.py) explains the intended
merge: copy estimator arrays into a full result file while retaining its
no-correction and other estimator arrays. New full runs do not require this merge
and write `results.pkl`; update notebook filenames accordingly.

**Reconciliation needed before claiming exact reproduction**

1. **Known-functional-form targeting mixes evaluation sizes.** The original
   full run has 10,000 evaluation customers; the July moon-only run has 100.
   The merged file reproduces the paper's moon WC mean of -29.74% and MAE of
   0.0098, but its rows do not all come from the manuscript's stated 10,000-customer
   design. The updated YAML now uses 10,000 for every row, following the stated design;
   rerunning it will change the historical moon row.
2. **Multiple-arms repetition count resolved.** The designated run now has
   1,000 repetitions. Its first 500 draws exactly match the deleted prior run;
   means and SDs remain close. See [the comparison](ab-paper-crosscheck/n-treatments-rerun.md).
   The printed Table E.6 values still reflect 500 repetitions and need updating.
   Figure 2's notebook and Figure 4's YAML still specify 500 versus 1,000 in their captions.
3. **Historical Bayesian results use the old gamma.** The April pickle uses
   0.8, while Table D.1 and the updated YAML use 0.6. For example, saved normalized
   moon bias at `(1,1.01)` is 12.65%, versus 3.77% in the paper.
4. **Some standard-bootstrap values differ from stored results.** At SNR 1%,
   Table 2 reports 37.49% mean WC and 190.41% SD; its notebook's saved arrays
   give 38.27% and 190.90%. The same stored values occur in the full-SNR,
   Bayesian, and exponent-sweep files. This is larger than rounding error.
   Identify the original run or update the manuscript from a designated run.
5. **Historical figure code has drifted.** Missing configs/results, obsolete
   imports, the malformed sample-splitting notebook, and Figure 3's changed
   parameters prevent an unmodified “Run All” reproduction.

For table generation, use [analysis.py](../src/winners_curse/analysis.py):
`generate_snr_sum_stats_latex_table`, `generate_noise_dist_sum_stats_latex_table`,
`generate_n_treatments_sum_stats_latex_table`,
`generate_bernoulli_sum_stats_latex_table`,
`generate_targeting_model_comparison_latex_table`, and
`generate_functional_form_latex_table`.

- Mean WC tables: `metric='wc'`, `stats='mean'`; normalize by the lift where the
  paper reports percentages and include the SD.
- Median tables E.12–E.14: same arrays with `stats='median'`,
  `include_std=False`, and two decimal places.
- MAE Tables 3, 6, E.4: same arrays with `metric='mae'`, `normalize=False`,
  `stats='mean'`, `include_std=False`, and four decimal places.
- Multiple equal-effect arms: `normalize=False`, since the lift is zero.

The notebooks contain additional MAE tables and exploratory plots that are not
in this manuscript. Their current formatting selections also vary: for example,
the main SNR cell requests zero decimals and the targeting model-comparison cell
currently requests medians. Select the appropriate options for each paper table.

The experiment scripts save raw arrays plus `config.yaml`, `config.json`, and
`experiment.log` under `results/<config-stem>_<timestamp>/`; they do not generate
or insert paper tables automatically. The Overleaf tables are pasted directly
into `main.tex`, with no table-file `\input` pipeline.

**Not required for the current manuscript**

- `ab_test_cond_bootstrap.yaml`: exploratory conditional-bootstrap comparison.
  Table E.5 uses **conditional selective inference**, which is already in
  `ab_test_snr_additional.yaml`; these are different estimators.
- `ab_test_moon_scaling.yaml` / `ab_test_moon.ipynb`: sample-size scaling
  exploration; not the active exponent-sweep table or Figure D.1. The config
  now contains scaled estimators only; historical files may still contain old rows.
- `structural.ipynb` / MNL assortment, `hillstrom.ipynb`, `selection_rule.ipynb`,
  and the other exploratory notebooks have no active result table/figure in the
  audited Overleaf `main.tex`.
- `subsampling_size_tradeoff.pdf` and `mnl_assort_capacity.pdf` are present among
  the Overleaf assets but are not included by the current manuscript.
- Table 1 and Appendix A are literature-derived parameters. They have no
  corresponding experiment script/YAML.
- `uv run pytest` verifies software behavior; it does not generate the paper's
  empirical or simulation results.

To keep subsequent runs identifiable, designate one output directory per row in
the checklist, retain its config and code revision, and record which tables and
figures consume it. Avoid selecting files solely by the newest timestamp or by
whether their directory name resembles a figure title.

**Batch runners (updated 2026-09-17)**

```bash
bash scripts/run_all_ab_tests.sh --dry-run
bash scripts/run_all_targeting_tests.sh --dry-run
bash scripts/run_all_ab_tests.sh
bash scripts/run_all_targeting_tests.sh
```

The scripts work from any directory, run configurations sequentially, and stop
on the first failure. Terminal output includes completed-config counts and
percentages, per-config elapsed times, live combination/repetition progress, and
a final summary (including failures/interruption). Percentages count configs,
not runtime; Python output is unbuffered. They run every matching simulation YAML, including the
conditional-bootstrap and scaled sample-size exploration configs. The A/B batch
currently has 10 configs; the targeting batch has 7. Upworthy and notebook-only
figures remain separate. The user completed the A/B batch on 2026-09-17; it is now the designated baseline.

For the combined imbalanced run, retain the complete result dict for Table E.7.
To use the existing SNR table helper for Table E.8, extract the fixed allocation:

```python
imbalanced_snr = {
    tau: measures
    for (allocation, tau), measures in results.items()
    if allocation == (4500, 500)
}
```

For the new targeting power sweep, select a named model before passing results
to the SNR analysis helpers:

```python
model_snr = {
    tau: measures
    for (model_name, tau), measures in results.items()
    if model_name == "causal_forest_depth_5"
}
```

Other model names are `known_functional_form`, `causal_forest_depth_2`, and
`causal_forest_depth_10`. The named models and their complete parameters are
preserved in each run's config metadata; model-by-SNR checkpoint/resume is
supported by `targeting_experiment.py --resume <prior_output_dir>`.
