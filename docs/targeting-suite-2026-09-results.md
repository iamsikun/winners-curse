# Targeting suite rerun, September 2026

Rerun of the targeting simulations after two changes to the runner: the projection
critical value in hybrid selective inference is now evaluated in closed form, and the
suite runs at 32 workers with memory-tiered job caps (see
`configs/targeting_forest_depth.yaml`). Neither change touches the estimators being
compared; the selective-inference change moves its own `val_est` by ~2e-4 (it replaces
a Monte Carlo estimate of a constant with the constant).

Batch log: `results/batch_targeting_20260919_214619.log`, started 2026-09-19 21:46.

**Status: partial.** Three of six configs are pinned below. `targeting_forest_depth`,
`targeting_forest_functional_form` and `targeting_forest_snr` were still running when
this was written and will be added when they finish.

All numbers are the mean winner's curse (estimated policy value minus true policy
value) over repeats, with its standard error, and the mean absolute winner's curse.
Positive mean WC is overstatement of the selected policy's value.

## targeting_correct_snr (known functional form, n = 2500, 1000 repeats)

Run: `results/targeting_correct_snr_20260920_032752`, 1 combination, 129 s.

tau = (1, 1.01):

| estimator | mean WC | SE | MAE |
|---|---|---|---|
| No correction | 0.0090 | 0.0003 | 0.0112 |
| Standard bootstrap | 0.0011 | 0.0004 | 0.0098 |
| m-out-of-n bootstrap | -0.0032 | 0.0003 | 0.0091 |
| Sample splitting | -0.0010 | 0.0006 | 0.0143 |
| Empirical Bayes | 0.0082 | 0.0003 | 0.0107 |
| Hybrid SI | -0.0064 | 0.0006 | 0.0153 |

The standard bootstrap removes 88% of the naive bias and is the only correction whose
mean WC is within two SE of zero. Empirical Bayes barely moves (0.0090 -> 0.0082).

## targeting_depth_compare (causal forest, depth sweep, n = 2500, 1000 repeats)

Run: `results/targeting_depth_compare_20260920_033004`, 3 combinations, 10,482 s.

| depth | estimator | mean WC | SE | MAE |
|---|---|---|---|---|
| 2 | No correction | 0.0230 | 0.0009 | 0.0307 |
| 2 | Standard bootstrap | 0.0089 | 0.0010 | 0.0273 |
| 2 | m-out-of-n bootstrap | 0.0013 | 0.0010 | 0.0248 |
| 2 | Sample splitting | -0.0029 | 0.0014 | 0.0371 |
| 2 | Empirical Bayes | -0.0233 | 0.0010 | 0.0319 |
| 2 | Hybrid SI | -0.1908 | 0.0024 | 0.1909 |
| 5 | No correction | 0.0342 | 0.0008 | 0.0361 |
| 5 | Standard bootstrap | 0.0091 | 0.0009 | 0.0235 |
| 5 | m-out-of-n bootstrap | 0.0057 | 0.0008 | 0.0215 |
| 5 | Sample splitting | -0.0034 | 0.0013 | 0.0324 |
| 5 | Empirical Bayes | 0.0121 | 0.0008 | 0.0226 |
| 5 | Hybrid SI | -0.1427 | 0.0016 | 0.1429 |
| 10 | No correction | 0.0823 | 0.0008 | 0.0823 |
| 10 | Standard bootstrap | 0.0209 | 0.0009 | 0.0280 |
| 10 | m-out-of-n bootstrap | 0.0539 | 0.0008 | 0.0543 |
| 10 | Sample splitting | -0.0033 | 0.0013 | 0.0318 |
| 10 | Empirical Bayes | 0.0564 | 0.0008 | 0.0567 |
| 10 | Hybrid SI | -0.1556 | 0.0011 | 0.1556 |

The naive bias grows with depth (0.023 -> 0.082) and the standard bootstrap is the only
correction that keeps pace, holding MAE near 0.023-0.028 at every depth. m-out-of-n and
empirical Bayes both degrade sharply at depth 10. Hybrid SI overcorrects by an order of
magnitude more than the bias it is removing, at every depth.

## targeting_cf_bernoulli (causal forest, binary outcome, n = 2500, 1000 repeats)

Run: `results/targeting_cf_bernoulli_20260919_214621`, 6 combinations, 20,491 s.

| tau | estimator | mean WC | SE | MAE |
|---|---|---|---|---|
| (0.1, 0.101) | No correction | 0.0091 | 0.0001 | 0.0092 |
| (0.1, 0.101) | Standard bootstrap | 0.0018 | 0.0002 | 0.0042 |
| (0.1, 0.101) | m-out-of-n bootstrap | 0.0032 | 0.0001 | 0.0045 |
| (0.1, 0.101) | Sample splitting | -0.0000 | 0.0002 | 0.0052 |
| (0.1, 0.101) | Empirical Bayes | 0.0028 | 0.0001 | 0.0043 |
| (0.1, 0.101) | Hybrid SI | -0.0085 | 0.0002 | 0.0091 |
| (0.2, 0.201) | No correction | 0.0121 | 0.0002 | 0.0122 |
| (0.2, 0.201) | Standard bootstrap | 0.0018 | 0.0002 | 0.0051 |
| (0.2, 0.201) | m-out-of-n bootstrap | 0.0044 | 0.0002 | 0.0059 |
| (0.2, 0.201) | Sample splitting | -0.0005 | 0.0003 | 0.0068 |
| (0.2, 0.201) | Empirical Bayes | 0.0064 | 0.0002 | 0.0071 |
| (0.2, 0.201) | Hybrid SI | -0.0122 | 0.0003 | 0.0129 |
| (0.3, 0.301) | No correction | 0.0139 | 0.0002 | 0.0140 |
| (0.3, 0.301) | Standard bootstrap | 0.0023 | 0.0002 | 0.0057 |
| (0.3, 0.301) | m-out-of-n bootstrap | 0.0053 | 0.0002 | 0.0067 |
| (0.3, 0.301) | Sample splitting | 0.0003 | 0.0003 | 0.0076 |
| (0.3, 0.301) | Empirical Bayes | 0.0098 | 0.0002 | 0.0101 |
| (0.3, 0.301) | Hybrid SI | -0.0145 | 0.0003 | 0.0151 |
| (0.1, 0.105) | No correction | 0.0089 | 0.0001 | 0.0090 |
| (0.1, 0.105) | Standard bootstrap | 0.0016 | 0.0002 | 0.0042 |
| (0.1, 0.105) | m-out-of-n bootstrap | 0.0031 | 0.0001 | 0.0044 |
| (0.1, 0.105) | Sample splitting | -0.0001 | 0.0002 | 0.0053 |
| (0.1, 0.105) | Empirical Bayes | 0.0027 | 0.0001 | 0.0043 |
| (0.1, 0.105) | Hybrid SI | -0.0087 | 0.0002 | 0.0093 |
| (0.2, 0.205) | No correction | 0.0120 | 0.0002 | 0.0121 |
| (0.2, 0.205) | Standard bootstrap | 0.0018 | 0.0002 | 0.0052 |
| (0.2, 0.205) | m-out-of-n bootstrap | 0.0043 | 0.0002 | 0.0057 |
| (0.2, 0.205) | Sample splitting | -0.0005 | 0.0003 | 0.0068 |
| (0.2, 0.205) | Empirical Bayes | 0.0064 | 0.0002 | 0.0071 |
| (0.2, 0.205) | Hybrid SI | -0.0122 | 0.0003 | 0.0129 |
| (0.3, 0.305) | No correction | 0.0138 | 0.0002 | 0.0139 |
| (0.3, 0.305) | Standard bootstrap | 0.0021 | 0.0002 | 0.0056 |
| (0.3, 0.305) | m-out-of-n bootstrap | 0.0051 | 0.0002 | 0.0066 |
| (0.3, 0.305) | Sample splitting | 0.0000 | 0.0003 | 0.0078 |
| (0.3, 0.305) | Empirical Bayes | 0.0097 | 0.0002 | 0.0101 |
| (0.3, 0.305) | Hybrid SI | -0.0144 | 0.0003 | 0.0151 |

The naive bias scales with the base rate (0.009 at 0.1, 0.014 at 0.3) and is
insensitive to the gap between arms: the (0.1, 0.101) and (0.1, 0.105) rows agree to
within one SE for every estimator. The standard bootstrap gives the lowest MAE in every
cell; hybrid SI overcorrects to roughly the negative of the naive bias.

## Selective inference convergence

`fsolve` non-convergence in hybrid SI, as a share of customer-repeats: 0.38-0.42% in
both `targeting_correct_snr` and `targeting_cf_bernoulli`, 0.81-1.09% in
`targeting_depth_compare` (highest at depth 2). These are the cases where the winner
and runner-up are close enough that the truncation window collapses; the returned
iterate is kept, so they are counted rather than dropped.

## What is pinned

Per run directory: `config.yaml`, `config.json`, `experiment.log`, and
`results_slim.pkl`. The raw `results.pkl` stays untracked -- it is 153 MB to 917 MB per
run, of which more than 99.9% is `*_selection_arr`, the per-customer chosen treatment
for every repeat (10,000 int64 per repeat). Dropping only those fields takes each run
to under 1 MB while keeping every `wc`, `val_est`, `val_true` and diagnostic array the
tables above use.
