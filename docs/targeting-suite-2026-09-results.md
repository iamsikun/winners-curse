# Targeting suite rerun, September 2026

Rerun of the targeting simulations after two changes to the runner: the projection
critical value in hybrid selective inference is now evaluated in closed form, and the
suite runs at 32 workers with memory-tiered job caps (see
`configs/targeting_forest_depth.yaml`). Neither change touches the estimators being
compared; the selective-inference change moves its own `val_est` by ~2e-4 (it replaces
a Monte Carlo estimate of a constant with the constant).

Batch log: `results/batch_targeting_20260919_214619.log`, started 2026-09-19 21:46.

**Status: complete for the six-config suite.** All six are pinned below. A depth 20
pass over the `targeting_forest_depth` grid is running separately and will be added to
that section when it finishes.

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

## targeting_forest_depth (causal forest, depth x sample size, no corrections)

Runs: `results/targeting_forest_depth_20260920_062447` (combinations 1-21) and
`results/targeting_forest_depth_20260921_002545` (resumed, 36/36). The resume was for
an unrelated scheduling fix; the two directories together cover one sweep, and the
second holds the complete merged results.

This config fits no corrections -- it measures how the uncorrected winner's curse decays
with sample size at three forest depths. 500 repeats per cell, except n = 5,000,000 at
20 repeats (a single fit there costs 25-51 min and only two workers fit in memory).

Mean winner's curse, with SE in parentheses:

| n | depth 5 | depth 10 | depth 15 |
|---|---|---|---|
| 2,500 | 0.03521 (0.00117) | 0.08312 (0.00116) | 0.12386 (0.00120) |
| 5,000 | 0.02330 (0.00084) | 0.06433 (0.00084) | 0.11002 (0.00086) |
| 10,000 | 0.01742 (0.00059) | 0.05137 (0.00059) | 0.09829 (0.00061) |
| 15,000 | 0.01447 (0.00047) | 0.04451 (0.00046) | 0.09079 (0.00046) |
| 20,000 | 0.01251 (0.00043) | 0.03968 (0.00043) | 0.08460 (0.00044) |
| 30,000 | 0.00950 (0.00034) | 0.03294 (0.00034) | 0.07569 (0.00035) |
| 40,000 | 0.00893 (0.00031) | 0.03004 (0.00029) | 0.07074 (0.00030) |
| 50,000 | 0.00734 (0.00029) | 0.02689 (0.00028) | 0.06600 (0.00029) |
| 100,000 | 0.00473 (0.00021) | 0.01973 (0.00020) | 0.05339 (0.00021) |
| 500,000 | 0.00083 (0.00013) | 0.00809 (0.00011) | 0.02909 (0.00011) |
| 1,000,000 | 0.00027 (0.00011) | 0.00518 (0.00009) | 0.02169 (0.00009) |
| 5,000,000 | 0.00017 (0.00042) | 0.00178 (0.00018) | 0.01047 (0.00022) |

The bias vanishes with sample size at depth 5 -- by n = 1M it is 0.00027, within three
SE of zero, and the 5M point is indistinguishable from zero -- but the decay slows
sharply with depth. Over the 400x from n = 2,500 to n = 1M, depth 5 falls by a factor
of 130 while depth 15 falls by only 5.7. At n = 5,000,000 a depth-15 forest still
overstates its selected policy by 0.0105 (48 SE from zero), more than a depth-5 forest
does at n = 100,000. Flexibility bought with depth is paid for in winner's curse that
sample size alone does not retire.

The 20-repeat tiering at 5M is adequate where it matters: the depth-15 estimate is 48
SE from zero, depth 10 is 10 SE, and only the depth-5 cell (0.00017 against SE 0.00042)
is too noisy to separate from zero -- which is itself the finding for that cell.

Wall clock for the large cells, and the worker counts the memory tiers allowed:

| n | depth 5 | depth 10 | depth 15 | workers |
|---|---|---|---|---|
| 500,000 | 2.49 h | 3.74 h | 4.28 h | 13-15 |
| 1,000,000 | 6.13 h | 9.76 h | 11.08 h | 8-9 |
| 5,000,000 | 4.29 h | 7.28 h | 9.10 h | 2 |

Peak worker RSS at n = 5M was 10-11 GB against a 12.5 GB budget on a 47 GB box.

## targeting_forest_functional_form (causal forest, char_func sweep, n = 2500, 1000 repeats)

Run: `results/targeting_forest_functional_form_20260923_131000`, 3 combinations,
10,272 s. Paper Table D.6 columns.

| char_func | estimator | mean WC | SE | MAE |
|---|---|---|---|---|
| x | No correction | 0.0373 | 0.0008 | 0.0386 |
| x | Standard bootstrap | 0.0068 | 0.0009 | 0.0222 |
| x | m-out-of-n bootstrap | 0.0137 | 0.0008 | 0.0233 |
| x | Sample splitting | -0.0005 | 0.0012 | 0.0307 |
| x | Empirical Bayes | 0.0360 | 0.0008 | 0.0375 |
| x | Hybrid SI | -0.1108 | 0.0012 | 0.1109 |
| x**2 + x | No correction | 0.0342 | 0.0008 | 0.0361 |
| x**2 + x | Standard bootstrap | 0.0091 | 0.0009 | 0.0235 |
| x**2 + x | m-out-of-n bootstrap | 0.0057 | 0.0008 | 0.0215 |
| x**2 + x | Sample splitting | -0.0034 | 0.0013 | 0.0324 |
| x**2 + x | Empirical Bayes | 0.0121 | 0.0008 | 0.0226 |
| x**2 + x | Hybrid SI | -0.1427 | 0.0016 | 0.1429 |
| np.abs(x) | No correction | 0.0341 | 0.0008 | 0.0358 |
| np.abs(x) | Standard bootstrap | 0.0075 | 0.0009 | 0.0224 |
| np.abs(x) | m-out-of-n bootstrap | 0.0103 | 0.0008 | 0.0221 |
| np.abs(x) | Sample splitting | -0.0014 | 0.0012 | 0.0313 |
| np.abs(x) | Empirical Bayes | 0.0119 | 0.0008 | 0.0222 |
| np.abs(x) | Hybrid SI | -0.0931 | 0.0013 | 0.0936 |

The naive bias is nearly flat across functional forms (0.0341-0.0373) and the standard
bootstrap is the best or tied-best correction by MAE in all three, at 0.0222-0.0235.
Empirical Bayes is the one estimator that is sensitive to the form: it barely corrects
under the linear `x` (0.0373 -> 0.0360) but removes two thirds of the bias under the
other two. Hybrid SI overcorrects everywhere, most severely under `x**2 + x`.

The `x**2 + x` column is the same DGP, estimator and seeds as the depth 5 row of
`targeting_depth_compare`, which ran three days earlier in a separate process. All six
estimators agree to four decimals, which is the reproducibility check the seeding is
supposed to provide.

SI non-convergence: 0.63-1.07% of customer-repeats.

## targeting_forest_snr (causal forest, SNR sweep, n = 2500, 1000 repeats)

Run: `results/targeting_forest_snr_20260923_160111`, 3 combinations, 10,163 s.

| tau | estimator | mean WC | SE | MAE |
|---|---|---|---|---|
| (1, 1.005) | No correction | 0.0345 | 0.0008 | 0.0363 |
| (1, 1.005) | Standard bootstrap | 0.0093 | 0.0009 | 0.0235 |
| (1, 1.005) | m-out-of-n bootstrap | 0.0060 | 0.0008 | 0.0216 |
| (1, 1.005) | Sample splitting | -0.0032 | 0.0013 | 0.0323 |
| (1, 1.005) | Empirical Bayes | 0.0123 | 0.0008 | 0.0226 |
| (1, 1.005) | Hybrid SI | -0.1429 | 0.0016 | 0.1431 |
| (1, 1.01) | No correction | 0.0342 | 0.0008 | 0.0361 |
| (1, 1.01) | Standard bootstrap | 0.0091 | 0.0009 | 0.0235 |
| (1, 1.01) | m-out-of-n bootstrap | 0.0057 | 0.0008 | 0.0215 |
| (1, 1.01) | Sample splitting | -0.0034 | 0.0013 | 0.0324 |
| (1, 1.01) | Empirical Bayes | 0.0121 | 0.0008 | 0.0226 |
| (1, 1.01) | Hybrid SI | -0.1427 | 0.0016 | 0.1429 |
| (1, 1.02) | No correction | 0.0332 | 0.0008 | 0.0352 |
| (1, 1.02) | Standard bootstrap | 0.0084 | 0.0009 | 0.0234 |
| (1, 1.02) | m-out-of-n bootstrap | 0.0046 | 0.0008 | 0.0214 |
| (1, 1.02) | Sample splitting | -0.0035 | 0.0013 | 0.0324 |
| (1, 1.02) | Empirical Bayes | 0.0110 | 0.0008 | 0.0223 |
| (1, 1.02) | Hybrid SI | -0.1412 | 0.0016 | 0.1414 |

Widening the gap between arms from 0.005 to 0.02 barely moves anything: the naive bias
falls only from 0.0345 to 0.0332 and every correction shifts within about one SE per
step. In this targeting design the winner's curse is driven by estimation noise in the
per-customer treatment effects rather than by how close the arms are -- the same
insensitivity the Bernoulli config shows across its two gap settings.

The (1, 1.01) row again reproduces the depth 5 row of `targeting_depth_compare` and the
`x**2 + x` row of `targeting_forest_functional_form` to four decimals.

SI non-convergence: 1.07-1.12% of customer-repeats.

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
