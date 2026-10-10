# Targeting: m-out-of-n power sweep

Source run: `results/targeting_forest_moon_power_20261007_090403` (12/12 combinations, exit code 0).

- 2500 training rows per condition (control and two treatments), so N = 7500 pooled rows; 10000 target customers
- 1000 repeats per cell, 100 bootstrap draws per estimator
- Subsample drawn within each condition: m_k = floor(n_k^gamma) rows from each condition's n_k (about 2500) rows, so m is about 3 * floor(2500^gamma) in total: gamma=0.40 -> 22 per condition, gamma=0.50 -> 50, gamma=0.60 -> 109, gamma=0.70 -> 238, gamma=0.80 -> 518, gamma=0.90 -> 1131, gamma=0.95 -> 1675. Each bootstrap WC draw is scaled by sqrt(m/N) with m = sum(m_k).
- Supersedes the pooled-draw runs (m = floor(7500^gamma) over all rows): `results/targeting_forest_moon_power_20260917_004734`, `results/targeting_forest_functional_form_moon_power_20261005_133156`, `results/targeting_cf_bernoulli_moon_power_20261005_183928`. Their No Correction and Standard Bootstrap columns are identical to the runs below.
- The suite runs' `mn_bootstrap` (gamma=0.6) rows were produced with the pooled draw and have not been rerun.

All entries are `100 * wc / delta_tau`, Monte Carlo standard error in parentheses,
n = 1000 repeats. Positive = policy value still overstated (under-correction).


## Signed bias: mean(wc) / delta_tau

### Known Functional Form

| Estimator | dt=0.005 | dt=0.01 | dt=0.02 |
|---|---|---|---|
| No Correction | 198.2 (6.6) | 90.0 (3.4) | 32.9 (1.8) |
| Standard Bootstrap | 35.7 (7.6) | 11.4 (3.9) | -1.2 (2.0) |
| m-out-of-n, gamma=0.40 | -82.7 (6.7) | -50.4 (3.4) | -37.1 (1.8) |
| m-out-of-n, gamma=0.50 | -50.9 (6.7) | -34.4 (3.4) | -29.0 (1.8) |
| m-out-of-n, gamma=0.60 | -35.7 (6.7) | -26.6 (3.5) | -24.7 (1.9) |
| m-out-of-n, gamma=0.70 | -25.5 (6.8) | -21.2 (3.5) | -21.3 (1.9) |
| m-out-of-n, gamma=0.80 | -11.8 (7.0) | -13.8 (3.6) | -16.4 (1.9) |
| m-out-of-n, gamma=0.90 | 6.4 (7.2) | -3.8 (3.7) | -10.0 (2.0) |
| m-out-of-n, gamma=0.95 | 20.2 (7.4) | 3.4 (3.8) | -5.8 (2.0) |

### Causal Forest, max_depth=2

| Estimator | dt=0.005 | dt=0.01 | dt=0.02 |
|---|---|---|---|
| No Correction | 466.5 (18.9) | 230.3 (9.4) | 109.3 (4.8) |
| Standard Bootstrap | 182.9 (20.7) | 89.1 (10.4) | 40.1 (5.3) |
| m-out-of-n, gamma=0.40 | -15.2 (19.4) | -10.8 (9.7) | -11.8 (4.9) |
| m-out-of-n, gamma=0.50 | 15.9 (19.4) | 4.7 (9.7) | -4.2 (4.9) |
| m-out-of-n, gamma=0.60 | 42.7 (19.5) | 18.3 (9.8) | 2.9 (4.9) |
| m-out-of-n, gamma=0.70 | 45.5 (18.8) | 19.9 (9.5) | 4.1 (4.8) |
| m-out-of-n, gamma=0.80 | 73.7 (19.6) | 34.2 (9.8) | 11.5 (5.0) |
| m-out-of-n, gamma=0.90 | 104.6 (20.0) | 49.7 (10.0) | 20.0 (5.1) |
| m-out-of-n, gamma=0.95 | 135.3 (20.0) | 65.5 (10.0) | 28.4 (5.1) |

### Causal Forest, max_depth=5

| Estimator | dt=0.005 | dt=0.01 | dt=0.02 |
|---|---|---|---|
| No Correction | 690.3 (16.2) | 342.4 (8.1) | 165.8 (4.1) |
| Standard Bootstrap | 186.7 (17.8) | 91.4 (8.9) | 41.8 (4.5) |
| m-out-of-n, gamma=0.40 | 209.5 (16.7) | 101.5 (8.3) | 44.9 (4.2) |
| m-out-of-n, gamma=0.50 | 159.7 (16.8) | 76.8 (8.4) | 32.7 (4.2) |
| m-out-of-n, gamma=0.60 | 112.5 (16.9) | 53.4 (8.4) | 21.1 (4.2) |
| m-out-of-n, gamma=0.70 | 114.0 (16.6) | 54.1 (8.3) | 21.7 (4.2) |
| m-out-of-n, gamma=0.80 | 110.1 (17.0) | 52.6 (8.5) | 21.5 (4.3) |
| m-out-of-n, gamma=0.90 | 124.8 (17.2) | 60.2 (8.6) | 26.0 (4.3) |
| m-out-of-n, gamma=0.95 | 159.2 (17.4) | 77.6 (8.7) | 34.9 (4.4) |

### Causal Forest, max_depth=10

| Estimator | dt=0.005 | dt=0.01 | dt=0.02 |
|---|---|---|---|
| No Correction | 1648.9 (16.0) | 823.1 (8.0) | 408.5 (4.0) |
| Standard Bootstrap | 420.8 (17.4) | 209.4 (8.7) | 102.6 (4.4) |
| m-out-of-n, gamma=0.40 | 1172.4 (16.6) | 584.3 (8.3) | 289.1 (4.2) |
| m-out-of-n, gamma=0.50 | 1121.9 (16.6) | 559.5 (8.3) | 276.6 (4.2) |
| m-out-of-n, gamma=0.60 | 986.2 (16.7) | 491.6 (8.3) | 242.9 (4.2) |
| m-out-of-n, gamma=0.70 | 805.1 (16.5) | 401.2 (8.3) | 198.0 (4.1) |
| m-out-of-n, gamma=0.80 | 659.7 (16.9) | 328.8 (8.5) | 162.1 (4.2) |
| m-out-of-n, gamma=0.90 | 540.0 (17.0) | 268.9 (8.5) | 132.4 (4.3) |
| m-out-of-n, gamma=0.95 | 492.4 (17.2) | 245.2 (8.6) | 120.6 (4.3) |


## Mean absolute error: mean|wc| / delta_tau

### Known Functional Form

| Estimator | dt=0.005 | dt=0.01 | dt=0.02 |
|---|---|---|---|
| No Correction | 232.8 (5.3) | 111.9 (2.6) | 52.2 (1.3) |
| Standard Bootstrap | 192.5 (4.6) | 98.0 (2.3) | 52.1 (1.2) |
| m-out-of-n, gamma=0.40 | 185.0 (4.2) | 97.9 (2.2) | 56.8 (1.3) |
| m-out-of-n, gamma=0.50 | 175.2 (4.0) | 91.9 (2.1) | 52.9 (1.2) |
| m-out-of-n, gamma=0.60 | 174.4 (4.0) | 91.1 (2.1) | 52.0 (1.2) |
| m-out-of-n, gamma=0.70 | 175.3 (4.1) | 91.2 (2.1) | 51.5 (1.2) |
| m-out-of-n, gamma=0.80 | 178.0 (4.2) | 92.3 (2.1) | 51.5 (1.2) |
| m-out-of-n, gamma=0.90 | 182.9 (4.3) | 94.2 (2.2) | 51.5 (1.2) |
| m-out-of-n, gamma=0.95 | 186.7 (4.5) | 95.5 (2.3) | 51.6 (1.2) |

### Causal Forest, max_depth=2

| Estimator | dt=0.005 | dt=0.01 | dt=0.02 |
|---|---|---|---|
| No Correction | 617.0 (13.9) | 307.2 (6.9) | 151.1 (3.4) |
| Standard Bootstrap | 543.3 (12.9) | 272.8 (6.5) | 137.2 (3.3) |
| m-out-of-n, gamma=0.40 | 489.2 (11.7) | 245.6 (5.9) | 124.4 (3.0) |
| m-out-of-n, gamma=0.50 | 494.3 (11.5) | 248.2 (5.8) | 125.4 (2.9) |
| m-out-of-n, gamma=0.60 | 496.6 (11.7) | 248.8 (5.9) | 125.2 (3.0) |
| m-out-of-n, gamma=0.70 | 476.6 (11.4) | 239.3 (5.7) | 121.1 (2.9) |
| m-out-of-n, gamma=0.80 | 501.5 (11.7) | 251.4 (5.9) | 126.4 (3.0) |
| m-out-of-n, gamma=0.90 | 512.0 (12.2) | 256.9 (6.1) | 129.0 (3.1) |
| m-out-of-n, gamma=0.95 | 525.5 (11.9) | 263.7 (6.0) | 132.7 (3.0) |

### Causal Forest, max_depth=5

| Estimator | dt=0.005 | dt=0.01 | dt=0.02 |
|---|---|---|---|
| No Correction | 726.0 (14.5) | 360.8 (7.3) | 175.9 (3.6) |
| Standard Bootstrap | 470.6 (11.4) | 235.0 (5.7) | 116.8 (2.8) |
| m-out-of-n, gamma=0.40 | 453.2 (10.8) | 225.8 (5.4) | 111.5 (2.7) |
| m-out-of-n, gamma=0.50 | 447.1 (10.4) | 223.1 (5.2) | 110.7 (2.6) |
| m-out-of-n, gamma=0.60 | 438.5 (10.2) | 219.0 (5.1) | 109.2 (2.5) |
| m-out-of-n, gamma=0.70 | 425.6 (10.3) | 212.4 (5.2) | 105.7 (2.6) |
| m-out-of-n, gamma=0.80 | 443.1 (10.3) | 221.3 (5.1) | 110.1 (2.6) |
| m-out-of-n, gamma=0.90 | 448.3 (10.5) | 223.9 (5.2) | 111.6 (2.6) |
| m-out-of-n, gamma=0.95 | 462.5 (10.6) | 230.9 (5.3) | 115.0 (2.6) |

### Causal Forest, max_depth=10

| Estimator | dt=0.005 | dt=0.01 | dt=0.02 |
|---|---|---|---|
| No Correction | 1648.9 (16.0) | 823.1 (8.0) | 408.5 (4.0) |
| Standard Bootstrap | 560.8 (12.9) | 279.9 (6.4) | 138.8 (3.2) |
| m-out-of-n, gamma=0.40 | 1175.8 (16.4) | 586.1 (8.2) | 290.2 (4.1) |
| m-out-of-n, gamma=0.50 | 1126.2 (16.3) | 561.8 (8.2) | 277.9 (4.1) |
| m-out-of-n, gamma=0.60 | 997.9 (16.0) | 497.6 (8.0) | 246.1 (4.0) |
| m-out-of-n, gamma=0.70 | 831.9 (15.2) | 414.9 (7.6) | 205.2 (3.8) |
| m-out-of-n, gamma=0.80 | 716.0 (14.5) | 357.3 (7.2) | 176.9 (3.6) |
| m-out-of-n, gamma=0.90 | 634.6 (13.4) | 316.7 (6.7) | 156.8 (3.3) |
| m-out-of-n, gamma=0.95 | 602.9 (13.2) | 300.9 (6.6) | 149.2 (3.3) |


## Best gamma by MAE

| Model | best gamma per SNR | within 2 SE of best | beats standard bootstrap |
|---|---|---|---|
| Known Functional Form | 0.60, 0.60, 0.70 | 0.50, 0.60, 0.70, 0.80, 0.90, 0.95 | yes |
| Causal Forest, max_depth=2 | 0.70, 0.70, 0.70 | 0.40, 0.50, 0.60, 0.70, 0.80 | yes |
| Causal Forest, max_depth=5 | 0.70, 0.70, 0.70 | 0.50, 0.60, 0.70, 0.80 | yes |
| Causal Forest, max_depth=10 | 0.95, 0.95, 0.95 | 0.95 | no |


## Extension: functional form and binary outcomes

Fills the gamma grid for the two depth-5 studies the sweep above did not cover.
Same estimators, repeats, bootstrap draws and per-condition subsampling as above; DGP, seed
and forest identical to the corresponding suite config.

- Functional form: `results/targeting_forest_functional_form_moon_power_20261006_112916`
  (2/2 combinations, exit code 0), config `targeting_forest_functional_form_moon_power.yaml`.
  g(x)=x**2+x is the `causal_forest_depth_5, dt=0.01` cell above.
- Binary: `results/targeting_cf_bernoulli_moon_power_20261006_165029`
  (6/6 combinations, exit code 0), config `targeting_cf_bernoulli_moon_power.yaml`.
  Columns are (p1, p2); entries are scaled by delta_p = p2 - p1.
- In every cell, No Correction and Standard Bootstrap reproduce the suite runs
  (`targeting_forest_functional_form_20260923_131000`, `targeting_cf_bernoulli_20260919_214621`)
  exactly.
- Binary caveat: each m-out-of-n draw is redrawn until every (treatment, outcome) stratum
  has >= 2 rows. At gamma=0.40 only ~32% / 84% / 97% of draws qualify for p1 = 0.1 / 0.2 / 0.3,
  and ~86% at gamma=0.50, p1=0.1, so those cells are conditional on outcome support.
  (Under the old pooled draw these were ~6% / 36% / 66% and ~49%.)

### Signed bias: mean(wc) / delta_tau

### Functional form, causal forest max_depth=5, (tau1, tau2) = (1, 1.01)

| Estimator | g(x)=x | g(x)=\|x\| |
|---|---|---|
| No Correction | 373.2 (7.9) | 340.8 (7.9) |
| Standard Bootstrap | 67.8 (8.6) | 75.0 (8.6) |
| m-out-of-n, gamma=0.40 | 221.0 (8.1) | 178.6 (8.1) |
| m-out-of-n, gamma=0.50 | 166.7 (8.2) | 130.8 (8.2) |
| m-out-of-n, gamma=0.60 | 105.4 (8.1) | 82.9 (8.1) |
| m-out-of-n, gamma=0.70 | 80.5 (8.1) | 72.6 (8.1) |
| m-out-of-n, gamma=0.80 | 66.7 (8.3) | 56.4 (8.3) |
| m-out-of-n, gamma=0.90 | 65.2 (8.3) | 62.5 (8.4) |
| m-out-of-n, gamma=0.95 | 70.3 (8.4) | 70.6 (8.5) |

### Binary outcome, causal forest max_depth=5

| Estimator | p=(0.1, 0.101) | p=(0.2, 0.201) | p=(0.3, 0.301) | p=(0.1, 0.105) | p=(0.2, 0.205) | p=(0.3, 0.305) |
|---|---|---|---|---|---|---|
| No Correction | 909.7 (14.7) | 1210.6 (18.2) | 1392.3 (20.3) | 178.9 (2.9) | 239.9 (3.7) | 277.0 (4.1) |
| Standard Bootstrap | 179.1 (15.5) | 184.1 (19.8) | 229.0 (21.8) | 32.4 (3.1) | 36.7 (4.0) | 41.7 (4.3) |
| m-out-of-n, gamma=0.40 | 184.9 (15.2) | 618.9 (18.5) | 816.7 (20.5) | 36.5 (3.1) | 121.4 (3.7) | 160.1 (4.1) |
| m-out-of-n, gamma=0.50 | 355.0 (14.7) | 539.4 (18.5) | 656.6 (20.4) | 68.2 (3.0) | 107.3 (3.7) | 126.8 (4.0) |
| m-out-of-n, gamma=0.60 | 251.8 (14.5) | 318.7 (18.2) | 375.6 (20.5) | 47.7 (2.9) | 61.4 (3.7) | 73.6 (4.1) |
| m-out-of-n, gamma=0.70 | 146.0 (14.6) | 165.0 (18.4) | 211.4 (20.4) | 26.0 (2.9) | 30.0 (3.7) | 37.9 (4.1) |
| m-out-of-n, gamma=0.80 | 125.9 (14.7) | 132.2 (18.6) | 184.5 (20.4) | 21.1 (2.9) | 27.0 (3.8) | 33.2 (4.2) |
| m-out-of-n, gamma=0.90 | 146.9 (15.0) | 166.7 (19.6) | 201.2 (20.8) | 26.6 (3.1) | 32.7 (3.9) | 36.0 (4.2) |
| m-out-of-n, gamma=0.95 | 151.1 (15.2) | 173.0 (19.5) | 202.6 (21.4) | 29.8 (3.1) | 34.6 (3.9) | 41.5 (4.3) |

### Mean absolute error: mean|wc| / delta_tau

### Functional form, causal forest max_depth=5, (tau1, tau2) = (1, 1.01)

| Estimator | g(x)=x | g(x)=\|x\| |
|---|---|---|
| No Correction | 386.4 (7.2) | 358.3 (7.1) |
| Standard Bootstrap | 221.9 (5.4) | 224.3 (5.5) |
| m-out-of-n, gamma=0.40 | 274.7 (6.2) | 251.3 (5.9) |
| m-out-of-n, gamma=0.50 | 249.8 (5.7) | 233.9 (5.5) |
| m-out-of-n, gamma=0.60 | 221.9 (5.2) | 216.6 (5.1) |
| m-out-of-n, gamma=0.70 | 211.6 (5.2) | 210.4 (5.2) |
| m-out-of-n, gamma=0.80 | 218.0 (5.1) | 216.7 (5.1) |
| m-out-of-n, gamma=0.90 | 218.0 (5.1) | 218.5 (5.2) |
| m-out-of-n, gamma=0.95 | 222.3 (5.1) | 223.0 (5.2) |

### Binary outcome, causal forest max_depth=5

| Estimator | p=(0.1, 0.101) | p=(0.2, 0.201) | p=(0.3, 0.301) | p=(0.1, 0.105) | p=(0.2, 0.205) | p=(0.3, 0.305) |
|---|---|---|---|---|---|---|
| No Correction | 916.1 (14.2) | 1217.4 (17.7) | 1396.4 (20.0) | 180.7 (2.8) | 241.8 (3.6) | 277.9 (4.0) |
| Standard Bootstrap | 417.3 (9.9) | 511.2 (12.8) | 569.9 (14.2) | 83.2 (1.9) | 103.6 (2.6) | 113.0 (2.8) |
| m-out-of-n, gamma=0.40 | 409.7 (9.9) | 701.5 (15.2) | 876.3 (17.8) | 81.0 (2.0) | 138.2 (3.1) | 172.9 (3.5) |
| m-out-of-n, gamma=0.50 | 469.2 (11.0) | 644.2 (14.7) | 753.0 (16.8) | 93.4 (2.2) | 128.2 (3.0) | 147.8 (3.2) |
| m-out-of-n, gamma=0.60 | 414.8 (10.1) | 513.2 (13.0) | 599.3 (14.2) | 81.6 (2.0) | 102.2 (2.6) | 116.6 (2.9) |
| m-out-of-n, gamma=0.70 | 383.3 (9.4) | 473.7 (11.8) | 534.5 (13.3) | 75.7 (1.9) | 94.8 (2.4) | 106.3 (2.6) |
| m-out-of-n, gamma=0.80 | 378.1 (9.5) | 470.3 (11.9) | 530.4 (13.0) | 75.2 (1.8) | 96.4 (2.4) | 106.9 (2.6) |
| m-out-of-n, gamma=0.90 | 387.8 (9.8) | 494.7 (12.9) | 541.3 (13.5) | 80.8 (2.0) | 100.9 (2.5) | 109.0 (2.7) |
| m-out-of-n, gamma=0.95 | 401.5 (9.6) | 502.2 (12.6) | 556.4 (13.8) | 80.4 (2.0) | 99.7 (2.5) | 111.9 (2.8) |

### Best gamma by MAE

| Model | best gamma per cell | within 2 SE of best | beats standard bootstrap |
|---|---|---|---|
| Functional form, max_depth=5 | 0.70, 0.70 | 0.60, 0.70, 0.80, 0.90 | yes |
| Binary outcome, max_depth=5 | 0.80, 0.80, 0.80, 0.80, 0.70, 0.70 | 0.70, 0.80, 0.90, 0.95 | yes |
