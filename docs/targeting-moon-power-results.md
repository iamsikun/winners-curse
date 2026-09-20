# Targeting: m-out-of-n power sweep

Source run: `results/targeting_forest_moon_power_20260917_004734` (12/12 combinations, exit code 0).
Resumed from `results/targeting_forest_moon_power_20260917_000232`, which holds the
three Known-Functional-Form combinations.

- N = 2500 training rows, 10000 target customers
- 1000 repeats per cell, 100 bootstrap draws per estimator
- Subsample size m = floor(N^gamma): gamma=0.40 -> m=22, gamma=0.50 -> m=50, gamma=0.60 -> m=109, gamma=0.70 -> m=239, gamma=0.80 -> m=522, gamma=0.90 -> m=1143, gamma=0.95 -> m=1690

All entries are `100 * wc / delta_tau`, Monte Carlo standard error in parentheses,
n = 1000 repeats. Positive = policy value still overstated (under-correction).


## Signed bias: mean(wc) / delta_tau

### Known Functional Form

| Estimator | dt=0.005 | dt=0.01 | dt=0.02 |
|---|---|---|---|
| No Correction | 198.2 (6.6) | 90.0 (3.4) | 32.9 (1.8) |
| Standard Bootstrap | 35.7 (7.6) | 11.4 (3.9) | -1.2 (2.0) |
| m-out-of-n, gamma=0.40 | -187.1 (6.8) | -102.6 (3.5) | -63.3 (1.9) |
| m-out-of-n, gamma=0.50 | -75.6 (6.6) | -46.8 (3.4) | -35.3 (1.8) |
| m-out-of-n, gamma=0.60 | -44.9 (6.7) | -31.3 (3.4) | -27.3 (1.8) |
| m-out-of-n, gamma=0.70 | -30.9 (6.7) | -24.0 (3.5) | -23.1 (1.9) |
| m-out-of-n, gamma=0.80 | -17.9 (6.9) | -17.0 (3.6) | -18.4 (1.9) |
| m-out-of-n, gamma=0.90 | 4.2 (7.2) | -5.0 (3.7) | -10.8 (2.0) |
| m-out-of-n, gamma=0.95 | 18.1 (7.4) | 2.4 (3.8) | -6.4 (2.0) |

### Causal Forest, max_depth=2

| Estimator | dt=0.005 | dt=0.01 | dt=0.02 |
|---|---|---|---|
| No Correction | 466.5 (18.9) | 230.3 (9.4) | 109.3 (4.8) |
| Standard Bootstrap | 182.9 (20.7) | 89.1 (10.4) | 40.1 (5.3) |
| m-out-of-n, gamma=0.40 | 8.3 (19.3) | 0.9 (9.7) | -6.0 (4.9) |
| m-out-of-n, gamma=0.50 | -31.8 (19.2) | -18.9 (9.6) | -15.9 (4.9) |
| m-out-of-n, gamma=0.60 | 26.5 (19.4) | 10.2 (9.7) | -1.3 (4.9) |
| m-out-of-n, gamma=0.70 | 46.8 (19.0) | 20.4 (9.6) | 4.0 (4.8) |
| m-out-of-n, gamma=0.80 | 63.3 (19.5) | 29.0 (9.8) | 8.9 (5.0) |
| m-out-of-n, gamma=0.90 | 101.5 (19.8) | 48.3 (9.9) | 19.0 (5.0) |
| m-out-of-n, gamma=0.95 | 131.6 (20.0) | 63.5 (10.0) | 27.0 (5.1) |

### Causal Forest, max_depth=5

| Estimator | dt=0.005 | dt=0.01 | dt=0.02 |
|---|---|---|---|
| No Correction | 690.3 (16.2) | 342.4 (8.1) | 165.8 (4.1) |
| Standard Bootstrap | 186.7 (17.8) | 91.4 (8.9) | 41.8 (4.5) |
| m-out-of-n, gamma=0.40 | 233.2 (16.8) | 113.3 (8.4) | 50.7 (4.2) |
| m-out-of-n, gamma=0.50 | 191.0 (16.5) | 92.4 (8.3) | 40.3 (4.1) |
| m-out-of-n, gamma=0.60 | 111.4 (16.8) | 52.8 (8.4) | 20.7 (4.2) |
| m-out-of-n, gamma=0.70 | 123.5 (16.7) | 59.0 (8.3) | 24.3 (4.2) |
| m-out-of-n, gamma=0.80 | 107.7 (17.0) | 51.5 (8.5) | 20.9 (4.2) |
| m-out-of-n, gamma=0.90 | 121.1 (17.1) | 58.2 (8.6) | 24.8 (4.3) |
| m-out-of-n, gamma=0.95 | 149.2 (17.4) | 72.6 (8.7) | 32.2 (4.3) |

### Causal Forest, max_depth=10

| Estimator | dt=0.005 | dt=0.01 | dt=0.02 |
|---|---|---|---|
| No Correction | 1648.9 (16.0) | 823.1 (8.0) | 408.5 (4.0) |
| Standard Bootstrap | 420.8 (17.4) | 209.4 (8.7) | 102.6 (4.4) |
| m-out-of-n, gamma=0.40 | 1195.8 (16.8) | 596.0 (8.4) | 294.9 (4.2) |
| m-out-of-n, gamma=0.50 | 1154.9 (16.4) | 575.8 (8.2) | 284.7 (4.1) |
| m-out-of-n, gamma=0.60 | 1072.5 (16.5) | 534.8 (8.3) | 264.5 (4.1) |
| m-out-of-n, gamma=0.70 | 875.4 (16.6) | 436.3 (8.3) | 215.4 (4.2) |
| m-out-of-n, gamma=0.80 | 700.3 (16.9) | 349.0 (8.4) | 172.1 (4.2) |
| m-out-of-n, gamma=0.90 | 556.8 (16.9) | 277.2 (8.5) | 136.3 (4.2) |
| m-out-of-n, gamma=0.95 | 496.4 (17.1) | 247.0 (8.6) | 121.5 (4.3) |


## Mean absolute error: mean|wc| / delta_tau

### Known Functional Form

| Estimator | dt=0.005 | dt=0.01 | dt=0.02 |
|---|---|---|---|
| No Correction | 232.8 (5.3) | 111.9 (2.6) | 52.2 (1.3) |
| Standard Bootstrap | 192.5 (4.6) | 98.0 (2.3) | 52.1 (1.2) |
| m-out-of-n, gamma=0.40 | 236.2 (5.0) | 125.5 (2.6) | 72.7 (1.5) |
| m-out-of-n, gamma=0.50 | 179.9 (4.1) | 95.2 (2.1) | 55.5 (1.2) |
| m-out-of-n, gamma=0.60 | 175.5 (4.0) | 91.8 (2.1) | 52.6 (1.2) |
| m-out-of-n, gamma=0.70 | 174.3 (4.0) | 90.7 (2.1) | 51.5 (1.2) |
| m-out-of-n, gamma=0.80 | 176.4 (4.1) | 91.7 (2.1) | 51.4 (1.2) |
| m-out-of-n, gamma=0.90 | 181.9 (4.3) | 93.8 (2.2) | 51.4 (1.2) |
| m-out-of-n, gamma=0.95 | 187.2 (4.5) | 95.9 (2.3) | 51.7 (1.2) |

### Causal Forest, max_depth=2

| Estimator | dt=0.005 | dt=0.01 | dt=0.02 |
|---|---|---|---|
| No Correction | 617.0 (13.9) | 307.2 (6.9) | 151.1 (3.4) |
| Standard Bootstrap | 543.3 (12.9) | 272.8 (6.5) | 137.2 (3.3) |
| m-out-of-n, gamma=0.40 | 485.6 (11.6) | 243.8 (5.8) | 123.4 (2.9) |
| m-out-of-n, gamma=0.50 | 491.3 (11.3) | 247.0 (5.7) | 125.9 (2.9) |
| m-out-of-n, gamma=0.60 | 489.9 (11.6) | 245.8 (5.8) | 124.4 (2.9) |
| m-out-of-n, gamma=0.70 | 486.4 (11.3) | 244.1 (5.7) | 123.2 (2.9) |
| m-out-of-n, gamma=0.80 | 501.6 (11.6) | 251.5 (5.8) | 127.1 (2.9) |
| m-out-of-n, gamma=0.90 | 509.2 (12.0) | 255.3 (6.0) | 128.2 (3.0) |
| m-out-of-n, gamma=0.95 | 519.8 (12.1) | 260.7 (6.0) | 131.0 (3.0) |

### Causal Forest, max_depth=5

| Estimator | dt=0.005 | dt=0.01 | dt=0.02 |
|---|---|---|---|
| No Correction | 726.0 (14.5) | 360.8 (7.3) | 175.9 (3.6) |
| Standard Bootstrap | 470.6 (11.4) | 235.0 (5.7) | 116.8 (2.8) |
| m-out-of-n, gamma=0.40 | 462.8 (11.0) | 230.6 (5.5) | 113.7 (2.7) |
| m-out-of-n, gamma=0.50 | 447.1 (10.4) | 223.0 (5.2) | 110.6 (2.6) |
| m-out-of-n, gamma=0.60 | 440.1 (10.0) | 219.7 (5.0) | 109.2 (2.5) |
| m-out-of-n, gamma=0.70 | 433.7 (10.2) | 216.5 (5.1) | 108.0 (2.5) |
| m-out-of-n, gamma=0.80 | 441.7 (10.2) | 220.6 (5.1) | 109.8 (2.5) |
| m-out-of-n, gamma=0.90 | 444.8 (10.5) | 222.2 (5.2) | 110.7 (2.6) |
| m-out-of-n, gamma=0.95 | 459.9 (10.6) | 229.5 (5.3) | 114.1 (2.6) |

### Causal Forest, max_depth=10

| Estimator | dt=0.005 | dt=0.01 | dt=0.02 |
|---|---|---|---|
| No Correction | 1648.9 (16.0) | 823.1 (8.0) | 408.5 (4.0) |
| Standard Bootstrap | 560.8 (12.9) | 279.9 (6.4) | 138.8 (3.2) |
| m-out-of-n, gamma=0.40 | 1199.8 (16.5) | 598.2 (8.2) | 296.1 (4.1) |
| m-out-of-n, gamma=0.50 | 1159.2 (16.1) | 578.1 (8.0) | 285.9 (4.0) |
| m-out-of-n, gamma=0.60 | 1081.1 (15.9) | 539.1 (8.0) | 266.8 (4.0) |
| m-out-of-n, gamma=0.70 | 895.3 (15.5) | 446.3 (7.8) | 220.7 (3.9) |
| m-out-of-n, gamma=0.80 | 745.4 (14.8) | 371.9 (7.4) | 184.0 (3.7) |
| m-out-of-n, gamma=0.90 | 644.3 (13.4) | 321.5 (6.7) | 159.2 (3.3) |
| m-out-of-n, gamma=0.95 | 603.7 (13.2) | 301.1 (6.6) | 149.3 (3.3) |


## Best gamma by MAE

| Model | best gamma per SNR | within 2 SE of best | beats standard bootstrap |
|---|---|---|---|
| Known Functional Form | 0.70, 0.70, 0.80 | 0.50, 0.60, 0.70, 0.80, 0.90, 0.95 | yes |
| Causal Forest, max_depth=2 | 0.40, 0.40, 0.70 | 0.40, 0.50, 0.60, 0.70, 0.80, 0.90 | yes |
| Causal Forest, max_depth=5 | 0.70, 0.70, 0.70 | 0.50, 0.60, 0.70, 0.80, 0.90 | yes |
| Causal Forest, max_depth=10 | 0.95, 0.95, 0.95 | 0.95 | no |
