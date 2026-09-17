# Multiple-treatment rerun: 500 versus 1,000 repetitions

All 30 arrays finite with 1000 draws; first 500 draws exactly reproduce prior run; means differ by at most 3 Monte Carlo SE accounting for shared draws; SDs differ by at most 10%. These are descriptive stability checks, not formal hypothesis tests.

Previous: `results/ab_test_n_treatments_20260917_121102`
Replacement: `results/ab_test_n_treatments_20260917_122850`

All checks passed: **True**.

| Arms | Estimator | Old mean | New mean | Old SD | New SD | Mean change / SE |
|---|---|---:|---:|---:|---:|---:|
| 2 | nc | 0.011610 | 0.012096 | 0.016689 | 0.016584 | 0.93 |
| 2 | standard_bootstrap | 0.003680 | 0.004258 | 0.018864 | 0.018673 | 0.98 |
| 2 | moon_bootstrap | 0.000640 | 0.001064 | 0.016877 | 0.016825 | 0.80 |
| 2 | sample_splitting | -0.000992 | 0.000807 | 0.028454 | 0.028631 | 1.99 |
| 2 | eb_normal | 0.005063 | 0.005410 | 0.015567 | 0.015434 | 0.71 |
| 2 | hybrid_si | -0.005170 | -0.003835 | 0.030573 | 0.030174 | 1.40 |
| 4 | nc | 0.020446 | 0.020376 | 0.014351 | 0.014400 | 0.15 |
| 4 | standard_bootstrap | 0.005794 | 0.005727 | 0.017402 | 0.017477 | 0.12 |
| 4 | moon_bootstrap | 0.000349 | 0.000250 | 0.014690 | 0.014724 | 0.21 |
| 4 | sample_splitting | -0.000154 | -0.000438 | 0.027908 | 0.028226 | 0.32 |
| 4 | eb_normal | 0.009037 | 0.008983 | 0.012867 | 0.012928 | 0.13 |
| 4 | hybrid_si | -0.004372 | -0.004283 | 0.031769 | 0.031716 | 0.09 |
| 6 | nc | 0.024414 | 0.025209 | 0.013194 | 0.013150 | 1.91 |
| 6 | standard_bootstrap | 0.006324 | 0.007309 | 0.016606 | 0.016624 | 1.88 |
| 6 | moon_bootstrap | -0.000308 | 0.000462 | 0.013533 | 0.013423 | 1.82 |
| 6 | sample_splitting | 0.000449 | 0.001280 | 0.027973 | 0.028146 | 0.93 |
| 6 | eb_normal | 0.011029 | 0.011581 | 0.011521 | 0.011458 | 1.52 |
| 6 | hybrid_si | -0.006251 | -0.004091 | 0.033034 | 0.032762 | 2.09 |
| 8 | nc | 0.027602 | 0.027980 | 0.012644 | 0.012428 | 0.96 |
| 8 | standard_bootstrap | 0.007208 | 0.007737 | 0.016183 | 0.016028 | 1.04 |
| 8 | moon_bootstrap | -0.000168 | 0.000172 | 0.012952 | 0.012698 | 0.85 |
| 8 | sample_splitting | -0.000051 | -0.000135 | 0.027149 | 0.027919 | 0.10 |
| 8 | eb_normal | 0.012708 | 0.012880 | 0.010927 | 0.010627 | 0.51 |
| 8 | hybrid_si | -0.004728 | -0.003485 | 0.031962 | 0.032091 | 1.22 |
| 10 | nc | 0.029990 | 0.030083 | 0.011724 | 0.011659 | 0.25 |
| 10 | standard_bootstrap | 0.008124 | 0.008176 | 0.015225 | 0.015230 | 0.11 |
| 10 | moon_bootstrap | -0.000077 | -0.000008 | 0.011981 | 0.011890 | 0.18 |
| 10 | sample_splitting | -0.000008 | -0.000999 | 0.027266 | 0.027226 | 1.15 |
| 10 | eb_normal | 0.013785 | 0.013869 | 0.009912 | 0.009774 | 0.27 |
| 10 | hybrid_si | -0.003145 | -0.003336 | 0.031802 | 0.031912 | 0.19 |
