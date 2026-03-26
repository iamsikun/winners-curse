# Bootstrap Correction for the Winner's Curse

## Winner's Curse

Given unbiased estimators $\hat{\tau}_j$ for the expected reward $\tau_j$ of each option $a_j \in \mathcal{A}$, the estimator $\hat{\tau}_{\hat{j}^*}$ of the empirical winner $\hat{j}^* = \arg\max_{j} \hat{\tau}_j$ overestimates its true performance:

$$
\text{WC} := \mathbb{E}_{\mathcal{D}}\!\left[\hat{\tau}_{\hat{j}^*} - \tau_{\hat{j}^*}\right] \geq 0
$$

This bias arises because the option whose effectiveness is most overestimated is more likely to be selected. The magnitude depends on the signal-to-noise ratio $\Delta\tau / \sigma$, the number of alternatives $J$, and the sample size $N$.

## Generic Bootstrap Correction Framework

Both algorithms share the same correction logic (implemented in `bootstrap_correction_estimator`):

1. Compute empirical estimates $\hat{\tau}_j$ from data $\mathcal{D}$.
2. Select the empirical winner $\hat{j}^* = \arg\max_j \hat{\tau}_j$.
3. Compute the naive value $V_{\text{naive}} = f(\hat{j}^*, \hat{\tau})$.
4. For $b = 1, \dots, B$: draw a bootstrap sample $\mathcal{D}^{(b)}$, compute bootstrap estimates $\hat{\tau}^{(b)}_j$, select bootstrap winner $\hat{j}^{*(b)}$, and compute bias:

$$
\text{WC}^{(b)} = f\!\left(\hat{j}^{*(b)},\; \hat{\tau}^{(b)}\right) - f\!\left(\hat{j}^{*(b)},\; \hat{\tau}\right)
$$

5. Estimate bias: $\widehat{\text{WC}} = \frac{1}{B}\sum_{b=1}^{B} \text{WC}^{(b)}$.
6. Return corrected estimate: $V_{\text{corrected}} = V_{\text{naive}} - \widehat{\text{WC}}$.

The key idea: bootstrap estimates $\hat{\tau}^{(b)}$ relate to empirical estimates $\hat{\tau}$ in the same way that $\hat{\tau}$ relates to the unknown true values $\tau$. The cross-evaluation term $f(\hat{j}^{*(b)}, \hat{\tau})$ acts as a proxy for $f(\hat{j}^{*(b)}, \tau)$, allowing us to estimate the selection bias.

## Algorithm 1: Standard Bootstrap

Draw $N$ observations **with replacement** from $\mathcal{D}$ to form each bootstrap sample.

| Step | Description |
|------|-------------|
| 1 | Compute $\hat{\tau}_j$ for all $j$ from $\mathcal{D}$ |
| 2 | Select $\hat{j}^* = \arg\max_j \hat{\tau}_j$ |
| 3 | For $b = 1, \dots, B$: sample $\mathcal{D}^{(b)}$ by drawing $N$ observations with replacement |
| 4 | Compute $\hat{\tau}^{(b)}_j$ from $\mathcal{D}^{(b)}$; select $\hat{j}^{*(b)} = \arg\max_j \hat{\tau}^{(b)}_j$ |
| 5 | $\widehat{\text{WC}} = B^{-1} \sum_b \left[\hat{\tau}^{(b)}_{\hat{j}^{*(b)}} - \hat{\tau}_{\hat{j}^{*(b)}}\right]$ |
| 6 | Return $\hat{\tau}_{\hat{j}^*} - \widehat{\text{WC}}$ |

## Algorithm 2: m-out-of-n Bootstrap

Draw $m < N$ observations with replacement, where $m = \lfloor N^{\gamma} \rfloor$ and $\gamma \in (0, 1)$ (default $\gamma = 0.95$).

| Step | Description |
|------|-------------|
| 1 | Compute $\hat{\tau}_j$ for all $j$ from $\mathcal{D}$ |
| 2 | Select $\hat{j}^* = \arg\max_j \hat{\tau}_j$ |
| 3 | For $b = 1, \dots, B$: sample $\mathcal{D}^{(b)}$ by drawing $m = \lfloor N^{\gamma} \rfloor$ observations with replacement |
| 4 | Compute $\hat{\tau}^{(b)}_j$ from $\mathcal{D}^{(b)}$; select $\hat{j}^{*(b)} = \arg\max_j \hat{\tau}^{(b)}_j$ |
| 5 | $\widehat{\text{WC}} = B^{-1} \sum_b \left[\hat{\tau}^{(b)}_{\hat{j}^{*(b)}} - \hat{\tau}_{\hat{j}^{*(b)}}\right]$ |
| 6 | Return $\hat{\tau}_{\hat{j}^*} - \widehat{\text{WC}}$ |

The **only difference** from the standard bootstrap is the subsample size in step 3.

## Why m-out-of-n?

The winner's curse statistic involves the $\max$ operator, which is **not fully Hadamard differentiable**. When two or more options have nearly equal estimated values, the $\arg\max$ has a discontinuity (a "kink") that causes the standard bootstrap to converge slowly.

Subsampling with $m < N$ introduces additional sampling variability that effectively smoothes this kink: the winner identity switches more frequently across bootstrap draws when the signal-to-noise ratio is small, producing a more accurate bias estimate.

The power $\gamma$ trades off two competing effects:
- **$\gamma \to 1$** (closer to standard bootstrap): less estimation noise, but the discontinuity bias reappears.
- **$\gamma$ too small**: excessive sampling variability causes overcorrection.

The default $\gamma = 0.95$ provides a good balance across typical marketing experiment settings.

## 2026-03-26: Scaling the m-out-of-n Bootstrap Correction

### Motivation

The m-out-of-n bootstrap estimates WC from bootstrap samples of size $m < N$. Because the estimation error scales as $\sigma / \sqrt{n}$, the bootstrap estimates $\hat{\tau}^{(b)}$ (based on $m$ observations) have variance $\sigma^2 / m$, which is larger than the variance of the original estimates $\sigma^2 / N$. The bootstrap world is therefore **noisier** than the real world, and the resulting WC estimate is inflated — explaining the overcorrection we observe.

To rescale the bias from the $m$-observation scale to the $N$-observation scale, we multiply by the ratio of standard deviations:

$$\widehat{\text{WC}}_{\text{scaled}} = \sqrt{\frac{m}{N}} \cdot \widehat{\text{WC}}_{m\text{-out-of-}n}$$

Since $m < N$, this factor is $< 1$ and shrinks the correction. The experiment below tests whether this scaling makes the m-out-of-n bootstrap unbiased.

### Experiment design

**Setting**: 2-arm continuous A/B test, Gaussian noise $\sigma = 1$, 1000 repeats per cell.

**Dimensions** (2-D sweep):

| Dimension | Values | Purpose |
|-----------|--------|---------|
| `tau_list` (SNR) | `[1, 1.005]`, `[1, 1.01]`, `[1, 1.02]`, `[1, 1.05]` | Low-to-high SNR; WC and the $\max$ kink are most pronounced at low SNR |
| `sample_size` | `500`, `1000`, `2500`, `5000` | The scaling factor $\sqrt{m/N}$ depends on $N$; varying $N$ tests whether the correction works across sample sizes |

**Estimators** (6 arms):

| Key | Method | Description |
|-----|--------|-------------|
| (built-in) | No correction | Naive $\hat{\tau}_{\hat{j}^*}$ |
| `standard_bootstrap` | Standard bootstrap | $N$-out-of-$N$ resampling, baseline correction |
| `moon_080_unscaled` | m-out-of-n, $\gamma = 0.8$, unscaled | Aggressive subsampling, no rescaling (expect large overcorrection) |
| `moon_080_scaled` | m-out-of-n, $\gamma = 0.8$, scaled by $\sqrt{m/N}$ | Same subsampling, with rescaling |
| `moon_095_unscaled` | m-out-of-n, $\gamma = 0.95$, unscaled | Mild subsampling, no rescaling (current default) |
| `moon_095_scaled` | m-out-of-n, $\gamma = 0.95$, scaled by $\sqrt{m/N}$ | Same subsampling, with rescaling |

### Implementation notes

- The scaling can be applied per bootstrap draw via a custom `wc_func` that multiplies each $\text{WC}^{(b)}$ by $\sqrt{m/N}$, or equivalently by scaling the mean bias in `bootstrap_correction_estimator`. Per-draw scaling via `wc_func` is preferred since it reuses the existing framework without modifying `bootstrap.py`.
- New estimator function (or a `scale_correction` flag on `get_wc_moon_boot`) needed in `ab_test.py`.
- Config file: `configs/ab_test_moon_scaling.yaml`.

### Expected results

1. **Unscaled m-out-of-n overcorrects**: negative mean WC (corrected estimate undershoots truth), worsening as $\gamma$ decreases and as $N$ increases (because $m/N$ shrinks for fixed $\gamma$, amplifying the noise gap).
2. **Scaled m-out-of-n is approximately unbiased**: mean WC near zero across all cells.
3. **$\gamma = 0.8$ unscaled overcorrects more than $\gamma = 0.95$ unscaled**: confirming the noise-level explanation.
4. **Standard bootstrap**: works well at high SNR, may show slight positive bias at very low SNR (the $\max$ kink issue).
5. **Sample size dimension**: overcorrection of unscaled m-out-of-n should be relatively stable across $N$ (since $m = N^\gamma$ keeps the ratio $m/N = N^{\gamma - 1}$ changing slowly), but the absolute WC magnitude should decrease with $N$ for all methods.
