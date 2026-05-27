# Scaling the m-out-of-n Bootstrap Correction

## Problem: the m-out-of-n bootstrap overcorrects

The standard bootstrap correction for the Winner's Curse resamples $N$ observations (with replacement) from the original $N$, estimates bias in this bootstrap world, and subtracts it from the naive estimate. This works because bootstrap estimates have roughly the same variance as the original estimates — both are based on $N$ effective observations — so the bootstrap bias is at the right scale.

The m-out-of-n (moon) bootstrap resamples only $m = \lfloor N^\gamma \rfloor$ observations, with $\gamma \in (0,1)$. The motivation is theoretical: the $\max$ operator in $\hat{j}^* = \arg\max_j \hat{\tau}_j$ is not Hadamard differentiable at ties, and the standard bootstrap can be inconsistent when two arms are close. Subsampling with $m < N$ smoothes this discontinuity by increasing sampling variability, so the winner identity switches more freely across bootstrap draws.

But this extra variability has a side effect. Because each bootstrap sample has only $m$ observations, the bootstrap treatment effect estimates have variance $\sigma^2 / m$ — larger than the variance $\sigma^2 / N$ of the original estimates. The bootstrap world is noisier than the real world. The winner in each bootstrap draw is more likely to be a noise artifact, and the gap between the bootstrap-winner's value and its cross-evaluated value is inflated. The resulting bias estimate is too large, and subtracting it overcorrects:

$$\text{Corrected value} = \underbrace{\hat{\tau}_{\hat{j}^*}}_{\text{naive}} - \underbrace{\widehat{\text{WC}}_{m\text{-out-of-}n}}_{\text{too large}} \;<\; \tau_{\hat{j}^*}$$

The overcorrection worsens as $\gamma$ decreases (more aggressive subsampling, larger noise gap) and is present across all sample sizes and signal-to-noise ratios.

## Solution: scale by $\sqrt{m/N}$

The fix is to rescale each bootstrap bias draw from the $m$-observation noise level to the $N$-observation noise level:

$$\widehat{\text{WC}}_{\text{scaled}} = \sqrt{\frac{m}{N}} \;\cdot\; \widehat{\text{WC}}_{m\text{-out-of-}n}$$

The corrected estimate becomes:

$$\hat{\tau}_{\hat{j}^*} \;-\; \sqrt{\frac{m}{N}}\;\cdot\;\widehat{\text{WC}}_{m\text{-out-of-}n}$$

## Why it works

The Winner's Curse bias is proportional to the noise level of the estimates. In the A/B test setting with sample means, the estimation error for each arm scales as $\sigma / \sqrt{n}$ where $n$ is the sample size. Therefore:

- At sample size $N$: the WC is proportional to $\sigma / \sqrt{N}$.
- At sample size $m$: the WC is proportional to $\sigma / \sqrt{m}$.

The ratio between the two is:

$$\frac{\text{WC at scale } N}{\text{WC at scale } m} = \frac{\sigma / \sqrt{N}}{\sigma / \sqrt{m}} = \sqrt{\frac{m}{N}}$$

Since $m < N$, this factor is less than 1 and shrinks the correction — exactly what is needed. The scaling converts a bias measurement taken in the noisier bootstrap world back to the noise level of the original data.

### Intuition

Think of the m-out-of-n bootstrap as measuring the Winner's Curse with a magnifying glass: by using fewer observations, it amplifies the noise, making the bias easier to detect but also making it appear larger than it actually is. The $\sqrt{m/N}$ factor is the demagnification that converts the measurement back to the original scale.

## Experimental evidence

We ran a 2D experiment sweeping over sample size ($N \in \{500, 1000, 2500, 5000, 7500, 10000\}$) and signal-to-noise ratio ($\Delta\tau \in \{0.005, 0.01, 0.02\}$), with 1000 repetitions per cell. Results are in `results/ab_test_moon_scaling_20260326_113057/`.

### Key findings

**1. Unscaled moon overcorrects, and the overcorrection grows with smaller $\gamma$.**

At $N = 2500$, $\Delta\tau = 0.01$, mean bias ($\times 10^4$):

| Method | Mean Bias |
|--------|-----------|
| No Correction | +113.8 |
| Standard Bootstrap | +38.3 |
| moon $\gamma = 0.95$ (unscaled) | +12.1 |
| moon $\gamma = 0.80$ (unscaled) | -107.6 |
| moon $\gamma = 0.70$ (unscaled) | -231.7 |

The unscaled $\gamma = 0.70$ overcorrects so aggressively that its bias magnitude exceeds the original Winner's Curse, just with the opposite sign.

**2. Scaling by $\sqrt{m/N}$ largely eliminates the overcorrection.**

Same cell, with scaling applied:

| Method | Mean Bias |
|--------|-----------|
| moon $\gamma = 0.95$ (scaled) | +30.2 |
| moon $\gamma = 0.80$ (scaled) | +12.6 |
| moon $\gamma = 0.70$ (scaled) | +7.0 |

The scaled variants have small positive bias, comparable to the standard bootstrap. In fact, the more aggressive subsampling ($\gamma = 0.70$) produces the *least* biased scaled estimate, suggesting that the smoothing benefit of smaller $m$ is preserved while the noise inflation is corrected.

**3. The pattern holds across sample sizes.**

Mean bias ($\times 10^4$) for $\gamma = 0.80$, $\Delta\tau = 0.01$:

| N | Unscaled | Scaled |
|---|----------|--------|
| 500 | -155.8 | +35.6 |
| 1000 | -137.0 | +21.4 |
| 2500 | -107.6 | +12.6 |
| 5000 | -107.9 | -10.2 |
| 10000 | -83.9 | -7.6 |

The unscaled version overcorrects at every $N$. The scaled version stays close to zero, with a slight tendency toward overcorrection only at large $N$ where the original WC is already small.

**4. At low SNR and small $N$, the scaled moon outperforms the standard bootstrap.**

At $N = 500$, $\Delta\tau = 0.005$, the standard bootstrap has bias +81.0 ($\times 10^4$) while the scaled moon ($\gamma = 0.70$) has bias +25.8. This is the regime where the $\max$ kink matters most, and subsampling's smoothing effect provides genuine benefit — now without the overcorrection penalty.

## Implementation

The scaling is applied per bootstrap draw via a custom `wc_func` in `get_wc_moon_boot` (`src/winners_curse/ab_test.py`). Each $\text{WC}^{(b)}$ is multiplied by $\sqrt{m/N}$ before averaging, which keeps the full bootstrap distribution at the correct scale.

As of the rename, `moon` refers to the scaled algorithm everywhere (A/B, targeting, structural); the original unscaled variant is preserved as `moon_unscaled` for comparison.

Config: `configs/ab_test_moon_scaling.yaml`. Notebook: `notebooks/ab_test_moon.ipynb`.
