# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research codebase for the paper "The Winner's Curse in Data-Driven Decision Making: Evidence and Solution" ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5930537)). Uses simulations to demonstrate the Winner's Curse bias and proposes a Bootstrap-based correction, compared against Sample Splitting, Empirical Bayes, and Selective Inference.

Two experiment types: **A/B Testing** (selecting best treatment arm) and **Personalized Targeting** (HTE estimation via Causal Forests).

## Project memory

When the user refers to "the paper", they mean `~/research/winners-curse-paper` (`/Users/iamsikun/research/winners-curse-paper`). Use that separate repository for paper-related work.

## Commands

```bash
# Setup
uv sync

# Run experiments
uv run python scripts/ab_test_experiment.py --config configs/ab_test_snr.yaml
uv run python scripts/targeting_experiment.py --config configs/targeting_correct_snr.yaml

# Tests
uv run pytest

# Notebooks
uv run jupyter lab
```

## Architecture

### Core flow
1. **Config** (YAML in `configs/`) defines DGP params, estimators, and comparative statics
2. **Scripts** (`scripts/ab_test_experiment.py`, `scripts/targeting_experiment.py`) load config, run parameter sweeps, save results to timestamped `results/` directories
3. **Experiment infra** (`src/winners_curse/experiments.py`) provides shared config loading, logging, output directory management, and parameter sweep logic
4. **DGP** (`dgp.py`) — `ABTest` and `Targeting` classes generate synthetic data
5. **Estimators** (`ab_test.py`, `targeting.py`) implement correction methods: bootstrap, sample splitting, empirical Bayes, selective inference, jackknife, plugin, k-fold CV
6. **Bootstrap framework** (`bootstrap.py`) is a generic estimator→optimizer→evaluator pipeline used by both experiment types
7. **Analysis** (`analysis.py`) computes summary stats and generates LaTeX tables; notebooks in `notebooks/` for interactive exploration

### Key design decisions
- **Max 2 varying parameters** in comparative statics sweeps — scripts error if more than 2 parameter lists vary simultaneously
- **Result keys** adapt to what varies: 1 param → simple keys (e.g., tau tuples), 2 params → composite keys (e.g., `(depth, tau)`)
- **Parallelism**: outer loop uses joblib `Parallel`; inner bootstrap jobs typically set `n_jobs=1` to avoid nested contention
- **Memory-tiered workers**: each worker holds a full training sample, so `parallel.max_jobs_by_sample_size` (keys = smallest `sample_size` an entry applies to, with a measured `gb_per_job`) lowers `parallel.max_jobs` per combo and re-caps it against free memory at launch; `experiment_params.n_repeats_by_sample_size` tiers repeats the same way
- **OpenBLAS threading** must be disabled (`OPENBLAS_NUM_THREADS=1` etc.) before numpy imports — scripts do this at top of file
- **Random variables** (`variables.py`) are config-driven: YAML specifies `type: UnivariateGaussian` with params, which gets instantiated into `RandomVariable` subclasses

### Docstring style
NumPy-style (`Params:`, `Returns:`).
