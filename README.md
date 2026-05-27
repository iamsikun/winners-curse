# The Winner's Curse in Data-Driven Decision Making: Evidence and Solutions

This repository contains the code and reproduction materials for the paper *"The Winner's Curse in Data-Driven Decision Making: Evidence and Solution"*.

**📄 [Read the Paper on SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5930537)**

## Overview

Data-driven decision-making often involves selecting the "best" option from a set of alternatives based on noisy estimates (e.g., selecting the best treatment in an A/B test or the best policy in personalized targeting). This selection process introduces a bias known as the **Winner's Curse**: the chosen option's estimated value is likely systematically higher than its true value because the selection rule favors positive estimation errors.

This repository uses simulations to:
1.  **Empirically demonstrate** the magnitude of the winner's curse in common settings (A/B testing and personalized targeting).
2.  **Propose a Bootstrap-based solution** to correct for this bias.
3.  **Compare** the proposed correction against existing methods, including:
    *   Sample Splitting
    *   Empirical Bayes
    *   Selective Inference

## Repository Structure

The project is organized as follows:

*   **`src/winners_curse/`**: The core Python package containing implementations of estimators, the bootstrap correction, and experimental utilities.
*   **`scripts/`**: Entry points for running simulations.
    *   `ab_test_experiment.py`: Runs A/B testing simulations.
    *   `targeting_experiment.py`: Runs personalized targeting (Heterogeneous Treatment Effect) simulations.
*   **`configs/`**: YAML configuration files controlling simulation parameters (sample size, noise levels, number of treatments, etc.).
*   **`notebooks/`**: Jupyter notebooks for data analysis, figure generation, and smaller-scale examples.
*   **`results/`**: (Created at runtime) Stores the output of experiments, including logs and results in CSV/JSON formats.

## Installation

This project uses modern Python packaging standards.

### Prerequisites
*   Python >= 3.13

### Using `uv` (Recommended)
This project contains a `uv.lock` file for reproducible environments.

```bash
# Sync dependencies
uv sync

# Run scripts directly
uv run scripts/ab_test_experiment.py
```

### Using `pip`
You can also install the package in editable mode:

```bash
pip install -e .
```

## Running Experiments

There are two major sets of experiments corresponding to the sections of the paper: **A/B Testing** and **Personalized Targeting**.

### 1. A/B Testing Experiments

The `scripts/ab_test_experiment.py` script simulates A/B tests to evaluate how the winner's curse affects the selection of the best treatment arm and how different estimators perform.

**Usage:**

```bash
python scripts/ab_test_experiment.py --config <path_to_config>
```

**Examples:**

Run the standard Signal-to-Noise Ratio (SNR) experiment:
```bash
python scripts/ab_test_experiment.py --config configs/ab_test_snr.yaml
```

Run an experiment with varying numbers of treatments:
```bash
python scripts/ab_test_experiment.py --config configs/ab_test_n_treatments.yaml
```

**Available Configs (`configs/`):**
*   `ab_test_snr.yaml`: Varies Signal-to-Noise Ratio.
*   `ab_test_n_treatments.yaml`: Varies the number of treatment arms.
*   `ab_test_sample_size.yaml`: Varies sample size.
*   `ab_test_bernoulli.yaml`: Binary outcome simulations.

### 2. Personalized Targeting Experiments

The `scripts/targeting_experiment.py` script simulates personalized policy learning (e.g., using Causal Forests) to evaluate the winner's curse when targeting specific segments of a population.

**Usage:**

```bash
python scripts/targeting_experiment.py --config <path_to_config>
```

**Examples:**

Run the standard targeting experiment:
```bash
python scripts/targeting_experiment.py --config configs/targeting_correct_snr.yaml
```

Run experiments comparing forest depth:
```bash
python scripts/targeting_experiment.py --config configs/targeting_forest_depth.yaml
```

**Available Configs (`configs/`):**
*   `targeting_correct_snr.yaml`: Standard targeting simulation.
*   `targeting_forest_depth.yaml`: Analysis of Causal Forest depth hyperparameters.
*   `targeting_cf_bernoulli.yaml`: Binary outcome targeting.

## Configuration Guide

Experiments are defined by YAML files in the `configs/` directory. These files control:

*   **`dgp_params`**: Data Generating Process settings (e.g., true effect size, noise level).
*   **`experiment_params`**: Simulation settings (e.g., number of repetitions, varying parameters like `tau` or `sample_size`).
*   **`estimators_dict`**: Which estimators to run (e.g., `Naive`, `BootstrapCorrection`, `SampleSplitting`).
*   **`optimization_params`**: How the "winner" is selected (e.g., `select_higher_effect`).

**Example Config Structure:**
```yaml
experiment_params:
  n_simulations: 1000
  varying_params:
    - tau  # The parameter to sweep over
    
dgp_params:
  n_treatments: 10
  
estimators_dict:
  Naive:
    run: true
  BootstrapCorrection:
    run: true
    n_bootstraps: 100
```

## Results and Output

Running an experiment will create a timestamped directory in `results/<experiment_name>/`.

Inside the result folder, you will find:
*   `config.yaml`: A copy of the configuration used.
*   `experiment.log`: Detailed logs of the run.
*   `results.csv` / `results.pkl`: The raw simulation data containing bias, RMSE, and other metrics for each estimator.

To analyze these results, refer to the notebooks in the `notebooks/` directory (e.g., `result_analysis.ipynb` or `visualization.py` in `src`).

## License

MIT License
