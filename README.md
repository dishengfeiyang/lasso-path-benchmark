# Lasso and Group Lasso Regularization Path Benchmark

This project implements and compares a variety of **Lasso** and **Group Lasso** regularization path methods, providing a unified interface, standardized benchmarking, result visualization, and scalability analysis. The goal is to build a clean, reproducible, and easily extensible framework for algorithm evaluation.

---

## Features

- **Unified interface**: All methods inherit from `BasePathMethod` and implement `fit(X, y, lambdas, groups=None)`, returning a dictionary containing coefficient paths, objective values, sparsity levels, timing, and the lambda sequence.
- **Broad method coverage**: Includes classic methods (LARS, coordinate descent, ADMM) and newer methods (Celer, FISTA, Screening CD, Gap Safe CD, Anderson CD, Elastic Net, Block CD, Group LARS).
- **Rich benchmarking**: Covers real and synthetic datasets, with support for custom sample size, dimensionality, correlation, group structure, etc.
- **Complete result storage**: Summary metrics saved as CSV, full paths saved as `.npz` for further analysis.
- **Scalability experiments**: Analyze time and memory as a function of sample size, feature dimension, and number of path points.
- **Automated tests**: Use `pytest` to verify interface consistency and numerical correctness.

---

## Directory Structure

    .
    ├── methods/                 # Implementation of all methods, one file per method
    │   ├── _base.py             # Abstract base class
    │   ├── lars.py              # LARS
    │   ├── coordinate_descent.py# Coordinate descent
    │   ├── admm.py              # ADMM
    │   ├── celer.py             # Celer
    │   ├── fista.py             # FISTA
    │   ├── screening_cd.py      # Strong rules + coordinate descent
    │   ├── gap_safe_cd.py       # Gap safe screening + coordinate descent
    │   ├── anderson_cd.py       # Anderson accelerated coordinate descent
    │   ├── elastic_net.py       # Elastic Net
    │   ├── block_cd.py          # Block coordinate descent (Group Lasso)
    │   └── group_lars.py        # Group LARS
    ├── benchmarks/              # Benchmarking scripts and configuration
    │   ├── run_benchmarks.py    # Main benchmark runner
    │   ├── run_scaling.py       # Scalability experiment script
    │   ├── config.yaml          # Experiment configuration (methods, datasets, lambda settings)
    │   ├── metrics.py           # Evaluation metrics
    │   └── utils.py             # Utility functions (lambda sequence generation)
    ├── data/                    # Data module
    │   ├── synthetic.py         # Synthetic data generator
    │   ├── real_datasets.py     # Real dataset loaders
    │   └── groups/              # Group information (auto-generated)
    ├── tests/                   # Automated tests
    │   ├── test_methods.py
    │   └── test_data.py
    ├── outputs/                 # Experiment results (auto-generated)
    │   ├── tables/              # Summary CSV files
    │   ├── results/             # Detailed path .npz files
    │   └── figures/             # Figures
    ├── scripts/
    │   └── plot_results.py      # Plotting script
    ├── environment.yml          # Conda environment configuration
    ├── requirements.txt         # Pip dependencies
    ├── .gitignore
    └── README.md

---

## Installation

### Using Conda (recommended)

    conda env create -f environment.yml
    conda activate lasso-benchmark

### Using pip

    pip install -r requirements.txt

---

## Quick Start

### 1. Run Benchmark Suite

    python -m benchmarks.run_benchmarks

This command will:
- Load the datasets and methods specified in `benchmarks/config.yaml`.
- Generate an adaptive lambda sequence for each dataset.
- Run all methods, recording time, memory, objective value, sparsity, etc.
- Save summary results to `outputs/tables/results.csv`.
- Save detailed coefficient paths to `outputs/results/*.npz`.

### 2. Run Scalability Experiments

    python -m benchmarks.run_scaling

This script varies sample size, feature dimension, and number of path points, recording time and memory for three representative methods. Results are saved to `outputs/tables/scaling_results.csv`.

### 3. Generate Figures

    python scripts/plot_results.py

Figures will be saved to `outputs/figures/`, including:
- Time and memory comparison bar charts (log scale)
- Objective value facet plots (Lasso / Group Lasso / Elastic Net)
- Final sparsity comparison
- Screening ratio plot
- Path accuracy plot (relative to LARS or coordinate descent)
- Scalability curves
---

## Dataset Description

| Dataset | Type | Samples | Features | Groups | Source |
|---------|------|---------|----------|--------|--------|
| California Housing | Real | ~20640 | 8 | None | scikit-learn |
| Diabetes | Real | 442 | 10 | None | scikit-learn |
| 20 Newsgroups | Real | ~18000 | 500 (TF-IDF) | Evenly divided into 20 groups | scikit-learn |
| Olivetti Faces | Real | 400 | 4096 | 8×8 spatial blocks | scikit-learn |
| synth_n100_p50 | Synthetic | 100 | 50 | 10 groups | Custom |
| synth_n500_p200_highcorr | Synthetic | 500 | 200 | 20 groups, high correlation | Custom |

- Real datasets are automatically downloaded on first use and cached in `data_cache/`.
- Group information is generated and saved to `data/groups/` for reproducibility.

---

## Adding a New Method

1. Create a new Python file under `methods/`, inheriting from `BasePathMethod`.
2. Implement the `fit(X, y, lambdas, groups=None)` method, returning a dictionary containing the following keys:
   - `coef_path` : shape `(n_features, n_lambdas)`
   - `objective` : shape `(n_lambdas,)`
   - `sparsity` : shape `(n_lambdas,)`
   - `timing`   : shape `(n_lambdas,)`
   - `lambdas`  : shape `(n_lambdas,)`
3. Register the class in the `METHODS` dictionary in `benchmarks/run_benchmarks.py`.
4. If you want to include the method in scalability experiments, add it to `benchmarks/run_scaling.py` accordingly.
5. Rerun the benchmark suite to automatically compare the new method.

Example (pseudocode):

    from methods._base import BasePathMethod

    class MyMethod(BasePathMethod):
        def __init__(self):
            super().__init__("MyMethod")

        def fit(self, X, y, lambdas, groups=None):
            # Implement path solver
            return {
                "coef_path": ...,
                "objective": ...,
                "sparsity": ...,
                "timing": ...,
                "lambdas": ...
            }

---

## Testing

Run all tests:

    pytest tests/

Test coverage includes:
- Interface consistency: all methods return required keys.
- Output shape correctness.
- Lasso method objective values within 5% relative error compared to reference (coordinate descent).
- Synthetic data generator shapes.
- Real dataset loaders return correct formats.

---

## Reproducibility

- Full experiment configuration is in `benchmarks/config.yaml`; you can adjust method list, dataset parameters, and lambda settings.
- Lambda sequence is generated adaptively: `λ_max` is computed from the data, and the minimum lambda is `ratio * λ_max` (currently `ratio = 0.01`).
- Random seeds are fixed for reproducibility (e.g., synthetic data `random_state=42`).

---

## Technical Report

The technical report will integrate method descriptions, experimental setup, result figures, and discussion. Please refer to the project documentation or submitted version.

---

## License

This project is for academic research and teaching purposes only. Please comply with the respective dataset licenses when using.