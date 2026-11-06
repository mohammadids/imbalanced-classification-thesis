# Supervised Classification of Unbalanced Datasets

Research code accompanying the Bachelor's thesis by Mohammad Mohammadi.

The project compares Logistic Regression, Random Forest, and XGBoost on the
`satimage`, `ozone_level`, and `abalone_19` binary classification datasets
under five imbalance treatments: original data, random oversampling, random
undersampling, SMOTE, and class weighting.

## Contents

- `src/`: data loading, preprocessing, sampling, models, evaluation, and plots
- `tests/`: integrity and behavior tests
- `notebooks/`: analysis and experiment notebooks with generated outputs removed
- `run_*.py`: validation and experiment entry points
- `requirements.txt`: installable dependencies

Generated results, predictions, logs, archives, and thesis documents are not
included in this code-only repository.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

On macOS, XGBoost may also require the OpenMP runtime:

```bash
brew install libomp
```

## Validation

```bash
.venv/bin/python -m pytest -q --ignore=tests/test_final_artifacts.py
```

The validation scripts load the benchmark datasets through
`imblearn.datasets.fetch_datasets`. The full experiment entry point is
`run_experiments.py`.
