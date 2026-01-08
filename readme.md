# ML Week 1 — ML Fundamentals Baseline 

A compact, reproducible ML pipeline for a structured regression task. This project focuses on correct data handling, leakage-free preprocessing, a simple baseline model, and clear evaluation/interpretation of results.

---

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Data & Preprocessing](#data--preprocessing)
- [Modeling & Evaluation](#modeling--evaluation)
- [Experiments & Findings](#experiments--findings)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This repository demonstrates a production-minded ML workflow: split the data before preprocessing, use scikit-learn Pipelines and ColumnTransformer to avoid leakage, and keep experiments reproducible and easy to inspect.

## Quick Start ⚡

Prerequisites:
- Python 3.8+ (this project was developed with Python 3.12)
- pip, virtualenv

Steps:
1. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install required packages:

   ```bash
   pip install pandas numpy scikit-learn
   ```

3. Run training (simple usage):

   ```bash
   python train.py
   ```

The script trains a baseline model and prints evaluation results (RMSE) on the test set.

## Project Structure 📁

```
ml-week1/
├── data/
│   └── data.csv      # raw dataset
├── train.py          # main training script
├── README.md         # this file
└── venv/             # virtual environment (optional)
```

## Data & Preprocessing 🔧

- Data is loaded from `data/data.csv` using pandas.
- Split into train/test sets (80/20) before any preprocessing to avoid leakage.

Preprocessing pipeline (scikit-learn):
- Numerical features: median imputation (SimpleImputer), StandardScaler
- Categorical features: most-frequent imputation, OneHotEncoder(handle_unknown="ignore")
- Implemented with `Pipeline` and `ColumnTransformer` for safety and reproducibility.

## Modeling & Evaluation 📊

- Baseline model: **Linear Regression** — chosen for simplicity and interpretability.
- Metric: **RMSE (Root Mean Squared Error)** on the test set — appropriate for regression and interpretable in target units.
- Model interpretation: map learned coefficients back to original feature names (including one-hot columns) and rank features by absolute coefficient magnitude.

## Experiments & Findings 🧪

- Log-target experiment: trained the model on `log1p(y)` and inverted predictions with `expm1`.
- Observation: RMSE on the original scale increased for this dataset, which highlights trade-offs between optimizing relative vs absolute error.

## Contributing

PRs and issues welcome. Keep changes small and focused; include updated tests or a short description of experiments when adding functionality.

## License

Add a license if needed (e.g., MIT). If you want, I can add a `LICENSE` file for you.

---

If you'd like, I can also:
- add a `requirements.txt`
- add a brief usage example with command-line arguments
- add an example `results/` folder and sample outputs

Feel free to tell me which you'd prefer next! ✨