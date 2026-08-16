# Multiple Linear Regression from Scratch — Diabetes Blood Pressure Prediction

A Multiple Linear Regression model built from scratch using the **Normal Equation**
(closed-form solution — no gradient descent, no learning rate), predicting blood
pressure from multiple patient features in the sklearn Diabetes dataset. Validated
against scikit-learn's `LinearRegression`.

This project follows on from a single-feature (BMI-only) regression baseline, to show
how much predictive power multiple real features add over one.

## Model

`MultiLinearRegression` (`MultiLinearRegression.py`) solves for all coefficients in a
single step:

```
θ = (XᵀX)⁻¹ Xᵀy
```

No iteration, no convergence tuning — unlike Gradient Descent, the exact
least-squares solution is computed directly.

## Data & Features

- **Dataset:** sklearn's built-in Diabetes dataset (442 patients, 10 baseline features)
- **Target:** `bp` (blood pressure)
- **Features used:** `bmi`, `triglycerides`, `glucose`, `age` — the four real,
  clinically available features with the strongest correlation to `bp`
- **Excluded:** the dataset's `target` column (disease progression score) — that's
  the label for a *different* prediction task, not something available when
  predicting blood pressure in practice. Including it would be data leakage.
- **Split:** 80/20 train/test (`random_state=42`), matching the earlier SLR and
  Gradient Descent projects for a fair R² comparison

## Results

| Model | Features | R² |
|---|---|---|
| SLR (previous project) | BMI only | 0.233 |
| **MLR (this project)** | bmi, triglycerides, glucose, age | **0.374** |

| Metric | Value |
|---|---|
| MAE | 0.0270 |
| MSE | 0.0012 |
| RMSE | 0.0346 |
| R² | 0.3742 |

The scratch implementation's coefficients and R² match scikit-learn's
`LinearRegression` exactly, confirming the Normal Equation is implemented correctly.

## Takeaway

Going from one feature (BMI) to four real, non-leaky features raised R² from 0.233
to 0.374 — blood pressure here is explained by a combination of factors, not any
single one. This is the core motivation for multiple regression over simple
regression: most real-world outcomes are multi-causal.

## Files

- `main.ipynb` — full walkthrough: data loading, EDA, feature selection, model
  fitting, evaluation, and validation against scikit-learn
- `MultiLinearRegression.py` — the from-scratch model implementation

## Running it

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook main.ipynb
```