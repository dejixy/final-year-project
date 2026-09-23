# XGBoost build — results (152 rows, 3 sites, 51 conditions)

Data: cimmyt_lowland_tropical_merged.csv (Cotzocón Am 65; Pochutla Aw 60; Peto Aw 27).
Features: intake variables + storage technology + physics-derived features (EMC gap, degree-days, humid-days,
open-storage exposure). All 8 end-of-storage columns excluded (leakage).
Hyperparameters: fixed conservative defaults (depth 3, lr 0.03, 400 trees, subsample/colsample 0.8), not tuned —
nested tuning degraded results on the earlier small-site run. XGBoost uses tree_method="exact", n_jobs=1 so results reproduce across machines.
Target: raw % weight loss (log1p tested, worse: RMSE 2.340 vs 2.298).

## Ceiling
Replicate noise floor RMSE = 1.637 pp -> maximum attainable R² ≈ 0.683.

## Primary: condition-grouped 5-fold CV, mean ± SD over 5 seeds
| Model | RMSE | MAE | R² | Bias (obs−pred) |
|---|---|---|---|---|
| **XGBoost** | **2.296 ± 0.149** | **1.294** | **0.374 ± 0.079** | −0.01 |
| GBM (Friedman) | 2.331 ± 0.124 | 1.300 | 0.356 | −0.04 |
| OLS (baseline) | 2.522 ± 0.141 | 1.584 | 0.245 | −0.19 |
| Ridge | 2.543 ± 0.127 | 1.571 | 0.234 | −0.13 |
| Mean predictor | 2.954 ± 0.036 | 2.059 | −0.032 | 0.00 |

XGBoost RMSE 95% bootstrap CI: 1.82–2.80.
Wilcoxon (paired squared errors, Holm-corrected): vs mean p = 0.0002; vs OLS p = 0.048; vs ridge p = 0.091; vs GBM p = 0.28.

## Humid (Am) rows only, same predictions
XGBoost R² 0.336, RMSE 2.31 (mean predictor R² −0.01).

## Storage-technology ranking
Spearman ρ = 0.95 between observed and predicted mean loss across 9 technologies.
Hermetic: observed 0.56 pp, predicted 0.66 pp. Non-hermetic: observed 3.24 pp, predicted 2.99 pp.

## Secondary: leave-one-site-out (3 folds)
XGBoost R² 0.194 (Cotzocón −0.02, Pochutla 0.25, Peto 0.33); GBM 0.124; OLS −0.67; mean −0.03.

## SHAP top features
exposure_open, hermetic, tech_PP, grain impurities, GTI, Hrmin, humid_days_70, RH_mean, Tmin, Hrmax.