# Rootly — Employee Attrition Prediction with Explainable ML

Applied AI course project (Startup Pitch track). I built a tabular ML pipeline that flags employees at risk of leaving
and explains *why* in plain language, so a manager can act on it.

## What it does
Predicts whether an employee will leave, using the public **IBM HR Analytics Attrition** dataset
(1,470 employees, 35 columns, ~16% attrition). Each flagged employee gets a risk score and the top SHAP-based drivers.

## Results (held-out test set, n=294, 47 leavers)
| Model | Precision | Recall | PR-AUC |
|---|---|---|---|
| Naive (predict "stays") | 0.00 | 0.00 | — |
| **Logistic regression** | 0.345 | 0.617 | **0.545** |
| XGBoost (tuned) | 0.377 | 0.426 | 0.466 |

**Finding:** the simple linear model beat every tree ensemble I tried — with ~1.2k training rows, gradient boosting
overfits rather than helps. So logistic regression is the primary model, and its SHAP explanations are exact and cheap.

*(Full pipeline, scripts, and demo dashboard coming in the commits below.)*
