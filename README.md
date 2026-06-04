# NYC Fatal Crash Risk Analysis — Ford vs Non-Ford

## Overview
Logistic regression analysis on **65,096 NYC crash records** to quantify whether Ford vehicles are statistically associated with higher fatal crash outcomes, after controlling for vehicle type, driver demographics, licensing status, and situational factors.

This project demonstrates end-to-end statistical inference: from data cleaning and feature engineering to model evaluation, sub-sample analysis, and business recommendations.

---

## Key Findings

| Finding | Result |
|---|---|
| Main Ford effect | β = 0.869, **p = 0.046** (statistically significant) |
| Odds Ratio | OR ≈ 2.38 — Ford vehicles have ~2.4× higher odds of fatal crash |
| Female drivers | β = 1.604, **p = 0.044** — effect concentrated among female drivers |
| Newer vehicles | β = 1.053, p = 0.072 — borderline significant among vehicles 0–5 years old |
| Model ROC-AUC | 0.7508 — fair discriminative ability given extreme class imbalance |

---

## Technical Highlights

- **Feature Engineering** — 15+ features: Ford indicator, vehicle age (crash year − manufacturing year), vehicle-type dummies, license status, time-of-day flags
- **Class Imbalance** — 37 fatal crashes out of 65,096 (0.057%); no rebalancing applied to preserve real-world distribution for valid coefficient interpretation
- **Model Evaluation** — ROC-AUC, optimal threshold selection via ROC curve, confusion matrix comparison at default vs optimal threshold
- **Sub-Sample Analysis** — 7 sensitivity models stratified by vehicle type, driver sex, and vehicle age to test for heterogeneous effects
- **Feature Importance** — Coefficient-based visualization of all 15 predictors
- **Business Recommendations** — 4 actionable safety recommendations for Ford management

---

## Visualizations

### Feature Coefficients
![Feature Importance](outputs/feature_importance.png)

### Ford Effect Across Sub-Sample Models
![Ford Coefficient Sub-samples](outputs/ford_coef_subsamples.png)

### Fatality Rate by Vehicle Type
![Fatality by Vehicle Type](outputs/fatality_by_vehicle_type.png)

### Model Evaluation — ROC Curve
![ROC Curve](outputs/roc_curve.png)

---

## Project Structure