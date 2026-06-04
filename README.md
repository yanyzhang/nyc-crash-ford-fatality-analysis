# Vehicle Fatality Risk Analysis — Ford vs Non-Ford

## Overview
Logistic regression analysis on 65,096 NYC crash records 
to quantify whether Ford vehicles are associated with higher 
fatal crash outcomes, after controlling for vehicle type, 
driver demographics, and situational factors.

**Key Findings:**
- Ford vehicles show statistically significant higher fatal 
  crash risk (β = 0.869, p = 0.046)
- Effect concentrated among female drivers (β = 1.604, p = 0.044)
- Effect potentially stronger among newer vehicles (β = 1.053, p = 0.072)

## Tools
Python | Statsmodels | Pandas | Scikit-learn | Matplotlib

## Data Source
NYC Open Data — Motor Vehicle Collisions (NYPD)  
https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95

## Files
- `ford_fatality_analysis.ipynb` — Full analysis notebook
- `Ford_on_death_rate.py` — Analysis script
- `outputs/` — Generated visualizations
- `report/` — Final written report

## Key Visualizations
![Feature Importance](outputs/feature_importance.png)
![Ford Coefficient Across Models](outputs/ford_coef_subsamples.png)