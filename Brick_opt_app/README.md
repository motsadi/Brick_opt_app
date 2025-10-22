# BrickMix Inverse Design (XGBoost + Streamlit)

This app trains **XGBoost** models (one per target) on your dataset, explains them with **SHAP**, and performs **inverse design** to find optimal *source mixing ratios* and (optionally) process parameters that satisfy user‑defined quality constraints for bricks.

## Key features
- **Auto-detect schema**: first column is treated as `Source`; **last 5 columns** are outputs/targets.
- **XGBoost-only** modeling with train/validation/test split.
- **EDA**: summary stats, correlations, scatter matrix.
- **Model diagnostics**: parity plots, cross-validated metrics, feature importances.
- **Explainability**: SHAP summary (beeswarm) & per-feature dependence for each target.
- **Inverse design / optimisation**:
  - Decision variables: **ratios for Sources** (nonnegative, sum to 1). Optionally allow optimisation of selected process parameters (per bounds).
  - Constraints (defaults; editable in UI):
    - Strength ≥ 20 MPa
    - Water Absorption ≤ 12 %
    - Linear Shrinkage within [3 %, 8 %] (default; you can change)
    - Porosity within [12 %, 22 %] (default; you can change)
    - **Maximise** Rich/Dark Red (higher is better)
  - Optimiser: **scipy.optimize.differential_evolution** with penalties for constraint violations.
  - Result shows predicted outputs and the implied **minerals/elements** composition from the chosen ratios.

## Project structure
```
brick_opt_app/
  ├── streamlit_app.py
  ├── utils.py
  ├── requirements.txt
  ├── README.md
  ├── data/
  │   └── Revised_Data_Arnold.xlsx 
  └── pages/
      ├── 1_📊_EDA.py
      ├── 2_🤖_Model_XGBoost.py
      └── 3_🎯_Inverse_Design.py




---

**Notes**
- The app detects inputs vs outputs automatically: **inputs = all columns except the first and the last 5; outputs = last 5**.
- Categorical **Source** column is required and must be the **first** column.
- If you change the number of targets, update the "Targets (last 5 columns)" note in the UI sidebar accordingly.
