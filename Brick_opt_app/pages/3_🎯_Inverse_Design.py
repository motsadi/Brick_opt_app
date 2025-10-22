
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution

from utils import load_dataset, source_means, mix_inputs_from_ratios

st.set_page_config(page_title="Inverse Design", page_icon="🎯", layout="wide")
st.title("🎯 Inverse Design: Constrained Optimisation over Source Ratios")

df0 = st.session_state.get("dataset", None)
models = st.session_state.get("models", None)
input_cols = st.session_state.get("train_columns", None)
target_cols = st.session_state.get("targets", None)

if any(x is None for x in [df0, models, input_cols, target_cols]):
    st.warning("Please train models on the **Model** page first.")
    st.stop()

df, source_col, input_cols_check, target_cols_check = load_dataset(df0)
assert input_cols == input_cols_check, "Input columns mismatch with the Model page."
assert target_cols == target_cols_check, "Target columns mismatch with the Model page."

# Let user choose which sources to consider in the mix
src_means = source_means(df, source_col, input_cols)
all_sources = src_means[source_col].tolist()
st.subheader("Select Sources to Mix")
sel_sources = st.multiselect("Pick 2–10 sources for the optimisation", all_sources, default=all_sources[:5])

# WR label and minimum share
wr_label = st.text_input("Waste Rock source label (must match in data)", value="WR")
min_wr = st.slider("Minimum WR share", 0.0, 0.9, 0.3, 0.05)
must_have_wr = st.checkbox("Require WR to be the largest share", value=True)

# Guarantee WR is included if present
if wr_label in all_sources and wr_label not in sel_sources:
    sel_sources = [wr_label] + sel_sources

if len(sel_sources) < 2:
    st.stop()

src_means_sel = src_means[src_means[source_col].isin(sel_sources)].reset_index(drop=True)

st.subheader("Quality Constraints & Objective")
# Map targets
default_map = {
    "Strength (MPa)": None,
    "Water Absorption (%)": None,
    "Rich/Dark Red (score)": None,
    "Linear Shrinkage (%)": None,
    "Porosity (%)": None
}

colmap = {}
for label, current in default_map.items():
    col = st.selectbox(f"Map target for **{label}**:", target_cols, index=0, key=f"map_{label}")
    colmap[label] = col

# Constraints
c1, c2, c3 = st.columns(3)
with c1:
    strength_min = st.number_input("Strength ≥", min_value=0.0, value=20.0, step=0.5)
with c2:
    wa_max = st.number_input("Water Absorption ≤", min_value=0.0, value=12.0, step=0.5)
with c3:
    maximize_color = st.checkbox("Maximise Rich/Dark Red", value=True)

c4, c5 = st.columns(2)
with c4:
    ls_min = st.number_input("Linear Shrinkage min", min_value=0.0, value=3.0, step=0.5)
    ls_max = st.number_input("Linear Shrinkage max", min_value=ls_min, value=8.0, step=0.5)
with c5:
    por_min = st.number_input("Porosity min", min_value=0.0, value=12.0, step=0.5)
    por_max = st.number_input("Porosity max", min_value=por_min, value=22.0, step=0.5)

# Predictor wrapper
def predict_outputs(x_input: pd.Series) -> pd.Series:
    Xdf = pd.DataFrame([x_input], columns=input_cols)
    preds = {}
    for t, m in models.items():
        preds[t] = m.predict(Xdf)[0]
    return pd.Series(preds)

# Objective with penalties (WR constraints + quality constraints)
def objective(ratios_raw):
    ratios = np.maximum(0, ratios_raw)
    if ratios.sum() == 0:
        ratios = np.ones_like(ratios) / len(ratios)
    else:
        ratios = ratios / ratios.sum()

    x_mix = mix_inputs_from_ratios(src_means_sel, ratios, source_col=src_means_sel.columns[0])
    y = predict_outputs(x_mix)

    f_strength = y[colmap["Strength (MPa)"]]
    f_wa = y[colmap["Water Absorption (%)"]]
    f_color = y[colmap["Rich/Dark Red (score)"]]
    f_ls = y[colmap["Linear Shrinkage (%)"]]
    f_por = y[colmap["Porosity (%)"]]

    pen = 0.0
    # Quality penalties
    pen += max(0.0, strength_min - f_strength)**2
    pen += max(0.0, f_wa - wa_max)**2
    pen += max(0.0, ls_min - f_ls)**2 + max(0.0, f_ls - ls_max)**2
    pen += max(0.0, por_min - f_por)**2 + max(0.0, f_por - por_max)**2

    # WR penalties
    if wr_label in src_means_sel[source_col].values:
        idx_wr = list(src_means_sel[source_col].values).index(wr_label)
        r_wr = ratios[idx_wr]
        # minimum share
        pen += max(0.0, min_wr - r_wr)**2 * 100.0
        if must_have_wr:
            # WR must be >= every other share
            diffs = ratios - r_wr
            diffs[idx_wr] = 0.0
            pen += np.sum(np.clip(diffs, 0.0, None)**2) * 500.0  # penalise any other being larger

    # Objective: maximise color by minimising negative color; fall back to WA-Strength if color missing
    obj = 0.0
    if maximize_color and np.isfinite(f_color):
        obj -= f_color
    else:
        obj += f_wa - f_strength
    obj += 1000.0 * pen
    return obj

st.subheader("Optimisation Settings")
pop = st.slider("Population size", 10, 200, 60, 10)
iters = st.slider("Generations", 10, 500, 120, 10)
seed = st.number_input("Random seed", 0, 100000, 42, 1)

do_opt = st.button("Run Optimisation")
if do_opt:
    dim = len(src_means_sel)
    bounds = [(0.0, 1.0)] * dim
    res = differential_evolution(objective, bounds=bounds, maxiter=iters, popsize=max(1, pop//10), seed=seed, polish=True, tol=1e-6)
    ratios = np.maximum(0, res.x)
    ratios = ratios / ratios.sum()

    st.success("Optimisation complete.")
    out = pd.DataFrame({
        "Source": src_means_sel.iloc[:,0],
        "Ratio": ratios
    }).sort_values("Ratio", ascending=False).reset_index(drop=True)

    st.subheader("Optimal Source Ratios (WR constrained)")
    st.dataframe(out.style.format({"Ratio": "{:.3f}"}))

    # Implied composition
    x_mix = mix_inputs_from_ratios(src_means_sel, ratios, source_col=src_means_sel.columns[0])
    st.subheader("Implied Minerals / Elements / Process Parameters (weighted inputs)")
    st.dataframe(pd.DataFrame(x_mix).rename(columns={0:"value"}))

    # Predicted outputs
    y_pred = predict_outputs(x_mix)
    st.subheader("Predicted Outputs")
    st.dataframe(pd.DataFrame(y_pred, columns=["Prediction"]))

    st.download_button("Download ratios as CSV", out.to_csv(index=False).encode("utf-8"), "optimal_ratios.csv", "text/csv")
else:
    st.info("Configure constraints then click **Run Optimisation**.")
