
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

from utils import load_dataset, clean_df, train_xgb_multioutput, predict_multi, cv_metrics, shap_values_for_target

st.set_page_config(page_title="Model", page_icon="🤖", layout="wide")
st.title("🤖 XGBoost Modeling & SHAP")

df = st.session_state.get("dataset", None)
if df is None:
    st.warning("No dataset found in session. Please go to the Home page and load data.")
    st.stop()

df, source_col, input_cols, target_cols = load_dataset(df)

st.sidebar.subheader("Train / Test Split")
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random seed", 0, 10_000, 42, 1)

X = df[input_cols].copy()
Y = df[target_cols].copy()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

st.write(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

st.sidebar.subheader("XGBoost Hyperparameters (shared)")
n_estimators = st.sidebar.slider("n_estimators", 100, 1500, 600, 50)
learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.3, 0.05, 0.01)
max_depth = st.sidebar.slider("max_depth", 2, 12, 6, 1)
subsample = st.sidebar.slider("subsample", 0.5, 1.0, 0.9, 0.05)
colsample_bytree = st.sidebar.slider("colsample_bytree", 0.5, 1.0, 0.9, 0.05)
reg_alpha = st.sidebar.slider("reg_alpha", 0.0, 2.0, 0.0, 0.1)
reg_lambda = st.sidebar.slider("reg_lambda", 0.0, 3.0, 1.0, 0.1)

params = dict(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    subsample=subsample,
    colsample_bytree=colsample_bytree,
    reg_alpha=reg_alpha,
    reg_lambda=reg_lambda,
    random_state=random_state
)

# Train models (one per target)
from utils import train_xgb_multioutput, predict_multi
models = train_xgb_multioutput(X_train, Y_train, params=params, random_state=random_state)

# Evaluate on test
Y_pred_test = predict_multi(models, X_test)

metrics = []
for t in target_cols:
    r2 = r2_score(Y_test[t], Y_pred_test[t])
    rmse = mean_squared_error(Y_test[t], Y_pred_test[t], squared=False)
    mae = mean_absolute_error(Y_test[t], Y_pred_test[t])
    metrics.append(dict(Target=t, R2=r2, RMSE=rmse, MAE=mae))

st.subheader("Test Metrics")
st.dataframe(pd.DataFrame(metrics).round(4))

st.subheader("Parity Plots")
tabs = st.tabs(target_cols)
for tab, t in zip(tabs, target_cols):
    with tab:
        fig = px.scatter(x=Y_test[t], y=Y_pred_test[t], labels={'x':'True', 'y':'Predicted'}, title=f"Parity: {t}")
        fig.add_shape(type='line', x0=float(Y_test[t].min()), y0=float(Y_test[t].min()),
                      x1=float(Y_test[t].max()), y1=float(Y_test[t].max()))
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Feature Importance (gain) per target")
tabs2 = st.tabs(target_cols)
for tab, t in zip(tabs2, target_cols):
    with tab:
        booster = models[t].get_booster()
        score = booster.get_score(importance_type='gain')
        imp = pd.DataFrame({'feature': list(score.keys()), 'gain': list(score.values())}).sort_values('gain', ascending=False)
        st.dataframe(imp)

st.subheader("SHAP Explanations")
st.caption("Using TreeExplainer on a sample of the training set for speed.")
sample_n = st.slider("Rows for SHAP sample", 50, min(1000, len(X_train)), min(300, len(X_train)), 50)
X_sample = X_train.sample(sample_n, random_state=random_state)
tabs3 = st.tabs(target_cols)
for tab, t in zip(tabs3, target_cols):
    with tab:
        model = models[t]
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sample)
        st.write("**Summary (beeswarm)**")
        fig = shap.plots.beeswarm(shap.Explanation(values=sv, base_values=explainer.expected_value, data=X_sample.values, feature_names=X_sample.columns), show=False)
        st.pyplot(fig, clear_figure=True)
        st.write("**Dependence plot (top feature)**")
        # pick top feature by mean |shap|
        mean_abs = np.abs(sv).mean(axis=0)
        top_idx = int(np.argmax(mean_abs))
        top_feat = X_sample.columns[top_idx]
        fig2 = shap.plots.scatter(shap.Explanation(values=sv[:, top_idx], base_values=explainer.expected_value, data=X_sample[top_feat].values, feature_names=[top_feat]), show=False)
        st.pyplot(fig2, clear_figure=True)

st.success("Models trained and diagnostics generated. Move to **Inverse Design** to optimise mixes.")
st.session_state["models"] = models
st.session_state["train_columns"] = input_cols
st.session_state["targets"] = target_cols
st.session_state["X_source_df"] = df[[source_col] + input_cols].copy()
