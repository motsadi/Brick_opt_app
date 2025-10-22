
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils import load_dataset, clean_df

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")
st.title("📊 Exploratory Data Analysis")

df = st.session_state.get("dataset", None)
if df is None:
    st.warning("No dataset found in session. Please go to the Home page and load data.")
    st.stop()

df, source_col, input_cols, target_cols = load_dataset(df)

st.subheader("Schema")
c1, c2 = st.columns(2)
with c1:
    st.write("**Source column**:", source_col)
with c2:
    st.write("**Targets (last 5 columns)**:", target_cols)

st.subheader("Summary")
st.write(df.describe(include='all'))

st.subheader("Correlation Heatmap (inputs vs targets)")
corr = df[input_cols + target_cols].corr()
fig = px.imshow(corr, aspect="auto", title="Correlation Matrix")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Scatter Matrix (selected columns)")
sel_cols = st.multiselect("Choose up to 8 columns", input_cols + target_cols, default=target_cols[:3] + input_cols[:5])
sel_cols = sel_cols[:8]
if len(sel_cols) >= 2:
    fig2 = px.scatter_matrix(df[sel_cols])
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Select at least two columns for a scatter matrix.")

st.markdown("---")
st.header("🧭 Principal Component Analysis (PCA) on Inputs")
st.caption("We standardise input features, then compute PCA. View scree plot, loadings, and a Source-coloured biplot.")

# Standardise inputs
X = df[input_cols].copy().astype(float)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# PCA
n_comp = st.slider("Number of components", 2, min(len(input_cols), 12), min(5, len(input_cols)), 1)
pca = PCA(n_components=n_comp, random_state=42)
X_pca = pca.fit_transform(X_std)

# Scree plot
exp_var = pca.explained_variance_ratio_
scree_df = pd.DataFrame({"PC": [f"PC{i+1}" for i in range(len(exp_var))], "ExplainedVariance": exp_var})
fig_scree = px.bar(scree_df, x="PC", y="ExplainedVariance", title="Scree Plot (Explained Variance Ratio)")
st.plotly_chart(fig_scree, use_container_width=True)

# Loadings
loadings = pd.DataFrame(pca.components_.T, index=input_cols, columns=[f"PC{i+1}" for i in range(len(exp_var))])
st.subheader("PCA Loadings (feature weights per PC)")
st.dataframe(loadings.style.format("{:.3f}"))

# Biplot (PC1 vs PC2)
if n_comp >= 2:
    st.subheader("Biplot (PC1 vs PC2)")
    bi_df = pd.DataFrame(X_pca[:, :2], columns=["PC1", "PC2"])
    bi_df[source_col] = df[source_col].values
    fig_bi = px.scatter(bi_df, x="PC1", y="PC2", color=source_col, title="Scores Scatter (by Source)")
    st.plotly_chart(fig_bi, use_container_width=True)

    st.caption("Feature loading vectors (scaled for display).")
    # Simple vector table for loadings on PC1/PC2
    vec = loadings[["PC1", "PC2"]].copy()
    st.dataframe(vec.style.format("{:.3f}"))
else:
    st.info("Increase components to at least 2 to see the biplot.")
