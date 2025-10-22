import streamlit as st
import pandas as pd
import numpy as np
from utils import load_dataset, clean_df

st.set_page_config(page_title="BrickMix Inverse Design", page_icon="🧱", layout="wide")

st.title("🧱 BrickMix Inverse Design (XGBoost)")
st.markdown("""
This app trains **XGBoost** on your data, explains models with **SHAP**, and performs **inverse design** to find **source mixing ratios** that satisfy brick quality constraints.
- **Assumptions**: first column is `Source`; **last 5 columns** are targets.
- Use the sidebar to upload/choose data.
""")

with st.sidebar:
    st.header("Data")
    up = st.file_uploader("Upload your Excel (*.xlsx)", type=["xlsx"])
    if up is not None:
        df = pd.read_excel(up)
    else:
        st.caption("Using sample data packaged in the repo: `Revised_Data_Arnold.xlsx` (tries ./data/ too)")
        # Try multiple likely locations
        from pathlib import Path
        candidates = [
            Path("data/Revised_Data_Arnold.xlsx"),
            Path("Revised_Data_Arnold.xlsx"),
            Path(__file__).parent / "data" / "Revised_Data_Arnold.xlsx",
            Path(__file__).parent / "Revised_Data_Arnold.xlsx",
        ]
        excel_path = None
        for c in candidates:
            if c.exists():
                excel_path = c
                break
        if excel_path is None:
            st.error("Could not find sample Excel. Please upload an .xlsx via the sidebar.")
            st.stop()
        df = pd.read_excel(excel_path)

df = clean_df(df)
st.dataframe(df.head())

# Cache dataset schema in session_state
if "dataset" not in st.session_state:
    st.session_state["dataset"] = df
