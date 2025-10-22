
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import shap

def load_dataset(df: pd.DataFrame):
    """
    Expects: first column = Source (categorical), last 5 columns = targets.
    Returns: df, source_col, input_cols, target_cols
    """
    assert df.shape[1] >= 7, "Dataset must have at least 7 columns: Source + inputs + 5 targets."
    df = df.copy()
    source_col = df.columns[0]
    target_cols = df.columns[-5:].tolist()
    input_cols = df.columns[1:-5].tolist()
    return df, source_col, input_cols, target_cols

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop rows that are fully NA in inputs/targets
    df = df.dropna(how="all")
    # Forward-fill / back-fill minor gaps if any
    df = df.fillna(method="ffill").fillna(method="bfill")
    return df

def split_data(X, Y, test_size=0.2, random_state=42):
    return train_test_split(X, Y, test_size=test_size, random_state=random_state)

def train_xgb_multioutput(X: pd.DataFrame, Y: pd.DataFrame, params: dict=None, random_state: int=42):
    """
    Trains one XGBRegressor per target.
    Returns dict of {target: model}
    """
    if params is None:
        params = dict(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=random_state,
        )
    models = {}
    for col in Y.columns:
        model = XGBRegressor(**params)
        model.fit(X, Y[col])
        models[col] = model
    return models

def predict_multi(models: Dict[str, XGBRegressor], X: pd.DataFrame) -> pd.DataFrame:
    preds = {}
    for t, m in models.items():
        preds[t] = m.predict(X)
    return pd.DataFrame(preds, index=X.index)

def cv_metrics(models, X, Y):
    # Simple holdout metrics using training fit (for quick diagnostics)
    preds = predict_multi(models, X)
    out = []
    for t in Y.columns:
        r2 = r2_score(Y[t], preds[t])
        rmse = mean_squared_error(Y[t], preds[t], squared=False)
        mae = mean_absolute_error(Y[t], preds[t])
        out.append(dict(target=t, R2=r2, RMSE=rmse, MAE=mae))
    return pd.DataFrame(out)

def shap_values_for_target(model: XGBRegressor, X_sample: pd.DataFrame, max_display=15):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)
    return explainer, sv

def source_means(df: pd.DataFrame, source_col: str, input_cols: List[str]) -> pd.DataFrame:
    """Compute mean input composition per Source."""
    return df.groupby(source_col)[input_cols].mean().reset_index()

def mix_inputs_from_ratios(source_means_df: pd.DataFrame, ratios: np.ndarray, source_col: str) -> pd.Series:
    """Weighted average of inputs given ratios aligned with source_means_df rows."""
    A = source_means_df.drop(columns=[source_col]).values  # shape (S, F)
    x = (ratios.reshape(-1,1) * A).sum(axis=0)
    return pd.Series(x, index=source_means_df.drop(columns=[source_col]).columns)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def make_bounds_for_params(X: pd.DataFrame, fixed: Dict[str, Optional[float]]=None, widen: float=0.05):
    """
    For each column in X, return (low, high).
    If a param is fixed (value provided), we keep it fixed by returning (val,val).
    """
    bounds = {}
    for c in X.columns:
        lo = X[c].min()
        hi = X[c].max()
        span = hi - lo
        lo -= widen*span
        hi += widen*span
        if fixed and (c in fixed) and (fixed[c] is not None):
            bounds[c] = (fixed[c], fixed[c])
        else:
            bounds[c] = (lo, hi)
    return bounds
