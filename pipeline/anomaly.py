"""Anomaly detection: rolling z-score baseline + Isolation Forest primary model."""
import pandas as pd
import numpy as np

def rolling_zscore(series: pd.Series, window: int = 7) -> pd.Series:
    roll_mean = series.rolling(window, min_periods=3).mean()
    roll_std = series.rolling(window, min_periods=3).std()
    return (series - roll_mean) / roll_std

def isolation_forest_flags(df: pd.DataFrame, feature_cols: list[str]):
    from sklearn.ensemble import IsolationForest
    model = IsolationForest(contamination=0.05, random_state=42)
    df = df.copy()
    df["anomaly_score"] = model.fit_predict(df[feature_cols])
    return df
