"""Isolation Forest on a multi-feature daily matrix: total cost, spend
across the top N categories (rest bucketed into 'Other'), and day-of-week.

Capped at top-N categories rather than all raw categories -- with ~69 daily
samples, using all ~49 raw category columns causes curse-of-dimensionality
noise (most categories are sparse/near-zero most days, so the model partly
isolates on that noise instead of real spend patterns). Capping keeps the
feature space meaningfully smaller than the sample size while still giving
the model real multivariate signal to work with.
"""
import pandas as pd
from sklearn.ensemble import IsolationForest


def daily_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per day: total cost. Still used for the chart's
    line/y-axis regardless of how many features the model itself uses."""
    return (
        df.groupby("Date", as_index=False)["CostInBillingCurrency"]
        .sum()
        .sort_values("Date")
        .reset_index(drop=True)
    )


def build_daily_feature_matrix(df: pd.DataFrame, top_n_categories: int = 8) -> pd.DataFrame:
    """Builds a per-day feature matrix: total cost, cost per top-N category
    (bucketing the rest into 'Other'), and day-of-week."""
    top_categories = (
        df.groupby("MeterCategory")["CostInBillingCurrency"].sum()
        .sort_values(ascending=False).head(top_n_categories).index
    )

    work = df.copy()
    work["CategoryBucket"] = work["MeterCategory"].where(
        work["MeterCategory"].isin(top_categories), "Other"
    )

    pivot = (
        work.groupby(["Date", "CategoryBucket"])["CostInBillingCurrency"]
        .sum()
        .unstack(fill_value=0.0)
    )
    pivot.columns = [f"cost_{c.replace(' ', '_')}" for c in pivot.columns]

    daily_total = df.groupby("Date")["CostInBillingCurrency"].sum().rename("total_cost")

    features = pivot.join(daily_total).reset_index()
    features["day_of_week"] = features["Date"].dt.dayofweek
    features = features.sort_values("Date").reset_index(drop=True)
    return features


def flag_anomalies(features: pd.DataFrame, contamination: float = 0.07) -> pd.DataFrame:
    """Fits Isolation Forest on the full multi-feature matrix (all columns
    except Date). contamination=0.07 = starting assumption that ~7% of days
    are unusual a tunable business decision, not a fixed truth.
    """
    feature_cols = [c for c in features.columns if c != "Date"]
    X = features[feature_cols].values

    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    pred = model.fit_predict(X)  # -1 = anomaly, 1 = normal
    scores = model.decision_function(X)

    out = features[["Date", "total_cost"]].copy()
    out["is_anomaly"] = pred == -1
    out["anomaly_score"] = scores
    return out.sort_values("Date").reset_index(drop=True)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ingest import load_all
    from clean import clean

    df = load_all()
    df = clean(df)

    features = build_daily_feature_matrix(df)
    flagged = flag_anomalies(features)

    print(f"Feature matrix shape: {features.shape}")
    print(f"Feature columns: {[c for c in features.columns if c != 'Date']}")
    print()
    print(flagged[flagged["is_anomaly"]])
    print(f"\n{flagged['is_anomaly'].sum()} anomalous day(s) out of {len(flagged)}")