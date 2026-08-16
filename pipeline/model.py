"""ONE model: Isolation Forest trained on daily total cost only.
Deliberately single-feature -- simple to run, simple to explain, no
hidden complexity to defend under questioning.
"""
import pandas as pd
from sklearn.ensemble import IsolationForest


def daily_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per day: total cost."""
    return (
        df.groupby("Date", as_index=False)["CostInBillingCurrency"]
        .sum()
        .sort_values("Date")
        .reset_index(drop=True)
    )


def flag_anomalies(daily_df: pd.DataFrame, contamination: float = 0.07) -> pd.DataFrame:
    """Fits Isolation Forest on daily total cost (single feature).
    contamination=0.07 = starting assumption that ~7% of days are unusual;
    a tunable business decision, not a fixed truth.
    """
    out = daily_df.copy()
    X = out[["CostInBillingCurrency"]].values

    model = IsolationForest(contamination=contamination, random_state=42)
    pred = model.fit_predict(X)  # -1 = anomaly, 1 = normal
    out["is_anomaly"] = pred == -1
    out["anomaly_score"] = model.decision_function(X)
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ingest import load_all
    from clean import clean

    df = load_all()
    df = clean(df)

    daily = daily_totals(df)
    flagged = flag_anomalies(daily)

    print(flagged)
    print(f"\n{flagged['is_anomaly'].sum()} anomalous day(s) out of {len(flagged)}")
