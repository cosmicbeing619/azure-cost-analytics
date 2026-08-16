"""Azure Cost Anomaly Dashboard — Streamlit version.

Runs the pipeline live (ingest -> clean -> model -> forecast) every time
data changes, using Streamlit's cache so it doesn't recompute on every
UI interaction, only when the underlying data actually changes.
"""
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import altair as alt

sys.path.insert(0, str(Path(__file__).resolve().parent / "pipeline"))
from ingest import load_all
from clean import clean
from model import daily_totals, build_daily_feature_matrix, flag_anomalies
from forecast import forecast_daily_cost

st.set_page_config(page_title="Azure Cost Anomaly Dashboard", layout="centered")


@st.cache_data
def get_pipeline_output(_data_dir_signature: str):
    """_data_dir_signature is just a cache key (see below) -- forces
    Streamlit to recompute when the files in /data actually change,
    not on every rerun."""
    df = load_all()
    df = clean(df)
    daily = daily_totals(df)
    features = build_daily_feature_matrix(df)
    flagged = flag_anomalies(features)
    forecast_df = forecast_daily_cost(daily, horizon_days=14)
    return df, flagged, forecast_df


def data_signature() -> str:
    """Cheap cache-busting signature: filenames + sizes of everything in /data.
    Changes whenever a file is added/modified, which invalidates the cache
    above and forces a fresh pipeline run."""
    data_dir = Path(__file__).resolve().parent / "data"
    files = sorted(data_dir.glob("*.csv"))
    return "|".join(f"{f.name}:{f.stat().st_size}" for f in files)


# ---------- Load ----------
df, flagged, forecast_df = get_pipeline_output(data_signature())

# ---------- Header ----------
st.title("Azure Cost Anomaly Dashboard")
st.caption(
    f"{len(df):,} rows · {df['Date'].min().date()} → {df['Date'].max().date()} "
    f"· {int(flagged['is_anomaly'].sum())} anomalous day(s) flagged"
)

# ---------- Chart 1: Anomaly-flagged trend ----------
st.subheader("Daily Cost — Anomalies Flagged (Isolation Forest)")

line = alt.Chart(flagged).mark_line(color="#2563eb").encode(
    x=alt.X("Date:T", title="Date"),
    y=alt.Y("total_cost:Q", title="Daily Cost ($)"),
    tooltip=["Date:T", alt.Tooltip("total_cost:Q", format="$.2f")],
)
points = alt.Chart(flagged[flagged["is_anomaly"]]).mark_circle(color="#dc2626", size=100).encode(
    x="Date:T", y="total_cost:Q",
    tooltip=["Date:T", alt.Tooltip("total_cost:Q", format="$.2f")],
)
st.altair_chart((line + points).properties(height=360), use_container_width=True)

with st.expander("Flagged anomaly days"):
    st.dataframe(
        flagged[flagged["is_anomaly"]][["Date", "total_cost", "anomaly_score"]]
        .rename(columns={"total_cost": "Cost ($)"})
        .reset_index(drop=True)
    )

# ---------- Chart 2: Forecast ----------
st.subheader("14-Day Forecast (SARIMA, weekly seasonality)")

actual = forecast_df[forecast_df["type"] == "actual"]
future = forecast_df[forecast_df["type"] == "forecast"]

actual_line = alt.Chart(actual).mark_line(color="#2563eb").encode(
    x=alt.X("Date:T", title="Date"),
    y=alt.Y("CostInBillingCurrency:Q", title="Daily Cost ($)"),
)
forecast_line = alt.Chart(future).mark_line(color="#16a34a", strokeDash=[5, 3]).encode(
    x="Date:T", y="CostInBillingCurrency:Q",
)
band = alt.Chart(future).mark_area(opacity=0.15, color="#16a34a").encode(
    x="Date:T", y="lower:Q", y2="upper:Q"
)
st.altair_chart((actual_line + band + forecast_line).properties(height=320), use_container_width=True)
st.caption("Green dashed line = forecast, shaded band = 80% confidence interval.")

# ---------- Footer note ----------
st.divider()
st.caption(
    "Pipeline: ingest -> clean -> Isolation Forest (daily total cost, single feature) "
    "-> SARIMA forecast (weekly seasonality only, given ~2-3 months of history). "
    "Re-runs automatically whenever new CSV files are added to /data."
)
