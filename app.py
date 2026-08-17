"""Azure Cost Anomaly Dashboard — Streamlit version.

Runs the pipeline (ingest -> clean -> feature matrix -> forecast) once per
data change (cached), then applies Isolation Forest live so the
contamination slider updates anomaly flags instantly without re-ingesting.
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

st.set_page_config(page_title="Azure Cost Anomaly Dashboard", layout="wide", page_icon="☁️")

COLOR_ACTUAL = "#2563eb"
COLOR_ANOMALY = "#dc2626"
COLOR_FORECAST = "#16a34a"


# ---------- Cached: everything that doesn't depend on the contamination slider ----------

def data_signature() -> str:
    """Cache-busting signature: filenames + sizes of everything in /data.
    Changes whenever a file is added/modified -- forces a fresh pipeline run."""
    data_dir = Path(__file__).resolve().parent / "data"
    files = sorted(data_dir.glob("*.csv"))
    return "|".join(f"{f.name}:{f.stat().st_size}" for f in files)


@st.cache_data(show_spinner="Loading and processing data...")
def get_base_pipeline(_signature: str):
    df = load_all()
    df = clean(df)
    daily = daily_totals(df)
    features = build_daily_feature_matrix(df)
    return df, daily, features


df, daily, features = get_base_pipeline(data_signature())


# ---------- Sidebar: live controls ----------

st.sidebar.title("☁️ Controls")
st.sidebar.markdown("### Contamination Rate")
contamination = st.sidebar.slider(
    "Expected anomaly rate", min_value=0.02, max_value=0.20, value=0.07, step=0.01,
    help="What fraction of days should Isolation Forest treat as anomalous. "
         "This is a tunable business decision, not a fixed truth -- move it "
         "and watch which days get flagged."
)

# recomputed live (fast, not cached) so the slider feels instant
flagged = flag_anomalies(features, contamination=contamination)

st.sidebar.markdown(f"**{int(flagged['is_anomaly'].sum())} day(s)** flagged at this sensitivity")
st.sidebar.divider()
st.sidebar.caption(
    "Pipeline: ingest → clean → Isolation Forest "
    "(total cost + top categories + day-of-week)"
    "Drop a new CSV into /data and push to update."
)


# ---------- Header + KPI cards ----------

st.title("Azure Cost Anomaly Dashboard")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Spend", f"${df['CostInBillingCurrency'].sum():,.2f}")
col2.metric("Days Analyzed", f"{df['Date'].nunique()}")
col3.metric("Anomalies Flagged", f"{int(flagged['is_anomaly'].sum())}")
col4.metric(
    "Date Range",
    f"{df['Date'].min().strftime('%b %d')} – {df['Date'].max().strftime('%b %d')}"
)

st.divider()


# ---------- Tabs ----------

tab1, tab2 = st.tabs(["📈 Anomaly Detection", "🔍 Day Inspector"])

with tab1:
    st.subheader("Daily Cost — Anomalies Flagged (Isolation Forest)")

    line = alt.Chart(flagged).mark_line(color=COLOR_ACTUAL).encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("total_cost:Q", title="Daily Cost ($)"),
        tooltip=["Date:T", alt.Tooltip("total_cost:Q", format="$.2f")],
    )
    points = alt.Chart(flagged[flagged["is_anomaly"]]).mark_circle(color=COLOR_ANOMALY, size=120).encode(
        x="Date:T", y="total_cost:Q",
        tooltip=["Date:T", alt.Tooltip("total_cost:Q", format="$.2f"), "anomaly_score:Q"],
    )
    st.altair_chart((line + points).properties(height=380), use_container_width=True)

    anomaly_table = (
        flagged[flagged["is_anomaly"]][["Date", "total_cost", "anomaly_score"]]
        .rename(columns={"total_cost": "Cost ($)", "anomaly_score": "Anomaly Score"})
        .sort_values("Anomaly Score")
        .reset_index(drop=True)
    )
    st.markdown("**Flagged days** (most anomalous first)")
    st.dataframe(anomaly_table, use_container_width=True)

    st.download_button(
        "⬇ Download flagged anomalies (CSV)",
        anomaly_table.to_csv(index=False).encode("utf-8"),
        file_name="flagged_anomalies.csv",
        mime="text/csv",
    )

with tab2:
    st.subheader("Inspect a specific day")

    flagged_dates = sorted(flagged[flagged["is_anomaly"]]["Date"].dt.date.tolist())
    all_dates = sorted(flagged["Date"].dt.date.tolist())

    default_choice = flagged_dates[0] if flagged_dates else all_dates[0]
    selected_date = st.selectbox(
        "Choose a date (flagged anomalies marked with ⚠️)",
        options=all_dates,
        index=all_dates.index(default_choice),
        format_func=lambda d: f"⚠️ {d}" if d in flagged_dates else str(d),
    )

    day_df = df[df["Date"].dt.date == selected_date]
    day_total = day_df["CostInBillingCurrency"].sum()
    is_flagged = selected_date in flagged_dates

    c1, c2 = st.columns(2)
    c1.metric("Total cost this day", f"${day_total:,.2f}")
    c2.metric("Status", "⚠️ Anomaly" if is_flagged else "✅ Normal")

    st.markdown("**Cost breakdown by category**")
    cat_breakdown = (
        day_df.groupby("MeterCategory")["CostInBillingCurrency"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .head(10)
    )
    bar = alt.Chart(cat_breakdown).mark_bar(color=COLOR_ANOMALY if is_flagged else COLOR_ACTUAL).encode(
        x=alt.X("CostInBillingCurrency:Q", title="Cost ($)"),
        y=alt.Y("MeterCategory:N", sort="-x", title=None),
        tooltip=["MeterCategory:N", alt.Tooltip("CostInBillingCurrency:Q", format="$.2f")],
    ).properties(height=300)
    st.altair_chart(bar, use_container_width=True)

    if is_flagged:
        st.info(
            "This day was flagged because its overall spend pattern — total cost, "
            "category mix, or day-of-week combination — didn't resemble most other "
            "days in the dataset. Use the category breakdown above to judge whether "
            "that's a real issue or an expected one-off."
        )