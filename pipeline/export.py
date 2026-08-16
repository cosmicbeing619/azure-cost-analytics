"""Runs ingest -> clean -> model, writes exactly ONE data file and ONE
chart spec for the dashboard. Nothing else.
"""
import json
import sys
from pathlib import Path

import altair as alt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest import load_all
from clean import clean
from model import daily_totals, flag_anomalies
from forecast import forecast_daily_cost

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_chart(flagged_df):
    line = (
        alt.Chart(flagged_df)
        .mark_line(color="#2563eb")
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("CostInBillingCurrency:Q", title="Daily Cost ($)"),
            tooltip=["Date:T", alt.Tooltip("CostInBillingCurrency:Q", format="$.2f")],
        )
    )
    points = (
        alt.Chart(flagged_df[flagged_df["is_anomaly"]])
        .mark_circle(color="#dc2626", size=100)
        .encode(
            x="Date:T",
            y="CostInBillingCurrency:Q",
            tooltip=["Date:T", alt.Tooltip("CostInBillingCurrency:Q", format="$.2f")],
        )
    )
    chart = (line + points).properties(
        width="container", height=360,
        title="Daily Azure Cost — Anomalies Flagged (Isolation Forest)"
    )
    return chart.to_dict()


def build_forecast_chart(forecast_df):
    import altair as alt

    actual = forecast_df[forecast_df["type"] == "actual"]
    future = forecast_df[forecast_df["type"] == "forecast"]

    actual_line = alt.Chart(actual).mark_line(color="#2563eb").encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("CostInBillingCurrency:Q", title="Daily Cost ($)"),
        tooltip=["Date:T", alt.Tooltip("CostInBillingCurrency:Q", format="$.2f")],
    )
    forecast_line = alt.Chart(future).mark_line(color="#16a34a", strokeDash=[5, 3]).encode(
        x="Date:T", y="CostInBillingCurrency:Q",
        tooltip=["Date:T", alt.Tooltip("CostInBillingCurrency:Q", format="$.2f")],
    )
    band = alt.Chart(future).mark_area(opacity=0.15, color="#16a34a").encode(
        x="Date:T", y="lower:Q", y2="upper:Q"
    )
    chart = (actual_line + band + forecast_line).properties(
        width="container", height=320,
        title="Daily Cost — 14-Day Forecast (SARIMA, weekly seasonality)"
    )
    return chart.to_dict()


def run():
    print("Loading and cleaning data...")
    df = load_all()
    df = clean(df)

    print("Running anomaly detection...")
    daily = daily_totals(df)
    flagged = flag_anomalies(daily)

    print(f"{flagged['is_anomaly'].sum()} anomalous day(s) flagged out of {len(flagged)}")

    # one data file
    data_out = flagged.copy()
    data_out["Date"] = data_out["Date"].astype(str)
    (OUT_DIR / "daily_cost.json").write_text(
        json.dumps(data_out.to_dict(orient="records"), indent=2)
    )
    print(f"wrote {OUT_DIR / 'daily_cost.json'}")

    # one chart spec
    (OUT_DIR / "chart.json").write_text(json.dumps(build_chart(flagged), indent=2, default=str))
    print(f"wrote {OUT_DIR / 'chart.json'}")

    print("Running forecast...")
    forecast_df = forecast_daily_cost(daily, horizon_days=14)
    (OUT_DIR / "forecast_chart.json").write_text(
        json.dumps(build_forecast_chart(forecast_df), indent=2, default=str)
    )
    print(f"wrote {OUT_DIR / 'forecast_chart.json'}")


if __name__ == "__main__":
    run()
