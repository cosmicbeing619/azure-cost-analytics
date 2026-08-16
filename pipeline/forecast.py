"""ONE small forecast: basic SARIMA on daily total cost, weekly seasonality only.

Why weekly (m=7) and not monthly seasonality: with ~69 days of history that's
roughly 9-10 full weekly cycles -- enough for SARIMA to learn a real weekly
pattern. A monthly seasonal term would need many more monthly cycles than we
have and would just be fitting noise -- same reasoning that ruled out SARIMA
for the anomaly model earlier, but weekly seasonality specifically survives
that objection.
"""
import pandas as pd


def forecast_daily_cost(daily_df: pd.DataFrame, horizon_days: int = 14) -> pd.DataFrame:
    """daily_df: output of model.daily_totals() -- columns Date, CostInBillingCurrency.
    Returns historical + forecasted rows with confidence bounds, one combined table.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    series = daily_df.set_index("Date")["CostInBillingCurrency"]
    series.index = pd.DatetimeIndex(series.index).to_period("D").to_timestamp()

    # basic order: (1,1,1) trend, weekly seasonal (1,1,1,7) -- deliberately simple,
    # not grid-searched, so it stays explainable in one sentence.
    model = SARIMAX(
        series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    fit = model.fit(disp=False)

    forecast = fit.get_forecast(steps=horizon_days)
    mean = forecast.predicted_mean
    ci = forecast.conf_int(alpha=0.2)  # 80% interval -- readable, not overconfident

    hist = daily_df.copy()
    hist["type"] = "actual"
    hist["lower"] = None
    hist["upper"] = None

    future = pd.DataFrame({
        "Date": mean.index,
        "CostInBillingCurrency": mean.values,
        "type": "forecast",
        "lower": ci.iloc[:, 0].values,
        "upper": ci.iloc[:, 1].values,
    })

    combined = pd.concat([hist, future], ignore_index=True)
    return combined


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ingest import load_all
    from clean import clean
    from model import daily_totals

    df = load_all()
    df = clean(df)
    daily = daily_totals(df)

    result = forecast_daily_cost(daily)
    print(result.tail(20))
