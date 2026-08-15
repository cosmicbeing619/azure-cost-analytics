"""Core EDA aggregations: cost overview, concentration/Pareto, hierarchy rollup."""
import pandas as pd

def cost_by_category_over_time(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["Date", "MeterCategory"])["CostInBillingCurrency"].sum().reset_index()

def concentration(df: pd.DataFrame, column: str) -> pd.DataFrame:
    ranked = df.groupby(column)["CostInBillingCurrency"].sum().sort_values(ascending=False)
    total = ranked.sum()
    return (ranked / total * 100).cumsum().reset_index(name="cumulative_pct")
