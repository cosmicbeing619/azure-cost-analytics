"""Cleans and type-casts the raw ingested dataframe."""
import pandas as pd

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["CostInBillingCurrency"] = pd.to_numeric(df["CostInBillingCurrency"], errors="coerce")
    # TODO: decide how negative costs/credits are handled, document decision here
    return df
