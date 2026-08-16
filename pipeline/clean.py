"""Cleans and type-casts the raw ingested dataframe.

Data quality findings (checked against the real Dec/Jan/Feb dataset, 75,648 rows):
- No nulls in any column.
- No negative CostInBillingCurrency values (no credits/refunds present in this data) —
  if a future data drop introduces negatives, they are NOT silently dropped here;
  add explicit handling if/when they appear (see flag_negative_costs below).
- ~9% of rows are zero-cost (legitimate — free-tier services like the F1 App Service
  plan). These are kept, not dropped: they're meaningful for a "free vs. paid tier
  mix" view, and dropping them would understate resource/activity counts.
"""
import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df["CostInBillingCurrency"] = pd.to_numeric(df["CostInBillingCurrency"], errors="coerce")

    # Safety net: if parsing ever produces NaNs (bad row in a future data drop),
    # don't silently drop them — flag for review instead.
    bad_cost_rows = df["CostInBillingCurrency"].isnull().sum()
    if bad_cost_rows:
        print(f"WARNING: {bad_cost_rows} rows had unparseable cost values — review before proceeding")

    df = flag_negative_costs(df)

    return df


def flag_negative_costs(df: pd.DataFrame) -> pd.DataFrame:
    """No negative costs exist in the current dataset. This function exists so that
    if a future data drop introduces credits/refunds, they get flagged explicitly
    rather than silently mishandled by downstream aggregation or anomaly detection.
    """
    df = df.copy()
    df["is_credit"] = df["CostInBillingCurrency"] < 0
    n_credits = df["is_credit"].sum()
    if n_credits:
        print(f"NOTE: {n_credits} credit/refund rows detected — currently kept as-is, "
              f"consider excluding from anomaly detection to avoid false 'drop' flags")
    return df


def tier_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Free-tier vs paid-tier row counts and cost, per MeterCategory — a genuine
    insight surfaced by the zero-cost rows rather than something to discard."""
    df = df.copy()
    df["tier"] = df["CostInBillingCurrency"].apply(lambda c: "free" if c == 0 else "paid")
    return (
        df.groupby(["MeterCategory", "tier"])
        .agg(rows=("CostInBillingCurrency", "size"), total_cost=("CostInBillingCurrency", "sum"))
        .reset_index()
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ingest import load_all

    df = load_all()
    df = clean(df)

    print(df.dtypes)
    print()
    print("Credit rows:", df["is_credit"].sum())
    print()
    print(tier_mix(df).head(10))
