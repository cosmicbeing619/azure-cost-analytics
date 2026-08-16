import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import altair as alt

    alt.data_transformers.enable("vegafusion") 
    return alt, pd


@app.cell
def _():
    file_paths = [
        "2022-12.csv",
        "2023-01.csv",
        "2023-02.csv",
    ]
    return (file_paths,)


@app.cell
def _(file_paths, pd):
    dfs = []

    for file_path in file_paths:
        df_month = pd.read_csv(file_path)
        df_month["SourceFile"] = file_path
        dfs.append(df_month)

    df = pd.concat(dfs, ignore_index=True)
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df["SourceFile"].value_counts()
    return


@app.cell
def _(df, pd):
    df["Date"] = pd.to_datetime(df["Date"])
    df["CostInBillingCurrency"] = pd.to_numeric(
        df["CostInBillingCurrency"],
        errors="coerce"
    )
    return


@app.cell
def _(alt, df):
    cost_distribution = (
        alt.Chart(df)
        .transform_filter(
            alt.datum.CostInBillingCurrency > 0
        )
        .mark_bar()
        .encode(
            x=alt.X(
                "CostInBillingCurrency:Q",
                bin=alt.Bin(maxbins=50),
                title="Cost"
            ),
            y=alt.Y(
                "count():Q",
                title="Number of Records"
            ),
            tooltip=[
                alt.Tooltip("count():Q", title="Records")
            ]
        )
        .properties(
            title="Distribution of Azure Costs",
            width=700,
            height=400
        )
    )

    cost_distribution
    return


@app.cell
def _(df):
    daily_cost = (
        df.groupby("Date", as_index=False)
          ["CostInBillingCurrency"]
          .sum()
    )
    return (daily_cost,)


@app.cell
def _(alt, daily_cost):
    daily_cost_chart = (
        alt.Chart(daily_cost)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "Date:T",
                title="Date"
            ),
            y=alt.Y(
                "CostInBillingCurrency:Q",
                title="Total Cost"
            ),
            tooltip=[
                alt.Tooltip("Date:T", title="Date"),
                alt.Tooltip(
                    "CostInBillingCurrency:Q",
                    title="Cost",
                    format=",.2f"
                )
            ]
        )
        .properties(
            title="Daily Azure Cost Over Time",
            width=800,
            height=400
        )
        .interactive()
    )

    daily_cost_chart
    return


@app.cell
def _(df):
    monthly_cost = (
        df.assign(Month=df["Date"].dt.to_period("M").dt.to_timestamp())
          .groupby("Month", as_index=False)
          ["CostInBillingCurrency"]
          .sum()
    )
    return (monthly_cost,)


@app.cell
def _(alt, monthly_cost):
    monthly_cost_chart = (
        alt.Chart(monthly_cost)
        .mark_line(point=True)
        .encode(
            x=alt.X("Month:T", title="Month"),
            y=alt.Y(
                "CostInBillingCurrency:Q",
                title="Total Cost"
            ),
            tooltip=[
                alt.Tooltip("Month:T", title="Month"),
                alt.Tooltip(
                    "CostInBillingCurrency:Q",
                    title="Cost",
                    format=",.2f"
                )
            ]
        )
        .properties(
            title="Monthly Azure Cost",
            width=700,
            height=400
        )
    )

    monthly_cost_chart
    return


@app.cell
def _(df):
    cost_df = df["CostInBillingCurrency"].copy()
    return (cost_df,)


@app.cell
def _(alt, cost_df):
    boxplot = (
        alt.Chart(cost_df)
        .mark_boxplot(
            extent="min-max",
            size=50
        )
        .encode(
            y=alt.Y(
                "CostInBillingCurrency:Q",
                title="Cost"
            )
        )
        .properties(
            title="Distribution of Azure Costs",
            width=200,
            height=500
        )
    )

    boxplot
    return


if __name__ == "__main__":
    app.run()
