# Marimo notebook — run with: marimo edit notebooks/eda.py
import marimo

app = marimo.App()

@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import altair as alt
    return mo, pd, alt

if __name__ == "__main__":
    app.run()
