"""Reads all CSV files currently in /data and returns a combined dataframe.
Re-running this after a new file is dropped in /data picks it up automatically —
no code changes needed.
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def load_all():
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(files)} file(s), {len(df)} rows total")
    return df

if __name__ == "__main__":
    load_all()
