"""Chains the full pipeline and writes JSON outputs into docs/data/
for the static dashboard to consume."""
import json
from pathlib import Path
from ingest import load_all
from clean import clean

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def run():
    df = load_all()
    df = clean(df)
    # TODO: call analyze.py + anomaly.py, export results as JSON here
    summary = {"total_cost": float(df["CostInBillingCurrency"].sum()), "rows": len(df)}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Wrote summary.json")

if __name__ == "__main__":
    run()
