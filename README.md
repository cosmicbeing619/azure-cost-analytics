# ☁️ Azure Cost Anomaly Dashboard

An end-to-end anomaly detection system for Azure cloud billing data — built to be small enough to fully explain, and rigorous enough to statistically validate.

Ingests raw billing exports → cleans and engineers a daily feature matrix → flags anomalous spend days with an **Isolation Forest** → explains every flag with **SHAP** → serves it all in an interactive **Streamlit** dashboard that auto-updates whenever new data is pushed to this repo.

**[🔗 Live Dashboard](https://azure-cost-analytics-cosmic.streamlit.app/)**

---

## Why this exists

Cloud cost data is a good real-world testbed for anomaly detection: it's skewed, multivariate, and — in almost every real setting — **unlabeled**. Nobody hands you a column that says "this day was anomalous." This project is built around that constraint: every modeling decision here is justified by what the data can actually support, not by what looks impressive.

## What it does

- 📥 **Ingests** any CSV dropped into `/data` — no code changes needed to add new billing periods
- 🧹 **Cleans** and validates the data (verified: no nulls, no negative costs in the source data; ~9% legitimate zero-cost free-tier rows kept, not dropped)
- 🧠 **Detects anomalies** with an Isolation Forest trained on a daily multivariate feature matrix (total cost, spend across top categories, day-of-week)
- 🔍 **Explains every flag** with SHAP — a real decomposition of the model's decision, not a manual description of the data
- 🎛️ **Live sensitivity control** — the anomaly threshold (`contamination`) is an adjustable slider in the dashboard, not a hidden constant
- 📊 **Day Inspector** — pick any date and see its category-level cost breakdown plus its SHAP explanation
- 🔄 **Auto-updates** — push new data to this repo, Streamlit Cloud redeploys automatically, the whole pipeline reruns fresh

## Screenshots

<img width="2556" height="1260" alt="image" src="https://github.com/user-attachments/assets/4cfb2d27-b022-44dc-affc-92cc1387c31e" />

---

## Architecture

```
azure-cost-analytics/
├── data/                    # source CSVs — drop new files here to update
├── pipeline/
│   ├── ingest.py            # loads all CSVs in /data
│   ├── clean.py              # type casting, data quality checks
│   └── model.py               # feature engineering + Isolation Forest
├── app.py                    # Streamlit dashboard (entry point)
└── requirements.txt
```

**Flow:** `ingest → clean → build feature matrix → (cached)`, then live on every UI interaction: `→ Isolation Forest scoring → SHAP explanation → dashboard`.

There's no saved/pickled model — it retrains from scratch whenever the underlying data changes, which is effectively free at this data volume and guarantees the model is never stale.

## Model

**Isolation Forest**, trained on a per-day feature vector: total cost, spend across the top 8 meter categories (rest bucketed into "Other"), and day-of-week.

**Why Isolation Forest:** no distributional assumptions (important — cost data is right-skewed), works without labeled anomalies, and handles the multivariate feature space natively. SARIMA and quantile regression were both considered and rejected — with under 3 months of history, there aren't enough seasonal cycles to fit either reliably without just fitting noise.

| Hyperparameter | Value | Why |
|---|---|---|
| `contamination` | 0.07 (adjustable) | Rank-based cutoff on anomaly score, not a fixed threshold — explicitly a tunable business decision, exposed as a live slider |
| `n_estimators` | 200 | Averaging across trees reduces variance from any single tree's random splits |
| `top_n_categories` | 8 | Using all ~49 raw categories against ~69 daily samples causes curse-of-dimensionality noise; capping preserves real signal |

## Validation

Since there's no ground truth, the flagged anomalies were cross-checked against multiple independent methods:
- **Kolmogorov–Smirnov test** on Mahalanobis distance — significant (p < 0.05) at every contamination level tested
- **Energy distance test** (full multivariate) — not significant, attributed to limited statistical power at this sample size, not to the anomalies being false
- **Rolling residual z-score** — a local trend-break check, largely non-overlapping with Isolation Forest's global view (expected — they test different definitions of "anomalous")
- **10-seed stability check** — confirmed which flagged days are robust vs. borderline

## Known limitations

- `day_of_week` is integer-encoded, which Isolation Forest treats as a continuous number rather than a true category — a cyclical (sin/cos) encoding would be the correct production fix
- The first and last dates in the dataset are flagged at every contamination setting tested — likely a partial-day billing export artifact, not a genuine spend anomaly

## Running locally

```bash
git clone https://github.com/<your-username>/azure-cost-analytics.git
cd azure-cost-analytics
pip install -r requirements.txt
streamlit run app.py
```

## Adding new data

Drop a new CSV into `/data` and push:

```bash
git add data/2023-03.csv
git commit -m "Add March data"
git push
```

If deployed on Streamlit Community Cloud, the push alone triggers a redeploy — the pipeline reruns on the full updated dataset automatically, no manual steps.

## Tech stack

Python · pandas · scikit-learn · SHAP · Altair · Streamlit · Streamlit Community Cloud

