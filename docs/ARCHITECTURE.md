# Architecture

## Current implementation

```text
CIC-IDS2017 CSV files
        ├── week3.py / cicids_eda.py  (single-file exploratory analysis)
        └── week3_preprocessing.py    (full in-memory clean + CSV export)
                                          │
                                          ▼
                              cicids2017_cleaned.csv
                                          ├── ml/data.py (stream validation + class-capped sample)
                                          ├── ml/splitting.py (exact-row fingerprint group split)
                                          ├── ml/train.py (bounded LR / RF / XGBoost comparison; artifacts saved)
                                          └── dashboard.py (offline EDA)
                                                               │
                                  app/main.py ─► lifespan model loader ─► prediction service
                                      ├── /health, /ready, /models
                                      └── /predict, /predict/batch
                                                          ▲
                                                          │ same-origin dev CORS
                                          frontend/ (React + Vite)
                                          ├── artifact-derived 77-feature form
                                          ├── browser CSV validation + batch results
                                          └── API-backed health/readiness/model status
```

The React frontend consumes the existing API and does not train models. Its feature schema is generated from the saved artifact metadata; it refuses to submit if the API model name, version, or class list differs. CSV files are parsed and validated in the browser before bounded batch requests. The development API allows the exact Vite origin `http://localhost:5173`.

The API loads the saved XGBoost model once and serves validated single and bounded-batch requests. The bounded training pipeline has saved supervised model artifacts and actual class-capped subset metrics. The project does not implement packet capture, an alert workflow, persistent operational history, or real-time prevention. Dashboard figures describe offline dataset analysis, not live network activity.

## Intended staged architecture

```text
Offline CIC-IDS2017 records ──► validated preprocessing ──► train/validation/test data
                                                               │
                                                               ▼
                                                 supervised model + feature schema
                                                               │
                               optional independent anomaly detector ──┤
                                                               ▼
                                               inference and risk/alert services
                                                               ▼
                                             FastAPI ──► SQL database
                                               │                 │
                                               └──── dashboard ─┘

Approved feedback ──► auditable offline review ──► candidate retraining/evaluation
```

The intended architecture is a plan, not an implemented system. Real-time ingestion would be a separate future extension and would require its own tested data source and operational safeguards.

## Data and model boundaries

- `Label` is the supervised target and must never enter the model feature matrix.
- Data-dependent transformations must be fitted using training data only and reused unchanged for validation, test, and inference.
- Exact duplicate handling and the split method must be recorded. Stratification preserves class proportions but does not by itself establish generalization to future time periods or unseen capture environments.
- Supervised class scores and anomaly scores have different meanings; an anomaly is not by itself a confirmed attack.
- Model artifacts must carry or reference feature order/schema and version/configuration metadata.
- The inference API checks the feature schema against the saved bundle, reorders inputs to its stored order, and rejects missing, extra, non-numeric, or non-finite values.
- Feedback is evidence for a future offline candidate model, not an automatic production update.

## Operational boundary

No cloud resources or external response actions are part of the current project. A future response module should begin in dry-run mode. AWS remains a separately authorized deployment stage.
