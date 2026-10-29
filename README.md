# Adaptive IDS

A local CIC-IDS2017 network-flow classification project. It includes data preprocessing and offline analysis, a bounded machine-learning pipeline, a saved XGBoost model, a FastAPI inference service, and a React frontend.

The frontend sends requests to the existing API. Predictions use the saved model; the application does not train a model at startup. The current model was trained on a class-capped subset, so its evaluation results should not be treated as performance on the naturally distributed CIC-IDS2017 dataset.

## Run the application locally

Prerequisites: Python dependencies from the repository requirements files, Node.js 22.12 or newer, and the saved model artifact under `artifacts/models/`.

The CIC-IDS2017 source CSVs and generated `cicids2017_cleaned.csv` are intentionally excluded from Git because of their size. Keep or obtain the source data locally and follow [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md) for expected paths and preprocessing. The saved model artifacts are included.

Start the API from the project root:

```bash
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal, start the React frontend:

```bash
cd frontend
npm ci
npm run dev
```

Open `http://localhost:5173`. The API provides `/health`, `/ready`, `/models`, `/predict`, and `/predict/batch`. See [frontend/README.md](frontend/README.md) for configuration, CSV requirements, schema synchronization, tests, and build instructions. See [docs/API.md](docs/API.md) for the request and response contracts.

## Project areas

- `app/` — FastAPI service and saved-model inference.
- `ml/` — bounded training, evaluation, and artifact utilities.
- `frontend/` — React, TypeScript, and Vite UI.
- `week3_preprocessing.py`, `week3.py`, `cicids_eda.py` — preprocessing and exploratory analysis.
- `dashboard.py` — Streamlit dashboard for offline dataset exploration.
- `docs/` — architecture, API, data pipeline, model evaluation, and project status.

## Checks

From the project root, run the Python suite:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

From `frontend/`, run the frontend tests and production build:

```bash
npm test
npm run build
```

## Current limits

This project does not perform live packet capture, persistent prediction history, independent anomaly detection, or automated threat response. The frontend only sends CSV records to the configured local API; it does not upload files to an external service. The API loads trusted local model artifacts and does not accept artifact uploads.
