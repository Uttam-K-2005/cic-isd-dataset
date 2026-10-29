# CIC-IDS2017 frontend

React, TypeScript, and Vite interface for the existing FastAPI inference service. It displays live API and model status, accepts one flow using the saved model's ordered feature schema, and parses bounded CSV batches in the browser. It does not train or update models.

## Run locally

Prerequisites: Node.js 22.12 or newer and the repository Python environment with API dependencies installed.

Start the API from the repository root in one terminal:

```bash
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Start the frontend in another terminal:

```bash
cd frontend
npm ci
cp .env.example .env.local  # optional; defaults point to local API
npm run dev
```

Open `http://localhost:5173`. The API enables CORS for this exact development origin. Set `VITE_API_BASE_URL` in `.env.local` to change the API address. Keep `VITE_MAX_BATCH_SIZE` at or below the API's `CICIDS_MAX_BATCH_SIZE` (default 64); oversized files are rejected and never silently truncated.

## Schema maintenance

`src/data/model-schema.json` is generated from the active saved artifact metadata. If the active artifact changes, regenerate it from the repository root:

```bash
.venv/bin/python frontend/scripts/sync_model_schema.py \
  --artifact-dir artifacts/models/xgboost_20261029T152352Z \
  --dataset cicids2017_cleaned.csv
```

The script validates the 77 ordered numeric model inputs and scans the dataset to mark which features contain negative source values. The target `Label` and repeated legacy header are not model inputs. At runtime, the overview compares API model name, version, and supported classes to this generated schema; a mismatch disables inference until the schema is synced.

## Checks and production build

```bash
npm test
npm run build
npm run preview
```

## API contract used

| UI feature | API |
| --- | --- |
| API and model status | `GET /health`, `GET /ready`, `GET /models` |
| Single flow | `POST /predict` with the 77 raw numeric features |
| CSV batch | `POST /predict/batch` with `{"records": [...]}` |

CSV parsing and validation happen in the browser. Each submitted record must have exactly the model feature names, finite numeric values representable by float32, and no negative values for fields that had none in the source data. Files are capped at 5 MB and the configured batch size. Results are the API's actual predictions and can be downloaded as CSV. No sample metrics or placeholder predictions are shown. No history, database, anomaly score, packet capture, or response action is implemented.
