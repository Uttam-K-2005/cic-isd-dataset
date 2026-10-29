# CIC-IDS2017 inference API

## Model and input contract

The service loads the existing XGBoost joblib bundle once in FastAPI lifespan startup. It does not train or update a model. The default trusted local bundle is `artifacts/models/xgboost_20261029T152352Z`; `metadata.json` accompanies `model.joblib`. Set `CICIDS_MODEL_ARTIFACT` to select another compatible artifact directory.

The bundle is authoritative for the 77 expected feature names and their order. The loader compares those names, model name/version, and the 15-label mapping with `metadata.json`, then checks the feature count and encoded class order exposed by XGBoost. Values are numeric CIC-IDS2017 flow statistics converted to finite float32 inputs. The training pipeline verified and removed the duplicated `Fwd Header Length.1` column; it is not an inference feature. No scaler or other fitted transformation is used for XGBoost. The `Label` target is never accepted as an input feature.

Requests must contain every feature exactly once, with no missing or extra keys. Each value must be a finite JSON number (booleans, strings, nulls, NaN, and infinities are rejected). Features are reordered to the bundle’s stored order before prediction. Missing/extra keys, non-numeric values, and non-finite values return HTTP 422. Request validation never fills missing values.

## Endpoints

| Method and path | Behavior |
| --- | --- |
| `GET /health` | Process health only; returns `{"status":"ok"}` even when a model cannot load. |
| `GET /ready` | Returns 200 when the model is loaded, otherwise 503. |
| `GET /models` | Safe model name, version, and supported label list; returns 503 when unavailable. |
| `POST /predict` | One feature object at the top level; returns one prediction. |
| `POST /predict/batch` | Body is `{"records":[feature_object,...]}`; default maximum is 64 records. |

Batch size is configurable with `CICIDS_MAX_BATCH_SIZE` from 1 to 256. Empty and oversized batches return 422. A model that is missing or incompatible leaves the process healthy but not ready; predictions then return 503 without exposing internal paths or stack traces.

For local browser development, CORS allows the Vite origin `http://localhost:5173` for GET and POST with the JSON content type, without credentials. No other origins are enabled.

Example response:

```json
{
  "request_id": "8c498a96-1be3-4c03-b80c-9d8d9d5db47e",
  "predicted_class": "BENIGN",
  "is_attack": false,
  "probabilities": {
    "BENIGN": 0.9,
    "Bot": 0.01,
    "DDoS": 0.03,
    "DoS GoldenEye": 0.01,
    "DoS Hulk": 0.01,
    "DoS Slowhttptest": 0.005,
    "DoS slowloris": 0.005,
    "FTP-Patator": 0.005,
    "Heartbleed": 0.001,
    "Infiltration": 0.001,
    "PortScan": 0.01,
    "SSH-Patator": 0.005,
    "Web Attack � Brute Force": 0.003,
    "Web Attack � Sql Injection": 0.001,
    "Web Attack � XSS": 0.004
  },
  "model_name": "xgboost",
  "model_version": "20261029T152352Z",
  "predicted_at": "2026-10-29T15:30:00+00:00"
}
```

The probability object contains every supported class in the exact order stored in the bundle. These are illustrative model probabilities, not calibrated confidence. `is_attack` is derived only from the predicted class (`false` for `BENIGN`). The API does not return an anomaly score, threat-risk score, or ground-truth label.

## Run locally

From the project root, reuse the existing environment:

```bash
source .venv/bin/activate
python -m pip install -r requirements-api.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The server loads the default artifact from the relative path above. To choose a different trusted bundle and batch limit:

```bash
CICIDS_MODEL_ARTIFACT=artifacts/models/xgboost_20261029T152352Z \
CICIDS_MAX_BATCH_SIZE=64 \
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open interactive documentation at `http://127.0.0.1:8000/docs`. Check health and readiness:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl http://127.0.0.1:8000/models
```

To send the first real numeric flow record from the existing cleaned CSV without printing the record, stream it directly into the request. The target and legacy duplicate column are removed:

```bash
python -c 'import json,pandas as pd; r=pd.read_csv("cicids2017_cleaned.csv",nrows=1).iloc[0]; print(json.dumps({k:float(r[k]) for k in r.index if k not in ("Label","Fwd Header Length.1")},allow_nan=False))' \
| curl -sS -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @-
```

For a hand-authored payload, copy all 77 feature names and numeric values from the bundle schema; an example JSON object is kept in the HTTP request body, not a project file. The API needs no full-dataset scan or training at startup.

## Trust and limits

`model.joblib` is a pickle-based format that can execute code while loading. Only load a bundle created by this project from a trusted local source; the API does not accept artifact uploads. Model predictions inherit the evaluation limitations described in [MODEL_EVALUATION.md](MODEL_EVALUATION.md): the initial training run used a class-capped subset, and no capture-held-out or natural-prevalence evaluation has been completed. This is an offline inference API, not live packet capture or an automated response system.
