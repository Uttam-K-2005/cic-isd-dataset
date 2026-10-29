# Implementation plan

This plan is based on the repository audit dated 2026-10-29. Completion means acceptance criteria were checked; a plan item alone is not evidence of implementation.

| Stage | Work | Acceptance evidence | Status |
| --- | --- | --- | --- |
| 1. Audit and plan | Inventory files, dependencies, data/artifacts, and gaps; record architecture and staged work. | Audit and plan agree with inspected workspace; no unverified ML claims. | Complete for the visible workspace; Git history unavailable. |
| 2. Preprocessing | Make input discovery, schema/label checks, invalid-value policy, duplicate policy, and output paths explicit and reproducible. Choose a memory-aware execution strategy and split protocol. | Small fixture validation plus a documented full-data command; report row counts and label counts from an actual run. Preserve the current cleaned artifact until replacement is explicitly arranged. | Implemented; synthetic tests pass; all 2,520,798 cleaned rows streamed and validated. Source preprocessing provenance/global duplicate history remains unknown. |
| 3. ML baseline | Train Logistic Regression, Random Forest, and XGBoost using fixed group-aware splits; save feature schema, configuration, metrics, and model artifacts. | Actual validation and final subset-test metrics, per-class scores, confusion matrix, false-positive rates, AUC where available, and saved artifacts. | Bounded 500-per-class comparison complete; all metrics/artifacts saved and inference smoke checked. Natural-prevalence/capture-held-out evaluation remains unavailable from current merged CSV. |
| 4. Reusable inference and anomaly score | Add artifact loading and prediction; keep anomaly scoring independent of supervised class probabilities. | No fake prediction when artifacts are missing; small-fixture checks and documented detector strategy. | Single-record supervised inference implemented and smoke checked; independent anomaly detector not started. |
| 5. API | Health/readiness, model metadata, bounded prediction endpoints, validation, clear errors, and offloaded synchronous inference. | API tests for valid/invalid/missing schema and unavailable artifacts. | Implemented and tested with saved XGBoost artifact; no retraining or database added. |
| 6. Persistence and alerts | Add SQLAlchemy models, migrations, bounded history queries, alert creation and acknowledgement. | SQLite integration tests and PostgreSQL configuration documented. | Not started |
| 7. Threat response and feedback | Dry-run response levels and auditable feedback; keep feedback separate from model promotion. | Tests prove no external action occurs in simulation and unapproved feedback cannot promote a model. | Not started |
| 8. Dashboard and frontend | Preserve offline EDA and add API-backed operational views only when those services exist. Clearly label offline vs operational data. | React frontend validates artifact-derived inputs, calls live status and prediction endpoints, and handles bounded CSV batches. | React frontend implemented with schema mismatch guard, single and batch prediction, API-backed overview, and responsive UI. Persistent history remains unavailable; Streamlit dashboard remains offline EDA. |
| 9. Quality and security | Add focused tests, lint/format/type checks as justified, input limits, safe logging, and configuration. | Commands run and results recorded; limitations documented. | Backend API tests and frontend unit/component tests cover validation and result handling; lint/type/CI automation remains unconfigured. |
| 10. Local deployment and CI | Docker/Compose and CI after app/artifact packaging is settled. | Clean local startup and CI checks. Do not include full source dataset in API image. | Not started |
| 11. AWS preparation | Document small deployment path, configuration, cost considerations, and cleanup. | Instructions reproducible; no resources provisioned absent explicit authorization. | Not started |

## Immediate next step

The next milestone is persistence and alert history. Preserve the current offline model evaluation caveats when exposing predictions. Do not report capped per-class test metrics as natural-prevalence performance.

Large model training, full-dataset preprocessing, cloud provisioning, and publishing are not automatic steps.
