# Project audit

Audit date: 2026-10-29

## Scope and method

Inspected the workspace and ran the 35-test `unittest` suite, a chunked validation pass over the cleaned dataset, a bounded model comparison, a saved-model inference smoke check, and a local Uvicorn HTTP check. The working directory is not a Git repository (`git status` reports that it is not a repository), so a version-control diff or history could not be inspected.

## What exists

| Area | Status | Evidence and limits |
| --- | --- | --- |
| Dataset | Present and streamed for validation | Eight source CSVs total about 845 MB. `cicids2017_cleaned.csv` is 880,193,731 bytes and 2,520,798 rows. A chunked pass validated all expected numeric columns and rows, verified the legacy repeated feature column is equal, and counted all 15 labels. No preprocessing report is present, so output provenance and global duplicate history remain unknown. |
| Preprocessing | Implemented and covered by synthetic tests; not rerun on source data | `week3_preprocessing.py` validates headers, processes chunks, reports quality counts, checks the repeated CICIDS `Fwd Header Length` column for equality, and writes timestamped outputs without replacement. It removes exact duplicates within each chunk only; global duplicates are not checked. Three observed labels contain the literal replacement character `�`; that spelling is preserved and added to the known label set. |
| EDA | Implemented, not validated in this audit | `week3.py` examines one Monday file for missing/infinite values, negative flow duration, labels, and a univariate outlier count. `cicids_eda.py` plots flow duration for that same file. |
| Dashboard | Implemented, not validated in this audit | `dashboard.py` provides offline dataset summaries, class distributions, and sampled feature plots. It loads the entire cleaned CSV into memory. Performance sections are placeholders. Displayed quality values are hard-coded claims based on prior preprocessing and are not rechecked by the app. |
| ML training/evaluation/inference | Bounded subset comparison implemented and run | `ml/` includes chunked data validation and capped sampling, fingerprint-group splits, model definitions, metrics, artifact I/O, and single-record inference. All three models were trained on at most 6,068 sampled records; saved artifacts and test metrics exist under `artifacts/`. Results are not natural-prevalence estimates. |
| Anomaly detection / alerting / response / feedback | Missing | No implementation found. |
| API and React frontend | Implemented; API exercised over local HTTP and frontend test/build checks added | `app/` provides health/readiness, metadata, single prediction, bounded batch prediction, startup artifact validation, strict feature checks, response metadata, and exact-origin Vite CORS. `frontend/` provides a metadata-derived form, browser CSV validation, batch results, and live API status. No retraining occurs. See `docs/API.md` and `frontend/README.md`. |
| Database | Missing by design for this milestone | No SQLAlchemy, PostgreSQL, or migration code added. |
| Tests / quality automation | Backend and frontend unit tests present | Backend `unittest discover -s tests -v` and frontend `npm test` cover service/API contracts, schema/CSV validation, and UI results. Frontend `npm run build` performs TypeScript checking and production bundling. No formatter, linter, or CI configuration exists. `dataset/test.py` remains empty. |
| Deployment | Missing | No Docker, Compose, or AWS infrastructure files found. |
| Dependencies | ML and API stacks installed in project environment | Existing `requirements.txt` still omits Matplotlib used by EDA. `requirements-ml.txt` pins scikit-learn and XGBoost. `requirements-api.txt` declares FastAPI, Uvicorn, and the test client. XGBoost also needs the machine-level Homebrew `libomp` runtime on this Mac. |
| Documentation | Added and updated | Audit, implementation plan, architecture, data pipeline, model evaluation, and API documents describe verified implementation state and limits. |

## Working directory and resource notes

- Available disk space: approximately 351 GiB. Physical RAM is 16 GiB (`os.sysconf`).
- The source CSVs total approximately 845 MB and the cleaned CSV is approximately 839 MB. These compressed/on-disk sizes do not indicate peak memory use; the current preprocessing performs a full in-memory concatenation and deduplication and may require several times the file size in RAM.
- Header inspection verified all eight source files have the same normalized 79-column schema with `Label`; one duplicated `Fwd Header Length` is the reason the raw header contains a repeated name. A sample of 1,000 rows from one file showed the duplicate values match; the pipeline checks all processed chunks before removing that repeated column.
- Resource estimate for a 100,000-row chunk: approximately 191 MB working memory (sample-based, with 3× factor); approximately 1.11 GB output space (1.25× total input bytes). Free space observed was approximately 377 GB. These are estimates, not run measurements.
- The local `.venv` has Python 3.14.5, pandas 3.0.5, NumPy 2.5.2, scikit-learn 1.9.1, and XGBoost 3.4.1. Streamlit, Plotly, and Matplotlib are present.
- The existing cleaned CSV was scanned in 25,000-row chunks. Measured label distribution: BENIGN 2,095,057; Bot 1,948; DDoS 128,014; DoS GoldenEye 10,286; DoS Hulk 172,846; DoS Slowhttptest 5,228; DoS slowloris 5,385; FTP-Patator 5,931; Heartbleed 11; Infiltration 36; PortScan 90,694; SSH-Patator 3,219; Web Attack replacement-character Brute Force 1,470; Sql Injection 21; XSS 652. It is severely imbalanced.
- A 10,000-row memory sample estimated 1.61 GB for the loaded full DataFrame and about 6.43 GB for a 4× training working set. These are planning estimates. The bounded models were fit on 3,639 rows; fit times were 0.09 s (LR), 0.32 s (RF), and 1.45 s (XGBoost), not a full-data run.
- No credentials or secrets were inspected.
- The preprocessing implementation and documentation changed after the initial audit. No source dataset or existing cleaned CSV was written by these changes or by the tests.

## Recommended next work

1. Review `docs/MODEL_EVALUATION.md` and its saved 500-per-class comparison before deciding whether a larger experiment is worthwhile.
2. Recover or retain source capture provenance for a scenario-held-out evaluation. The current merged CSV does not support capture-file grouping and the capped test set does not preserve natural class prevalence.
3. Consider exact global duplicate handling as a separate data workflow; current splits group exact duplicates by fingerprint but do not remove repeated flows.
4. Add persistence and bounded alert history; keep anomaly detection independent of supervised class predictions.

The listed tests and bounded evaluation were run as stated. The project is not production-ready; operational ingestion, alerting, response, deployment, and natural-prevalence evaluation remain future work.
