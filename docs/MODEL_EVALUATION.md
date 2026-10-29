# Model training and evaluation

## Dataset inspection

`cicids2017_cleaned.csv` exists (2,520,798 rows, 79 parsed columns, 880,193,731 bytes). A streaming, chunked pass read every row without materializing the file in memory: all 77 expected numeric features were valid and finite, the legacy duplicate `Fwd Header Length.1` matched `Fwd Header Length`, and all 15 observed labels were recognized by the project after adding the three literal `Web Attack � ...` values. Those replacement characters occur in the dataset itself and are preserved as labels; they indicate lost source punctuation and need provenance review before presenting those names as canonical.

No preprocessing report exists for this file, so its creation history and global duplicate count are unknown. Its row count matches the historical cleaned-row count, but this alone does not prove every prior transformation. The chunked validation above confirms its current schema and usable numeric values.

The local machine reports 16 GiB physical memory and about 351 GiB free disk. A 10,000-row sample estimated a full DataFrame of about 1.61 GB and a 4× training working set of about 6.43 GB. The 4× factor is a planning estimate, not measured peak memory. The implemented initial comparison avoids that full training working set by retaining at most 2,000 rows per class (about 30,000 rows; classes with fewer examples retain all their records) while streaming through and validating the full CSV.

## Split design

The sample is divided into 60% train, 20% validation, and 20% test, stratified by label at the exact-row group level. A deterministic pandas row fingerprint is calculated over all features plus `Label`; identical records receive the same group and cannot cross partitions. Hash collisions can group distinct records together, but cannot split exact duplicates apart. The split asserts that fingerprint sets are disjoint.

Validation macro F1 selects the preferred model. Test predictions are calculated only after that selection; latency probes use validation rows. The test set is still drawn from the class-capped sample, so it does **not** preserve the natural dataset class distribution. Its metrics are diagnostic subset metrics, not population performance. Rare classes have very small test support (for example, only 11 Heartbleed records exist in the whole file). Do not use these results to claim realistic operational false-alarm rates.

A simple row-level stratified split can put repeated or closely related traffic in both train and test and can look overly optimistic. Grouping exact duplicate rows addresses exact repetition in the sampled data. The cleaned file no longer records source capture filenames, so this implementation cannot hold out entire capture periods/scenarios. Near-duplicate and scenario overlap remain limitations. A future source-file evaluation should preserve provenance during preprocessing and hold out complete capture files where all classes remain evaluable.

## Features and preprocessing

- `Label` is separated from model inputs and encoded explicitly for XGBoost. The artifact stores the exact label class order and feature order.
- All expected input columns are numeric flow features. There are no categorical inputs to encode; a schema mismatch is rejected rather than silently accepted.
- Logistic Regression uses `StandardScaler` inside a scikit-learn pipeline. The scaler is fitted only when that pipeline is fitted on training rows.
- Random Forest uses raw numeric features; scaling is unnecessary for this tree model.
- XGBoost uses `multi:softprob`, class count from training labels, `hist`, and two worker threads by default.
- Logistic Regression and Random Forest use class weights; XGBoost receives inverse-frequency sample weights. The capped sample is approximately balanced for common classes, so this mostly protects the rare classes with fewer than the cap; it does not represent deployment prevalence.
- No oversampling is used. SMOTE on millions of rows would add cost and synthetic flows can be unrealistic; it must be evaluated separately and only on training data.

## Models and metrics

The common implementation compares Logistic Regression, Random Forest, and XGBoost with the same sample, partitions, label mapping, and seed. Defaults are modest: Logistic Regression with `lbfgs` and `max_iter=500`; Random Forest with 150 trees, depth 20, leaf minimum 2, and `n_jobs=2`; XGBoost with 150 histogram trees, depth 8, and `n_jobs=2`. Convergence warnings are captured in the report.

Metrics are computed from the confusion matrix:

- Accuracy = correct predictions / all predictions.
- Per-class precision = TP / (TP + FP); recall = TP / (TP + FN); F1 is their harmonic mean. Zero-denominator values are set to 0.
- Macro precision, recall, and F1 are unweighted averages across all labels. Weighted F1 averages per-class F1 by test support.
- For each class as one-vs-rest, false-positive rate = FP / (FP + TN). Both macro OVR FPR and micro OVR FPR (sum FP divided by summed OVR negatives) are saved.
- When scikit-learn is installed, one-vs-rest ROC-AUC and average precision are also calculated per class and macro-averaged. These measures are not substitutes for reporting class support and natural-prevalence metrics.

The bounded comparison was run with scikit-learn 1.9.1 and XGBoost 3.4.1. On this macOS Apple Silicon environment, XGBoost also required the separately installed Homebrew `libomp` runtime. See the [scikit-learn installation guide](https://scikit-learn.org/stable/install) and [XGBoost installation guide](https://xgboost.readthedocs.io/en/stable/install.html) for environment setup.

Install optional model dependencies in the active environment:

```bash
.venv/bin/python -m pip install -r requirements-ml.txt
```

Run a bounded data-validation/split dry run (no fit):

```bash
.venv/bin/python -m ml.train --data cicids2017_cleaned.csv --per-class 2000 --dry-run
```

Run all three bounded subset comparisons:

```bash
.venv/bin/python -m ml.train --data cicids2017_cleaned.csv --per-class 2000 --n-jobs 2
```

Choose just Logistic Regression and Random Forest if XGBoost is intentionally unavailable:

```bash
.venv/bin/python -m ml.train --models logistic_regression,random_forest --per-class 2000 --n-jobs 2
```

Artifacts and metrics use timestamped paths under `artifacts/models/` and `artifacts/metrics/`; existing files are not replaced. `joblib` artifacts use pickle serialization and must only be loaded from trusted project-generated files.

## Bounded comparison results

Run on 2026-10-29 with seed 42, 500 sampled rows per class maximum, 6,068 sample rows total, and the fixed exact-row grouped 60/20/20 split (3,639 train / 1,216 validation / 1,213 test). The full 2,520,798-row CSV was streamed and validated. Model training itself took 0.09 s for Logistic Regression, 0.32 s for Random Forest, and 1.45 s for XGBoost on this machine. Test metrics below are from the held-out test partition after selecting XGBoost from validation macro F1.

| Model | Validation macro F1 | Test accuracy | Test macro F1 | Test weighted F1 | Test macro OVR ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8111 | 0.8500 | 0.8152 | 0.8374 | 0.9810 |
| Random Forest | 0.9052 | 0.9233 | 0.9049 | 0.9222 | 0.9910 |
| XGBoost | **0.9124** | **0.9233** | **0.9260** | **0.9232** | **0.9957** |

The validation winner is XGBoost. These metrics are actual measurements for this capped sample, not estimates for natural CIC-IDS2017 prevalence. Rare classes remain weakly supported: the test set contains 2 Heartbleed and 4 Web Attack Sql Injection rows (the full file contains 11 and 21 respectively), so those per-class scores are unstable. The capped sampling also changes class prevalence, making its false-positive rates unsuitable as operational alert rates. Exact duplicate grouping does not prevent near-duplicate or capture-scenario leakage; the merged CSV has no source-capture provenance.

Saved outputs are in `artifacts/metrics/comparison_20261029T152352Z/evaluation.json` and the matching timestamped directories under `artifacts/models/`. A single-record prediction from the saved XGBoost bundle was exercised successfully and returned 15 class probabilities. No natural-prevalence or capture-held-out evaluation has been run.
