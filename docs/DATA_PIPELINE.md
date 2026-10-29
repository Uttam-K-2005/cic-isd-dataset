# Data pipeline

## Current preprocessing command

The script processes the CIC-IDS2017 input files one at a time and in bounded pandas chunks. It validates all headers before reading rows, strips header whitespace, requires `Label`, and rejects inconsistent schemas. It preserves source attack-label spelling apart from trimming surrounding whitespace and rejects labels outside the documented CIC-IDS2017 label set.

```bash
.venv/bin/python week3_preprocessing.py \
  --input-dir MachineLearningCVE \
  --output-dir outputs \
  --chunk-size 100000 \
  --seed 42
```

The command creates a timestamped cleaned CSV and a JSON quality report in the chosen output directory. It refuses to replace an output or report with the same name. The original CSVs and the pre-existing `cicids2017_cleaned.csv` are not output targets.

Header-only inspection found eight input files, each with 79 raw columns and the same order after whitespace normalization. The canonical schema has 77 expected model feature columns plus `Label`. The raw header repeats `Fwd Header Length`; the pipeline retains the first and verifies the second is identical chunk by chunk before dropping it. The existing cleaned file has 79 parsed columns because pandas previously renamed the repeat to `Fwd Header Length.1`.

The cleaned CSV scan measured 2,520,798 rows. A label-only streaming scan of the eight raw files measured 2,830,743 source rows. The cleaned file matches the previously reported cleaned row count, but no run-specific report accompanies it. Its three Web Attack labels contain a literal Unicode replacement character; the preprocessing label allowlist preserves these exact observed values rather than silently renaming them.

For `--chunk-size 100000`, a 10,000-row sample from the first source file estimated about 191 MB of chunk working memory using a 3× DataFrame memory multiplier. Estimated output space was about 1.11 GB (1.25× total source bytes); available space at inspection was about 377 GB. These are estimates, not measured peak use or output size. The pipeline records the same estimate at run start and refuses to start if its output-space estimate exceeds free disk space.

## Quality and duplicate policy

- Every numeric feature is parsed as numeric. Non-numeric values are counted and become invalid values; rows with missing/invalid features or a missing target are removed, with counts recorded in the report.
- Positive and negative infinity are counted separately and treated as invalid feature values.
- The CIC-IDS2017 source repeats the `Fwd Header Length` header. After header normalization, the pipeline compares the repeated values in each chunk and drops the repeated column only when every value matches. Any mismatch aborts the run.
- Exact duplicate rows are removed within each chunk only. Duplicates across chunk and file boundaries are not checked. The report states this limitation. A full global deduplication could require a large external index and additional disk I/O; it should be an explicit follow-up with an assessed disk budget, not an invisible memory-heavy operation.
- The report includes input files, byte sizes, per-file row counts, column/schema details, invalid-value counts, duplicate counts, removed rows, final label distribution, configuration, and output paths. These are measured during an actual run; inspecting the schema does not run the pipeline.

## Feature and label boundary

`Label` is the target. `feature_columns` in the report explicitly excludes it and the verified redundant repeated header. No identifier or metadata exclusion list has been evaluated yet; this dataset contains flow-statistic fields but no separately defined record ID. Review each feature for leakage before training, especially if future data sources add timestamps, capture identifiers, or fields derived from the target.

The preprocessing command does not fit an imputer, encoder, scaler, feature selector, or model. The `ml` training package validates and chunk-scans the existing cleaned CSV, then keeps a reproducible per-class capped sample in memory. It rejects missing/nonnumeric/nonfinite model features. All current CICIDS features are numeric, so no categorical encoder is needed; any future categorical schema is rejected until an explicit encoder with unseen-category behavior is added. Logistic Regression fits its scaler inside a pipeline on the train split only. Tree models receive unscaled features.

## Future evaluation split

The ML baseline fingerprints sampled feature+target rows and assigns each exact-row group to a single train/validation/test partition with approximate per-class balance. This prevents exact duplicate rows from crossing the three sample partitions without deleting legitimate repeated flows. Related/near-duplicate flows may still cross. The cleaned CSV lost source-file provenance, so capture-period grouping cannot be done there. Validation chooses among models; test metrics are calculated after selection. Because the initial sample is capped per class, the test sample does not have natural CICIDS2017 prevalence and is only diagnostic.

No model training or full-dataset preprocessing was run as part of implementing this pipeline. The full data run should first be planned with the selected chunk size and available memory/disk in mind.
