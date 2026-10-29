"""Chunked, auditable CIC-IDS2017 CSV preprocessing.

The command never replaces an existing output. Exact deduplication is limited
to each input chunk; duplicates spanning chunks/files are reported as unchecked.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

LABEL_COLUMN = "Label"
KNOWN_LABELS = {
    "BENIGN", "Bot", "DDoS", "DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest",
    "DoS slowloris", "FTP-Patator", "Heartbleed", "Infiltration", "PortScan",
    "SSH-Patator", "Web Attack - Brute Force", "Web Attack - Sql Injection",
    "Web Attack - XSS", "Web Attack � Brute Force", "Web Attack � Sql Injection",
    "Web Attack � XSS",
}
EXPECTED_FEATURE_COLUMNS = tuple("""Destination Port
Flow Duration
Total Fwd Packets
Total Backward Packets
Total Length of Fwd Packets
Total Length of Bwd Packets
Fwd Packet Length Max
Fwd Packet Length Min
Fwd Packet Length Mean
Fwd Packet Length Std
Bwd Packet Length Max
Bwd Packet Length Min
Bwd Packet Length Mean
Bwd Packet Length Std
Flow Bytes/s
Flow Packets/s
Flow IAT Mean
Flow IAT Std
Flow IAT Max
Flow IAT Min
Fwd IAT Total
Fwd IAT Mean
Fwd IAT Std
Fwd IAT Max
Fwd IAT Min
Bwd IAT Total
Bwd IAT Mean
Bwd IAT Std
Bwd IAT Max
Bwd IAT Min
Fwd PSH Flags
Bwd PSH Flags
Fwd URG Flags
Bwd URG Flags
Fwd Header Length
Bwd Header Length
Fwd Packets/s
Bwd Packets/s
Min Packet Length
Max Packet Length
Packet Length Mean
Packet Length Std
Packet Length Variance
FIN Flag Count
SYN Flag Count
RST Flag Count
PSH Flag Count
ACK Flag Count
URG Flag Count
CWE Flag Count
ECE Flag Count
Down/Up Ratio
Average Packet Size
Avg Fwd Segment Size
Avg Bwd Segment Size
Fwd Avg Bytes/Bulk
Fwd Avg Packets/Bulk
Fwd Avg Bulk Rate
Bwd Avg Bytes/Bulk
Bwd Avg Packets/Bulk
Bwd Avg Bulk Rate
Subflow Fwd Packets
Subflow Fwd Bytes
Subflow Bwd Packets
Subflow Bwd Bytes
Init_Win_bytes_forward
Init_Win_bytes_backward
act_data_pkt_fwd
min_seg_size_forward
Active Mean
Active Std
Active Max
Active Min
Idle Mean
Idle Std
Idle Max
Idle Min""".splitlines())


@dataclass(frozen=True)
class PreprocessingConfig:
    input_dir: str = "MachineLearningCVE"
    output_dir: str = "outputs"
    chunk_size: int = 100_000
    random_seed: int = 42
    expected_labels: tuple[str, ...] = tuple(sorted(KNOWN_LABELS))
    expected_feature_columns: tuple[str, ...] = EXPECTED_FEATURE_COLUMNS
    drop_rows_with_invalid_features: bool = True
    deduplication_scope: str = "within_chunk_only"


def normalize_headers(headers: Iterable[str]) -> tuple[list[str], list[str]]:
    """Strip header whitespace and assign deterministic names to duplicates.

    CIC-IDS2017 repeats ``Fwd Header Length``. That extra column is accepted
    under a generated name and must be proven identical to its first copy
    during processing before it is removed.
    """
    names: list[str] = []
    seen: dict[str, int] = {}
    duplicates: list[str] = []
    for raw in headers:
        name = raw.strip()
        if not name:
            raise ValueError("CSV contains an empty column name after whitespace normalization")
        count = seen.get(name, 0) + 1
        seen[name] = count
        if count > 1:
            duplicates.append(name)
            name = f"{name}__duplicate_{count}"
        names.append(name)
    return names, duplicates


def read_header(path: Path) -> tuple[list[str], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        try:
            raw = next(csv.reader(stream))
        except StopIteration as exc:
            raise ValueError(f"Input CSV is empty: {path}") from exc
    return normalize_headers(raw)


def validate_schema(paths: list[Path], expected_features: Iterable[str] | None = EXPECTED_FEATURE_COLUMNS) -> tuple[list[str], dict[str, Any]]:
    """Validate headers only; does not read dataset rows into memory."""
    if not paths:
        raise ValueError("No CSV input files were found")
    reference: list[str] | None = None
    per_file: list[dict[str, Any]] = []
    for path in paths:
        names, duplicates = read_header(path)
        if LABEL_COLUMN not in names:
            raise ValueError(f"Required target column {LABEL_COLUMN!r} is missing from {path}")
        if expected_features is not None:
            actual_features = [name for name in names if name != LABEL_COLUMN and "__duplicate_" not in name]
            expected = list(expected_features)
            missing = sorted(set(expected) - set(actual_features))
            unexpected = sorted(set(actual_features) - set(expected))
            if missing or unexpected:
                raise ValueError(f"Unexpected feature schema in {path}; missing={missing}, unexpected={unexpected}")
        if reference is None:
            reference = names
        elif names != reference:
            missing = [column for column in reference if column not in names]
            extra = [column for column in names if column not in reference]
            raise ValueError(f"Inconsistent schema in {path}; missing={missing}, extra={extra}")
        per_file.append({"file": str(path), "column_count": len(names), "normalized_duplicates": duplicates})
    assert reference is not None
    return reference, {"schema_by_file": per_file, "schema_differences": []}


def clean_chunk(frame: pd.DataFrame, feature_columns: list[str], expected_labels: set[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize labels, verify duplicate columns, and drop invalid rows with counts."""
    info: dict[str, Any] = {
        "input_rows": int(len(frame)), "duplicate_columns_collapsed": [],
        "missing_values_by_column": {}, "infinite_values_by_column": {},
        "non_numeric_values_by_column": {}, "rows_missing_label": 0,
        "rows_with_invalid_features": 0, "rows_removed_for_invalid_or_missing": 0,
        "exact_duplicates_within_chunk": 0,
    }
    for column in frame.columns:
        if "__duplicate_" not in column:
            continue
        original = column.rsplit("__duplicate_", 1)[0]
        left = pd.to_numeric(frame[original], errors="coerce")
        right = pd.to_numeric(frame[column], errors="coerce")
        same = left.eq(right) | (left.isna() & right.isna())
        if not bool(same.all()):
            raise ValueError(f"Repeated source column {original!r} contains different values; refusing to discard it")
        frame.drop(columns=[column], inplace=True)
        info["duplicate_columns_collapsed"].append(original)

    if LABEL_COLUMN not in frame:
        raise ValueError(f"Required target column {LABEL_COLUMN!r} is missing")
    labels = frame[LABEL_COLUMN].astype("string").str.strip()
    observed = set(labels.dropna().unique().tolist())
    unknown = sorted(observed - expected_labels)
    if unknown:
        raise ValueError(f"Unexpected target labels: {unknown}")
    info["label_counts_before_filter"] = {str(k): int(v) for k, v in labels.value_counts(dropna=False).items()}
    info["rows_missing_label"] = int(labels.isna().sum())
    frame[LABEL_COLUMN] = labels

    invalid = pd.Series(False, index=frame.index)
    for column in feature_columns:
        raw = frame[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        non_numeric = raw.notna() & numeric.isna()
        if non_numeric.any():
            info["non_numeric_values_by_column"][column] = int(non_numeric.sum())
        missing = numeric.isna()
        if missing.any():
            info["missing_values_by_column"][column] = int(missing.sum())
        finite = np.isfinite(numeric.to_numpy(dtype="float64", na_value=np.nan))
        infinite = numeric.notna().to_numpy() & ~finite
        if infinite.any():
            info["infinite_values_by_column"][column] = int(infinite.sum())
            numeric = numeric.mask(~np.isfinite(numeric))
        frame[column] = numeric
        invalid |= numeric.isna()

    missing_target = frame[LABEL_COLUMN].isna()
    info["rows_with_invalid_features"] = int(invalid.sum())
    rows_to_remove = invalid | missing_target
    info["rows_removed_for_invalid_or_missing"] = int(rows_to_remove.sum())
    frame = frame.loc[~rows_to_remove].copy()
    before_dedup = len(frame)
    frame.drop_duplicates(inplace=True)
    info["exact_duplicates_within_chunk"] = int(before_dedup - len(frame))
    info["output_rows"] = int(len(frame))
    info["label_counts_after_filter"] = {str(k): int(v) for k, v in frame[LABEL_COLUMN].value_counts().items()}
    return frame, info


def _merge_counts(target: dict[str, int], values: dict[str, int]) -> None:
    for key, value in values.items():
        target[key] = target.get(key, 0) + value


def _publish_no_replace(temp_path: Path, destination: Path) -> None:
    """Atomically create destination without replacing an existing file."""
    os.link(temp_path, destination)


def estimate_resources(paths: list[Path], columns: list[str], chunk_size: int, output_dir: Path) -> dict[str, int | str]:
    """Estimate chunk memory from a small sample; this is not a measured peak."""
    sample_rows = min(10_000, chunk_size)
    sample = pd.read_csv(paths[0], names=columns, header=0, nrows=sample_rows, low_memory=False)
    sample_bytes = int(sample.memory_usage(index=True, deep=True).sum())
    estimated_chunk_bytes = math.ceil(sample_bytes * chunk_size / max(1, len(sample)) * 3)
    total_input_bytes = sum(path.stat().st_size for path in paths)
    estimated_output_bytes = math.ceil(total_input_bytes * 1.25)
    free_bytes = shutil.disk_usage(output_dir).free
    return {
        "sample_rows": len(sample), "estimated_chunk_working_memory_bytes": estimated_chunk_bytes,
        "estimated_output_disk_bytes": estimated_output_bytes,
        "input_bytes_total": total_input_bytes, "available_disk_bytes_at_start": free_bytes,
        "method": "sample DataFrame deep memory scaled to chunk size with a 3x working-copy factor; output estimated at 1.25x source bytes",
    }


def run(config: PreprocessingConfig) -> tuple[Path, Path, dict[str, Any]]:
    if config.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    input_dir, output_dir = Path(config.input_dir), Path(config.output_dir)
    paths = sorted(input_dir.glob("*.csv"))
    columns, schema_info = validate_schema(paths, config.expected_feature_columns)
    output_dir.mkdir(parents=True, exist_ok=True)
    resource_estimate = estimate_resources(paths, columns, config.chunk_size, output_dir)
    if resource_estimate["available_disk_bytes_at_start"] < resource_estimate["estimated_output_disk_bytes"]:
        raise OSError("Estimated output may exceed available disk space; choose another output directory or lower the data scope")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"cicids2017_cleaned_{stamp}.csv"
    report_path = output_dir / f"cicids2017_preprocessing_{stamp}.json"
    if output_path.exists() or report_path.exists():
        raise FileExistsError(f"Refusing to overwrite output/report: {output_path} or {report_path}")

    output_fd, output_tmp_name = tempfile.mkstemp(prefix=".cicids-output-", suffix=".tmp", dir=output_dir)
    report_fd, report_tmp_name = tempfile.mkstemp(prefix=".cicids-report-", suffix=".tmp", dir=output_dir)
    os.close(report_fd)
    output_tmp, report_tmp = Path(output_tmp_name), Path(report_tmp_name)
    feature_columns = [column for column in columns if column != LABEL_COLUMN and "__duplicate_" not in column]
    file_reports: list[dict[str, Any]] = []
    total_output_rows = 0
    total_rows = 0
    combined_labels: dict[str, int] = {}
    try:
        with os.fdopen(output_fd, "w", encoding="utf-8", newline="") as output_stream:
            wrote_header = False
            for path in paths:
                file_info: dict[str, Any] = {
                    "filename": str(path), "input_bytes": path.stat().st_size,
                    "input_rows": 0, "output_rows": 0, "chunks": 0,
                    "exact_duplicates_within_chunk": 0, "rows_with_invalid_features": 0,
                    "rows_removed_for_invalid_or_missing": 0,
                    "rows_missing_label": 0, "missing_values_by_column": {},
                    "infinite_values_by_column": {}, "non_numeric_values_by_column": {},
                    "duplicate_columns_collapsed": [],
                }
                reader = pd.read_csv(path, names=columns, header=0, chunksize=config.chunk_size, low_memory=False)
                for chunk in reader:
                    cleaned, info = clean_chunk(chunk, feature_columns, set(config.expected_labels))
                    file_info["chunks"] += 1
                    file_info["input_rows"] += info["input_rows"]
                    file_info["output_rows"] += info["output_rows"]
                    file_info["exact_duplicates_within_chunk"] += info["exact_duplicates_within_chunk"]
                    file_info["rows_with_invalid_features"] += info["rows_with_invalid_features"]
                    file_info["rows_removed_for_invalid_or_missing"] += info["rows_removed_for_invalid_or_missing"]
                    file_info["rows_missing_label"] += info["rows_missing_label"]
                    file_info["duplicate_columns_collapsed"] = sorted(set(file_info["duplicate_columns_collapsed"] + info["duplicate_columns_collapsed"]))
                    for key in ("missing_values_by_column", "infinite_values_by_column", "non_numeric_values_by_column"):
                        _merge_counts(file_info[key], info[key])
                    _merge_counts(combined_labels, info["label_counts_after_filter"])
                    if not wrote_header:
                        cleaned.to_csv(output_stream, index=False, header=True)
                        wrote_header = True
                    elif not cleaned.empty:
                        cleaned.to_csv(output_stream, index=False, header=False)
                if file_info["chunks"] == 0:
                    raise ValueError(f"Input CSV has a header but no data rows: {path}")
                total_rows += file_info["input_rows"]
                total_output_rows += file_info["output_rows"]
                file_reports.append(file_info)
                print(f"Processed {path.name}: {file_info['input_rows']:,} rows -> {file_info['output_rows']:,}")
            if not wrote_header:
                raise ValueError("No input records were available to preprocess")

        report = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": asdict(config), "input_files": file_reports,
            "schema": schema_info, "columns": columns,
            "resource_estimate": resource_estimate,
            "feature_columns": feature_columns, "target_column": LABEL_COLUMN,
            "total_input_rows": total_rows, "final_row_count": total_output_rows,
            "rows_removed_for_missing_or_invalid_values": sum(x["rows_removed_for_invalid_or_missing"] for x in file_reports),
            "duplicates_removed_within_chunks": sum(x["exact_duplicates_within_chunk"] for x in file_reports),
            "cross_chunk_or_cross_file_duplicates": "not checked; deduplication is strictly within each chunk",
            "final_label_distribution": combined_labels,
            "output_csv": str(output_path), "report_json": str(report_path),
            "counts_are": "measured during this run; no full-dataset run is performed by inspection mode",
        }
        report_tmp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        _publish_no_replace(output_tmp, output_path)
        try:
            _publish_no_replace(report_tmp, report_path)
        except Exception:
            output_path.unlink(missing_ok=True)
            raise
        return output_path, report_path, report
    finally:
        output_tmp.unlink(missing_ok=True)
        report_tmp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="MachineLearningCVE")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42, help="Recorded for reproducibility; preprocessing is deterministic")
    args = parser.parse_args()
    config = PreprocessingConfig(input_dir=args.input_dir, output_dir=args.output_dir,
                                 chunk_size=args.chunk_size, random_seed=args.seed)
    output_path, report_path, report = run(config)
    print(f"Output: {output_path}\nReport: {report_path}\nRows: {report['final_row_count']:,}")
    print("Label distribution:")
    for label, count in sorted(report["final_label_distribution"].items()):
        print(f"  {label}: {count:,}")


if __name__ == "__main__":
    main()
