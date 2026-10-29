"""Schema validation and bounded-memory CIC-IDS2017 sampling."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from week3_preprocessing import EXPECTED_FEATURE_COLUMNS, KNOWN_LABELS, LABEL_COLUMN

LEGACY_DUPLICATE_COLUMN = "Fwd Header Length.1"


class LabelCodec:
    """Stable, explicit target encoding independent of model libraries."""

    def __init__(self, classes: list[str] | tuple[str, ...]):
        self.classes = tuple(sorted(set(classes)))
        if not self.classes:
            raise ValueError("Label classes cannot be empty")
        self.to_index = {label: i for i, label in enumerate(self.classes)}

    def encode(self, values: list[str] | pd.Series) -> np.ndarray:
        unknown = sorted(set(map(str, values)) - set(self.to_index))
        if unknown:
            raise ValueError(f"Unknown labels: {unknown}")
        return np.asarray([self.to_index[str(value)] for value in values], dtype=np.int32)

    def decode(self, values: list[int] | np.ndarray) -> list[str]:
        if any(int(value) < 0 or int(value) >= len(self.classes) for value in values):
            raise ValueError("Encoded label is outside the fitted class mapping")
        return [self.classes[int(value)] for value in values]


def separate_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if LABEL_COLUMN not in frame.columns:
        raise ValueError(f"Target column {LABEL_COLUMN!r} is required")
    features = [column for column in frame.columns if column != LABEL_COLUMN]
    if LABEL_COLUMN in features:
        raise AssertionError("Target leakage into feature matrix")
    return frame.loc[:, features].copy(), frame[LABEL_COLUMN].astype(str).copy()


def canonical_columns(path: str | Path) -> tuple[list[str], list[str]]:
    """Validate cleaned CSV header, returning canonical feature order and actual columns."""
    actual = [str(c).strip() for c in pd.read_csv(path, nrows=0).columns]
    if len(actual) != len(set(actual)):
        raise ValueError("CSV still has duplicate column names after pandas header parsing")
    expected = list(EXPECTED_FEATURE_COLUMNS)
    without_legacy_duplicate = [c for c in actual if c != LEGACY_DUPLICATE_COLUMN]
    if set(without_legacy_duplicate) != set(expected + [LABEL_COLUMN]):
        missing = sorted(set(expected + [LABEL_COLUMN]) - set(without_legacy_duplicate))
        extra = sorted(set(without_legacy_duplicate) - set(expected + [LABEL_COLUMN]))
        raise ValueError(f"Dataset schema mismatch; missing={missing}, unexpected={extra}")
    if [c for c in without_legacy_duplicate if c != LABEL_COLUMN] != expected:
        raise ValueError("Dataset feature order differs from the expected CIC-IDS2017 schema")
    if actual[-1] != LABEL_COLUMN:
        raise ValueError("Target column Label must be present and last in the cleaned dataset")
    return expected, actual


def _validate_chunk(frame: pd.DataFrame, features: list[str], labels: set[str]) -> pd.DataFrame:
    """Validate feature types and the legacy duplicate feature without dropping rows."""
    if LEGACY_DUPLICATE_COLUMN in frame.columns:
        left = pd.to_numeric(frame["Fwd Header Length"], errors="coerce")
        right = pd.to_numeric(frame[LEGACY_DUPLICATE_COLUMN], errors="coerce")
        if not bool((left.eq(right) | (left.isna() & right.isna())).all()):
            raise ValueError("Fwd Header Length.1 differs from Fwd Header Length")
        frame = frame.drop(columns=[LEGACY_DUPLICATE_COLUMN])
    target = frame[LABEL_COLUMN].astype("string").str.strip()
    unexpected = sorted(set(target.dropna().unique()) - labels)
    if unexpected:
        raise ValueError(f"Unexpected target labels: {unexpected}")
    frame[LABEL_COLUMN] = target
    invalid_counts: dict[str, int] = {}
    for column in features:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid = numeric.isna() | ~np.isfinite(numeric.to_numpy(dtype="float64", na_value=np.nan))
        if invalid.any():
            invalid_counts[column] = int(invalid.sum())
        frame[column] = numeric
    if invalid_counts:
        raise ValueError(f"Missing, non-numeric, or infinite feature values: {invalid_counts}")
    if frame[LABEL_COLUMN].isna().any():
        raise ValueError("Dataset contains rows without a Label")
    return frame


def stratified_sample(
    path: str | Path,
    per_class: int = 2_000,
    seed: int = 42,
    chunk_size: int = 25_000,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Scan and validate the entire CSV, retaining a seeded per-class reservoir.

    This is bounded in memory but reads the complete CSV. Returned samples are
    class-capped, so their class proportions are not representative of CIC-IDS2017.
    """
    if per_class < 3 or chunk_size < 1:
        raise ValueError("per_class must be >= 3 and chunk_size must be positive")
    features, actual_columns = canonical_columns(path)
    reservoirs: dict[str, pd.DataFrame] = {}
    priorities: dict[str, np.ndarray] = {}
    totals: Counter[str] = Counter()
    rng = np.random.default_rng(seed)
    rows_scanned = 0
    for chunk in pd.read_csv(path, usecols=actual_columns, chunksize=chunk_size, low_memory=False):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        chunk = _validate_chunk(chunk, features, set(KNOWN_LABELS))
        rows_scanned += len(chunk)
        labels = chunk[LABEL_COLUMN].astype(str)
        totals.update(labels.value_counts().to_dict())
        row_priorities = rng.random(len(chunk))
        for label in labels.unique():
            indices = np.flatnonzero(labels.to_numpy() == label)
            current = reservoirs.get(label)
            if current is None:
                candidates = chunk.iloc[indices].copy()
                scores = row_priorities[indices]
            else:
                candidates = pd.concat([current, chunk.iloc[indices]], ignore_index=True)
                scores = np.concatenate([priorities[label], row_priorities[indices]])
            keep_count = min(per_class, len(candidates))
            keep = np.argpartition(scores, keep_count - 1)[:keep_count]
            reservoirs[label] = candidates.iloc[keep].reset_index(drop=True)
            priorities[label] = scores[keep]
    if rows_scanned == 0:
        raise ValueError("Dataset contains no records")
    sampled = pd.concat(reservoirs.values(), ignore_index=True)
    sampled_features, y = separate_features_target(sampled)
    report = {
        "dataset_path": str(path), "rows_scanned_and_validated": rows_scanned,
        "full_label_distribution": dict(sorted(totals.items())),
        "sample_label_distribution": dict(sorted(Counter(y).items())),
        "sample_rows": len(sampled), "per_class_cap": per_class,
        "seed": seed, "chunk_size": chunk_size,
        "sample_is_class_capped_and_not_population_representative": True,
    }
    return sampled_features.loc[:, features], y, report


def estimate_full_load_memory(path: str | Path, total_rows: int, sample_rows: int = 10_000) -> dict[str, int | str]:
    features, actual_columns = canonical_columns(path)
    sample = pd.read_csv(path, usecols=actual_columns, nrows=sample_rows, low_memory=False)
    sample_bytes = int(sample.memory_usage(deep=True).sum())
    estimated_dataset_bytes = int(sample_bytes * total_rows / max(1, len(sample)))
    return {
        "rows_estimate": total_rows, "feature_count": len(features),
        "estimated_loaded_frame_bytes": estimated_dataset_bytes,
        "estimated_training_working_set_bytes": estimated_dataset_bytes * 4,
        "estimate_method": "sample DataFrame memory scaled to rows; 4x factor for split arrays, scaling, and model working copies",
    }
