"""Deterministic class-stratified splitting that keeps exact duplicate rows together."""

from __future__ import annotations

import numpy as np
import pandas as pd


def exact_row_groups(features: pd.DataFrame, labels: pd.Series) -> np.ndarray:
    """Fingerprint feature+target rows; identical records always receive one group."""
    combined = features.copy()
    combined["__target__"] = labels.astype(str).to_numpy()
    return pd.util.hash_pandas_object(combined, index=False, categorize=True).to_numpy(dtype="uint64")


def grouped_stratified_split(
    features: pd.DataFrame,
    labels: pd.Series,
    seed: int = 42,
    proportions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Assign identical rows to one partition and approximately stratify labels.

    Records are assigned by class-level fingerprint groups. Hash collisions can
    group distinct records together, but cannot put identical records in different
    partitions. This guards exact duplicates, not related or near-duplicate flows.
    """
    if len(features) != len(labels) or len(features) == 0:
        raise ValueError("Features and labels must have equal, nonzero length")
    if len(proportions) != 3 or any(p <= 0 for p in proportions) or not np.isclose(sum(proportions), 1):
        raise ValueError("proportions must be three positive values summing to 1")
    groups = exact_row_groups(features, labels)
    assignment: dict[int, int] = {}
    rng = np.random.default_rng(seed)
    for label in sorted(labels.astype(str).unique()):
        positions = np.flatnonzero(labels.astype(str).to_numpy() == label)
        group_counts = pd.Series(groups[positions]).value_counts()
        if len(group_counts) < 3:
            raise ValueError(f"Class {label!r} has fewer than 3 distinct exact-row groups; cannot populate all splits")
        shuffled = rng.permutation(group_counts.index.to_numpy())
        ordered = sorted(shuffled, key=lambda group: -int(group_counts[group]))
        targets = np.asarray(proportions) * len(positions)
        assigned = np.zeros(3, dtype=int)
        for index, group in enumerate(ordered):
            if index < 3:
                split = index
            else:
                ratios = assigned / np.maximum(targets, 1)
                split = int(np.argmin(ratios))
            assignment[int(group)] = split
            assigned[split] += int(group_counts[group])
    split_ids = np.fromiter((assignment[int(group)] for group in groups), dtype=np.int8, count=len(groups))
    train = np.flatnonzero(split_ids == 0)
    validation = np.flatnonzero(split_ids == 1)
    test = np.flatnonzero(split_ids == 2)
    sets = [set(groups[idx].tolist()) for idx in (train, validation, test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise AssertionError("Exact-row group leaked across partitions")
    return train, validation, test, {
        "method": "class-stratified exact-row fingerprint group split",
        "seed": seed, "requested_proportions": list(proportions),
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "distinct_groups": {"train": len(sets[0]), "validation": len(sets[1]), "test": len(sets[2])},
        "class_rows": {
            part: {str(k): int(v) for k, v in labels.iloc[idx].value_counts().items()}
            for part, idx in zip(("train", "validation", "test"), (train, validation, test))
        },
    }
