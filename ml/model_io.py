"""Versioned trusted-local model artifact persistence and feature validation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def save_bundle(bundle: dict[str, Any], artifact_root: str | Path, model_name: str) -> Path:
    import joblib

    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = root / f"{model_name}_{version}"
    destination.mkdir(exist_ok=False)
    joblib.dump(bundle, destination / "model.joblib", compress=3)
    metadata = {key: value for key, value in bundle.items() if key not in {"model", "label_codec"}}
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return destination


def load_bundle(artifact_dir: str | Path) -> dict[str, Any]:
    path = Path(artifact_dir) / "model.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact is missing: {path}")
    import joblib

    # joblib/pickle can execute code: only load artifacts created by this project
    # from a trusted local source. Do not accept user uploads here.
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or not bundle.get("feature_names"):
        raise ValueError("Model artifact has no valid feature schema")
    return bundle


def align_features(record: dict[str, Any], feature_names: list[str]):
    expected, received = set(feature_names), set(record)
    missing, unexpected = sorted(expected - received), sorted(received - expected)
    if missing or unexpected:
        raise ValueError(f"Feature mismatch; missing={missing}, unexpected={unexpected}")
    import pandas as pd
    import numpy as np

    try:
        values = np.asarray([record[name] for name in feature_names], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Every model feature must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError("Model features must be finite; NaN and infinity are not accepted")
    return pd.DataFrame([values], columns=feature_names)
