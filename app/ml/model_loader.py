"""Load and validate the existing trusted XGBoost artifact without training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.model_io import load_bundle


def load_xgboost_bundle(artifact_dir: str | Path) -> dict[str, Any]:
    directory = Path(artifact_dir)
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError("Model metadata is missing")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Model metadata is unreadable") from exc

    # This is joblib/pickle deserialization. Only load project-generated files
    # from a trusted local artifact directory; request uploads are never loaded.
    bundle = load_bundle(directory)
    feature_names = bundle.get("feature_names")
    label_classes = bundle.get("label_classes")
    if bundle.get("model_name") != "xgboost":
        raise ValueError("Configured artifact is not an XGBoost model")
    if not isinstance(feature_names, list) or not feature_names or not all(isinstance(name, str) for name in feature_names):
        raise ValueError("Model artifact has an invalid feature schema")
    if len(feature_names) != len(set(feature_names)) or "Label" in feature_names:
        raise ValueError("Model artifact feature schema is duplicated or contains the target")
    if not isinstance(label_classes, list) or not label_classes or not all(isinstance(label, str) for label in label_classes):
        raise ValueError("Model artifact has an invalid class mapping")
    if len(label_classes) != len(set(label_classes)):
        raise ValueError("Model artifact class mapping contains duplicate labels")
    if metadata.get("feature_names") != feature_names or metadata.get("label_classes") != label_classes:
        raise ValueError("Model metadata schema does not match the serialized artifact")
    for key in ("model_name", "model_version"):
        if metadata.get(key) != bundle.get(key):
            raise ValueError(f"Model metadata {key} does not match the serialized artifact")
    model = bundle.get("model")
    if model is None or not callable(getattr(model, "predict", None)) or not callable(getattr(model, "predict_proba", None)):
        raise ValueError("XGBoost artifact does not support prediction and probabilities")
    try:
        model_classes = np.asarray(model.classes_)
        feature_count = int(model.get_booster().num_features())
    except Exception as exc:
        raise ValueError("Serialized XGBoost model is incompatible") from exc
    if not np.array_equal(model_classes, np.arange(len(label_classes))):
        raise ValueError("XGBoost class order does not match the stored label mapping")
    if feature_count != len(feature_names):
        raise ValueError("XGBoost feature count does not match the stored feature schema")
    return bundle
