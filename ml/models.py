"""Baseline model definitions; heavy ML libraries are imported only on use."""

from __future__ import annotations

from typing import Any

import numpy as np


def build_models(seed: int = 42, n_jobs: int = 2, requested: list[str] | None = None,
                 class_count: int | None = None) -> dict[str, Any]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required. Install requirements-ml.txt in the active environment.") from exc
    available = {"logistic_regression", "random_forest", "xgboost"}
    requested = requested or sorted(available)
    unknown = sorted(set(requested) - available)
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}")
    models: dict[str, Any] = {}
    if "logistic_regression" in requested:
        models["logistic_regression"] = make_pipeline(
            StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=500, class_weight="balanced", random_state=seed)
        )
    if "random_forest" in requested:
        models["random_forest"] = RandomForestClassifier(
            n_estimators=150, max_depth=20, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", random_state=seed, n_jobs=n_jobs,
        )
    if "xgboost" in requested:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("XGBoost is required for the requested comparison. Install requirements-ml.txt; no substitute will be used.") from exc
        models["xgboost"] = XGBClassifier(
            objective="multi:softprob", eval_metric="mlogloss", tree_method="hist",
            num_class=class_count,
            n_estimators=150, max_depth=8, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, random_state=seed, n_jobs=n_jobs,
        )
    return models


def fit_model(model: Any, name: str, x_train: np.ndarray, y_train: np.ndarray, labels: list[str]) -> Any:
    if name == "xgboost":
        mapping = {label: index for index, label in enumerate(labels)}
        encoded = np.asarray([mapping[value] for value in y_train], dtype=np.int32)
        counts = np.bincount(encoded, minlength=len(labels))
        weights = np.asarray([len(encoded) / (len(labels) * counts[value]) for value in encoded])
        model.fit(x_train, encoded, sample_weight=weights)
    else:
        model.fit(x_train, y_train)
    return model


def predict_labels(model: Any, name: str, x: np.ndarray, labels: list[str]) -> np.ndarray:
    predictions = model.predict(x)
    if name == "xgboost":
        return np.asarray([labels[int(value)] for value in predictions])
    return predictions.astype(str)
