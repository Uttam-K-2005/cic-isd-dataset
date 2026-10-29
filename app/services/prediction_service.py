"""Feature validation and inference using an already-loaded model bundle."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from ml.model_io import align_features


class PredictionService:
    def __init__(self, bundle: dict[str, Any]):
        self.bundle = bundle
        self.feature_names: list[str] = bundle["feature_names"]
        self.label_classes: list[str] = bundle["label_classes"]
        self.model_name: str = bundle["model_name"]
        self.model_version: str | None = bundle.get("model_version")
        self.model = bundle["model"]

    def predict(self, record: dict[str, Any], request_id: UUID | None = None) -> dict[str, Any]:
        return self.predict_many([record], request_id)[0]

    def predict_many(self, records: list[dict[str, Any]], request_id: UUID | None = None) -> list[dict[str, Any]]:
        if not records:
            raise ValueError("At least one record is required")
        aligned = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"Record {index} must be a JSON object")
            for name, value in record.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"Feature {name!r} in record {index} must be numeric")
                try:
                    finite = math.isfinite(float(value))
                except (OverflowError, ValueError):
                    finite = False
                if not finite:
                    raise ValueError(f"Feature {name!r} in record {index} must be finite")
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    row = align_features(record, self.feature_names).iloc[0].to_numpy(dtype=np.float32)
                if not np.isfinite(row).all():
                    raise ValueError("values must be representable as finite float32 numbers")
                aligned.append(row)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid feature record at index {index}: {exc}") from exc
        values = np.stack(aligned)
        encoded_predictions = np.asarray(self.model.predict(values)).reshape(-1)
        probability_matrix = np.asarray(self.model.predict_proba(values), dtype=np.float64)
        if len(encoded_predictions) != len(records) or probability_matrix.shape != (len(records), len(self.label_classes)):
            raise RuntimeError("Model returned an incompatible prediction shape")
        classes: list[str] = []
        for encoded in encoded_predictions:
            index = int(encoded)
            if index < 0 or index >= len(self.label_classes):
                raise RuntimeError("Model returned a class outside the stored mapping")
            classes.append(self.label_classes[index])
        if not np.isfinite(probability_matrix).all():
            raise RuntimeError("Model returned non-finite probabilities")
        total = probability_matrix.sum(axis=1)
        if not np.allclose(total, 1.0, rtol=1e-4, atol=1e-5):
            raise RuntimeError("Model probabilities do not sum to one")
        now = datetime.now(timezone.utc).isoformat()
        correlation_id = request_id or uuid4()
        return [
            {
                "request_id": correlation_id,
                "predicted_class": predicted,
                "is_attack": predicted != "BENIGN",
                "probabilities": {label: float(probability) for label, probability in zip(self.label_classes, row)},
                "model_name": self.model_name,
                "model_version": self.model_version,
                "predicted_at": now,
            }
            for predicted, row in zip(classes, probability_matrix)
        ]
