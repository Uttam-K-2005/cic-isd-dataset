"""Predict one flow from a trusted local model artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.model_io import align_features, load_bundle


def predict_record(artifact_dir: str | Path, record: dict[str, Any]) -> dict[str, Any]:
    bundle = load_bundle(artifact_dir)
    frame = align_features(record, bundle["feature_names"])
    values = frame.to_numpy(dtype=np.float32)
    model = bundle["model"]
    predicted = model.predict(values)[0]
    if bundle["model_name"] == "xgboost":
        predicted = bundle["label_classes"][int(predicted)]
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(values)[0].tolist()
    return {
        "predicted_label": str(predicted), "model_name": bundle["model_name"],
        "probabilities": probabilities, "label_classes": bundle["label_classes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir")
    parser.add_argument("record_json", help="Path to a JSON object containing every expected feature")
    args = parser.parse_args()
    record = json.loads(Path(args.record_json).read_text(encoding="utf-8"))
    print(json.dumps(predict_record(args.artifact_dir, record), indent=2))


if __name__ == "__main__":
    main()
