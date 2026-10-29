"""Export the saved model's input schema and observed numeric-domain metadata.

The feature names and label order come from the saved model metadata. Negative
input allowance is profiled from every row of the local cleaned CSV; the CSV is
read only and never copied into the frontend bundle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "artifacts/models/xgboost_20261029T152352Z"
DEFAULT_DATASET = ROOT / "cicids2017_cleaned.csv"
OUTPUT = ROOT / "frontend/src/data/model-schema.json"


def feature_group(name: str) -> str:
    if "Bulk" in name:
        return "Bulk statistics"
    if "Flag" in name:
        return "TCP flags"
    if any(token in name for token in ("IAT", "Duration", "Active", "Idle", "/s", "Down/Up")):
        return "Timing and rates"
    if any(token in name for token in ("Header", "Win_bytes", "seg_size")):
        return "Headers and windows"
    if "Subflow" in name:
        return "Subflow metrics"
    return "Packet counts and sizes"


def export_schema(artifact_dir: Path, dataset_path: Path, output_path: Path, chunk_size: int = 50_000) -> dict[str, Any]:
    metadata_path = artifact_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_names = metadata.get("feature_names")
    if not isinstance(feature_names, list) or len(feature_names) != 77 or "Label" in feature_names:
        raise ValueError("Saved artifact metadata does not contain the expected 77-feature inference schema")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Saved model feature names must be unique")

    negative_counts = {name: 0 for name in feature_names}
    minimums = {name: float("inf") for name in feature_names}
    scanned_rows = 0
    for chunk in pd.read_csv(dataset_path, usecols=feature_names, chunksize=chunk_size, low_memory=False):
        scanned_rows += len(chunk)
        for name in feature_names:
            values = pd.to_numeric(chunk[name], errors="coerce")
            minimums[name] = min(minimums[name], float(values.min()))
            negative_counts[name] += int((values < 0).sum())
    if scanned_rows == 0:
        raise ValueError("Dataset contains no rows to profile")

    features = [
        {
            "name": name,
            "group": feature_group(name),
            "dtype": "number",
            "allowsNegative": negative_counts[name] > 0,
            "minimumObserved": minimums[name],
            "negativeRowsObserved": negative_counts[name],
        }
        for name in feature_names
    ]
    result = {
        "modelName": metadata["model_name"],
        "modelVersion": metadata["model_version"],
        "labelClasses": metadata["label_classes"],
        "inputDtype": "float32",
        "featureCount": len(features),
        "batchLimitDefault": 64,
        "featureDomainSource": "full local cleaned CSV scan; negative inputs enabled only when observed in source data",
        "profiledRows": scanned_rows,
        "features": features,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()
    result = export_schema(args.artifact_dir, args.dataset, args.output, args.chunk_size)
    print(f"Exported {result['featureCount']} features for {result['modelName']} {result['modelVersion']}")
    print(f"Profiled {result['profiledRows']:,} local rows; wrote {args.output}")


if __name__ == "__main__":
    main()
