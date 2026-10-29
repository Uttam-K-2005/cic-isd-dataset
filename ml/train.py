"""Train Logistic Regression, Random Forest, and XGBoost on a bounded sample."""

from __future__ import annotations

import argparse
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ml.data import estimate_full_load_memory, stratified_sample
from ml.metrics import auc_metrics, classification_metrics
from ml.model_io import save_bundle
from ml.models import build_models, fit_model, predict_labels
from ml.splitting import grouped_stratified_split


def run_training(
    data_path: str | Path,
    artifact_root: str | Path = "artifacts/models",
    metrics_root: str | Path = "artifacts/metrics",
    seed: int = 42,
    per_class: int = 2_000,
    chunk_size: int = 25_000,
    n_jobs: int = 2,
    model_names: list[str] | None = None,
) -> dict[str, Any]:
    features, target, sample_report = stratified_sample(data_path, per_class, seed, chunk_size)
    full_memory_estimate = estimate_full_load_memory(data_path, sample_report["rows_scanned_and_validated"])
    sample_report["estimated_sample_working_set_bytes"] = int(features.memory_usage(index=True, deep=True).sum() * 4)
    train_idx, validation_idx, test_idx, split_report = grouped_stratified_split(features, target, seed)
    labels = sorted(target.unique().tolist())
    train_x = features.iloc[train_idx].to_numpy(dtype=np.float32, copy=True)
    validation_x = features.iloc[validation_idx].to_numpy(dtype=np.float32, copy=True)
    test_x = features.iloc[test_idx].to_numpy(dtype=np.float32, copy=True)
    train_y = target.iloc[train_idx].to_numpy(dtype=str)
    validation_y = target.iloc[validation_idx].to_numpy(dtype=str)
    test_y = target.iloc[test_idx].to_numpy(dtype=str)
    model_config = {"seed": seed, "n_jobs": n_jobs, "per_class_cap": per_class,
                    "chunk_size": chunk_size, "feature_count": len(features.columns),
                    "label_classes": labels,
                    "preprocessing": "strict finite numeric schema; Fwd Header Length.1 verified and removed",
                    "class_weights": "balanced for Logistic Regression and Random Forest; inverse-frequency sample weights for XGBoost"}
    selected_names = model_names or ["logistic_regression", "random_forest", "xgboost"]
    if not selected_names:
        raise ValueError("At least one model must be selected")
    if len(set(selected_names)) != len(selected_names):
        raise ValueError("Model names must not be repeated")
    if n_jobs < 1 or n_jobs > 2:
        raise ValueError("n_jobs must be 1 or 2 to keep this laptop responsive")
    models = build_models(seed=seed, n_jobs=n_jobs, requested=selected_names, class_count=len(labels))
    hyperparameters = {
        "logistic_regression": {"scaler": "StandardScaler", "solver": "lbfgs", "max_iter": 500, "class_weight": "balanced"},
        "random_forest": {"n_estimators": 150, "max_depth": 20, "min_samples_leaf": 2,
                          "max_features": "sqrt", "class_weight": "balanced_subsample", "n_jobs": n_jobs},
        "xgboost": {"objective": "multi:softprob", "num_class": len(labels), "tree_method": "hist",
                    "n_estimators": 150, "max_depth": 8, "learning_rate": 0.1,
                    "subsample": 0.8, "colsample_bytree": 0.8, "n_jobs": n_jobs},
    }
    model_config["hyperparameters"] = {name: hyperparameters[name] for name in selected_names}

    validation_results: dict[str, Any] = {}
    fitted: dict[str, Any] = {}
    latencies: dict[str, float] = {}
    for name in selected_names:
        model = models[name]
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit_model(model, name, train_x, train_y, labels)
        train_seconds = time.perf_counter() - start
        convergence_warnings = [str(item.message) for item in caught if "conver" in str(item.message).lower()]
        predictions = predict_labels(model, name, validation_x, labels)
        validation_probabilities = model.predict_proba(validation_x)
        validation_results[name] = {
            "metrics": classification_metrics(validation_y, predictions, labels),
            "auc_metrics": auc_metrics(validation_y, validation_probabilities, labels),
            "training_seconds": train_seconds, "convergence_warnings": convergence_warnings,
        }
        fitted[name] = model
        started = time.perf_counter()
        probe_size = min(1000, len(validation_x))
        predict_labels(model, name, validation_x[:probe_size], labels)
        latencies[name] = (time.perf_counter() - started) / probe_size

    best_name = max(selected_names, key=lambda name: validation_results[name]["metrics"]["macro_f1"])
    # The test partition is first used here, after validation-based model selection.
    test_results = {}
    for name, model in fitted.items():
        predictions = predict_labels(model, name, test_x, labels)
        probabilities = model.predict_proba(test_x)
        test_results[name] = {
            "metrics": classification_metrics(test_y, predictions, labels),
            "auc_metrics": auc_metrics(test_y, probabilities, labels),
        }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metrics_dir = Path(metrics_root) / f"comparison_{timestamp}"
    metrics_dir.mkdir(parents=True, exist_ok=False)
    artifact_paths: dict[str, str] = {}
    for name, model in fitted.items():
        bundle = {
            "model": model, "model_name": name,
            "feature_names": list(features.columns), "label_classes": labels,
            "seed": seed, "config": model_config,
            "hyperparameters": hyperparameters[name],
            "split_method": split_report["method"],
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_version": timestamp,
            "python_version": platform.python_version(), "numpy_version": np.__version__,
            "dependency_versions": _dependency_versions(),
        }
        artifact_paths[name] = str(save_bundle(bundle, artifact_root, name))
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": sample_report, "split": split_report, "configuration": model_config,
        "validation_results": validation_results, "selected_by_validation_macro_f1": best_name,
        "test_results": test_results, "inference_seconds_per_record_on_1000_record_probe": latencies,
        "artifact_paths": artifact_paths, "dependency_versions": _dependency_versions(),
        "result_scope": "class-capped subset; metrics are not estimates for the natural CIC-IDS2017 population",
        "limitations": [
            "Test rows are drawn from a per-class capped sample, not the natural dataset prevalence.",
            "Exact duplicate rows are grouped, but related/near-duplicate flows can cross partitions.",
            "The cleaned CSV has no capture-file provenance, so scenario-held-out evaluation is unavailable.",
        ],
    }
    result["dataset_memory_estimate"] = full_memory_estimate
    metrics_path = metrics_dir / "evaluation.json"
    result["metrics_path"] = str(metrics_path)
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return result


def _dependency_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {"python": platform.python_version(), "numpy": np.__version__}
    for module, key in (("pandas", "pandas"), ("sklearn", "scikit_learn"), ("xgboost", "xgboost")):
        try:
            imported = __import__(module)
            result[key] = imported.__version__
        except ImportError:
            result[key] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="cicids2017_cleaned.csv")
    parser.add_argument("--artifact-root", default="artifacts/models")
    parser.add_argument("--metrics-root", default="artifacts/metrics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-class", type=int, default=2_000,
                        help="Reservoir cap per class; this creates a class-capped subset, not population-representative metrics")
    parser.add_argument("--chunk-size", type=int, default=25_000)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--models", default="logistic_regression,random_forest,xgboost",
                        help="Comma-separated baseline names; XGBoost is required by default")
    parser.add_argument("--dry-run", action="store_true", help="Scan and validate data, report sample/split and estimates; do not fit models")
    args = parser.parse_args()
    if args.dry_run:
        from ml.splitting import grouped_stratified_split
        x, y, report = stratified_sample(args.data, args.per_class, args.seed, args.chunk_size)
        _, _, _, split = grouped_stratified_split(x, y, args.seed)
        report["split"] = split
        report["estimated_sample_working_set_bytes"] = int(x.memory_usage(index=True, deep=True).sum() * 4)
        report["estimated_full_load_memory"] = estimate_full_load_memory(args.data, report["rows_scanned_and_validated"])
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    result = run_training(args.data, args.artifact_root, args.metrics_root, args.seed,
                          args.per_class, args.chunk_size, args.n_jobs,
                          [name.strip() for name in args.models.split(",") if name.strip()])
    print(f"Validation winner: {result['selected_by_validation_macro_f1']}")
    print(f"Metrics: {result['metrics_path']}")
    for name, path in result["artifact_paths"].items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
