"""Dependency-light multiclass evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def classification_metrics(y_true: list[str] | np.ndarray, y_pred: list[str] | np.ndarray, labels: list[str]) -> dict[str, Any]:
    if len(y_true) != len(y_pred) or not labels:
        raise ValueError("y_true/y_pred must have equal length and labels cannot be empty")
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    index = {label: i for i, label in enumerate(labels)}
    for truth, predicted in zip(y_true, y_pred):
        if truth not in index or predicted not in index:
            raise ValueError("Encountered a label outside the declared label order")
        matrix[index[truth], index[predicted]] += 1
    total = int(matrix.sum())
    per_class: dict[str, dict[str, float | int]] = {}
    precisions, recalls, f1s, supports, false_positives, true_negatives = [], [], [], [], [], []
    for i, label in enumerate(labels):
        tp = int(matrix[i, i])
        fp = int(matrix[:, i].sum() - tp)
        fn = int(matrix[i, :].sum() - tp)
        tn = total - tp - fp - fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        support = int(matrix[i, :].sum())
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support, "false_positive_rate_ovr": fpr}
        precisions.append(precision); recalls.append(recall); f1s.append(f1); supports.append(support)
        false_positives.append(fp); true_negatives.append(tn)
    support_total = sum(supports)
    return {
        "accuracy": float(np.trace(matrix) / total) if total else 0.0,
        "macro_precision": float(np.mean(precisions)), "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "weighted_f1": float(np.average(f1s, weights=supports)) if support_total else 0.0,
        "macro_false_positive_rate_ovr": float(np.mean([v["false_positive_rate_ovr"] for v in per_class.values()])),
        "micro_false_positive_rate_ovr": float(sum(false_positives) / max(1, sum(fp + tn for fp, tn in zip(false_positives, true_negatives)))),
        "per_class": per_class, "confusion_matrix": matrix.tolist(), "labels": labels,
    }


def auc_metrics(y_true: list[str] | np.ndarray, probabilities: np.ndarray, labels: list[str]) -> dict[str, Any]:
    """Calculate one-vs-rest ROC-AUC and average precision when sklearn is installed."""
    if probabilities.shape != (len(y_true), len(labels)):
        raise ValueError("Probability matrix shape must be rows × declared labels")
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError:
        return {"available": False, "reason": "scikit-learn is not installed"}
    y_array = np.asarray(y_true)
    per_class: dict[str, dict[str, float | None]] = {}
    roc_values, pr_values = [], []
    for index, label in enumerate(labels):
        truth = y_array == label
        if truth.all() or (~truth).all():
            roc = pr = None
        else:
            roc = float(roc_auc_score(truth, probabilities[:, index]))
            pr = float(average_precision_score(truth, probabilities[:, index]))
            roc_values.append(roc)
            pr_values.append(pr)
        per_class[label] = {"roc_auc_ovr": roc, "average_precision_ovr": pr}
    return {
        "available": True,
        "macro_roc_auc_ovr": float(np.mean(roc_values)) if roc_values else None,
        "macro_average_precision_ovr": float(np.mean(pr_values)) if pr_values else None,
        "per_class": per_class,
    }
