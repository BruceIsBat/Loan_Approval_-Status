"""Evaluation helpers for model training and reporting."""

from __future__ import annotations

from typing import Any, Dict

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(y_true: Any, y_pred: Any) -> Dict[str, Any]:
    """Compute a compact evaluation report for binary classification."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Print metrics in a readable format."""
    print("Evaluation metrics:")
    for key, value in metrics.items():
        if key == "confusion_matrix":
            print(f"  {key}: {value}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
