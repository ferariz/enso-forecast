"""Evaluation metrics and reporting for ENSO phase prediction.

Provides:
- per-model, per-target metrics (accuracy, F1 macro, confusion matrix)
- comparison table across models and baselines
- lead-time performance curves
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)

from src.utils.logging import get_logger

logger = get_logger(__name__)

LABEL_ORDER = ["La Niña", "Neutral", "El Niño"]


def evaluate(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
) -> dict[str, Any]:
    """Compute accuracy, macro-F1, per-class F1, and confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc   = accuracy_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm    = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    f1_pc = f1_score(
        y_true, y_pred,
        labels=LABEL_ORDER,
        average=None,
        zero_division=0,
    )

    result = {
        "label":      label,
        "accuracy":   round(acc, 4),
        "f1_macro":   round(f1, 4),
        "f1_per_class": {
            cls: round(v, 4)
            for cls, v in zip(LABEL_ORDER, f1_pc)
        },
        "confusion_matrix": cm.tolist(),
        "n_samples":  len(y_true),
    }

    logger.info(
        f"{label} | acc={acc:.3f} | f1_macro={f1:.3f} | n={len(y_true)}"
    )
    return result


def evaluate_all(
    y_true: pd.Series,
    predictions: dict[str, np.ndarray],
    target: str,
) -> dict[str, dict]:
    """Evaluate multiple models (and baselines) for one target.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    predictions:
        Dict mapping model_name → predicted labels array.
    target:
        Name of the target (e.g. ``'enso_t3'``), used for logging.
    """
    results = {}
    for name, y_pred in predictions.items():
        mask = ~pd.isnull(y_true) & ~pd.isnull(pd.Series(y_pred, index=y_true.index))
        results[name] = evaluate(y_true[mask], np.asarray(y_pred)[mask.values], label=f"{target}/{name}")
    return results


def results_to_dataframe(results: dict[str, dict[str, dict]]) -> pd.DataFrame:
    """Flatten nested {target: {model: metrics}} into a tidy DataFrame."""
    rows = []
    for target, model_results in results.items():
        for model, metrics in model_results.items():
            rows.append({
                "target":   target,
                "model":    model,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "n":        metrics["n_samples"],
                **{f"f1_{k.replace(' ', '_').lower()}": v
                   for k, v in metrics["f1_per_class"].items()},
            })
    return pd.DataFrame(rows).sort_values(["target", "f1_macro"], ascending=[True, False])


def save_metrics(results: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info(f"Metrics saved → {path}")
