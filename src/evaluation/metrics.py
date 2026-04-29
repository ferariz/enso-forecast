"""Evaluation metrics for ENSO phase prediction.

Primary metric: macro F1 — weights all three classes equally,
regardless of how often Neutral dominates the label distribution.

Secondary: accuracy, per-class F1, confusion matrix.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)

LABEL_ORDER = ["La Niña", "Neutral", "El Niño"]


def evaluate(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    name:   str = "",
) -> dict[str, Any]:
    """Compute accuracy, macro-F1, per-class F1, confusion matrix.

    Parameters
    ----------
    y_true : array-like
        Ground-truth phase strings.
    y_pred : array-like
        Predicted phase strings.
    name : str
        Label for logging (e.g. "enso_t3/lightgbm").

    Returns
    -------
    dict with keys: name, accuracy, f1_macro, f1_per_class,
                    confusion_matrix, n_samples.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Drop rows where either is NaN / None
    mask   = pd.notna(y_true) & pd.notna(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    acc    = accuracy_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred, average="macro", zero_division=0,
                      labels=LABEL_ORDER)
    f1_pc  = f1_score(y_true, y_pred, average=None, zero_division=0,
                      labels=LABEL_ORDER)
    cm     = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)

    result = {
        "name":       name,
        "accuracy":   round(float(acc), 4),
        "f1_macro":   round(float(f1),  4),
        "f1_per_class": {
            cls: round(float(v), 4)
            for cls, v in zip(LABEL_ORDER, f1_pc)
        },
        "confusion_matrix": cm.tolist(),
        "n_samples":  int(len(y_true)),
    }

    print(f"[metrics] {name:40s}  "
          f"acc={acc:.3f}  f1_macro={f1:.3f}  n={len(y_true)}")
    return result


def compare(
    y_true:      pd.Series,
    predictions: dict[str, np.ndarray],
    target:      str,
) -> dict[str, dict]:
    """Evaluate multiple models for one target in one call.

    Parameters
    ----------
    y_true : pd.Series
        Ground-truth labels.
    predictions : dict
        model_name → predicted labels array.
    target : str
        Target name for logging (e.g. "enso_t3").

    Returns
    -------
    dict mapping model_name → metrics dict.
    """
    return {
        name: evaluate(y_true, y_pred, name=f"{target}/{name}")
        for name, y_pred in predictions.items()
    }


def to_dataframe(results: dict[str, dict[str, dict]]) -> pd.DataFrame:
    """Flatten {target: {model: metrics}} into a tidy DataFrame."""
    rows = []
    for target, model_results in results.items():
        for model, m in model_results.items():
            rows.append({
                "target":              target,
                "model":               model,
                "accuracy":            m["accuracy"],
                "f1_macro":            m["f1_macro"],
                "f1_la_nina":          m["f1_per_class"].get("La Niña", 0),
                "f1_neutral":          m["f1_per_class"].get("Neutral",  0),
                "f1_el_nino":          m["f1_per_class"].get("El Niño",  0),
                "n":                   m["n_samples"],
            })
    return (
        pd.DataFrame(rows)
          .sort_values(["target", "f1_macro"], ascending=[True, False])
          .reset_index(drop=True)
    )


def save(results: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"[metrics] Saved → {path}")
