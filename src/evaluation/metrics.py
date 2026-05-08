"""Evaluation metrics for ENSO phase prediction.

Primary metric: macro F1 — weights all three classes equally,
regardless of how often Neutral dominates the label distribution.

Secondary: accuracy, per-class F1, confusion matrix.

Spring barrier stratification: evaluate_by_spring_barrier() splits
the test set by crosses_spring_tL and reports metrics separately for
each regime — physically interpretable and scientifically meaningful.
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
    """Evaluate multiple models for one target in one call."""
    return {
        name: evaluate(y_true, y_pred, name=f"{target}/{name}")
        for name, y_pred in predictions.items()
    }


def evaluate_by_spring_barrier(
    y_true:        pd.Series,
    y_pred:        np.ndarray,
    crosses_spring: pd.Series,
    name:          str = "",
) -> dict[str, dict]:
    """Evaluate separately for spring-crossing and non-spring-crossing forecasts.

    This stratification reveals whether forecast skill degrades significantly
    when the forecast window passes through boreal spring (MAM) — the
    spring predictability barrier.

    Physical interpretation
    -----------------------
    crosses_spring=True:  forecast window contains MAM months.
                          Expect lower skill — phase-locking disrupts
                          the persistence of anomalies through spring.

    crosses_spring=False: forecast window avoids MAM.
                          Expect higher skill AND higher operational impact
                          (these forecasts target DJF, the season of
                          peak ENSO influence globally).

    Parameters
    ----------
    y_true : pd.Series
        Ground-truth labels, indexed by date.
    y_pred : np.ndarray
        Predicted labels, same length as y_true.
    crosses_spring : pd.Series
        Boolean series indicating spring-crossing, same index as y_true.
        Typically the crosses_spring_tL column from the dataset.
    name : str
        Model/target label for logging.

    Returns
    -------
    dict with keys:
        "crosses_spring"     → metrics for spring-crossing forecasts
        "no_spring_barrier"  → metrics for non-crossing forecasts
        "difference"         → F1 improvement when barrier is absent
    """
    y_pred_series = pd.Series(
        np.asarray(y_pred),
        index=y_true.index
    )

    # Align mask: valid labels + spring regime
    valid = pd.notna(y_true) & pd.notna(y_pred_series)
    cs_mask  = valid & crosses_spring.astype(bool)
    ncs_mask = valid & ~crosses_spring.astype(bool)

    result = {}

    for mask, key, label in [
        (cs_mask,  "crosses_spring",    f"{name}/crosses_spring"),
        (ncs_mask, "no_spring_barrier", f"{name}/no_spring_barrier"),
    ]:
        if mask.sum() == 0:
            print(f"[metrics] {label}: no samples — skipping")
            result[key] = None
            continue
        result[key] = evaluate(
            y_true[mask],
            y_pred_series[mask].values,
            name=label,
        )

    # F1 difference: how much does the barrier cost?
    if result.get("crosses_spring") and result.get("no_spring_barrier"):
        delta = (result["no_spring_barrier"]["f1_macro"]
                 - result["crosses_spring"]["f1_macro"])
        result["difference"] = round(delta, 4)
        print(
            f"[metrics] {name} spring barrier cost: "
            f"ΔF1 = {delta:+.3f} "
            f"({'no barrier better' if delta > 0 else 'barrier better — unexpected'})"
        )

    return result


def evaluate_by_init_month(
    y_true:   pd.Series,
    y_pred:   np.ndarray,
    name:     str = "",
) -> pd.DataFrame:
    """Compute F1 macro for each calendar month of initialization.

    Returns a DataFrame with columns: month, f1_macro, n_samples.
    Used to plot the "F1 vs initialization month" curve which directly
    visualises the spring predictability barrier as a step function.

    Parameters
    ----------
    y_true : pd.Series
        Ground-truth labels with a DatetimeIndex.
    y_pred : np.ndarray
        Predicted labels, same length as y_true.
    name : str
        Label for logging.
    """
    y_pred_series = pd.Series(np.asarray(y_pred), index=y_true.index)
    valid = pd.notna(y_true) & pd.notna(y_pred_series)

    rows = []
    for month in range(1, 13):
        mask = valid & (y_true.index.month == month)
        if mask.sum() < 3:
            rows.append({"month": month, "f1_macro": float("nan"), "n": 0})
            continue
        f1 = f1_score(
            y_true[mask],
            y_pred_series[mask].values,
            average="macro",
            zero_division=0,
            labels=LABEL_ORDER,
        )
        rows.append({"month": month, "f1_macro": round(float(f1), 4),
                     "n": int(mask.sum())})

    df = pd.DataFrame(rows)
    return df


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
