"""SHAP-based feature importance for the LightGBM model.

Generates:
  - Global SHAP values (TreeExplainer, efficient for tree models)
  - Per-class mean |SHAP| ranking
  - Summary beeswarm plots (via evaluation.plots)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

LABEL_ORDER = ["La Niña", "Neutral", "El Niño"]


def compute_shap_values(
    trainer,          # ModelTrainer with a LightGBM estimator
    X: pd.DataFrame,
    output_dir: Path | None = None,
) -> tuple[list, pd.DataFrame]:
    """Compute SHAP values using TreeExplainer.

    Returns
    -------
    shap_values:
        List of shap arrays, one per class.
    importance_df:
        DataFrame of mean |SHAP| per feature, sorted descending.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shap is required: pip install shap")

    estimator = trainer.estimator
    explainer  = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X[trainer.feature_names_].values)

    # mean |SHAP| per feature, averaged across classes
    mean_abs = np.abs(np.stack(shap_values)).mean(axis=(0, 1))  # (n_classes, n_samples, n_features)
    importance_df = pd.DataFrame({
        "feature": trainer.feature_names_,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    logger.info("SHAP values computed")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(output_dir / "shap_importance.csv", index=False)
        logger.info(f"SHAP importance table saved → {output_dir / 'shap_importance.csv'}")

    return shap_values, importance_df
