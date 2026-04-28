"""Model training pipeline.

Wraps sklearn-compatible estimators with a consistent interface for
training, prediction, and serialisation.  Handles class-label encoding
internally so downstream code always works with human-readable strings.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

from src.utils.logging import get_logger

logger = get_logger(__name__)

LABEL_ORDER = ["La Niña", "Neutral", "El Niño"]   # fixed encoding


def _build_estimator(model_name: str, params: dict[str, Any]):
    """Instantiate a scikit-learn-compatible estimator from config."""
    if model_name == "logistic_regression":
        return LogisticRegression(**params)
    elif model_name == "random_forest":
        return RandomForestClassifier(**params)
    elif model_name == "lightgbm":
        if not LGB_AVAILABLE:
            raise ImportError("lightgbm is not installed. Run: pip install lightgbm")
        return lgb.LGBMClassifier(**params)
    else:
        raise ValueError(f"Unknown model: {model_name!r}")


class ModelTrainer:
    """Train a single model for a single target horizon."""

    def __init__(self, model_name: str, params: dict[str, Any]):
        self.model_name = model_name
        self.params = params
        self.estimator = _build_estimator(model_name, params)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(LABEL_ORDER)
        self.feature_names_: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> "ModelTrainer":
        self.feature_names_ = list(X_train.columns)
        y_enc = self.label_encoder.transform(y_train)
        logger.info(
            f"Training {self.model_name} | "
            f"n_samples={len(X_train)} | "
            f"n_features={X_train.shape[1]}"
        )
        self.estimator.fit(X_train.values, y_enc)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return string class predictions."""
        y_enc = self.estimator.predict(X[self.feature_names_].values)
        return self.label_encoder.inverse_transform(y_enc)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return class probabilities as a DataFrame."""
        proba = self.estimator.predict_proba(X[self.feature_names_].values)
        classes = self.label_encoder.inverse_transform(self.estimator.classes_)
        return pd.DataFrame(proba, index=X.index, columns=classes)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ModelTrainer":
        return joblib.load(path)


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target: str,
    config: dict[str, Any],
    output_dir: Path = Path("outputs/models"),
) -> dict[str, ModelTrainer]:
    """Train all enabled models for one target and persist to disk.

    Returns a dict mapping model_name → fitted ModelTrainer.
    """
    trained: dict[str, ModelTrainer] = {}

    for model_name, model_cfg in config.get("models", {}).items():
        if not model_cfg.get("enabled", True):
            continue
        params = model_cfg.get("params", {})
        trainer = ModelTrainer(model_name=model_name, params=params)
        trainer.fit(X_train, y_train)
        save_path = output_dir / target / f"{model_name}.joblib"
        trainer.save(save_path)
        trained[model_name] = trainer

    return trained
