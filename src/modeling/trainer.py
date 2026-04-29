"""Model training wrapper.

A single ModelTrainer class handles any sklearn-compatible estimator.
Label encoding (string ↔ int) is managed internally — callers always
work with human-readable phase strings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Fixed label order — consistent across all models and evaluation
LABEL_ORDER = ["La Niña", "Neutral", "El Niño"]


def _build_estimator(name: str, params: dict[str, Any]):
    """Instantiate an estimator from a name + params dict."""
    if name == "logistic_regression":
        return LogisticRegression(**params)
    elif name == "random_forest":
        return RandomForestClassifier(**params)
    elif name == "lightgbm":
        try:
            import lightgbm as lgb
            return lgb.LGBMClassifier(**params)
        except ImportError:
            raise ImportError("lightgbm not installed — run: pip install lightgbm")
    else:
        raise ValueError(f"Unknown model name: '{name}'")


class ModelTrainer:
    """Train, predict, and persist a single classification model.

    Parameters
    ----------
    model_name : str
        One of: logistic_regression | random_forest | lightgbm
    params : dict
        Hyperparameters passed directly to the estimator constructor.
    """

    def __init__(self, model_name: str, params: dict[str, Any]):
        self.model_name = model_name
        self.params     = params
        self.estimator  = _build_estimator(model_name, params)

        # Fit LabelEncoder on the fixed order so encoding is deterministic
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(LABEL_ORDER)

        self.feature_names_: list[str] = []
        self.is_fitted_: bool = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "ModelTrainer":
        self.feature_names_ = list(X_train.columns)
        y_enc = self.label_encoder.transform(y_train)

        print(f"[trainer] Fitting {self.model_name} | "
              f"n={len(X_train)} samples | "
              f"p={X_train.shape[1]} features")

        self.estimator.fit(X_train.values, y_enc)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return string class predictions."""
        self._check_fitted()
        X_aligned = X[self.feature_names_]
        y_enc = self.estimator.predict(X_aligned.values)
        return self.label_encoder.inverse_transform(y_enc)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return class probabilities as a DataFrame with class-name columns."""
        self._check_fitted()
        X_aligned = X[self.feature_names_]
        proba  = self.estimator.predict_proba(X_aligned.values)
        # estimator.classes_ gives encoded ints — decode to strings
        classes = self.label_encoder.inverse_transform(self.estimator.classes_)
        return pd.DataFrame(proba, index=X.index, columns=classes)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"[trainer] Saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ModelTrainer":
        return joblib.load(path)

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before predict()")
