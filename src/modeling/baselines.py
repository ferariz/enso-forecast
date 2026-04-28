"""Climatology and persistence baselines.

These are the minimum benchmarks any trained model must beat.

Climatology baseline
--------------------
Predicts the most frequent ENSO phase in the *training set* for every
sample — equivalent to a zero-skill forecaster that knows the climate.

Persistence baseline
--------------------
Predicts that the ENSO phase at time t will persist unchanged at t+L.
This is a deceptively strong baseline for short lead times.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ClimatologyBaseline:
    """Always predict the modal class from training data."""

    def __init__(self):
        self.most_frequent_class_: str | None = None

    def fit(self, y_train: pd.Series) -> "ClimatologyBaseline":
        self.most_frequent_class_ = y_train.value_counts().idxmax()
        logger.info(f"ClimatologyBaseline: most frequent class = {self.most_frequent_class_!r}")
        return self

    def predict(self, n: int) -> np.ndarray:
        if self.most_frequent_class_ is None:
            raise RuntimeError("Call fit() before predict()")
        return np.array([self.most_frequent_class_] * n)


class PersistenceBaseline:
    """Predict enso_phase_t for all horizons (no change assumption)."""

    def predict(self, enso_phase_t: pd.Series) -> np.ndarray:
        """Return current phase as the forecast — horizon-agnostic."""
        return enso_phase_t.values
