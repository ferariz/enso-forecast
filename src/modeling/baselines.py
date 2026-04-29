"""Climatology and persistence baselines.

These are the minimum benchmarks any trained model must beat.
A model that cannot outperform persistence at t+1 or climatology
at t+6 has no practical value.

Climatology
-----------
Predicts the most frequent ENSO phase in the training set for every
sample — equivalent to always saying "probably Neutral".

Persistence
-----------
Predicts that the current ENSO phase at t will persist unchanged at t+L.
Deceptively strong at short lead times because ENSO events last 6–18 months.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class ClimatologyBaseline:
    """Always predict the modal class from training data."""

    def __init__(self):
        self.most_frequent_class_: str | None = None

    def fit(self, y_train: pd.Series) -> "ClimatologyBaseline":
        self.most_frequent_class_ = y_train.value_counts().idxmax()
        print(f"[climatology] Most frequent class: '{self.most_frequent_class_}'")
        return self

    def predict(self, n: int) -> np.ndarray:
        if self.most_frequent_class_ is None:
            raise RuntimeError("Call fit() before predict()")
        return np.array([self.most_frequent_class_] * n)


class PersistenceBaseline:
    """Predict current phase as the forecast for any horizon."""

    def predict(self, enso_phase_t: pd.Series) -> np.ndarray:
        """Return current phase as forecast — no fitting needed.

        Parameters
        ----------
        enso_phase_t : pd.Series
            The enso_phase column for the rows being predicted
            (i.e. the phase at time t, not t+L).
        """
        return enso_phase_t.values
