"""ENSO phase labeling following the ONI convention.

The Oceanic Niño Index (ONI) uses a 3-month running mean of the Niño 3.4
SST anomaly, with thresholds ±0.5 °C sustained for ≥5 consecutive months.

For predictive ML purposes we use the same rolling-mean smoothing but apply
the threshold without the "sustained" requirement — appropriate for a monthly
classification target.

Labels
------
    "El Niño"  : rolling_mean(nino34_anom) > +0.5
    "La Niña"  : rolling_mean(nino34_anom) < -0.5
    "Neutral"  : otherwise

Targets generated
-----------------
    enso_phase  : label at current time t (reference)
    enso_t1     : label at t + 1 month
    enso_t3     : label at t + 3 months
    enso_t6     : label at t + 6 months

CRITICAL: shift() is applied to the smoothed series BEFORE assigning to
features, so the target at horizon L is the label that will be KNOWN at
t + L — not information from the future leaking into t.
"""
from __future__ import annotations

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

PHASE_MAP = {
    "El Niño": 2,
    "Neutral":  1,
    "La Niña":  0,
}

EL_NINO_THRESH =  0.5   # °C
LA_NINA_THRESH = -0.5   # °C
ROLLING_WINDOW = 3      # months


def _phase_from_value(series: pd.Series) -> pd.Series:
    """Vectorised mapping from anomaly value to phase string."""
    phase = pd.Series("Neutral", index=series.index, dtype=object)
    phase[series >  EL_NINO_THRESH] = "El Niño"
    phase[series <  LA_NINA_THRESH] = "La Niña"
    return phase


def label(
    df: pd.DataFrame,
    source_col: str = "nino34_anom",
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Add ENSO phase label columns to *df*.

    Parameters
    ----------
    df:
        DataFrame with a monthly DatetimeIndex.
    source_col:
        Column used for thresholding. Default ``nino34_anom``.
    horizons:
        Lead-time months to generate targets for. Default [1, 3, 6].

    Returns
    -------
    pd.DataFrame
        Original df plus ``enso_phase`` and ``enso_tL`` target columns.
    """
    if horizons is None:
        horizons = [1, 3, 6]

    df = df.copy()

    # 3-month centred rolling mean of the anomaly
    smoothed = df[source_col].rolling(window=ROLLING_WINDOW, center=False, min_periods=2).mean()

    # Current phase (reference label, NOT a target to predict from t)
    df["enso_phase"] = _phase_from_value(smoothed)

    # Future targets — shift smoothed series backward by L months
    # shift(-L) pulls the future value into the current row.
    # This creates the TARGET: what phase will be observed at t+L.
    for L in horizons:
        col = f"enso_t{L}"
        future_smoothed = smoothed.shift(-L)
        df[col] = _phase_from_value(future_smoothed)
        n_nan = df[col].isna().sum()
        if n_nan:
            logger.debug(f"Target {col}: {n_nan} NaN rows at end (expected due to shift)")

    n_phases = df["enso_phase"].value_counts().to_dict()
    logger.info(f"ENSO phase distribution: {n_phases}")

    return df
