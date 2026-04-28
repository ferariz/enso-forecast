"""ENSO phase labeling following the ONI convention.

The Oceanic Niño Index (ONI) uses a 3-month backward rolling mean of the
Niño 3.4 SST anomaly with thresholds ±0.5 °C.

Labels
------
    "El Niño"  : rolling_mean(nino34_anom, 3) >  +0.5
    "La Niña"  : rolling_mean(nino34_anom, 3) <  -0.5
    "Neutral"  : otherwise

Targets
-------
    enso_phase  : phase at current time t (reference, not a prediction target)
    enso_t1     : phase at t + 1 month
    enso_t3     : phase at t + 3 months
    enso_t6     : phase at t + 6 months

SHIFT LOGIC (critical):
    future_smoothed = smoothed.shift(-L)
    enso_tL = _phase_from_value(future_smoothed)

    shift(-L) pulls the value L steps ahead into the current row.
    This means enso_t1 at row i equals enso_phase at row i+1.
    The last L rows will be NaN — expected and correct.
"""
from __future__ import annotations

import pandas as pd

# ONI thresholds (°C)
EL_NINO_THRESH =  0.5
LA_NINA_THRESH = -0.5

# Rolling window for smoothing (months)
SMOOTH_WINDOW = 3

# Default prediction horizons (months)
DEFAULT_HORIZONS = [1, 3, 6]


def _smooth(series: pd.Series) -> pd.Series:
    """Apply 3-month backward rolling mean (min 2 periods to avoid all-NaN start)."""
    return series.rolling(window=SMOOTH_WINDOW, min_periods=2).mean()


def _phase_from_value(series: pd.Series) -> pd.Series:
    """Map a continuous anomaly series to ENSO phase strings."""
    phase = pd.Series("Neutral", index=series.index, dtype=object)
    phase[series >  EL_NINO_THRESH] = "El Niño"
    phase[series <  LA_NINA_THRESH] = "La Niña"
    # Propagate NaN — if the smoothed value was NaN, phase should be NaN too
    phase = phase.where(series.notna(), other=None)
    return phase


def label(
    df: pd.DataFrame,
    source_col: str = "nino34_anom",
    horizons: list[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Add ENSO phase columns to df.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with a monthly DatetimeIndex.
    source_col : str
        Column to threshold. Default "nino34_anom".
    horizons : list[int]
        Lead times in months to generate targets for.

    Returns
    -------
    pd.DataFrame
        Original df plus enso_phase and enso_tL columns.
    """
    if source_col not in df.columns:
        raise ValueError(
            f"Source column '{source_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()

    # Smooth the anomaly series (backward-looking — no leakage)
    smoothed = _smooth(df[source_col])

    # Current phase (reference label at time t)
    df["enso_phase"] = _phase_from_value(smoothed)

    # Future targets — one per horizon
    for L in horizons:
        col = f"enso_t{L}"
        future_smoothed = smoothed.shift(-L)
        df[col] = _phase_from_value(future_smoothed)

    # Report
    dist = df["enso_phase"].value_counts().to_dict()
    print(f"[labeling] Phase distribution at t: {dist}")
    for L in horizons:
        n_nan = df[f"enso_t{L}"].isna().sum()
        print(f"[labeling] enso_t{L}: {n_nan} NaN rows at end (expected: {L})")

    return df
