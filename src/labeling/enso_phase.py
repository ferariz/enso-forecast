"""ENSO phase labeling following the ONI convention.

The Oceanic Niño Index (ONI) uses a 3-month backward rolling mean of the
Niño 3.4 SST anomaly with thresholds ±0.5 °C.

Labels
------
    "El Niño"  : rolling_mean(nino34_anom, 3) >  +0.5
    "La Niña"  : rolling_mean(nino34_anom, 3) <  -0.5
    "Neutral"  : otherwise

Targets (per horizon L)
-----------------------
    enso_phase   : phase at current time t (reference, not a prediction target)
    enso_tL      : ENSO phase at t+L (classification target)
    nino34_tL    : smoothed Niño 3.4 anomaly (°C) at t+L (regression target)

The regression target nino34_tL is the same 3-month smoothed series used
to derive enso_tL — they are consistent by construction. This allows users
to treat ENSO prediction as either a classification or regression problem,
and to observe the "regression to the mean" bias that emerges during spring.

SHIFT LOGIC (critical):
    future_smoothed = smoothed.shift(-L)
    enso_tL  = _phase_from_value(future_smoothed)   # classification
    nino34_tL = future_smoothed                      # regression (same series)

    shift(-L) pulls the value L steps ahead into the current row.
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
    """Add ENSO phase and Niño 3.4 anomaly target columns to df.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with a monthly DatetimeIndex.
    source_col : str
        Column to smooth and threshold. Default "nino34_anom".
    horizons : list[int]
        Lead times in months to generate targets for.

    Returns
    -------
    pd.DataFrame
        Original df plus:
          - enso_phase         : phase at t (reference)
          - enso_tL            : phase at t+L (classification target)
          - nino34_tL          : smoothed Niño 3.4 anomaly at t+L (regression target)
    """
    if source_col not in df.columns:
        raise ValueError(
            f"Source column '{source_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()

    # Smooth the anomaly series (backward-looking — no leakage)
    smoothed = _smooth(df[source_col])

    # Reference: current smoothed anomaly and phase at t
    df["nino34_smoothed"] = smoothed                    # kept for reference/debugging
    df["enso_phase"]      = _phase_from_value(smoothed)

    # Future targets — one classification + one regression per horizon
    for L in horizons:
        future_smoothed = smoothed.shift(-L)

        # Classification target: phase string
        df[f"enso_t{L}"]   = _phase_from_value(future_smoothed)

        # Regression target: smoothed anomaly value (°C)
        # Same series as used for classification — consistent by construction.
        # Round to 2 decimal places to match NOAA publication precision.
        df[f"nino34_t{L}"] = future_smoothed.round(2)

    # ── Reporting ─────────────────────────────────────────────────────────────
    dist = df["enso_phase"].value_counts().to_dict()
    print(f"[labeling] Phase distribution at t: {dist}")

    for L in horizons:
        n_nan      = df[f"enso_t{L}"].isna().sum()
        shift_nan  = L
        source_nan = max(0, n_nan - shift_nan)
        msg  = f"[labeling] enso_t{L} / nino34_t{L}: {n_nan} NaN rows"
        msg += f" ({shift_nan} from horizon shift"
        if source_nan > 0:
            msg += f", {source_nan} from missing source data"
        msg += ")"
        print(msg)

    return df
