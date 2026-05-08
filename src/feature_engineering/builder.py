"""Feature engineering for ENSO prediction.

All features are strictly backward-looking:
  - lags use shift(+L), pulling past values forward
  - rolling windows use only past observations (min_periods enforced)
  - diffs reflect past-to-present change only

Target and metadata columns are never modified.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# These columns are passed through untouched
TARGET_COLS = {"enso_phase", "enso_t1", "enso_t3", "enso_t6"}

# MAM = boreal spring months
SPRING_MONTHS = {3, 4, 5}


def _add_lags(df: pd.DataFrame, col: str, lags: list[int]) -> pd.DataFrame:
    """Add lagged columns: {col}_lag{L} = value L months ago.

    shift(+L) shifts values DOWN by L rows, so row i gets the value
    that was at row i-L. This is strictly past information.
    """
    for L in lags:
        df[f"{col}_lag{L}"] = df[col].shift(L)
    return df


def _add_rolling_mean(df: pd.DataFrame, col: str, windows: list[int]) -> pd.DataFrame:
    """Add backward rolling mean: {col}_rm{W}.

    min_periods=2 avoids NaN on the very first row while still requiring
    at least some history.
    """
    for W in windows:
        df[f"{col}_rm{W}"] = df[col].rolling(window=W, min_periods=2).mean()
    return df


def _add_rolling_std(df: pd.DataFrame, col: str, windows: list[int]) -> pd.DataFrame:
    """Add backward rolling std: {col}_rstd{W}."""
    for W in windows:
        df[f"{col}_rstd{W}"] = df[col].rolling(window=W, min_periods=2).std()
    return df


def _add_diff(df: pd.DataFrame, col: str, periods: list[int]) -> pd.DataFrame:
    """Add first differences: {col}_diff{P} = x_t - x_{t-P}."""
    for P in periods:
        df[f"{col}_diff{P}"] = df[col].diff(P)
    return df


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode month-of-year as sin/cos to capture seasonal cycle.

    sin/cos encoding preserves the circular structure of the calendar:
    the distance between December and January is the same as between
    any other consecutive months.
    """
    month = df.index.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["month"] = month
    df["year"]  = df.index.year
    return df


def _forecast_crosses_spring(init_month: int, horizon: int) -> bool:
    """Return True if the forecast window [t+1, t+horizon] contains any MAM month.

    Parameters
    ----------
    init_month : int
        Calendar month of initialization (1=Jan, 12=Dec).
    horizon : int
        Forecast lead time in months.

    Physical rationale
    ------------------
    The boreal spring predictability barrier (SPB) causes a sharp drop in
    ENSO forecast skill for windows that pass through MAM (March–May).
    This is due to the phase-locking of ENSO to the annual cycle: SST
    anomalies in the central-eastern Pacific tend to peak in boreal winter
    (DJF) and decay through spring, making spring a "barrier" that
    disrupts the persistence of anomalies into the following season.

    For L=6, the months that cross spring are: Sep, Oct, Nov, Dec, Jan,
    Feb, Mar, Apr (8 months). May–Aug do NOT cross spring and therefore
    have both higher forecast skill and higher operational impact since
    their targets land in DJF.
    """
    forecast_months = {(init_month + l - 1) % 12 + 1 for l in range(1, horizon + 1)}
    return bool(forecast_months & SPRING_MONTHS)


def _add_spring_barrier_features(
    df: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    """Add crosses_spring_tL boolean features for each forecast horizon.

    These features encode whether a given forecast crosses the boreal
    spring predictability barrier (MAM). They are computed from the
    initialization month only — strictly backward-looking.

    A tree model (LightGBM, RF) will use these as regime indicators,
    applying different logic to spring-crossing vs non-spring-crossing
    forecasts. In a linear model, an interaction term with the main
    ENSO indices would be needed instead.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a monthly DatetimeIndex.
    horizons : list[int]
        Forecast horizons in months (e.g. [1, 3, 6]).
    """
    month = df.index.month
    for L in horizons:
        col = f"crosses_spring_t{L}"
        df[col] = [
            _forecast_crosses_spring(m, L) for m in month
        ]
        n_true  = df[col].sum()
        n_false = (~df[col]).sum()
        print(f"[features] {col}: {n_true} cross spring, {n_false} do not")
    return df


def build_features(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Apply all configured feature transformations.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned and labeled DataFrame with a monthly DatetimeIndex.
    config : dict
        The ``features`` section of configs/features.yaml.

    Returns
    -------
    pd.DataFrame
        Original columns plus all engineered features.
        Target columns (enso_phase, enso_t*) are untouched.
    """
    df = df.copy()
    tf = config.get("transformations", {})
    base_vars = config.get("base_variables", [])

    for col in base_vars:
        if col not in df.columns:
            print(f"[features] WARNING: '{col}' not in DataFrame — skipping")
            continue

        if tf.get("lags", {}).get("enabled", True):
            df = _add_lags(df, col, tf["lags"].get("months", [1, 3, 6]))

        if tf.get("rolling_mean", {}).get("enabled", True):
            df = _add_rolling_mean(df, col, tf["rolling_mean"].get("windows", [3]))

        if tf.get("rolling_std", {}).get("enabled", True):
            df = _add_rolling_std(df, col, tf["rolling_std"].get("windows", [3]))

        if tf.get("diff", {}).get("enabled", True):
            df = _add_diff(df, col, tf["diff"].get("periods", [1]))

    df = _add_calendar_features(df)

    # Spring barrier features — one boolean per forecast horizon
    # Horizons are read from the targets list in modeling.yaml, but we
    # default to [1, 3, 6] to match DEFAULT_HORIZONS in labeling.
    horizons = [1, 3, 6]
    df = _add_spring_barrier_features(df, horizons)

    feature_cols = get_feature_columns(df)
    print(f"[features] Built {len(feature_cols)} feature columns "
          f"from {len(base_vars)} base variables")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature column names — everything except targets and index."""
    exclude = TARGET_COLS | {"date"}
    return [c for c in df.columns if c not in exclude]
