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

    feature_cols = get_feature_columns(df)
    print(f"[features] Built {len(feature_cols)} feature columns "
          f"from {len(base_vars)} base variables")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature column names — everything except targets and index."""
    exclude = TARGET_COLS | {"date"}
    return [c for c in df.columns if c not in exclude]
