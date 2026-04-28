"""Feature engineering for ENSO prediction.

All features are strictly backward-looking:
  - lags use past values only (shift(+L))
  - rolling statistics use only past observations (min_periods enforced)
  - differences reflect past trends

No future information enters any feature column.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Columns that are metadata / labels — never transform these
_NON_FEATURE_COLS = {
    "enso_phase", "enso_t1", "enso_t3", "enso_t6",
}


def _add_lags(df: pd.DataFrame, col: str, lags: list[int]) -> pd.DataFrame:
    """Add lagged columns: col_lag{L} = value L months ago."""
    for L in lags:
        df[f"{col}_lag{L}"] = df[col].shift(L)
    return df


def _add_rolling_mean(df: pd.DataFrame, col: str, windows: list[int]) -> pd.DataFrame:
    """Add backward-looking rolling mean: col_rm{W}."""
    for W in windows:
        df[f"{col}_rm{W}"] = (
            df[col].rolling(window=W, min_periods=max(1, W // 2)).mean()
        )
    return df


def _add_rolling_std(df: pd.DataFrame, col: str, windows: list[int]) -> pd.DataFrame:
    """Add backward-looking rolling std: col_rstd{W}."""
    for W in windows:
        df[f"{col}_rstd{W}"] = (
            df[col].rolling(window=W, min_periods=max(2, W // 2)).std()
        )
    return df


def _add_diff(df: pd.DataFrame, col: str, periods: list[int]) -> pd.DataFrame:
    """Add first differences: col_diff{P} = col_t − col_{t−P}."""
    for P in periods:
        df[f"{col}_diff{P}"] = df[col].diff(P)
    return df


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Month-of-year encoded as sin/cos for cyclical continuity."""
    month = df.index.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["year"] = df.index.year
    df["month"] = month
    return df


def build_features(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Apply all configured feature transformations to *df*.

    Parameters
    ----------
    df:
        Cleaned DataFrame (output of preprocessing + labeling).
    config:
        The ``features`` section of ``configs/features.yaml``.

    Returns
    -------
    pd.DataFrame
        DataFrame with original columns plus all engineered features.
        Target and metadata columns are untouched.
    """
    df = df.copy()
    tf = config.get("transformations", {})

    base_vars = config.get("base_variables", [])

    for col in base_vars:
        if col not in df.columns:
            logger.warning(f"Feature column {col!r} not found in DataFrame — skipping")
            continue

        if tf.get("lags", {}).get("enabled", True):
            lags = tf["lags"].get("months", [1, 3, 6])
            df = _add_lags(df, col, lags)

        if tf.get("rolling_mean", {}).get("enabled", True):
            windows = tf["rolling_mean"].get("windows", [3])
            df = _add_rolling_mean(df, col, windows)

        if tf.get("rolling_std", {}).get("enabled", True):
            windows = tf["rolling_std"].get("windows", [3])
            df = _add_rolling_std(df, col, windows)

        if tf.get("diff", {}).get("enabled", True):
            periods = tf["diff"].get("periods", [1])
            df = _add_diff(df, col, periods)

    # MJO features — if present and enabled, sin/cos already computed in ingestion
    mjo_cfg = config.get("mjo", {})
    if mjo_cfg.get("enabled", False):
        for mjo_col in ["mjo_sin", "mjo_cos", "mjo_amplitude", "rmm1", "rmm2"]:
            if mjo_col in df.columns:
                if tf.get("lags", {}).get("enabled", True):
                    df = _add_lags(df, mjo_col, tf["lags"].get("months", [1, 3, 6]))

    # Calendar features (always added)
    df = _add_calendar_features(df)

    n_features = len([c for c in df.columns if c not in _NON_FEATURE_COLS
                      and c not in ("date",)])
    logger.info(f"Feature engineering complete: {n_features} total feature columns")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature (predictor) column names, excluding targets and metadata."""
    exclude = _NON_FEATURE_COLS | {"date"}
    return [c for c in df.columns if c not in exclude]
