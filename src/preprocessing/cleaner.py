"""Preprocessing: time-range filtering, missing-value handling, type enforcement.

All operations here are applied BEFORE feature engineering and labeling,
ensuring a clean, consistent base for downstream steps.
"""
from __future__ import annotations

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Columns we consider critical — rows missing these will be dropped
CRITICAL_COLS = ["nino34_anom"]


def filter_time_range(
    df: pd.DataFrame,
    start: str | None = "1980-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Clip DataFrame to [start, end] on its DatetimeIndex."""
    if start:
        df = df.loc[df.index >= pd.Timestamp(start)]
    if end:
        df = df.loc[df.index <= pd.Timestamp(end)]
    logger.info(f"After time-range filter: {len(df)} rows ({df.index[0]} → {df.index[-1]})")
    return df


def enforce_monthly_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure index is MonthStart-aligned (MS frequency)."""
    df = df.copy()
    df.index = df.index.to_period("M").to_timestamp("MS")
    # Fill any gaps with NaN rows to keep a regular monthly grid
    full_idx = pd.date_range(df.index[0], df.index[-1], freq="MS")
    df = df.reindex(full_idx)
    n_gaps = df.index.difference(df.index.dropna()).shape[0]
    if n_gaps:
        logger.warning(f"Inserted {n_gaps} missing months into the monthly grid")
    return df


def handle_missing(
    df: pd.DataFrame,
    strategy: str = "forward_fill",
    max_gap: int = 3,
) -> pd.DataFrame:
    """Impute missing values.

    Parameters
    ----------
    strategy:
        ``'forward_fill'`` — propagate last valid observation (causal, safe).
        ``'drop'``          — drop rows with *any* NaN in CRITICAL_COLS.
    max_gap:
        Maximum consecutive months to forward-fill. Longer gaps remain NaN.
    """
    df = df.copy()

    if strategy == "forward_fill":
        df = df.fillna(method="ffill", limit=max_gap)
        n_remaining = df.isnull().any(axis=1).sum()
        logger.info(f"After forward-fill (max_gap={max_gap}): {n_remaining} rows still have NaNs")
    elif strategy == "drop":
        before = len(df)
        df = df.dropna(subset=CRITICAL_COLS)
        logger.info(f"Dropped {before - len(df)} rows missing critical columns")
    else:
        raise ValueError(f"Unknown missing-value strategy: {strategy!r}")

    return df


def clean(
    df: pd.DataFrame,
    start: str | None = "1980-01",
    end: str | None = None,
    missing_strategy: str = "forward_fill",
    max_gap: int = 3,
) -> pd.DataFrame:
    """Full preprocessing pipeline: filter → regularise → impute."""
    df = filter_time_range(df, start=start, end=end)
    df = enforce_monthly_index(df)
    df = handle_missing(df, strategy=missing_strategy, max_gap=max_gap)
    return df
