"""Preprocessing: time filtering, monthly grid enforcement, gap imputation.

This module sits between raw ingestion and feature engineering.
It makes no physical assumptions — it only ensures the DataFrame is:
  - within the configured time range
  - on a regular monthly grid (no missing months)
  - reasonably free of short gaps (forward-filled up to max_gap months)

Longer gaps are left as NaN so they remain visible downstream.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def filter_time_range(
    df: pd.DataFrame,
    start: str | None = "1980-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Clip DataFrame to [start, end] on its DatetimeIndex.

    Parameters
    ----------
    start : str | None
        First month to keep, e.g. "1980-01". None = no lower bound.
    end : str | None
        Last month to keep, e.g. "2023-12". None = keep all available.
    """
    if start:
        df = df.loc[df.index >= pd.Timestamp(start)]
    if end:
        df = df.loc[df.index <= pd.Timestamp(end)]

    print(f"[cleaner] After time filter: {len(df)} rows  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


def enforce_monthly_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the index is a complete, gapless monthly grid (MS frequency).

    Any months present in the expected range but missing from the data
    are inserted as NaN rows. This makes gaps explicit rather than hidden.
    """
    # Align existing index to month-start (MS)
    df = df.copy()
    df.index = df.index.to_period("M").to_timestamp()  # defaults to month-start

    # Build the expected complete grid
    full_index = pd.date_range(df.index[0], df.index[-1], freq="MS")

    # Reindex — missing months become NaN rows
    df = df.reindex(full_index)
    df.index.name = "date"

    n_inserted = df.isnull().all(axis=1).sum()
    if n_inserted > 0:
        print(f"[cleaner] Inserted {n_inserted} missing months as NaN rows")
    else:
        print("[cleaner] Monthly grid is already complete — no gaps found")

    return df


def impute_gaps(
    df: pd.DataFrame,
    max_gap: int = 3,
) -> pd.DataFrame:
    """Forward-fill gaps of at most max_gap consecutive months.

    Gaps longer than max_gap are left as NaN — they will be visible
    in the missing-value report from the registry and handled explicitly.

    Parameters
    ----------
    max_gap : int
        Maximum number of consecutive NaN months to fill.
        Default 3 — fills instrument dropouts, not structural absences.
    """
    df = df.copy()
    before = df.isnull().sum().sum()
    df = df.ffill(limit=max_gap)
    after = df.isnull().sum().sum()

    filled = before - after
    if filled > 0:
        print(f"[cleaner] Forward-filled {filled} values (max_gap={max_gap})")
    remaining = df.isnull().sum()
    remaining = remaining[remaining > 0]
    if not remaining.empty:
        print(f"[cleaner] Remaining NaNs after imputation:\n{remaining.to_string()}")

    return df


def clean(
    df: pd.DataFrame,
    start: str | None = "1980-01",
    end: str | None = None,
    max_gap: int = 3,
) -> pd.DataFrame:
    """Full preprocessing pipeline: filter → regularise → impute.

    Parameters
    ----------
    df : pd.DataFrame
        Raw merged DataFrame from the registry.
    start : str | None
        Start of time range (from config).
    end : str | None
        End of time range. None = all available data.
    max_gap : int
        Max consecutive months to forward-fill.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame on a regular monthly MS grid.
    """
    df = filter_time_range(df, start=start, end=end)
    df = coerce_numeric(df)          # ensure all columns are float before grid ops
    df = enforce_monthly_grid(df)
    df = impute_gaps(df, max_gap=max_gap)
    return df


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Force all columns to numeric dtype, converting unparseable values to NaN.

    This catches cases where the raw file had missing-value sentinels
    jammed against real values without whitespace separation, causing
    read_csv to store the entire token as a string in an object column.

    E.g. "2.34-999.9-999.9..." → NaN (correctly treated as missing).
    """
    df = df.copy()
    object_cols = df.select_dtypes(include="object").columns.tolist()
    if object_cols:
        print(f"[cleaner] Coercing {len(object_cols)} object columns to numeric: "
              f"{object_cols}")
        for col in object_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
