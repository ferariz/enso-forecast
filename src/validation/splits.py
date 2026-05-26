"""Temporal split utilities for ENSO time series.

All splits respect strict time ordering.
No random splitting, no shuffling, ever.

Two splitting strategies:

1. ``time_split`` — single hard split into train / val / test.
   Used for final evaluation and Kaggle benchmarking.

2. ``walk_forward_cv`` — rolling walk-forward cross-validation.
   Used for hyperparameter tuning, model comparison with confidence
   intervals, and statistical significance testing (paper).

Walk-forward CV design
----------------------
For a climate time series with strong autocorrelation, standard K-fold
leaks future information. Walk-forward CV preserves temporal ordering:

    Fold 1:  train [1980–1994]  test [1995–1996]
    Fold 2:  train [1980–1996]  test [1997–1998]
    ...
    Fold N:  train [1980–2012]  test [2013–2014]

The test set (2019–present) is always held out and never used in CV.

A minimum gap between train end and test start can be specified to
prevent leakage from features with long lags (e.g. lag-6 features
require a 6-month gap between train and test windows).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import pandas as pd


@dataclass
class TemporalSplit:
    """Container for a single train / val / test temporal split."""
    train: pd.DataFrame
    val:   pd.DataFrame | None
    test:  pd.DataFrame


@dataclass
class WalkForwardFold:
    """Container for one fold of walk-forward cross-validation."""
    fold_index:  int
    train:       pd.DataFrame
    test:        pd.DataFrame
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    test_start:  pd.Timestamp
    test_end:    pd.Timestamp

    def __repr__(self) -> str:
        return (
            f"Fold {self.fold_index:2d}: "
            f"train [{self.train_start.date()} → {self.train_end.date()}] "
            f"({len(self.train):3d} months) | "
            f"test  [{self.test_start.date()} → {self.test_end.date()}] "
            f"({len(self.test):2d} months)"
        )


def time_split(
    df: pd.DataFrame,
    train_end:  str,
    test_start: str,
    val_start:  str | None = None,
    val_end:    str | None = None,
) -> TemporalSplit:
    """Hard time-based split into train, optional val, and test.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with a monthly DatetimeIndex.
    train_end : str
        Last month included in training, e.g. "2015-12".
    test_start : str
        First month of the held-out test set, e.g. "2019-01".
    val_start : str | None
        First month of validation window. If None, the gap between
        train_end and test_start is used as validation.
    val_end : str | None
        Last month of validation window.

    Returns
    -------
    TemporalSplit
        Named fields: train, val (may be None), test.
    """
    train = df.loc[: pd.Timestamp(train_end)]
    test  = df.loc[pd.Timestamp(test_start) :]

    if val_start and val_end:
        val = df.loc[pd.Timestamp(val_start) : pd.Timestamp(val_end)]
    else:
        # Use the gap between train and test as implicit validation
        gap_start = train.index[-1] + pd.DateOffset(months=1)
        gap_end   = test.index[0]  - pd.DateOffset(months=1)
        val = df.loc[gap_start:gap_end] if gap_start <= gap_end else None

    _report(train, val, test)
    return TemporalSplit(train=train, val=val, test=test)


def walk_forward_cv(
    df:               pd.DataFrame,
    cv_end:           str,
    test_window:      int = 12,
    min_train_months: int = 120,
    gap_months:       int = 0,
    step_months:      int | None = None,
    expanding:        bool = True,
) -> list[WalkForwardFold]:
    """Generate walk-forward cross-validation folds.

    The held-out test set (after ``cv_end``) is never touched.
    All folds are built from the data up to ``cv_end``.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with a monthly DatetimeIndex.
    cv_end : str
        Last month available for CV (= test_start - 1 month).
        E.g. "2018-12" if the held-out test starts at 2019-01.
    test_window : int
        Number of months in each CV test window. Default 12 (1 year).
    min_train_months : int
        Minimum number of training months required before a fold is
        created. Default 120 (10 years) — ensures enough history for
        feature lags and rolling windows to be meaningful.
    gap_months : int
        Months to skip between train end and test start. Use this to
        prevent leakage from long-lag features. Default 0 (no gap).
        For a dataset with max lag 6, set gap_months=6.
    step_months : int | None
        How many months to advance between folds. Default = test_window
        (non-overlapping test sets). Set smaller for more folds.
    expanding : bool
        If True (default), training window expands with each fold.
        If False, training window is rolling (fixed length = min_train_months).

    Returns
    -------
    list[WalkForwardFold]
        Ordered list of folds, earliest first.

    Examples
    --------
    >>> folds = walk_forward_cv(df, cv_end="2018-12",
    ...                         test_window=12, min_train_months=120)
    >>> for fold in folds:
    ...     print(fold)
    Fold  1: train [1980-01 → 1989-12] (120 months) | test [1990-01 → 1990-12] (12 months)
    Fold  2: train [1980-01 → 1990-12] (132 months) | test [1991-01 → 1991-12] (12 months)
    ...
    """
    if step_months is None:
        step_months = test_window

    cv_end_ts = pd.Timestamp(cv_end) + pd.offsets.MonthEnd(0)
    df_cv     = df.loc[: cv_end_ts].copy()

    if len(df_cv) == 0:
        raise ValueError(f"No data found up to cv_end={cv_end}")

    start_ts = df_cv.index[0]
    folds    = []
    fold_idx = 1

    # First possible test start: after min_train + gap
    first_test_start = start_ts + pd.DateOffset(months=min_train_months + gap_months)

    # Align to first day of month
    if first_test_start not in df_cv.index:
        first_test_start = df_cv.index[df_cv.index >= first_test_start][0] \
            if any(df_cv.index >= first_test_start) else None

    if first_test_start is None:
        raise ValueError(
            f"Not enough data for CV. Need at least "
            f"{min_train_months + gap_months} months before cv_end={cv_end}."
        )

    test_start = first_test_start

    while True:
        test_end = test_start + pd.DateOffset(months=test_window - 1)
        test_end = test_end + pd.offsets.MonthEnd(0)

        # Stop if test window exceeds available CV data
        if test_end > cv_end_ts:
            break

        # Train window
        train_end = test_start - pd.DateOffset(months=gap_months + 1)
        train_end = train_end + pd.offsets.MonthEnd(0)

        if expanding:
            train_start = start_ts
        else:
            train_start = train_end - pd.DateOffset(months=min_train_months - 1)
            train_start = train_start.replace(day=1)

        train_df = df_cv.loc[train_start:train_end]
        test_df  = df_cv.loc[test_start:test_end]

        if len(train_df) < min_train_months:
            test_start += pd.DateOffset(months=step_months)
            continue

        folds.append(WalkForwardFold(
            fold_index  = fold_idx,
            train       = train_df,
            test        = test_df,
            train_start = train_df.index[0],
            train_end   = train_df.index[-1],
            test_start  = test_df.index[0],
            test_end    = test_df.index[-1],
        ))
        fold_idx   += 1
        test_start += pd.DateOffset(months=step_months)

    print(f"[cv] Generated {len(folds)} walk-forward folds")
    print(f"[cv] CV range: {folds[0].test_start.date()} → {folds[-1].test_end.date()}")
    print(f"[cv] Train window: {'expanding' if expanding else 'rolling'} "
          f"| test_window={test_window}m | gap={gap_months}m")
    return folds


def _report(train, val, test):
    val_str = (
        f"val   [{val.index[0].date()} → {val.index[-1].date()}] "
        f"({len(val)} months)"
        if val is not None else "val   [none]"
    )
    print(
        f"[splits] train [{train.index[0].date()} → {train.index[-1].date()}] "
        f"({len(train)} months)\n"
        f"[splits] {val_str}\n"
        f"[splits] test  [{test.index[0].date()} → {test.index[-1].date()}] "
        f"({len(test)} months)"
    )
