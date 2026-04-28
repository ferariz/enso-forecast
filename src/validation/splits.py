"""Temporal split utilities for ENSO time series.

⚠️  All splits respect time order — no shuffling, no random splits.
     Future data NEVER enters the training fold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TemporalSplit:
    """Container for a single train/val/test temporal split."""
    train: pd.DataFrame
    val:   pd.DataFrame | None
    test:  pd.DataFrame


def time_split(
    df: pd.DataFrame,
    train_end:   str,
    test_start:  str,
    val_start:   str | None = None,
    val_end:     str | None = None,
) -> TemporalSplit:
    """Hard time-based split.

    Parameters
    ----------
    df:
        Full dataset with a DatetimeIndex (monthly).
    train_end:
        Last month (inclusive) of the training set, e.g. ``"2015-12"``.
    test_start:
        First month (inclusive) of the test set, e.g. ``"2019-01"``.
    val_start / val_end:
        Optional explicit validation window. If None, the gap between
        *train_end* and *test_start* is used as validation.
    """
    train = df.loc[:pd.Timestamp(train_end)]

    # Validation
    if val_start and val_end:
        val = df.loc[pd.Timestamp(val_start):pd.Timestamp(val_end)]
    elif pd.Timestamp(train_end) < pd.Timestamp(test_start):
        val_start_ts = df.loc[pd.Timestamp(train_end):].index[1]  # month after train_end
        val_end_ts = df.loc[:pd.Timestamp(test_start)].index[-2]  # month before test_start
        val = df.loc[val_start_ts:val_end_ts] if val_start_ts <= val_end_ts else None
    else:
        val = None

    test = df.loc[pd.Timestamp(test_start):]

    _log_split(train, val, test)
    return TemporalSplit(train=train, val=val, test=test)


def walk_forward_splits(
    df: pd.DataFrame,
    initial_train_years: int = 15,
    step_months: int = 12,
    horizon_months: int = 12,
) -> Generator[TemporalSplit, None, None]:
    """Yield expanding-window walk-forward splits.

    Each iteration:
      - Training window expands by *step_months*
      - Validation window is the next *horizon_months*

    Parameters
    ----------
    initial_train_years:
        Minimum training history before the first fold.
    step_months:
        How many months to advance the training cutoff each fold.
    horizon_months:
        Length of the validation (held-out) window per fold.
    """
    start = df.index[0]
    end   = df.index[-1]

    train_end = start + pd.DateOffset(years=initial_train_years) - pd.DateOffset(months=1)
    fold = 0

    while True:
        val_start = train_end + pd.DateOffset(months=1)
        val_end   = val_start + pd.DateOffset(months=horizon_months - 1)

        if val_end > end:
            break

        train = df.loc[:train_end]
        val   = df.loc[val_start:val_end]

        logger.debug(
            f"Fold {fold}: train [{train.index[0].date()} → {train.index[-1].date()}] "
            f"| val [{val.index[0].date()} → {val.index[-1].date()}]"
        )
        yield TemporalSplit(train=train, val=val, test=val)

        train_end = train_end + pd.DateOffset(months=step_months)
        fold += 1


def _log_split(train, val, test):
    val_info = (
        f"val [{val.index[0].date()} → {val.index[-1].date()}] ({len(val)} months)"
        if val is not None else "no val set"
    )
    logger.info(
        f"Temporal split — "
        f"train [{train.index[0].date()} → {train.index[-1].date()}] ({len(train)} months) | "
        f"{val_info} | "
        f"test [{test.index[0].date()} → {test.index[-1].date()}] ({len(test)} months)"
    )
