"""Temporal split utilities for ENSO time series.

All splits respect strict time ordering.
No random splitting, no shuffling, ever.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TemporalSplit:
    """Container for a single train / val / test temporal split."""
    train: pd.DataFrame
    val:   pd.DataFrame | None
    test:  pd.DataFrame


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
