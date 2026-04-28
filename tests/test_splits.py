"""Unit tests for temporal split utilities."""
import pandas as pd
import numpy as np
import pytest

from src.validation.splits import time_split, walk_forward_splits


def _make_monthly_df(start="1990-01", periods=360) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="MS")
    return pd.DataFrame({"x": np.arange(periods)}, index=idx)


class TestTimeSplit:
    def test_train_ends_at_correct_date(self):
        df = _make_monthly_df()
        split = time_split(df, train_end="2010-12", test_start="2012-01")
        assert split.train.index[-1] <= pd.Timestamp("2010-12")

    def test_test_starts_at_correct_date(self):
        df = _make_monthly_df()
        split = time_split(df, train_end="2010-12", test_start="2012-01")
        assert split.test.index[0] >= pd.Timestamp("2012-01")

    def test_no_overlap_train_test(self):
        df = _make_monthly_df()
        split = time_split(df, train_end="2010-12", test_start="2012-01")
        overlap = set(split.train.index) & set(split.test.index)
        assert len(overlap) == 0

    def test_validation_set_between_train_and_test(self):
        df = _make_monthly_df()
        split = time_split(
            df,
            train_end="2010-12",
            test_start="2015-01",
            val_start="2011-01",
            val_end="2014-12",
        )
        assert split.val is not None
        assert split.val.index[0]  >= pd.Timestamp("2011-01")
        assert split.val.index[-1] <= pd.Timestamp("2014-12")

    def test_future_does_not_enter_train(self):
        df = _make_monthly_df()
        split = time_split(df, train_end="2010-12", test_start="2012-01")
        # The x values in train should all be less than those in test
        assert split.train["x"].max() < split.test["x"].min()


class TestWalkForward:
    def test_yields_multiple_folds(self):
        df = _make_monthly_df(periods=360)  # 30 years
        folds = list(walk_forward_splits(df, initial_train_years=15, step_months=12, horizon_months=12))
        assert len(folds) >= 2

    def test_train_expands_monotonically(self):
        df = _make_monthly_df(periods=360)
        folds = list(walk_forward_splits(df, initial_train_years=15, step_months=12, horizon_months=12))
        train_lengths = [len(f.train) for f in folds]
        assert train_lengths == sorted(train_lengths)

    def test_val_does_not_overlap_train(self):
        df = _make_monthly_df(periods=360)
        for fold in walk_forward_splits(df, initial_train_years=15, step_months=12, horizon_months=12):
            overlap = set(fold.train.index) & set(fold.val.index)
            assert len(overlap) == 0
