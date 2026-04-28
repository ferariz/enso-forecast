"""Unit tests for leakage detection checks."""
import pandas as pd
import numpy as np
import pytest

from src.validation.leakage_check import (
    check_no_future_lags,
    check_index_monotonic,
    check_no_target_in_features,
    run_all_checks,
)


def _clean_df():
    idx = pd.date_range("1990-01", periods=50, freq="MS")
    df = pd.DataFrame({
        "nino34_anom":      np.random.randn(50),
        "nino34_anom_lag1": np.random.randn(50),
        "nino34_anom_lag3": np.random.randn(50),
        "enso_phase":       ["Neutral"] * 50,
        "enso_t1":          ["Neutral"] * 50,
        "enso_t3":          ["Neutral"] * 50,
        "enso_t6":          ["Neutral"] * 50,
    }, index=idx)
    return df


class TestLeakageChecks:
    def test_no_future_lags_passes_on_clean_df(self):
        df = _clean_df()
        assert check_no_future_lags(df) is True

    def test_no_future_lags_fails_on_lag0(self):
        df = _clean_df()
        df["nino34_anom_lag0"] = 0.0
        assert check_no_future_lags(df) is False

    def test_no_future_lags_fails_on_negative_lag(self):
        df = _clean_df()
        df["nino34_anom_lag-1"] = 0.0
        assert check_no_future_lags(df) is False

    def test_monotonic_index_passes(self):
        df = _clean_df()
        assert check_index_monotonic(df) is True

    def test_non_monotonic_index_fails(self):
        df = _clean_df()
        df = pd.concat([df.iloc[10:], df.iloc[:10]])  # shuffle dates
        assert check_index_monotonic(df) is False

    def test_no_target_in_features_passes(self):
        feature_cols = ["nino34_anom", "nino34_anom_lag1"]
        assert check_no_target_in_features(_clean_df(), feature_cols) is True

    def test_target_in_features_fails(self):
        feature_cols = ["nino34_anom", "enso_t1"]  # enso_t1 is a target!
        assert check_no_target_in_features(_clean_df(), feature_cols) is False
