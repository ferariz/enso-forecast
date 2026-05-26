"""Unit tests for leakage detection checks."""
import pandas as pd
import numpy as np
import pytest

from src.validation.leakage_check import (
    check_no_future_lags,
    check_index_monotonic,
    check_no_targets_in_features,
    check_target_shift,
    run_leakage_checks,
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
        assert check_no_future_lags(df)[0] is True

    def test_no_future_lags_fails_on_lag0(self):
        df = _clean_df()
        df["nino34_anom_lag0"] = 0.0
        assert check_no_future_lags(df)[0] is False

    def test_no_future_lags_fails_on_negative_lag(self):
        df = _clean_df()
        df["nino34_anom_lag-1"] = 0.0
        assert check_no_future_lags(df)[0] is False

    def test_monotonic_index_passes(self):
        df = _clean_df()
        assert check_index_monotonic(df)[0] is True

    def test_non_monotonic_index_fails(self):
        df = _clean_df()
        df = pd.concat([df.iloc[10:], df.iloc[:10]])
        assert check_index_monotonic(df)[0] is False

    def test_no_targets_in_features_passes(self):
        feature_cols = ["nino34_anom", "nino34_anom_lag1"]
        assert check_no_targets_in_features(feature_cols)[0] is True

    def test_target_in_features_fails(self):
        feature_cols = ["nino34_anom", "enso_t1"]
        assert check_no_targets_in_features(feature_cols)[0] is False

    def test_run_all_checks_passes_on_clean_df(self):
        """run_leakage_checks should not raise on a clean dataset."""
        df = _clean_df()
        # Add proper shift relationship for enso_t1
        smoothed = df["nino34_anom"].rolling(3, min_periods=2).mean()
        def phase(v):
            if pd.isna(v): return None
            if v > 0.5: return "El Niño"
            if v < -0.5: return "La Niña"
            return "Neutral"
        df["enso_phase"] = smoothed.apply(phase)
        df["enso_t1"]    = smoothed.shift(-1).apply(phase)
        try:
            run_leakage_checks(df, ["nino34_anom", "nino34_anom_lag1"])
        except Exception as e:
            pytest.fail(f"run_leakage_checks raised on clean df: {e}")
