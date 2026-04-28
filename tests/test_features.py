"""Unit tests for feature engineering — focus on backward-looking guarantee."""
import pandas as pd
import numpy as np
import pytest

from src.feature_engineering.builder import build_features, get_feature_columns


def _base_config():
    return {
        "base_variables": ["nino34_anom", "soi"],
        "transformations": {
            "lags":         {"enabled": True,  "months": [1, 3]},
            "rolling_mean": {"enabled": True,  "windows": [3]},
            "rolling_std":  {"enabled": True,  "windows": [3]},
            "diff":         {"enabled": True,  "periods": [1]},
        },
        "mjo": {"enabled": False},
    }


def _make_df(n: int = 30) -> pd.DataFrame:
    idx = pd.date_range("1990-01", periods=n, freq="MS")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "nino34_anom": rng.normal(0, 0.5, n),
        "soi":         rng.normal(0, 1.0, n),
        "enso_phase":  ["Neutral"] * n,
        "enso_t1":     ["Neutral"] * n,
        "enso_t3":     ["Neutral"] * n,
        "enso_t6":     ["Neutral"] * n,
    }, index=idx)


class TestBuildFeatures:
    def test_lag_columns_created(self):
        df = build_features(_make_df(), _base_config())
        assert "nino34_anom_lag1" in df.columns
        assert "nino34_anom_lag3" in df.columns

    def test_no_negative_lag_columns(self):
        """No future-leaking lag columns should exist."""
        df = build_features(_make_df(), _base_config())
        for col in df.columns:
            if "_lag" in col:
                lag_val = int(col.split("_lag")[-1])
                assert lag_val > 0, f"Non-positive lag found: {col}"

    def test_rolling_mean_columns_created(self):
        df = build_features(_make_df(), _base_config())
        assert "nino34_anom_rm3" in df.columns

    def test_diff_columns_created(self):
        df = build_features(_make_df(), _base_config())
        assert "nino34_anom_diff1" in df.columns

    def test_calendar_features_created(self):
        df = build_features(_make_df(), _base_config())
        assert "month_sin" in df.columns
        assert "month_cos" in df.columns

    def test_lag1_equals_shifted_source(self):
        """nino34_anom_lag1 at row i must equal nino34_anom at row i-1."""
        df_raw = _make_df()
        df = build_features(df_raw, _base_config())
        for i in range(1, len(df)):
            assert np.isclose(
                df["nino34_anom_lag1"].iloc[i],
                df_raw["nino34_anom"].iloc[i - 1],
                equal_nan=True,
            )

    def test_get_feature_columns_excludes_targets(self):
        df = build_features(_make_df(), _base_config())
        feat_cols = get_feature_columns(df)
        for t in ["enso_phase", "enso_t1", "enso_t3", "enso_t6"]:
            assert t not in feat_cols

    def test_missing_base_var_skipped_gracefully(self):
        """Should warn but not crash when a base variable is absent."""
        cfg = _base_config()
        cfg["base_variables"] = ["nino34_anom", "nonexistent_col"]
        df = build_features(_make_df(), cfg)
        assert "nino34_anom_lag1" in df.columns  # real col still processed
