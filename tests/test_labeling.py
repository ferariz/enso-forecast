"""Unit tests for ENSO phase labeling."""
import numpy as np
import pandas as pd
import pytest

from src.labeling.enso_phase import (
    label,
    EL_NINO_THRESH,
    LA_NINA_THRESH,
    SMOOTH_WINDOW,
    DEFAULT_HORIZONS,
)


def _make_df(values: list[float], start: str = "1990-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(values), freq="MS")
    return pd.DataFrame({"nino34_anom": values}, index=idx)


class TestLabel:
    def test_el_nino_detection(self):
        vals = [0.6, 0.7, 0.8] + [0.0] * 10
        df = label(_make_df(vals))
        assert "El Niño" in df["enso_phase"].values

    def test_la_nina_detection(self):
        vals = [-0.6, -0.7, -0.8] + [0.0] * 10
        df = label(_make_df(vals))
        assert "La Niña" in df["enso_phase"].values

    def test_neutral_detection(self):
        vals = [0.1, -0.1, 0.2, -0.2] * 5
        df = label(_make_df(vals))
        assert "Neutral" in df["enso_phase"].values

    def test_classification_target_columns_created(self):
        df = label(_make_df([0.0] * 20))
        for col in ["enso_phase", "enso_t1", "enso_t3", "enso_t6"]:
            assert col in df.columns, f"Missing target column: {col}"

    def test_regression_target_columns_created(self):
        df = label(_make_df([0.0] * 20))
        for col in ["nino34_t1", "nino34_t3", "nino34_t6"]:
            assert col in df.columns, f"Missing regression target: {col}"

    def test_target_t6_has_trailing_nans(self):
        df = label(_make_df([0.5] * 20))
        assert df["enso_t6"].isna().sum() >= 6

    def test_regression_target_t6_has_trailing_nans(self):
        df = label(_make_df([0.5] * 20))
        assert df["nino34_t6"].isna().sum() >= 6

    def test_no_leakage_in_classification_targets(self):
        """enso_t1 at row i must equal enso_phase at row i+1."""
        vals = ([0.8] * 6) + ([-0.8] * 6) + ([0.0] * 12)
        df = label(_make_df(vals))
        for i in range(len(df) - 1):
            if pd.notna(df["enso_t1"].iloc[i]) and pd.notna(df["enso_phase"].iloc[i + 1]):
                assert df["enso_t1"].iloc[i] == df["enso_phase"].iloc[i + 1]

    def test_regression_and_classification_consistent(self):
        """nino34_t3 > threshold iff enso_t3 == 'El Niño'."""
        vals = [float(i) * 0.1 for i in range(-10, 30)]
        df = label(_make_df(vals))
        valid = df[["nino34_t3", "enso_t3"]].dropna()
        el_nino_mask = valid["nino34_t3"] > EL_NINO_THRESH
        assert (valid.loc[el_nino_mask, "enso_t3"] == "El Niño").all()
        la_nina_mask = valid["nino34_t3"] < LA_NINA_THRESH
        assert (valid.loc[la_nina_mask, "enso_t3"] == "La Niña").all()

    def test_custom_horizons(self):
        df = label(_make_df([0.0] * 20), horizons=[2, 4])
        assert "enso_t2" in df.columns
        assert "enso_t4" in df.columns
        assert "enso_t1" not in df.columns

    def test_smooth_window_constant(self):
        assert SMOOTH_WINDOW == 3

    def test_thresholds(self):
        assert EL_NINO_THRESH ==  0.5
        assert LA_NINA_THRESH == -0.5
