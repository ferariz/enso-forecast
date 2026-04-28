"""Unit tests for ENSO phase labeling."""
import numpy as np
import pandas as pd
import pytest

from src.labeling.enso_phase import label, EL_NINO_THRESH, LA_NINA_THRESH, ROLLING_WINDOW


def _make_df(values: list[float], start: str = "1990-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(values), freq="MS")
    return pd.DataFrame({"nino34_anom": values}, index=idx)


class TestLabel:
    def test_el_nino_detection(self):
        # Three consecutive months above threshold → rolling mean > threshold
        vals = [0.6, 0.7, 0.8] + [0.0] * 10
        df = label(_make_df(vals))
        # After warm-up period, first stable label should be El Niño
        assert "El Niño" in df["enso_phase"].values

    def test_la_nina_detection(self):
        vals = [-0.6, -0.7, -0.8] + [0.0] * 10
        df = label(_make_df(vals))
        assert "La Niña" in df["enso_phase"].values

    def test_neutral_detection(self):
        vals = [0.1, -0.1, 0.2, -0.2] * 5
        df = label(_make_df(vals))
        assert "Neutral" in df["enso_phase"].values

    def test_target_columns_created(self):
        df = label(_make_df([0.0] * 20))
        for col in ["enso_phase", "enso_t1", "enso_t3", "enso_t6"]:
            assert col in df.columns, f"Missing target column: {col}"

    def test_target_t6_has_trailing_nans(self):
        """enso_t6 must have NaN in the last 6 rows (shift creates gaps)."""
        df = label(_make_df([0.5] * 20))
        assert df["enso_t6"].isna().sum() >= 6

    def test_no_leakage_in_targets(self):
        """Target at horizon L must equal phase of the row L steps forward."""
        vals = ([0.8] * 6) + ([-0.8] * 6) + ([0.0] * 12)
        df = label(_make_df(vals))
        # enso_t1 at row i should match enso_phase at row i+1
        for i in range(len(df) - 1):
            if pd.notna(df["enso_t1"].iloc[i]) and pd.notna(df["enso_phase"].iloc[i + 1]):
                assert df["enso_t1"].iloc[i] == df["enso_phase"].iloc[i + 1]

    def test_custom_horizons(self):
        df = label(_make_df([0.0] * 20), horizons=[2, 4])
        assert "enso_t2" in df.columns
        assert "enso_t4" in df.columns
        assert "enso_t1" not in df.columns
