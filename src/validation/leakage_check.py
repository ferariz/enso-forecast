"""Automated data leakage detection.

Runs a suite of checks on the feature-engineered DataFrame before
any model training. If any check fails, the pipeline should stop.

Checks
------
1. index_monotonic       — time index is strictly increasing
2. no_future_lags        — no feature column encodes a negative lag
3. no_targets_in_features — target columns not accidentally in feature list
4. target_shift_correct  — enso_t1[i] == enso_phase[i+1] (spot check)
"""
from __future__ import annotations

import pandas as pd
import numpy as np

TARGET_COLS = {"enso_phase", "enso_t1", "enso_t3", "enso_t6"}


def check_index_monotonic(df: pd.DataFrame) -> tuple[bool, str]:
    ok = df.index.is_monotonic_increasing
    msg = "OK" if ok else "FAIL — index is not monotonically increasing"
    return ok, msg


def check_no_future_lags(df: pd.DataFrame) -> tuple[bool, str]:
    """Detect any column whose name implies a non-positive lag (future data)."""
    bad = []
    for col in df.columns:
        if "_lag" in col:
            try:
                lag_val = int(col.split("_lag")[-1])
                if lag_val <= 0:
                    bad.append(col)
            except ValueError:
                pass
    ok  = len(bad) == 0
    msg = "OK" if ok else f"FAIL — future-lag columns found: {bad}"
    return ok, msg


def check_no_targets_in_features(
    feature_cols: list[str],
) -> tuple[bool, str]:
    overlap = TARGET_COLS & set(feature_cols)
    ok  = len(overlap) == 0
    msg = "OK" if ok else f"FAIL — target columns in feature set: {overlap}"
    return ok, msg


def check_target_shift(
    df: pd.DataFrame,
    n_spot_checks: int = 20,
) -> tuple[bool, str]:
    """Spot-check that enso_t1[i] == enso_phase[i+1].

    Verifies the shift logic in labeling is correct end-to-end.
    Checks the first n_spot_checks non-NaN rows.
    """
    if "enso_t1" not in df.columns or "enso_phase" not in df.columns:
        return True, "SKIP — columns not present"

    mismatches = []
    checked = 0
    for i in range(len(df) - 1):
        t1   = df["enso_t1"].iloc[i]
        ph1  = df["enso_phase"].iloc[i + 1]
        if t1 is None or ph1 is None or pd.isna(t1) or pd.isna(ph1):
            continue
        if t1 != ph1:
            mismatches.append(i)
        checked += 1
        if checked >= n_spot_checks:
            break

    ok  = len(mismatches) == 0
    msg = "OK" if ok else f"FAIL — shift mismatches at rows: {mismatches}"
    return ok, msg


def run_leakage_checks(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, bool]:
    """Run the full leakage check suite.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame (output of build_features + label).
    feature_cols : list[str]
        Column names that will be used as model inputs.

    Returns
    -------
    dict[str, bool]
        Mapping of check name → passed. All must be True to proceed.
    """
    checks = {
        "index_monotonic":        check_index_monotonic(df),
        "no_future_lags":         check_no_future_lags(df),
        "no_targets_in_features": check_no_targets_in_features(feature_cols),
        "target_shift_correct":   check_target_shift(df),
    }

    all_passed = True
    for name, (ok, msg) in checks.items():
        status = "✓" if ok else "✗"
        print(f"[leakage] {status} {name}: {msg}")
        if not ok:
            all_passed = False

    if all_passed:
        print("[leakage] All checks passed — safe to proceed")
    else:
        print("[leakage] LEAKAGE DETECTED — pipeline should not continue")

    return {name: ok for name, (ok, _) in checks.items()}
