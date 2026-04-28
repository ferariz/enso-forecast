"""Data leakage detection utilities.

Runs automated checks to verify that no future information contaminates
the feature set. Called as part of the dataset build pipeline.
"""
from __future__ import annotations

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

TARGET_COLS = {"enso_phase", "enso_t1", "enso_t3", "enso_t6"}


def check_no_future_lags(df: pd.DataFrame) -> bool:
    """Verify that no feature column name implies a negative lag (future).

    Looks for columns named ``*_lagN`` where N ≤ 0.
    """
    bad = []
    for col in df.columns:
        if "_lag" in col:
            try:
                lag_val = int(col.split("_lag")[-1])
                if lag_val <= 0:
                    bad.append(col)
            except ValueError:
                pass
    if bad:
        logger.error(f"[LEAKAGE] Future-lag columns detected: {bad}")
        return False
    logger.info("[OK] No future-lag columns found")
    return True


def check_feature_target_correlation_in_time(
    df: pd.DataFrame,
    target: str = "enso_t1",
    threshold: float = 0.99,
) -> bool:
    """Flag features with suspiciously high correlation with a future target.

    A near-perfect correlation (ρ > threshold) between a feature and a
    shifted target often signals data leakage.  This is a heuristic — high
    correlation with the right physical features (e.g. nino34_anom and
    enso_t1) is expected, so the threshold is intentionally set very high.
    """
    if target not in df.columns:
        logger.warning(f"Target {target!r} not found — skipping correlation check")
        return True

    # Encode target as numeric for correlation
    enc = {"El Niño": 2, "Neutral": 1, "La Niña": 0}
    y = df[target].map(enc)

    feature_cols = [c for c in df.columns if c not in TARGET_COLS
                    and pd.api.types.is_numeric_dtype(df[c])]

    suspect = []
    for col in feature_cols:
        if df[col].nunique() < 2:
            continue
        corr = df[col].corr(y)
        if abs(corr) > threshold:
            suspect.append((col, round(corr, 4)))

    if suspect:
        logger.warning(
            f"[LEAKAGE WARNING] Features with |corr| > {threshold} vs {target}: {suspect}"
        )
        return False

    logger.info(f"[OK] No feature exceeds correlation threshold {threshold} with {target}")
    return True


def check_index_monotonic(df: pd.DataFrame) -> bool:
    """Ensure the time index is strictly monotonically increasing."""
    if not df.index.is_monotonic_increasing:
        logger.error("[LEAKAGE] DataFrame index is NOT monotonically increasing — time ordering violated")
        return False
    logger.info("[OK] Index is monotonically increasing")
    return True


def check_no_target_in_features(df: pd.DataFrame, feature_cols: list[str]) -> bool:
    """Ensure none of the target columns appear in the feature set."""
    overlap = TARGET_COLS & set(feature_cols)
    if overlap:
        logger.error(f"[LEAKAGE] Target columns in feature set: {overlap}")
        return False
    logger.info("[OK] No target columns in feature set")
    return True


def run_all_checks(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, bool]:
    """Run the full leakage check suite and return a results dict."""
    results = {
        "no_future_lags":              check_no_future_lags(df),
        "index_monotonic":             check_index_monotonic(df),
        "no_target_in_features":       check_no_target_in_features(df, feature_cols),
        "correlation_enso_t1":         check_feature_target_correlation_in_time(df, "enso_t1"),
        "correlation_enso_t3":         check_feature_target_correlation_in_time(df, "enso_t3"),
        "correlation_enso_t6":         check_feature_target_correlation_in_time(df, "enso_t6"),
    }
    passed = sum(results.values())
    total  = len(results)
    logger.info(f"Leakage checks: {passed}/{total} passed")
    if passed < total:
        failed = [k for k, v in results.items() if not v]
        logger.error(f"FAILED checks: {failed}")
    return results
