"""Kaggle export pipeline.

Produces a self-contained dataset bundle in data/kaggle_export/:

    enso_train.parquet       training set (1980 – test_start)
    enso_test.parquet        test set (test_start – present)
    data_dictionary.csv      column descriptions and dtypes
    metadata.json            dataset provenance and label encoding
    dataset-metadata.json    Kaggle CLI metadata (title, license, etc.)

Design goals:
  - Well-named, intuitive columns
  - No missing values in critical columns
  - Integer-encoded targets alongside string labels
  - Completely reproducible from the processed dataset
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# ── Column rename map: internal name → export name ───────────────────────────
# Only base variable names need renaming — derived features (lags, rm, etc.)
# are renamed automatically by prefix substitution.
BASE_RENAMES = {
    "nino34_anom":  "sst_anom_nino34",
    "nino12_anom":  "sst_anom_nino12",
    "nino3_anom":   "sst_anom_nino3",
    "nino4_anom":   "sst_anom_nino4",
    "soi":          "southern_oscillation_index",
    "zwnd850_anom": "zonal_wind_850_anom",
}

# ── Column descriptions for the data dictionary ───────────────────────────────
DESCRIPTIONS: dict[str, str] = {
    "date": "Monthly timestamp (first day of month)",
    "year": "Calendar year",
    "month": "Calendar month (1–12)",
    "month_sin": "Sine encoding of month — captures seasonal cycle without discontinuity",
    "month_cos": "Cosine encoding of month",

    "sst_anom_nino34":  "Niño 3.4 SST anomaly (°C) — primary ENSO index (5°S–5°N, 120°W–170°W)",
    "sst_anom_nino12":  "Niño 1+2 SST anomaly (°C) — far eastern Pacific (0°–10°S, 80°W–90°W)",
    "sst_anom_nino3":   "Niño 3 SST anomaly (°C) — central/eastern Pacific (5°S–5°N, 90°W–150°W)",
    "sst_anom_nino4":   "Niño 4 SST anomaly (°C) — western Pacific warm pool (5°S–5°N, 150°W–160°E)",
    "southern_oscillation_index": "SOI — normalised pressure difference Tahiti − Darwin. Negative = El Niño.",
    "zonal_wind_850_anom": "850 hPa equatorial zonal wind anomaly (m/s) — Walker circulation proxy",

    "enso_phase":  "ENSO phase at time t (reference): El Niño / Neutral / La Niña",
    "enso_t1":     "TARGET: ENSO phase 1 month ahead",
    "enso_t3":     "TARGET: ENSO phase 3 months ahead",
    "enso_t6":     "TARGET: ENSO phase 6 months ahead",
    "enso_t1_int": "Integer-encoded enso_t1 (La Niña=0, Neutral=1, El Niño=2)",
    "enso_t3_int": "Integer-encoded enso_t3",
    "enso_t6_int": "Integer-encoded enso_t6",
}

LABEL_MAP = {"El Niño": 2, "Neutral": 1, "La Niña": 0}
TARGETS   = ["enso_t1", "enso_t3", "enso_t6"]


def _build_rename_map(columns: list[str]) -> dict[str, str]:
    """Build full rename map including derived features (lags, rm, etc.)."""
    rename = {}
    for col in columns:
        for old, new in BASE_RENAMES.items():
            if col == old:
                rename[col] = new
            elif col.startswith(old + "_"):
                rename[col] = col.replace(old, new, 1)
    return rename


def _drop_unrenamed_raw_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Drop raw SST columns that were not renamed (nino12, nino3, nino34, nino4).

    These appear when the raw ingestion columns were not fully cleaned up.
    The anomaly columns (renamed to sst_anom_*) are the ones we keep.
    """
    raw_raw = {"nino12", "nino3", "nino34", "nino4",
               "nino12_anom", "nino3_anom", "nino34_anom", "nino4_anom",
               "soi", "zwnd850_anom"}
    drop = [c for c in df.columns if c in raw_raw]
    if drop:
        df = df.drop(columns=drop)
    return df


def _encode_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add integer-encoded target columns alongside string labels."""
    df = df.copy()
    for t in TARGETS:
        if t in df.columns:
            df[f"{t}_int"] = df[t].map(LABEL_MAP)
    return df


def _build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        rows.append({
            "column":      col,
            "dtype":       str(df[col].dtype),
            "n_missing":   int(df[col].isna().sum()),
            "description": DESCRIPTIONS.get(col, _infer_description(col)),
        })
    return pd.DataFrame(rows)


def _infer_description(col: str) -> str:
    """Generate a description for derived feature columns."""
    for base, readable in BASE_RENAMES.items():
        new = BASE_RENAMES.get(base, base)
        if f"_lag" in col:
            lag = col.split("_lag")[-1]
            return f"{readable} lagged {lag} month(s)"
        if f"_rm" in col:
            w = col.split("_rm")[-1]
            return f"{readable} {w}-month backward rolling mean"
        if f"_rstd" in col:
            w = col.split("_rstd")[-1]
            return f"{readable} {w}-month rolling standard deviation"
        if f"_diff" in col:
            p = col.split("_diff")[-1]
            return f"{readable} {p}-month first difference (tendency)"
    return ""


def _build_metadata(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, Any]:
    feature_cols = [
        c for c in train.columns
        if c not in {"date"} | set(TARGETS) | {f"{t}_int" for t in TARGETS}
        | {"enso_phase"}
        and train[c].dtype != object
    ]
    return {
        "dataset_name":    "ENSO Early Phase Prediction",
        "version":         "1.0.0",
        "created_at":      datetime.now(timezone.utc).isoformat(),
        "train_range":     [str(train["date"].min().date()),
                            str(train["date"].max().date())],
        "test_range":      [str(test["date"].min().date()),
                            str(test["date"].max().date())],
        "n_train":         len(train),
        "n_test":          len(test),
        "n_features":      len(feature_cols),
        "targets":         TARGETS,
        "label_map":       LABEL_MAP,
        "temporal_resolution": "monthly",
        "labeling_method": "3-month rolling mean of Niño 3.4 anomaly, thresholds ±0.5°C (ONI convention)",
        "data_sources": [
            "NOAA CPC Niño indices (ERSSTv5, base 1991–2020)",
            "NOAA CPC Southern Oscillation Index",
            "NOAA CPC 850 hPa equatorial zonal wind",
        ],
        "benchmark_f1_macro": {
            "enso_t1": {"lightgbm": 0.945, "persistence": 0.858},
            "enso_t3": {"logistic_regression": 0.790, "persistence": 0.610},
            "enso_t6": {"random_forest": 0.556, "persistence": 0.419},
        },
        "test_period_note": "Test set (2019–2026) includes the strong 2020–2023 triple-dip La Niña.",
    }


def _build_kaggle_metadata(output_dir: Path) -> dict:
    """Generate dataset-metadata.json for the Kaggle CLI."""
    return {
        "title": "ENSO Early Phase Prediction",
        "id": "ferariz/enso-early-phase-prediction",
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": [
            "earth and nature", "climate and weather",
            "classification", "time series", "feature engineering"
        ],
        "collaborators": [],
        "data": [
            {"description": "Training set (1980–2018)", "name": "enso_train.parquet"},
            {"description": "Test set (2019–present)",  "name": "enso_test.parquet"},
            {"description": "Column descriptions",      "name": "data_dictionary.csv"},
            {"description": "Dataset provenance",       "name": "metadata.json"},
        ],
    }


def export(
    df: pd.DataFrame,
    output_dir: str | Path = "data/kaggle_export",
    test_start: str = "2019-01",
) -> None:
    """Run the full Kaggle export pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Fully processed dataset (features + targets). DatetimeIndex or
        with a 'date' column.
    output_dir : Path
        Destination directory (created if absent).
    test_start : str
        First month of the test set, e.g. "2019-01".
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Reset index → date column ──────────────────────────────────────────
    df = df.copy()
    if df.index.name == "date" or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"])

    # ── 2. Drop unrenamed raw SST columns ─────────────────────────────────────
    df = _drop_unrenamed_raw_cols(df)

    # ── 3. Rename columns to export names ─────────────────────────────────────
    rename_map = _build_rename_map(list(df.columns))
    df = df.rename(columns=rename_map)

    # ── 4. Encode targets as integers ─────────────────────────────────────────
    df = _encode_targets(df)

    # ── 5. Drop rows missing all targets ──────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=TARGETS, how="all")
    if len(df) < before:
        print(f"[export] Dropped {before - len(df)} rows missing all targets")

    # ── 6. Split train / test ─────────────────────────────────────────────────
    test_ts  = pd.Timestamp(test_start)
    train_df = df[df["date"] < test_ts].copy()
    test_df  = df[df["date"] >= test_ts].copy()

    print(f"[export] Train: {len(train_df)} rows "
          f"({train_df['date'].min().date()} → {train_df['date'].max().date()})")
    print(f"[export] Test:  {len(test_df)} rows "
          f"({test_df['date'].min().date()} → {test_df['date'].max().date()})")

    # ── 7. Save data files ────────────────────────────────────────────────────
    train_df.to_parquet(output_dir / "enso_train.parquet", index=False)
    test_df.to_parquet( output_dir / "enso_test.parquet",  index=False)
    print(f"[export] Parquet files saved → {output_dir}")

    # ── 8. Data dictionary ────────────────────────────────────────────────────
    dd = _build_data_dictionary(train_df)
    dd.to_csv(output_dir / "data_dictionary.csv", index=False)
    print(f"[export] Data dictionary saved ({len(dd)} columns documented)")

    # ── 9. Metadata ───────────────────────────────────────────────────────────
    meta = _build_metadata(train_df, test_df)
    with (output_dir / "metadata.json").open("w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    # ── 10. Kaggle CLI metadata ───────────────────────────────────────────────
    kaggle_meta = _build_kaggle_metadata(output_dir)
    with (output_dir / "dataset-metadata.json").open("w") as fh:
        json.dump(kaggle_meta, fh, indent=2)

    print(f"[export] metadata.json and dataset-metadata.json saved")
    print(f"[export] ✓ Export complete → {output_dir}")
