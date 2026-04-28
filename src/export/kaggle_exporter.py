"""Export pipeline for Kaggle dataset production.

Produces:
  - enso_train.parquet / .csv
  - enso_test.parquet  / .csv
  - data_dictionary.csv
  - metadata.json

Design goals:
  - Well-named, intuitive columns
  - No missing values in critical columns
  - Compact, reproducible, portable
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ── Column descriptions for the data dictionary ──────────────────────────────
COLUMN_DESCRIPTIONS: dict[str, str] = {
    "date":                     "Monthly timestamp (first day of month)",
    "year":                     "Calendar year",
    "month":                    "Calendar month (1–12)",
    "month_sin":                "Sine encoding of month (captures seasonal cycle)",
    "month_cos":                "Cosine encoding of month (captures seasonal cycle)",

    # SST anomaly columns
    "sst_anom_nino34":          "Niño 3.4 SST anomaly (°C) — primary ENSO index (5°S–5°N, 120°W–170°W)",
    "sst_anom_nino12":          "Niño 1+2 SST anomaly (°C) — eastern tropical Pacific (0°–10°S, 80°W–90°W)",
    "sst_anom_nino3":           "Niño 3 SST anomaly (°C) — central/eastern Pacific (5°S–5°N, 90°W–150°W)",
    "sst_anom_nino4":           "Niño 4 SST anomaly (°C) — western Pacific (5°S–5°N, 150°W–160°E)",
    "southern_oscillation_index": "Southern Oscillation Index — normalised pressure difference Tahiti−Darwin (dimensionless). Negative = El Niño.",
    "zonal_wind_850_anom":      "850 hPa equatorial zonal wind anomaly (m/s) — Walker circulation proxy",

    # Lagged features
    "sst_anom_nino34_lag1":     "Niño 3.4 anomaly 1 month prior",
    "sst_anom_nino34_lag3":     "Niño 3.4 anomaly 3 months prior",
    "sst_anom_nino34_lag6":     "Niño 3.4 anomaly 6 months prior",

    # Rolling statistics
    "sst_anom_nino34_rm3":      "3-month rolling mean of Niño 3.4 anomaly (≈ ONI)",
    "sst_anom_nino34_rstd3":    "3-month rolling std of Niño 3.4 anomaly",
    "sst_anom_nino34_diff1":    "Month-over-month change in Niño 3.4 anomaly",

    # Reference label
    "enso_phase":               "ENSO phase at time t (reference, not a prediction target): El Niño / Neutral / La Niña",

    # Targets
    "enso_t1":                  "TARGET: ENSO phase 1 month ahead",
    "enso_t3":                  "TARGET: ENSO phase 3 months ahead",
    "enso_t6":                  "TARGET: ENSO phase 6 months ahead",
    "enso_t1_int":              "Integer-encoded enso_t1 (La Niña=0, Neutral=1, El Niño=2)",
    "enso_t3_int":              "Integer-encoded enso_t3",
    "enso_t6_int":              "Integer-encoded enso_t6",
}


def _apply_column_renames(df: pd.DataFrame, rename_map: dict[str, str]) -> pd.DataFrame:
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


def _encode_targets(df: pd.DataFrame, targets: list[str], label_map: dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    for t in targets:
        if t in df.columns:
            df[f"{t}_int"] = df[t].map(label_map)
    return df


def _drop_rows_missing_critical(df: pd.DataFrame, critical_cols: list[str]) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=[c for c in critical_cols if c in df.columns])
    if len(df) < before:
        logger.warning(f"Dropped {before - len(df)} rows with missing critical values")
    return df


def _build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        rows.append({
            "column":      col,
            "dtype":       str(df[col].dtype),
            "n_missing":   int(df[col].isna().sum()),
            "description": COLUMN_DESCRIPTIONS.get(col, ""),
        })
    return pd.DataFrame(rows)


def _build_metadata(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_name":    "ENSO Early Phase Prediction",
        "version":         "1.0.0",
        "created_at":      datetime.now(timezone.utc).isoformat(),
        "train_range":     [str(train["date"].min()), str(train["date"].max())],
        "test_range":      [str(test["date"].min()),  str(test["date"].max())],
        "n_train":         len(train),
        "n_test":          len(test),
        "n_features":      len([c for c in train.columns
                                 if c not in {"date", "enso_phase",
                                              "enso_t1", "enso_t3", "enso_t6",
                                              "enso_t1_int", "enso_t3_int", "enso_t6_int"}]),
        "targets":         config.get("targets", ["enso_t1", "enso_t3", "enso_t6"]),
        "label_map":       config.get("label_map", {}),
        "temporal_resolution": "monthly",
        "labeling_method": "3-month rolling mean of Niño 3.4 anomaly, thresholds ±0.5°C",
        "data_sources":    ["NOAA CPC Niño indices (ERSSTv5)", "NOAA CPC SOI", "NOAA CPC 850 hPa zonal wind"],
    }


def export(
    df: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path = "data/kaggle_export",
) -> None:
    """Run the full Kaggle export pipeline.

    Parameters
    ----------
    df:
        Fully processed dataset (features + targets).
    config:
        The ``export`` section of ``configs/export.yaml``.
    output_dir:
        Destination directory (created if absent).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()

    # ── 1. Rename columns to user-friendly names ──────────────────────────────
    df = _apply_column_renames(df, config.get("column_renames", {}))

    # ── 2. Encode targets as integers ─────────────────────────────────────────
    targets = config.get("targets", ["enso_t1", "enso_t3", "enso_t6"])
    label_map = config.get("label_map", {"El Niño": 2, "Neutral": 1, "La Niña": 0})
    df = _encode_targets(df, targets, label_map)

    # ── 3. Reset index → date column ──────────────────────────────────────────
    if df.index.name == "date":
        df = df.reset_index()
    elif "date" not in df.columns:
        df.insert(0, "date", df.index)

    # ── 4. Drop rows missing any target ───────────────────────────────────────
    df = _drop_rows_missing_critical(df, targets)

    # ── 5. Split train / test ─────────────────────────────────────────────────
    test_start = pd.Timestamp(config.get("test_start", "2019-01"))
    train_df   = df[df["date"] < test_start].copy()
    test_df    = df[df["date"] >= test_start].copy()

    logger.info(
        f"Export split — train: {len(train_df)} rows | test: {len(test_df)} rows"
    )

    # ── 6. Write data files ───────────────────────────────────────────────────
    fmt = config.get("format", "parquet")
    train_name = config.get("files", {}).get("train", "enso_train")
    test_name  = config.get("files", {}).get("test",  "enso_test")

    if fmt == "parquet":
        train_df.to_parquet(output_dir / f"{Path(train_name).stem}.parquet", index=False)
        test_df.to_parquet( output_dir / f"{Path(test_name).stem}.parquet",  index=False)
    else:  # csv fallback
        train_df.to_csv(output_dir / f"{Path(train_name).stem}.csv", index=False)
        test_df.to_csv( output_dir / f"{Path(test_name).stem}.csv",  index=False)

    logger.info(f"Train dataset saved → {output_dir}")
    logger.info(f"Test  dataset saved → {output_dir}")

    # ── 7. Data dictionary ────────────────────────────────────────────────────
    dd = _build_data_dictionary(train_df)
    dd_path = output_dir / config.get("files", {}).get("data_dictionary", "data_dictionary.csv")
    dd.to_csv(dd_path, index=False)
    logger.info(f"Data dictionary saved → {dd_path}")

    # ── 8. Metadata ───────────────────────────────────────────────────────────
    meta = _build_metadata(train_df, test_df, config)
    meta_path = output_dir / config.get("files", {}).get("metadata", "metadata.json")
    with meta_path.open("w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    logger.info(f"Metadata saved → {meta_path}")
