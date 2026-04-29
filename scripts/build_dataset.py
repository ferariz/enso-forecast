#!/usr/bin/env python
"""Build the ENSO dataset end-to-end.

Steps:
    1. Load configs
    2. Ingest raw data from NOAA (or use cached files in data/raw/)
    3. Preprocess (time filter, monthly grid, gap imputation)
    4. Label (ENSO phase at t, targets at t+1, t+3, t+6)
    5. Engineer features (lags, rolling stats, calendar)
    6. Run leakage checks
    7. Save to data/processed/enso_dataset.parquet

Run from repo root:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --config-dir configs --out data/processed/enso_dataset.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_all
from src.utils.io import write_parquet
from src.ingestion.registry import build_raw_dataset
from src.preprocessing.cleaner import clean
from src.labeling.enso_phase import label
from src.feature_engineering.builder import build_features, get_feature_columns
from src.validation.leakage_check import run_leakage_checks


def parse_args():
    p = argparse.ArgumentParser(description="Build ENSO dataset")
    p.add_argument(
        "--config-dir",
        default="configs",
        help="Directory containing YAML configs (default: configs)",
    )
    p.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory for cached raw files (default: data/raw)",
    )
    p.add_argument(
        "--out",
        default="data/processed/enso_dataset.parquet",
        help="Output path for processed dataset",
    )
    return p.parse_args()


def main():
    args = parse_args()

    config_dir = Path(args.config_dir)
    raw_dir    = Path(args.raw_dir)
    out_path   = Path(args.out)

    print("=" * 60)
    print("  ENSO Dataset Build Pipeline")
    print("=" * 60)

    # ── 1. Load configs ───────────────────────────────────────────
    print("\n[1/6] Loading configs...")
    cfg = load_all(config_dir)
    print(f"      Loaded: {list(cfg.keys())}")

    # ── 2. Ingest ─────────────────────────────────────────────────
    print("\n[2/6] Ingesting raw data...")
    raw_df = build_raw_dataset(
        config=cfg["data_sources"],
        raw_dir=raw_dir,
    )

    # ── 3. Preprocess ─────────────────────────────────────────────
    print("\n[3/6] Preprocessing...")
    tr = cfg["data_sources"].get("time_range", {})
    clean_df = clean(
        raw_df,
        start=tr.get("start"),
        end=tr.get("end"),
    )

    # ── 4. Label ──────────────────────────────────────────────────
    print("\n[4/6] Labeling ENSO phases...")
    labeled_df = label(clean_df)

    # ── 5. Feature engineering ────────────────────────────────────
    print("\n[5/6] Engineering features...")
    featured_df = build_features(labeled_df, config=cfg["features"])

    # ── 6. Leakage checks ─────────────────────────────────────────
    print("\n[6/6] Running leakage checks...")
    feature_cols = get_feature_columns(featured_df)
    # enso_phase is a reference column, not a model input
    feature_cols = [c for c in feature_cols if c != "enso_phase"]

    checks = run_leakage_checks(featured_df, feature_cols)

    if not all(checks.values()):
        print("\nERROR: Leakage checks failed. Dataset NOT saved.")
        sys.exit(1)

    # ── Save ──────────────────────────────────────────────────────
    print(f"\nSaving dataset...")
    write_parquet(featured_df, out_path)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Build complete")
    print("=" * 60)
    print(f"  Rows:     {len(featured_df)}")
    print(f"  Columns:  {len(featured_df.columns)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Date range: {featured_df.index[0].date()} → "
          f"{featured_df.index[-1].date()}")
    print(f"  Output:   {out_path}")
    print()

    phase_dist = featured_df["enso_phase"].value_counts()
    print("  Phase distribution:")
    for phase, count in phase_dist.items():
        pct = count / len(featured_df) * 100
        print(f"    {phase:10s}  {count:4d} months  ({pct:.1f}%)")
    print()


if __name__ == "__main__":
    main()
