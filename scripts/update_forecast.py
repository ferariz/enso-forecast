#!/usr/bin/env python
"""Automated forecast update script.

Runs the full pipeline to update the ENSO forecast with the latest
NOAA data. Intended to be run monthly, and especially before the
August 2026 post-barrier update.

Steps
-----
1. Re-download all raw NOAA files (force refresh)
2. Rebuild the processed dataset
3. Retrain all models
4. Re-export the Kaggle dataset bundle
5. Print a summary of what changed

Usage
-----
    python scripts/update_forecast.py
    python scripts/update_forecast.py --skip-retrain
    python scripts/update_forecast.py --skip-export
    python scripts/update_forecast.py --dry-run   # show what would run

Typical schedule
----------------
    Monthly:  python scripts/update_forecast.py --skip-retrain
    Seasonal: python scripts/update_forecast.py  (full retrain)
    August:   python scripts/update_forecast.py  (full retrain + manual notebook update)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Update ENSO forecast with latest NOAA data")
    p.add_argument("--skip-retrain", action="store_true",
                   help="Skip model retraining (faster, use if only data changed)")
    p.add_argument("--skip-export",  action="store_true",
                   help="Skip Kaggle export step")
    p.add_argument("--dry-run",      action="store_true",
                   help="Print steps without executing")
    p.add_argument("--dataset",      default="data/processed/enso_dataset.parquet")
    p.add_argument("--config-dir",   default="configs")
    return p.parse_args()


def run(cmd: str, dry_run: bool = False) -> int:
    """Run a shell command, print output in real time."""
    print(f"\n$ {cmd}")
    if dry_run:
        print("  [dry-run — skipped]")
        return 0
    result = subprocess.run(cmd, shell=True)
    return result.returncode


def get_last_real_date(dataset_path: str) -> str | None:
    """Return the last date with real (non-forward-filled) nino34 data."""
    try:
        import pandas as pd
        df = pd.read_parquet(dataset_path)
        if "date" in df.columns:
            df = df.set_index("date")
        key_cols = ["nino34_anom", "soi", "zwnd850_anom"]
        available = [c for c in key_cols if c in df.columns]
        if not available:
            return None
        last = df[available].dropna().index[-1]
        return str(last.date())
    except Exception:
        return None


def main():
    args    = parse_args()
    dry_run = args.dry_run

    print("=" * 60)
    print("  ENSO Forecast Update Pipeline")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    if dry_run:
        print("\n  ⚠️  DRY RUN — no commands will be executed\n")

    # ── Step 0: show current state ────────────────────────────────────────────
    dataset_path = args.dataset
    before_date  = get_last_real_date(dataset_path)
    print(f"\n  Current last data point: {before_date or 'unknown'}")

    # ── Step 1: force re-download all raw files ───────────────────────────────
    print("\n[1/4] Deleting cached raw files to force re-download...")
    raw_files = [
        "data/raw/nino_indices_raw.txt",
        "data/raw/soi_raw.txt",
        "data/raw/zwnd850_raw.txt",
        "data/raw/wwv_raw.txt",
    ]
    for f in raw_files:
        if dry_run:
            print(f"  [dry-run] would delete {f}")
        else:
            path = Path(f)
            if path.exists():
                path.unlink()
                print(f"  Deleted {f}")
            else:
                print(f"  Not found (skipping): {f}")

    # ── Step 2: rebuild dataset ───────────────────────────────────────────────
    print("\n[2/4] Rebuilding dataset with fresh NOAA data...")
    rc = run(f"python scripts/build_dataset.py --config-dir {args.config_dir}",
             dry_run=dry_run)
    if rc != 0:
        print(f"\n❌ build_dataset.py failed (exit code {rc})")
        sys.exit(rc)

    after_date = get_last_real_date(dataset_path) if not dry_run else "unknown"
    if before_date and after_date and before_date != after_date:
        print(f"\n  ✓ Data updated: {before_date} → {after_date}")
    elif not dry_run:
        print(f"\n  — No new data (still at {after_date})")

    # ── Step 3: retrain models ────────────────────────────────────────────────
    if args.skip_retrain:
        print("\n[3/4] Skipping model retraining (--skip-retrain)")
    else:
        print("\n[3/4] Retraining all models...")
        rc = run("python scripts/train_models.py", dry_run=dry_run)
        if rc != 0:
            print(f"\n❌ train_models.py failed (exit code {rc})")
            sys.exit(rc)

    # ── Step 4: export Kaggle dataset ─────────────────────────────────────────
    if args.skip_export:
        print("\n[4/4] Skipping Kaggle export (--skip-export)")
    else:
        print("\n[4/4] Exporting Kaggle dataset bundle...")
        rc = run("python scripts/export_kaggle_dataset.py", dry_run=dry_run)
        if rc != 0:
            print(f"\n❌ export_kaggle_dataset.py failed (exit code {rc})")
            sys.exit(rc)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Update complete")
    print("=" * 60)
    if not dry_run and not args.skip_export:
        print("""
  Next steps:
  1. Review outputs/metrics/results.json — did benchmarks change?
  2. Run the 2026 forecast notebook to update predictions:
       jupyter lab notebooks/research/enso_2026_forecast.ipynb
  3. If benchmarks improved, update Kaggle:
       kaggle datasets version -p data/kaggle_export/ -m "Monthly update YYYY-MM"
  4. Commit:
       git add outputs/metrics/ data/
       git commit -m "chore: monthly data update YYYY-MM"
""")


if __name__ == "__main__":
    main()
