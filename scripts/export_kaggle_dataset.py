#!/usr/bin/env python
"""Export the Kaggle-ready dataset bundle.

Reads data/processed/enso_dataset.parquet and produces:
    data/kaggle_export/
        enso_train.parquet
        enso_test.parquet
        data_dictionary.csv
        metadata.json
        dataset-metadata.json   ← used by `kaggle datasets create`

Run from repo root:
    python scripts/export_kaggle_dataset.py
    python scripts/export_kaggle_dataset.py --out data/kaggle_export

To publish on Kaggle (first time):
    pip install kaggle
    # Put ~/.kaggle/kaggle.json API key in place
    kaggle datasets create -p data/kaggle_export/

To update an existing dataset:
    kaggle datasets version -p data/kaggle_export/ -m "describe your update"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_all
from src.utils.io import read_parquet
from src.export.kaggle_exporter import export


def parse_args():
    p = argparse.ArgumentParser(description="Export Kaggle dataset bundle")
    p.add_argument(
        "--dataset",
        default="data/processed/enso_dataset.parquet",
        help="Processed dataset to export from",
    )
    p.add_argument(
        "--out",
        default="data/kaggle_export",
        help="Output directory for export files",
    )
    p.add_argument(
        "--config-dir",
        default="configs",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_all(args.config_dir)

    test_start = cfg["export"].get("test_start", "2019-01")

    print("=" * 60)
    print("  Kaggle Export Pipeline")
    print("=" * 60)
    print(f"\n  Source:     {args.dataset}")
    print(f"  Output:     {args.out}")
    print(f"  Test start: {test_start}")
    print()

    df = read_parquet(args.dataset)

    export(df, output_dir=args.out, test_start=test_start)

    print()
    print("=" * 60)
    print("  To publish on Kaggle:")
    print("  kaggle datasets create -p data/kaggle_export/")
    print()
    print("  To update an existing dataset:")
    print("  kaggle datasets version -p data/kaggle_export/ -m 'your message'")
    print("=" * 60)


if __name__ == "__main__":
    main()
