#!/usr/bin/env python
"""Standalone script: build the ENSO dataset end-to-end.

Equivalent to running: enso build-dataset

Run from repo root:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --config-dir configs --out-path data/processed/enso_dataset.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer

if __name__ == "__main__":
    from src.cli import build_dataset
    typer.run(build_dataset)
