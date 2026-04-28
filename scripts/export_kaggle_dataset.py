#!/usr/bin/env python
"""Standalone script: export the Kaggle-ready dataset bundle."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer

if __name__ == "__main__":
    from src.cli import export_kaggle
    typer.run(export_kaggle)
