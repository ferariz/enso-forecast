#!/usr/bin/env python
"""Standalone script: train models for all (or a specific) target horizon.

Run from repo root:
    python scripts/run_training.py --target enso_t3
    python scripts/run_training.py  # trains t1, t3, t6
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Optional
import typer

app = typer.Typer()

@app.command()
def main(
    target: Optional[str] = typer.Option(
        None,
        help="Single target to train (enso_t1 | enso_t3 | enso_t6). If omitted, trains all three.",
    ),
    config_dir: Path = typer.Option(Path("configs")),
    dataset:    Path = typer.Option(Path("data/processed/enso_dataset.parquet")),
    output_dir: Path = typer.Option(Path("outputs")),
) -> None:
    from src.cli import train_model

    targets = [target] if target else ["enso_t1", "enso_t3", "enso_t6"]
    for t in targets:
        train_model(
            target=t,
            dataset=dataset,
            config_dir=config_dir,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    app()
