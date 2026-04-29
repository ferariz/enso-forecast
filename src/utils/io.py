"""I/O helpers for consistent read/write of processed data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)
    print(f"[io] Saved {len(df)} rows → {path}")


def read_parquet(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_parquet(path)
    print(f"[io] Loaded {len(df)} rows ← {path}")
    return df
