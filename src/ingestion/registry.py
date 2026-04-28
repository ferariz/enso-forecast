"""Registry: load all enabled sources and merge into one DataFrame.

Usage
-----
    from src.ingestion.registry import build_raw_dataset
    df = build_raw_dataset(config=cfg["data_sources"], raw_dir=Path("data/raw"))
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.ingestion import nino_indices, soi, zonal_wind


def build_raw_dataset(
    config: dict[str, Any],
    raw_dir: Path = Path("data/raw"),
) -> pd.DataFrame:
    """Fetch / load all enabled sources and outer-join on monthly date index.

    Parameters
    ----------
    config : dict
        The ``data_sources`` section of configs/data_sources.yaml.
    raw_dir : Path
        Directory where raw files are cached.

    Returns
    -------
    pd.DataFrame
        Monthly DatetimeIndex, all raw columns, no time filtering applied.
        Rows present in any source but not others will have NaN.
    """
    frames: list[pd.DataFrame] = []

    # ── Niño indices ──────────────────────────────────────────────────────────
    nino_df = nino_indices.load(raw_dir=raw_dir, url=config["nino_indices"]["url"])
    frames.append(nino_df)

    # ── SOI ───────────────────────────────────────────────────────────────────
    soi_df = soi.load(raw_dir=raw_dir, url=config["soi"]["url"])
    frames.append(soi_df)

    # ── Zonal wind ────────────────────────────────────────────────────────────
    zwnd_df = zonal_wind.load(raw_dir=raw_dir, url=config["zonal_wind_850"]["url"])
    frames.append(zwnd_df)

    # ── MJO (optional) ────────────────────────────────────────────────────────
    if config.get("mjo", {}).get("enabled", False):
        from src.ingestion import mjo
        mjo_df = mjo.load(raw_dir=raw_dir, url=config["mjo"]["url"])
        frames.append(mjo_df)

    # ── Merge on date index ────────────────────────────────────────────────────
    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.join(frame, how="outer")

    combined = combined.sort_index()

    print(
        f"\n[registry] Raw dataset assembled: "
        f"{combined.shape[1]} columns × {len(combined)} rows  "
        f"({combined.index[0].date()} → {combined.index[-1].date()})"
    )
    print(f"[registry] Missing values per column:\n{combined.isnull().sum().to_string()}")

    return combined
