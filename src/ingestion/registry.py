"""Unified ingestion registry — assembles all raw sources into one DataFrame.

Usage
-----
>>> from src.ingestion.registry import build_raw_dataset
>>> df = build_raw_dataset(config=cfg["data_sources"], raw_dir=Path("data/raw"))
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.ingestion import mjo, nino_indices, soi, zonal_wind
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_raw_dataset(
    config: dict[str, Any],
    raw_dir: Path = Path("data/raw"),
) -> pd.DataFrame:
    """Fetch / load all enabled data sources and outer-join on monthly date index.

    Parameters
    ----------
    config:
        The ``data_sources`` section of ``configs/data_sources.yaml``.
    raw_dir:
        Directory where raw files are cached.

    Returns
    -------
    pd.DataFrame
        Monthly index, all source columns, no filtering applied.
    """
    frames: list[pd.DataFrame] = []

    # ── Niño indices ──────────────────────────────────────────────────────────
    logger.info("Ingesting Niño indices")
    nino_df = nino_indices.load(
        raw_path=raw_dir / config["nino_indices"]["filename"],
        url=config["nino_indices"]["url"],
        raw_dir=raw_dir,
    )
    frames.append(nino_df)

    # ── SOI ───────────────────────────────────────────────────────────────────
    logger.info("Ingesting SOI")
    soi_df = soi.load(
        raw_path=raw_dir / config["soi"]["filename"],
        url=config["soi"]["url"],
        raw_dir=raw_dir,
    )
    frames.append(soi_df)

    # ── Zonal wind ────────────────────────────────────────────────────────────
    logger.info("Ingesting zonal wind")
    zwnd_df = zonal_wind.load(
        raw_path=raw_dir / config["zonal_wind_850"]["filename"],
        url=config["zonal_wind_850"]["url"],
        raw_dir=raw_dir,
    )
    frames.append(zwnd_df)

    # ── MJO (optional) ────────────────────────────────────────────────────────
    if config.get("mjo", {}).get("enabled", False):
        logger.info("Ingesting MJO")
        mjo_df = mjo.load(
            raw_path=raw_dir / config["mjo"]["filename"],
            url=config["mjo"]["url"],
            raw_dir=raw_dir,
        )
        frames.append(mjo_df)

    # ── Merge ─────────────────────────────────────────────────────────────────
    combined = frames[0].copy()
    for f in frames[1:]:
        combined = combined.join(f, how="outer")

    combined = combined.sort_index()
    logger.info(
        f"Raw dataset assembled: {combined.shape[1]} columns, "
        f"{len(combined)} rows ({combined.index[0]} → {combined.index[-1]})"
    )
    return combined
