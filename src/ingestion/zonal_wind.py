"""Loader for NOAA CPC 850 hPa equatorial zonal wind anomaly index.

Format is similar to SOI: wide (years × months), missing = -999.9.
This is a proxy for the Walker circulation strength — a key ENSO precursor.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ingestion.soi import fetch_raw as _fetch_raw
from src.ingestion.soi import parse_raw as _parse_soi_wide
from src.utils.logging import get_logger

logger = get_logger(__name__)

ZWND_URL = "https://www.cpc.ncep.noaa.gov/data/indices/zwnd200"


def load(
    raw_path: Path | None = None,
    url: str = ZWND_URL,
    save_raw: bool = True,
    raw_dir: Path = Path("data/raw"),
) -> pd.DataFrame:
    """Load equatorial 850 hPa zonal wind anomaly.

    Reuses the wide-format parser from the SOI module since the file
    layout is identical; just renames the value column.
    """
    if raw_path is None:
        raw_path = raw_dir / "zwnd850_raw.txt"

    if Path(raw_path).exists():
        logger.info(f"Loading zonal wind from cached file {raw_path}")
        text = Path(raw_path).read_text()
    else:
        save_target = raw_path if save_raw else None
        text = _fetch_raw(url=url, save_path=save_target)

    df = _parse_soi_wide(text).rename(columns={"soi": "zwnd850_anom"})
    return df
