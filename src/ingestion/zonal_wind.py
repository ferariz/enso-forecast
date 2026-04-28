"""Loader for NOAA CPC equatorial 850 hPa zonal wind anomaly.

File format is identical to SOI (wide, years × months, missing = -999.9).
This index measures the strength of the Walker circulation — westerly
anomalies signal weakening trade winds and precede El Niño development.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ingestion.soi import fetch_raw, parse as _parse_wide

URL = "https://www.cpc.ncep.noaa.gov/data/indices/zwnd200"


def load(
    raw_dir: Path = Path("data/raw"),
    url: str = URL,
) -> pd.DataFrame:
    """Load 850 hPa equatorial zonal wind anomaly.

    Reuses the SOI wide-format parser — only the column name differs.
    """
    cache = raw_dir / "zwnd850_raw.txt"

    if cache.exists():
        print(f"[zonal_wind] Loading from cache: {cache}")
        text = cache.read_text()
    else:
        print(f"[zonal_wind] Fetching from {url}")
        text = fetch_raw(url=url, save_path=cache)

    df = _parse_wide(text).rename(columns={"soi": "zwnd850_anom"})
    print(f"[zonal_wind] {len(df)} rows  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df
