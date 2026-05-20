"""Loader for NOAA PMEL Warm Water Volume (WWV) index.

WWV is the volume of water above the 20°C isotherm integrated over
the equatorial Pacific (5°S–5°N, 120°E–80°W). It is a direct measure
of subsurface heat content and a leading indicator of ENSO development.

Key physical relationship:
    - Positive WWV anomaly → thermocline deepening → El Niño precursor
    - Negative WWV anomaly → thermocline shoaling → La Niña precursor
    - WWV leads Niño 3.4 SST by approximately 2–3 quarters (Meinen & McPhaden 2000)

Source: NOAA PMEL GTMBA Project
URL:    https://www.pmel.noaa.gov/tao/wwv/data/wwv.dat
Format: YYYYMM  Volume(m³)  Anomaly(m³)

Reference:
    Meinen, C.S. and M.J. McPhaden, 2000: Observations of warm water
    volume changes in the equatorial Pacific and their relationship to
    El Niño and La Niña. J. Climate, 13, 3551-3559.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

WWV_URL  = "https://www.pmel.noaa.gov/tao/wwv/data/wwv.dat"
RAW_FILE = "data/raw/wwv_raw.txt"

# Normalisation base period — match NOAA's anomaly computation
BASE_PERIOD = ("1981-01", "2010-12")


def fetch(
    cache_path: str | Path = RAW_FILE,
    force: bool = False,
) -> str:
    """Download the WWV file and cache it locally.

    Parameters
    ----------
    cache_path : path to save the raw file
    force : re-download even if cache exists

    Returns
    -------
    str : raw file content
    """
    import urllib.request

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force:
        print(f"[wwv] Loading from cache: {cache_path}")
        return cache_path.read_text()

    print(f"[wwv] Fetching from {WWV_URL}")
    with urllib.request.urlopen(WWV_URL, timeout=30) as r:
        content = r.read().decode("utf-8", errors="replace")
    cache_path.write_text(content)
    print(f"[wwv] Saved raw file → {cache_path}")
    return content


def parse(content: str) -> pd.DataFrame:
    """Parse raw WWV file into a monthly DataFrame.

    Returns
    -------
    pd.DataFrame with DatetimeIndex and columns:
        wwv_volume  : absolute volume (m³)
        wwv_anom    : anomaly relative to NOAA base period (m³)
        wwv_anom_std: anomaly standardised to unit variance (dimensionless)
    """
    rows = []
    for line in content.splitlines():
        line = line.strip()
        # Skip header lines — data lines start with 6-digit YYYYMM
        if not line or not line[:6].isdigit():
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            yyyymm = parts[0]
            year   = int(yyyymm[:4])
            month  = int(yyyymm[4:6])
            volume = float(parts[1])
            anom   = float(parts[2])
            rows.append((year, month, volume, anom))
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(rows, columns=["year", "month", "wwv_volume", "wwv_anom"])
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    )
    df = df.set_index("date").drop(columns=["year", "month"])
    df = df.sort_index()

    # Standardise anomaly (zero mean, unit variance) over base period
    base = df.loc[BASE_PERIOD[0]:BASE_PERIOD[1], "wwv_anom"]
    df["wwv_anom_std"] = (df["wwv_anom"] - base.mean()) / base.std()

    n = len(df)
    t0 = df.index[0].strftime("%Y-%m-%d")
    t1 = df.index[-1].strftime("%Y-%m-%d")
    print(f"[wwv] Parsed {n} rows  ({t0} → {t1})")
    print(f"[wwv] Missing values: {df.isna().sum().to_dict()}")

    return df


def load(
    cache_path: str | Path = RAW_FILE,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch and parse the WWV index in one call."""
    content = fetch(cache_path=cache_path, force=force)
    return parse(content)
