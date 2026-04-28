"""Loader for NOAA CPC Niño indices (ERSSTv5 base period 1991–2020).

Raw data format (space-delimited ASCII):
    YR  MON  NINO1+2  ANOM  NINO3  ANOM  NINO4  ANOM  NINO3.4  ANOM
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Default NOAA URL (may need updating if NOAA changes layout)
NINO_URL = "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"

RAW_COLS = [
    "year", "month",
    "nino12",     "nino12_anom",
    "nino3",      "nino3_anom",
    "nino4",      "nino4_anom",
    "nino34",     "nino34_anom",
]


def fetch_raw(url: str = NINO_URL, save_path: Path | None = None) -> str:
    """Download raw ASCII text and optionally persist to *save_path*."""
    logger.info(f"Fetching Niño indices from {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    text = resp.text
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(text)
        logger.info(f"Raw Niño indices saved → {save_path}")
    return text


def parse_raw(text: str) -> pd.DataFrame:
    """Parse the raw ASCII table into a tidy DataFrame with a DatetimeIndex."""
    # Skip header line(s) — anything that doesn't start with a 4-digit year
    lines = [l for l in text.splitlines() if l.strip() and l.strip()[0].isdigit()]
    df = pd.read_csv(
        io.StringIO("\n".join(lines)),
        sep=r"\s+",
        header=None,
        names=RAW_COLS,
    )
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    )
    df = df.drop(columns=["year", "month"])
    df = df.set_index("date").sort_index()
    logger.info(f"Parsed Niño indices: {len(df)} rows ({df.index[0]} → {df.index[-1]})")
    return df


def load(
    raw_path: Path | None = None,
    url: str = NINO_URL,
    save_raw: bool = True,
    raw_dir: Path = Path("data/raw"),
) -> pd.DataFrame:
    """High-level entry point.

    If *raw_path* exists on disk, parse from file (no network call).
    Otherwise fetch from *url* and optionally save to *raw_dir*.
    """
    if raw_path is None:
        raw_path = raw_dir / "nino_indices_raw.txt"

    if Path(raw_path).exists():
        logger.info(f"Loading Niño indices from cached file {raw_path}")
        text = Path(raw_path).read_text()
    else:
        save_target = raw_path if save_raw else None
        text = fetch_raw(url=url, save_path=save_target)

    return parse_raw(text)
