"""Loader for NOAA CPC Niño SST indices (ERSSTv5, base period 1991-2020).

Raw file format (space-delimited):
    YR  MON  NINO1+2  ANOM  NINO3  ANOM  NINO4  ANOM  NINO3.4  ANOM

We keep both raw SST and anomaly columns. Downstream steps use anomalies;
raw SST is retained in interim for reference.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

URL = "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"

# Final column names after parsing the 10-column raw file
COLUMNS = [
    "nino12", "nino12_anom",
    "nino3",  "nino3_anom",
    "nino4",  "nino4_anom",
    "nino34", "nino34_anom",
]


def fetch_raw(url: str = URL, save_path: Path | None = None) -> str:
    """Download the raw ASCII file and optionally cache it to disk."""
    print(f"[nino_indices] Fetching from {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    text = resp.text
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(text)
        print(f"[nino_indices] Saved raw file → {save_path}")
    return text


def parse(text: str) -> pd.DataFrame:
    """Parse raw ASCII text into a monthly-indexed DataFrame."""
    # Keep only lines that start with a 4-digit year
    data_lines = [
        line for line in text.splitlines()
        if line.strip() and line.strip()[:4].isdigit()
    ]

    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        sep=r"\s+",
        header=None,
        names=["year", "month"] + COLUMNS,
    )

    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    )
    df = (
        df.drop(columns=["year", "month"])
          .set_index("date")
          .sort_index()
    )
    print(f"[nino_indices] Parsed {len(df)} rows  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


def load(
    raw_dir: Path = Path("data/raw"),
    url: str = URL,
) -> pd.DataFrame:
    """Load Niño indices: use cached file if available, otherwise fetch.

    Parameters
    ----------
    raw_dir : Path
        Directory where the raw file is cached.
    url : str
        Source URL (used only when cache is absent).

    Returns
    -------
    pd.DataFrame
        Monthly DatetimeIndex, columns: nino12, nino12_anom, ..., nino34_anom.
    """
    cache = raw_dir / "nino_indices_raw.txt"

    if cache.exists():
        print(f"[nino_indices] Loading from cache: {cache}")
        text = cache.read_text()
    else:
        text = fetch_raw(url=url, save_path=cache)

    return parse(text)
