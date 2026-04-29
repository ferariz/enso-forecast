"""Loader for NOAA CPC Southern Oscillation Index (SOI).

Raw file format — wide, years as rows, months as columns:
    YEAR  JAN  FEB  MAR  APR  MAY  JUN  JUL  AUG  SEP  OCT  NOV  DEC

Missing value sentinel: -999.9
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

URL = "https://www.cpc.ncep.noaa.gov/data/indices/soi"

MONTH_COLS = ["jan", "feb", "mar", "apr", "may", "jun",
              "jul", "aug", "sep", "oct", "nov", "dec"]

MISSING = -999.9


def fetch_raw(url: str = URL, save_path: Path | None = None) -> str:
    print(f"[soi] Fetching from {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    text = resp.text
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(text)
        print(f"[soi] Saved raw file → {save_path}")
    return text


def _is_valid_year_line(line: str) -> bool:
    """Return True only for lines whose first token is a plausible calendar year.

    The real NOAA SOI file contains:
    - Header rows starting with letters ("STANDARDIZED", "YEAR", etc.)
    - Data rows starting with a 4-digit year (1950–2099)
    - Sentinel rows where year=9999 (all-missing placeholder) — must exclude
    - Rows with no whitespace where -999.9 values run together — start with "-"
    """
    token = line.strip().split()[0] if line.strip() else ""
    if len(token) != 4 or not token.isdigit():
        return False
    year = int(token)
    return 1900 <= year <= 2099   # exclude 9999 sentinel


def parse(text: str) -> pd.DataFrame:
    """Parse wide-format SOI into a tidy monthly series."""
    data_lines = [
        line for line in text.splitlines()
        if _is_valid_year_line(line)
    ]

    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        sep=r"\s+",
        header=None,
        names=["year"] + MONTH_COLS,
    )

    # Wide → long
    df = df.melt(id_vars="year", var_name="month_str", value_name="soi")

    # Map month abbreviation to integer
    month_num = {m: i + 1 for i, m in enumerate(MONTH_COLS)}
    df["month"] = df["month_str"].map(month_num)

    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    )

    df = (
        df[["date", "soi"]]
          .set_index("date")
          .sort_index()
    )

    # Replace NOAA missing sentinel with NaN
    df["soi"] = df["soi"].replace(MISSING, float("nan"))

    print(f"[soi] Parsed {len(df)} rows  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


def load(
    raw_dir: Path = Path("data/raw"),
    url: str = URL,
) -> pd.DataFrame:
    """Load SOI: use cached file if available, otherwise fetch."""
    cache = raw_dir / "soi_raw.txt"

    if cache.exists():
        print(f"[soi] Loading from cache: {cache}")
        text = cache.read_text()
    else:
        text = fetch_raw(url=url, save_path=cache)

    return parse(text)
