"""Loader for NOAA CPC Southern Oscillation Index (SOI).

NOAA CPC SOI format: fixed-width, years as rows, months as columns.
Header: YEAR  JAN  FEB  MAR  APR  MAY  JUN  JUL  AUG  SEP  OCT  NOV  DEC
Missing value: -999.9
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

from src.utils.logging import get_logger

logger = get_logger(__name__)

SOI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/soi"

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]


def fetch_raw(url: str = SOI_URL, save_path: Path | None = None) -> str:
    logger.info(f"Fetching SOI from {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    text = resp.text
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(text)
        logger.info(f"Raw SOI saved → {save_path}")
    return text


def parse_raw(text: str) -> pd.DataFrame:
    """Reshape wide-format SOI (years × months) into a tidy monthly series."""
    lines = [l for l in text.splitlines() if l.strip() and l.strip()[0].isdigit()]
    df = pd.read_csv(
        io.StringIO("\n".join(lines)),
        sep=r"\s+",
        header=None,
        names=["year"] + MONTHS,
    )
    # Melt to long format
    df = df.melt(id_vars="year", var_name="month_str", value_name="soi")
    df["month"] = pd.Categorical(df["month_str"], categories=MONTHS, ordered=True).codes + 1
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    )
    df = df.set_index("date").sort_index()[["soi"]]
    # Replace NOAA missing value sentinel
    df["soi"] = df["soi"].replace(-999.9, float("nan"))
    logger.info(f"Parsed SOI: {len(df)} rows ({df.index[0]} → {df.index[-1]})")
    return df


def load(
    raw_path: Path | None = None,
    url: str = SOI_URL,
    save_raw: bool = True,
    raw_dir: Path = Path("data/raw"),
) -> pd.DataFrame:
    if raw_path is None:
        raw_path = raw_dir / "soi_raw.txt"

    if Path(raw_path).exists():
        logger.info(f"Loading SOI from cached file {raw_path}")
        text = Path(raw_path).read_text()
    else:
        save_target = raw_path if save_raw else None
        text = fetch_raw(url=url, save_path=save_target)

    return parse_raw(text)
