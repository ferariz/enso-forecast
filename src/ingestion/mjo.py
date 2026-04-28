"""Loader for BOM Real-time Multivariate MJO (RMM) index.

The RMM index characterises the MJO via two principal components (RMM1, RMM2)
derived from combined EOF analysis of OLR and 850/200 hPa zonal wind.

BOM format (space-delimited):
    year  month  day  RMM1  RMM2  phase  amplitude  source

We collapse daily → monthly by averaging RMM1, RMM2, amplitude.
Phase is re-encoded as sin/cos to preserve circular structure.

Reference:
    Wheeler & Hendon (2004), Mon. Wea. Rev.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.utils.logging import get_logger

logger = get_logger(__name__)

MJO_URL = "http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt"


def fetch_raw(url: str = MJO_URL, save_path: Path | None = None) -> str:
    logger.info(f"Fetching MJO RMM index from {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    text = resp.text
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(text)
    return text


def parse_raw(text: str, amplitude_threshold: float = 1.0) -> pd.DataFrame:
    """Parse and aggregate BOM RMM daily data to monthly.

    When MJO amplitude < *amplitude_threshold*, phase is considered
    ill-defined and sin/cos components are set to 0 (inactive MJO).
    """
    # Skip header lines
    lines = [l for l in text.splitlines()
             if l.strip() and l.strip()[0].isdigit()]
    df = pd.read_csv(
        io.StringIO("\n".join(lines)),
        sep=r"\s+",
        header=None,
        names=["year", "month", "day", "rmm1", "rmm2", "phase", "amplitude", "source"],
        usecols=["year", "month", "day", "rmm1", "rmm2", "phase", "amplitude"],
    )

    # Missing values flagged as 999.xxx in BOM data
    df = df.replace(999.0, float("nan"))
    df = df.replace({col: 999 for col in df.columns}, float("nan"))

    df["date_day"] = pd.to_datetime(df[["year", "month", "day"]])
    df["date"] = df["date_day"].dt.to_period("M").dt.to_timestamp()

    # Encode phase as unit-circle sin/cos (active MJO only)
    angle = 2 * np.pi * (df["phase"] - 1) / 8.0
    df["mjo_sin"] = np.where(df["amplitude"] >= amplitude_threshold, np.sin(angle), 0.0)
    df["mjo_cos"] = np.where(df["amplitude"] >= amplitude_threshold, np.cos(angle), 0.0)

    # Collapse to monthly means
    monthly = (
        df.groupby("date")[["rmm1", "rmm2", "amplitude", "mjo_sin", "mjo_cos"]]
        .mean()
        .rename(columns={"amplitude": "mjo_amplitude"})
    )
    monthly.index.name = "date"
    logger.info(f"Parsed MJO: {len(monthly)} monthly rows")
    return monthly


def load(
    raw_path: Path | None = None,
    url: str = MJO_URL,
    save_raw: bool = True,
    raw_dir: Path = Path("data/raw"),
    amplitude_threshold: float = 1.0,
) -> pd.DataFrame:
    if raw_path is None:
        raw_path = raw_dir / "mjo_raw.txt"

    if Path(raw_path).exists():
        text = Path(raw_path).read_text()
    else:
        save_target = raw_path if save_raw else None
        text = fetch_raw(url=url, save_path=save_target)

    return parse_raw(text, amplitude_threshold=amplitude_threshold)
