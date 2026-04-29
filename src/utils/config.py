"""Config loading utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load(path: str | Path) -> dict[str, Any]:
    """Load a single YAML config file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as fh:
        return yaml.safe_load(fh)


def load_all(config_dir: str | Path = "configs") -> dict[str, dict]:
    """Load all YAML files in config_dir, keyed by stem.

    Returns e.g. {"data_sources": {...}, "features": {...}, ...}
    """
    config_dir = Path(config_dir)
    return {
        yml.stem: load(yml)
        for yml in sorted(config_dir.glob("*.yaml"))
    }
