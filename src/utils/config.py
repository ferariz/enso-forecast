"""Config loading utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return as a plain dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as fh:
        return yaml.safe_load(fh)


def load_all_configs(config_dir: str | Path = "configs") -> dict[str, dict]:
    """Load all YAML files in *config_dir* and return as a nested dict keyed by stem."""
    config_dir = Path(config_dir)
    configs: dict[str, dict] = {}
    for yml in sorted(config_dir.glob("*.yaml")):
        configs[yml.stem] = load_config(yml)
    return configs
