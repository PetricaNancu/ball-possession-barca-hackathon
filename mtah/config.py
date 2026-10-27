"""YAML config: `configs/default.yaml` overridden by an optional file passed on the command line."""

from __future__ import annotations

from pathlib import Path

import yaml

from . import paths


def load_config(override: str | Path | None = None) -> dict:
    with open(paths.CONFIG_DIR / "default.yaml") as fh:
        config = yaml.safe_load(fh) or {}
    if override:
        with open(override) as fh:
            for key, value in (yaml.safe_load(fh) or {}).items():
                if isinstance(value, dict) and isinstance(config.get(key), dict):
                    config[key].update(value)
                else:
                    config[key] = value
    return config
