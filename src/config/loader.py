"""YAML configuration loader with recursive base inheritance.

The previous experiment entry points each had a slightly different shallow
``_base_`` merge.  Keeping the loader here makes configuration a stable API
without changing those entry points at once.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load *path*, recursively merge ``_base_``, and add resolved metadata."""
    path = Path(path).resolve()

    def read(current: Path, ancestry: tuple[Path, ...]) -> dict[str, Any]:
        if current in ancestry:
            chain = " -> ".join(str(item) for item in (*ancestry, current))
            raise ValueError(f"Cyclic YAML inheritance: {chain}")
        with current.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise TypeError(f"Configuration root must be a mapping: {current}")
        base_name = data.pop("_base_", None)
        if base_name is None:
            return data
        base_path = (current.parent / base_name).resolve()
        return _deep_merge(read(base_path, (*ancestry, current)), data)

    config = read(path, ())
    config.setdefault("runtime", {})["config_path"] = str(path)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Fail early for only the invariants shared by all supported runs."""
    env = config.get("env", {})
    if env.get("k_stas", env.get("k_stas_per_ap", 1)) < 1:
        raise ValueError("env.k_stas must be at least one")
    if env.get("max_episodes", 1) < 1:
        raise ValueError("env.max_episodes must be at least one")
    if config.get("train", {}).get("batch_size", 1) < 1:
        raise ValueError("train.batch_size must be at least one")
