"""Configuration loader for RedactIQ."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration, with env-var overrides."""
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # Allow environment variable overrides for key settings
    env_overrides = {
        "REDACTIQ_MODEL_NAME": ("model", "name"),
        "REDACTIQ_DEVICE": ("vllm", "device"),
        "REDACTIQ_PORT": ("serving", "port"),
        "REDACTIQ_LOG_LEVEL": ("monitoring", "log_level"),
        "REDACTIQ_QUANTIZATION": ("model", "quantization"),
    }
    for env_var, key_path in env_overrides.items():
        value = os.environ.get(env_var)
        if value is not None:
            obj = cfg
            for k in key_path[:-1]:
                obj = obj.setdefault(k, {})
            # Convert port to int
            if key_path[-1] == "port":
                value = int(value)
            obj[key_path[-1]] = value

    return cfg
