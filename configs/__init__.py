"""Configuration loading utilities.

Loads the centralized default.yaml and applies optional dotted-key overrides
supplied from command-line arguments, so every script can be configured without
editing source files.
"""
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent / "default.yaml"


def _set_dotted(cfg, dotted, value):
    node = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def load_config(path=None, overrides=None):
    """Return the config dict from ``path`` (default.yaml when omitted).

    ``overrides`` is a mapping of dotted keys to values; entries whose value is
    None are ignored so callers may pass argparse namespaces verbatim.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    for key, value in (overrides or {}).items():
        if value is not None:
            _set_dotted(cfg, key, value)
    return cfg
