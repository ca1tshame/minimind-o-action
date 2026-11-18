"""Configuration loading and workspace/path resolution.

All machine-specific locations are supplied at run time through
``--workspace-root`` or ``MINIMIND_WORKSPACE`` (or a ``paths.workspace_root``
entry); ``configs/default.yaml`` itself contains only relative paths.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

__all__ = [
    "DEFAULT_CONFIG",
    "REPO_ROOT",
    "apply_overrides",
    "load_config",
    "require_paths",
    "resolve_device",
    "resolve_paths",
    "workspace_root",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG: Dict[str, Any] = {}

_URL_KEYS = {"miniwob_url"}


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply dotted-key overrides in place, ignoring ``None`` values."""
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        parts = key.split(".")
        node: Dict[str, Any] = cfg
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = value
    return cfg


def load_config(path: Optional[str] = None,
                overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load a YAML config (default: ``configs/default.yaml``) and apply overrides."""
    if path is None:
        path = REPO_ROOT / "configs" / "default.yaml"
        if not Path(path).is_file():
            return apply_overrides(copy.deepcopy(DEFAULT_CONFIG), overrides)
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("config %s must be a mapping" % path)
    return apply_overrides(data, overrides)


def workspace_root(cfg: Dict[str, Any], cli_root: Optional[str] = None) -> Path:
    """Resolve the workspace root: CLI > MINIMIND_WORKSPACE > config > repo root."""
    candidate = cli_root or os.environ.get("MINIMIND_WORKSPACE")
    if not candidate:
        candidate = (cfg.get("paths") or {}).get("workspace_root")
    root = Path(candidate).expanduser().resolve() if candidate else REPO_ROOT
    if not root.is_dir():
        raise FileNotFoundError("workspace root does not exist: %s" % root)
    return root


def resolve_paths(cfg: Dict[str, Any], root: Path) -> Dict[str, Optional[Path]]:
    """Resolve every ``paths.*`` entry against ``root``.

    URLs are returned unchanged as strings under their own key; relative paths
    are resolved against the workspace root, ``~`` is expanded.  Absolute
    paths are kept as-is (useful for one-off runs) but never required.
    """
    resolved: Dict[str, Optional[Path]] = {}
    for key, value in (cfg.get("paths") or {}).items():
        if value is None:
            resolved[key] = None
        elif key in _URL_KEYS or (isinstance(value, str) and "://" in value):
            resolved[key] = value  # type: ignore[assignment]
        else:
            path = Path(str(value)).expanduser()
            resolved[key] = path if path.is_absolute() else (root / path)
    return resolved


def require_paths(paths: Dict[str, Any], keys: Iterable[str], *, kind: str = "file") -> None:
    """Fail early with a helpful message when required inputs are missing."""
    missing = []
    for key in keys:
        value = paths.get(key)
        if value is None:
            missing.append("%s (not configured)" % key)
            continue
        path = Path(value)
        ok = path.is_file() if kind == "file" else path.exists()
        if not ok:
            missing.append("%s (%s)" % (key, path))
    if missing:
        raise FileNotFoundError(
            "missing required %s(s): %s; set them in the config or pass "
            "--workspace-root/MINIMIND_WORKSPACE" % (kind, ", ".join(missing))
        )


def resolve_device(cfg: Dict[str, Any]) -> str:
    """Return the requested device, falling back to CPU when CUDA is absent."""
    device = str((cfg.get("device") or {}).get("device", "cuda"))
    if device == "cpu":
        return device
    try:
        import torch

        if device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
    except Exception:
        return "cpu"
    return device
