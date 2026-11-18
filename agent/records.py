"""Canonical GUI SFT/evaluation record validation (V14 DSL).

Replaces the V1 JSON record contract (moved to ``legacy/action_v1.py``).
Every record is one (episode, step) sample::

    {"episode_id", "step", "task", "seed", "instruction", "frame",
     "frame_sha256", "history", "error_feedback", "target_action",
     "action_kind", "data_source", "split", "reward", "success",
     "target_point"}

``target_action`` is the authoritative DSL string; ``action_kind`` and
``target_point`` must agree with it (click needs a point, type points at the
copied span, scroll carries both deltas).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.dsl import ActionParseError, parse_action
from agent.copy_resolve import CopyPointer
from agent.vision_tiles import TILE_NAMES

__all__ = ["RecordError", "SOURCES", "SPLITS", "validate_record"]

SOURCES = (
    "gui_owl_teacher",
    "v11_teacher",
    "oracle",
    "old_gui",
    "language_replay",
    "legacy",
)
SPLITS = ("train", "val", "test")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED = (
    "episode_id", "step", "task", "seed", "instruction", "target_action",
    "action_kind", "data_source", "split",
)


class RecordError(ValueError):
    """Raised when a record violates the canonical GUI schema."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_record(record: Dict[str, Any], *, require_frame: bool = False) -> Dict[str, Any]:
    """Validate one record and return a normalised copy (no file writes)."""
    if not isinstance(record, dict):
        raise RecordError("record must be a mapping")
    missing = [key for key in _REQUIRED if key not in record]
    if missing:
        raise RecordError("missing fields: %s" % ", ".join(missing))
    if record["split"] not in SPLITS:
        raise RecordError("split must be one of %s, got %r" % (SPLITS, record["split"]))
    if record["data_source"] not in SOURCES:
        raise RecordError("data_source must be one of %s, got %r"
                          % (SOURCES, record["data_source"]))
    try:
        step = int(record["step"])
        seed = int(record["seed"])
    except (TypeError, ValueError) as exc:
        raise RecordError("step and seed must be integers") from exc
    if step < 0:
        raise RecordError("step must be >= 0")
    episode_id = str(record["episode_id"])
    if not episode_id:
        raise RecordError("episode_id is empty")
    instruction = record["instruction"]
    if not isinstance(instruction, str) or not instruction.strip():
        raise RecordError("instruction must be non-empty text")
    try:
        kind, value = parse_action(record["target_action"])
        if instruction:
            parse_action(record["target_action"], instruction)
    except ActionParseError as exc:
        raise RecordError("invalid target_action: %s" % exc) from exc
    if record["action_kind"] != kind:
        raise RecordError("action_kind %r disagrees with target_action kind %r"
                          % (record["action_kind"], kind))
    point = record.get("target_point")
    if kind == "click":
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise RecordError("click record needs target_point [x, y]")
        if not all(0.0 <= float(v) <= 1.0 for v in point):
            raise RecordError("target_point must be in [0, 1]")
        point = [float(point[0]), float(point[1])]
    elif kind == "type" and isinstance(value, CopyPointer):
        point = [value.start, value.end]
    else:
        point = None if point is None else list(point)
    history = record.get("history") or []
    if not isinstance(history, list) or not all(isinstance(h, str) for h in history):
        raise RecordError("history must be a list of DSL strings")
    for entry in history:
        try:
            parse_action(entry, instruction)
        except ActionParseError as exc:
            raise RecordError("invalid history action %r: %s" % (entry, exc)) from exc
    error = record.get("error_feedback") or ""
    if not isinstance(error, str):
        raise RecordError("error_feedback must be text")
    reward = float(record.get("reward", 0.0) or 0.0)
    success = bool(record.get("success", False))
    frame = record.get("frame")
    frame_sha = record.get("frame_sha256")
    if require_frame:
        if not frame:
            raise RecordError("frame path is required")
        path = Path(frame)
        if not path.is_file():
            raise RecordError("frame not found: %s" % path)
        actual = _sha256_file(path)
        if frame_sha and actual != frame_sha:
            raise RecordError("frame sha256 mismatch for %s" % path)
        frame_sha = actual
    elif frame_sha and not _SHA256.match(str(frame_sha)):
        raise RecordError("frame_sha256 must be 64 lowercase hex characters")
    return {
        "episode_id": episode_id,
        "step": step,
        "task": str(record["task"]),
        "seed": seed,
        "instruction": instruction,
        "frame": frame,
        "frame_sha256": frame_sha,
        "history": history,
        "error_feedback": error,
        "target_action": record["target_action"],
        "action_kind": kind,
        "data_source": record["data_source"],
        "split": record["split"],
        "reward": reward,
        "success": success,
        "target_point": point,
    }
