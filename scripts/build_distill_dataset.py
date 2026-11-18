"""Merge the V11 teacher, V14 oracle and GUI-Owl sources into one dataset.

Pipeline (counts from the published V14 run)::

    raw sources            3010   V11 teacher 2259 + V14 oracle 705 + GUI-Owl 46
    after BLOCKED_SEEDS   -60    every frozen-evaluation seed is dropped
    after content dedup   -35    key = (frame_sha256, target_action, instruction[:80])
    after task balancing -132    easy_cap = 1200 rows per task
    = 2783 unique rows           2552 train / 231 validation

``2915`` is the number of *unique rows after dedup*; it is not the number of
rows trained on.  Only ``train.jsonl`` is trained on, ``validation.jsonl`` is
held out, and a frozen test set is never written into either split.

The ``frame`` field is relative to ``--frames-root`` (default: the workspace
root), which is the same root ``scripts/train_gui_sft.py`` reads frames from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.filter_teacher import (  # noqa: E402
    ACTION_KINDS,
    CANONICAL_FIELDS,
    DATA_SOURCES,
    HEX64,
    SPLITS,
    ActionDSLError,
    load_jsonl,
    parse_action_dsl,
    sha256_file,
    write_jsonl,
)

HARD_TASKS = (
    "click-button-sequence",
    "use-autocomplete",
    "click-pie",
    "click-option",
    "scroll-text",
    "click-link",
    "login-user-popup",
    "enter-text-2",
)

BLOCKED_SEEDS = frozenset(
    set(range(20001, 20017))
    | set(range(21001, 21009))
    | set(range(13101, 13109))
    | set(range(15101, 15109))
    | set(range(19101, 19109))
    | set(range(9101, 9113))
    | set(range(8111, 8114))
    | set(range(12101, 12131))
    | set(range(16101, 16107))
    | set(range(14101, 14200))
    | set(range(18101, 18104))
)

DEDUP_KEY_FIELDS = ("frame_sha256", "target_action", "instruction[:80]")


class RecordError(ValueError):
    """A source row that cannot be represented as a canonical record."""

    def __init__(self, reason, detail=""):
        message = reason if not detail else "%s: %s" % (reason, detail)
        super().__init__(message)
        self.reason = reason
        self.detail = detail


def _finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _first_str(mapping, *keys):
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _coerce_seed(value):
    if isinstance(value, bool):
        raise RecordError("seed_invalid", repr(value))
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    raise RecordError("seed_invalid", repr(value))


def gui_owl_native_dsl(value):
    """Convert a GUI-Owl adapter action dict into the constrained DSL."""
    if not isinstance(value, dict):
        raise RecordError("target_action_not_string", type(value).__name__)
    kind = str(value.get("action") or value.get("action_type") or "").lower()
    if kind == "click":
        point = value.get("point") or value.get("position")
        if point is None and "x" in value and "y" in value:
            point = [value.get("x"), value.get("y")]
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise RecordError("dsl_coordinate", "click without a 2D point")
        x, y = float(point[0]), float(point[1])
        if max(abs(x), abs(y)) > 1.5:
            x, y = x / 1000.0, y / 1000.0
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise RecordError("dsl_coordinate", "click outside [0,1]: %r" % (point,))
        return "<action>click(%.4f,%.4f)</action>" % (x, y)
    if kind == "type":
        text = value.get("text", value.get("value"))
        if not isinstance(text, str) or not text:
            raise RecordError("type_not_string", repr(text))
        return '<action>type(%s)</action>' % json.dumps(text, ensure_ascii=False)
    if kind in ("scroll", "scroll_at"):
        dx = float(value.get("dx", 0.0))
        dy = float(value.get("dy", 0.0))
        inside = bool(value.get("inside", True))
        if not (-1.0 <= dx <= 1.0 and -1.0 <= dy <= 1.0):
            raise RecordError("dsl_coordinate", "scroll outside [-1,1]")
        return "<action>scroll(%.4f,%.4f,%s)</action>" % (dx, dy, inside)
    if kind in ("stop", "done", "finished", "success"):
        return "<action>stop(done)</action>"
    raise RecordError("unsupported_action", kind or "<missing>")


def to_type_pointer(instruction, text):
    """Copy-pointer encoding: ``text`` -> ``<action>type(@S:E[|lower|upper])</action>``.

    Mirrors ``project/scripts/gui_v8_type_copy.py``: exact match first, then a
    lower-cased span equal to ``text``, then an upper-cased span. Returns
    ``None`` when the text cannot be copied from the instruction (the caller
    keeps the plain ``type("...")`` form).
    """
    if not isinstance(text, str) or not text or not isinstance(instruction, str):
        return None
    if len(text) > len(instruction):
        return None
    for transform in (None, "lower", "upper"):
        for start in range(0, len(instruction) - len(text) + 1):
            end = start + len(text)
            segment = instruction[start:end]
            if transform is None:
                if segment == text:
                    return "<action>type(@%d:%d)</action>" % (start, end)
            elif transform == "lower":
                if segment != text and segment.lower() == text:
                    return "<action>type(@%d:%d|lower)</action>" % (start, end)
            else:
                if segment != text and segment.upper() == text:
                    return "<action>type(@%d:%d|upper)</action>" % (start, end)
    return None


def apply_type_pointer(instruction, target_action):
    """Resolve a pointer-encoded ``type`` action back to its literal text."""
    try:
        kind, payload = parse_action_dsl(target_action, instruction)
    except ActionDSLError as exc:
        raise RecordError("target_action:" + exc.reason, exc.detail)
    if kind != "type":
        raise RecordError("not_a_type_action", target_action)
    return payload["text"]


def _normalized_frame(value):
    raw = str(value).strip().replace("\\", "/")
    if raw.startswith("/") or re.match(r"^[A-Za-z]:/", raw) or raw.startswith("//"):
        raise RecordError("frame_absolute", raw)
    if any(part == ".." for part in raw.split("/")):
        raise RecordError("frame_outside_frames_root", raw)
    return raw


def normalize_row(raw, data_source, frames_root=None, check_frames=True, split=None):
    """Coerce one source row into an exact canonical record (fail closed)."""
    if not isinstance(raw, dict):
        raise RecordError("row_not_object", type(raw).__name__)
    if data_source not in DATA_SOURCES:
        raise RecordError("unknown_data_source", str(data_source))

    task = _first_str(raw, "task", "task_id")
    if not task:
        raise RecordError("task_invalid")
    seed = _coerce_seed(raw.get("seed", 0))
    instruction = _first_str(raw, "instruction", "goal")
    if not instruction:
        raise RecordError("instruction_invalid")

    target = _first_str(raw, "target_action", "dsl")
    if not target and isinstance(raw.get("action"), dict):
        target = gui_owl_native_dsl(raw["action"])
    if not target:
        raise RecordError("missing_target_action")
    try:
        kind, payload = parse_action_dsl(target, instruction)
    except ActionDSLError as exc:
        raise RecordError(exc.reason, exc.detail)

    step = raw.get("step", 0)
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise RecordError("step_invalid", repr(step))

    frame = _first_str(raw, "frame", "frame_path")
    if not frame:
        raise RecordError("missing_frame")
    frame = _normalized_frame(frame)
    recorded_hash = raw.get("frame_sha256")
    digest = None
    if frames_root is not None:
        root = Path(frames_root).resolve()
        candidate = (root / frame).resolve()
        if candidate != root and root not in candidate.parents:
            raise RecordError("frame_outside_frames_root", frame)
        if candidate.is_file():
            digest = sha256_file(candidate)
            if recorded_hash is not None and str(recorded_hash).lower() != digest:
                raise RecordError(
                    "frame_hash_mismatch", "%s != %s" % (recorded_hash, digest)
                )
            frame = candidate.relative_to(root).as_posix()
        elif check_frames:
            raise RecordError("missing_frame", frame)
    if digest is None:
        if not isinstance(recorded_hash, str) or not HEX64.match(
            recorded_hash.lower()
        ):
            raise RecordError("missing_frame_hash", repr(recorded_hash))
        digest = recorded_hash.lower()

    reward = raw.get("reward", raw.get("env_reward"))
    if reward is not None:
        if not _finite(reward):
            raise RecordError("reward_invalid", repr(reward))
        reward = float(reward)
    success = raw.get("success")
    if success is not None and not isinstance(success, bool):
        raise RecordError("success_invalid", repr(success))

    history = raw.get("history")
    if history is None:
        history = []
    if not isinstance(history, list) or any(not isinstance(h, str) for h in history):
        raise RecordError("history_invalid")
    error_feedback = raw.get("error_feedback")
    if error_feedback is not None and not isinstance(error_feedback, str):
        raise RecordError("error_feedback_invalid")

    point = None
    if kind == "click":
        point = [payload["x"], payload["y"]]
        supplied = raw.get("target_point")
        if supplied is not None:
            if (
                not isinstance(supplied, (list, tuple))
                or len(supplied) != 2
                or not all(_finite(value) for value in supplied)
            ):
                raise RecordError("target_point_invalid", repr(supplied))
            if any(not (0.0 <= float(value) <= 1.0) for value in supplied):
                raise RecordError("target_point_out_of_range", repr(supplied))
            if (
                abs(float(supplied[0]) - point[0]) > 1e-6
                or abs(float(supplied[1]) - point[1]) > 1e-6
            ):
                raise RecordError("target_point_mismatch", repr(supplied))

    episode_id = raw.get("episode_id")
    episode_id = str(episode_id) if episode_id else "%s_%s" % (task, seed)

    resolved_split = split or raw.get("split") or "train"
    if resolved_split not in SPLITS:
        raise RecordError("split_invalid", str(resolved_split))

    record = {
        "episode_id": episode_id,
        "step": step,
        "task": task,
        "seed": seed,
        "instruction": instruction,
        "frame": frame,
        "frame_sha256": digest,
        "history": list(history),
        "error_feedback": error_feedback,
        "target_action": target.strip(),
        "action_kind": kind,
        "data_source": data_source,
        "split": resolved_split,
        "reward": reward,
        "success": success,
        "target_point": point,
    }
    problems = check_record(record)
    if problems:
        raise RecordError("schema", "; ".join(problems))
    return record


def check_record(record):
    """Return a list of schema/DSL/range problems for a canonical record."""
    if not isinstance(record, dict):
        return ["record_not_object"]
    problems = []
    missing = [field for field in CANONICAL_FIELDS if field not in record]
    extra = sorted(set(record) - set(CANONICAL_FIELDS))
    if missing:
        problems.append("missing_fields:" + ",".join(missing))
    if extra:
        problems.append("extra_fields:" + ",".join(extra))

    episode_id = record.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        problems.append("episode_id_invalid")
    step = record.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        problems.append("step_invalid")
    task = record.get("task")
    if not isinstance(task, str) or not task:
        problems.append("task_invalid")
    seed = record.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        problems.append("seed_invalid")
    instruction = record.get("instruction")
    if not isinstance(instruction, str) or not instruction:
        problems.append("instruction_invalid")

    frame = record.get("frame")
    if not isinstance(frame, str) or not frame:
        problems.append("frame_invalid")
    else:
        normalized = frame.replace("\\", "/")
        if (
            normalized.startswith("/")
            or re.match(r"^[A-Za-z]:/", normalized)
            or normalized.startswith("//")
        ):
            problems.append("frame_absolute")
        elif any(part == ".." for part in normalized.split("/")):
            problems.append("frame_outside_frames_root")

    digest = record.get("frame_sha256")
    if not isinstance(digest, str) or not HEX64.match(digest.lower()):
        problems.append("frame_sha256_invalid")
    history = record.get("history")
    if not isinstance(history, list) or any(
        not isinstance(item, str) for item in history
    ):
        problems.append("history_invalid")
    error_feedback = record.get("error_feedback")
    if error_feedback is not None and not isinstance(error_feedback, str):
        problems.append("error_feedback_invalid")

    action_kind = record.get("action_kind")
    if action_kind not in ACTION_KINDS:
        problems.append("action_kind_invalid")
    target = record.get("target_action")
    if not isinstance(target, str) or not target.strip():
        problems.append("target_action_invalid")
    else:
        try:
            parsed_kind, payload = parse_action_dsl(
                target, instruction if isinstance(instruction, str) else ""
            )
        except ActionDSLError as exc:
            problems.append("target_action:" + exc.reason)
        else:
            if parsed_kind != action_kind:
                problems.append("action_kind_mismatch")
            point = record.get("target_point")
            if parsed_kind == "click":
                if (
                    not isinstance(point, (list, tuple))
                    or len(point) != 2
                    or not all(_finite(value) for value in point)
                ):
                    problems.append("target_point_invalid")
                elif any(not (0.0 <= float(value) <= 1.0) for value in point):
                    problems.append("target_point_out_of_range")
                elif (
                    abs(float(point[0]) - payload["x"]) > 1e-6
                    or abs(float(point[1]) - payload["y"]) > 1e-6
                ):
                    problems.append("target_point_mismatch")
            elif point is not None:
                problems.append("target_point_unexpected")

    if record.get("data_source") not in DATA_SOURCES:
        problems.append("data_source_invalid")
    if record.get("split") not in SPLITS:
        problems.append("split_invalid")
    reward = record.get("reward")
    if reward is not None and not _finite(reward):
        problems.append("reward_invalid")
    success = record.get("success")
    if success is not None and not isinstance(success, bool):
        problems.append("success_invalid")
    return problems


def dedup_key(record):
    return (
        record["frame_sha256"],
        record["target_action"],
        record["instruction"][:80],
    )


def row_sort_key(record):
    return (
        record["task"],
        record["seed"],
        record["episode_id"],
        record["step"],
        record["frame_sha256"],
        record["target_action"],
    )


def load_frozen_test(path):
    """Load a frozen test manifest (JSON) or a frozen record list (JSONL)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("frozen test file not found: %s" % path)
    if path.suffix.lower() == ".jsonl":
        records = load_jsonl(path)
        payload = {"records": records}
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("frozen test JSON must be an object: %s" % path)
    records = payload.get("records") or []
    seeds = {int(value) for value in payload.get("seeds") or [] if _finite(value)}
    episodes = {str(value) for value in payload.get("episodes") or [] if value}
    tasks = {str(value) for value in payload.get("tasks") or [] if value}
    for record in records:
        if not isinstance(record, dict):
            continue
        if isinstance(record.get("seed"), int):
            seeds.add(record["seed"])
        if record.get("episode_id"):
            episodes.add(str(record["episode_id"]))
    return {
        "path": path,
        "tasks": tasks,
        "seeds": seeds,
        "episodes": episodes,
        "records": records,
        "task_level": bool(tasks),
        "errors": payload.get("errors") or [],
    }


def frozen_conflict(record, frozen):
    if not frozen:
        return None
    if frozen.get("task_level") and record["task"] in frozen["tasks"]:
        return "frozen_task"
    if record["seed"] in frozen["seeds"]:
        return "frozen_seed"
    if record["episode_id"] in frozen["episodes"]:
        return "frozen_episode"
    return None


def _counts(rows, order):
    return {source: sum(1 for row in rows if row["data_source"] == source) for source in order}


def _diff_counts(new, old):
    result = {}
    for source, value in old.items():
        difference = value - new.get(source, 0)
        if difference:
            result[source] = difference
    return result


def _stage(rows, order, removed=0, removed_by_source=None, **extra):
    stage = {
        "total": len(rows),
        "by_source": _counts(rows, order),
        "removed": removed,
    }
    if removed_by_source:
        stage["removed_by_source"] = removed_by_source
    stage.update(extra)
    return stage


def build_dataset(
    sources,
    source_paths=None,
    frames_root=None,
    check_frames=True,
    blocked_seeds=None,
    seed_block_exempt=("v5_old",),
    easy_cap=1200,
    hard_tasks=HARD_TASKS,
    split_seed=14,
    val_divisor=12,
    min_val=0,
    frozen=None,
    strict=False,
    workspace_root=None,
):
    """Run the merge pipeline on in-memory source rows.

    ``sources`` maps ``data_source`` -> iterable of raw rows.  Returns
    ``train``/``validation``/``test`` record lists plus a manifest dict.
    """
    blocked = set(BLOCKED_SEEDS if blocked_seeds is None else blocked_seeds)
    exempt = set(seed_block_exempt or ())
    hard = set(hard_tasks or ())
    sources = sources or {}
    order = [source for source in DATA_SOURCES if source in sources]
    order += sorted(source for source in sources if source not in DATA_SOURCES)

    normalized = []
    invalid = []
    raw_by_source = {}
    for source in order:
        rows = list(sources.get(source) or [])
        raw_by_source[source] = len(rows)
        for index, raw in enumerate(rows):
            try:
                normalized.append(
                    normalize_row(
                        raw,
                        source,
                        frames_root=frames_root,
                        check_frames=check_frames,
                    )
                )
            except RecordError as exc:
                invalid.append(
                    {
                        "data_source": source,
                        "index": index,
                        "reason": exc.reason,
                        "detail": exc.detail,
                    }
                )
    if strict and invalid:
        raise RecordError(
            "invalid_rows",
            "; ".join(
                "%s[%d]=%s" % (item["data_source"], item["index"], item["reason"])
                for item in invalid[:10]
            ),
        )

    normalized.sort(key=row_sort_key)

    # Stage 1: frozen-evaluation seeds are never trained on (legacy base exempt,
    # matching the V14 builder where the old oracle_v11/base rows kept their
    # seeds).
    seed_kept = [
        record
        for record in normalized
        if not (record["seed"] in blocked and record["data_source"] not in exempt)
    ]
    stage_seed = _stage(
        seed_kept,
        order,
        removed=len(normalized) - len(seed_kept),
        removed_by_source=_diff_counts(_counts(seed_kept, order), _counts(normalized, order)),
    )

    # Stage 2: frozen test exclusion (no-op when no frozen file was supplied).
    frozen_removed = Counter()
    if frozen:
        frozen_kept = []
        for record in seed_kept:
            reason = frozen_conflict(record, frozen)
            if reason:
                frozen_removed[reason] += 1
            else:
                frozen_kept.append(record)
    else:
        frozen_kept = seed_kept
    stage_frozen = _stage(
        frozen_kept,
        order,
        removed=len(seed_kept) - len(frozen_kept),
        removed_by_source=_diff_counts(_counts(frozen_kept, order), _counts(seed_kept, order)),
        removed_by_reason=dict(sorted(frozen_removed.items())),
    )

    # Stage 3: content dedup, keep-first over the deterministic sort order.
    seen = {}
    deduped = []
    for record in sorted(frozen_kept, key=row_sort_key):
        key = dedup_key(record)
        if key in seen:
            continue
        seen[key] = record["episode_id"]
        deduped.append(record)
    dedup_hash = hashlib.sha256(
        "\n".join(
            json.dumps(list(key), ensure_ascii=False) for key in sorted(seen)
        ).encode("utf-8")
    ).hexdigest()
    stage_dedup = _stage(
        deduped,
        order,
        removed=len(frozen_kept) - len(deduped),
        removed_by_source=_diff_counts(_counts(deduped, order), _counts(frozen_kept, order)),
        duplicate_keys=len(seen),
        key_fields=list(DEDUP_KEY_FIELDS),
    )

    # Stage 4: task balancing with the easy cap; HARD_TASKS keep every row.
    by_task = defaultdict(list)
    for record in deduped:
        by_task[record["task"]].append(record)
    balanced = []
    per_task = {}
    for task in sorted(by_task):
        rows = sorted(by_task[task], key=row_sort_key)
        cap = None if task in hard else easy_cap
        if cap is not None and len(rows) > cap:
            rng = random.Random((split_seed + zlib.crc32(task.encode("utf-8"))) & 0xFFFFFFFF)
            selected = sorted(rng.sample(rows, cap), key=row_sort_key)
        else:
            selected = rows
        balanced.extend(selected)
        per_task[task] = {
            "before": len(rows),
            "after": len(selected),
            "capped_by": "easy_cap" if len(selected) < len(rows) else None,
            "hard_task": task in hard,
        }
    balanced.sort(key=row_sort_key)
    stage_balance = _stage(
        balanced,
        order,
        removed=len(deduped) - len(balanced),
        removed_by_source=_diff_counts(_counts(balanced, order), _counts(deduped, order)),
        easy_cap=easy_cap,
        per_task=per_task,
    )

    # Stage 5: deterministic split. Row-level to reproduce the published
    # 2552/231 counts; the packaging step (prepare_gui_sft_data.py) re-splits by
    # episode so steps of one episode never straddle train/validation.
    shuffled = list(balanced)
    random.Random(split_seed).shuffle(shuffled)
    n_val = 0
    if val_divisor and val_divisor > 0:
        n_val = max(min_val, len(shuffled) // val_divisor)
    n_val = min(n_val, len(shuffled))
    validation = [dict(record, split="validation") for record in shuffled[:n_val]]
    train = [dict(record, split="train") for record in shuffled[n_val:]]
    train.sort(key=row_sort_key)
    validation.sort(key=row_sort_key)

    test_rows = []
    test_errors = []
    if frozen and frozen.get("records"):
        for raw in frozen["records"]:
            try:
                source = raw.get("data_source") if isinstance(raw, dict) else None
                if source not in DATA_SOURCES:
                    source = "v5_old"
                test_rows.append(
                    normalize_row(
                        raw,
                        source,
                        frames_root=frames_root,
                        check_frames=False,
                        split="validation",
                    )
                )
            except RecordError as exc:
                test_errors.append({"reason": exc.reason, "detail": exc.detail})
        test_rows.sort(key=row_sort_key)

    source_paths = source_paths or {}
    source_meta = {}
    for source in order:
        final_counts = _counts(train + validation + test_rows, order)
        source_meta[source] = {
            "input": source_paths.get(source),
            "raw_rows": raw_by_source.get(source, 0),
            "final_rows": final_counts.get(source, 0),
        }

    manifest = {
        "schema_version": "1.0",
        "generator": "scripts/build_distill_dataset.py",
        "canonical_schema": "data/schemas/record.schema.json",
        "description": (
            "Merged GUI SFT distillation dataset. Counts are unique rows, not "
            "training rows: only train.jsonl is trained on."
        ),
        "counts_are_unique_rows_not_training_rows": True,
        "params": {
            "easy_cap": easy_cap,
            "hard_tasks": sorted(hard),
            "split_seed": split_seed,
            "val_divisor": val_divisor,
            "min_val": min_val,
            "blocked_seed_count": len(blocked),
            "seed_block_exempt_sources": sorted(exempt),
            "dedup_key_fields": list(DEDUP_KEY_FIELDS),
        },
        "stages": {
            "raw": {
                "total": sum(raw_by_source.values()),
                "by_source": {source: raw_by_source.get(source, 0) for source in order},
                "invalid_rows": len(invalid),
            },
            "valid": _stage(normalized, order),
            "after_seed_block": stage_seed,
            "after_frozen_test": stage_frozen,
            "after_dedup": stage_dedup,
            "after_balance": stage_balance,
        },
        "unique_rows": len(deduped),
        "unique_rows_note": (
            "Unique rows after content dedup. 2915 in the published V14 run; it is "
            "NOT the number of rows trained on."
        ),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test_rows),
        "trained_rows": len(train),
        "dedup": {
            "hash": dedup_hash,
            "key_fields": list(DEDUP_KEY_FIELDS),
            "unique_keys": len(seen),
            "duplicates_removed": len(frozen_kept) - len(deduped),
        },
        "split": {
            "seed": split_seed,
            "strategy": "row-level shuffle; n_val = max(min_val, len // val_divisor)",
            "train": len(train),
            "validation": len(validation),
        },
        "sources": source_meta,
        "frozen_test": {
            "supplied": bool(frozen),
            "task_level": bool(frozen and frozen.get("task_level")),
            "tasks": sorted(frozen["tasks"]) if frozen else [],
            "episodes": sorted(frozen["episodes"]) if frozen else [],
            "seeds": sorted(frozen["seeds"]) if frozen else [],
            "excluded": dict(sorted(frozen_removed.items())),
            "errors": test_errors,
            "file": _display_path(frozen["path"], workspace_root) if frozen else None,
        },
        "invalid_rows": invalid[:50],
        "invalid_row_count": len(invalid),
    }
    return {
        "train": train,
        "validation": validation,
        "test": test_rows,
        "manifest": manifest,
    }


def _display_path(path, workspace_root):
    if not path:
        return None
    path = Path(path)
    if workspace_root:
        try:
            return path.resolve().relative_to(Path(workspace_root).resolve()).as_posix()
        except (ValueError, OSError):
            pass
    return path.name


def flatten_episode(episode):
    """Flatten an ``episodes/*.json`` document into row dicts with defaults."""
    if not isinstance(episode, dict):
        return []
    rows = episode.get("rows")
    if not isinstance(rows, list):
        return []
    defaults = {
        "task": episode.get("task") or episode.get("task_id"),
        "seed": episode.get("seed"),
        "instruction": episode.get("instruction") or episode.get("goal"),
        "episode_id": episode.get("episode_id"),
        "success": episode.get("success"),
        "reward": episode.get("reward", episode.get("env_reward")),
    }
    flattened = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        merged = dict(row)
        for key, value in defaults.items():
            if merged.get(key) is None and value is not None:
                merged[key] = value
        flattened.append(merged)
    return flattened


def load_source_rows(path, data_source):
    """Load a JSONL file or an oracle/collector output directory."""
    path = Path(path)
    if path.is_dir():
        episodes = sorted(path.glob("episodes/*.json"))
        if not episodes:
            episodes = sorted(p for p in path.glob("*.json") if p.is_file())
        rows = []
        for episode_path in episodes:
            with open(episode_path, "r", encoding="utf-8") as handle:
                rows.extend(flatten_episode(json.load(handle)))
        return rows, [p.as_posix() for p in episodes]
    return load_jsonl(path), [path.as_posix()]


def write_dataset(result, out_dir, extra_manifest=None):
    """Write train/validation/test JSONL plus manifest.json; return the manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in (
        ("train.jsonl", result["train"]),
        ("validation.jsonl", result["validation"]),
    ):
        path = write_jsonl(out_dir / name, rows)
        files[name] = {"rows": len(rows), "sha256": sha256_file(path)}
    if result.get("test"):
        path = write_jsonl(out_dir / "test.jsonl", result["test"])
        files["test.jsonl"] = {"rows": len(result["test"]), "sha256": sha256_file(path)}
    manifest = dict(result["manifest"])
    if extra_manifest:
        manifest.update(extra_manifest)
    manifest["files"] = files
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def _read_config(config_path):
    try:
        from configs import load_config
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("cannot import configs.load_config: %s" % exc)
    return load_config(config_path)


def _resolve(value, root):
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return Path(root) / path


def _parse_seeds(value):
    if value is None or str(value).lower() in ("default", "v14", "blocked"):
        return set(BLOCKED_SEEDS)
    if str(value).lower() in ("none", "0", ""):
        return set()
    seeds = set()
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            start, end = part.split("-", 1)
            seeds.update(range(int(start), int(end) + 1))
        else:
            seeds.add(int(part))
    return seeds


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Merge V11 teacher, V14 oracle and GUI-Owl data into one SFT dataset."
    )
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--v11-teacher", default=None, help="kept.jsonl from filter_teacher.py")
    parser.add_argument("--v14-oracle", default=None, help="collector output dir or JSONL")
    parser.add_argument("--gui-owl", default=None, help="GUI-Owl converted JSONL")
    parser.add_argument("--frames-root", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--frozen-test", default=None)
    parser.add_argument("--easy-cap", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--val-divisor", type=int, default=None)
    parser.add_argument("--min-val", type=int, default=None)
    parser.add_argument("--blocked-seeds", default=None, help="'default', 'none' or a comma/range list")
    parser.add_argument("--block-legacy", action="store_true", help="also block v5_old seeds")
    parser.add_argument("--no-check-frames", action="store_false", dest="check_frames")
    parser.set_defaults(check_frames=True, strict=False)
    parser.add_argument("--strict", action="store_true", help="fail on the first invalid row")
    args = parser.parse_args(argv)

    config = _read_config(args.config)
    distill = (config or {}).get("distill") or {}
    paths_cfg = (config or {}).get("paths") or {}
    root_value = args.workspace_root or paths_cfg.get("workspace_root")
    root = Path(root_value).resolve() if root_value else Path.cwd()
    sources_cfg = distill.get("sources") or {}

    source_args = {
        "v11_teacher": args.v11_teacher or sources_cfg.get("v11_teacher"),
        "v14_oracle": args.v14_oracle or sources_cfg.get("v14_oracle"),
        "gui_owl": args.gui_owl or sources_cfg.get("gui_owl"),
    }
    sources = {}
    source_paths = {}
    for source, value in source_args.items():
        if not value:
            continue
        path = _resolve(value, root)
        rows, episode_files = load_source_rows(path, source)
        sources[source] = rows
        source_paths[source] = _display_path(path, root)
        if path.is_dir():
            source_paths[source] = {
                "directory": _display_path(path, root),
                "episodes": len(episode_files),
            }
    if not sources:
        parser.error(
            "no sources given: pass --v11-teacher/--v14-oracle/--gui-owl "
            "or set distill.sources in the config"
        )

    frames_value = args.frames_root or distill.get("frames_root")
    frames_root = _resolve(frames_value, root) if frames_value else root
    easy_cap = args.easy_cap if args.easy_cap is not None else int(distill.get("easy_cap", 1200))
    split_seed = (
        args.split_seed if args.split_seed is not None else int(distill.get("split_seed", 14))
    )
    val_divisor = (
        args.val_divisor
        if args.val_divisor is not None
        else int(distill.get("val_divisor", 12))
    )
    min_val = args.min_val if args.min_val is not None else int(distill.get("min_val", 0))
    blocked = _parse_seeds(args.blocked_seeds) if args.blocked_seeds is not None else set(BLOCKED_SEEDS)
    exempt = () if args.block_legacy else ("v5_old",)
    frozen_value = args.frozen_test or distill.get("frozen_test")
    frozen = load_frozen_test(_resolve(frozen_value, root)) if frozen_value else None

    out_dir = _resolve(args.out, root) if args.out else root / "datasets" / "gui_distill_v14"

    result = build_dataset(
        sources,
        source_paths=source_paths,
        frames_root=frames_root,
        check_frames=args.check_frames,
        blocked_seeds=blocked,
        seed_block_exempt=exempt,
        easy_cap=easy_cap,
        split_seed=split_seed,
        val_divisor=val_divisor,
        min_val=min_val,
        frozen=frozen,
        strict=args.strict,
        workspace_root=root,
    )
    manifest = write_dataset(
        result,
        out_dir,
        extra_manifest={
            "workspace_root_display": ".",
            "frames_root": _display_path(frames_root, root),
        },
    )
    print(
        json.dumps(
            {
                "out": str(out_dir),
                "raw": manifest["stages"]["raw"]["total"],
                "after_seed_block": manifest["stages"]["after_seed_block"]["total"],
                "after_dedup": manifest["stages"]["after_dedup"]["total"],
                "after_balance": manifest["stages"]["after_balance"]["total"],
                "unique_rows": manifest["unique_rows"],
                "train_rows": manifest["train_rows"],
                "validation_rows": manifest["validation_rows"],
                "files": manifest["files"],
                "counts_are_unique_rows_not_training_rows": True,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
