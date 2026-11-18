"""Quality-filter teacher GUI trajectories into canonical SFT rows.

Port of ``project/gui_distillation_v11/filter_teacher.py`` from the research
workspace, with every machine path removed. Inputs are resolved from
``--workspace-root`` / ``--config`` or from explicit CLI arguments.

A row is kept only when all of the following hold:

* its episode is environment-verified successful (``success`` flag, and when
  present the ``status``/``reward`` cross-checks agree);
* ``target_action`` parses as the constrained action DSL: ``click(x,y)`` with
  finite coordinates in ``[0, 1]``, ``type("...")`` / ``type(@start:end[|lower|
  upper])`` resolved against the instruction, ``scroll(dx,dy)`` with finite
  deltas in ``[-1, 1]`` and a boolean ``inside`` argument, ``stop(done)``;
* the referenced screenshot exists under ``--frames-root`` and the recorded
  SHA-256 matches the bytes on disk (a hash is computed when absent);
* the click is not an exact repeat of an earlier click in the same episode;
* at most one ``stop`` per episode and only as the last step;
* optional per-task cap (``--max-per-task``) on the kept rows.

Rejected rows go to ``rejected.jsonl`` with a machine-readable ``reason``.
Kept rows are canonical records, see ``data/schemas/record.schema.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

CANONICAL_FIELDS = (
    "episode_id",
    "step",
    "task",
    "seed",
    "instruction",
    "frame",
    "frame_sha256",
    "history",
    "error_feedback",
    "target_action",
    "action_kind",
    "data_source",
    "split",
    "reward",
    "success",
    "target_point",
)

ACTION_KINDS = ("click", "type", "scroll", "stop")
DATA_SOURCES = ("v11_teacher", "v14_oracle", "gui_owl", "v5_old")
SPLITS = ("train", "validation")

HEX64 = re.compile(r"^[0-9a-f]{64}$")
CLICK_RE = re.compile(
    r"^<action>click\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)</action>$"
)
SCROLL_RE = re.compile(
    r"^<action>scroll\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*(?:,\s*(True|False|true|false)\s*)?\)</action>$"
)
STOP_RE = re.compile(r"^<action>stop\(\s*([A-Za-z_]+)\s*\)</action>$")
TYPE_JSON_RE = re.compile(r'^<action>type\(\s*(".*")\s*\)</action>$', re.DOTALL)
TYPE_POINTER_RE = re.compile(r"^<action>type\(@(\d+):(\d+)(?:\|(lower|upper))?\)</action>$")
COORD_TOKEN_RE = re.compile(r"[^0-9.,\-+eE\s]")


class RejectRow(Exception):
    """Row-level rejection carrying a stable machine-readable reason."""

    def __init__(self, reason, detail=""):
        message = reason if not detail else "%s: %s" % (reason, detail)
        super().__init__(message)
        self.reason = reason
        self.detail = detail


ActionDSLError = RejectRow


def _finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _normalized_argument(value, reason, detail):
    """Return ``float(value)`` or reject non-finite / non-numeric text."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise RejectRow(reason, detail)
    if not math.isfinite(number):
        raise RejectRow(reason, detail)
    return number


def parse_action_dsl(target_action, instruction=""):
    """Parse one ``<action>...</action>`` string.

    Returns ``(kind, payload)``. ``payload`` is ``{"x", "y"}`` for click,
    ``{"text"}`` for type (pointer references are resolved against
    ``instruction``), ``{"dx", "dy", "inside"}`` for scroll and ``{"status"}``
    for stop. Raises :class:`RejectRow` with reason ``dsl_syntax``,
    ``dsl_coordinate``, ``pointer_unresolved`` or ``type_not_string``.
    """
    if not isinstance(target_action, str):
        raise RejectRow("target_action_not_string", type(target_action).__name__)
    text = target_action.strip()
    if not text.startswith("<action>") or not text.endswith("</action>"):
        raise RejectRow("dsl_syntax", "missing <action> wrapper: %r" % text[:80])
    if len(text) > 512:
        raise RejectRow("dsl_syntax", "action longer than 512 chars")

    match = CLICK_RE.match(text)
    if match:
        x = _normalized_argument(match.group(1), "dsl_coordinate", text)
        y = _normalized_argument(match.group(2), "dsl_coordinate", text)
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise RejectRow("dsl_coordinate", "click outside [0,1]: %r" % text)
        return "click", {"x": x, "y": y}

    match = SCROLL_RE.match(text)
    if match:
        dx = _normalized_argument(match.group(1), "dsl_coordinate", text)
        dy = _normalized_argument(match.group(2), "dsl_coordinate", text)
        if not (-1.0 <= dx <= 1.0 and -1.0 <= dy <= 1.0):
            raise RejectRow("dsl_coordinate", "scroll outside [-1,1]: %r" % text)
        return "scroll", {
            "dx": dx,
            "dy": dy,
            "inside": (match.group(3) or "true").lower() == "true",
        }

    match = STOP_RE.match(text)
    if match:
        status = match.group(1)
        if status.lower() not in ("done", "fail", "failed", "success"):
            raise RejectRow("dsl_syntax", "unknown stop status: %r" % status)
        return "stop", {"status": status}

    match = TYPE_POINTER_RE.match(text)
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        transform = match.group(3)
        if end <= start:
            raise RejectRow("dsl_syntax", "empty pointer range: %r" % text)
        if end > len(instruction):
            raise RejectRow("pointer_unresolved", "range past instruction: %r" % text)
        segment = instruction[start:end]
        if transform == "lower":
            segment = segment.lower()
        elif transform == "upper":
            segment = segment.upper()
        if not segment.strip():
            raise RejectRow("pointer_unresolved", "empty pointer text: %r" % text)
        return "type", {"text": segment}

    match = TYPE_JSON_RE.match(text)
    if match:
        literal = match.group(1)
        try:
            value = json.loads(literal)
        except (ValueError, TypeError):
            raise RejectRow("dsl_syntax", "bad JSON string: %r" % text[:120])
        if not isinstance(value, str):
            raise RejectRow("type_not_string", "type() payload is not a string")
        if not value:
            raise RejectRow("dsl_syntax", "empty type text")
        return "type", {"text": value}

    if COORD_TOKEN_RE.search(text):
        raise RejectRow("dsl_syntax", "unknown action form: %r" % text[:120])
    raise RejectRow("dsl_syntax", "not a supported action: %r" % text[:120])


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise ValueError("%s:%d: invalid JSON: %s" % (path, line_number, exc))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False))
            handle.write("\n")
    return path


def resolve_frame(value, frames_root):
    """Resolve ``value`` under ``frames_root``; return ``(relative_posix, path)``."""
    if not isinstance(value, str) or not value.strip():
        raise RejectRow("missing_frame")
    raw = value.strip().replace("\\", "/")
    if raw.startswith("/") or re.match(r"^[A-Za-z]:/", raw) or raw.startswith("//"):
        raise RejectRow("frame_absolute", raw)
    root = Path(frames_root).resolve()
    candidate = (root / raw).resolve()
    if candidate != root and root not in candidate.parents:
        raise RejectRow("frame_outside_frames_root", raw)
    if not candidate.is_file():
        raise RejectRow("missing_frame", raw)
    return candidate.relative_to(root).as_posix(), candidate


def verify_episode_success(episode):
    """Return ``(ok, reason)`` using only recorded environment evidence."""
    if episode.get("success") is not True:
        return False, "ep_failed"
    if episode.get("verified") is False:
        return False, "ep_not_environment_verified"
    status = episode.get("status")
    if status is not None and str(status).strip().lower() != "success":
        return False, "ep_status_not_success"
    reward = episode.get("reward", episode.get("env_reward"))
    if reward is not None:
        if not _finite(reward):
            return False, "ep_reward_invalid"
        if float(reward) <= 0.0:
            return False, "ep_reward_nonpositive"
    return True, ""


def _reject(episode_id, task, seed, step, reason, detail=""):
    return {
        "episode_id": episode_id,
        "task": task,
        "seed": seed,
        "step": step,
        "reason": reason,
        "detail": str(detail)[:300],
    }


def _seed_key(seed):
    if isinstance(seed, bool):
        return (1, 0.0)
    if isinstance(seed, (int, float)):
        return (0, float(seed))
    return (2, 0.0)


def _episode_sort_key(episode, fallback):
    task = episode.get("task") or episode.get("task_id") or ""
    seed = episode.get("seed", 0)
    episode_id = episode.get("episode_id") or fallback
    return (str(task), _seed_key(seed), str(episode_id))


def _episode_files(episodes_dir):
    directory = Path(episodes_dir)
    if not directory.is_dir():
        raise FileNotFoundError("episodes directory not found: %s" % directory)
    files = sorted(directory.glob("ep_*.json"))
    if not files:
        files = sorted(p for p in directory.glob("*.json") if p.is_file())
    return files


def filter_directory(
    episodes_dir,
    frames_root,
    out_dir=None,
    teacher="GUI-Owl-1.5-8B",
    max_per_task=0,
    data_source="v11_teacher",
    split="train",
    workspace_root=None,
):
    """Filter ``episodes_dir/*.json`` and return kept/rejected rows plus stats.

    ``frames_root`` is the single root the canonical ``frame`` field is relative
    to (the same root ``scripts/train_gui_sft.py`` later reads frames from).
    """
    if data_source not in DATA_SOURCES:
        raise ValueError("unknown data_source: %r" % (data_source,))
    if split not in SPLITS:
        raise ValueError("unknown split: %r" % (split,))
    episodes_dir = Path(episodes_dir)
    frames_root = Path(frames_root)
    if not frames_root.is_dir():
        raise FileNotFoundError("frames root not found: %s" % frames_root)

    kept = []
    rejected = []
    stats = Counter()
    per_task_kept = Counter()
    per_task_rejected = Counter()
    episode_docs = []
    for path in _episode_files(episodes_dir):
        with open(path, "r", encoding="utf-8") as handle:
            doc = json.load(handle)
        if not isinstance(doc, dict):
            raise ValueError("episode file is not a JSON object: %s" % path)
        episode_docs.append((path, doc))

    episode_docs.sort(key=lambda item: _episode_sort_key(item[1], item[0].stem))

    for path, episode in episode_docs:
        stats["episodes"] += 1
        task = str(episode.get("task") or episode.get("task_id") or "")
        seed = episode.get("seed", 0)
        episode_id = str(
            episode.get("episode_id")
            or ("%s_%s" % (task, seed) if task else path.stem)
        )
        ok, reason = verify_episode_success(episode)
        if not ok:
            stats["episodes_rejected"] += 1
            rejected.append(
                _reject(episode_id, task, seed, None, reason, "episode=%s" % path.name)
            )
            continue
        stats["episodes_success"] += 1

        if not task:
            stats["episodes_rejected"] += 1
            rejected.append(_reject(episode_id, task, seed, None, "ep_task_missing"))
            continue
        if isinstance(seed, bool) or not isinstance(seed, int):
            stats["episodes_rejected"] += 1
            rejected.append(
                _reject(episode_id, task, seed, None, "ep_seed_invalid", repr(seed))
            )
            continue
        instruction = str(episode.get("instruction") or episode.get("goal") or "")
        if not instruction:
            stats["episodes_rejected"] += 1
            rejected.append(
                _reject(episode_id, task, seed, None, "ep_instruction_missing")
            )
            continue

        rows = episode.get("rows")
        if not isinstance(rows, list) or not rows:
            stats["episodes_rejected"] += 1
            rejected.append(_reject(episode_id, task, seed, None, "ep_rows_missing"))
            continue

        seen_clicks = set()
        stop_seen = False
        next_history = [
            str(item)
            for item in (episode.get("history") or [])
            if isinstance(item, str)
        ]

        for index, row in enumerate(rows):
            stats["rows_seen"] += 1
            if not isinstance(row, dict):
                rejected.append(
                    _reject(episode_id, task, seed, None, "row_not_object")
                )
                continue
            step = row.get("step", index)
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                rejected.append(
                    _reject(episode_id, task, seed, None, "step_invalid", repr(step))
                )
                continue
            target = row.get("target_action")
            if target is None:
                target = row.get("dsl")
            try:
                if not isinstance(target, str) or not target.strip():
                    raise RejectRow("missing_target_action")
                kind, payload = parse_action_dsl(target, instruction)
                if kind == "stop":
                    if stop_seen:
                        raise RejectRow("repeat_stop", target)
                    if index != len(rows) - 1:
                        raise RejectRow("stop_not_terminal", target)
                    stop_seen = True
                elif stop_seen:
                    raise RejectRow("action_after_stop", target)

                reward = row.get("reward")
                if reward is None:
                    reward = episode.get("reward", episode.get("env_reward"))
                if reward is not None:
                    if not _finite(reward):
                        raise RejectRow("reward_invalid", repr(reward))
                    reward = float(reward)

                history = row.get("history")
                if history is None:
                    history = list(next_history)
                if not isinstance(history, list) or any(
                    not isinstance(item, str) for item in history
                ):
                    raise RejectRow("history_invalid")
                if len(history) > 64:
                    raise RejectRow("history_too_long", str(len(history)))

                error_feedback = row.get("error_feedback")
                if error_feedback is None:
                    error_feedback = row.get("last_action_error")
                if error_feedback is not None and not isinstance(
                    error_feedback, str
                ):
                    raise RejectRow("error_feedback_invalid")

                rel_frame, frame_path = resolve_frame(
                    row.get("frame") if row.get("frame") is not None else row.get("frame_path"),
                    frames_root,
                )
                digest = sha256_file(frame_path)
                recorded_hash = row.get("frame_sha256")
                if recorded_hash is not None:
                    if not isinstance(recorded_hash, str) or not HEX64.match(
                        recorded_hash.lower()
                    ):
                        raise RejectRow("frame_hash_invalid", repr(recorded_hash))
                    if recorded_hash.lower() != digest:
                        raise RejectRow(
                            "frame_hash_mismatch",
                            "%s != %s" % (recorded_hash, digest),
                        )

                target_point = None
                if kind == "click":
                    target_point = [payload["x"], payload["y"]]
                    supplied = row.get("target_point")
                    if supplied is not None:
                        if (
                            not isinstance(supplied, (list, tuple))
                            or len(supplied) != 2
                            or not all(_finite(value) for value in supplied)
                        ):
                            raise RejectRow("target_point_invalid", repr(supplied))
                        if (
                            abs(float(supplied[0]) - payload["x"]) > 1e-6
                            or abs(float(supplied[1]) - payload["y"]) > 1e-6
                        ):
                            raise RejectRow(
                                "target_point_mismatch",
                                "%r != %r" % (list(supplied), target_point),
                            )

                if kind == "click":
                    click_key = (round(payload["x"], 4), round(payload["y"], 4))
                    if click_key in seen_clicks:
                        raise RejectRow("repeat_click", target)
                    seen_clicks.add(click_key)
            except RejectRow as exc:
                stats["rows_rejected"] += 1
                stats["reject_" + exc.reason] += 1
                per_task_rejected[task] += 1
                rejected.append(
                    _reject(episode_id, task, seed, step, exc.reason, exc.detail or target)
                )
                continue

            if kind == "type":
                next_history = list(history) + [target]
            elif kind in ("click", "scroll", "stop"):
                next_history = list(history) + [target]

            kept.append(
                {
                    "episode_id": episode_id,
                    "step": step,
                    "task": task,
                    "seed": seed,
                    "instruction": instruction,
                    "frame": rel_frame,
                    "frame_sha256": digest,
                    "history": list(history),
                    "error_feedback": error_feedback,
                    "target_action": target.strip(),
                    "action_kind": kind,
                    "data_source": data_source,
                    "split": split,
                    "reward": reward,
                    "success": True,
                    "target_point": target_point,
                }
            )
            stats["rows_kept"] += 1
            per_task_kept[task] += 1

    kept.sort(
        key=lambda record: (
            record["task"],
            record["seed"],
            record["episode_id"],
            record["step"],
            record["frame"],
        )
    )
    rejected.sort(
        key=lambda record: (
            str(record.get("task") or ""),
            _seed_key(record.get("seed")),
            str(record.get("episode_id") or ""),
            -1 if record.get("step") is None else int(record["step"]),
            str(record.get("reason") or ""),
        )
    )

    if max_per_task and max_per_task > 0:
        by_task = defaultdict(list)
        for record in kept:
            by_task[record["task"]].append(record)
        capped = []
        for task in sorted(by_task):
            rows = by_task[task]
            if len(rows) > max_per_task:
                capped.extend(rows[:max_per_task])
                for record in rows[max_per_task:]:
                    stats["rows_rejected"] += 1
                    stats["reject_task_cap"] += 1
                    per_task_rejected[task] += 1
                    rejected.append(
                        _reject(
                            record["episode_id"],
                            task,
                            record["seed"],
                            record["step"],
                            "task_cap",
                            "max_per_task=%d" % max_per_task,
                        )
                    )
            else:
                capped.extend(rows)
        kept = sorted(
            capped,
            key=lambda record: (
                record["task"],
                record["seed"],
                record["episode_id"],
                record["step"],
                record["frame"],
            ),
        )
        rejected.sort(
            key=lambda record: (
                str(record.get("task") or ""),
                _seed_key(record.get("seed")),
                str(record.get("episode_id") or ""),
                -1 if record.get("step") is None else int(record["step"]),
                str(record.get("reason") or ""),
            )
        )

    kinds = Counter(record["action_kind"] for record in kept)
    report = {
        "generator": "scripts/filter_teacher.py",
        "schema_version": "1.0",
        "params": {
            "teacher": teacher,
            "data_source": data_source,
            "split": split,
            "max_per_task": max_per_task,
        },
        "input": {
            "episodes_dir": _display_path(episodes_dir, workspace_root),
            "episodes": stats["episodes"],
            "frames_root": _display_path(frames_root, workspace_root),
        },
        "episodes": {
            "seen": stats["episodes"],
            "verified_success": stats["episodes_success"],
            "rejected": stats["episodes_rejected"],
        },
        "rows": {
            "seen": stats["rows_seen"],
            "kept": stats["rows_kept"],
            "rejected": stats["rows_rejected"],
        },
        "reject_reasons": {
            key[len("reject_"):]: value
            for key, value in sorted(stats.items())
            if key.startswith("reject_")
        },
        "action_kinds": {key: kinds[key] for key in sorted(kinds)},
        "per_task": {
            task: {
                "kept": per_task_kept.get(task, 0),
                "rejected": per_task_rejected.get(task, 0),
            }
            for task in sorted(set(per_task_kept) | set(per_task_rejected))
        },
        "counts_are_unique_rows_not_training_rows": True,
    }
    return {"kept": kept, "rejected": rejected, "stats": report}


def write_outputs(result, out_dir):
    """Write kept/rejected JSONL and the stats report; return ``(stats, files)``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in (
        ("kept.jsonl", result["kept"]),
        ("rejected.jsonl", result["rejected"]),
    ):
        path = write_jsonl(out_dir / name, rows)
        files[name] = {"rows": len(rows), "sha256": sha256_file(path)}
    stats = dict(result["stats"])
    stats["files"] = files
    (out_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return stats, files


def _display_path(path, workspace_root):
    path = Path(path)
    if workspace_root:
        try:
            return path.resolve().relative_to(Path(workspace_root).resolve()).as_posix()
        except (ValueError, OSError):
            pass
    return path.name


def _read_config(config_path):
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from configs import load_config
    except ImportError as exc:  # pragma: no cover - repo always ships configs/
        raise SystemExit("cannot import configs.load_config: %s" % exc)
    return load_config(config_path)


def _workspace_root(cli_value, config):
    if cli_value:
        return Path(cli_value)
    paths = (config or {}).get("paths") or {}
    configured = paths.get("workspace_root")
    if configured:
        return Path(configured)
    return Path.cwd()


def _resolve(value, root):
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return Path(root) / path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Filter teacher GUI trajectories into canonical SFT rows."
    )
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--episodes-dir", default=None)
    parser.add_argument("--frames-root", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--teacher", default=None)
    parser.add_argument("--max-per-task", type=int, default=None)
    parser.add_argument("--data-source", default="v11_teacher", choices=DATA_SOURCES)
    parser.add_argument("--split", default="train", choices=SPLITS)
    args = parser.parse_args(argv)

    config = _read_config(args.config)
    distill = (config or {}).get("distill") or {}
    root = _workspace_root(args.workspace_root, config).resolve()

    episodes_value = args.episodes_dir or distill.get("teacher_episodes")
    if not episodes_value:
        parser.error(
            "--episodes-dir is required (or set distill.teacher_episodes in the config)"
        )
    frames_value = args.frames_root or distill.get("frames_root")
    teacher = args.teacher or distill.get("teacher_model") or "GUI-Owl-1.5-8B"
    max_per_task = args.max_per_task
    if max_per_task is None:
        max_per_task = int(distill.get("max_rows_per_task") or 0)

    episodes_dir = _resolve(episodes_value, root)
    frames_root = _resolve(frames_value, root) if frames_value else root
    out_dir = _resolve(args.out, root) if args.out else root / "datasets" / "v11_teacher_filtered"

    result = filter_directory(
        episodes_dir,
        frames_root,
        teacher=teacher,
        max_per_task=max_per_task,
        data_source=args.data_source,
        split=args.split,
        workspace_root=root,
    )
    stats, files = write_outputs(result, out_dir)
    print(
        json.dumps(
            {
                "kept": len(result["kept"]),
                "rejected": len(result["rejected"]),
                "out": str(out_dir),
                "files": files,
                "counts_are_unique_rows_not_training_rows": True,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
