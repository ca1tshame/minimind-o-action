"""Package filtered GUI records into the canonical SFT JSONL splits.

Reads records produced by ``scripts/filter_teacher.py`` (or any canonical
record file), re-validates every row, removes content duplicates, splits whole
*episodes* into train/validation so steps of one episode never straddle the
boundary, and keeps a frozen test set completely out of both splits.

Output (under ``--out``)::

    sft.jsonl          all shipped rows (= train + validation)
    train.jsonl
    validation.jsonl
    test.jsonl         only when a frozen test file was supplied
    manifest.json      counts, frozen-test keys, file SHA-256 values

Frozen test manifests are JSON objects with ``tasks``/``seeds``/``episodes``
(whole-task exclusions only when ``tasks`` is non-empty), or plain JSONL record
lists, which are treated as episode/seed-level exclusions.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.build_distill_dataset import (  # noqa: E402
    RecordError,
    check_record,
    dedup_key,
    frozen_conflict,
    load_frozen_test,
    normalize_row,
    row_sort_key,
)
from scripts.filter_teacher import (  # noqa: E402
    DATA_SOURCES,
    load_jsonl,
    sha256_file,
    write_jsonl,
)


class IsolationError(ValueError):
    """A frozen-test row leaked (or would leak) into train/validation."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__(
            "frozen test isolation violated: " + "; ".join(self.problems[:5])
        )


def load_records(paths):
    """Load records from JSONL files and/or directories containing them."""
    records = []
    files = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            candidates = sorted(path.glob("*.jsonl"))
            candidates = [p for p in candidates if p.name != "rejected.jsonl"]
            if not candidates:
                raise FileNotFoundError("no *.jsonl records under %s" % path)
            for candidate in candidates:
                records.extend(load_jsonl(candidate))
                files.append(candidate)
        else:
            records.extend(load_jsonl(path))
            files.append(path)
    return records, files


def _normalize_all(records, frames_root=None, check_frames=False):
    normalized = []
    errors = []
    for index, raw in enumerate(records):
        source = raw.get("data_source") if isinstance(raw, dict) else None
        if source not in DATA_SOURCES:
            errors.append({"index": index, "reason": "data_source_invalid"})
            continue
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
            errors.append({"index": index, "reason": exc.reason, "detail": exc.detail})
    return normalized, errors


def dedup_records(records):
    """Keep the first row for each content key over a deterministic order."""
    unique = []
    seen = set()
    duplicates = 0
    for record in sorted(records, key=row_sort_key):
        key = dedup_key(record)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(record)
    return unique, duplicates


def split_by_episode(records, seed=14, val_divisor=12, min_val=0):
    """Assign whole episodes to validation until the row target is reached."""
    by_episode = defaultdict(list)
    for record in records:
        by_episode[record["episode_id"]].append(record)
    episode_ids = sorted(by_episode)
    random.Random(seed).shuffle(episode_ids)
    target = 0
    if val_divisor and val_divisor > 0:
        target = max(int(min_val), len(records) // int(val_divisor))
    train = []
    validation = []
    for episode_id in episode_ids:
        rows = sorted(by_episode[episode_id], key=row_sort_key)
        if len(validation) < target:
            validation.extend(rows)
        else:
            train.extend(rows)
    train.sort(key=row_sort_key)
    validation.sort(key=row_sort_key)
    return train, validation


def assert_isolated(train, validation, test, frozen_tasks=(), frozen_seeds=None):
    """Raise :class:`IsolationError` if a frozen test row can reach train/val."""
    problems = []
    test_episodes = {record["episode_id"] for record in test}
    test_seeds = set(frozen_seeds or ())
    test_seeds.update(record["seed"] for record in test)
    declared_tasks = set(frozen_tasks or ())
    for split_name, rows in (("train", train), ("validation", validation)):
        for record in rows:
            if record["episode_id"] in test_episodes:
                problems.append(
                    "%s episode %s is a frozen test episode"
                    % (split_name, record["episode_id"])
                )
            elif record["seed"] in test_seeds:
                problems.append(
                    "%s row %s/%s uses frozen test seed %s"
                    % (
                        split_name,
                        record["task"],
                        record["episode_id"],
                        record["seed"],
                    )
                )
            elif declared_tasks and record["task"] in declared_tasks:
                problems.append(
                    "%s row uses frozen test task %s" % (split_name, record["task"])
                )
    if problems:
        raise IsolationError(problems)
    return True


def package(
    records,
    frozen=None,
    split_seed=14,
    val_divisor=12,
    min_val=0,
    keep_split=False,
    frames_root=None,
    check_frames=False,
    source_paths=None,
    strict=True,
):
    """Validate, dedup, split and isolation-check canonical records."""
    normalized, errors = _normalize_all(records, frames_root, check_frames)
    if errors and strict:
        raise RecordError(
            "invalid_rows",
            "; ".join(
                "record[%s]=%s" % (item["index"], item["reason"]) for item in errors[:10]
            ),
        )
    unique, duplicates = dedup_records(normalized)

    frozen_tasks = set((frozen or {}).get("tasks") or ())
    frozen_seeds = set((frozen or {}).get("seeds") or ()) if frozen else set()

    kept = []
    excluded = Counter()
    for record in unique:
        reason = frozen_conflict(record, frozen)
        if reason:
            excluded[reason] += 1
            continue
        kept.append(record)

    test = []
    test_errors = []
    if frozen and frozen.get("records"):
        test, test_errors = _normalize_all(
            frozen["records"], frames_root, check_frames=False
        )
        for record in test:
            record["split"] = "validation"
        if test_errors and strict:
            raise RecordError(
                "frozen_test_invalid",
                "; ".join(
                    "record[%s]=%s" % (item["index"], item["reason"])
                    for item in test_errors[:10]
                ),
            )

    if keep_split:
        train = [r for r in kept if r["split"] == "train"]
        validation = [r for r in kept if r["split"] == "validation"]
    else:
        train, validation = split_by_episode(
            kept, seed=split_seed, val_divisor=val_divisor, min_val=min_val
        )
        # The file a record ships in is authoritative: relabel so every row's
        # "split" agrees with the split file it lands in.
        for record in train:
            record["split"] = "train"
        for record in validation:
            record["split"] = "validation"

    assert_isolated(train, validation, test, frozen_tasks, frozen_seeds)

    shipped = sorted(train + validation, key=row_sort_key)
    sources = Counter(record["data_source"] for record in shipped)
    manifest = {
        "schema_version": "1.0",
        "generator": "scripts/prepare_gui_sft_data.py",
        "canonical_schema": "data/schemas/record.schema.json",
        "description": (
            "Canonical GUI SFT data. Counts are unique rows; only train.jsonl is "
            "trained on and the frozen test set is never in either split."
        ),
        "counts_are_unique_rows_not_training_rows": True,
        "counts": {
            "input_rows": len(records),
            "valid_rows": len(normalized),
            "invalid_rows": len(errors),
            "after_dedup": len(unique),
            "duplicates_removed": duplicates,
            "excluded_frozen": dict(sorted(excluded.items())),
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "shipped": len(shipped),
        },
        "split": {
            "method": "keep-split" if keep_split else "episode-level deterministic shuffle",
            "seed": split_seed,
            "val_divisor": val_divisor,
            "min_val": min_val,
            "validation_target_rows": max(int(min_val), len(kept) // int(val_divisor))
            if val_divisor
            else 0,
        },
        "isolation": {
            "checked": True,
            "frozen_tasks": sorted(frozen_tasks),
            "frozen_seeds": sorted(frozen_seeds),
            "frozen_episodes": len((frozen or {}).get("episodes") or ()),
        },
        "sources": {source: sources.get(source, 0) for source in sorted(sources)},
        "frozen_test": {
            "supplied": bool(frozen),
            "task_level": bool(frozen and frozen.get("task_level")),
            "tasks": sorted(frozen_tasks),
            "excluded": dict(sorted(excluded.items())),
            "errors": test_errors,
        },
        "input_files": list(source_paths or []),
        "invalid_rows": errors[:50],
    }
    return {
        "train": train,
        "validation": validation,
        "test": test,
        "all": shipped,
        "manifest": manifest,
    }


def write_outputs(result, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in (
        ("sft.jsonl", result["all"]),
        ("train.jsonl", result["train"]),
        ("validation.jsonl", result["validation"]),
    ):
        path = write_jsonl(out_dir / name, rows)
        files[name] = {"rows": len(rows), "sha256": sha256_file(path)}
    if result["test"]:
        path = write_jsonl(out_dir / "test.jsonl", result["test"])
        files["test.jsonl"] = {"rows": len(result["test"]), "sha256": sha256_file(path)}
    manifest = dict(result["manifest"])
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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Package filtered GUI records into canonical SFT splits."
    )
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--records", nargs="+", default=None, help="record JSONL files/dirs")
    parser.add_argument("--frozen-test", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--val-divisor", type=int, default=None)
    parser.add_argument("--min-val", type=int, default=None)
    parser.add_argument("--keep-split", action="store_true", help="use the split field on each record")
    parser.add_argument("--frames-root", default=None)
    parser.add_argument("--check-frames", action="store_true", help="require frame files on disk")
    args = parser.parse_args(argv)

    config = _read_config(args.config)
    distill = (config or {}).get("distill") or {}
    paths_cfg = (config or {}).get("paths") or {}
    root_value = args.workspace_root or paths_cfg.get("workspace_root")
    root = Path(root_value).resolve() if root_value else Path.cwd()

    record_values = args.records or distill.get("sft_records")
    if isinstance(record_values, str):
        record_values = [record_values]
    if not record_values:
        parser.error("--records is required (or set distill.sft_records in the config)")
    paths = [_resolve(value, root) for value in record_values]
    records, input_files = load_records(paths)

    frozen_value = args.frozen_test or distill.get("frozen_test")
    frozen = load_frozen_test(_resolve(frozen_value, root)) if frozen_value else None

    frames_value = args.frames_root or distill.get("frames_root")
    frames_root = _resolve(frames_value, root) if frames_value else root
    split_seed = (
        args.split_seed if args.split_seed is not None else int(distill.get("split_seed", 14))
    )
    val_divisor = (
        args.val_divisor
        if args.val_divisor is not None
        else int(distill.get("val_divisor", 12))
    )
    min_val = args.min_val if args.min_val is not None else int(distill.get("min_val", 0))
    out_dir = _resolve(args.out, root) if args.out else root / "datasets" / "gui_sft"

    result = package(
        records,
        frozen=frozen,
        split_seed=split_seed,
        val_divisor=val_divisor,
        min_val=min_val,
        keep_split=args.keep_split,
        frames_root=frames_root,
        check_frames=args.check_frames,
        source_paths=[str(path.name) for path in input_files],
    )
    manifest = write_outputs(result, out_dir)
    print(
        json.dumps(
            {
                "out": str(out_dir),
                **manifest["counts"],
                "files": manifest["files"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
