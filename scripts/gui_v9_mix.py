"""Mix V9 grounding episodes into GUI SFT data at a configured ratio.

The V9 ablation added a small grounding set on top of the V8 mixture instead of
replacing it. This script reproduces that merge portably:

* base SFT ``train.jsonl`` / ``validation.jsonl`` rows are kept untouched;
* grounding episodes are split into train/validation by a hash of
  ``(task, seed)`` so every step of one episode stays in one split;
* ``--ratio`` is the fraction of grounding rows in the *final* training set,
  i.e. ``ground / (base + ground)``, so ``ratio=0.1`` with 90 base rows adds
  10 grounding rows;
* ground rows are capped to that target, sorted deterministically, and their
  ``frame`` paths are rewritten relative to ``--frames-prefix``;
* a manifest records row counts, the achieved ratio and sha256 of both written
  files. No absolute paths are stored.

Frame paths are workspace-relative POSIX paths. The script refuses absolute
paths, drive letters and ``..`` segments so the manifest stays portable.
"""
import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path, PurePosixPath

FILES = {"train": "train.jsonl", "validation": "validation.jsonl"}


def read_jsonl(path):
    """Read a JSONL file into a list of dict rows."""
    rows = []
    with Path(path).open(encoding="utf8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("%s:%d is not a JSON object" % (path, number))
            rows.append(row)
    return rows


def dump_rows(rows):
    """Serialise rows to canonical JSONL bytes (sorted keys, UTF-8)."""
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf8")


def write_rows(path, rows):
    """Write canonical JSONL and return ``(sha256, row_count)``."""
    payload = dump_rows(rows)
    Path(path).write_bytes(payload)
    return hashlib.sha256(payload).hexdigest(), len(rows)


def session_key(row):
    """Return the episode key used for the train/validation split.

    An explicit ``session_key`` field wins; otherwise the key is
    ``v9__<task>_<seed>`` exactly as in the original V9 mixture.
    """
    explicit = row.get("session_key")
    if explicit:
        return str(explicit)
    task = row.get("task")
    seed = row.get("seed")
    if not task or seed is None:
        raise ValueError("grounding row needs task and seed (or session_key)")
    return "v9__%s_%s" % (task, seed)


def is_validation(key, val_percent=15, seed=0):
    """Deterministic hash split; ``val_percent=15`` matches the V8/V9 split."""
    if not 0 <= val_percent <= 100:
        raise ValueError("val_percent must be in [0, 100]")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    salt = ("v9:%d:" % seed) if seed else "v9:"
    digest = hashlib.sha256((salt + key).encode("utf8")).hexdigest()
    return int(digest[:8], 16) % 100 < val_percent


def plan_mix(n_base, ratio):
    """Rows of grounding data needed for ``ground / (base + ground) == ratio``."""
    if not 0.0 < ratio < 1.0:
        raise ValueError("ratio must be in (0, 1)")
    if n_base < 0:
        raise ValueError("n_base must be non-negative")
    if n_base == 0:
        return 0
    return int(math.floor(n_base * ratio / (1.0 - ratio) + 0.5))


def _relative(path, what):
    text = str(path).replace("\\", "/")
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts or (pure.parts and ":" in pure.parts[0]):
        raise ValueError("%s must be a workspace-relative path: %r" % (what, path))
    return pure


def join_frames(prefix, frame):
    """Join an optional frames prefix with one row's frame path."""
    rel = _relative(frame, "frame")
    if prefix is None:
        return rel.as_posix()
    base = _relative(prefix, "frames_prefix")
    return (base / rel).as_posix() if rel.parts else base.as_posix()


def _prepare(row, split, prefix):
    prepared = dict(row)
    prepared["mix_source"] = "v9_ground"
    prepared["split"] = split
    prepared["session_key"] = session_key(row)
    prepared["frame"] = join_frames(prefix, row["frame"])
    return prepared


def build_mix(base_train, base_validation, ground_rows, ratio,
              frames_prefix=None, val_percent=15, seed=0):
    """Merge grounding rows into base SFT rows (pure; no filesystem writes).

    Returns ``{"train": [...], "validation": [...], "manifest": {...}}``. The
    input lists are not mutated. Grounding rows are ordered by session, then
    episode, then step/frame, so the result does not depend on input order.
    """
    for name, rows in (("base_train", base_train),
                       ("base_validation", base_validation),
                       ("ground_rows", ground_rows)):
        if not isinstance(rows, (list, tuple)):
            raise ValueError("%s must be a sequence of rows" % name)
    if not ground_rows:
        raise ValueError("no grounding rows to mix")
    for row in ground_rows:
        if not isinstance(row, dict) or "frame" not in row:
            raise ValueError("grounding rows need a frame field")
    target = plan_mix(len(base_train), ratio)

    validation_keys = {
        session_key(row) for row in ground_rows
        if is_validation(session_key(row), val_percent, seed)
    }
    train_pool = [row for row in ground_rows if session_key(row) not in validation_keys]

    def order(row):
        return (session_key(row), str(row.get("episode_id", "")),
                int(row.get("step", 0)), str(row.get("frame", "")))

    train_pool.sort(key=order)
    used = train_pool[:target]
    dropped = len(train_pool) - len(used)
    validation_rows = [row for row in ground_rows if session_key(row) in validation_keys]
    validation_rows.sort(key=order)

    train = [dict(row) for row in base_train]
    train.extend(_prepare(row, "train", frames_prefix) for row in used)
    validation = [dict(row) for row in base_validation]
    validation.extend(_prepare(row, "validation", frames_prefix) for row in validation_rows)

    total = len(train)
    if total and used:
        achieved = round(len(used) / float(total), 4)
    else:
        achieved = 0.0
    manifest = {
        "kind": "v9_grounding_mix",
        "ratio_target": round(float(ratio), 6),
        "ratio_achieved": achieved,
        "val_percent": int(val_percent),
        "split_seed": int(seed),
        "frames_prefix": "" if frames_prefix is None else _relative(frames_prefix, "frames_prefix").as_posix(),
        "base_rows": len(base_train),
        "base_val_rows": len(base_validation),
        "ground_rows": len(ground_rows),
        "ground_train_available": len(train_pool),
        "ground_in_train": len(used),
        "ground_in_validation": len(validation_rows),
        "ground_dropped": dropped,
        "rows_with_target_point": sum(1 for row in train if row.get("target_point")),
        "total_train": total,
        "val_rows": len(validation),
        "kinds": dict(sorted(Counter(
            str(row.get("action_kind") or "unknown") for row in train
        ).items())),
        "files": {
            "train": {"sha256": hashlib.sha256(dump_rows(train)).hexdigest(), "rows": total},
            "validation": {"sha256": hashlib.sha256(dump_rows(validation)).hexdigest(),
                           "rows": len(validation)},
        },
    }
    return {"train": train, "validation": validation, "manifest": manifest}


def write_mix(out_dir, mixed):
    """Write ``train.jsonl``, ``validation.jsonl`` and the manifest.

    The destination must not already exist, so an interrupted or repeated run
    cannot silently overwrite a previous mixture.
    """
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError("output directory already exists: %s" % out)
    out.mkdir(parents=True)
    manifest = mixed["manifest"]
    for split, name in FILES.items():
        digest, rows = write_rows(out / name, mixed[split])
        expected = manifest["files"][split]
        if digest != expected["sha256"] or rows != expected["rows"]:
            raise ValueError("manifest/file mismatch for %s" % split)
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf8"
    )
    return manifest


def load_ground_rows(path):
    """Load grounding rows from a JSONL file or a directory of episode files."""
    path = Path(path)
    if path.is_dir():
        rows = []
        for item in sorted(path.iterdir()):
            if item.suffix == ".jsonl":
                rows.extend(read_jsonl(item))
            elif item.suffix == ".json":
                obj = json.loads(item.read_text(encoding="utf8"))
                if isinstance(obj, list):
                    rows.extend(obj)
                elif isinstance(obj, dict):
                    rows.append(obj)
                else:
                    raise ValueError("%s does not hold an object or list" % item)
        return rows
    return read_jsonl(path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Mix V9 grounding episodes into GUI SFT data."
    )
    parser.add_argument("--base-train", required=True, help="base train JSONL")
    parser.add_argument("--base-validation", required=True, help="base validation JSONL")
    parser.add_argument("--ground", required=True,
                        help="grounding JSONL, or a directory of episode files")
    parser.add_argument("--out", required=True, help="new output directory")
    parser.add_argument("--ratio", type=float, default=0.1,
                        help="ground fraction of the final training set")
    parser.add_argument("--frames-prefix", default=None,
                        help="workspace-relative prefix for ground frames")
    parser.add_argument("--val-percent", type=int, default=15)
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args(argv)

    mixed = build_mix(
        read_jsonl(args.base_train),
        read_jsonl(args.base_validation),
        load_ground_rows(args.ground),
        args.ratio,
        frames_prefix=args.frames_prefix,
        val_percent=args.val_percent,
        seed=args.split_seed,
    )
    manifest = write_mix(args.out, mixed)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
