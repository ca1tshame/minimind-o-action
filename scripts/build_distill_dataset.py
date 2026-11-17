"""Build the V11 teacher-distillation SFT dataset with source tags and balance.

Base = V8 mixed (V6 broad + V7/V8 balanced, session-deduped). Adds V10 and V11
teacher rows plus V11 scroll oracle. The oracle-only variant excludes teacher
rows. All inputs are local paths supplied via CLI; no fixed machine paths.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.open(encoding="utf8")] if path.exists() else []


def dedup(rows):
    seen, output = set(), []
    for row in rows:
        key = (row.get("episode_id", ""), row.get("step", 0), row.get("data_source", ""))
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build V11 distillation dataset.")
    parser.add_argument("--config")
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-train", required=True)
    parser.add_argument("--base-val", required=True)
    parser.add_argument("--teacher-v10", default="")
    parser.add_argument("--teacher-v11", default="")
    parser.add_argument("--scroll", default="")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base = load_jsonl(args.base_train)
    for row in base:
        row["data_source"] = "oracle_base"

    teacher_v10 = []
    v10_root = Path(args.teacher_v10) if args.teacher_v10 else None
    if v10_root:
        for relative in ("dataset_owl_v1/sft/teacher_success.jsonl",
                         "dataset_extra_owl/sft/teacher_success.jsonl"):
            for row in load_jsonl(v10_root / relative):
                frame = Path(row.get("frame", ""))
                resolved = frame if frame.exists() else v10_root / frame
                if not resolved.exists():
                    continue
                row["frame"] = str(resolved)
                row.setdefault("step", 0)
                row["data_source"] = "teacher_v10"
                teacher_v10.append(row)

    teacher_v11 = []
    for row in load_jsonl(args.teacher_v11):
        frame = Path(row.get("frame", ""))
        if not frame.exists():
            continue
        row["frame"] = str(frame)
        row["data_source"] = "teacher_v11"
        teacher_v11.append(row)

    scroll = []
    scroll_root = Path(args.scroll) if args.scroll else None
    if scroll_root and (scroll_root / "episodes").exists():
        for path in sorted((scroll_root / "episodes").glob("*.json")):
            episode = json.loads(path.read_text(encoding="utf8"))
            for row in episode.get("rows", []):
                if row.get("action_kind") != "scroll":
                    continue
                frame = scroll_root / "frames" / row["frame"]
                if not frame.exists():
                    continue
                scroll.append({
                    "episode_id": episode["task"] + "_" + str(episode["seed"]),
                    "task_id": episode["task"], "seed": episode["seed"],
                    "instruction": row["instruction"],
                    "history": row.get("history", []),
                    "frame": str(frame),
                    "target_action": row["target_action"],
                    "action_kind": "scroll",
                    "data_source": "oracle_scroll",
                    "source": "MiniWoB oracle v11",
                    "provenance": "oracle_success",
                    "split": "train",
                })

    for collection in (base, teacher_v10, teacher_v11, scroll):
        for row in collection:
            row.setdefault("step", 0)
            frame = row.get("frame")
            if frame and Path(frame).exists() and "frame_sha256" not in row:
                row["frame_sha256"] = hashlib.sha256(Path(frame).read_bytes()).hexdigest()

    teacher = teacher_v10 + teacher_v11
    all_rows = dedup(base + teacher + scroll)
    oracle_only = dedup(base + scroll)

    validation = load_jsonl(args.base_val)
    for row in validation:
        row["data_source"] = "oracle_base_val"

    names = {
        "train": all_rows,
        "train_oracle_only": oracle_only,
        "validation": validation,
    }
    for name, rows in names.items():
        with (out / (name + ".jsonl")).open("w", encoding="utf8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "base_rows": len(base),
        "teacher_v10": len(teacher_v10),
        "teacher_v11": len(teacher_v11),
        "scroll_oracle": len(scroll),
        "total_with_teacher": len(all_rows),
        "total_oracle_only": len(oracle_only),
        "kinds_with_teacher": dict(Counter(r.get("action_kind") for r in all_rows)),
        "kinds_oracle_only": dict(Counter(r.get("action_kind") for r in oracle_only)),
        "sources": dict(Counter(r.get("data_source") for r in all_rows)),
        "files": {
            name: {"rows": len(rows), "sha256": sha(out / (name + ".jsonl"))}
            for name, rows in names.items()
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf8"
    )
    print("V11_DATA", json.dumps({
        "total": len(all_rows), "oracle_only": len(oracle_only),
        "sources": manifest["sources"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
