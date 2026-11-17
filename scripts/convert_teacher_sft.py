"""Convert environment-verified teacher trajectories into GUI-SFT rows.

Only steps from episodes whose final reward came from the environment count as
positive demonstrations. Failed episodes are kept as diagnostics, never as SFT
targets. Every row carries provenance (oracle / teacher / evaluation).
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert teacher trajectories to SFT rows.")
    parser.add_argument("--config")
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--frames-key", default="frames")
    parser.add_argument("--out", required=True)
    parser.add_argument("--teacher", required=True)
    args = parser.parse_args(argv)

    source = Path(args.eval_dir)
    out = Path(args.out)
    (out / "sft").mkdir(parents=True, exist_ok=True)
    good, bad = [], []
    stats = {"episodes": 0, "success": 0, "skipped_steps": 0}
    for path in sorted(source.glob("ep_*.json")):
        data = json.loads(path.read_text(encoding="utf8"))
        stats["episodes"] += 1
        episode_ok = bool(data.get("success"))
        stats["success"] += int(episode_ok)
        rows = []
        for row in data.get("rows", []):
            if not row.get("dsl") or row.get("kind") == "unsupported":
                stats["skipped_steps"] += 1
                continue
            rows.append({
                "episode_id": "%s_%s" % (data["task"], data["seed"]),
                "step": row["step"],
                "instruction": data["goal"],
                "history": row.get("history", []),
                "target_action": row["dsl"],
                "action_kind": row.get("kind"),
                "frame": str(source.name) + "/" + args.frames_key + "/" + row["frame"],
                "source": "teacher:" + args.teacher,
                "teacher_model": data.get("model"),
                "env_reward": row.get("reward"),
                "terminated": row.get("terminated"),
                "split": "train",
                "provenance": "teacher_success" if episode_ok else "teacher_failure",
            })
        (good if episode_ok else bad).extend(rows)

    success_path = out / "sft" / "teacher_success.jsonl"
    failure_path = out / "sft" / "teacher_failure.jsonl"
    for path, rows in ((success_path, good), (failure_path, bad)):
        with path.open("w", encoding="utf8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "teacher": args.teacher,
        "episodes": stats["episodes"],
        "success_episodes": stats["success"],
        "success_rate": round(stats["success"] / max(1, stats["episodes"]), 3),
        "sft_rows": len(good),
        "failure_rows_kept_for_diagnostics": len(bad),
        "skipped_steps": stats["skipped_steps"],
        "files": {
            "teacher_success": {"rows": len(good), "sha256": sha(success_path)},
            "teacher_failure": {"rows": len(bad), "sha256": sha(failure_path)},
        },
        "note": "positive rows only from environment-verified successful episodes",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print("CONVERTED", json.dumps(manifest, ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()
