"""Aggregate MiniWoB++ episode traces into the unified hist_used=6 metrics.

All rates are computed from environment-verified success (terminal reward > 0),
never from model self-reports. A single protocol's numbers are only comparable
to another collected under the same history-used setting.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def summarize_episodes(episodes):
    total = len(episodes)
    successes = sum(1 for ep in episodes if ep.get("success"))
    rewards = [float(ep.get("final_reward", 0.0)) for ep in episodes]
    by_status = defaultdict(int)
    by_task = defaultdict(lambda: [0, 0])
    for ep in episodes:
        by_status[ep.get("status", "unknown")] += 1
        task = ep.get("task", "unknown")
        by_task[task][1] += 1
        if ep.get("success"):
            by_task[task][0] += 1
    return {
        "episodes": total,
        "successes": successes,
        "success_rate": round(successes / total, 4) if total else 0.0,
        "mean_reward": round(sum(rewards) / total, 4) if total else 0.0,
        "by_status": dict(by_status),
        "by_task": {k: {"success": v[0], "total": v[1]} for k, v in sorted(by_task.items())},
    }


def load_traces(path):
    path = Path(path)
    if path.is_dir():
        episode_file = path / "episodes.jsonl"
        if not episode_file.exists():
            raise FileNotFoundError(episode_file)
        path = episode_file
    return [json.loads(line) for line in path.open(encoding="utf8") if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize MiniWoB++ closed-loop results.")
    parser.add_argument("--traces", required=True, help="episodes.jsonl or a result directory")
    parser.add_argument("--config")
    parser.add_argument("--hist-used", type=int)
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    episodes = load_traces(args.traces)
    result = summarize_episodes(episodes)
    result["hist_used"] = args.hist_used
    result["success_definition"] = "environment terminated with positive reward"
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf8")
    print(text)


if __name__ == "__main__":
    main()
