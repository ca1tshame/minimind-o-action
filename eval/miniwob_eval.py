"""Real BrowserGym MiniWoB++ evaluation for MiniMind-O Action.

The formal protocol is 15 tasks x 16 seeds = 240 episodes with
``hist_used=6``; ``--protocol historical`` reproduces it exactly.  Episodes
are written as JSONL (``final_reward`` and ``reward`` both present) and
aggregated by :mod:`eval.metrics`.

Example::

    python eval/miniwob_eval.py --workspace-root /data/minimind \\
        --checkpoint checkpoints/gui_v14/payload/model.safetensors \\
        --protocol historical --out outputs/eval/v14_hist240
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.environment import configure_browser, run_episode
from configs import (apply_overrides, load_config, resolve_paths, workspace_root)
from eval.metrics import load_traces, summarize_episodes


def resolve_episodes(cfg, protocol, subset=None, limit=0):
    """Return an ordered list of ``(task, seed)`` pairs for a protocol."""
    protocols = (cfg.get("eval") or {}).get("protocols") or {}
    if protocol not in protocols:
        raise ValueError("unknown protocol %r (have %s)"
                         % (protocol, ", ".join(sorted(protocols))))
    spec = protocols[protocol]
    if "tasks" in spec and "seeds" in spec:
        tasks = list(spec["tasks"])
        if subset and subset != "all":
            subset_spec = protocols.get(subset)
            if not subset_spec or "tasks" not in subset_spec:
                raise ValueError("unknown task subset %r" % subset)
            allowed = set(subset_spec["tasks"])
            tasks = [task for task in tasks if task in allowed]
        pairs = [(task, int(seed)) for task in tasks for seed in spec["seeds"]]
    elif protocol == "unseen":
        seed = int(spec["seed"])
        count = int(spec.get("tasks_per_seed", 32))
        all_tasks = []
        for name in ("seen", "related_unseen", "distinct_unseen"):
            all_tasks.extend(protocols.get(name, {}).get("tasks", []))
        all_tasks = sorted(set(all_tasks))
        if not all_tasks:
            raise ValueError("unseen protocol needs task lists in config")
        pairs = [(all_tasks[i % len(all_tasks)], seed) for i in range(count)]
    elif protocol == "long":
        pairs = [(task, int(seed)) for task in spec["tasks"] for seed in spec["seeds"]]
    else:
        raise ValueError("protocol %r needs explicit tasks and seeds" % protocol)
    if limit and limit > 0:
        pairs = pairs[: int(limit)]
    return pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--protocol", default="historical")
    parser.add_argument("--subset", default=None,
                        help="filter historical tasks to seen/related_unseen/distinct_unseen")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-new", type=int, default=None)
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--hist-used", type=int, default=None)
    parser.add_argument("--out", default="outputs/eval")
    parser.add_argument("--no-headless", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    apply_overrides(cfg, {"paths.workspace_root": args.workspace_root})
    root = workspace_root(cfg, args.workspace_root)
    resolved = resolve_paths(cfg, root)
    for key in ("miniwob_url", "playwright_browsers_path"):
        if resolved.get(key):
            cfg.setdefault("paths", {})[key] = resolved[key]
    configure_browser(cfg)

    from scripts.inference import build_predictor

    policy = build_predictor(cfg, workspace=root, checkpoint=args.checkpoint,
                             device=args.device)
    pairs = resolve_episodes(cfg, args.protocol, args.subset, args.limit)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    episode_path = out_dir / "episodes.jsonl"
    if episode_path.exists():
        raise FileExistsError("refusing to append to an existing run: %s" % episode_path)
    episodes = []
    for index, (task, seed) in enumerate(pairs, 1):
        trace = run_episode(
            policy, task, cfg, seed=seed, headless=not args.no_headless,
            max_steps=args.max_steps, max_new_tokens=args.max_new, context=args.context,
            hist_used=args.hist_used, record_path=episode_path,
            instruction=(cfg.get("eval") or {}).get("instruction"),
        )
        episodes.append(trace)
        print("[%d/%d] %s seed=%s success=%s reward=%.3f steps=%d %s"
              % (index, len(pairs), task, seed, trace["success"], trace["final_reward"],
                 trace["num_steps"], trace["status"]))
    summary = summarize_episodes(episodes)
    summary.update({"protocol": args.protocol, "subset": args.subset})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
