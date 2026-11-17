"""MiniWoB++ closed-loop evaluation under the unified hist_used=6 protocol.

The policy consumes only the goal, the screenshot tiles, and its own previous
actions. Success is the environment's terminal reward; no DOM internals or
oracle labels are read at inference. This is the evaluation logic used for the
V11/V12/V13 main-line models.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from agent.ctx_pack import compact_history
from configs import load_config

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_prompt(goal, image_tokens, history_text):
    return (
        "<|im_start|>user\n%s\nTask screenshot follows.%s<|vision_end|>\n"
        "%s<|im_end|>\n<|im_start|>assistant\n"
        % (goal, image_tokens, history_text)
    )


def run_suite(cfg, model, tasks, seeds, output, max_steps, hist_used, limit=0, resume=False):
    """Run the cross-product of tasks and seeds, writing one JSONL per episode."""
    from agent.environment import make_env
    from agent.dsl import parse_action, to_browsergym

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    screenshot_dir = output / "screenshots"
    screenshot_dir.mkdir(exist_ok=True)

    done = set()
    episode_file = output / "episodes.jsonl"
    if resume and episode_file.exists():
        for line in episode_file.open(encoding="utf8"):
            try:
                entry = json.loads(line)
                done.add((entry["task"], entry["seed"]))
            except Exception:
                continue

    image_tokens = "<|image_pad|>" * cfg["eval"]["vision_tokens"]
    history_budget = cfg["eval"]["history_reserve_tokens"] + 136
    cases = [(task, seed) for task in tasks for seed in seeds]
    if limit:
        cases = cases[:limit]
    cases = [case for case in cases if case not in done]

    with episode_file.open("a", encoding="utf8", buffering=1) as handle:
        for task, seed in cases:
            env = None
            steps = []
            goal = ""
            status = "error"
            reward = 0.0
            history = []
            started = time.time()
            try:
                env = make_env(task, headless=True)
                observation, _ = env.reset(seed=seed)
                goal = observation.get("goal", "")
                for step in range(max_steps):
                    screen = observation["screenshot"]
                    height, width = screen.shape[:2]
                    history_text, _ = compact_history(
                        history, model.tokenizer, history_budget,
                        kind_only_after=cfg["eval"]["kind_only_after"],
                    )
                    prompt = build_prompt(goal, image_tokens, history_text)
                    try:
                        text = model.predict(screen, prompt)
                    except ValueError as exc:
                        if "context budget" not in str(exc):
                            raise
                        status = "context_overflow"
                        steps.append({"step": step, "error": str(exc)})
                        break
                    row = {
                        "step": step, "output": text,
                        "history_used": min(hist_used, len(history)),
                    }
                    try:
                        action = parse_action(text)
                        command = to_browsergym(action, width, height)
                        if command is None:
                            status = "model_stop"
                            steps.append(row)
                            break
                        row["executor"] = command
                    except ValueError as exc:
                        row["executor_error"] = str(exc)
                        command = "noop(0)"
                    observation, reward, terminated, truncated, _ = env.step(command)
                    row.update({
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    })
                    steps.append(row)
                    history.append(text)
                    if terminated or truncated:
                        status = "environment_terminated" if terminated else "environment_truncated"
                        break
                else:
                    status = "max_steps"
            except Exception as exc:
                status = "environment_error"
                steps.append({"exception": type(exc).__name__ + ": " + str(exc)[:300]})
            finally:
                if env is not None:
                    env.close()
            episode = {
                "task": task, "seed": seed, "goal": goal,
                "success": bool(status == "environment_terminated" and reward > 0),
                "reward": float(reward), "status": status,
                "steps": steps, "seconds": time.time() - started,
            }
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
            print(task, seed, status, "reward", reward, flush=True)

    meta = {
        "condition": {
            "hist_used": hist_used,
            "context": cfg["eval"]["context"],
            "vision_tokens": cfg["eval"]["vision_tokens"],
        },
        "tasks": list(tasks), "seeds": list(seeds),
        "success_definition": "environment terminated with positive reward",
    }
    (output / "config.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf8"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="MiniWoB++ closed-loop evaluation.")
    parser.add_argument("--config")
    parser.add_argument("--workspace-root")
    parser.add_argument("--protocol", choices=["historical", "unseen", "long"], default="historical")
    parser.add_argument("--output", required=True)
    parser.add_argument("--hist-used", type=int)
    parser.add_argument("--context", type=int)
    parser.add_argument("--vision-tokens", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, {
        "paths.workspace_root": args.workspace_root,
        "eval.hist_used": args.hist_used,
        "eval.context": args.context,
        "eval.vision_tokens": args.vision_tokens,
    })
    eval_cfg = cfg["eval"]
    protocol = cfg["eval"]["protocols"][args.protocol]
    if args.protocol == "historical":
        tasks = protocol["tasks"]
        seeds = protocol["seeds"]
    elif args.protocol == "unseen":
        tasks = protocol["tasks"] if "tasks" in protocol else []
        seeds = [protocol["seed"]]
    else:
        tasks = protocol["tasks"] if "tasks" in protocol else []
        seeds = [protocol["seed"]]

    if not tasks:
        raise ValueError("protocol %s has no task list in config" % args.protocol)

    from scripts.inference import build_predictor
    model = build_predictor(cfg)
    run_suite(
        cfg, model, tasks, seeds, args.output,
        args.max_steps or eval_cfg["max_steps"],
        eval_cfg["hist_used"], args.limit or eval_cfg["limit"], args.resume,
    )


if __name__ == "__main__":
    main()
