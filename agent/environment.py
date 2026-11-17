"""BrowserGym / MiniWoB++ closed-loop interaction.

The policy only ever sees the goal, the screenshot, and its own previous
actions. Task success is the environment's terminal reward, never a model
self-report and never an oracle label.
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

from transformers import AutoTokenizer

from agent.ctx_pack import compact_history
from agent.dsl import parse_action, to_browsergym
from agent.copy_resolve import resolve_or_fallback
from configs import load_config

ACTION_RE = re.compile(r"<action>(click|scroll|type|stop|longpress)(.*?)</action>")


def configure_browser(cfg):
    """Point Playwright and MiniWoB at local assets from config (no baked paths)."""
    paths = cfg["paths"]
    browsers = paths.get("playwright_browsers_path")
    miniwob = paths.get("miniwob_url")
    if browsers:
        os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(browsers))
    if miniwob:
        os.environ.setdefault("MINIWOB_URL", str(miniwob))


def load_tokenizer(cfg):
    tokenizer_dir = cfg["paths"].get("tokenizer_dir")
    if not tokenizer_dir:
        raise ValueError("paths.tokenizer_dir must be set")
    return AutoTokenizer.from_pretrained(
        str(tokenizer_dir), local_files_only=True, trust_remote_code=False
    )


def make_env(task, headless=True, delay=0.05):
    """Create a MiniWoB++ BrowserGym environment with the coordinate action set."""
    import gymnasium as gym
    from browsergym.core.action.highlevel import HighLevelActionSet
    import browsergym.miniwob  # noqa: F401  registers the environments

    action_set = HighLevelActionSet(subsets=["coord"], multiaction=False)
    return gym.make(
        "browsergym/miniwob." + task,
        headless=headless,
        action_mapping=action_set.to_python_code,
        pre_observation_delay=delay,
    )


def decode_action(text, width, height, goal):
    """Turn a model completion into a BrowserGym command, or None for stop."""
    matches = ACTION_RE.findall(text)
    if not matches:
        return None, None, "unparsable"
    kind, raw = matches[-1]
    if kind == "click":
        try:
            cx, cy = [float(v) for v in raw.split(",")][:2]
            if 0 <= cx <= 1 and 0 <= cy <= 1:
                return "mouse_click(%d,%d)" % (cx * width, cy * height), kind, None
        except ValueError:
            pass
        return None, kind, "invalid_coordinates"
    if kind == "type":
        argument = raw.strip().strip('"')
        resolved, meta = resolve_or_fallback(argument, goal)
        return "keyboard_type(%r)" % resolved[:200], kind, None if meta.get("ok") else "copy_fallback"
    if kind == "scroll":
        try:
            cx, cy = [float(v) for v in raw.split(",")][:2]
            return "scroll_at(%d,%d,0,%d)" % (
                cx * width, cy * height, -0.9 * height
            ), kind, None
        except ValueError:
            return None, kind, "invalid_coordinates"
    if kind == "stop":
        return "noop(0)", kind, None
    return None, kind, "unsupported"


def build_prompt(goal, image_tokens, history_text, error=None):
    """Construct the chat-formatted prompt with history and optional error hint."""
    prompt = (
        "<|im_start|>user\n%s\nTask screenshot follows.%s<|vision_end|>\n"
        % (goal, image_tokens)
    )
    prompt += history_text
    if error:
        prompt += "Last action failed: %s\n" % error[:60]
    prompt += "<|im_end|>\n<|im_start|>assistant\n"
    return prompt


def run_episode(env, model, tokenizer, cfg, task, seed, max_steps=None):
    """Run one closed-loop episode and return a trace dict.

    ``model`` must expose ``predict(screen_rgb, prompt) -> completion_text``.
    This keeps the environment loop independent of any particular model build.
    """
    eval_cfg = cfg["eval"]
    steps_cfg = cfg.get("inference", {})
    max_steps = max_steps or steps_cfg.get("max_steps", 6)
    history_budget = eval_cfg.get("history_reserve_tokens", 32) + 136
    image_tokens = "<|image_pad|>" * eval_cfg.get("vision_tokens", 320)

    trace = {"task": task, "seed": seed, "steps": []}
    obs, _ = env.reset(seed=seed)
    goal = obs.get("goal", "")
    history, error, reward = [], None, 0.0
    status = "max_steps"

    try:
        for step in range(max_steps):
            screen = obs["screenshot"]
            height, width = screen.shape[:2]
            history_text, _ = compact_history(history, tokenizer, history_budget)
            prompt = build_prompt(goal, image_tokens, history_text, error)
            try:
                text = model.predict(screen, prompt)
            except ValueError as exc:
                if "context budget" not in str(exc):
                    raise
                status = "context_overflow"
                trace["steps"].append({"step": step, "error": str(exc)})
                break

            command, kind, note = decode_action(text, width, height, goal)
            trace["steps"].append({
                "step": step, "model_output": text,
                "command": command, "kind": kind, "note": note,
            })
            if command is None:
                status = "model_stop"
                break
            history.append("<action>%s(%s)</action>" % (kind, text))
            obs, reward, terminated, truncated, _ = env.step(command)
            error = (
                str(obs["last_action_error"])[:60]
                if obs.get("last_action_error") else None
            )
            if terminated or truncated:
                status = "environment_terminated" if terminated else "environment_truncated"
                break
    finally:
        env.close()

    trace["success"] = bool(status == "environment_terminated" and reward > 0)
    trace["final_reward"] = float(reward)
    trace["status"] = status
    return trace


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a single MiniWoB++ episode.")
    parser.add_argument("--config")
    parser.add_argument("--workspace-root")
    parser.add_argument("--task")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--out", default="demo_trace.json")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, {
        "paths.workspace_root": args.workspace_root,
        "eval.tasks.0": args.task,
    })
    configure_browser(cfg)
    task = args.task or cfg["inference"]["default_task"]
    seed = args.seed if args.seed is not None else cfg["eval"]["protocols"]["unseen"]["seed"]

    # A concrete model adapter is provided by scripts/inference.py; importing it
    # lazily keeps this module usable for environment-only checks.
    from scripts.inference import build_predictor

    model = build_predictor(cfg)
    tokenizer = load_tokenizer(cfg)
    env = make_env(task)
    trace = run_episode(env, model, tokenizer, cfg, task, seed, args.max_steps)
    Path(args.out).write_text(
        json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("RESULT", trace["status"], "reward", trace["final_reward"])
    print("trace written to", args.out)


if __name__ == "__main__":
    main()
