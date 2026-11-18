"""MiniWoB++ / BrowserGym execution for the V14 GUI protocol.

Prompt assembly lives in :mod:`agent.ctx_pack` and action parsing in
:mod:`agent.dsl`; this module only wires them to a real BrowserGym
environment, feeds environment errors back to the policy and records a
complete episode trace.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from agent.ctx_pack import ContextOverflow, assemble_prompt, build_user_prompt
from agent.dsl import ActionParseError, format_action, parse_action, to_browsergym

__all__ = [
    "EpisodeResult",
    "build_prompt",
    "configure_browser",
    "decode_action",
    "make_env",
    "run_episode",
]


def configure_browser(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Export the BrowserGym/MiniWoB environment variables.

    ``paths.miniwob_url`` must point at the MiniWoB++ html directory (a
    ``file://`` URL).  ``paths.playwright_browsers_path`` is optional when the
    Playwright browsers live in their default location.  Values from the
    config win over the process environment; nothing machine-specific is
    hard-coded here.
    """
    paths = cfg.get("paths") or {}
    url = paths.get("miniwob_url")
    browsers = paths.get("playwright_browsers_path")
    exported: Dict[str, str] = {}
    if url:
        os.environ["MINIWOB_URL"] = str(url)
        exported["MINIWOB_URL"] = str(url)
    elif not os.environ.get("MINIWOB_URL"):
        raise RuntimeError(
            "MINIWOB_URL is not set: pass --workspace-root with a config that "
            "sets paths.miniwob_url or export MINIWOB_URL to the MiniWoB++ "
            "html directory (file:///.../miniwob/html/miniwob/)"
        )
    else:
        exported["MINIWOB_URL"] = os.environ["MINIWOB_URL"]
    if browsers:
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(browsers)
        exported["PLAYWRIGHT_BROWSERS_PATH"] = str(browsers)
    return exported


def make_env(task: str, headless: bool = True, delay: float = 0.05):
    """Create a BrowserGym MiniWoB++ environment with the coordinate action set."""
    try:
        import gymnasium as gym
        import browsergym.miniwob  # noqa: F401  (registers the environments)
        from browsergym.core.action.highlevel import HighLevelActionSet
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise RuntimeError(
            "BrowserGym is not installed: pip install 'minimind-o-action[eval]'"
        ) from exc
    action_set = HighLevelActionSet(subsets=["coord"], strict=False, multiaction=False)
    kwargs = {
        "action_mapping": action_set.to_python_code,
        "headless": headless,
    }
    # BrowserGym renamed this argument over time: older releases accept
    # ``pre_observation_delay`` (seconds) and newer ones ``wait_for_action``
    # (a callable). Passing the wrong one raises TypeError from the env
    # constructor, so probe the signature instead of hard-coding a version.
    try:
        import inspect

        from browsergym.core.env import BrowserEnv

        params = inspect.signature(BrowserEnv.__init__).parameters
    except Exception:  # pragma: no cover - introspection is best-effort
        params = {}
    if "wait_for_action" in params:
        kwargs["wait_for_action"] = lambda *_: delay
    elif "pre_observation_delay" in params:
        kwargs["pre_observation_delay"] = delay
    env = gym.make("browsergym/miniwob." + task, **kwargs)
    return env


def build_prompt(goal: str, image_tokens: int, history_text: str = "", error: str = "") -> str:
    """Render the V14 user turn (kept for backwards compatibility)."""
    return build_user_prompt(goal, image_tokens, history_text, error)


def decode_action(text: str, width: int, height: int, instruction: Optional[str] = None):
    """Parse a raw model output into ``(browsergym_command, kind, value)``.

    ``command`` is ``None`` for ``stop``; the caller must terminate.
    """
    kind, value = parse_action(text, instruction)
    command = to_browsergym((kind, value), width, height, instruction)
    return command, kind, value


@dataclass
class EpisodeResult:
    task: str
    seed: Optional[int]
    success: bool
    final_reward: float
    steps: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    error: Optional[str] = None
    prompt_tokens: int = 0
    seconds: float = 0.0

    def to_trace(self, instruction: str = "") -> Dict[str, Any]:
        return {
            "task": self.task,
            "seed": self.seed,
            "instruction": instruction,
            "success": self.success,
            "final_reward": self.final_reward,
            "reward": self.final_reward,
            "status": self.status,
            "error": self.error,
            "steps": self.steps,
            "num_steps": len(self.steps),
            "prompt_tokens": self.prompt_tokens,
            "seconds": round(self.seconds, 3),
        }


def run_episode(policy: Any, task: str, cfg: Dict[str, Any], *, seed: Optional[int] = None,
                headless: bool = True, max_steps: Optional[int] = None,
                max_new_tokens: Optional[int] = None, context: Optional[int] = None,
                hist_used: Optional[int] = None, vision_tokens: Optional[int] = None,
                record_path: Optional[Path] = None, instruction: Optional[str] = None,
                frames_dir: Optional[Path] = None, delay: float = 0.05) -> Dict[str, Any]:
    """Run one MiniWoB episode with a real policy.

    ``policy`` must expose ``predict(screen_rgb, prompt) -> str`` and a
    ``tokenizer`` attribute (``GUIPolicy`` from ``scripts/inference.py`` does).
    The task instruction comes from the environment goal unless overridden.
    """
    eval_cfg = cfg.get("eval") or {}
    max_steps = int(max_steps if max_steps is not None else eval_cfg.get("max_steps", 8))
    max_new_tokens = int(max_new_tokens if max_new_tokens is not None
                         else eval_cfg.get("max_new", 48))
    context = int(context if context is not None else eval_cfg.get("context", 512))
    hist_used = int(hist_used if hist_used is not None else eval_cfg.get("hist_used", 6))
    vision_tokens = int(vision_tokens if vision_tokens is not None
                        else eval_cfg.get("vision_tokens", 320))
    tokenizer = policy.tokenizer
    env = make_env(task, headless=headless, delay=delay)
    started = time.time()
    history: List[str] = []
    error = ""
    steps: List[Dict[str, Any]] = []
    result = EpisodeResult(task=task, seed=seed, success=False, final_reward=0.0)
    try:
        obs, _info = env.reset(seed=seed)
        goal = instruction or str(obs.get("goal") or task)
        terminated = truncated = False
        for step in range(max_steps):
            try:
                packed = assemble_prompt(
                    tokenizer, goal, vision_tokens, history, context, max_new_tokens,
                    error=error, hist_used=hist_used,
                )
            except ContextOverflow as exc:
                result.status = "context_overflow"
                result.error = str(exc)
                break
            result.prompt_tokens = max(result.prompt_tokens, packed.input_tokens)
            record: Dict[str, Any] = {
                "step": step,
                "input_tokens": packed.input_tokens,
                "history_used": packed.history_used,
                "history_stubbed": packed.history_stubbed,
                "history_dropped": packed.history_dropped,
                "error_feedback": error,
            }
            screen = np.asarray(obs["screenshot"], dtype=np.uint8)
            if frames_dir is not None:
                from PIL import Image

                frame_dir = Path(frames_dir)
                frame_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(screen).save(frame_dir / ("step_%03d.png" % step))
            try:
                text = policy.predict(screen, packed.prompt)
            except Exception as exc:  # model/runtime failure ends the episode
                result.status = "policy_error"
                result.error = "%s: %s" % (type(exc).__name__, exc)
                record["output"] = ""
                steps.append(record)
                break
            record["output"] = text
            try:
                # Coordinates must be scaled by the *screenshot* size, not the
                # CSS viewport: MiniWoB renders at 498x321 while the viewport is
                # 332x214 (1.5x), so using the viewport silently misclicks
                # everything. The screenshot is what the model actually saw.
                shot_h, shot_w = screen.shape[:2]
                command, kind, value = decode_action(text, shot_w, shot_h, goal)
            except ActionParseError as exc:
                error = str(exc)[:120]
                record["parse_error"] = error
                steps.append(record)
                continue
            record["action"] = format_action(kind, value)
            record["action_kind"] = kind
            if kind == "stop":
                record["stopped"] = value or "done"
                steps.append(record)
                result.status = "model_stop"
                break
            obs, reward, terminated, truncated, info = env.step(command)
            record["reward"] = float(reward)
            last_error = ""
            if isinstance(info, dict):
                last_error = str(info.get("last_action_error") or "")
            record["env_error"] = last_error
            error = last_error[:120]
            history.append(format_action(kind, value))
            steps.append(record)
            result.final_reward = float(reward)
            if terminated or truncated:
                break
        result.success = bool(terminated and result.final_reward > 0)
        if result.status == "ok":
            result.status = "success" if result.success else (
                "truncated" if truncated else "failed")
    finally:
        result.steps = steps
        result.seconds = time.time() - started
        result.prompt_tokens = result.prompt_tokens
        try:
            env.close()
        except Exception:
            pass
    trace = result.to_trace(goal)
    if record_path is not None:
        record_path = Path(record_path)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        with record_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
    return trace
