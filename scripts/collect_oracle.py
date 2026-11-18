"""Collect environment-verified oracle trajectories on BrowserGym/MiniWoB.

Port of ``project/gui_distillation_v14/oracle_collect_v14.py``.  The collector
executes a real plan against the live environment (DOM is used only to produce
training labels), saves one JPEG per executed step, and records the target action
that the environment actually accepted.  Success is *never* fabricated: an
episode counts as successful only when the environment terminates with a
positive reward, and only such episodes contribute rows to
``trajectories.jsonl``.

The module is import-safe without BrowserGym: the dependency is imported lazily
by :func:`collect` / :func:`main`, which raise a clear ``SystemExit`` when it is
missing.

Output layout (defaults, all under ``--out``)::

    frames/v14_<task>_<seed>_<step>.jpg
    episodes/<task>__<seed>.json      all attempted episodes incl. failures
    trajectories.jsonl                canonical rows from successful episodes
    failures.jsonl                    attempted-but-unsuccessful episodes
    run.json                          counts and the exact task/seed plan

Then hand the trajectory file to the builder::

    python scripts/collect_oracle.py --out datasets/oracle_v14 --handoff

which runs ``scripts/build_distill_dataset.py`` with ``--v14-oracle`` pointing at
the produced ``trajectories.jsonl`` (other sources come from the config).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.filter_teacher import (  # noqa: E402
    ActionDSLError,
    parse_action_dsl,
    sha256_file,
    write_jsonl,
)

DEFAULT_TASKS = (
    "click-button-sequence",
    "use-autocomplete",
    "click-pie",
    "click-option",
    "scroll-text",
    "enter-text-2",
    "login-user-popup",
    "click-link",
)

# Tasks whose V14 oracle rollouts never reached a positive reward; they are kept
# in the default task list for completeness but are expected to yield no rows.
ZERO_SUCCESS_TASKS = ("click-pie", "click-option", "click-link", "login-user-popup")

DATASET_KIND = "v14_oracle"


def _require_browsergym():
    try:
        import gymnasium as gym  # noqa: F401
        import browsergym.miniwob  # noqa: F401
        from browsergym.core.action.highlevel import HighLevelActionSet  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "collect_oracle.py needs BrowserGym + MiniWoB (gymnasium, browsergym, "
            "playwright). Install them and re-run; import failed: %s" % exc
        )


def goal_text(goal):
    """Extract the key user-supplied string from a MiniWoB goal, or None."""
    if not isinstance(goal, str):
        return None
    match = re.search(r'"([^"]+)"', goal)
    if match:
        return match.group(1)
    match = re.search(r"labeled\s+([\w\-]+)", goal, re.I)
    if match:
        return match.group(1)
    match = re.search(r"^Select\s+([\w\-]+)\s+and", goal, re.I)
    if match:
        return match.group(1)
    match = re.search(r'starts with\s+"([^"]+)"', goal, re.I)
    if match:
        return match.group(1)
    return None


def click_xy(page, locator, idx=0):
    """Return normalised click coordinates of a DOM element, or None."""
    try:
        box = locator.nth(idx).bounding_box()
        if not box:
            return None
        viewport = page.viewport_size
        x = (box["x"] + box["width"] / 2.0) / viewport["width"]
        y = (box["y"] + box["height"] / 2.0) / viewport["height"]
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            return x, y
    except Exception:
        pass
    return None


def build_plan(task, goal, page):
    """Return ``(plan, error)`` where plan is a list of ``(kind, payload)``."""
    lowered = (goal or "").lower()

    if task == "click-pie":
        key = goal_text(goal)
        trigger = page.locator("svg").first
        xy = click_xy(page, trigger)
        if not xy:
            box = trigger.evaluate(
                "(e)=>{const r=e.getBoundingClientRect();"
                "return {x:r.x,y:r.y,width:r.width,height:r.height}}"
            )
            viewport = page.viewport_size
            if box and box["width"] > 0:
                xy = (
                    (box["x"] + box["width"] / 2.0) / viewport["width"],
                    (box["y"] + box["height"] / 2.0) / viewport["height"],
                )
        if not xy:
            return None, "no_svg"
        texts = page.locator("svg text")
        target_idx = None
        if key:
            for index in range(min(texts.count(), 20)):
                if (texts.nth(index).text_content() or "").strip().lower() == key.lower():
                    target_idx = index
                    break
        return [("click_xy", xy), ("click_pie_text", target_idx if target_idx is not None else 0)], None

    if task == "click-option":
        key = goal_text(goal)
        if not key:
            return None, "no_key"
        return [("click_checkbox_text", key), ("click_sel", "#subbtn")], None

    if task == "click-link":
        key = goal_text(goal)
        if not key:
            return None, "no_key"
        return [("click_link_text", key)], None

    if task == "login-user-popup":
        match = re.search(
            r'username\s+"([^"]+)"\s+and\s+(?:the\s+)?password\s+"([^"]+)"', goal, re.I
        )
        if not match:
            return None, "unparseable"
        return [
            ("click_sel", "#username"),
            ("type", match.group(1)),
            ("click_sel", "#password"),
            ("type", match.group(2)),
            ("click_sel", "#subbtn"),
            ("handle_popup", None),
        ], None

    if task == "enter-text-2":
        match = re.search(r'Type\s+"([^"]+)".*all (upper|lower) case', goal, re.I)
        if not match:
            return None, "unparseable"
        text = match.group(1)
        text = text.upper() if match.group(2).lower() == "upper" else text.lower()
        return [("click_sel", "#tt"), ("type", text), ("click_sel", "#subbtn")], None
    if task == "enter-text":
        match = re.search(r'Type\s+"([^"]+)"', goal, re.I)
        if not match:
            return None, "unparseable"
        return [("click_sel", "#tt"), ("type", match.group(1)), ("click_sel", "#subbtn")], None
    if task == "login-user":
        match = re.search(
            r'username\s+"([^"]+)"\s+and\s+(?:the\s+)?password\s+"([^"]+)"', goal, re.I
        )
        if not match:
            return None, "unparseable"
        return [
            ("click_sel", "#username"),
            ("type", match.group(1)),
            ("click_sel", "#password"),
            ("type", match.group(2)),
            ("click_sel", "#subbtn"),
        ], None
    if task == "scroll-text":
        down = "bottom" in lowered
        return [("scroll", 0.9 if down else -0.9), ("scroll", 0.9 if down else -0.9)], None
    if task == "scroll-text-2":
        down = "bottom" in lowered
        return [("scroll", 0.9 if down else -0.9), ("click_sel", "#subbtn")], None
    if task == "use-autocomplete":
        return [
            ("click_sel", "#tags"),
            ("type_first", None),
            ("click_menu", 0),
            ("click_sel", "#subbtn"),
        ], None
    if task == "click-checkboxes":
        return [("check_many", None), ("click_sel", "#subbtn")], None
    if task == "click-button-sequence":
        return [("click_sel", "#subbtn"), ("click_sel", "#subbtn2")], None
    return None, "no_plan"


def step_action(kind, payload, env, width, height):
    """Translate one plan step into ``(dsl, executor_command)``.

    ``width``/``height`` must be the screenshot size (not the viewport size).
    ``dsl`` is ``None`` for environment actions that the action DSL cannot
    express (for example pressing Enter to dismiss an alert); those steps are
    executed but not recorded as training targets.
    """
    page = env.unwrapped.page

    def click_command(x, y):
        return (
            "<action>click(%.4f,%.4f)</action>" % (x, y),
            "mouse_click(%d,%d)" % (x * width, y * height),
        )

    if kind == "click_sel":
        xy = click_xy(page, page.locator(payload))
        return click_command(*xy) if xy else None
    if kind == "click_xy":
        return click_command(*payload)
    if kind == "click_pie_text":
        texts = page.locator("svg text")
        idx = 0 if payload is None else payload
        xy = click_xy(page, texts, idx)
        return click_command(*xy) if xy else None
    if kind == "click_link_text":
        locator = page.locator("span").filter(has_text=payload)
        xy = click_xy(page, locator)
        return click_command(*xy) if xy else None
    if kind == "click_checkbox_text":
        boxes = page.locator("input[type=radio], input[type=checkbox]")
        for index in range(boxes.count()):
            parent = boxes.nth(index).locator("xpath=..")
            if payload.lower() in parent.inner_text().lower():
                xy = click_xy(page, boxes, index)
                if xy:
                    return click_command(*xy)
        return None
    if kind == "click_menu":
        xy = click_xy(page, page.locator(".ui-menu-item"), payload or 0)
        return click_command(*xy) if xy else None
    if kind == "type":
        return (
            '<action>type(%s)</action>' % json.dumps(payload, ensure_ascii=False),
            "keyboard_type(%r)" % payload,
        )
    if kind == "type_first":
        field = page.locator("#tags")
        field.click()
        prefix = ""
        match = re.search(
            r'starts with\s+"([^"]+)"', page.locator("body").inner_text(), re.I
        )
        if match:
            prefix = match.group(1)
        if not prefix:
            prefix = "Roma"
        field.fill(prefix)
        page.wait_for_timeout(400)
        options = page.locator(".ui-menu-item")
        text = prefix
        for index in range(options.count()):
            candidate = options.nth(index).inner_text().strip()
            if prefix.lower() in candidate.lower():
                text = candidate
                break
        return (
            '<action>type(%s)</action>' % json.dumps(text, ensure_ascii=False),
            "keyboard_type(%r)" % text,
        )
    if kind == "scroll":
        dy = float(payload)
        return (
            "<action>scroll(0.0000,%.4f)</action>" % dy,
            "scroll_at(%d,%d,0,%d)" % (width // 2, height // 2, int(-dy * height)),
        )
    if kind == "check_many":
        boxes = page.locator("input[type=checkbox]")
        commands = []
        for index in range(min(4, boxes.count())):
            xy = click_xy(page, boxes, index)
            if xy:
                commands.append(click_command(*xy))
        return ("MULTI", commands) if commands else None
    if kind == "handle_popup":
        return (None, "keyboard_press('Enter')")
    return None


def _save_frame(screen, path):
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - guarded by _require_browsergym
        raise SystemExit("saving frames needs Pillow: %s" % exc)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(screen).convert("RGB").save(path, "JPEG", quality=90)
    return sha256_file(path)


def _frame_name(task, seed, step):
    return "v14_%s_%d_%04d.jpg" % (task, seed, step)


def _relative_frame(path, frames_root):
    root = Path(frames_root).resolve()
    resolved = Path(path).resolve()
    if resolved != root and root not in resolved.parents:
        raise SystemExit(
            "frame %s is outside --frames-root %s; choose an --out inside the frames root"
            % (resolved, root)
        )
    return resolved.relative_to(root).as_posix()


def _make_row(task, seed, goal, step, history, dsl, frame, digest):
    try:
        kind, payload = parse_action_dsl(dsl, goal)
    except ActionDSLError as exc:
        raise SystemExit(
            "collector produced an unparseable label (%s): %s" % (exc.reason, dsl)
        )
    point = [payload["x"], payload["y"]] if kind == "click" else None
    return {
        "episode_id": "%s_%d" % (task, seed),
        "step": step,
        "task": task,
        "seed": seed,
        "instruction": goal,
        "frame": frame,
        "frame_sha256": digest,
        "history": list(history),
        "error_feedback": None,
        "target_action": dsl,
        "action_kind": kind,
        "data_source": DATASET_KIND,
        "split": "train",
        "reward": None,
        "success": None,
        "target_point": point,
    }


def _make_env(task, seed, action_set, headless=True):
    import gymnasium as gym
    import browsergym.miniwob  # noqa: F401

    return gym.make(
        "browsergym/miniwob." + task,
        headless=headless,
        action_mapping=action_set.to_python_code,
        pre_observation_delay=0.05,
    )


def _run_episode(task, seed, env_factory, frames_dir, frames_root):
    """Execute one episode; return its JSON document (rows are canonical)."""
    env = env_factory(task, seed)
    rows = []
    history = []
    status = "init"
    reward = 0.0
    terminated = False
    truncated = False
    goal = ""
    plan = None
    plan_error = None

    def record(screen, dsl):
        step = len(rows)
        name = _frame_name(task, seed, step)
        digest = _save_frame(screen, frames_dir / name)
        rows.append(
            _make_row(
                task,
                seed,
                goal,
                step,
                history[-8:],
                dsl,
                _relative_frame(frames_dir / name, frames_root),
                digest,
            )
        )

    try:
        obs, _ = env.reset(seed=seed)
        goal = str(obs.get("goal") or "")
        plan, plan_error = build_plan(task, goal, env.unwrapped.page)
        if plan is None:
            status = plan_error or "no_plan"
        else:
            for kind, payload in plan:
                screen = obs["screenshot"]
                height, width = screen.shape[:2]
                action = step_action(kind, payload, env, width, height)
                if action is None:
                    status = "no_target"
                    break
                dsl, command = action
                if dsl == "MULTI":
                    for multi_dsl, multi_command in command:
                        record(screen, multi_dsl)
                        obs, reward, terminated, truncated, info = env.step(multi_command)
                        history.append(multi_dsl)
                        if obs.get("last_action_error"):
                            history[-1] += " [executor error]"
                        if terminated or truncated:
                            break
                    if terminated or truncated:
                        break
                    continue
                if dsl is not None:
                    record(screen, dsl)
                obs, reward, terminated, truncated, info = env.step(command)
                if dsl is not None:
                    history.append(dsl)
                    if obs.get("last_action_error"):
                        history[-1] += " [executor error]"
                if terminated or truncated:
                    break
            success = bool(terminated and float(reward) > 0.0)
            status = "success" if success else ("no_reward" if rows else status)
    finally:
        try:
            env.close()
        except Exception:
            pass

    reward = float(reward) if isinstance(reward, (int, float)) else 0.0
    success = bool(terminated and reward > 0.0)
    if success:
        status = "success"
    for row in rows:
        row["reward"] = reward
        row["success"] = success
    document = {
        "kind": DATASET_KIND,
        "episode_id": "%s_%d" % (task, seed),
        "task": task,
        "seed": seed,
        "instruction": goal,
        "status": status,
        "success": success,
        "reward": reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "plan_error": plan_error,
        "plan": [list(item) for item in plan] if plan else None,
        "rows": rows,
    }
    return document


def collect(
    out_dir,
    tasks=None,
    seeds=None,
    frames_root=None,
    headless=True,
    overwrite=False,
    env_factory=None,
    progress=None,
):
    """Run the oracle plan over ``tasks`` x ``seeds`` and write the output tree."""
    tasks = [str(task) for task in (tasks or DEFAULT_TASKS)]
    seeds = [int(seed) for seed in (seeds or [])]
    if not tasks:
        raise ValueError("no tasks given")
    if not seeds:
        raise ValueError("no seeds given")
    out_dir = Path(out_dir)
    frames_root = Path(frames_root).resolve() if frames_root else out_dir.resolve()
    frames_dir = out_dir / "frames"
    episodes_dir = out_dir / "episodes"
    frames_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)

    if env_factory is None:
        _require_browsergym()
        from browsergym.core.action.highlevel import HighLevelActionSet

        action_set = HighLevelActionSet(subsets=["coord"], multiaction=False)
        env_factory = lambda task, seed: _make_env(task, seed, action_set, headless)  # noqa: E731

    trajectory = []
    failures = []
    attempted = 0
    succeeded = 0
    skipped = 0
    log = progress or (lambda message: None)

    for task in tasks:
        for seed in seeds:
            marker = episodes_dir / ("%s__%d.json" % (task, seed))
            if marker.exists() and not overwrite:
                document = json.loads(marker.read_text(encoding="utf-8"))
                skipped += 1
            else:
                try:
                    document = _run_episode(
                        task, seed, env_factory, frames_dir, frames_root
                    )
                except Exception as exc:  # never fake a success on a crashed rollout
                    document = {
                        "kind": DATASET_KIND,
                        "episode_id": "%s_%d" % (task, seed),
                        "task": task,
                        "seed": seed,
                        "instruction": "",
                        "status": "collector_error",
                        "success": False,
                        "reward": 0.0,
                        "terminated": False,
                        "truncated": False,
                        "plan_error": None,
                        "plan": None,
                        "rows": [],
                        "error": "%s: %s" % (type(exc).__name__, exc),
                    }
                marker.write_text(
                    json.dumps(document, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                attempted += 1
                log("oracle %s seed=%d status=%s" % (task, seed, document["status"]))
            if document.get("success") is True:
                succeeded += 1
                trajectory.extend(document.get("rows") or [])
            else:
                failures.append(
                    {
                        "episode_id": document.get("episode_id"),
                        "task": task,
                        "seed": seed,
                        "status": document.get("status"),
                        "reward": document.get("reward"),
                        "success": False,
                        "steps": len(document.get("rows") or []),
                    }
                )

    trajectory.sort(key=lambda row: (row["task"], row["seed"], row["step"]))
    failures.sort(key=lambda row: (row["task"], row["seed"]))
    write_jsonl(out_dir / "trajectories.jsonl", trajectory)
    write_jsonl(out_dir / "failures.jsonl", failures)
    summary = {
        "generator": "scripts/collect_oracle.py",
        "data_source": DATASET_KIND,
        "tasks": tasks,
        "seeds": seeds,
        "attempted_episodes": attempted,
        "skipped_existing": skipped,
        "successful_episodes": succeeded,
        "failed_episodes": len(failures),
        "rows": len(trajectory),
        "success_rule": "success = bool(terminated and reward > 0); failed episodes contribute no rows",
        "zero_success_tasks": list(ZERO_SUCCESS_TASKS),
        "files": {
            "trajectories.jsonl": {
                "rows": len(trajectory),
                "sha256": sha256_file(out_dir / "trajectories.jsonl"),
            },
            "failures.jsonl": {
                "rows": len(failures),
                "sha256": sha256_file(out_dir / "failures.jsonl"),
            },
        },
    }
    (out_dir / "run.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


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


def _seed_list(value, seed_lo, num_seeds):
    if value:
        seeds = []
        for part in str(value).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part[1:]:
                start, end = part.split("-", 1)
                seeds.extend(range(int(start), int(end) + 1))
            else:
                seeds.append(int(part))
        return seeds
    return list(range(int(seed_lo), int(seed_lo) + int(num_seeds)))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Collect environment-verified MiniWoB oracle trajectories."
    )
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None, help="output dir (default: datasets/oracle_v14)")
    parser.add_argument("--frames-root", default=None)
    parser.add_argument("--tasks", default=None, help="comma-separated MiniWoB task list")
    parser.add_argument("--seeds", default=None, help="comma/range seed list, e.g. 20101-20140")
    parser.add_argument("--seed-lo", type=int, default=None)
    parser.add_argument("--num-seeds", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="re-run episodes with a marker")
    parser.add_argument("--headful", action="store_true", help="run with a visible browser")
    parser.add_argument("--handoff", action="store_true", help="run the dataset builder afterwards")
    parser.add_argument("--build-out", default=None)
    parser.add_argument("--v11-teacher", default=None, help="extra source passed to the builder")
    parser.add_argument("--gui-owl", default=None, help="extra source passed to the builder")
    parser.add_argument("--frozen-test", default=None, help="frozen test file passed to the builder")
    args = parser.parse_args(argv)

    config = _read_config(args.config)
    distill = (config or {}).get("distill") or {}
    oracle_cfg = distill.get("oracle") or {}
    paths_cfg = (config or {}).get("paths") or {}
    root_value = args.workspace_root or paths_cfg.get("workspace_root")
    root = Path(root_value).resolve() if root_value else Path.cwd()

    out_value = args.out or oracle_cfg.get("out") or "datasets/oracle_v14"
    out_dir = _resolve(out_value, root)
    frames_value = args.frames_root or distill.get("frames_root")
    frames_root = _resolve(frames_value, root) if frames_value else root.resolve()

    task_value = args.tasks or oracle_cfg.get("tasks") or ",".join(DEFAULT_TASKS)
    tasks = [task.strip() for task in str(task_value).split(",") if task.strip()]
    seed_lo = args.seed_lo if args.seed_lo is not None else int(oracle_cfg.get("seed_lo", 20101))
    num_seeds = (
        args.num_seeds if args.num_seeds is not None else int(oracle_cfg.get("num_seeds", 40))
    )
    seeds = _seed_list(args.seeds, seed_lo, num_seeds)

    summary = collect(
        out_dir,
        tasks=tasks,
        seeds=seeds,
        frames_root=frames_root,
        headless=not args.headful,
        overwrite=args.overwrite,
        progress=lambda message: print("ORACLE_COLLECT " + message, flush=True),
    )
    print(json.dumps({"out": str(out_dir), **summary["files"]}, ensure_ascii=False))

    if args.handoff:
        from scripts import build_distill_dataset as builder

        builder_argv = [
            "--workspace-root",
            str(root),
            "--frames-root",
            str(frames_root),
            "--v14-oracle",
            str(out_dir / "trajectories.jsonl"),
        ]
        if args.v11_teacher:
            builder_argv += ["--v11-teacher", args.v11_teacher]
        if args.gui_owl:
            builder_argv += ["--gui-owl", args.gui_owl]
        if args.frozen_test:
            builder_argv += ["--frozen-test", args.frozen_test]
        if args.build_out:
            builder_argv += ["--out", args.build_out]
        if args.config:
            builder_argv += ["--config", args.config]
        print("ORACLE_HANDOFF build_distill_dataset.py", flush=True)
        return builder.main(builder_argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
