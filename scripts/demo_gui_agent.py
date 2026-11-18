"""One-episode demo of a trained MiniMind-O Action checkpoint on MiniWoB++.

The demo loads a real action checkpoint (frozen SigLIP2 -> 5 tiles ->
tile compressor -> projector -> MiniMind) and runs a single BrowserGym
episode, then writes the episode trace as JSON.  It is deliberately strict:

* a missing or mismatched checkpoint, tokenizer or vision-encoder directory
  aborts with a clear message instead of running a randomly initialised model;
* a missing BrowserGym install aborts with the install command instead of
  replaying a recorded trace or inventing rewards;
* the JSON trace is written only after ``agent.environment.run_episode``
  returns, so the file always describes a real rollout.

Loading reuses :func:`scripts.train_grpo.build_action_model`, which attaches
the tile compressor *before* applying the checkpoint.  Doing it in the other
order silently leaves ``compressor``/``tile_pos`` random while the checkpoint
reports success, which is the bug this shared loader prevents.

Example::

    python scripts/demo_gui_agent.py \\
        --checkpoint outputs/experiments/grpo/checkpoints/step_000040/payload/model.safetensors \\
        --task click-dialog --seed 20001 --show \\
        --output outputs/demo/click-dialog_20001.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SEED = 20001


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one real MiniWoB++ episode with a trained action checkpoint."
    )
    parser.add_argument("--config", default=None,
                        help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--workspace-root", default=None,
                        help="root all relative config paths resolve against")
    parser.add_argument("--checkpoint", default=None,
                        help="action checkpoint (default: paths.checkpoint)")
    parser.add_argument("--device", default=None, help="cpu / cuda / cuda:0")
    parser.add_argument("--task", default=None,
                        help="MiniWoB task (default: inference.default_task)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-new", type=int, default=None)
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--hist-used", type=int, default=None)
    parser.add_argument("--output", default=None,
                        help="trace JSON path (default: outputs/demo/<task>_seed<seed>.json)")
    parser.add_argument("--show", action="store_true",
                        help="run the browser headed instead of headless")
    parser.add_argument("--frames-dir", default=None,
                        help="save one screenshot per step here (for record_demo_gif.py)")
    return parser.parse_args(argv)


def load_cfg(args: argparse.Namespace):
    from configs import load_config

    return load_config(args.config, {
        "paths.workspace_root": args.workspace_root,
        "paths.checkpoint": args.checkpoint,
        "device.device": args.device,
    })


def resolve_checkpoint(cfg, args: argparse.Namespace) -> Path:
    from configs import workspace_root

    from scripts.train_grpo import resolve_asset

    root = workspace_root(cfg, args.workspace_root)
    checkpoint = resolve_asset(
        root, (cfg.get("paths") or {}).get("checkpoint")
    )
    if checkpoint is None:
        raise SystemExit(
            "no action checkpoint configured: pass --checkpoint or set "
            "paths.checkpoint in the config. Refusing to run an untrained model "
            "and present its output as a demo."
        )
    if not Path(checkpoint).is_file():
        raise SystemExit(
            "action checkpoint not found: %s. Refusing to run an untrained model "
            "and present its output as a demo." % checkpoint
        )
    return Path(checkpoint)


def require_browsergym() -> None:
    try:
        import browsergym.miniwob  # noqa: F401  (import registers the tasks)
    except ImportError as exc:
        raise SystemExit(
            "BrowserGym is not installed, so a real MiniWoB episode cannot run. "
            "Install the eval extra: pip install 'minimind-o-action[eval]'. "
            "No trace was written: this demo never fabricates an episode."
        ) from exc


def build_policy(cfg, checkpoint: Path, device_name: str):
    """Build the model through the shared, order-correct action loader."""
    import torch

    from configs import resolve_device

    from scripts.train_grpo import (
        GuiPolicy,
        build_action_model,
        load_tokenizer,
        rl_settings,
    )

    settings = rl_settings(cfg)
    device = torch.device(device_name or resolve_device(cfg))
    tokenizer = load_tokenizer(cfg, workspace_root(cfg))
    model, report = build_action_model(
        cfg, checkpoint, device=device, settings=settings
    )
    model.eval()
    policy = GuiPolicy(
        model,
        tokenizer,
        device=device,
        vision_tokens=settings["vision_tokens"],
        context=settings["context"],
        max_new=settings["max_new"],
    )
    return policy, report, device


def workspace_root(cfg):
    from configs import workspace_root as _workspace_root

    return _workspace_root(cfg)


def main(argv=None) -> int:
    args = parse_args(argv)

    cfg = load_cfg(args)
    root = workspace_root(cfg)
    checkpoint = resolve_checkpoint(cfg, args)
    require_browsergym()
    # Imported only after the BrowserGym check so a missing install produces
    # this module's install hint instead of a bare ImportError.
    from agent.environment import configure_browser, run_episode

    try:
        configure_browser(cfg)
    except RuntimeError as exc:
        raise SystemExit(
            "MiniWoB is not reachable: %s\nSet paths.miniwob_url in the config or "
            "MINIWOB_URL in the environment." % exc
        ) from exc

    policy, report, device = build_policy(cfg, checkpoint, args.device)
    print("LOADED", json.dumps({
        "checkpoint": report.get("path"),
        "renamed_keys": report.get("renamed_keys"),
        "dropped_vision_encoder_keys": report.get("dropped_vision_encoder_keys"),
        "missing_keys": len(report.get("missing_keys") or []),
        "unexpected_keys": len(report.get("unexpected_keys") or []),
        "parameters": report.get("parameters"),
        "device": str(device),
    }, ensure_ascii=False))

    task = args.task or (cfg.get("inference") or {}).get("default_task", "click-dialog")
    trace = run_episode(
        policy,
        task,
        cfg,
        seed=args.seed,
        headless=not args.show,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new,
        context=args.context,
        hist_used=args.hist_used,
        vision_tokens=policy.vision_tokens,
        frames_dir=Path(args.frames_dir) if args.frames_dir else (
            root / "outputs" / "demo" / (task + "_seed%d_frames" % args.seed)
        ),
    )

    output = Path(args.output) if args.output else (
        root / "outputs" / "demo" / ("%s_seed%d.json" % (task, args.seed))
    )
    if not output.is_absolute():
        output = Path.cwd() / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf8")

    print(json.dumps({key: value for key, value in trace.items() if key != "steps"},
                     indent=2, ensure_ascii=False))
    for step in trace.get("steps", []):
        print("STEP %s | %s | %s | reward=%s" % (
            step.get("step"),
            step.get("action") or step.get("action_kind") or "?",
            str(step.get("output", ""))[:120].replace("\n", " "),
            step.get("reward", 0.0),
        ))
    print("TRACE", output)
    return 0 if trace.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
