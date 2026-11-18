"""Assemble a real rollout trace + screenshots into a GitHub-sized GIF.

The recorder never invents an episode: it requires the JSON trace produced by
``scripts/demo_gui_agent.py`` and the per-step screenshots saved next to it.
Each frame is annotated with the step number, the sampled action, the
environment reward and the action error, so the GIF is a replay of a real
rollout and nothing else.

Usage::

    python scripts/demo_gui_agent.py --checkpoint ... --task click-dialog \
        --seed 20001 --output outputs/demo/click-dialog_20001.json
    python scripts/record_demo_gif.py \
        --trace outputs/demo/click-dialog_20001.json \
        --frames outputs/demo/click-dialog_20001_frames \
        --out assets/demo/click-dialog_20001.gif
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

__all__ = ["build_gif", "main"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSansMono.ttf", 14)
    except OSError:
        return ImageFont.load_default()


def build_gif(trace_path: Path, frames_dir: Path, out_path: Path, *, width: int = 960,
              fps: int = 2, note: str = "") -> dict:
    trace = json.loads(Path(trace_path).read_text(encoding="utf8"))
    steps = trace.get("steps") or []
    if not steps:
        raise SystemExit("trace has no steps: refusing to build an empty GIF")
    frames = []
    font = _font()
    for step in steps:
        number = int(step.get("step", 0))
        path = Path(frames_dir) / ("step_%03d.png" % number)
        if not path.is_file():
            raise SystemExit("missing real screenshot for step %d: %s" % (number, path))
        image = Image.open(path).convert("RGB")
        ratio = width / image.width
        image = image.resize((width, int(image.height * ratio)), Image.BILINEAR)
        draw = ImageDraw.Draw(image)
        lines = [
            "step %d  %s" % (number, step.get("action") or step.get("action_kind") or "?"),
            "reward %s" % step.get("reward"),
        ]
        if step.get("env_error"):
            lines.append("env error: %s" % step["env_error"])
        if step.get("parse_error"):
            lines.append("parse error: %s" % step["parse_error"])
        draw.rectangle([0, 0, image.width, 16 * len(lines) + 10], fill=(0, 0, 0))
        for row, text in enumerate(lines):
            draw.text((6, 4 + 16 * row), text, fill=(255, 255, 255), font=font)
        frames.append(image.convert("P", palette=Image.ADAPTIVE))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / max(fps, 1)), loop=0, optimize=True)
    meta = {
        "trace": str(trace_path),
        "trace_sha256": _sha256(Path(trace_path)),
        "frames_dir": str(frames_dir),
        "gif": str(out_path),
        "gif_sha256": _sha256(out_path),
        "task": trace.get("task"),
        "seed": trace.get("seed"),
        "success": trace.get("success"),
        "final_reward": trace.get("final_reward"),
        "steps": len(steps),
        "replay_note": note or "real rollout recorded from the student checkpoint",
        "fps": fps,
    }
    out_path.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    return meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--out", default="assets/demo/episode.gif")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--note", default="")
    args = parser.parse_args(argv)
    meta = build_gif(Path(args.trace), Path(args.frames), Path(args.out),
                     width=args.width, fps=args.fps, note=args.note)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
