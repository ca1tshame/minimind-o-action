"""Collect grounding targets from BrowserGym/MiniWoB DOM element boxes.

Two modes:

**Offline (default)** -- ``--episodes`` reads a JSONL of recorded episodes. Each
line describes one MiniWoB observation::

    {"episode_id": "click-dialog_6201", "task": "click-dialog", "seed": 6201,
     "instruction": "Close the dialog", "viewport": [1280, 720],
     "frame": "frames/click-dialog_6201/step_0.jpg", "history": [],
     "elements": [{"selector": "button.ui-button",
                   "bbox": [640, 300, 120, 40], "space": "pixels"}]}

and one training row is emitted per element. This mode needs no browser and no
BrowserGym import, so the module is importable on a plain CPU machine.

**Live** -- ``--live`` drives MiniWoB++ through BrowserGym, asks the page for
element bounding boxes and writes frames plus a JSONL. ``gymnasium`` and
``browsergym`` are imported lazily inside :func:`collect_live`, so importing
this module never requires them.

Both modes use :func:`models.grounding.point_to_token` on the box centre, so
the emitted ``target_token`` / ``target_offset`` are the same targets the head
is trained and evaluated with.
"""
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import load_config
from models.grounding import point_to_token

SOURCE = "browsergym-miniwob"
SMALL_TARGET_AREA = 0.01
IMAGE_QUALITY = 90
TASKS = (
    "click-button-sequence", "click-dialog", "click-option",
    "click-checkboxes", "click-pie", "click-test",
)


def plan(task, goal=""):
    """Return the selector groups to click for a task, in step order.

    Each group is a tuple ``(parent_selector, child_selector, ...)``; the last
    entry is the clickable element. This is the DOM plan from the original
    collector, kept as data so it can be reviewed without a browser.
    """
    if task == "click-button-sequence":
        return [("#subbtn",), ("#subbtn2",)]
    if task == "click-dialog":
        return [("button.ui-button",)]
    if task == "click-option":
        return [("select#dropdown", "option")]
    if task == "click-checkboxes":
        match = re.search(r"Select (\d+)", goal or "")
        count = int(match.group(1)) if match else 1
        return [("input[type=checkbox]",)] * max(1, count)
    if task == "click-pie":
        return [("#pie-slice-0",)]
    if task == "click-test":
        return [("#subbtn",)]
    raise ValueError("no DOM plan for task %r" % (task,))


def _check_relative(path, name):
    text = str(path).replace("\\", "/")
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts or (pure.parts and ":" in pure.parts[0]):
        raise ValueError("%s must be a workspace-relative path: %r" % (name, path))
    return pure.as_posix()


def normalize_box(box, viewport, space="pixels"):
    """Normalise ``[x, y, w, h]`` to viewport coordinates in ``[0, 1]``.

    ``space="pixels"`` divides by the viewport width/height; ``"normalized"``
    accepts already-normalised boxes. Boxes extending outside the viewport are
    clipped; empty or inverted boxes raise ``ValueError``.
    """
    if space not in ("pixels", "normalized"):
        raise ValueError("space must be 'pixels' or 'normalized'")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("bbox must be [x, y, w, h]")
    values = [float(value) for value in box]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("bbox values must be finite")
    width, height = (float(viewport[0]), float(viewport[1]))
    if width <= 0 or height <= 0:
        raise ValueError("viewport must be positive")
    if space == "pixels":
        values = [values[0] / width, values[1] / height,
                  values[2] / width, values[3] / height]
    x, y, w, h = values
    if w <= 0 or h <= 0:
        raise ValueError("bbox must have positive size")
    x0 = min(max(x, 0.0), 1.0)
    y0 = min(max(y, 0.0), 1.0)
    x1 = min(max(x + w, 0.0), 1.0)
    y1 = min(max(y + h, 0.0), 1.0)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("bbox lies outside the viewport")
    return [x0, y0, x1 - x0, y1 - y0]


def row_from_element(episode, element, viewport):
    """Build one grounding training row from an episode and a DOM element."""
    if not isinstance(episode, dict) or not isinstance(element, dict):
        raise ValueError("episode and element must be objects")
    instruction = str(episode.get("instruction", "")).strip()
    if not instruction:
        raise ValueError("episode is missing a non-empty instruction")
    frame = _check_relative(episode.get("frame"), "frame")
    task = str(episode.get("task", "")).strip()
    seed = episode.get("seed")
    if not task or seed is None:
        raise ValueError("episode needs task and seed")
    viewport = [float(viewport[0]), float(viewport[1])]
    bbox = normalize_box(element.get("bbox"), viewport,
                         str(element.get("space", episode.get("space", "pixels"))))
    cx, cy = bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0
    token, ox, oy = point_to_token(cx, cy)
    step = int(episode.get("step", 0))
    history = [str(item) for item in (episode.get("history") or [])]
    row = {
        "episode_id": str(episode.get("episode_id") or "%s_%s" % (task, seed)),
        "step": step,
        "instruction": instruction,
        "history": history,
        "target_action": "<action>click(%.4f,%.4f)</action>" % (cx, cy),
        "action_kind": "click",
        "target_bbox": [round(value, 6) for value in bbox],
        "target_point": [round(cx, 6), round(cy, 6)],
        "bbox_area": round(bbox[2] * bbox[3], 6),
        "small_target": bool(bbox[2] * bbox[3] < SMALL_TARGET_AREA),
        "target_token": token,
        "target_offset": [round(ox, 6), round(oy, 6)],
        "frame": frame,
        "viewport": [int(viewport[0]), int(viewport[1])],
        "source": SOURCE,
        "split": "train",
        "task": task,
        "seed": seed,
    }
    selector = element.get("selector")
    if selector:
        row["selector"] = str(selector)
    return row


def collect_rows(episodes, workspace_root=None, check_frames=False):
    """Turn recorded episodes into grounding rows.

    When ``check_frames`` is set, every row's frame must exist under
    ``workspace_root``; this catches path drift before a training run instead
    of at the first batch.
    """
    if not isinstance(episodes, (list, tuple)) or not episodes:
        raise ValueError("episodes must be a non-empty sequence")
    root = Path(workspace_root) if workspace_root else None
    if check_frames and root is None:
        raise ValueError("--check-frames requires --workspace-root")
    rows = []
    for episode in episodes:
        viewport = episode.get("viewport")
        if not isinstance(viewport, (list, tuple)) or len(viewport) != 2:
            raise ValueError("episode must carry a [width, height] viewport")
        elements = episode.get("elements")
        if not isinstance(elements, (list, tuple)) or not elements:
            raise ValueError("episode must carry at least one element")
        for element in elements:
            row = row_from_element(episode, element, viewport)
            if check_frames:
                path = root / row["frame"]
                if not path.is_file() or path.stat().st_size == 0:
                    raise FileNotFoundError("missing frame: %s" % row["frame"])
            rows.append(row)
    return rows


def write_rows(path, rows):
    """Write canonical JSONL and return its sha256."""
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf8")
    Path(path).write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Live BrowserGym collection (optional; imported lazily)
# ---------------------------------------------------------------------------

def _browsergym():
    try:
        import gymnasium as gym
        import browsergym.miniwob  # noqa: F401  (registers the environments)
        from browsergym.core.action.highlevel import HighLevelActionSet
    except ImportError as exc:
        raise SystemExit(
            "live collection needs gymnasium and browsergym; "
            "use --episodes for offline collection"
        ) from exc
    return gym, HighLevelActionSet


def _first_box(page, selectors):
    """Return the first visible bounding box for a selector chain."""
    locator = page.locator(selectors[-1]).first
    box = locator.bounding_box()
    if box is None:
        raise ValueError("no bounding box for selectors %r" % (selectors,))
    return box


def collect_live(args):
    """Drive MiniWoB++ and record one click per planned selector.

    This is a best-effort port of the research collector: BrowserGym and
    Playwright versions differ, so page access is guarded and failures name the
    missing API. The offline path is the stable interface.
    """
    gym, action_set_cls = _browsergym()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("output directory already exists: %s" % out)
    frames = out / "frames"
    frames.mkdir(parents=True)
    rows = []
    for task in args.tasks:
        for seed in range(args.seed_lo, args.seed_lo + args.seeds):
            env = gym.make(
                "browsergym/miniwob." + task,
                headless=True,
                action_mapping=action_set_cls(
                    subsets=["coord"], multiaction=False
                ).to_python_code,
            )
            try:
                obs, _ = env.reset(seed=seed)
                page = env.unwrapped.page
                viewport = page.viewport_size
                if not viewport:
                    raise ValueError("page has no viewport size")
                goal = obs.get("goal") or obs.get("instruction") or ""
                episode_id = "%s_%d" % (task, seed)
                history = []
                for index, selectors in enumerate(plan(task, goal)):
                    box = _first_box(page, selectors)
                    frame = "frames/%s/step_%d.jpg" % (episode_id, index)
                    target = frames / episode_id / ("step_%d.jpg" % index)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=str(target), type="jpeg", quality=IMAGE_QUALITY)
                    episode = {
                        "episode_id": episode_id,
                        "task": task,
                        "seed": seed,
                        "step": index,
                        "instruction": goal,
                        "history": list(history),
                        "viewport": [viewport["width"], viewport["height"]],
                        "frame": frame,
                        "elements": [{"selector": selectors[-1], "bbox": [
                            box["x"], box["y"], box["width"], box["height"]
                        ]}],
                    }
                    row = row_from_element(episode, episode["elements"][0],
                                           episode["viewport"])
                    rows.append(row)
                    cx, cy = row["target_point"]
                    action = "mouse_click(%d,%d)" % (
                        int(cx * (viewport["width"] - 1)),
                        int(cy * (viewport["height"] - 1)),
                    )
                    obs, _, terminated, truncated, _ = env.step(action)
                    history.append(row["target_action"])
                    if terminated or truncated:
                        break
            finally:
                env.close()
    rows_path = out / "episodes.jsonl"
    digest = write_rows(rows_path, rows)
    print(json.dumps({"rows": len(rows), "sha256": digest}, indent=2))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Collect DOM-box grounding targets (offline or live)."
    )
    parser.add_argument("--config", help="YAML config for workspace defaults")
    parser.add_argument("--workspace-root", help="workspace root for frame checks")
    parser.add_argument("--episodes", help="JSONL of recorded episodes (offline mode)")
    parser.add_argument("--out", help="output JSONL (offline) or directory (live)")
    parser.add_argument("--check-frames", action="store_true",
                        help="verify every frame exists under the workspace root")
    parser.add_argument("--live", action="store_true", help="drive BrowserGym/MiniWoB++")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--seed-lo", type=int, default=6201)
    args = parser.parse_args(argv)

    cfg = load_config(args.config, {"paths.workspace_root": args.workspace_root})
    workspace_root = args.workspace_root or cfg["paths"].get("workspace_root")

    if args.live:
        if not args.out:
            parser.error("--live requires --out")
        collect_live(args)
        return

    if not args.episodes or not args.out:
        parser.error("offline mode requires --episodes and --out")
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("output file already exists: %s" % out)
    rows = collect_rows(
        [json.loads(line) for line in Path(args.episodes).open(encoding="utf8")
         if line.strip()],
        workspace_root=workspace_root,
        check_frames=args.check_frames,
    )
    digest = write_rows(out, rows)
    print(json.dumps({"rows": len(rows), "sha256": digest,
                      "first_target_token": rows[0]["target_token"] if rows else None},
                     indent=2))


if __name__ == "__main__":
    main()
