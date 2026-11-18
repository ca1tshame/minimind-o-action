"""Evaluate the V9 grounding head against the DSL output path.

The point of the V9 ablation is that one checkpoint exposes two independent
output paths and both are measured on the same rows:

* **head path** -- the CenterNet heatmap/offset head over the 320 visual
  patches decodes a click point directly;
* **DSL path** -- the same language backbone greedily generates an
  ``<action>click(x,y)</action>`` string, which is parsed into a point.

The head path only exists for checkpoints that carry ``ground_head.*`` weights.
For headless checkpoints (the V8 control and the V9 DSL baseline) the head
columns are reported as ``n/a`` rather than as zero, because they were never
measured.

Hit = predicted point inside the oracle DOM bounding box. Error is the
Euclidean distance to the oracle bbox centre in viewport-normalised
coordinates. The numbers in ``--result-json`` mode are read from an existing
result file, never recomputed or invented.
"""
import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, load_model
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.vision_tiles import attach_tile_compression, preprocess_tiles
from configs import load_config
from models import (
    IMAGE_ID, IMAGE_TOKEN, IN_DIM, MiniMindConfig, MiniMindForCausalLM,
    FrozenSigLIP2, GuiActionModel,
)
from models.grounding import NUM_TOKENS, GroundingHead, decode_points
from models.vision import lm_hidden_forward
from scripts.gui_v9_grounding import pool_instruction, resolve_device

CLICK = re.compile(r"click\(\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*,\s*"
                   r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*\)")


def parse_click(text):
    """Parse the first ``click(x, y)`` in ``text``; ``None`` if absent/invalid."""
    if not isinstance(text, str):
        return None
    match = CLICK.search(text)
    if not match:
        return None
    try:
        x, y = float(match.group(1)), float(match.group(2))
    except ValueError:
        return None
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return None
    if not (torch.isfinite(torch.tensor(x)) and torch.isfinite(torch.tensor(y))):
        return None
    return x, y


def inside(point, bbox):
    """True when a normalised point lies inside the oracle bbox (inclusive)."""
    if point is None:
        return False
    x, y = point
    return (bbox[0] <= x <= bbox[0] + bbox[2]) and (bbox[1] <= y <= bbox[1] + bbox[3])


def _distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def read_rows(path, limit=0):
    """Read evaluation rows that carry an oracle bbox, point and frame."""
    rows = []
    with Path(path).open(encoding="utf8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("target_point") and row.get("target_bbox") and row.get("frame"):
                rows.append(row)
    return rows[:limit] if limit else rows


def encode_prompt(record, tokenizer, image_tokens):
    """Tokenise the V9 GUI prompt with ``image_tokens`` image markers."""
    history = record.get("history") or []
    history_text = ""
    if history:
        history_text = "Previous actions: " + "; ".join(history[:2]) + ".\n"
    prompt = (
        "<|im_start|>user\n%s\nTask screenshot follows.%s<|vision_end|>\n"
        "%s<|im_end|>\n<|im_start|>assistant\n"
        % (record["instruction"], IMAGE_TOKEN * image_tokens, history_text)
    )
    return [tokenizer.bos_token_id] + tokenizer(
        prompt, add_special_tokens=False
    )["input_ids"]


def _report_table(entries):
    """Format model metrics as a fixed-width table.

    Accuracy is recomputed from hits and n when both are present, so a rate
    stored with three decimals cannot turn 3/80 into 3.70% instead of 3.75%.
    A headless checkpoint (the V8 control and the V9 DSL baseline) is shown as
    ``n/a`` on the head columns: the original harness stored zeros there, but
    that path never ran.
    """
    header = ("model", "n", "head hit", "head rate", "dsl hit", "dsl rate",
              "head err", "dsl err")
    lines = ["%-28s %5s %9s %10s %8s %9s %9s %8s" % header]
    for name, metrics in entries:
        n = metrics.get("n")
        head_available = metrics.get("head_available")
        if head_available is None:
            head_available = bool(
                metrics.get("head_err_mean") is not None
                or metrics.get("head_hit") or metrics.get("head_invalid")
            )
        head_hit = metrics.get("head_hit") if head_available else None

        def rate(hit, stored, available=True):
            if not available:
                return "n/a"
            if hit is not None and n:
                return "%.2f%%" % (100.0 * hit / n)
            if stored is not None:
                return "%.2f%%" % (100.0 * stored)
            return "n/a"

        def number(value):
            return "n/a" if value is None else "%.4f" % value

        lines.append("%-28s %5s %9s %10s %8s %9s %9s %8s" % (
            name, n if n is not None else "n/a",
            head_hit if head_hit is not None else "n/a",
            rate(head_hit, metrics.get("head_hit_rate"), head_available),
            metrics.get("dsl_hit", "n/a"),
            rate(metrics.get("dsl_hit"), metrics.get("dsl_hit_rate")),
            number(metrics.get("head_err_mean") if head_available else None),
            number(metrics.get("dsl_err_mean")),
        ))
    return "\n".join(lines)


def report_result_json(path):
    """Print head vs DSL accuracy read from an existing result JSON.

    Returns the parsed structure so callers can consume it programmatically.
    """
    result = json.loads(Path(path).read_text(encoding="utf8"))
    entries = [(name, metrics) for name, metrics in result.items()
               if isinstance(metrics, dict)]
    print(_report_table(entries))
    print("head path: heatmap/offset head; dsl path: greedy DSL text; "
          "both from the same checkpoint when the result file includes both.")
    return result


def _state_keys(path):
    return set(load_file(str(path), device="cpu").keys())


def build_model(cfg, args, image_tokens):
    """Instantiate the action model and load one checkpoint for both paths."""
    model_cfg = json.loads(
        (ROOT / cfg["paths"]["model_config"]).read_text(encoding="utf8")
    )
    lm = MiniMindForCausalLM(MiniMindConfig(**model_cfg))
    model = GuiActionModel(lm)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(str(checkpoint))
    keys = _state_keys(checkpoint)
    has_compressor = any(key.startswith("compressor.") for key in keys)
    has_head = any(key.startswith("ground_head.") for key in keys)
    if args.require_head and not has_head:
        raise ValueError("checkpoint has no ground_head.* weights")
    if has_compressor:
        attach_tile_compression(model, image_tokens)
    elif not args.no_compressor:
        print("[gui_v9_ground_eval] checkpoint has no compressor; "
              "using the plain projector inject", file=sys.stderr)
    if has_head:
        if lm.config.hidden_size != IN_DIM:
            raise ValueError("grounding head expects hidden_size == in_dim (%d)" % IN_DIM)
        model.ground_head = GroundingHead(lm.config.hidden_size)
    missing, unexpected = load_model(model, str(checkpoint), strict=False, device="cpu")
    unexpected = [key for key in unexpected if not key.startswith("ground_head.")]
    allowed = {key for key in missing if key.startswith("ground_head.")}
    if unexpected or set(missing) - allowed:
        raise ValueError("checkpoint mismatch: missing=%s unexpected=%s" % (
            sorted(set(missing) - allowed)[:5], unexpected[:5]))
    return model


def _frame_path(record, workspace_root, frames_dir):
    frame = str(record["frame"]).replace("\\", "/")
    if Path(frame).is_absolute() or ".." in Path(frame).parts:
        raise ValueError("frame must be workspace-relative: %r" % frame)
    candidates = []
    if frames_dir:
        candidates.append(Path(frames_dir) / frame)
    candidates.append(Path(workspace_root) / frame)
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size:
            return candidate
    raise FileNotFoundError("frame not found: %s" % frame)


def evaluate(cfg, args):
    """Run both output paths on every row and return the metric dict."""
    from PIL import Image

    rows = read_rows(args.rows, args.limit)
    if not rows:
        raise ValueError("no rows with target_bbox/target_point in %s" % args.rows)
    device = resolve_device(cfg, args.device)
    image_tokens = int(args.vision_tokens or cfg["eval"]["vision_tokens"])
    if image_tokens != NUM_TOKENS:
        raise ValueError("grounding evaluation requires %d visual tokens" % NUM_TOKENS)
    model = build_model(cfg, args, image_tokens)
    head = getattr(model, "ground_head", None)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    encoder = FrozenSigLIP2(cfg["paths"]["vision_encoder_dir"]).to(device).eval()
    preprocess = cfg["vision"].get("preprocess") or {
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
    }
    tokenizer_dir = cfg["paths"]["tokenizer_dir"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.convert_tokens_to_ids(IMAGE_TOKEN) != IMAGE_ID:
        raise ValueError("image token id mismatch")
    workspace_root = cfg["paths"].get("workspace_root") or str(ROOT)
    frames_dir = args.frames_dir or cfg["paths"].get("gui_frames_dir")

    metrics = {
        "n": len(rows),
        "head_hit": 0 if head is not None else None,
        "head_err": [],
        "dsl_hit": 0,
        "dsl_err": [],
        "head_invalid": 0 if head is not None else None,
        "dsl_invalid": 0,
        "small_n": 0,
        "small_hit": 0,
        "head_available": head is not None,
    }
    started = time.time()
    for row in rows:
        raw = _frame_path(row, workspace_root, frames_dir).read_bytes()
        image = Image.open(io.BytesIO(raw))
        pixels, _ = preprocess_tiles(image, preprocess)
        pixels = pixels.to(device)
        ids = encode_prompt(row, tokenizer, image_tokens)
        if len(ids) > model.lm.config.max_position_embeddings:
            raise ValueError("prompt exceeds the model context")
        tokens = torch.tensor([ids], device=device)
        mask = torch.ones_like(tokens)
        with torch.inference_mode():
            features = encoder(pixel_values=pixels).last_hidden_state
            features = features.reshape(1, NUM_TOKENS, IN_DIM).float()
            hidden = model.inject(
                tokens, model.lm.model.embed_tokens(tokens), mask, [[features[0]]]
            )
            instruction = pool_instruction(hidden, tokens)
            point = None
            if head is not None:
                heat, offsets = head(features, instruction)
                points, _ = decode_points(heat, offsets)
                point = points[0][0]

            generated = []
            step_hidden, step_tokens, step_mask = hidden, tokens, mask
            for _ in range(args.max_new):
                logits = lm_hidden_forward(model.lm, step_hidden, step_mask)
                token = int(logits[0, -1].float().argmax())
                if token in (tokenizer.eos_token_id, 2):
                    break
                generated.append(token)
                step_tokens = torch.cat(
                    [step_tokens, torch.tensor([[token]], device=device)], dim=1
                )
                step_mask = torch.ones_like(step_tokens)
                step_hidden = torch.cat(
                    [step_hidden, model.lm.model.embed_tokens(step_tokens[:, -1:])],
                    dim=1,
                )
                text = tokenizer.decode(generated, skip_special_tokens=False)
                if CLICK.search(text):
                    break
        dsl = parse_click(tokenizer.decode(generated, skip_special_tokens=False))
        bbox, target = row["target_bbox"], row["target_point"]
        if head is not None:
            if point is None:
                metrics["head_invalid"] += 1
            else:
                metrics["head_hit"] += int(inside(point, bbox))
                metrics["head_err"].append(_distance(point, target))
                if row.get("small_target"):
                    metrics["small_n"] += 1
                    metrics["small_hit"] += int(inside(point, bbox))
        if dsl is None:
            metrics["dsl_invalid"] += 1
        else:
            metrics["dsl_hit"] += int(inside(dsl, bbox))
            metrics["dsl_err"].append(_distance(dsl, target))

    head_errors = metrics.pop("head_err")
    dsl_errors = metrics.pop("dsl_err")
    metrics["head_err_mean"] = (
        round(sum(head_errors) / len(head_errors), 4) if head_errors else None
    )
    metrics["dsl_err_mean"] = (
        round(sum(dsl_errors) / len(dsl_errors), 4) if dsl_errors else None
    )
    metrics["head_hit_rate"] = (
        round(metrics["head_hit"] / len(rows), 3) if head is not None else None
    )
    metrics["dsl_hit_rate"] = round(metrics["dsl_hit"] / len(rows), 3)
    metrics["small_hit_rate"] = (
        round(metrics["small_hit"] / metrics["small_n"], 3)
        if metrics["small_n"] else None
    )
    metrics["seconds"] = round(time.time() - started, 1)
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["vision_tokens"] = image_tokens
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate the V9 grounding head and/or report head-vs-DSL results."
    )
    parser.add_argument("--config", help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--workspace-root")
    parser.add_argument("--rows", help="JSONL rows with target_bbox/target_point")
    parser.add_argument("--checkpoint", help="action-model checkpoint (.safetensors)")
    parser.add_argument("--name", help="result key (default: checkpoint directory name)")
    parser.add_argument("--out", help="write result JSON here")
    parser.add_argument("--result-json", help="report an existing result JSON and exit")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", help="torch device override")
    parser.add_argument("--vision-tokens", type=int)
    parser.add_argument("--max-new", type=int, default=24)
    parser.add_argument("--tokenizer-dir")
    parser.add_argument("--vision-encoder-dir")
    parser.add_argument("--frames-dir")
    parser.add_argument("--require-head", action="store_true")
    parser.add_argument("--no-compressor", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.result_json and not args.checkpoint:
        report_result_json(args.result_json)
        return

    cfg = load_config(args.config, {
        "paths.workspace_root": args.workspace_root,
        "paths.tokenizer_dir": args.tokenizer_dir,
        "paths.vision_encoder_dir": args.vision_encoder_dir,
        "device.device": args.device,
    })
    for key in ("tokenizer_dir", "vision_encoder_dir", "model_config"):
        if not cfg["paths"].get(key):
            raise SystemExit("config is missing paths.%s; pass it on the CLI" % key)
    if not args.rows:
        raise SystemExit("--rows is required")
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required (or use --result-json)")
    metrics = evaluate(cfg, args)
    name = args.name or Path(args.checkpoint).parent.name
    result = {name: metrics}
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out)
        if out.exists() and not args.overwrite:
            raise FileExistsError("result exists: %s (pass --overwrite)" % out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf8")
    print(_report_table([(name, metrics)]))
    print(text, end="")


if __name__ == "__main__":
    main()
