"""Grounding head entry point and loss plumbing for the V9 ablation.

The reusable pieces live in :mod:`models.grounding`; this module is the small
portable harness around them:

* :func:`build_head` sizes the head from ``model.hidden_size`` in
  ``configs/default.yaml`` and moves it to a :class:`torch.device`.
* :func:`pool_instruction` mean-pools the LM states over text positions, the
  instruction vector the head is conditioned on.
* :func:`grounding_loss` is the exact training objective used by the original
  V9 run: focal heatmap loss over all 320 cells plus L1 offset loss on the
  labelled click cells only.

The original research script hard-coded CUDA and HPC paths; this module takes
the device and every path from config/CLI instead. Run it directly to print the
head contract (and optionally verify a checkpoint)::

    python scripts/gui_v9_grounding.py --config configs/default.yaml

Integration into the SFT loop (``scripts/gui_v6_train.py`` in the research
tree) is one call per batch for click rows that carry ``target_point``::

    head = build_head(cfg, device=device)
    loss, stats = grounding_loss(head, feats, instr_vec, points, device,
                                 lambda_heat=1.0, lambda_off=1.0)
    total = ce_loss + stats["loss"]
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import load_config
from models.grounding import (
    IMAGE_TOKEN_ID, NUM_TOKENS, GroundingHead, build_targets, decode_points,
    focal_loss, point_to_token,
)


def resolve_device(cfg, override=None):
    """Return the configured torch device, falling back to CPU with a notice.

    ``override`` wins over ``device.device``. A configured CUDA device that is
    not available falls back to CPU so the tool stays runnable on CPU-only
    hosts; the fallback is printed rather than silently applied.
    """
    name = override or cfg["device"]["device"]
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[grounding] CUDA requested but unavailable; using CPU", file=sys.stderr)
        return torch.device("cpu")
    return device


def build_head(cfg, device=None, hidden=None):
    """Build :class:`GroundingHead` sized by ``model.hidden_size``.

    ``hidden`` overrides the head's internal width (the original used 256).
    The head input width is always the LM hidden size, because the vision
    features and the pooled instruction are both projected into that space.
    """
    dim = int(cfg["model"]["hidden_size"])
    head = GroundingHead(dim=dim, hidden=int(hidden or 256))
    if device is not None:
        head = head.to(torch.device(device))
    return head


def pool_instruction(hidden, tokens, image_token_id=IMAGE_TOKEN_ID):
    """Mean-pool LM states over text positions.

    ``hidden`` is ``[B, L, D]`` and ``tokens`` is ``[B, L]``. Image-marker and
    padding positions are excluded, matching the original V9 conditioning
    vector. Raises ``ValueError`` if a row has no text positions.
    """
    if hidden.ndim != 3 or tokens.ndim != 2 or hidden.shape[:2] != tokens.shape:
        raise ValueError("expected hidden [B, L, D] and tokens [B, L]")
    text = (tokens != image_token_id) & (tokens != 0)
    counts = text.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("row without text positions")
    vector = (hidden * text.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)
    return vector.to(hidden.dtype)


def grounding_loss(head, feats, instr_vec, points, device, lambda_heat=1.0,
                   lambda_off=1.0, sigma=1.5):
    """Compute the V9 grounding objective.

    ``feats`` is ``[B, 320, D]``, ``instr_vec`` is ``[B, D]`` and ``points`` is
    the per-row ``(token|None, ox, oy)`` triples accepted by
    :func:`models.grounding.build_targets`. Returns ``(loss, stats)`` where
    ``stats`` holds detached floats for logging and the number of labelled
    rows. Offset loss uses only rows that carry a click target, so unlabelled
    rows still contribute heatmap negatives.
    """
    heat_t, off_t, mask = build_targets(points, device, sigma=sigma)
    logits, pred_off = head(feats, instr_vec)
    heat_loss = focal_loss(logits, heat_t)
    labelled = mask.nonzero(as_tuple=False).squeeze(1)
    if int(labelled.numel()):
        tokens = torch.tensor(
            [int(points[int(b)][0]) for b in labelled.tolist()],
            dtype=torch.long, device=pred_off.device,
        )
        offset_loss = F.l1_loss(pred_off[labelled, tokens], off_t[labelled, tokens])
    else:
        offset_loss = pred_off.sum() * 0.0
    loss = float(lambda_heat) * heat_loss + float(lambda_off) * offset_loss
    stats = {
        "loss": float(loss.detach()),
        "heat_loss": float(heat_loss.detach()),
        "offset_loss": float(offset_loss.detach()),
        "labelled": int(labelled.numel()),
    }
    return loss, stats


def _read_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")
    return torch.load(str(path), map_location="cpu", weights_only=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Print the V9 grounding head contract or verify a head checkpoint."
    )
    parser.add_argument("--config", help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--workspace-root", help="workspace root override")
    parser.add_argument("--device", help="torch device override, e.g. cpu or cuda")
    parser.add_argument("--head-hidden", type=int, default=256)
    parser.add_argument("--checkpoint", help="head state dict (.safetensors or .pt)")
    parser.add_argument("--forward-check", action="store_true",
                        help="run one forward pass on zero features")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, {
        "paths.workspace_root": args.workspace_root,
        "device.device": args.device,
    })
    device = resolve_device(cfg, args.device)
    head = build_head(cfg, device=device, hidden=args.head_hidden)
    head.eval()
    report = {
        "tokens": NUM_TOKENS,
        "hidden_size": int(cfg["model"]["hidden_size"]),
        "head_hidden": head.hidden,
        "parameters": sum(p.numel() for p in head.parameters()),
        "trainable": sum(p.numel() for p in head.parameters() if p.requires_grad),
        "device": str(device),
        "image_token_id": IMAGE_TOKEN_ID,
        "checkpoint": None,
        "forward_check": None,
    }

    if args.checkpoint:
        state = _read_checkpoint(args.checkpoint)
        state = {k[len("ground_head."):] if k.startswith("ground_head.") else k: v
                 for k, v in state.items()}
        head.load_state_dict(state, strict=True)
        report["checkpoint"] = str(Path(args.checkpoint))

    if args.forward_check:
        with torch.inference_mode():
            feats = torch.zeros(1, NUM_TOKENS, head.dim, device=device)
            instr = torch.zeros(1, head.dim, device=device)
            heat, offsets = head(feats, instr)
            points, scores = decode_points(heat, offsets)
        report["forward_check"] = {
            "heat_shape": list(heat.shape),
            "offset_shape": list(offsets.shape),
            "point": list(points[0][0]),
            "score": float(scores[0, 0]),
        }

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
