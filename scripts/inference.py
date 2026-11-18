"""Formal MiniMind-O Action GUI inference entry point.

Loads a V14 action checkpoint exactly as trained (frozen SigLIP2 -> 5 tiles ->
TileCompressor/tile_pos -> projector -> MiniMind), assembles the token-aware
prompt and runs real BrowserGym episodes.

Example::

    python scripts/inference.py --workspace-root /data/minimind \\
        --checkpoint checkpoints/gui_v14/payload/model.safetensors \\
        --task click-dialog --seed 20001 --record outputs/demo/episodes.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image
import torch

from agent.ctx_pack import ContextOverflow, assemble_prompt, count_tokens
from agent.dsl import parse_action, to_browsergym
from agent.environment import configure_browser, run_episode
from agent.vision_tiles import preprocess_tiles
from configs import (REPO_ROOT as CONFIG_ROOT, load_config, require_paths,
                     resolve_device, resolve_paths, workspace_root)
from models.minimind import MiniMindConfig, MiniMindForCausalLM
from agent.vision_tiles import attach_tile_compression
from models.vision import (FrozenSigLIP2, GuiActionModel,
                           action_parameter_report, lm_hidden_forward,
                           load_action_model, load_lm_checkpoint, preprocess_rgb)

DEFAULT_EOS = 2


class GUIPolicy:
    """Single-step policy: screenshot + prompt -> generated action DSL text."""

    def __init__(self, model, encoder, tokenizer, device="cpu", context=512,
                 max_new=48, dtype=torch.bfloat16, max_position=None,
                 vision_preprocess=None):
        self.model = model
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.context = int(context)
        self.max_new = int(max_new)
        self.dtype = dtype
        self.max_position = int(max_position or model.lm.config.max_position_embeddings)
        self.vision_preprocess = vision_preprocess or {
            "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
        }
        self.bos = int(getattr(tokenizer, "bos_token_id", 1) or 1)
        self.eos = int(getattr(tokenizer, "eos_token_id", DEFAULT_EOS) or DEFAULT_EOS)

    def encode_prompt(self, prompt):
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        return [self.bos] + list(ids)

    def check_budget(self, ids):
        if len(ids) + self.max_new > self.context:
            raise ContextOverflow(
                "prompt has %d tokens + %d reserved > context %d"
                % (len(ids), self.max_new, self.context)
            )
        if len(ids) + self.max_new > self.max_position:
            raise ContextOverflow(
                "prompt exceeds the model's %d position embeddings" % self.max_position
            )

    @torch.no_grad()
    def predict(self, screen_rgb, prompt):
        ids = self.encode_prompt(prompt)
        self.check_budget(ids)
        image = screen_rgb
        if isinstance(image, np.ndarray):
            image = Image.fromarray(np.asarray(image, dtype=np.uint8))
        tiles, _boxes = preprocess_tiles(image, self.vision_preprocess)
        tiles = tiles.to(self.device)
        features = self.encoder(tiles)               # (5, 64, 768) Tensor, not an object
        features = features.reshape(1, -1, features.shape[-1])
        tokens = torch.tensor([ids], dtype=torch.long, device=self.device)
        hidden = self.model.lm.model.embed_tokens(tokens)
        mask = torch.ones_like(tokens)
        hidden = self.model.inject(tokens, hidden, mask, [[features[0]]])
        generated = []
        for _ in range(self.max_new):
            logits = lm_hidden_forward(self.model.lm, hidden, mask)
            next_id = int(torch.argmax(logits[0, -1]).item())
            if next_id == self.eos or next_id == 0:
                break
            generated.append(next_id)
            step = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
            hidden = torch.cat([hidden, self.model.lm.model.embed_tokens(step)], dim=1)
            mask = torch.ones((1, hidden.shape[1]), dtype=torch.long, device=self.device)
        return self.tokenizer.decode(generated)


def _config_path(value):
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    for base in (CONFIG_ROOT, Path.cwd()):
        candidate = base / path
        if candidate.is_file():
            return candidate
    return CONFIG_ROOT / path


def build_predictor(cfg, workspace=None, checkpoint=None, device=None):
    """Build the full V14 action model with a validated checkpoint load."""
    root = Path(workspace).resolve() if workspace else workspace_root(cfg)
    paths = resolve_paths(cfg, root)
    for key in ("tokenizer_dir", "vision_encoder_dir"):
        require_paths(paths, [key], kind="dir")
    checkpoint = checkpoint or paths.get("checkpoint")
    if not checkpoint:
        raise FileNotFoundError("no --checkpoint given and paths.checkpoint is null")
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError("checkpoint not found: %s" % checkpoint)
    model_cfg_path = _config_path((cfg.get("paths") or {}).get("model_config")
                                  or "configs/model_config.json")
    with open(model_cfg_path, "r", encoding="utf-8") as handle:
        model_cfg = json.load(handle)
    device = device or resolve_device(cfg)

    lm = MiniMindForCausalLM(MiniMindConfig(**model_cfg))
    load_lm_checkpoint(lm, checkpoint, expected_parameters=(cfg.get("model") or {}).get("parameters"))
    model = GuiActionModel(lm, in_dim=(cfg.get("vision") or {}).get("in_dim", 768))
    vision_tokens = int((cfg.get("eval") or {}).get("vision_tokens",
                                                    (cfg.get("vision") or {}).get("tokens_per_image", 320)))
    attach_tile_compression(model, vision_tokens)
    report = load_action_model(model, checkpoint)
    encoder = FrozenSigLIP2(paths["vision_encoder_dir"])
    model = model.to(device).eval()
    encoder = encoder.to(device)
    tokenizer_dir = Path(paths["tokenizer_dir"])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    eval_cfg = cfg.get("eval") or {}
    policy = GUIPolicy(
        model, encoder, tokenizer, device=device,
        context=int(eval_cfg.get("context", 512)), max_new=int(eval_cfg.get("max_new", 48)),
        dtype=torch.bfloat16 if (cfg.get("device") or {}).get("precision") == "bf16" else torch.float32,
        vision_preprocess=(cfg.get("vision") or {}).get("preprocess"),
    )
    policy.load_report = report
    policy.parameter_report = action_parameter_report(model)
    return policy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--seed", type=int, default=20001)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-new", type=int, default=None)
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--hist-used", type=int, default=None)
    parser.add_argument("--record", default=None, help="episode JSONL output path")
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--dump-load-report", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    overrides = {"paths.workspace_root": args.workspace_root}
    from configs import apply_overrides

    apply_overrides(cfg, overrides)
    root = workspace_root(cfg, args.workspace_root)
    resolved = resolve_paths(cfg, root)
    for key in ("miniwob_url", "playwright_browsers_path"):
        if resolved.get(key):
            cfg.setdefault("paths", {})[key] = resolved[key]
    configure_browser(cfg)
    policy = build_predictor(cfg, workspace=args.workspace_root,
                             checkpoint=args.checkpoint, device=args.device)
    if args.dump_load_report:
        print(json.dumps(policy.load_report, indent=2, default=str))
    task = args.task or (cfg.get("inference") or {}).get("default_task", "click-dialog")
    trace = run_episode(
        policy, task, cfg, seed=args.seed, headless=not args.no_headless,
        max_steps=args.max_steps, max_new_tokens=args.max_new, context=args.context,
        hist_used=args.hist_used, record_path=args.record,
    )
    print(json.dumps({k: v for k, v in trace.items() if k != "steps"}, indent=2))
    return 0 if trace.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
