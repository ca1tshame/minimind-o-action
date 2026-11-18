"""V11/V14 GUI SFT trainer (faithful port of ``project/scripts/gui_v6_train.py``).

Recipe reproduced here:

* 63,912,192-parameter Dense MiniMind backbone, strict load of the P1.5/P1.6
  checkpoint, optional V11 action-checkpoint init (projector/compressor/tile_pos);
* frozen SigLIP2 encoder, 5 tiles per screenshot, 320 vision tokens
  (64 -> 64/64/64/64/64 per tile through TileCompressor + tile_pos);
* three weighted streams per optimizer step: new GUI 4 samples (0.75),
  language replay 1 (0.15), old GUI replay 2 (0.10);
* only layers.7 + final norm + projector + compressor + tile_pos train;
* per-action-token CE, class-weighted by action kind
  {click: 0.35, type: 1.0, scroll: 2.0, stop: 2.0};
* gradient accumulation with ``zero_grad`` only at the real accumulation
  boundary, truncated-sample accounting, validation, resumable checkpoints.

The trainer never silently falls back to a random projector: every loaded
checkpoint is validated by :func:`models.vision.load_action_model`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer

from agent.vision_tiles import attach_tile_compression, preprocess_tiles
from configs import (apply_overrides, load_config, require_paths, resolve_device,
                     resolve_paths, workspace_root)
from models.minimind import MiniMindConfig, MiniMindForCausalLM
from models.vision import (ActionWeightError, FrozenSigLIP2, GuiActionModel,
                           action_parameter_report, lm_hidden_forward,
                           load_action_model, load_lm_checkpoint)
from scripts.lib.checkpoint import digest, load_checkpoint, save_checkpoint
from scripts.lib.sft_data import SFTTooLong, collate_sft, encode_record

TRAINABLE_PREFIXES = (
    "lm.model.layers.7.",
    "lm.model.norm.",
    "proj.",
    "compressor.",
    "tile_pos.",
)


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError("empty dataset: %s" % path)
    return records


class GUIStream:
    """Canonical GUI records with frames, cycled in shuffled epochs."""

    def __init__(self, path, frames_dir, tokenizer, n_visual, context, hist_cap,
                 vision_preprocess, seed=0, batch_size=1):
        self.records = read_jsonl(path)
        self.frames_dir = Path(frames_dir) if frames_dir else None
        self.tokenizer = tokenizer
        self.n_visual = int(n_visual)
        self.context = int(context)
        self.hist_cap = int(hist_cap)
        self.vision_preprocess = vision_preprocess
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = -1
        self.order = []
        self.cursor = 0
        self.truncated = 0

    def _frame_path(self, record):
        frame = record.get("frame")
        if not frame:
            raise ValueError("record %s has no frame" % record.get("episode_id"))
        path = Path(frame)
        if not path.is_absolute():
            if self.frames_dir is None:
                raise ValueError("relative frame but no frames dir configured")
            path = self.frames_dir / path
        if not path.is_file():
            raise FileNotFoundError("frame missing: %s" % path)
        return path

    def _encode(self, record):
        history = list(record.get("history") or [])[-self.hist_cap:]
        error = record.get("error_feedback") or ""
        target = record["target_action"] + "<|im_end|>"
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        base = ("<|im_start|>user\n%s\nTask screenshot follows.%s<|vision_end|>\n"
                % (record["instruction"], "<|image_pad|>" * self.n_visual))
        while True:
            text = base
            if history:
                text += "Previous actions: " + "; ".join(history) + ".\n"
            if error:
                text += "Last action failed: %s\n" % error
            text += "<|im_end|>\n<|im_start|>assistant\n"
            ids = [self.tokenizer.bos_token_id] + self.tokenizer(
                text, add_special_tokens=False)["input_ids"]
            if len(ids) + len(target_ids) <= self.context:
                break
            if not history:
                raise SFTTooLong("instruction+target exceeds context %d" % self.context)
            history = history[1:]
        labels = [-100] * len(ids) + list(target_ids)
        ids = ids + list(target_ids)
        image = Image.open(self._frame_path(record)).convert("RGB")
        tiles, _boxes = preprocess_tiles(np.asarray(image, dtype=np.uint8),
                                         self.vision_preprocess)
        return {
            "input_ids": ids,
            "labels": labels,
            "tiles": tiles,
            "kind": record.get("action_kind") or "click",
            "sample_id": "%s:%s" % (record.get("episode_id"), record.get("step")),
        }

    def _reshuffle(self):
        self.epoch += 1
        self.order = list(range(len(self.records)))
        random.Random(self.seed + self.epoch).shuffle(self.order)
        self.cursor = 0

    def next_batch(self):
        batch, tries = [], 0
        while len(batch) < self.batch_size:
            if self.cursor >= len(self.order):
                self._reshuffle()
            index = self.order[self.cursor]
            self.cursor += 1
            try:
                batch.append(self._encode(self.records[index]))
            except SFTTooLong:
                self.truncated += 1
                tries += 1
                if tries > len(self.records) * 2:
                    raise RuntimeError("every record exceeds context %d" % self.context)
        return batch

    def batches(self):
        while True:
            yield self.next_batch()


class LanguageStream:
    """Language replay stream using the validated chat-template encoder."""

    def __init__(self, path, tokenizer, max_length, batch_size=1, seed=0):
        self.records = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = -1
        self.order = []
        self.cursor = 0
        self.truncated = 0

    def _reshuffle(self):
        self.epoch += 1
        self.order = list(range(len(self.records)))
        random.Random(self.seed + 1000 + self.epoch).shuffle(self.order)
        self.cursor = 0

    def next_batch(self):
        samples = []
        while len(samples) < self.batch_size:
            if self.cursor >= len(self.order):
                self._reshuffle()
            record = self.records[self.order[self.cursor]]
            self.cursor += 1
            try:
                samples.append(encode_record(record, self.tokenizer,
                                             max_length=self.max_length,
                                             overflow="truncate_complete_assistant"))
            except SFTTooLong:
                self.truncated += 1
                if self.cursor > len(self.records) * 2:
                    raise RuntimeError("no language sample fits max_length")
        return collate_sft(samples, pad_id=self.tokenizer.pad_token_id or 0)


def pad_gui_batch(batch, pad_id=0):
    width = max(len(item["input_ids"]) for item in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for row, item in enumerate(batch):
        length = len(item["input_ids"])
        ids[row, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
        labels[row, :length] = torch.tensor(item["labels"], dtype=torch.long)
        mask[row, :length] = 1
    tiles = [item["tiles"] for item in batch]
    return {"input_ids": ids, "labels": labels, "mask": mask, "tiles": tiles,
            "kinds": [item["kind"] for item in batch]}


def forward_gui(model, batch, device):
    tokens = batch["input_ids"].to(device)
    mask = batch["mask"].to(device)
    labels = batch["labels"].to(device)
    hidden = model.lm.model.embed_tokens(tokens)
    features = [tiles.to(device) for tiles in batch["tiles"]]
    encoded = []
    for tiles in features:
        feats = model.vision_encoder(tiles)  # (5, 64, 768) Tensor
        # inject() takes one list of media features per sample.
        encoded.append([feats.reshape(-1, feats.shape[-1])])
    hidden = model.inject(tokens, hidden, mask, encoded)
    logits = lm_hidden_forward(model.lm, hidden, mask)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                         labels[:, 1:].reshape(-1), reduction="none", ignore_index=-100)
    ce = ce.view(labels.shape[0], -1)
    counts = (labels[:, 1:] != -100).sum(1).clamp_min(1)
    per_sample = ce.sum(1) / counts
    return per_sample, batch["kinds"]


def forward_language(model, batch, device):
    tokens = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    hidden = model.lm.model.embed_tokens(tokens)
    logits = lm_hidden_forward(model.lm, hidden, mask)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                         labels[:, 1:].reshape(-1), reduction="none", ignore_index=-100)
    counts = (labels[:, 1:] != -100).sum().clamp_min(1)
    return ce.sum() / counts


def build_model(cfg, paths, device, context, vision_tokens):
    with open(Path(paths["model_config"]), "r", encoding="utf-8") as handle:
        model_cfg = json.load(handle)
    model_cfg["max_position_embeddings"] = max(
        int(model_cfg.get("max_position_embeddings") or 0), context + 128)
    lm = MiniMindForCausalLM(MiniMindConfig(**model_cfg))
    base = paths.get("base_checkpoint")
    init = paths.get("init_checkpoint")
    if not base and not init:
        raise FileNotFoundError("set paths.base_checkpoint (P1.5/P1.6) or paths.init_checkpoint")
    load_lm_checkpoint(lm, base or init,
                       expected_parameters=(cfg.get("model") or {}).get("parameters"))
    model = GuiActionModel(lm)
    attach_tile_compression(model, vision_tokens)
    if init:
        load_action_model(model, init)
    else:
        print("WARNING: no init_checkpoint; projector/compressor start from scratch", flush=True)
    encoder = FrozenSigLIP2(paths["vision_encoder_dir"], dtype=torch.float32)
    object.__setattr__(model, "vision_encoder", encoder)
    preprocess_path = Path(paths["vision_encoder_dir"]) / "preprocessor_config.json"
    if preprocess_path.is_file():
        vision_preprocess = json.loads(preprocess_path.read_text(encoding="utf-8"))
    else:
        vision_preprocess = (cfg.get("vision") or {}).get("preprocess") or {
            "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
        }
    for param in model.parameters():
        param.requires_grad_(False)
    for name, param in model.named_parameters():
        if name.startswith(TRAINABLE_PREFIXES):
            param.requires_grad_(True)
    model.to(device)
    model.lm.eval()
    encoder.eval()
    return model, vision_preprocess


def build_optimizer(model, recipe):
    groups = {"lm": [], "proj": [], "compressor": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("proj."):
            groups["proj"].append(param)
        elif name.startswith(("compressor.", "tile_pos.")):
            groups["compressor"].append(param)
        else:
            groups["lm"].append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": groups["lm"], "lr": recipe["learning_rate_lm"]},
            {"params": groups["proj"], "lr": recipe["learning_rate_proj"]},
            {"params": groups["compressor"], "lr": recipe["learning_rate_compressor"]},
        ],
        betas=tuple(recipe["betas"]), eps=recipe["eps"],
        weight_decay=recipe["weight_decay"],
    )
    return optimizer


def cosine_schedule(optimizer, warmup, total):
    def factor(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def validate(model, gui_val, lang_val, device, kind_weights, batches):
    model.eval()
    totals = {"gui": 0.0, "language": 0.0}
    for _ in range(max(1, batches)):
        batch = pad_gui_batch(gui_val.next_batch())
        per, kinds = forward_gui(model, batch, device)
        weights = torch.tensor([kind_weights.get(kind, 1.0) for kind in kinds], device=device)
        totals["gui"] += float((per * weights).sum() / weights.sum())
        if lang_val is not None:
            totals["language"] += float(forward_language(model, lang_val.next_batch(), device))
    count = max(1, batches)
    model.train()
    return {"val_gui_loss": totals["gui"] / count,
            "val_language_loss": totals["language"] / count}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--train-data", default=None)
    parser.add_argument("--val-data", default=None)
    parser.add_argument("--language-data", default=None)
    parser.add_argument("--old-gui-data", default=None)
    parser.add_argument("--frames-dir", default=None)
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--vision-encoder-dir", default=None)
    parser.add_argument("--name", default="gui_sft")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--vision-tokens", type=int, default=None)
    parser.add_argument("--hist-cap", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume-tag", default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--limit-steps", type=int, default=0, help="smoke-test cap")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    overrides = {
        "paths.workspace_root": args.workspace_root,
        "paths.init_checkpoint": args.init_checkpoint,
        "paths.base_checkpoint": args.base_checkpoint,
        "paths.gui_records_train": args.train_data,
        "paths.gui_records_val": args.val_data,
        "paths.language_data": args.language_data,
        "paths.old_gui_data": args.old_gui_data,
        "paths.gui_frames_dir": args.frames_dir,
        "paths.tokenizer_dir": args.tokenizer_dir,
        "paths.vision_encoder_dir": args.vision_encoder_dir,
    }
    apply_overrides(cfg, overrides)
    root = workspace_root(cfg, args.workspace_root)
    paths = resolve_paths(cfg, root)
    if paths.get("model_config"):
        candidate = Path(str(paths["model_config"]))
        if not candidate.is_file():
            repo_candidate = REPO_ROOT / str((cfg.get("paths") or {}).get("model_config"))
            if repo_candidate.is_file():
                paths["model_config"] = repo_candidate
    require_paths(paths, ["tokenizer_dir", "vision_encoder_dir"], kind="dir")
    require_paths(paths, ["gui_records_train"], kind="file")
    recipe_cfg = cfg.get("gui_sft") or {}
    recipe = {
        "steps": int(args.steps or recipe_cfg.get("steps", 700)),
        "context": int(args.context or recipe_cfg.get("context", 512)),
        "vision_tokens": int(args.vision_tokens or recipe_cfg.get("vision_tokens", 320)),
        "hist_cap": int(args.hist_cap or recipe_cfg.get("hist_cap", 6)),
        "batch_new_gui": int(recipe_cfg.get("batch_new_gui", 4)),
        "batch_language": int(recipe_cfg.get("batch_language", 1)),
        "batch_old_gui": int(recipe_cfg.get("batch_old_gui", 2)),
        "stream_weights": recipe_cfg.get("stream_weights") or
        {"new_gui": 0.75, "language": 0.15, "old_gui": 0.10},
        "kind_weights": recipe_cfg.get("kind_weights") or
        {"click": 0.35, "type": 1.0, "scroll": 2.0, "stop": 2.0},
        "learning_rate_lm": float(recipe_cfg.get("learning_rate_lm", 1e-6)),
        "learning_rate_proj": float(recipe_cfg.get("learning_rate_proj", 2e-5)),
        "learning_rate_compressor": float(recipe_cfg.get("learning_rate_compressor", 2e-5)),
        "weight_decay": float(recipe_cfg.get("weight_decay", 0.01)),
        "betas": list(recipe_cfg.get("betas", [0.9, 0.95])),
        "eps": float(recipe_cfg.get("eps", 1e-8)),
        "grad_clip": float(recipe_cfg.get("grad_clip", 1.0)),
        "warmup_steps": int(recipe_cfg.get("warmup_steps", 20)),
        "accumulation": int(recipe_cfg.get("gradient_accumulation_steps", 1)),
        "eval_interval": int(args.eval_interval or recipe_cfg.get("eval_interval", 100)),
        "validation_batches": int(recipe_cfg.get("validation_batches", 8)),
        "checkpoint_interval": int(args.checkpoint_interval
                                   or recipe_cfg.get("checkpoint_interval", 100)),
        "seed": int(args.seed or recipe_cfg.get("seed", 4225)),
    }
    device = torch.device(args.device or resolve_device(cfg))
    seed = recipe["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model, vision_preprocess = build_model(cfg, paths, device, recipe["context"],
                                           recipe["vision_tokens"])
    tokenizer = AutoTokenizer.from_pretrained(str(paths["tokenizer_dir"]),
                                              local_files_only=True)
    frames_dir = paths.get("gui_frames_dir")
    new_gui = GUIStream(paths["gui_records_train"], frames_dir, tokenizer,
                        recipe["vision_tokens"], recipe["context"], recipe["hist_cap"],
                        vision_preprocess, seed, recipe["batch_new_gui"])
    old_gui_path = paths.get("old_gui_data") or paths["gui_records_train"]
    old_gui = GUIStream(old_gui_path, frames_dir, tokenizer, recipe["vision_tokens"],
                        recipe["context"], recipe["hist_cap"], vision_preprocess,
                        seed + 2, recipe["batch_old_gui"])
    lang_path = paths.get("language_data")
    language = (LanguageStream(lang_path, tokenizer, recipe["context"],
                               recipe["batch_language"], seed + 1) if lang_path else None)
    gui_val = (GUIStream(paths["gui_records_val"], frames_dir, tokenizer,
                         recipe["vision_tokens"], recipe["context"], recipe["hist_cap"],
                         vision_preprocess, seed + 3, 1)
               if paths.get("gui_records_val") else None)

    optimizer = build_optimizer(model, recipe)
    total_updates = recipe["steps"] if not args.limit_steps else min(
        recipe["steps"], args.limit_steps)
    scheduler = cosine_schedule(optimizer, recipe["warmup_steps"], total_updates)
    exp_dir = Path(paths.get("experiment_dir") or (root / "outputs" / "experiments"))
    exp_dir = exp_dir / args.name
    contract = {
        "format": "minimind-o-action/gui-sft-v1",
        "context": recipe["context"],
        "vision_tokens": recipe["vision_tokens"],
        "hist_cap": recipe["hist_cap"],
        "kind_weights": recipe["kind_weights"],
        "stream_weights": recipe["stream_weights"],
        "steps": total_updates,
        "seed": seed,
        "init_checkpoint": str(paths.get("init_checkpoint") or ""),
        "base_checkpoint": str(paths.get("base_checkpoint") or ""),
        "train_data_sha256": digest(paths["gui_records_train"]),
        "val_data_sha256": digest(paths["gui_records_val"]) if paths.get("gui_records_val") else None,
        "language_data_sha256": digest(lang_path) if lang_path else None,
        "old_gui_data_sha256": digest(old_gui_path),
    }
    progress = {"step": 0, "epoch": 0, "next_sample": 0, "supervised_tokens": 0,
                "truncated_samples": 0, "order": seed, "accumulation_boundary": True}
    if args.resume_tag:
        progress = load_checkpoint(exp_dir, args.resume_tag, model, optimizer, scheduler,
                                   contract)
    print(json.dumps({"parameters": action_parameter_report(model),
                      "trainable": sum(p.numel() for p in model.parameters()
                                       if p.requires_grad)}, indent=2), flush=True)

    weights = recipe["stream_weights"]
    metrics_path = exp_dir / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    step = int(progress["step"])
    accumulation = max(1, recipe["accumulation"])
    while step < total_updates:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _micro in range(accumulation):
            gui_batch = pad_gui_batch(new_gui.next_batch())
            per, kinds = forward_gui(model, gui_batch, device)
            kw = torch.tensor([recipe["kind_weights"].get(k, 1.0) for k in kinds],
                              device=device)
            gui_loss = (per * kw).sum() / kw.sum()
            loss = weights["new_gui"] * gui_loss
            lang_loss = torch.zeros((), device=device)
            if language is not None:
                lang_loss = forward_language(model, language.next_batch(), device)
                loss = loss + weights["language"] * lang_loss
            old_batch = pad_gui_batch(old_gui.next_batch())
            old_per, old_kinds = forward_gui(model, old_batch, device)
            old_kw = torch.tensor([recipe["kind_weights"].get(k, 1.0) for k in old_kinds],
                                  device=device)
            old_loss = (old_per * old_kw).sum() / old_kw.sum()
            loss = loss + weights["old_gui"] * old_loss
            supervised = int((gui_batch["labels"][:, 1:] != -100).sum())
            supervised += int((old_batch["labels"][:, 1:] != -100).sum())
            (loss / accumulation).backward()
            accum_loss += float(loss)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], recipe["grad_clip"])
        optimizer.step()
        scheduler.step()
        step += 1
        progress.update({
            "step": step, "epoch": new_gui.epoch, "next_sample": new_gui.cursor,
            "supervised_tokens": int(progress["supervised_tokens"]) + supervised,
            "truncated_samples": int(new_gui.truncated + old_gui.truncated +
                                     (language.truncated if language else 0)),
            "order": seed, "accumulation_boundary": True,
        })
        record = {"step": step, "loss": accum_loss / accumulation,
                  "grad_norm": float(grad_norm), "lr": scheduler.get_last_lr()[0],
                  "truncated_samples": progress["truncated_samples"],
                  "seconds": round(time.time() - started, 1)}
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if step % recipe["eval_interval"] == 0 and gui_val is not None:
            val = validate(model, gui_val, language, device, recipe["kind_weights"],
                           recipe["validation_batches"])
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, **val}) + "\n")
            print(json.dumps({"step": step, **val}), flush=True)
        if step % recipe["checkpoint_interval"] == 0 or step == total_updates:
            save_checkpoint(exp_dir, "step_%06d" % step, model, optimizer, scheduler,
                            progress, contract)
    save_checkpoint(exp_dir, "final_%06d" % step, model, optimizer, scheduler, progress,
                    contract)
    print("done: %s" % exp_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
