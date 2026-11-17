"""GUI Action SFT: frozen SigLIP2 + trainable projector + Full-SFT backbone.

Visual features replace 64 image-marker tokens; loss is computed only on the
assistant action span. The frozen encoder lives outside nn.Module children and
is never included in the checkpoint state_dict.
"""
import argparse
import json
import math
import os
import random
import signal
import sys
import time
import uuid
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_model
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import load_config
from models import (
    MiniMindConfig, MiniMindForCausalLM, GuiActionModel,
    FrozenSigLIP2, IMAGE_TOKEN, IMAGE_ID, TOKENS_PER_IMAGE,
)
from models.vision import preprocess_rgb, lm_hidden_forward
from scripts.lib.checkpoint import digest, save_checkpoint, load_checkpoint


def encode_record(record, tokenizer, max_length, image_tokens):
    history = ""
    if record.get("history"):
        history = "Previous actions: " + "; ".join(record["history"]) + ".\n"
    prompt = (
        "<|im_start|>user\n%s\nTask screenshot follows.%s<|vision_end|>\n"
        "%s<|im_end|>\n<|im_start|>assistant\n"
        % (record["instruction"], IMAGE_TOKEN * image_tokens, history)
    )
    target = record["target_action"] + "<|im_end|>"
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    ids = [tokenizer.bos_token_id] + prompt_ids + target_ids
    if len(ids) > max_length:
        raise ValueError("record exceeds max_length; refusing to truncate target")
    labels = [-100] * (len(ids) - len(target_ids)) + list(target_ids)
    runs = sum(1 for i, token in enumerate(ids) if token == IMAGE_ID and (i == 0 or ids[i - 1] != IMAGE_ID))
    if runs != 1:
        raise ValueError("expected exactly 1 image marker run")
    return {"input_ids": ids, "labels": labels, "frame": record["frame"]}


def build_model(cfg):
    paths = cfg["paths"]
    model_cfg = json.loads((ROOT / paths["model_config"]).read_text(encoding="utf8"))
    lm = MiniMindForCausalLM(MiniMindConfig(**model_cfg))
    base = Path(paths["base_checkpoint"])
    missing, unexpected = load_model(lm, str(base), strict=True, device="cpu")
    if missing or unexpected:
        raise ValueError("strict lm load failed")
    model = GuiActionModel(lm)
    encoder = FrozenSigLIP2(paths["vision_encoder_dir"])
    object.__setattr__(model, "vision_encoder", encoder)
    object.__setattr__(model, "vision_preprocess", cfg["vision"].get("preprocess", {
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
    }))
    return model


def collate(encoded, image_tokens, device):
    width = max(len(x["input_ids"]) for x in encoded)
    ii = torch.stack([
        torch.nn.functional.pad(torch.tensor(x["input_ids"]), (0, width - len(x["input_ids"])), value=0)
        for x in encoded
    ]).to(device)
    ll = torch.stack([
        torch.nn.functional.pad(torch.tensor(x["labels"]), (0, width - len(x["labels"])), value=-100)
        for x in encoded
    ]).to(device)
    mask = (ii != 0).long()
    return ii, ll, mask


def main(argv=None):
    parser = argparse.ArgumentParser(description="GUI action SFT (explicit execution).")
    parser.add_argument("--config")
    parser.add_argument("--workspace-root")
    parser.add_argument("--records-train")
    parser.add_argument("--records-val")
    parser.add_argument("--frames-dir")
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--vision-encoder-dir")
    parser.add_argument("--tokenizer-dir")
    parser.add_argument("--experiment")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--execute-training", action="store_true")
    parser.add_argument("--resume-tag")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, {
        "paths.workspace_root": args.workspace_root,
        "paths.gui_records_train": args.records_train,
        "paths.gui_records_val": args.records_val,
        "paths.gui_frames_dir": args.frames_dir,
        "paths.base_checkpoint": args.base_checkpoint,
        "paths.vision_encoder_dir": args.vision_encoder_dir,
        "paths.tokenizer_dir": args.tokenizer_dir,
    })
    paths, gcfg = cfg["paths"], cfg["gui_sft"]
    if args.check:
        print("GUI_CONFIG_VALID_TRAINING_NOT_EXECUTED")
        return
    if not args.execute_training:
        raise PermissionError("pass --check or --execute-training")

    root = Path(paths["workspace_root"])
    train = [json.loads(line) for line in Path(paths["gui_records_train"]).open(encoding="utf8")]
    val = [json.loads(line) for line in Path(paths["gui_records_val"]).open(encoding="utf8")]
    frames_dir = Path(paths["gui_frames_dir"])
    tokenizer_dir = Path(paths["tokenizer_dir"])
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, trust_remote_code=False)
    if tokenizer.convert_tokens_to_ids(IMAGE_TOKEN) != IMAGE_ID:
        raise ValueError("image token id mismatch")

    model = build_model(cfg)
    device = torch.device(cfg["device"]["device"])
    encoder = model.vision_encoder
    encoder.to(device).eval()
    model.proj.to(device)
    model.lm.to(device)
    vision_preprocess = model.vision_preprocess
    bf16 = cfg["device"]["precision"] == "bf16" and device.type == "cuda"
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if bf16 else nullcontext

    groups = [
        {"params": list(model.proj.parameters()), "lr": gcfg["learning_rate_proj"]},
        {"params": list(model.lm.parameters()), "lr": gcfg["learning_rate_lm"]},
    ]
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=gcfg["weight_decay"])

    def scale(step):
        if step < gcfg["warmup_steps"]:
            return float(step + 1) / gcfg["warmup_steps"]
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(
            math.pi * min(1.0, (step - gcfg["warmup_steps"]) /
                          max(1, gcfg["max_updates"] - gcfg["warmup_steps"]))
        ))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    experiment = Path(args.experiment or root / paths["experiment_dir"] / "gui_sft")
    checkpoints = experiment / "checkpoints"
    if args.resume_tag:
        if not experiment.is_dir():
            raise FileNotFoundError("experiment missing")
    else:
        if experiment.exists():
            raise FileExistsError("experiment occupied")
        experiment.mkdir(parents=True)
        checkpoints.mkdir()

    seed = gcfg.get("seed", cfg["device"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    contract = {"config": gcfg, "frames_dir": str(frames_dir), "torch_version": str(torch.__version__)}
    progress = {"step": 0, "epoch": 0, "order": [], "next_sample": 0,
                "supervised_tokens": 0, "accumulation_boundary": True}
    if args.resume_tag:
        progress = load_checkpoint(checkpoints, args.resume_tag, model, optimizer, scheduler, contract)

    stop = {"r": False}
    def halt(signum, frame):
        stop["r"] = True
    for sig in ("SIGTERM", "SIGINT", "SIGUSR1"):
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), halt)
    deadline = time.monotonic() + gcfg["wall_seconds"] - gcfg["reserve_seconds"]
    image_tokens = cfg["vision"]["tokens_per_image"]

    def featurize(records):
        features = []
        for record in records:
            raw = (frames_dir / record["frame"]).read_bytes()
            pixels = preprocess_rgb(raw, vision_preprocess).to(device)
            with torch.no_grad():
                feature = encoder(pixel_values=pixels).last_hidden_state[0]
            features.append(feature.detach().float().cpu())
        return [f.to(device) for f in features]

    log_path = experiment / "metrics.jsonl"
    def log(obj):
        with log_path.open("a", encoding="utf8") as handle:
            handle.write(json.dumps(obj, ensure_ascii=False) + "\n")

    while progress["step"] < gcfg["max_updates"] and progress["epoch"] < gcfg["epochs"]:
        if stop["r"] or time.monotonic() >= deadline:
            break
        if not progress["order"]:
            order = list(range(len(train)))
            random.Random(seed + progress["epoch"]).shuffle(order)
            progress["order"] = order
        order = progress["order"]
        pos = progress["next_sample"]
        if pos >= len(order):
            progress.update(epoch=progress["epoch"] + 1, next_sample=0, order=[])
            continue
        chunk_ids = order[pos:pos + gcfg["batch_size"]]
        chunk = [train[i] for i in chunk_ids]
        encoded = [encode_record(record, tokenizer, gcfg["max_length"], image_tokens) for record in chunk]
        ii, ll, mask = collate(encoded, image_tokens, device)
        features = featurize(chunk)
        hidden = model.lm.model.embed_tokens(ii)
        hidden = model.inject(ii, hidden, mask, [[f] for f in features])
        with amp():
            logits = lm_hidden_forward(model.lm, hidden, mask).float()
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), ll[:, 1:].reshape(-1), ignore_index=-100
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.proj.parameters()) + list(model.lm.parameters()), gcfg["grad_clip"]
        )
        optimizer.step()
        scheduler.step()
        progress.update(
            step=progress["step"] + 1, next_sample=pos + len(chunk_ids),
            supervised_tokens=progress["supervised_tokens"] + int(ll[:, 1:].ne(-100).sum()),
        )
        if progress["step"] == 1 or progress["step"] % 10 == 0:
            log({"type": "train", "step": progress["step"], "loss": float(loss.detach())})
        if progress["step"] % gcfg["checkpoint_interval"] == 0:
            tag = "step_%08d_%s" % (progress["step"], uuid.uuid4().hex[:8])
            save_checkpoint(checkpoints, tag, model, optimizer, scheduler, progress, contract)

    tag = "final_%08d" % progress["step"]
    save_checkpoint(checkpoints, tag, model, optimizer, scheduler, progress, contract)
    print("GUI_SFT_DONE", "step", progress["step"], "tag", tag)


if __name__ == "__main__":
    main()
