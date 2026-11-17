"""Full SFT for the language backbone (746,672-row validated subset).

Auditable SFT trainer with explicit execution gating. Data and checkpoint paths
come from configuration; optional SHA-256 pins enforce frozen datasets when set.
"""
import argparse
import hashlib
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
from models import MiniMindConfig, MiniMindForCausalLM
from scripts.lib.checkpoint import digest, save_checkpoint, load_checkpoint
from scripts.lib.sft_data import (
    LanguageSFTDataset, build_index, collate_sft,
)


def load_backbone(cfg, tokenizer_dir, config_path, base_checkpoint, sequence_length):
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir), local_files_only=True, trust_remote_code=False
    )
    if len(tokenizer) != 6400 or (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) != (1, 2, 0):
        raise ValueError("tokenizer contract mismatch")
    model_cfg = json.loads(Path(config_path).read_text(encoding="utf8"))
    model_cfg["max_position_embeddings"] = sequence_length
    model = MiniMindForCausalLM(MiniMindConfig(**model_cfg))
    missing, unexpected = load_model(model, str(base_checkpoint), strict=True, device="cpu")
    if missing or unexpected:
        raise ValueError("strict weight load failed")
    if model.model.embed_tokens.weight is not model.lm_head.weight:
        raise ValueError("tied embedding contract failed")
    if sum(p.numel() for p in model.parameters()) != 63912192:
        raise ValueError("model parameter count mismatch")
    return tokenizer, model


def evaluate(model, dataset, batch_size, device, amp, limit=0):
    model.eval()
    weighted, count, truncated = 0.0, 0, 0
    samples = dataset if not limit else [dataset[i] for i in range(min(limit, len(dataset)))]
    with torch.inference_mode():
        for i in range(0, len(samples), batch_size):
            batch = collate_sft(samples[i:i + batch_size])
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            mask = batch["attention_mask"].to(device)
            with amp():
                out = model(x, labels=y, attention_mask=mask)
            if not torch.isfinite(out.loss):
                raise FloatingPointError("nonfinite validation")
            weighted += float(out.loss) * batch["supervised_tokens"]
            count += batch["supervised_tokens"]
            truncated += batch["truncated_samples"]
    if not count:
        raise ValueError("no supervised validation tokens")
    model.train()
    nll = weighted / count
    return {"nll": nll, "ppl": math.exp(nll), "supervised_tokens": count,
            "samples": len(samples), "truncated_samples": truncated}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Full SFT for the MiniMind backbone (explicit execution).")
    parser.add_argument("--config")
    parser.add_argument("--workspace-root")
    parser.add_argument("--data")
    parser.add_argument("--tokenizer-dir")
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--experiment")
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--resume-tag")
    parser.add_argument("--execute-training", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, {
        "paths.workspace_root": args.workspace_root,
        "paths.sft_data": args.data,
        "paths.tokenizer_dir": args.tokenizer_dir,
        "paths.base_checkpoint": args.base_checkpoint,
    })
    paths, scfg = cfg["paths"], cfg["sft"]
    if args.check_config:
        print("CONFIG_VALID_TRAINING_NOT_EXECUTED")
        return
    if not args.execute_training:
        raise PermissionError("pass --check-config to validate, or --execute-training to train")

    root = Path(paths["workspace_root"])
    data = Path(paths["sft_data"])
    tokenizer_dir = Path(paths["tokenizer_dir"])
    base_checkpoint = Path(paths["base_checkpoint"])
    config_path = ROOT / paths["model_config"]

    index = build_index(data, scfg.get("seed", 42), scfg["val_percent"])
    expected = scfg.get("data_sha256")
    if expected and index["data_sha256"] != expected:
        raise ValueError("SFT data hash mismatch")
    tokenizer, model = load_backbone(
        cfg, tokenizer_dir, config_path, base_checkpoint, scfg["max_length"]
    )

    device = torch.device(cfg["device"]["device"])
    if device.type not in ("cpu", "cuda"):
        raise ValueError("unsupported device")
    amp = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if scfg.get("precision", cfg["device"]["precision"]) == "bf16"
        else nullcontext
    )
    if amp is not nullcontext and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bf16 requires a supported CUDA device")

    experiment = Path(args.experiment or root / paths["experiment_dir"] / "full_sft")
    checkpoints = experiment / "checkpoints"
    if args.resume_tag:
        if not experiment.is_dir():
            raise FileNotFoundError("resume experiment missing")
    else:
        if experiment.exists():
            raise FileExistsError("experiment target occupied")
        experiment.mkdir(parents=True)
        checkpoints.mkdir()
        (experiment / "step_journal.jsonl").write_text("", encoding="utf8")

    seed = scfg.get("seed", cfg["device"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=scfg["learning_rate"], betas=(0.9, 0.999),
        eps=1e-8, weight_decay=scfg["weight_decay"],
    )

    def lr_scale(step):
        if step < scfg["warmup_steps"]:
            return float(step + 1) / scfg["warmup_steps"]
        return 0.1 + 0.9 * 0.5 * (
            1 + math.cos(math.pi * min(1.0, (step - scfg["warmup_steps"]) /
                                       max(1, scfg["max_updates"] - scfg["warmup_steps"])))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    train = LanguageSFTDataset(data, index["splits"]["train"], tokenizer, scfg["max_length"], scfg["overflow_policy"])
    val_index = index["splits"]["validation"][:scfg["validation_max_samples"]]
    valid = LanguageSFTDataset(data, val_index, tokenizer, scfg["max_length"], scfg["overflow_policy"])
    contract = {
        "data_sha256": index["data_sha256"],
        "config": scfg, "torch_version": str(torch.__version__),
    }
    progress = {
        "step": 0, "epoch": 0, "next_sample": 0,
        "supervised_tokens": 0, "truncated_samples": 0,
        "order": [], "accumulation_boundary": True,
    }
    if args.resume_tag:
        progress = load_checkpoint(checkpoints, args.resume_tag, model, optimizer, scheduler, contract)

    stop = {"requested": False}
    def halt(signum, frame):
        stop["requested"] = True
    signal.signal(signal.SIGTERM, halt)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, halt)
    start = time.monotonic()

    def needs_stop():
        return stop["requested"] or time.monotonic() - start >= scfg["wall_seconds"] - scfg["reserve_seconds"]

    def log(obj):
        with (experiment / "metrics.jsonl").open("a", encoding="utf8") as handle:
            handle.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def save(reason):
        tag = "stop_%08d_%s" % (progress["step"], uuid.uuid4().hex[:8])
        save_checkpoint(checkpoints, tag, model, optimizer, scheduler, progress, contract)
        log({"type": "checkpoint", "step": progress["step"], "tag": tag, "reason": reason})

    while progress["epoch"] < scfg["epochs"] and progress["step"] < scfg["max_updates"]:
        if needs_stop():
            save("stopped")
            return
        if not progress["order"]:
            order = list(range(len(train)))
            random.Random(seed + progress["epoch"]).shuffle(order)
            progress["order"] = order
        order = progress["order"]
        pos = progress["next_sample"]
        if pos >= len(order):
            progress.update(epoch=progress["epoch"] + 1, next_sample=0, order=[])
            continue
        end = min(pos + scfg["batch_size"] * scfg["gradient_accumulation_steps"], len(order))
        micro = [collate_sft([train[order[k]] for k in range(j, min(j + scfg["batch_size"], end))])
                 for j in range(pos, end, scfg["batch_size"])]
        effective = sum(b["supervised_tokens"] for b in micro)
        if effective <= 0:
            raise ValueError("no supervised tokens")
        optimizer.zero_grad(set_to_none=True)
        weighted = 0.0
        for batch in micro:
            with amp():
                out = model(batch["input_ids"].to(device), labels=batch["labels"].to(device),
                            attention_mask=batch["attention_mask"].to(device))
            (out.loss * (batch["supervised_tokens"] / effective)).backward()
            weighted += float(out.loss.detach()) * batch["supervised_tokens"]
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), scfg["grad_clip"])
        optimizer.step()
        scheduler.step()
        progress.update(
            step=progress["step"] + 1, next_sample=end,
            supervised_tokens=progress["supervised_tokens"] + effective,
            truncated_samples=progress["truncated_samples"] + sum(b["truncated_samples"] for b in micro),
        )
        log({"type": "train", "step": progress["step"], "loss": weighted / effective,
             "grad_norm": float(norm), "lr": optimizer.param_groups[0]["lr"]})
        if progress["step"] % scfg["eval_interval"] == 0:
            result = evaluate(model, valid, scfg["batch_size"], device, amp, scfg["validation_max_samples"])
            log({"type": "validation", "step": progress["step"], **result})
        if progress["step"] % scfg["checkpoint_interval"] == 0:
            tag = "step_%08d" % progress["step"]
            save_checkpoint(checkpoints, tag, model, optimizer, scheduler, progress, contract)
            log({"type": "checkpoint", "step": progress["step"], "tag": tag})

    result = evaluate(model, valid, scfg["batch_size"], device, amp, scfg["validation_max_samples"])
    log({"type": "validation_final", "step": progress["step"], **result})
    tag = "final_%08d" % progress["step"]
    save_checkpoint(checkpoints, tag, model, optimizer, scheduler, progress, contract)
    print("FULL_SFT_DONE", "step", progress["step"], "tag", tag)


if __name__ == "__main__":
    main()
