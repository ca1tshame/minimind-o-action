"""P1.6 long-context continuation runner.

Reads P1.5 weights (not optimizer state). Candidate data plans are rejected
until their immutable manifest is explicitly marked training-ready. Execution
and paths are fully config/CLI driven.
"""
import argparse
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from safetensors.torch import load_model
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import load_config
from models import MiniMindConfig, MiniMindForCausalLM
from scripts.lib.checkpoint import digest, save_checkpoint, load_checkpoint
from scripts.lib.p16 import (
    make_window, loss_from_logits, token_weighted_update, choose_start,
)


def verify_base(checkpoint_dir):
    ready = json.loads((checkpoint_dir / "READY.json").read_text())
    if ready["payload"] != "payload":
        raise ValueError("invalid checkpoint payload")
    payload = checkpoint_dir / ready["payload"]
    if digest(payload / "manifest.json") != ready["manifest_sha256"]:
        raise ValueError("P1.5 READY mismatch")
    return payload / "model.safetensors"


def load_plan(plan_dir, expected_data_sha, expected_token_sha):
    manifest = json.loads((plan_dir / "manifest.json").read_text())
    if manifest.get("fixture") or manifest.get("data_sha256") != expected_data_sha:
        raise ValueError("not a verified full data plan")
    rows = {}
    for name in ("train", "validation"):
        path = plan_dir / (name + ".jsonl")
        if digest(path) != manifest[name + "_plan_sha256"]:
            raise ValueError("plan hash mismatch")
        rows[name] = [json.loads(line) for line in path.open(encoding="utf8")]
        if len(rows[name]) != manifest[name + "_rows"]:
            raise ValueError("plan length mismatch")
    train_ids = {r["sha256"] for r in rows["train"]}
    val_ids = {r["sha256"] for r in rows["validation"]}
    if len(train_ids) != len(rows["train"]) or train_ids & val_ids:
        raise ValueError("exact duplicate or split leak")
    if not any(r["context"] == 2048 for r in rows["train"]):
        raise ValueError("no true 2048 windows")
    return manifest, rows


def materialize(handle, record, tokenizer, epoch, device):
    handle.seek(record["offset"])
    text = json.loads(handle.readline())["text"]
    import hashlib
    if hashlib.sha256(text.encode()).hexdigest() != record["sha256"]:
        raise ValueError("document mismatch")
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) != record["tokens"]:
        raise ValueError("unexpected token count")
    start = choose_start(record["sha256"], epoch, len(ids), record["context"])
    window = make_window(
        ids, record["context"], tokenizer.bos_token_id,
        tokenizer.eos_token_id, tokenizer.pad_token_id, start=start,
    )
    return (
        {k: torch.tensor([window[k]], dtype=torch.long, device=device)
         for k in ("input_ids", "targets", "attention_mask")},
        window,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="P1.6 long-context training (explicit execution).")
    parser.add_argument("--config")
    parser.add_argument("--workspace-root")
    parser.add_argument("--plan")
    parser.add_argument("--data")
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--tokenizer-dir")
    parser.add_argument("--experiment")
    parser.add_argument("--resume")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute-training", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run == args.execute_training:
        parser.error("choose exactly one mode")

    cfg = load_config(args.config, {
        "paths.workspace_root": args.workspace_root,
        "paths.tokenizer_dir": args.tokenizer_dir,
    })
    paths, pcfg = cfg["paths"], cfg["long_pretrain"]
    root = Path(paths["workspace_root"])
    plan_dir = Path(args.plan)
    data = Path(args.data)
    base_checkpoint = Path(args.base_checkpoint)
    tokenizer_dir = Path(paths["tokenizer_dir"])
    experiment = Path(args.experiment or root / paths["experiment_dir"] / "pretrain_p16")

    meta, rows = load_plan(plan_dir, pcfg.get("data_sha256"), None)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, trust_remote_code=False)

    model_cfg = json.loads((ROOT / paths["model_config"]).read_text(encoding="utf8"))
    model = MiniMindForCausalLM(MiniMindConfig(**model_cfg))
    weights = verify_base(base_checkpoint)
    missing, extra = load_model(model, str(weights), strict=True, device="cpu")
    if missing or extra:
        raise ValueError("model load mismatch")
    if sum(p.numel() for p in model.parameters()) != cfg["model"]["parameters"]:
        raise ValueError("model parameter count mismatch")

    n = len(rows["train"])
    total = (n + pcfg["accumulation"] - 1) // pcfg["accumulation"]
    warmup = min(pcfg["warmup_updates"], total - 1)

    if args.dry_run:
        with data.open("rb") as handle:
            record = next(r for r in rows["train"] if r["context"] == 2048)
            example, window = materialize(handle, record, tokenizer, 0, "cpu")
        model.eval()
        with torch.inference_mode():
            loss = loss_from_logits(model(example["input_ids"]).logits, example["targets"])
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite loss")
        print("P16_CPU_DRY_RUN", json.dumps({
            "rows": n, "planned_updates": total,
            "valid_targets": window["valid_tokens"],
            "loss": float(loss), "gpu_used": False,
        }))
        return

    if meta.get("status") != "TRAINING_READY":
        raise PermissionError("candidate plan has not been reviewed and frozen")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    experiment.mkdir(parents=True, exist_ok=True)
    with FileLock(str(experiment / "training.lock"), timeout=0):
        torch.manual_seed(pcfg.get("seed", cfg["device"]["seed"]))
        model = model.cuda().train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=pcfg["learning_rate"], weight_decay=pcfg["weight_decay"]
        )
        total_steps = total
        def lr_scale(step):
            if not 0 <= step <= total_steps or not 0 <= warmup < total_steps:
                raise ValueError("schedule bounds")
            if warmup and step < warmup:
                return pcfg["learning_rate"] * (step + 1) / warmup
            return pcfg["learning_rate"] * (
                0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
            )

        class LambdaScheduler:
            def __init__(self):
                self.step = 0
            def advance(self, step):
                self.step = step
                for group in optimizer.param_groups:
                    group["lr"] = lr_scale(step)
            def state_dict(self):
                return {"step": self.step, "total": total_steps, "warmup": warmup, "lr": pcfg["learning_rate"]}
            def load_state_dict(self, state):
                if (state["total"], state["warmup"], state["lr"]) != (total_steps, warmup, pcfg["learning_rate"]):
                    raise ValueError("schedule mismatch")
                self.advance(state["step"])

        scheduler = LambdaScheduler()
        contract = {
            "base_model_sha256": digest(weights),
            "plan_sha256": digest(plan_dir / "manifest.json"),
            "runner_sha256": digest(Path(__file__)),
        }
        progress = {
            "step": 0, "epoch": 0, "next_sample": 0,
            "supervised_tokens": 0, "truncated_samples": 0,
            "accumulation_boundary": True,
        }
        checkpoints = experiment / "checkpoints"
        if args.resume:
            progress = load_checkpoint(checkpoints, args.resume, model, optimizer, scheduler, contract)
        stop = [False]
        def halt(signum, frame):
            stop[0] = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, halt)
        log = (experiment / "metrics.jsonl").open("a", encoding="utf8", buffering=1)
        def emit(obj):
            log.write(json.dumps(obj) + "\n")
            log.flush()
        def save(reason):
            tag = "step_%08d_%s" % (progress["step"], reason)
            save_checkpoint(checkpoints, tag, model, optimizer, scheduler, progress, contract)
            emit({"type": "checkpoint", "tag": tag, "step": progress["step"]})

        with data.open("rb") as handle:
            while progress["next_sample"] < n:
                if stop[0]:
                    save("stopped")
                    log.close()
                    return
                part = rows["train"][progress["next_sample"]:progress["next_sample"] + pcfg["accumulation"]]
                examples = []
                for record in part:
                    example, window = materialize(handle, record, tokenizer, 0, "cuda")
                    examples.append(example)
                scheduler.advance(progress["step"])
                autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
                result = token_weighted_update(model, optimizer, examples, pcfg["grad_clip"], autocast)
                progress["step"] += 1
                progress["next_sample"] += len(part)
                progress["supervised_tokens"] += result["targets"]
                emit({"type": "train", "step": progress["step"], **result})
                if progress["step"] % pcfg["checkpoint_interval"] == 0:
                    save("periodic")
            progress["epoch"] = 1
            save("final")
            emit({"type": "complete", "step": progress["step"]})
        log.close()


if __name__ == "__main__":
    main()
