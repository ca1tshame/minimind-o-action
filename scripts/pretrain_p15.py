"""P1.5 full-data pretraining entry point.

Execution is explicit: pass --execute-training to run, otherwise the script
prints a validated plan only. Data, index, tokenizer, and output paths all come
from configuration and may be overridden on the command line.
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
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import load_config
from models import MiniMindConfig, MiniMindForCausalLM
from scripts.lib.checkpoint import digest, save_checkpoint, load_checkpoint
from scripts.lib.schedule import CosineSchedule, plan as step_plan, finish_epoch


def load_index(config, index_dir):
    manifest_path = index_dir / "manifest.json"
    if config.get("index_manifest_sha256") and digest(manifest_path) != config["index_manifest_sha256"]:
        raise ValueError("frozen index manifest mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    if manifest["fixture"] or manifest["source_sha256"] != config["data_sha256"] or manifest["max_length"] != config["max_length"]:
        raise ValueError("data or index mismatch")
    arrays = {}
    for split in ("train", "validation"):
        path = index_dir / (split + ".offsets.u64")
        if path.stat().st_size != manifest["counts"][split] * 8 or digest(path) != manifest[split + "_offsets_sha256"]:
            raise ValueError("offset index mismatch: " + split)
        arrays[split] = np.memmap(path, mode="r", dtype="<u8")
    return manifest, arrays


def make_batch(handle, offsets, tokenizer, max_length, device):
    texts = []
    for offset in offsets:
        handle.seek(int(offset))
        text = json.loads(handle.readline())["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("bad indexed document")
        texts.append(text)
    values = tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
    rows, truncated = [], 0
    for ids in values:
        truncated += int(len(ids) > max_length - 2)
        ids = [tokenizer.bos_token_id] + ids[:max_length - 2] + [tokenizer.eos_token_id]
        rows.append(ids + [tokenizer.pad_token_id] * (max_length - len(ids)))
    x = torch.tensor(rows, dtype=torch.long, device=device)
    y = x.clone()
    y[x == tokenizer.pad_token_id] = -100
    if not bool((y[:, 1:] != -100).any()):
        raise ValueError("no supervised targets")
    return x, y, int((y[:, 1:] != -100).sum()), truncated


def main(argv=None):
    parser = argparse.ArgumentParser(description="P1.5 full-data pretraining (explicit execution).")
    parser.add_argument("--config")
    parser.add_argument("--workspace-root")
    parser.add_argument("--data")
    parser.add_argument("--index")
    parser.add_argument("--tokenizer-dir")
    parser.add_argument("--experiment")
    parser.add_argument("--resume")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute-training", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run == args.execute_training:
        parser.error("choose exactly one of --dry-run or --execute-training")

    cfg = load_config(args.config, {
        "paths.workspace_root": args.workspace_root,
        "paths.pretrain_data": args.data,
        "paths.pretrain_index": args.index,
        "paths.tokenizer_dir": args.tokenizer_dir,
    })
    paths, pcfg = cfg["paths"], cfg["pretrain"]
    root = Path(paths["workspace_root"])
    data = Path(paths["pretrain_data"])
    index = Path(paths["pretrain_index"])
    tokenizer_dir = Path(paths["tokenizer_dir"])
    experiment = Path(args.experiment or root / paths["experiment_dir"] / "pretrain_p15")

    manifest, arrays = load_index(pcfg, index)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, trust_remote_code=False)
    model_cfg = json.loads((ROOT / paths["model_config"]).read_text(encoding="utf8"))
    model_cfg["max_position_embeddings"] = pcfg["max_length"]
    model = MiniMindForCausalLM(MiniMindConfig(**model_cfg))
    if sum(p.numel() for p in model.parameters()) != cfg["model"]["parameters"]:
        raise ValueError("model parameter count mismatch")

    plan = step_plan(
        len(arrays["train"]), pcfg["micro_batch_size"],
        pcfg["gradient_accumulation_steps"], pcfg["epochs"],
    )
    if args.dry_run:
        with data.open("rb") as handle:
            x, y, targets, truncated = make_batch(
                handle, arrays["train"][:1], tokenizer, pcfg["max_length"], "cpu"
            )
        model.eval()
        with torch.inference_mode():
            loss = model(x, labels=y).loss
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite dry-run loss")
        print("CPU_DRY_RUN", json.dumps({
            "plan": plan, "train_rows": len(arrays["train"]),
            "validation_rows": len(arrays["validation"]),
            "input_shape": list(x.shape), "targets": targets,
            "loss": float(loss), "gpu_used": False,
        }))
        return

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("one BF16 CUDA GPU is required for pretraining")
    experiment.mkdir(parents=True, exist_ok=True)
    with FileLock(str(experiment / "training.lock"), timeout=0):
        contract = {
            "config": pcfg, "data_sha256": pcfg.get("data_sha256"),
            "plan": plan, "runner_sha256": digest(Path(__file__)),
        }
        torch.manual_seed(pcfg.get("seed", cfg["device"]["seed"]))
        model = model.cuda()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=pcfg["learning_rate"],
            weight_decay=pcfg["weight_decay"],
        )
        scheduler = CosineSchedule(optimizer, pcfg["learning_rate"], plan["total_microbatches"])
        progress = {
            "step": 0, "epoch": 0, "next_sample": 0,
            "supervised_tokens": 0, "truncated_samples": 0,
            "micro": 0, "accumulation_boundary": True,
        }
        checkpoints = experiment / "checkpoints"
        if args.resume:
            progress = load_checkpoint(checkpoints, args.resume, model, optimizer, scheduler, contract)
        log = (experiment / "metrics.jsonl").open("a", encoding="utf8", buffering=1)
        stop = [False]

        def halt(signum, frame):
            stop[0] = True

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, halt)
        if hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1, halt)

        def emit(obj):
            log.write(json.dumps(obj) + "\n")
            log.flush()

        def save(reason):
            tag = "step_%08d_%s" % (progress["step"], reason)
            save_checkpoint(checkpoints, tag, model, optimizer, scheduler, progress, contract)
            emit({"type": "checkpoint", "tag": tag, "step": progress["step"]})

        def evaluate(handle):
            model.eval()
            total, count = 0.0, 0
            with torch.inference_mode():
                for start in range(0, min(len(arrays["validation"]), pcfg["eval_samples"]), pcfg["micro_batch_size"]):
                    x, y, k, _ = make_batch(
                        handle, arrays["validation"][start:start + pcfg["micro_batch_size"]],
                        tokenizer, pcfg["max_length"], "cuda",
                    )
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = model(x, labels=y).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite validation")
                    total += float(loss) * k
                    count += k
            model.train()
            nll = total / count
            emit({"type": "validation", "step": progress["step"], "nll": nll, "ppl": math.exp(nll), "targets": count})

        with data.open("rb") as handle:
            if not args.resume:
                evaluate(handle)
            while progress["epoch"] < pcfg["epochs"]:
                generator = torch.Generator(device="cpu").manual_seed(
                    cfg["device"]["seed"] + progress["epoch"]
                )
                order = torch.randperm(len(arrays["train"]), generator=generator).numpy()
                model.train()
                while progress["next_sample"] < len(order):
                    if stop[0]:
                        save("stopped")
                        log.close()
                        return
                    optimizer.zero_grad(set_to_none=True)
                    loss_sum = 0.0
                    micro_in_step = 0
                    while (micro_in_step < pcfg["gradient_accumulation_steps"]
                           and progress["next_sample"] < len(order)):
                        start = progress["next_sample"]
                        selection = order[start:start + pcfg["micro_batch_size"]]
                        x, y, _, truncated = make_batch(
                            handle, arrays["train"][selection], tokenizer,
                            pcfg["max_length"], "cuda",
                        )
                        progress["truncated_samples"] += truncated
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            loss = model(x, labels=y).loss
                        (loss / pcfg["gradient_accumulation_steps"]).backward()
                        loss_sum += float(loss.detach())
                        micro_in_step += 1
                        progress["micro"] += 1
                        progress["next_sample"] += len(selection)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), pcfg["grad_clip"])
                    scheduler.advance(progress["micro"])
                    optimizer.step()
                    progress["step"] += 1
                    emit({
                        "type": "train", "step": progress["step"],
                        "loss": loss_sum / micro_in_step,
                        "grad_norm": float(norm),
                    })
                    if progress["step"] % pcfg["checkpoint_interval_updates"] == 0:
                        evaluate(handle)
                        save("periodic")
                finish_epoch(progress, len(order), plan, 0)
            evaluate(handle)
            save("final")
        log.close()


if __name__ == "__main__":
    main()
