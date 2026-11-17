"""Atomic, integrity-checked checkpoint save and resume.

Checkpoints write to a temporary directory, fsync payloads, and publish a
READY marker only after a full hash round-trip. Only locally generated state is
loaded with weights_only=True.
"""
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_model, save_model


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _sync_dir(path):
    if os.name == "nt":
        return
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "name": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def _restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((
        n["name"], np.array(n["keys"], dtype=np.uint32),
        n["position"], n["has_gauss"], n["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device count mismatch")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _safe_tag(tag):
    if not tag or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in tag
    ):
        raise ValueError("unsafe checkpoint tag")


def save_checkpoint(root, tag, model, optimizer, scheduler, progress, contract):
    _safe_tag(tag)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    dest = root / tag
    tmp = root / (tag + ".incomplete")
    if tmp.exists() or dest.exists():
        raise FileExistsError("checkpoint target occupied")
    if progress.get("accumulation_boundary") is not True:
        raise ValueError("not at accumulation boundary")
    for key in ("step", "epoch", "next_sample", "supervised_tokens", "truncated_samples", "order"):
        if key not in progress:
            raise ValueError("missing progress field: " + key)

    tmp.mkdir()
    save_model(model, str(tmp / "model.safetensors"))
    state = {
        "format_version": 1,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "progress": progress,
        "contract": contract,
        "rng": _rng_state(),
    }
    torch.save(state, tmp / "state.pt")
    for name in ("model.safetensors", "state.pt"):
        with (tmp / name).open("r+b") as handle:
            os.fsync(handle.fileno())
    probe = torch.load(tmp / "state.pt", map_location="cpu", weights_only=True)
    if probe["progress"] != progress or probe["contract"] != contract:
        raise ValueError("state round-trip mismatch")
    manifest = {
        "format_version": 1,
        "model_sha256": digest(tmp / "model.safetensors"),
        "state_sha256": digest(tmp / "state.pt"),
        "contract": contract,
        "progress": progress,
    }
    with (tmp / "manifest.json").open("x", encoding="utf8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    _sync_dir(tmp)

    dest.mkdir()
    _sync_dir(root)
    try:
        os.rename(tmp, dest / "payload")
    except Exception as exc:
        raise RuntimeError("checkpoint reservation incomplete; preserve both directories") from exc
    _sync_dir(dest)
    with (dest / "READY.json.part").open("x", encoding="utf8", newline="\n") as handle:
        json.dump({
            "format_version": 1,
            "payload": "payload",
            "manifest_sha256": digest(dest / "payload" / "manifest.json"),
        }, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.rename(dest / "READY.json.part", dest / "READY.json")
    _sync_dir(dest)
    _sync_dir(root)
    return dest


def load_checkpoint(root, tag, model, optimizer, scheduler, contract):
    _safe_tag(tag)
    root = Path(root).resolve()
    dest = root / tag
    if dest.is_symlink():
        raise ValueError("unsafe checkpoint")
    ready, base = dest / "READY.json", dest / "payload"
    if not ready.is_file() or ready.is_symlink() or base.is_symlink():
        raise ValueError("checkpoint incomplete")
    marker = json.loads(ready.read_text(encoding="utf8"))
    if marker.get("payload") != "payload" or digest(base / "manifest.json") != marker["manifest_sha256"]:
        raise ValueError("manifest hash mismatch")
    for name in ("manifest.json", "model.safetensors", "state.pt"):
        if (base / name).is_symlink():
            raise ValueError("checkpoint symlink refused")
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf8"))
    if (
        manifest["contract"] != contract
        or digest(base / "model.safetensors") != manifest["model_sha256"]
        or digest(base / "state.pt") != manifest["state_sha256"]
    ):
        raise ValueError("checkpoint contract or hash mismatch")
    state = torch.load(base / "state.pt", map_location="cpu", weights_only=True)
    if (
        state["contract"] != contract
        or state["progress"] != manifest["progress"]
        or state["progress"].get("accumulation_boundary") is not True
    ):
        raise ValueError("invalid resume state")
    missing, unexpected = load_model(model, str(base / "model.safetensors"), strict=True, device="cpu")
    if missing or unexpected:
        raise ValueError("checkpoint model mismatch")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    _restore_rng(state["rng"])
    return state["progress"]
