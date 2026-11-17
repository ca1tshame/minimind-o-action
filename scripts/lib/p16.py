"""P1.6 long-context causal windows and token-weighted update primitive."""
import hashlib
import random

import torch


def choose_start(document_hash, epoch, n, context):
    """Deterministically pick a window start from a document hash and epoch."""
    if epoch < 0 or n < 2 or context < 1:
        raise ValueError("bad window arguments")
    seed = int.from_bytes(
        hashlib.sha256((str(epoch) + ":" + document_hash).encode()).digest()[:8],
        "big",
    )
    return random.Random(seed).randrange(max(1, n - context))


def make_window(tokens, context, bos_id, eos_id, pad_id, start=0,
                natural_end_evidence=None):
    """Build an already-shifted causal window with right padding.

    Unknown record ends never create EOS labels. The model is called WITHOUT
    labels and cross-entropy is applied to logits and the returned targets.
    """
    if context < 1 or not tokens or start < 0 or start >= len(tokens):
        raise ValueError("invalid window")
    if natural_end_evidence is not None and (
        not isinstance(natural_end_evidence, str) or not natural_end_evidence.strip()
    ):
        raise ValueError("natural end requires provenance")
    if any(not isinstance(x, int) or x < 0 for x in tokens):
        raise ValueError("invalid token")
    if pad_id in tokens:
        raise ValueError("embedded PAD must be rejected and counted")
    prefix = [bos_id] if start == 0 else []
    seq = prefix + tokens[start:]
    verified_end = natural_end_evidence is not None
    if verified_end:
        seq.append(eos_id)
    seq = seq[:context + 1]
    if len(seq) < 2:
        raise ValueError("no true next token")
    x, y = seq[:-1], seq[1:]
    appended_eos = verified_end and len(prefix) + len(tokens) - start < len(seq)
    for i, token in enumerate(y):
        if token in (eos_id, pad_id) and not (
            token == eos_id and appended_eos and i == len(y) - 1
        ):
            y[i] = -100
    real = len(x)
    x = x + [pad_id] * (context - real)
    y = y + [-100] * (context - real)
    mask = [1] * real + [0] * (context - real)
    if all(t == -100 for t in y):
        raise ValueError("empty supervision")
    return {
        "input_ids": x, "targets": y, "attention_mask": mask,
        "start": start,
        "valid_tokens": sum(t != -100 for t in y),
        "verified_natural_end": appended_eos,
        "source_end_provenance": natural_end_evidence,
        "record_length": len(tokens),
    }


def loss_from_logits(logits, targets):
    if logits.shape[:-1] != targets.shape:
        raise ValueError("logit/target shape mismatch")
    if not torch.any(targets != -100):
        raise ValueError("empty supervision")
    return torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1), ignore_index=-100,
    )


def token_weighted_update(model, optimizer, examples, clip=1.0,
                          autocast_factory=None):
    """One token-weighted gradient update over B=1 aligned microbatches."""
    if not examples or clip <= 0:
        raise ValueError("invalid update")
    total = sum(int((e["targets"] != -100).sum()) for e in examples)
    if total <= 0:
        raise ValueError("no supervised tokens")
    optimizer.zero_grad(set_to_none=True)
    weighted = 0.0
    for example in examples:
        x, y, mask = example["input_ids"], example["targets"], example["attention_mask"]
        if x.ndim != 2 or x.shape != y.shape or x.shape != mask.shape or x.shape[0] != 1:
            raise ValueError("microbatch must be B=1 with aligned tensors")
        valid = int(mask.sum())
        if valid < 1 or not bool(torch.all(mask[0, :valid] == 1)) or not bool(torch.all(mask[0, valid:] == 0)):
            raise ValueError("invalid right padding")
        if not bool(torch.all(y[mask == 0] == -100)):
            raise ValueError("padding targets must be ignored")
        x, y = x[:, :valid], y[:, :valid]
        k = int((y != -100).sum())
        if not k:
            raise ValueError("empty example")
        if autocast_factory is None:
            logits = model(x).logits
            loss = loss_from_logits(logits, y)
        else:
            with autocast_factory():
                logits = model(x).logits
                loss = loss_from_logits(logits, y)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite loss")
        (loss * (k / total)).backward()
        weighted += float(loss.detach()) * k
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError("nonfinite gradient")
    optimizer.step()
    return {
        "targets": total, "nll": weighted / total,
        "grad_norm": float(norm), "microbatches": len(examples),
    }
