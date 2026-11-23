"""Vision-language action model.

Combines a frozen SigLIP2 vision encoder, a trainable cross-modal projector,
and the 63.9M Dense MiniMind language backbone. Visual features replace the
dedicated image-marker tokens in the embedding sequence, so the language model
consumes a single unified token stream.
"""
import io

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import SiglipVisionModel

from models.minimind import MiniMindConfig, MiniMindForCausalLM

# Vision marker contract inherited from the validated GUI action pipeline.
IMAGE_ID = 12
IMAGE_TOKEN = "<|image_pad|>"
TOKENS_PER_IMAGE = 64
IN_DIM = 768
IMG_SIZE = 256


class MMVisionProjector(nn.Module):
    """Trainable projector mapping frozen SigLIP2 features into LM hidden space."""

    def __init__(self, in_dim=IN_DIM, out_dim=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class FrozenSigLIP2(nn.Module):
    """Wrapper around the externally held, frozen SigLIP2 encoder.

    The encoder is loaded from a local directory only. It is never downloaded,
    never registered as a trainable child, and never written to checkpoints.
    """

    def __init__(self, model_dir, dtype=torch.float32):
        super().__init__()
        encoder = SiglipVisionModel.from_pretrained(
            str(model_dir), local_files_only=True, torch_dtype=dtype
        )
        for param in encoder.parameters():
            param.requires_grad_(False)
        # Registered as a real submodule on purpose: only then do ``.to(device)``,
        # ``.eval()`` and ``.half()`` reach the encoder. It is frozen via
        # requires_grad_(False) and it never lands in an action checkpoint
        # (GUIDE: the action model's state_dict does not contain it, and
        # ``load_action_model`` drops any ``vision_encoder.*`` keys).
        self.encoder = encoder.eval()

    def forward(self, pixel_values):
        with torch.no_grad():
            return self.encoder(pixel_values=pixel_values).last_hidden_state


def inject_vision_features(tokens, hidden, mask, features, projector, in_dim=IN_DIM):
    """Replace each image-marker run with projected visual features.

    ``features`` is an ordered list (one list of 2-D tensors per batch sample).
    The number of marker runs must equal the number of media items per sample,
    and each feature's first dimension must equal its marker run length.
    """
    if tokens.ndim != 2 or hidden.ndim != 3 or hidden.shape[:2] != tokens.shape:
        raise ValueError("invalid token/embedding shape")
    if mask is not None and (mask.ndim != 2 or mask.shape != tokens.shape):
        raise ValueError("invalid attention mask shape")
    batch, length = tokens.shape
    if not isinstance(features, (list, tuple)) or len(features) != batch:
        raise ValueError("media batch mismatch")
    rows = []
    for b in range(batch):
        ids = tokens[b].tolist()
        spans = []
        i = 0
        while i < length:
            if ids[i] == IMAGE_ID:
                begin = i
                while i < length and ids[i] == IMAGE_ID:
                    i += 1
                if mask is not None and not bool(mask[b, begin:i].bool().all()):
                    raise ValueError("image marker on padded position")
                spans.append((begin, i))
            else:
                i += 1
        if len(spans) != len(features[b]):
            raise ValueError("marker/media count mismatch")
        parts = []
        cursor = 0
        for (begin, end), feature in zip(spans, features[b]):
            if not isinstance(feature, torch.Tensor) or feature.ndim != 2:
                raise ValueError("invalid feature tensor")
            if tuple(feature.shape) != (end - begin, in_dim):
                raise ValueError("feature shape/length mismatch")
            if feature.device != hidden.device or not bool(torch.isfinite(feature).all()):
                raise ValueError("invalid feature device or values")
            projected = projector(
                feature.to(dtype=next(projector.parameters()).dtype)
            ).to(hidden.dtype)
            if tuple(projected.shape) != (end - begin, hidden.shape[-1]):
                raise ValueError("projected feature shape mismatch")
            parts.extend((hidden[b, cursor:begin], projected))
            cursor = end
        parts.append(hidden[b, cursor:])
        row = torch.cat(parts, dim=0)
        if tuple(row.shape) != tuple(hidden[b].shape):
            raise ValueError("sequence length changed")
        rows.append(row)
    return torch.stack(rows)


class GuiActionModel(nn.Module):
    """Language backbone plus trainable vision projector for GUI actions."""

    def __init__(self, lm, in_dim=IN_DIM):
        super().__init__()
        self.lm = lm
        self.proj = MMVisionProjector(in_dim, lm.config.hidden_size)

    def inject(self, tokens, hidden, mask, features):
        return inject_vision_features(tokens, hidden, mask, features, self.proj)


def preprocess_rgb(jpeg_bytes, vision_preprocess):
    """Resize and normalize raw image bytes to the encoder's expected tensor."""
    with Image.open(io.BytesIO(jpeg_bytes)) as image:
        image = image.convert("RGB").resize(
            (IMG_SIZE, IMG_SIZE), Image.BILINEAR
        )
        array = np.asarray(image, dtype=np.float32)
    x = torch.from_numpy(array).permute(2, 0, 1) / 255.0
    mean = torch.tensor(vision_preprocess["image_mean"]).view(3, 1, 1)
    std = torch.tensor(vision_preprocess["image_std"]).view(3, 1, 1)
    return ((x - mean) / std).unsqueeze(0)


def lm_hidden_forward(lm, hidden, mask, start=0, past_key_values=None, use_cache=False):
    """Run the transformer stack over already-projected hidden states."""
    model = lm.model
    seq = hidden.shape[1]
    cos, sin = model.freqs_cos[start:start + seq], model.freqs_sin[start:start + seq]
    position_embeddings = (cos, sin)
    h = hidden
    past_key_values = past_key_values or [None] * len(model.layers)
    presents = []
    for layer, past_kv in zip(model.layers, past_key_values):
        h, present = layer(
            h, position_embeddings, past_key_value=past_kv,
            use_cache=use_cache, attention_mask=mask,
        )
        presents.append(present)
    logits = lm.lm_head(model.norm(h))
    if use_cache:
        return logits, presents
    return logits


def build_backbone(config):
    """Instantiate the language backbone from a config dict."""
    return MiniMindForCausalLM(MiniMindConfig(**config))


# ---------------------------------------------------------------------------
# Checkpoint loading contract
# ---------------------------------------------------------------------------

# The original MiniMind-O checkpoints store the projector as ``proj.net.*``
# (``nn.Sequential`` named ``net``); the clean rebuild calls it ``proj.mlp.*``.
PROJECTOR_KEY_MAP = (("proj.net.", "proj.mlp."),)


class ActionWeightError(RuntimeError):
    """A checkpoint cannot be loaded without leaving core modules random."""


def remap_action_keys(state):
    """Rename legacy projector keys and strip ``_orig_mod.`` compile prefixes."""
    mapped, renamed = {}, 0
    for key, value in state.items():
        new = key[len("_orig_mod."):] if key.startswith("_orig_mod.") else key
        for old, current in PROJECTOR_KEY_MAP:
            if new.startswith(old):
                new = current + new[len(old):]
                break
        renamed += int(new != key)
        mapped[new] = value
    return mapped, renamed


def action_parameter_report(model):
    """Return parameter counts per action-model component."""
    def count(prefix):
        return sum(p.numel() for name, p in model.named_parameters()
                   if name.startswith(prefix))

    return {
        "lm": sum(p.numel() for p in model.lm.parameters()),
        "proj": sum(p.numel() for p in model.proj.parameters()),
        "compressor": count("compressor."),
        "tile_pos": count("tile_pos."),
        "total": sum(p.numel() for p in model.parameters()),
    }


# Embedding names that MiniMind ties together (config.tie_word_embeddings=true).
# A checkpoint stores only one of each pair; the other is loaded via the shared
# tensor, so it must not be reported as a missing weight.
_TIED_PARTNERS = {
    "lm.model.embed_tokens.weight": "lm.lm_head.weight",
    "lm.lm_head.weight": "lm.model.embed_tokens.weight",
    "model.embed_tokens.weight": "lm_head.weight",
    "lm_head.weight": "model.embed_tokens.weight",
}


def _tie_intact(model) -> bool:
    """True when the language model really shares embedding and output weights."""
    lm = getattr(model, "lm", model)
    core = getattr(lm, "model", None)
    head = getattr(lm, "lm_head", None)
    embed = getattr(core, "embed_tokens", None) if core is not None else None
    if head is None or embed is None:
        return False
    return embed.weight is head.weight


def load_action_model(model, path, *, require_tile=True, allow_missing=(),
                      allow_extra=(), strict=True):
    """Load a V14 action checkpoint with explicit key validation.

    * legacy ``proj.net.*`` keys are mapped to ``proj.mlp.*``;
    * frozen ``vision_encoder.*`` weights are dropped (the encoder is loaded
      separately from ``paths.vision_encoder_dir``);
    * the TileCompressor and tile_pos must already be attached *and* present
      in the checkpoint when ``require_tile`` is true, so the model can never
      look "loaded" while two core modules are still random;
    * remaining missing/unexpected keys are an error unless explicitly
      allow-listed (``allow_missing`` / ``allow_extra`` prefixes).

    Returns a report dict describing exactly what happened.
    """
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - safetensors is a hard dep
        raise ActionWeightError("safetensors is required to load checkpoints") from exc
    path = str(path)
    state = load_file(path)
    state, renamed = remap_action_keys(state)
    dropped = [key for key in state if key.startswith("vision_encoder.")]
    for key in dropped:
        del state[key]

    if require_tile:
        if getattr(model, "compressor", None) is None or getattr(model, "tile_pos", None) is None:
            raise ActionWeightError(
                "attach_tile_compression() must run before load_action_model(); "
                "otherwise compressor/tile_pos are random"
            )
        for prefix in ("compressor.", "tile_pos."):
            if not any(key.startswith(prefix) for key in state):
                raise ActionWeightError(
                    "checkpoint %s has no %s weights: refusing to load a model "
                    "whose tile compression module would stay random" % (path, prefix)
                )

    incompatible = model.load_state_dict(state, strict=False)
    # Tied embeddings: MiniMind shares embed_tokens.weight with lm_head.weight
    # (config tie_word_embeddings=true), so a checkpoint stores only one of the
    # two names. Loading either updates the same tensor in place, which means
    # the other name shows up as "missing" while being fully loaded. Treat such
    # keys as satisfied only when the tied partner really is in the checkpoint,
    # and verify afterwards that the tie still holds.
    satisfied_tied = []
    for key in list(incompatible.missing_keys):
        partner = _TIED_PARTNERS.get(key)
        if partner and partner in state:
            satisfied_tied.append(key)
    missing = [k for k in incompatible.missing_keys
               if k not in satisfied_tied
               and not any(k.startswith(p) for p in allow_missing)]
    unexpected = [k for k in incompatible.unexpected_keys
                  if not any(k.startswith(p) for p in allow_extra)]
    if satisfied_tied and not _tie_intact(model):
        raise ActionWeightError(
            "checkpoint %s relies on tied embeddings (%s) but the model does not "
            "share embed_tokens.weight with lm_head.weight" % (path, satisfied_tied)
        )
    if strict and (missing or unexpected):
        raise ActionWeightError(
            "checkpoint %s does not match the model: missing=%s unexpected=%s"
            % (path, missing[:20], unexpected[:20])
        )
    counts = action_parameter_report(model)
    if require_tile and counts["compressor"] <= 0:
        raise ActionWeightError("compressor has no parameters after loading")
    return {
        "path": path,
        "renamed_keys": renamed,
        "dropped_vision_encoder_keys": len(dropped),
        "missing_keys": incompatible.missing_keys,
        "unexpected_keys": incompatible.unexpected_keys,
        "ignored_missing": missing,
        "ignored_unexpected": unexpected,
        "parameters": counts,
    }


def load_lm_checkpoint(lm, path, expected_parameters=None):
    """Strictly load the backbone weights of an action or language checkpoint.

    Action checkpoints store the backbone under ``lm.model.*`` / ``lm.lm_head.*``
    (because ``GuiActionModel`` wraps it); language checkpoints store the same
    tensors without the prefix.  Only backbone tensors are selected here, so a
    full V14 checkpoint can be used for the language parameters while the
    projector/compressor/tile_pos are validated separately.
    """
    from safetensors.torch import load_file

    raw, _renamed = remap_action_keys(load_file(str(path)))
    state = {}
    for key, value in raw.items():
        if key.startswith("lm."):
            state[key[3:]] = value
        elif key.startswith("model.") or key.startswith("lm_head."):
            state[key] = value
    incompatible = lm.load_state_dict(state, strict=False)
    satisfied_tied = []
    for key in list(incompatible.missing_keys):
        partner = _TIED_PARTNERS.get(key)
        if partner and partner in state:
            satisfied_tied.append(key)
    missing = [k for k in incompatible.missing_keys if k not in satisfied_tied]
    unexpected = list(incompatible.unexpected_keys)
    if satisfied_tied and not _tie_intact(lm):
        raise ActionWeightError(
            "language checkpoint %s relies on tied embeddings (%s) but the model "
            "does not share embed_tokens.weight with lm_head.weight"
            % (path, satisfied_tied)
        )
    if missing or unexpected:
        raise ActionWeightError(
            "language checkpoint %s mismatch: missing=%s unexpected=%s"
            % (path, missing[:20], unexpected[:20])
        )
    total = sum(p.numel() for p in lm.parameters())
    if expected_parameters is not None and total != int(expected_parameters):
        raise ActionWeightError(
            "language backbone has %d parameters, expected %d"
            % (total, int(expected_parameters))
        )
    return total
