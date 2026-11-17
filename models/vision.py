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
        object.__setattr__(self, "encoder", encoder.eval())

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


def lm_hidden_forward(lm, hidden, mask, start=0):
    """Run the transformer stack over already-projected hidden states."""
    model = lm.model
    seq = hidden.shape[1]
    cos, sin = model.freqs_cos[start:start + seq], model.freqs_sin[start:start + seq]
    position_embeddings = (cos, sin)
    h = hidden
    for layer in model.layers:
        h, _ = layer(
            h, position_embeddings, past_key_value=None,
            use_cache=False, attention_mask=mask,
        )
    return lm.lm_head(model.norm(h))


def build_backbone(config):
    """Instantiate the language backbone from a config dict."""
    return MiniMindForCausalLM(MiniMindConfig(**config))
