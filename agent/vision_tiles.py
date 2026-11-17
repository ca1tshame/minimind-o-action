"""Multi-scale visual tile compression over the frozen SigLIP2 encoder.

Five tiles (global + 2x2 local) each yield 64 patch tokens. Each tile keeps its
top-scoring tokens and merges the remainder into one summary token, so tokens
never merge across tile boundaries and every kept token still maps to its tile.
"""
import types

import numpy as np
import torch
from PIL import Image
from torch import nn

from models.vision import IN_DIM

TILE_NAMES = ("global", "top_left", "top_right", "bottom_left", "bottom_right")
LAYOUT = {
    320: (64, 64, 64, 64, 64),
    160: (40, 30, 30, 30, 30),
    128: (32, 24, 24, 24, 24),
    64: (64, 0, 0, 0, 0),
}


def tokens_per_tile(n):
    if n not in LAYOUT:
        raise ValueError("unsupported visual token budget: %d" % n)
    return LAYOUT[n]


def split_tiles(image):
    im = image.convert("RGB")
    w, h = im.size
    mx, my = w // 2, h // 2
    boxes = [(0, 0, w, h), (0, 0, mx, my), (mx, 0, w, my),
             (0, my, mx, h), (mx, my, w, h)]
    if min(w, h) < 2:
        raise ValueError("invalid source image")
    return [im.crop(b) for b in boxes], boxes


def preprocess_tiles(image, vision_preprocess):
    tiles, boxes = split_tiles(image)
    mean = torch.tensor(vision_preprocess["image_mean"]).view(3, 1, 1)
    std = torch.tensor(vision_preprocess["image_std"]).view(3, 1, 1)
    out = []
    for tile in tiles:
        arr = np.asarray(tile.resize((256, 256), Image.BILINEAR), dtype=np.float32)
        out.append((torch.from_numpy(arr).permute(2, 0, 1) / 255.0 - mean) / std)
    return torch.stack(out), boxes


class TileCompressor(nn.Module):
    """Per-tile top-k selection plus one soft-merged summary token."""

    def __init__(self, in_dim=IN_DIM, hidden=256):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden),
            nn.GELU(), nn.Linear(hidden, 1),
        )

    def forward(self, x, keep):
        if keep >= x.shape[0]:
            return x, None
        scores = self.score(x).squeeze(-1)
        idx = torch.topk(scores, keep - 1).indices.sort().values
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask[idx] = False
        weights = torch.softmax(scores[mask], dim=0).unsqueeze(-1)
        merged = (x[mask] * weights).sum(0, keepdim=True)
        return torch.cat([x[idx], merged], 0), idx


def attach_tile_compression(model, n_tokens):
    """Attach the learned per-tile compressor and a tile-position embedding.

    Replaces the model's ``inject`` method so visual features are compressed
    per tile before projection. The language backbone and projector are shared;
    this is the module that produced the 320-token multi-scale condition.
    """
    if hasattr(model, "tile_pos"):
        raise ValueError("tile compression already attached")
    per_tile = tokens_per_tile(n_tokens)
    model.compressor = TileCompressor(IN_DIM)
    model.tile_pos = nn.Embedding(len(TILE_NAMES), model.lm.config.hidden_size)
    nn.init.zeros_(model.tile_pos.weight)
    model.vision_layout = per_tile

    def inject(self, tokens, hidden, mask, features):
        rows = []
        for b in range(tokens.shape[0]):
            ids = tokens[b].tolist()
            spans = []
            i = 0
            while i < len(ids):
                if ids[i] == 12:
                    begin = i
                    while i < len(ids) and ids[i] == 12:
                        i += 1
                    if mask is not None and not bool(mask[b, begin:i].bool().all()):
                        raise ValueError("image marker on padding")
                    spans.append((begin, i))
                else:
                    i += 1
            if len(spans) != len(features[b]):
                raise ValueError("media count mismatch")
            cursor = 0
            pieces = []
            for (start, end), feat in zip(spans, features[b]):
                if tuple(feat.shape) != (320, IN_DIM):
                    raise ValueError("expected 5x64 media tokens")
                if end - start != n_tokens:
                    raise ValueError("marker count differs from visual budget")
                outs, tile_ids = [], []
                for t in range(len(TILE_NAMES)):
                    keep = per_tile[t]
                    if keep == 0:
                        continue
                    part = feat[t * 64:(t + 1) * 64].to(
                        dtype=next(self.compressor.parameters()).dtype
                    )
                    kept, _ = self.compressor(part, keep) if keep < 64 else (part, None)
                    outs.append(kept)
                    tile_ids.append(
                        torch.full((keep,), t, dtype=torch.long, device=kept.device)
                    )
                cat = torch.cat(outs, 0)
                tids = torch.cat(tile_ids, 0)
                projected = self.proj(cat) + self.tile_pos(tids)
                if tuple(projected.shape) != (n_tokens, hidden.shape[-1]):
                    raise ValueError("projected shape mismatch")
                pieces.extend([hidden[b, cursor:start], projected.to(hidden.dtype)])
                cursor = end
            pieces.append(hidden[b, cursor:])
            rows.append(torch.cat(pieces))
        return torch.stack(rows)

    model.inject = types.MethodType(inject, model)
    return model
