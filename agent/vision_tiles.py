"""Multi-scale visual tile compression over the frozen SigLIP2 encoder.

Five base tiles (global + 2x2 local) each yield 64 patch tokens. With a
384-token focus budget (V16-V18), a 6th native-resolution local crop around
the active workspace / previous click coordinate is appended so centered form
controls are never split along the 2x2 midline.
"""
import re
import types

import numpy as np
import torch
from PIL import Image
from torch import nn

from models.vision import IN_DIM

TILE_NAMES = ("global", "top_left", "top_right", "bottom_left", "bottom_right")
TILE_NAMES_FOCUS = TILE_NAMES + ("focus_crop",)
LAYOUT = {
    384: (64, 64, 64, 64, 64, 64),
    320: (64, 64, 64, 64, 64),
    160: (40, 30, 30, 30, 30),
    128: (32, 24, 24, 24, 24),
    64: (64, 0, 0, 0, 0),
}


def tokens_per_tile(n):
    if n not in LAYOUT:
        raise ValueError("unsupported visual token budget: %d" % n)
    return LAYOUT[n]


def focus_box_from_xy(w, h, fx, fy, cw=480, ch=360):
    """Return a clamped native-resolution crop box around normalized ``(fx, fy)``."""
    cw = min(max(2, int(cw)), w)
    ch = min(max(2, int(ch)), h)
    cx = int(round(min(max(float(fx), 0.0), 1.0) * (w - 1)))
    cy = int(round(min(max(float(fy), 0.0), 1.0) * (h - 1)))
    x0 = min(max(0, cx - cw // 2), max(0, w - cw))
    y0 = min(max(0, cy - ch // 2), max(0, h - ch))
    return (x0, y0, x0 + cw, y0 + ch)


def next_focus_xy(history=None, err=None, default=(0.50, 0.30)):
    """Deterministic focus-crop center shared identically by training and rollout."""
    if err:
        return default
    for item in reversed(list(history or ())):
        s = str(item)
        if "[action error]" in s or "<action>tab(" in s or "<action>key(" in s:
            return default
        m = re.search(r"<action>click\(([^,]+),([^)]+)\)</action>", s)
        if m:
            try:
                cx, cy = float(m.group(1)), float(m.group(2))
                if 0.0 <= cx <= 1.0 and 0.0 <= cy < 0.90:
                    return (cx, cy)
                return default
            except ValueError:
                return default
        if "<action>type(" in s:
            continue
        return default
    return default


def split_tiles(image, focus_box=None, focus_xy=None):
    im = image.convert("RGB")
    w, h = im.size
    mx, my = w // 2, h // 2
    boxes = [(0, 0, w, h), (0, 0, mx, my), (mx, 0, w, my),
             (0, my, mx, h), (mx, my, w, h)]
    if min(w, h) < 2:
        raise ValueError("invalid source image")
    if focus_box is None and focus_xy is not None:
        focus_box = focus_box_from_xy(w, h, focus_xy[0], focus_xy[1])
    if focus_box is not None:
        x0, y0, x1, y1 = [int(v) for v in focus_box]
        x0 = max(0, min(w - 2, x0))
        y0 = max(0, min(h - 2, y0))
        x1 = max(x0 + 2, min(w, x1))
        y1 = max(y0 + 2, min(h, y1))
        boxes.append((x0, y0, x1, y1))
    return [im.crop(b) for b in boxes], boxes


def preprocess_tiles(image, vision_preprocess, focus_box=None, focus_xy=None):
    tiles, boxes = split_tiles(image, focus_box=focus_box, focus_xy=focus_xy)
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
    tile_names = TILE_NAMES_FOCUS if len(per_tile) == 6 else TILE_NAMES
    expected_raw = len(tile_names) * 64
    model.compressor = TileCompressor(IN_DIM)
    model.tile_pos = nn.Embedding(len(tile_names), model.lm.config.hidden_size)
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
                if tuple(feat.shape) != (expected_raw, IN_DIM):
                    raise ValueError(f"expected {len(tile_names)}x64 media tokens")
                if end - start != n_tokens:
                    raise ValueError("marker count differs from visual budget")
                outs, tile_ids = [], []
                for t in range(len(tile_names)):
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
