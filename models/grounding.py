"""CenterNet-style visual grounding over the five SigLIP2 tile grids.

The head follows "Objects as Points" (Zhou et al., CenterNet): a per-patch
objectness heatmap plus a per-patch offset regression. Only the head and the
loss are new; the backbone stays the frozen SigLIP2 encoder and the trainable
projector, and the patch grid is the existing multi-scale tile layout (one
global tile plus the 2x2 quadrants).

Spatial contract (must match collection, training and evaluation):

* 5 tiles x 64 patch tokens = 320 heatmap cells, matching
  ``agent.vision_tiles.LAYOUT[320]`` (one global tile plus four quadrants).
* A 256 px tile encoded by SigLIP2-base-p32-256 yields an 8x8 patch grid, so
  one patch is 32 px.  The flat index is ``tile * 64 + row * 8 + col``.
* ``point_to_token`` maps a viewport-normalised click to ``(token, ox, oy)``
  where ``ox``/``oy`` are the sub-patch offsets in ``[0, 1)``.
* ``token_offset_to_point`` inverts that map: the point in tile space is
  ``((col + ox) * 32 / 256, (row + oy) * 32 / 256)``, then the tile's viewport
  region is applied (tile 0 is the full viewport, tiles 1..4 the quadrants).

Coordinates are always viewport-normalised: ``(0, 0)`` is the top-left corner
and ``(1, 1)`` the bottom-right corner.
"""
import torch
from torch import nn

TILES = 5
PATCH_GRID = 8
PATCHES_PER_TILE = PATCH_GRID * PATCH_GRID
NUM_TOKENS = TILES * PATCHES_PER_TILE
TILE_PIXELS = 256
PATCH_PIXELS = TILE_PIXELS // PATCH_GRID

# Names used by the original V9 research script; kept so ported call sites stay
# readable without changing the numeric contract.
GRID = PATCH_GRID
CELLS = PATCHES_PER_TILE
PATCH = PATCH_PIXELS
IMAGE_TOKEN_ID = 12

_NEIGHBOUR_WEIGHT = {
    (0, 0): 1.0,
    (0, 1): 0.5, (0, -1): 0.5, (1, 0): 0.5, (-1, 0): 0.5,
    (1, 1): 0.25, (1, -1): 0.25, (-1, 1): 0.25, (-1, -1): 0.25,
}


def patch_grid_indices(grid=PATCH_GRID):
    """Return one tile's ``grid x grid`` flat token offsets, row-major.

    The encoder uses ``grid=8`` (64 patches per tile, 320 tokens for five
    tiles). A different square grid is accepted for inspection only; it does
    not change ``NUM_TOKENS`` or the decoding contract.
    """
    if not isinstance(grid, int) or isinstance(grid, bool) or grid < 1:
        raise ValueError("grid must be a positive integer")
    return [[row * grid + col for col in range(grid)] for row in range(grid)]


def _check_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("%s must be an integer" % name)
    return value


def token_to_patch(token, grid=PATCH_GRID, tiles=TILES):
    """Map a flat heatmap index to ``(tile, row, col)``.

    ``token`` must be in ``[0, tiles * grid * grid)``. With the default grid
    this is 320 cells: ``tile = token // 64``, ``row = (token % 64) // 8``,
    ``col = token % 8``.
    """
    _check_int(token, "token")
    _check_int(grid, "grid")
    _check_int(tiles, "tiles")
    if grid < 1 or tiles < 1:
        raise ValueError("grid and tiles must be positive")
    if not 0 <= token < tiles * grid * grid:
        raise ValueError("token %d outside [0, %d)" % (token, tiles * grid * grid))
    tile, cell = divmod(token, grid * grid)
    row, col = divmod(cell, grid)
    return tile, row, col


def patch_to_token(tile, row, col, grid=PATCH_GRID, tiles=TILES):
    """Map ``(tile, row, col)`` back to the flat heatmap index."""
    for value, name in ((tile, "tile"), (row, "row"), (col, "col")):
        _check_int(value, name)
        if not 0 <= value < (tiles if name == "tile" else grid):
            raise ValueError("%s %d out of range" % (name, value))
    return tile * grid * grid + row * grid + col


def token_offset_to_point(token, ox, oy):
    """Decode one heatmap cell into a viewport-normalised ``(x, y)`` point.

    ``token`` selects tile/patch; ``ox``/``oy`` are the in-patch offsets
    (normally in ``[0, 1]``). The result is clamped to ``[0, 1]``.
    """
    _check_int(token, "token")
    tile, row, col = token_to_patch(token)
    ox = float(ox)
    oy = float(oy)
    if not torch.isfinite(torch.tensor([ox, oy])).all():
        raise ValueError("offsets must be finite")
    u = ((col + ox) * PATCH_PIXELS) / float(TILE_PIXELS)
    v = ((row + oy) * PATCH_PIXELS) / float(TILE_PIXELS)
    if tile == 0:
        x, y = u, v
    elif tile == 1:
        x, y = u * 0.5, v * 0.5
    elif tile == 2:
        x, y = u * 0.5 + 0.5, v * 0.5
    elif tile == 3:
        x, y = u * 0.5, v * 0.5 + 0.5
    else:
        x, y = u * 0.5 + 0.5, v * 0.5 + 0.5
    return min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)


def point_to_token(x, y):
    """Map a viewport-normalised point to ``(token, ox, oy)``.

    Points away from the quadrant borders (``0.12 < x < 0.88`` and
    ``0.12 < y < 0.88``) go to the global tile, which covers the whole
    viewport at full resolution; border and corner points use the matching
    quadrant. Offsets are clamped to ``[0, 0.999]`` so the decoded patch never
    leaves the cell it was assigned to.
    """
    x = float(x)
    y = float(y)
    if not torch.isfinite(torch.tensor([x, y])).all():
        raise ValueError("point must be finite")
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError("point must be normalised to [0, 1]")
    if 0.12 < x < 0.88 and 0.12 < y < 0.88:
        tile, u, v = 0, x, y
    elif x < 0.5 and y < 0.5:
        tile, u, v = 1, x * 2.0, y * 2.0
    elif x >= 0.5 and y < 0.5:
        tile, u, v = 2, (x - 0.5) * 2.0, y * 2.0
    elif x < 0.5 and y >= 0.5:
        tile, u, v = 3, x * 2.0, (y - 0.5) * 2.0
    else:
        tile, u, v = 4, (x - 0.5) * 2.0, (y - 0.5) * 2.0
    pu = u * float(TILE_PIXELS)
    pv = v * float(TILE_PIXELS)
    col = min(PATCH_GRID - 1, max(0, int(pu // PATCH_PIXELS)))
    row = min(PATCH_GRID - 1, max(0, int(pv // PATCH_PIXELS)))
    ox = pu / float(PATCH_PIXELS) - col
    oy = pv / float(PATCH_PIXELS) - row
    token = patch_to_token(tile, row, col)
    return token, float(min(max(ox, 0.0), 0.999)), float(min(max(oy, 0.0), 0.999))


def decode_points(heat, off, topk=1):
    """Decode the top-``topk`` points from heatmap logits and offsets.

    ``heat`` is ``[B, 320]`` logits, ``off`` is ``[B, 320, 2]`` in-patch
    offsets. Returns ``(points, scores)`` where ``points`` is a list of
    ``B`` lists of ``(x, y)`` tuples and ``scores`` is ``[B, topk]`` sigmoid
    probabilities.
    """
    if heat.ndim != 2 or off.ndim != 3 or off.shape[:2] != heat.shape:
        raise ValueError("expected heat [B, N] and off [B, N, 2]")
    if off.shape[2] != 2:
        raise ValueError("offsets must have last dimension 2")
    if heat.shape[1] != NUM_TOKENS:
        raise ValueError("expected %d heatmap cells, got %d" % (NUM_TOKENS, heat.shape[1]))
    _check_int(topk, "topk")
    if not 1 <= topk <= NUM_TOKENS:
        raise ValueError("topk must be in [1, %d]" % NUM_TOKENS)
    if heat.shape[0] < 1:
        raise ValueError("empty batch")
    scores = torch.sigmoid(heat)
    values, indices = scores.topk(topk, dim=1)
    out = []
    for b in range(heat.shape[0]):
        row = []
        for j in range(topk):
            token = int(indices[b, j])
            ox = float(off[b, token, 0])
            oy = float(off[b, token, 1])
            row.append(token_offset_to_point(token, ox, oy))
        out.append(row)
    return out, values


def gaussian_target(size=CELLS, center=0.0, sigma=1.0):
    """One-dimensional Gaussian bump used by the heatmap target builder."""
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError("size must be a positive integer")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return torch.exp(
        -((torch.arange(size, dtype=torch.float32) - float(center)) ** 2)
        / (2.0 * sigma * sigma)
    )


class GroundingHead(nn.Module):
    """Patch heatmap classifier plus per-patch ``(dx, dy)`` offset regressor.

    ``forward`` consumes ``feats`` ``[B, 320, D]`` (projected or raw vision
    tokens) and ``instr_vec`` ``[B, D]`` (mean-pooled LM text state) and
    returns ``(heat_logits [B, 320], offsets [B, 320, 2])`` with offsets
    squashed to ``[0, 1]``. The instruction branch conditions every patch, so
    the head localises the element named by the instruction rather than any
    salient patch.
    """

    def __init__(self, dim=768, hidden=256):
        super().__init__()
        if not isinstance(dim, int) or not isinstance(hidden, int) or dim < 1 or hidden < 1:
            raise ValueError("dim and hidden must be positive integers")
        self.feat = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.cond = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU()
        )
        self.hm = nn.Linear(hidden, 1)
        self.off = nn.Linear(hidden, 2)
        nn.init.constant_(self.hm.bias, -4.0)  # start with low objectness
        self.dim = dim
        self.hidden = hidden

    def forward(self, feats, instr_vec):
        """Return ``(heat_logits [B, 320], offsets [B, 320, 2])``."""
        if feats.ndim != 3 or feats.shape[1] != NUM_TOKENS:
            raise ValueError("feats must be [B, %d, D]" % NUM_TOKENS)
        if feats.shape[2] != self.dim:
            raise ValueError("feats last dimension must be %d" % self.dim)
        if instr_vec.ndim != 2 or instr_vec.shape != (feats.shape[0], self.dim):
            raise ValueError("instr_vec must be [B, D] matching feats")
        hidden = self.feat(feats) + self.cond(instr_vec).unsqueeze(1)
        return self.hm(hidden).squeeze(-1), torch.sigmoid(self.off(hidden))


def focal_loss(pred_logits, target, alpha=2.0, beta=4.0):
    """CenterNet focal loss over a Gaussian heatmap with 1.0 at the centre.

    ``pred_logits`` and ``target`` are ``[B, 320]``; ``target`` values lie in
    ``[0, 1]`` and positives are ``target >= 1.0``. The loss is normalised by
    the number of positives (clamped to one), exactly as in the original
    training loop.
    """
    if pred_logits.shape != target.shape:
        raise ValueError("logits and target must have the same shape")
    if target.numel() and bool((target < 0).any()):
        raise ValueError("target values must be non-negative")
    p = torch.sigmoid(pred_logits).clamp(1e-4, 1 - 1e-4)
    pos = (target >= 1.0).to(p.dtype)
    neg = 1.0 - pos
    pos_loss = -(1 - p).pow(alpha) * torch.log(p) * pos
    neg_loss = -(1 - target).pow(beta) * p.pow(alpha) * torch.log(1 - p) * neg
    npos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / npos


def build_targets(points, device, sigma=1.5):
    """Build heatmap, offset and mask targets for a batch of click points.

    ``points`` is a sequence of ``(token, ox, oy)`` triples; an unlabelled
    sample is ``(None, 0.0, 0.0)``. Returns ``heat [B, 320]``, ``off
    [B, 320, 2]`` and ``mask [B]`` on ``device``; only the clicked cell carries
    a non-zero offset and ``mask`` marks rows that received supervision.
    ``sigma`` is kept for parity with the original task definition and marks
    the (unused) Gaussian width, because the 3x3 neighbour weights below are
    the original hard-coded 0.5 ** (|dr| + |dc|).
    """
    if not isinstance(points, (list, tuple)) or not points:
        raise ValueError("points must be a non-empty sequence")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    device = torch.device(device)
    batch = len(points)
    heat = torch.zeros(batch, NUM_TOKENS, device=device)
    off = torch.zeros(batch, NUM_TOKENS, 2, device=device)
    mask = torch.zeros(batch, device=device)
    for b, point in enumerate(points):
        if not isinstance(point, (list, tuple)) or len(point) != 3:
            raise ValueError("each point must be (token|None, ox, oy)")
        token, ox, oy = point
        if token is None:
            continue
        _check_int(token, "token")
        if not 0 <= token < NUM_TOKENS:
            raise ValueError("token %d outside [0, %d)" % (token, NUM_TOKENS))
        ox = float(ox)
        oy = float(oy)
        if not torch.isfinite(torch.tensor([ox, oy])).all():
            raise ValueError("offsets must be finite")
        if not (-1.0 <= ox <= 1.0 and -1.0 <= oy <= 1.0):
            raise ValueError("offsets must lie in [-1, 1]")
        mask[b] = 1.0
        heat[b, token] = 1.0
        tile, row, col = token_to_patch(token)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = row + dr, col + dc
                if 0 <= rr < PATCH_GRID and 0 <= cc < PATCH_GRID:
                    neighbour = patch_to_token(tile, rr, cc)
                    value = _NEIGHBOUR_WEIGHT[(dr, dc)]
                    if float(heat[b, neighbour]) < value:
                        heat[b, neighbour] = value
        off[b, token, 0] = ox
        off[b, token, 1] = oy
    return heat, off, mask
