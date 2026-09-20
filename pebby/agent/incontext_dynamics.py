"""In-context transition prediction for LS20 rule variants (action permutations).

A *game* is one hidden permutation of the four agent actions onto the engine directions plus
seven sequential levels.  The model here is a small causal transformer over the game's step
sequence.  Its token for step ``t`` is built from public fields only -- the before-state,
the four neighbouring tile classes, the agent action, level index / boundary flag, and the
*previous* step's public outcome -- and it predicts the outcome of step ``t`` (movement class,
after shape / colour / rotation, life lost).  Nothing about the permutation is ever an input:
``engine_action``, ``action_map`` and ``variant_id`` are labels for evaluation and splitting.

The question the module serves: does the transformer infer the mapping from the game's own
earlier steps (accuracy rising with step index on held-out permutations), or does it merely
memorise the training permutations?  :class:`MemorylessBaseline` is the control: the same
token embedding and MLP head applied to the current step alone, *without* the previous-step
outcome features, so it can only learn the population prior over permutations.

Previous-step outcome features.  Token ``t`` includes ``prev_action`` (agent action at ``t-1``)
and ``prev_movement`` (movement class of step ``t-1`` computed from public before/after
positions).  A transformer could recover both from consecutive tokens, but exposing them
turns mapping inference into a one-layer "find earlier step with the same action, copy its
movement" circuit that a ~200k-parameter model learns quickly.  They are set to an "unknown"
class at step 0 and after a level change / reset.  Pass ``include_previous=False`` to drop
them (the memoryless baseline always drops them).

Windowing.  Games can exceed ``max_steps``.  Training samples windows of at most ``max_steps``
steps, starting at the game start with probability ``start_bias`` (the mapping has to be
inferred from the earliest steps) and at a random offset otherwise.  Evaluation runs the whole
game in one pass when it fits and otherwise streams with a sliding window of ``max_steps``
advanced by ``stride`` steps; predictions for later steps then see only the last ``max_steps``
steps of history, which is a limitation to keep in mind if games are much longer than the
window.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from pebby.agent.variant_metrics import (FIELDS, FIELD_SIZES, score_predictions, target_masks,
                                         targets_from_arrays)

FORMAT = "incontext-dynamics-v1"
GRID = 12
SHAPES, COLORS, ROTATIONS = 6, 4, 4
TILE_CLASSES = 10
ACTIONS = 4
LIVES_CLASSES = 8
LEVEL_CLASSES = 8
GOAL_BITS = 16
STEPS_NORM = 42.0
PREV_ACTION_NONE = ACTIONS            # 5-way: 0..3 or "no previous step"
PREV_MOVEMENT_UNKNOWN = 5             # 6-way: movement 0..4 or "unknown"
FORBIDDEN_INPUTS = ("engine_action", "action_map", "variant_id")
DEFAULT_LOSS_WEIGHTS = {"movement": 2.0, "shape": 1.0, "color": 1.0, "rotation": 1.0, "life_lost": 1.0}

INPUT_KEYS = ("position", "glyph", "lives", "level", "boundary", "action", "neighbours", "scalars",
              "prev_action", "prev_movement")


# ------------------------------------------------------------------ featurisation
def _clip_int(arrays, key, high):
    return np.clip(np.asarray(arrays[key]).astype(np.int64), 0, high - 1)


def featurize(arrays):
    """Per-step input tokens and targets from a game's arrays (public fields only).

    Returns a dict of tensors with a leading ``T`` axis::

        position   int64 [T, 2]   before x, y (0..11)
        glyph      int64 [T, 3]   before shape, color, rotation
        lives      int64 [T]      before lives (clipped to 0..7)
        level      int64 [T]      level_index (clipped to 0..7)
        boundary   int64 [T]      1 on step 0 and whenever level_index differs from the previous step
        action     int64 [T]      agent action 0..3
        neighbours int64 [T, 4]   tile class of the up/down/left/right cell (engine order)
        scalars    float32 [T, 17] steps_left / 42 and the 16 goals_mask bits
        prev_action   int64 [T]   agent action of step t-1 (4 = none)
        prev_movement int64 [T]   movement class of step t-1 (5 = unknown / level start)
        targets    {field: int64 [T]}   see pebby.agent.variant_metrics
        masks      {field: bool  [T]}   which steps are scored for each field

    ``engine_action``, ``action_map`` and ``variant_id`` are never read.
    """
    for key in FORBIDDEN_INPUTS:
        assert key not in INPUT_KEYS  # documentation-level guard; these are never touched below
    x = _clip_int(arrays, "before_player_x", GRID)
    y = _clip_int(arrays, "before_player_y", GRID)
    steps = x.shape[0]
    glyph = np.stack([_clip_int(arrays, "before_shape", SHAPES), _clip_int(arrays, "before_color", COLORS),
                      _clip_int(arrays, "before_rotation", ROTATIONS)], axis=1)
    action = _clip_int(arrays, "agent_action", ACTIONS)
    neighbours = np.clip(np.asarray(arrays["neighbours"]).astype(np.int64), 0, TILE_CLASSES - 1)
    if neighbours.shape != (steps, 4):
        raise ValueError(f"neighbours must be [T, 4], got {neighbours.shape}")
    steps_left = np.clip(np.asarray(arrays["before_steps_left"]).astype(np.float32) / STEPS_NORM, 0.0, 2.0)
    goals = np.asarray(arrays["before_goals_mask"]).astype(np.int64) & 0xFFFF
    goal_bits = ((goals[:, None] >> np.arange(GOAL_BITS)) & 1).astype(np.float32)
    scalars = np.concatenate([steps_left[:, None], goal_bits], axis=1)
    level_index = np.asarray(arrays["level_index"]).astype(np.int64)
    boundary = np.ones(steps, dtype=np.int64)          # step 0 and every first step of a new level
    if steps > 1:
        boundary[1:] = (level_index[1:] != level_index[:-1]).astype(np.int64)

    targets = targets_from_arrays(arrays)
    masks = target_masks(arrays)
    prev_action = np.full(steps, PREV_ACTION_NONE, dtype=np.int64)
    prev_movement = np.full(steps, PREV_MOVEMENT_UNKNOWN, dtype=np.int64)
    if steps > 1:
        prev_action[1:] = action[:-1]
        known = masks["movement"][:-1]
        prev_movement[1:] = np.where(known, targets["movement"][:-1], PREV_MOVEMENT_UNKNOWN)

    to = torch.from_numpy
    return {
        "position": to(np.stack([x, y], axis=1)),
        "glyph": to(glyph),
        "lives": to(_clip_int(arrays, "before_lives", LIVES_CLASSES)),
        "level": to(_clip_int(arrays, "level_index", LEVEL_CLASSES)),
        "boundary": to(boundary),
        "action": to(action),
        "neighbours": to(neighbours),
        "scalars": to(np.ascontiguousarray(scalars)),
        "prev_action": to(prev_action),
        "prev_movement": to(prev_movement),
        "targets": {name: to(np.ascontiguousarray(value)) for name, value in targets.items()},
        "masks": {name: to(np.ascontiguousarray(value)) for name, value in masks.items()},
    }


def slice_features(features, start, end):
    out = {key: features[key][start:end] for key in INPUT_KEYS}
    out["targets"] = {name: value[start:end] for name, value in features["targets"].items()}
    out["masks"] = {name: value[start:end] for name, value in features["masks"].items()}
    return out


def sample_window(steps, max_steps, rng, start_bias=0.7):
    """[start, end) of a training window: the whole game if it fits, else ``max_steps`` steps
    starting at 0 with probability ``start_bias`` and at a random offset otherwise."""
    if steps <= max_steps:
        return 0, steps
    if rng.random() < start_bias:
        return 0, max_steps
    start = int(rng.integers(0, steps - max_steps + 1))
    return start, start + max_steps


def collate(windows):
    """Right-pad a list of feature dicts to a batch; ``pad_mask`` is True on padded steps and
    all target masks are False there."""
    length = max(int(w["action"].shape[0]) for w in windows)
    batch = {}
    for key in INPUT_KEYS:
        pieces = []
        for w in windows:
            value = w[key]
            pad = length - value.shape[0]
            pieces.append(F.pad(value, (0, 0, 0, pad) if value.dim() == 2 else (0, pad)))
        batch[key] = torch.stack(pieces)
    batch["targets"] = {}
    batch["masks"] = {}
    for name in FIELDS:
        batch["targets"][name] = torch.stack(
            [F.pad(w["targets"][name], (0, length - w["targets"][name].shape[0])) for w in windows])
        batch["masks"][name] = torch.stack(
            [F.pad(w["masks"][name], (0, length - w["masks"][name].shape[0]), value=False) for w in windows])
    batch["pad_mask"] = torch.stack(
        [F.pad(torch.zeros(int(w["action"].shape[0]), dtype=torch.bool),
               (0, length - int(w["action"].shape[0])), value=True) for w in windows])
    return batch


def batch_to(batch, device):
    out = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            out[key] = {k: v.to(device) for k, v in value.items()}
        else:
            out[key] = value.to(device)
    return out


# ------------------------------------------------------------------ model
@dataclass
class InContextConfig:
    d_model: int = 64
    layers: int = 3
    heads: int = 4
    ffn_multiplier: int = 2
    max_steps: int = 1024
    dropout: float = 0.0
    include_previous: bool = True
    # embedding widths
    coord_dim: int = 8
    glyph_dim: int = 6
    small_dim: int = 4
    action_dim: int = 8
    tile_dim: int = 6


class TokenEmbedder(nn.Module):
    """Concatenated small embeddings of the public per-step fields, projected to d_model."""

    def __init__(self, config: InContextConfig):
        super().__init__()
        c = config
        self.include_previous = c.include_previous
        self.x = nn.Embedding(GRID, c.coord_dim)
        self.y = nn.Embedding(GRID, c.coord_dim)
        self.shape = nn.Embedding(SHAPES, c.glyph_dim)
        self.color = nn.Embedding(COLORS, c.glyph_dim)
        self.rotation = nn.Embedding(ROTATIONS, c.glyph_dim)
        self.lives = nn.Embedding(LIVES_CLASSES, c.small_dim)
        self.level = nn.Embedding(LEVEL_CLASSES, c.small_dim)
        self.boundary = nn.Embedding(2, c.small_dim)
        self.action = nn.Embedding(ACTIONS, c.action_dim)
        self.tile = nn.Embedding(TILE_CLASSES, c.tile_dim)
        self.direction = nn.Parameter(torch.zeros(4, c.tile_dim))  # marks which neighbour slot
        width = (2 * c.coord_dim + 3 * c.glyph_dim + 3 * c.small_dim + c.action_dim + 4 * c.tile_dim
                 + 1 + GOAL_BITS)
        if c.include_previous:
            self.prev_action = nn.Embedding(ACTIONS + 1, c.action_dim)
            self.prev_movement = nn.Embedding(6, c.action_dim)
            width += 2 * c.action_dim
        self.project = nn.Linear(width, c.d_model)

    def forward(self, batch):
        tiles = self.tile(batch["neighbours"]) + self.direction  # [B, T, 4, tile_dim]
        parts = [self.x(batch["position"][..., 0]), self.y(batch["position"][..., 1]),
                 self.shape(batch["glyph"][..., 0]), self.color(batch["glyph"][..., 1]),
                 self.rotation(batch["glyph"][..., 2]), self.lives(batch["lives"]),
                 self.level(batch["level"]), self.boundary(batch["boundary"]),
                 self.action(batch["action"]), tiles.flatten(-2), batch["scalars"]]
        if self.include_previous:
            parts += [self.prev_action(batch["prev_action"]), self.prev_movement(batch["prev_movement"])]
        return self.project(torch.cat(parts, dim=-1))


class Heads(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, sum(FIELD_SIZES[f] for f in FIELDS))

    def forward(self, hidden):
        logits = self.linear(self.norm(hidden))
        out, offset = {}, 0
        for name in FIELDS:
            out[name] = logits[..., offset:offset + FIELD_SIZES[name]]
            offset += FIELD_SIZES[name]
        return out


class InContextDynamics(nn.Module):
    """Causal transformer over a game's step tokens; logits for the five outcome fields at
    every step, conditioned on steps ``<= t`` (the current before-state and action included)."""

    memoryless = False

    def __init__(self, config: InContextConfig | None = None, **overrides):
        super().__init__()
        self.config = InContextConfig(**{**asdict(config or InContextConfig()), **overrides})
        c = self.config
        if c.d_model % c.heads:
            raise ValueError("d_model must be divisible by heads")
        self.embed = TokenEmbedder(c)
        self.positions = nn.Embedding(c.max_steps, c.d_model)
        layer = nn.TransformerEncoderLayer(c.d_model, c.heads, c.ffn_multiplier * c.d_model, c.dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, c.layers, enable_nested_tensor=False)
        self.heads = Heads(c.d_model)
        nn.init.normal_(self.positions.weight, std=0.02)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, batch):
        tokens = self.embed(batch)
        steps = tokens.shape[1]
        if steps > self.config.max_steps:
            raise ValueError(f"sequence of {steps} steps exceeds max_steps={self.config.max_steps}")
        hidden = tokens + self.positions(torch.arange(steps, device=tokens.device))[None]
        causal = torch.full((steps, steps), float("-inf"), device=tokens.device).triu(1)
        pad = batch.get("pad_mask")
        pad_mask = None
        if pad is not None:
            pad_mask = torch.zeros(pad.shape, dtype=hidden.dtype, device=tokens.device)
            pad_mask = pad_mask.masked_fill(pad, float("-inf"))
        hidden = self.encoder(hidden, mask=causal.to(hidden.dtype), src_key_padding_mask=pad_mask,
                              is_causal=True)
        return self.heads(hidden)


class MemorylessBaseline(nn.Module):
    """Control: the same token embedding and heads, an MLP over the current step only
    (no attention, no previous-step outcome features), so nothing can be inferred in-context."""

    memoryless = True

    def __init__(self, config: InContextConfig | None = None, **overrides):
        super().__init__()
        self.config = InContextConfig(**{**asdict(config or InContextConfig()), **overrides,
                                         "include_previous": False})
        c = self.config
        self.embed = TokenEmbedder(self.config)
        blocks = []
        for _ in range(c.layers):
            blocks += [nn.LayerNorm(c.d_model), nn.Linear(c.d_model, c.ffn_multiplier * c.d_model), nn.GELU(),
                       nn.Linear(c.ffn_multiplier * c.d_model, c.d_model)]
        self.mlp = nn.Sequential(*blocks)
        self.heads = Heads(c.d_model)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, batch):
        hidden = self.embed(batch)
        return self.heads(hidden + self.mlp(hidden))


def build_model(memoryless=False, **config):
    return MemorylessBaseline(**config) if memoryless else InContextDynamics(**config)


# ------------------------------------------------------------------ loss / checkpoints
def prediction_loss(logits, batch, weights=None):
    """Summed masked cross-entropy over the five heads (movement weighted 2x by default).
    Returns (total, {field: float})."""
    weights = {**DEFAULT_LOSS_WEIGHTS, **(weights or {})}
    total = logits["movement"].new_zeros(())
    parts = {}
    for name in FIELDS:
        mask = batch["masks"][name] & ~batch["pad_mask"]
        if not mask.any():
            parts[name] = 0.0
            continue
        ce = F.cross_entropy(logits[name][mask], batch["targets"][name][mask])
        total = total + weights[name] * ce
        parts[name] = float(ce.detach())
    return total, parts


def save_checkpoint(model, path, **extra):
    torch.save({"format": FORMAT, "memoryless": model.memoryless, "config": asdict(model.config),
                "state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError(f"unexpected checkpoint format {payload.get('format')!r}")
    model = build_model(payload["memoryless"], **payload["config"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload


# ------------------------------------------------------------------ prediction / evaluation
def _windows_for(steps, max_steps, stride):
    """(start, end, keep_from) windows covering [0, steps) with each step predicted once."""
    if steps <= max_steps:
        return [(0, steps, 0)]
    stride = max(1, min(stride or max(1, max_steps // 4), max_steps))
    windows = [(0, max_steps, 0)]
    covered = max_steps
    while covered < steps:
        start = min(covered - max_steps + stride, steps - max_steps)
        end = start + max_steps
        windows.append((start, end, covered - start))
        covered = end
    return windows


@torch.inference_mode()
def predict_game(model, features, device=None, max_steps=None, stride=None):
    """Greedy per-step predictions (dict of int64 numpy arrays, length T) for one game,
    streaming with a sliding window when the game is longer than ``max_steps``."""
    device = device or next(model.parameters()).device
    limit = max_steps or (model.config.max_steps if not model.memoryless else 1 << 30)
    steps = int(features["action"].shape[0])
    was_training = model.training
    model.eval()
    out = {name: np.zeros(steps, dtype=np.int64) for name in FIELDS}
    for start, end, keep_from in _windows_for(steps, limit, stride):
        batch = batch_to(collate([slice_features(features, start, end)]), device)
        logits = model(batch)
        for name in FIELDS:
            guess = logits[name][0].argmax(-1).cpu().numpy()
            out[name][start + keep_from:end] = guess[keep_from:]
    if was_training:
        model.train()
    return out


def evaluate_game(model, arrays, device=None, max_steps=None, stride=None, features=None):
    """Score one game: per-step joint and per-field correctness, the accuracy curve by step
    bin, ``steps_to_stable`` and movement-only accuracy (see variant_metrics.score_predictions)."""
    features = features if features is not None else featurize(arrays)
    pred = predict_game(model, features, device=device, max_steps=max_steps, stride=stride)
    return score_predictions(pred, arrays)


def load_game(path):
    """Read one NPZ into a plain dict of numpy arrays (scalars become 0-d arrays)."""
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}
