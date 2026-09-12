"""Learned perception of the carried HUD glyph (a public, fixed 6x6 crop).

The carried token is drawn at rows 55..60, columns 3..8 of every public frame
(see ``pebby/agent/model.py``, HUD crop notes). ``GlyphEncoder`` is a small
trainable classifier over that crop -- one-hot palette planes flattened to 576
features, one GELU hidden layer, 14 logits split into shape (6), colour (4) and
rotation (4). It is not a template matcher and reads no game state: only the
pixels of the crop enter. The world policy can optionally consume its grouped
softmax (``WorldModelConfig.glyph_recall``); ``glyph_train`` pretrains it on the
generated-only glyph NPZ splits built by ``tools/build_glyph_data.py``.

Depends on torch only; nothing here imports the game.
"""

import json
import os
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

GLYPH_FORMAT = "pebby.ls20-glyph-encoder.v1"
GLYPH_SOURCE = "generated_only"

# Crop geometry: half-open pixel ranges of the carried glyph in a 64x64 frame.
GLYPH_ROWS = (55, 61)
GLYPH_COLUMNS = (3, 9)
GLYPH_SIZE = GLYPH_ROWS[1] - GLYPH_ROWS[0]
PALETTE = 16
GLYPH_INPUTS = GLYPH_SIZE * GLYPH_SIZE * PALETTE  # 576
GLYPH_HIDDEN = 64
GLYPH_FIELDS = ("shape", "color", "rotation")
GLYPH_SIZES = (6, 4, 4)
GLYPH_CLASSES = sum(GLYPH_SIZES)  # 14

# Provenance every glyph checkpoint must carry (besides format/config/weights).
REQUIRED_METADATA = ("source", "train_seeds", "validation_seeds", "counts", "results")


def crop_glyph(frames):
    """``[..., 64, 64]`` frames -> ``[..., 6, 6]`` carried-glyph crop (a view)."""
    if frames.dim() < 2 or frames.shape[-2:] != (64, 64):
        raise ValueError(f"expected [..., 64, 64] frames, got {tuple(frames.shape)}")
    return frames[..., GLYPH_ROWS[0]:GLYPH_ROWS[1], GLYPH_COLUMNS[0]:GLYPH_COLUMNS[1]]


def glyph_probabilities(logits):
    """Grouped softmax: 14 logits -> 14 probabilities that sum to one per field."""
    return torch.cat([scores.softmax(-1) for scores in logits.split(GLYPH_SIZES, -1)], dim=-1)


def check_triples(triples, count=None):
    """``[N, 3]`` long labels inside the shape/colour/rotation ranges."""
    triples = torch.as_tensor(triples).long()
    if triples.dim() != 2 or triples.size(1) != 3 or (count is not None and triples.size(0) != count):
        raise ValueError(f"glyph triples must be [{'N' if count is None else count}, 3], got {tuple(triples.shape)}")
    limits = torch.tensor(GLYPH_SIZES, device=triples.device)
    if bool(((triples < 0) | (triples >= limits)).any()):
        raise ValueError("glyph triples must lie in 0..5 (shape), 0..3 (colour), 0..3 (rotation)")
    return triples


class GlyphEncoder(nn.Module):
    """One-hot 6x6 crop -> 14 logits (shape 6 | colour 4 | rotation 4)."""

    def __init__(self, hidden=GLYPH_HIDDEN):
        super().__init__()
        if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 1:
            raise ValueError("hidden must be a positive integer")
        self.hidden = hidden
        self.mlp = nn.Sequential(nn.Linear(GLYPH_INPUTS, hidden), nn.GELU(), nn.Linear(hidden, GLYPH_CLASSES))

    def config(self):
        return {"format": GLYPH_FORMAT, "hidden": self.hidden, "inputs": GLYPH_INPUTS,
                "classes": list(GLYPH_SIZES), "crop": {"rows": list(GLYPH_ROWS), "columns": list(GLYPH_COLUMNS)}}

    @classmethod
    def from_config(cls, config):
        config = dict(config or {})
        expected = cls(config.get("hidden", GLYPH_HIDDEN)).config()
        if config != expected:
            raise ValueError(f"glyph encoder config mismatch: expected {expected}, got {config}")
        return cls(expected["hidden"])

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, glyphs):
        """``[..., 6, 6]`` colour indices -> ``[..., 14]`` logits."""
        glyphs = glyphs if torch.is_tensor(glyphs) else torch.as_tensor(glyphs)
        if glyphs.shape[-2:] != (GLYPH_SIZE, GLYPH_SIZE):
            raise ValueError(f"glyphs must be [..., {GLYPH_SIZE}, {GLYPH_SIZE}], got {tuple(glyphs.shape)}")
        glyphs = glyphs.to(device=self.mlp[0].weight.device, dtype=torch.long)
        if bool(((glyphs < 0) | (glyphs >= PALETTE)).any()):
            raise ValueError("glyph pixels must be palette indices 0..15")
        planes = F.one_hot(glyphs, num_classes=PALETTE).flatten(-3).to(self.mlp[0].weight.dtype)
        return self.mlp(planes)

    def classify_frames(self, frames):
        """``[..., 64, 64]`` frames -> logits of their carried glyph."""
        return self(crop_glyph(frames if torch.is_tensor(frames) else torch.as_tensor(frames)))

    @staticmethod
    def probabilities(logits):
        return glyph_probabilities(logits)

    @staticmethod
    def loss(logits, triples):
        """Cross-entropy averaged over the three fields (and the rows)."""
        triples = check_triples(triples, logits.size(0)).to(logits.device)
        return torch.stack([F.cross_entropy(scores, triples[:, index])
                            for index, scores in enumerate(logits.split(GLYPH_SIZES, -1))]).mean()

    @staticmethod
    def accuracies(logits, triples):
        """Per-field accuracies ``[3]`` and the joint (all three right) accuracy."""
        triples = check_triples(triples, logits.size(0)).to(logits.device)
        hits = torch.stack([scores.argmax(-1) == triples[:, index]
                            for index, scores in enumerate(logits.split(GLYPH_SIZES, -1))], dim=-1)
        return hits.float().mean(0), hits.all(-1).float().mean()


# ------------------------------------------------------------------- IO
def validate_glyph_metadata(metadata):
    """Provenance a glyph checkpoint must carry: generated-only, disjoint seeds, counts, results."""
    missing = [key for key in REQUIRED_METADATA if key not in metadata]
    if missing:
        raise ValueError(f"glyph checkpoint lacks provenance: {missing}")
    if metadata["source"] != GLYPH_SOURCE:
        raise ValueError(f"glyph checkpoint source must be {GLYPH_SOURCE!r}, got {metadata['source']!r}")
    seeds = {}
    for split in ("train_seeds", "validation_seeds"):
        values = metadata[split]
        if not isinstance(values, list) or not values or not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            raise ValueError(f"glyph checkpoint {split} must be a non-empty list of integers")
        seeds[split] = set(values)
        if len(seeds[split]) != len(values):
            raise ValueError(f"glyph checkpoint {split} repeats level seeds")
    if seeds["train_seeds"] & seeds["validation_seeds"]:
        raise ValueError("glyph checkpoint train and validation seeds overlap")
    for key in ("counts", "results"):
        if not isinstance(metadata[key], dict) or not metadata[key]:
            raise ValueError(f"glyph checkpoint {key} must be a non-empty dictionary")


def save_glyph_checkpoint(path, model, **metadata):
    """Atomically save ``model`` with mandatory provenance metadata."""
    if not isinstance(model, GlyphEncoder):
        raise ValueError("only a GlyphEncoder can be saved as a glyph checkpoint")
    reserved = {"format", "config", "parameters", "weights"} & metadata.keys()
    if reserved:
        raise ValueError(f"reserved checkpoint metadata: {sorted(reserved)}")
    validate_glyph_metadata(metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"format": GLYPH_FORMAT, "config": model.config(), "parameters": model.parameter_count(),
                  **metadata, "weights": {key: value.detach().to("cpu") for key, value in model.state_dict().items()}}
    json.dumps({key: value for key, value in checkpoint.items() if key != "weights"}, allow_nan=False)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return checkpoint


def load_glyph_checkpoint(path, device="cpu"):
    """Rebuild a saved encoder strictly; returns ``(model, checkpoint)``."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != GLYPH_FORMAT:
        raise ValueError(f"unsupported glyph checkpoint format {checkpoint.get('format') if isinstance(checkpoint, dict) else None!r}, "
                         f"expected {GLYPH_FORMAT!r}")
    if not isinstance(checkpoint.get("config"), dict) or not isinstance(checkpoint.get("weights"), dict):
        raise ValueError("glyph checkpoint needs a config dictionary and a weights dictionary")
    validate_glyph_metadata(checkpoint)
    model = GlyphEncoder.from_config(checkpoint["config"])
    model.load_state_dict(checkpoint["weights"], strict=True)
    if checkpoint.get("parameters") != model.parameter_count():
        raise ValueError("glyph checkpoint parameter count does not match the rebuilt network")
    return model.to(device).eval(), checkpoint
