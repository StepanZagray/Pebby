"""Experimental WorldPolicy with explicit learned public player-location recall.

The original projector features retain their exact ordering. A detached copy
of the learned player softmax contributes twelve row marginals followed by
twelve column marginals. Only this appended copy is detached: state recall,
the direct policy and query readout can still train the player head. No labels,
environment state, persistent state or inference-time normalization are added.

Use this module's checkpoint functions explicitly; the global policy loader
does not recognize this experimental format. Migration checks architecture and
weights, while the calling trainer owns source-data and checkpoint provenance.
"""

import json
import os
from pathlib import Path
import tempfile

import torch
from torch import nn

from .world_model import CELLS, GRID_COLS, GRID_ROWS, WorldModelConfig, WorldPolicy

POSITION_RECALL_FORMAT = "pebby.ls20-world-position-policy.v1"
POSITION_RECALL_VERSION = "row_column_marginals_v1"
POSITION_INPUTS = GRID_ROWS + GRID_COLS


def _position_source():
    return {"source": "player_head_softmax_of_public_refined_cells",
            "grid_rows": GRID_ROWS, "grid_columns": GRID_COLS,
            "ordering": "row_marginals_then_column_marginals",
            "detach": "appended_feature_copy_only"}


class PositionRecallPolicy(WorldPolicy):
    """The same public-history policy contract, with 24 appended features."""

    checkpoint_format = POSITION_RECALL_FORMAT

    def __init__(self, config=None, **overrides):
        if isinstance(config, dict):
            config = {**config, **overrides}
            version = config.pop("position_recall", POSITION_RECALL_VERSION)
            if version != POSITION_RECALL_VERSION:
                raise ValueError(f"unsupported position_recall {version!r}")
            config = WorldModelConfig.from_dict(config)
            overrides = {}
        super().__init__(config, **overrides)
        first = self.projector[0]
        self.position_inputs = POSITION_INPUTS
        extended = nn.Linear(first.in_features + POSITION_INPUTS, first.out_features,
                             bias=first.bias is not None,
                             device=first.weight.device, dtype=first.weight.dtype)
        with torch.no_grad():
            extended.weight.zero_()
            extended.weight[:, :first.in_features].copy_(first.weight)
            if first.bias is not None:
                extended.bias.copy_(first.bias)
        self.projector[0] = extended

    def config(self):
        return {**super().config(), "position_recall": POSITION_RECALL_VERSION}

    def projector_layout(self):
        layout = super().projector_layout()
        return [*layout, ("position", sum(width for _, _, width in layout), POSITION_INPUTS)]

    def position_features(self, state):
        """Refined public tokens -> detached row-then-column marginals [B, 24]."""
        expected = (self.tokens, self.cfg.channels)
        if state.ndim != 3 or tuple(state.shape[1:]) != expected:
            raise ValueError(f"state must be [B, {expected[0]}, {expected[1]}]")
        weights = self.player_weights(state[:, :CELLS])[1]
        if tuple(weights.shape) != (state.size(0), CELLS):
            raise ValueError(f"player weights must be [B, {CELLS}]")
        grid = weights.reshape(-1, GRID_ROWS, GRID_COLS)
        return torch.cat((grid.sum(dim=2), grid.sum(dim=1)), dim=-1).detach()

    def projector_inputs(self, state, raw_hud=None, glyph=None):
        # Validate shapes before the inherited projector can flatten bad input.
        position = self.position_features(state)
        batch = state.size(0)
        if self.cfg.state_recall and (raw_hud is None or tuple(raw_hud.shape) !=
                                     (batch, self.cfg.hud_tokens * self.cfg.channels)):
            raise ValueError("state_recall needs raw HUD [B, hud_tokens * channels]")
        if self.cfg.glyph_recall and (glyph is None or tuple(glyph.shape) !=
                                     (batch, self.glyph_inputs)):
            raise ValueError("glyph_recall needs glyph probabilities [B, 14]")
        original = super().projector_inputs(state, raw_hud, glyph)
        expected = self.base_inputs + self.recall_inputs + self.glyph_inputs
        if tuple(original.shape) != (batch, expected):
            raise ValueError("original projector input dimensions do not match config")
        return torch.cat((original, position.to(original.dtype)), dim=-1)


def _validate_weights(weights, expected):
    if not isinstance(weights, dict) or set(weights) != set(expected):
        raise ValueError("checkpoint weight keys do not match the model")
    for key, value in weights.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape:
            raise ValueError(f"checkpoint weight shape does not match: {key}")
        if value.dtype != expected[key].dtype or value.layout != torch.strided:
            raise ValueError(f"checkpoint weight dtype/layout does not match: {key}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"checkpoint weight is not finite: {key}")


def initialize_from_base(target, source):
    """Strict weights-only migration; only 24 zero columns are new.

    Source and target must have identical WorldModelConfig. Every parameter
    and buffer is copied, with the sole shape change at projector.0.weight.
    Forward outputs are mathematically preserved, subject to floating-point
    accumulation differences from the wider matrix multiplication.
    """
    if type(target) is not PositionRecallPolicy or type(source) is not WorldPolicy:
        raise ValueError("migration requires PositionRecallPolicy target and base WorldPolicy source")
    if target.cfg != source.cfg:
        raise ValueError("migration requires identical base configs")
    expected_layout = source.projector_layout()
    if target.projector_layout() != [*expected_layout,
                                    ("position", source.projector[0].in_features, POSITION_INPUTS)]:
        raise ValueError("migration projector layouts do not match")
    expected = target.state_dict()
    weights = {key: value.detach().clone() for key, value in source.state_dict().items()}
    key = "projector.0.weight"
    width = sum(width for _, _, width in expected_layout)
    if key not in weights or weights[key].shape != (target.projector[0].out_features, width):
        raise ValueError("source first projector shape does not match config")
    padded = weights[key].new_zeros((target.projector[0].out_features, width + POSITION_INPUTS))
    padded[:, :width] = weights[key]
    weights[key] = padded
    _validate_weights(weights, expected)
    target.load_state_dict(weights, strict=True)
    return [key]


def _validate_metadata(checkpoint):
    try:
        json.dumps({key: value for key, value in checkpoint.items() if key != "weights"},
                   allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint metadata must be finite JSON") from error


def save_checkpoint(path, model, **metadata):
    """Atomically save this subtype with finite JSON metadata and explicit semantics."""
    if type(model) is not PositionRecallPolicy:
        raise ValueError("save_checkpoint requires PositionRecallPolicy")
    reserved = {"format", "config", "parameters", "position_source", "weights"} & metadata.keys()
    if reserved:
        raise ValueError(f"reserved checkpoint metadata: {sorted(reserved)}")
    checkpoint = {"format": POSITION_RECALL_FORMAT, "config": model.config(),
                  "parameters": model.parameter_count(), "position_source": _position_source(),
                  **metadata, "weights": {key: value.detach().cpu().clone()
                                          for key, value in model.state_dict().items()}}
    _validate_metadata(checkpoint)
    _validate_weights(checkpoint["weights"], model.state_dict())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return checkpoint


def load_checkpoint(path, device="cpu"):
    """Load only this experiment's format, validate strictly, and return (eval model, metadata)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != POSITION_RECALL_FORMAT:
        raise ValueError(f"unsupported checkpoint format; expected {POSITION_RECALL_FORMAT!r}")
    _validate_metadata(checkpoint)
    config = checkpoint.get("config")
    expected_keys = set(WorldModelConfig().as_dict()) | {"position_recall"}
    if not isinstance(config, dict) or set(config) != expected_keys:
        raise ValueError("checkpoint config must contain exactly the full position-policy config")
    if config["position_recall"] != POSITION_RECALL_VERSION:
        raise ValueError("unsupported position_recall version")
    if checkpoint.get("position_source") != _position_source():
        raise ValueError("checkpoint position_source semantics do not match")
    model = PositionRecallPolicy(config)
    if type(checkpoint.get("parameters")) is not int or checkpoint["parameters"] != model.parameter_count():
        raise ValueError("checkpoint parameter count does not match the rebuilt network")
    _validate_weights(checkpoint.get("weights"), model.state_dict())
    model.load_state_dict(checkpoint["weights"], strict=True)
    return model.to(device).eval(), checkpoint
