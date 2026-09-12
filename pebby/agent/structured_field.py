"""Frozen public-history field encoder for the structured world-model path.

The encoder combines two frozen public-pixel models with the frozen world
encoder.  It has no engine, label, route, or gold-visibility inputs.  Its only
runtime inputs are the causal H8 frame history and the corresponding validity
and previous-action arrays.

Field layout is ``[B, 148, 96]``: 144 row-major board cells followed by four
pooled HUD tokens.  Channels are state (0:48), dense appearance (48:70),
carried-glyph probabilities (70:84), predicted public visibility (84), and
reserved zeros (85:96).  The visibility channel is a learned fallible signal;
it never gates or masks another channel.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import torch
from torch import nn

from .cell_visibility import CellVisibility, FORMAT as VISIBILITY_FORMAT
from .world_model import CELLS, GRID_COLS, GRID_ROWS, WORLD_MODEL_FORMAT, load_world_checkpoint


FIELD_FORMAT = "pebby.ls20-structured-field.v1"
FIELD_TOKENS = CELLS + 4
FIELD_CHANNELS = 96
STATE_CHANNELS = 48
APPEARANCE_CHANNELS = 22
GLYPH_CHANNELS = 14

DEFAULT_WORLD_CHECKPOINT = Path("checkpoints/ls20-world-cell-recall-b1024.pt")
DEFAULT_VISIBILITY_CHECKPOINT = Path("checkpoints/ls20-cell-visibility-initial-200.pt")


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def _required_hash(path: Path, expected: str, description: str) -> str:
    if not path.exists():
        raise ValueError(f"missing provenance source for {description}: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"provenance hash mismatch for {description}: {path}")
    return actual


def _verify_world_sources(checkpoint: dict, checkpoint_path: Path) -> dict:
    if checkpoint.get("format") != WORLD_MODEL_FORMAT:
        raise ValueError(f"unsupported world checkpoint format: {checkpoint.get('format')!r}")
    source = checkpoint.get("cell_source")
    if not isinstance(source, dict) or source.get("format") != "pebby.cell-appearance.v1":
        raise ValueError("world checkpoint lacks a bound generated cell-appearance source")
    source_path = Path(source.get("path", ""))
    _required_hash(source_path, source.get("sha256", ""), "cell appearance checkpoint")
    for metadata_key, hash_key in (("bank", "bank_sha256"), ("proof", "proof_sha256")):
        if metadata_key in source:
            _required_hash(Path(source[metadata_key]), source.get(hash_key, ""), metadata_key)
    if not isinstance(source.get("train_seeds"), list):
        raise ValueError("cell-appearance source must expose a training seed list")
    if source.get("validation_used_for_training_or_selection") is not False:
        raise ValueError("cell-appearance source permits validation selection")
    return {
        "format": source["format"],
        "checkpoint": str(source_path),
        "sha256": source["sha256"],
        "bank": source.get("bank"),
        "bank_sha256": source.get("bank_sha256"),
        "proof": source.get("proof"),
        "proof_sha256": source.get("proof_sha256"),
        "train_levels": len(set(source["train_seeds"])),
        "validation_levels": len(set(source.get("validation_seeds", []))),
        "bound_world_checkpoint": str(checkpoint_path),
    }


def _verify_visibility_checkpoint(checkpoint: dict, checkpoint_path: Path) -> dict:
    if checkpoint.get("format") != VISIBILITY_FORMAT:
        raise ValueError(f"unsupported visibility checkpoint format: {checkpoint.get('format')!r}")
    if checkpoint.get("input_contract") != "public full64x64 integer palette frame only":
        raise ValueError("visibility checkpoint input contract is not public-frame-only")
    if checkpoint.get("validation_used_for_training_or_selection") is not False:
        raise ValueError("visibility checkpoint provenance permits validation selection")
    model = CellVisibility()
    if checkpoint.get("parameters") != model.parameter_count():
        raise ValueError("visibility checkpoint parameter count mismatch")
    hashes = checkpoint.get("source_hashes")
    if not isinstance(hashes, dict):
        raise ValueError("visibility checkpoint lacks source hashes")
    source_hashes = {}
    for source, expected in hashes.items():
        source_path = Path(source)
        source_hashes[source] = _required_hash(source_path, expected, "visibility source")
    return {
        "format": checkpoint["format"],
        "checkpoint": str(checkpoint_path),
        "sha256": _sha256(checkpoint_path),
        "parameters": model.parameter_count(),
        "source_hashes": source_hashes,
        "input_contract": checkpoint["input_contract"],
        "initial_state_only": bool(checkpoint.get("initial_state_only")),
    }


class StructuredFieldEncoder(nn.Module):
    """Frozen public H8 history -> structured board/HUD field tensor."""

    def __init__(self, world_checkpoint=DEFAULT_WORLD_CHECKPOINT,
                 visibility_checkpoint=DEFAULT_VISIBILITY_CHECKPOINT, device="cpu"):
        super().__init__()
        self.world_checkpoint_path = Path(world_checkpoint)
        self.visibility_checkpoint_path = Path(visibility_checkpoint)
        world_sha_before = _sha256(self.world_checkpoint_path)
        self.world, world_checkpoint_data = load_world_checkpoint(self.world_checkpoint_path, device)
        world_sha_after = _sha256(self.world_checkpoint_path)
        if world_sha_before != world_sha_after:
            raise ValueError("world checkpoint changed while loading frozen weights")
        self.visibility = CellVisibility().to(device)
        visibility_sha_before = _sha256(self.visibility_checkpoint_path)
        visibility_checkpoint_data = torch.load(self.visibility_checkpoint_path, map_location="cpu", weights_only=True)
        visibility_sha_after = _sha256(self.visibility_checkpoint_path)
        if visibility_sha_before != visibility_sha_after:
            raise ValueError("visibility checkpoint changed while loading frozen weights")
        self.visibility.load_state_dict(visibility_checkpoint_data["weights"], strict=True)
        self.visibility.to(device)

        config = self.world.config()
        if config.get("channels") != STATE_CHANNELS:
            raise ValueError(f"structured field requires 48 state channels, got {config.get('channels')}")
        if config.get("history") != 8:
            raise ValueError(f"structured field requires H8 world encoder, got H{config.get('history')}")
        if not config.get("glyph_recall") or not config.get("cell_recall"):
            raise ValueError("structured field requires glyph_recall and cell_recall world features")
        if self.world.tokens != CELLS + config.get("hud_tokens", 0):
            raise ValueError("world token count does not match board plus HUD configuration")
        if config.get("hud_tokens", 0) < 1 or config["hud_tokens"] % 4:
            raise ValueError("HUD token count must be divisible into four output HUD tokens")

        self._config = {
            "format": FIELD_FORMAT,
            "tokens": FIELD_TOKENS,
            "board_tokens": CELLS,
            "hud_tokens": 4,
            "channels": FIELD_CHANNELS,
            "state_channels": STATE_CHANNELS,
            "appearance_channels": APPEARANCE_CHANNELS,
            "glyph_channels": GLYPH_CHANNELS,
            "visibility_channel": 84,
            "reserved_channels": [85, 96],
            "world": config,
            "world_parameters": self.world.parameter_count(),
            "visibility_parameters": self.visibility.parameter_count(),
        }
        self._sources = {
            "world_checkpoint": {"path": str(self.world_checkpoint_path),
                                 "sha256": world_sha_after,
                                 "format": world_checkpoint_data["format"]},
            "cell_appearance": _verify_world_sources(world_checkpoint_data, self.world_checkpoint_path),
            "visibility": _verify_visibility_checkpoint(visibility_checkpoint_data,
                                                          self.visibility_checkpoint_path),
        }
        self._sources["visibility"]["sha256"] = visibility_sha_after
        self._freeze()

    def _freeze(self):
        # Bypass this class's train override while recursively setting the two
        # source modules to eval mode.
        nn.Module.train(self, False)
        self.world.requires_grad_(False)
        self.visibility.requires_grad_(False)

    def train(self, mode=True):
        # This object is an evaluation boundary; accidentally entering train
        # mode must not change dropout/norm behavior in a frozen source.
        self._freeze()
        return self

    def config(self):
        return copy.deepcopy(self._config)

    def metadata(self):
        return {"config": self.config(), "sources": copy.deepcopy(self._sources),
                "parameter_counts": self.parameter_counts()}

    def parameter_counts(self):
        return {"world": self.world.parameter_count(),
                "visibility": self.visibility.parameter_count(),
                "total": self.world.parameter_count() + self.visibility.parameter_count()}

    @classmethod
    def from_checkpoints(cls, world_checkpoint=DEFAULT_WORLD_CHECKPOINT,
                         visibility_checkpoint=DEFAULT_VISIBILITY_CHECKPOINT, device="cpu"):
        return cls(world_checkpoint, visibility_checkpoint, device)

    @staticmethod
    def _inputs(frames, history_valid, previous_actions, device):
        frames = torch.as_tensor(frames)
        if frames.ndim != 4 or tuple(frames.shape[-2:]) != (64, 64):
            raise ValueError("frames must be [B, 8, 64, 64]")
        if frames.shape[1] != 8 or frames.shape[0] < 1:
            raise ValueError("structured field requires a non-empty H8 frame history")
        if frames.dtype.is_floating_point or frames.dtype == torch.bool:
            raise ValueError("frames must contain integer palette indices")
        if bool(((frames < 0) | (frames > 15)).any()):
            raise ValueError("palette indices must be within 0..15")
        batch = frames.shape[0]
        if history_valid is None:
            history_valid = torch.ones((batch, 8), dtype=torch.bool)
        else:
            history_valid = torch.as_tensor(history_valid)
            if history_valid.dtype != torch.bool:
                raise ValueError("history_valid must contain boolean values")
        if previous_actions is None:
            previous_actions = torch.full((batch, 8), -1, dtype=torch.long)
        else:
            previous_actions = torch.as_tensor(previous_actions)
            if previous_actions.dtype == torch.bool or previous_actions.dtype.is_floating_point:
                raise ValueError("previous_actions must contain integer action indices")
            previous_actions = previous_actions.long()
        if tuple(history_valid.shape) != (batch, 8) or tuple(previous_actions.shape) != (batch, 8):
            raise ValueError("history_valid and previous_actions must be [B, 8]")
        if not bool(history_valid[:, -1].all()):
            raise ValueError("the current history slot must be valid")
        if bool(((previous_actions < -1) | (previous_actions >= 4)).any()):
            raise ValueError("previous_actions must be in -1..3")
        if bool((previous_actions[~history_valid] != -1).any()):
            raise ValueError("padded history actions must be -1")
        if bool((history_valid[:, :-1] & ~history_valid[:, 1:]).any()):
            raise ValueError("history_valid must be a left-padded contiguous suffix")
        return frames.to(device=device, dtype=torch.long), history_valid.to(device), previous_actions.to(device)

    @torch.no_grad()
    def forward(self, frames, history_valid=None, previous_actions=None):
        device = next(self.world.parameters()).device
        frames, history_valid, previous_actions = self._inputs(frames, history_valid, previous_actions, device)
        encoding = self.world.encode(frames, history_valid, previous_actions)
        state = encoding["state"]
        expected_tokens = CELLS + self.world.cfg.hud_tokens
        if tuple(state.shape[1:]) != (expected_tokens, STATE_CHANNELS):
            raise RuntimeError(f"unexpected world state shape {tuple(state.shape)}")
        if self.world.cfg.hud_tokens % 4:
            raise RuntimeError("world HUD token count cannot be pooled into four tokens")
        hud_factor = self.world.cfg.hud_tokens // 4
        field = state.new_zeros((frames.shape[0], FIELD_TOKENS, FIELD_CHANNELS))
        field[:, :CELLS, :STATE_CHANNELS] = state[:, :CELLS]
        hud = state[:, CELLS:].reshape(frames.shape[0], 4, hud_factor, STATE_CHANNELS).mean(2)
        field[:, CELLS:, :STATE_CHANNELS] = hud

        role_logits, *attribute_logits = self.world.cell_appearance(frames[:, -1])
        appearance = torch.cat([role_logits.sigmoid(), *(logits.softmax(-1) for logits in attribute_logits)], dim=-1)
        if tuple(appearance.shape) != (frames.shape[0], CELLS, APPEARANCE_CHANNELS):
            raise RuntimeError(f"unexpected appearance shape {tuple(appearance.shape)}")
        field[:, :CELLS, 48:70] = appearance

        glyph = encoding.get("glyph")
        if glyph is None or tuple(glyph.shape) != (frames.shape[0], GLYPH_CHANNELS):
            raise RuntimeError("world checkpoint did not return grouped carried-glyph probabilities")
        # Glyph state is global public evidence, so it is broadcast across board
        # and HUD tokens. It is not a label or a hidden-state lookup.
        field[:, :, 70:84] = glyph[:, None, :]

        visibility = self.visibility(frames[:, -1]).sigmoid()
        if tuple(visibility.shape) != (frames.shape[0], CELLS):
            raise RuntimeError(f"unexpected visibility shape {tuple(visibility.shape)}")
        field[:, :CELLS, 84] = visibility
        # HUD is rendered in the public frame and has no board-cell fog label.
        field[:, CELLS:, 84] = 1.
        return field


def load_structured_field_encoder(world_checkpoint=DEFAULT_WORLD_CHECKPOINT,
                                  visibility_checkpoint=DEFAULT_VISIBILITY_CHECKPOINT,
                                  device="cpu"):
    return StructuredFieldEncoder(world_checkpoint, visibility_checkpoint, device)


FrozenPublicFieldAssembler = StructuredFieldEncoder


__all__ = ["FIELD_FORMAT", "StructuredFieldEncoder", "FrozenPublicFieldAssembler",
           "load_structured_field_encoder"]
