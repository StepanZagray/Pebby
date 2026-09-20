"""Causal visual policy and world-model auxiliaries for whole ARC-AGI-3 games.

The policy side of this module has a deliberately narrow data boundary.  At
decision time ``t`` it can use the public frame and state at ``t``, legal
actions, and the action/event observed at ``t-1``.  The chosen action at ``t``
and state ``t+1`` are accepted only by the separate transition-prediction
head.  Teacher targets and provenance never enter the model input mapping.

One loaded item is one complete sequential game.  The recurrent memory is not
reset at level boundaries and has no fixed history window; callers reset it by
creating a fresh ``initial_memory`` for a new game.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import Dataset


FORMAT = "pebby-multigame-model-v1"
FRAME_SIZE = 64
# Side of the coarse feature grid left by three stride-2 convolutions (64 -> 8).
GRID_SIZE = FRAME_SIZE // 8
PALETTE_SIZE = 16
ACTION_COUNT = 8
CLICK_ACTION = 6
COORDINATE_NONE = -1
EVENT_NAMES = ("level_boundary", "terminal", "won")

# These are the only keys selected by encode_history/policy.  In particular,
# executed_action_*, next_*, target_*, source, source_id, and planner fields are
# absent.  Keeping the list public lets trainers assert the boundary directly.
MODEL_INPUT_KEYS = (
    "frames",
    "previous_action_id",
    "previous_action_x",
    "previous_action_y",
    "previous_level_boundary",
    "terminal",
    "won",
    "legal_action_mask",
    "padding_mask",
)

_PUBLIC_REQUIRED = (
    "frames",
    "legal_action_mask",
    "state",
    "level_index",
    "levels_completed",
    "terminal",
    "won",
    "action_id",
    "action_x",
    "action_y",
    "level_boundary",
)
_TEACHER_REQUIRED = ("target_action_id", "target_action_x", "target_action_y", "source")


@dataclass(frozen=True)
class GameSequence:
    """Validated, causal arrays for one whole game.

    Policy targets are optional so the same public loader can feed evaluation.
    Executed actions and successor fields are auxiliary labels, never policy
    inputs.  Arrays are unpadded; batching adds right padding.
    """

    frames: np.ndarray
    previous_action_id: np.ndarray
    previous_action_x: np.ndarray
    previous_action_y: np.ndarray
    previous_level_boundary: np.ndarray
    terminal: np.ndarray
    won: np.ndarray
    legal_action_mask: np.ndarray
    executed_action_id: np.ndarray
    executed_action_x: np.ndarray
    executed_action_y: np.ndarray
    next_frames: np.ndarray
    next_level_boundary: np.ndarray
    next_terminal: np.ndarray
    next_won: np.ndarray
    target_action_id: np.ndarray | None = None
    target_action_x: np.ndarray | None = None
    target_action_y: np.ndarray | None = None
    target_valid: np.ndarray | None = None
    action_source: np.ndarray | None = None
    # Optional set-valued click label: bool [T,64,64], true on every pixel the
    # collector verified to be equivalent to the exact teacher click.  ``None``
    # for public-only games and for teacher files written before regions
    # existed; those fall back to the exact pixel (see ``click_region_or_exact``).
    target_click_region: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    @property
    def supervised(self) -> bool:
        return self.target_action_id is not None

    @property
    def has_click_regions(self) -> bool:
        return self.target_click_region is not None


def click_region_or_exact(game: GameSequence) -> np.ndarray:
    """Bool [T,64,64] click label: the stored region, else the exact pixel."""
    if game.target_click_region is not None:
        return game.target_click_region
    if game.target_action_id is None:
        raise ValueError("click regions require a supervised game")
    region = np.zeros((len(game), FRAME_SIZE, FRAME_SIZE), dtype=np.bool_)
    assert game.target_valid is not None and game.target_action_x is not None
    assert game.target_action_y is not None
    rows = np.flatnonzero(game.target_valid & (game.target_action_id == CLICK_ACTION))
    region[rows, game.target_action_y[rows], game.target_action_x[rows]] = True
    return region


def _read_npz(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _require_keys(arrays: Mapping[str, np.ndarray], keys: Sequence[str], *, kind: str) -> None:
    missing = [key for key in keys if key not in arrays]
    if missing:
        raise ValueError(f"{kind} NPZ is missing required arrays: {', '.join(missing)}")


def _require_shape(array: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")


def _require_bool(array: np.ndarray, name: str) -> None:
    if array.dtype != np.bool_:
        raise ValueError(f"{name} must have bool dtype, got {array.dtype}")


def _require_integer(array: np.ndarray, name: str) -> None:
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must have integer dtype, got {array.dtype}")


def _validate_action_triples(
    action_id: np.ndarray,
    action_x: np.ndarray,
    action_y: np.ndarray,
    *,
    name: str,
    allow_missing: bool,
) -> np.ndarray:
    if not (action_id.ndim == action_x.ndim == action_y.ndim == 1):
        raise ValueError(f"{name} action arrays must be one-dimensional")
    if not (len(action_id) == len(action_x) == len(action_y)):
        raise ValueError(f"{name} action arrays must have identical lengths")
    for key, value in (("id", action_id), ("x", action_x), ("y", action_y)):
        _require_integer(value, f"{name}_action_{key}")
    valid = action_id != COORDINATE_NONE if allow_missing else np.ones(len(action_id), dtype=np.bool_)
    if np.any(valid & ((action_id < 1) | (action_id >= ACTION_COUNT))):
        raise ValueError(f"{name} action IDs must be in 1..7" + (" or -1" if allow_missing else ""))
    if allow_missing and np.any(~valid & (action_id != COORDINATE_NONE)):
        raise ValueError(f"{name} missing action IDs must be -1")
    click = valid & (action_id == CLICK_ACTION)
    if np.any(click & ((action_x < 0) | (action_x >= FRAME_SIZE) |
                       (action_y < 0) | (action_y >= FRAME_SIZE))):
        raise ValueError(f"{name} click coordinates must be integers in 0..63")
    if np.any(~click & ((action_x != COORDINATE_NONE) | (action_y != COORDINATE_NONE))):
        raise ValueError(f"{name} non-click or missing actions must use coordinates -1,-1")
    return valid


def _validate_public(arrays: Mapping[str, np.ndarray]) -> int:
    _require_keys(arrays, _PUBLIC_REQUIRED, kind="public")
    leaked = sorted(key for key in arrays if key.startswith("target_") or key in {"source_id", "teacher_plan"})
    if leaked:
        raise ValueError(f"public NPZ contains private/teacher arrays: {', '.join(leaked)}")

    frames = arrays["frames"]
    if frames.ndim != 3 or frames.shape[1:] != (FRAME_SIZE, FRAME_SIZE):
        raise ValueError(f"frames must be [N+1,64,64], got {frames.shape}")
    if frames.dtype != np.uint8:
        raise ValueError(f"frames must have uint8 dtype, got {frames.dtype}")
    if len(frames) < 2:
        raise ValueError("a whole-game file must contain at least one transition")
    if np.any(frames >= PALETTE_SIZE):
        raise ValueError("frames contain palette values outside 0..15")
    steps = len(frames) - 1

    _require_shape(arrays["legal_action_mask"], (steps + 1, ACTION_COUNT), "legal_action_mask")
    _require_bool(arrays["legal_action_mask"], "legal_action_mask")
    for key in ("state", "level_index", "levels_completed", "terminal", "won"):
        _require_shape(arrays[key], (steps + 1,), key)
    for key in ("terminal", "won"):
        _require_bool(arrays[key], key)
    for key in ("level_index", "levels_completed"):
        _require_integer(arrays[key], key)
        if np.any(arrays[key] < 0):
            raise ValueError(f"{key} cannot be negative")
    if np.any(arrays["won"] & ~arrays["terminal"]):
        raise ValueError("won states must also be terminal")
    if np.any(arrays["terminal"][:-1]):
        raise ValueError("a terminal public state cannot have a following executed action")

    for key in ("action_id", "action_x", "action_y", "level_boundary"):
        _require_shape(arrays[key], (steps,), key)
    _require_bool(arrays["level_boundary"], "level_boundary")
    _validate_action_triples(
        arrays["action_id"], arrays["action_x"], arrays["action_y"],
        name="executed", allow_missing=False,
    )
    if np.any(arrays["legal_action_mask"][:-1, 0]):
        raise ValueError("RESET/action 0 must not be advertised as a policy action")
    if np.any(~arrays["legal_action_mask"][np.arange(steps), arrays["action_id"].astype(np.int64)]):
        raise ValueError("an executed action is illegal in its aligned public state")
    if np.any(~arrays["legal_action_mask"][:-1, 1:].any(axis=1)):
        raise ValueError("every decision state must advertise at least one action in 1..7")

    completed = arrays["levels_completed"].astype(np.int64)
    delta = completed[1:] - completed[:-1]
    if np.any((delta < 0) | (delta > 1)):
        raise ValueError("levels_completed must be monotone and advance at most once per transition")
    if not np.array_equal(arrays["level_boundary"], delta == 1):
        raise ValueError("level_boundary must exactly match levels_completed transitions")
    return steps


def _sequence_from_public(arrays: Mapping[str, np.ndarray]) -> GameSequence:
    steps = _validate_public(arrays)
    action_id = arrays["action_id"].astype(np.int64, copy=True)
    action_x = arrays["action_x"].astype(np.int64, copy=True)
    action_y = arrays["action_y"].astype(np.int64, copy=True)
    boundary = arrays["level_boundary"].astype(np.bool_, copy=True)

    previous_id = np.full(steps, COORDINATE_NONE, dtype=np.int64)
    previous_x = np.full(steps, COORDINATE_NONE, dtype=np.int64)
    previous_y = np.full(steps, COORDINATE_NONE, dtype=np.int64)
    previous_boundary = np.zeros(steps, dtype=np.bool_)
    if steps > 1:
        previous_id[1:] = action_id[:-1]
        previous_x[1:] = action_x[:-1]
        previous_y[1:] = action_y[:-1]
        previous_boundary[1:] = boundary[:-1]

    return GameSequence(
        frames=arrays["frames"][:-1].copy(),
        previous_action_id=previous_id,
        previous_action_x=previous_x,
        previous_action_y=previous_y,
        previous_level_boundary=previous_boundary,
        terminal=arrays["terminal"][:-1].astype(np.bool_, copy=True),
        won=arrays["won"][:-1].astype(np.bool_, copy=True),
        legal_action_mask=arrays["legal_action_mask"][:-1].astype(np.bool_, copy=True),
        executed_action_id=action_id,
        executed_action_x=action_x,
        executed_action_y=action_y,
        next_frames=arrays["frames"][1:].copy(),
        next_level_boundary=boundary,
        next_terminal=arrays["terminal"][1:].astype(np.bool_, copy=True),
        next_won=arrays["won"][1:].astype(np.bool_, copy=True),
    )


def load_public_game(path: str | Path) -> GameSequence:
    """Load one public collector NPZ without creating policy labels."""
    return _sequence_from_public(_read_npz(path))


def load_supervised_game(public_path: str | Path, teacher_path: str | Path) -> GameSequence:
    """Load one whole game and its physically separate aligned teacher labels."""
    sequence = load_public_game(public_path)
    teacher = _read_npz(teacher_path)
    _require_keys(teacher, _TEACHER_REQUIRED, kind="teacher")
    forbidden = sorted(set(teacher) & set(_PUBLIC_REQUIRED))
    if forbidden:
        raise ValueError(f"teacher NPZ contains public model arrays: {', '.join(forbidden)}")
    steps = len(sequence)
    for key, value in teacher.items():
        if value.ndim == 0 or value.shape[0] != steps:
            raise ValueError(f"teacher array {key} must align to {steps} transitions, got {value.shape}")
    # Validate original dtypes before normalization.  Casting first would turn
    # malformed floats/bools into apparently valid integer action labels.
    raw_target_id = teacher["target_action_id"]
    raw_target_x = teacher["target_action_x"]
    raw_target_y = teacher["target_action_y"]
    target_valid = _validate_action_triples(
        raw_target_id, raw_target_x, raw_target_y, name="target", allow_missing=True,
    )
    target_id = raw_target_id.astype(np.int64, copy=True)
    target_x = raw_target_x.astype(np.int64, copy=True)
    target_y = raw_target_y.astype(np.int64, copy=True)
    source = teacher["source"]
    _require_integer(source, "teacher source")
    if np.any((source < 0) | (source > 1)):
        raise ValueError("teacher source must be 0 (non-teacher) or 1 (exact teacher)")
    rows = np.arange(steps)[target_valid]
    if np.any(~sequence.legal_action_mask[rows, target_id[target_valid]]):
        raise ValueError("a teacher target is illegal in its aligned public state")
    exact = source == 1
    if np.any(exact & ~target_valid):
        raise ValueError("exact-teacher source rows must have a target action")
    if (not np.array_equal(target_id[exact], sequence.executed_action_id[exact]) or
            not np.array_equal(target_x[exact], sequence.executed_action_x[exact]) or
            not np.array_equal(target_y[exact], sequence.executed_action_y[exact])):
        raise ValueError("exact-teacher targets do not match their aligned executed transitions")
    region = _validate_click_region(teacher, target_id, target_x, target_y, target_valid)
    return GameSequence(
        **{field: getattr(sequence, field) for field in sequence.__dataclass_fields__
           if not field.startswith("target_") and field != "action_source"},
        target_action_id=target_id,
        target_action_x=target_x,
        target_action_y=target_y,
        target_valid=target_valid,
        action_source=source.astype(np.int64, copy=True),
        target_click_region=region,
    )


def _validate_click_region(
    teacher: Mapping[str, np.ndarray],
    target_id: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    target_valid: np.ndarray,
) -> np.ndarray | None:
    """Validate an optional ``click_region_mask`` against the exact targets."""
    if "click_region_mask" not in teacher:
        if "click_region_size" in teacher:
            raise ValueError("teacher click_region_size requires click_region_mask")
        return None
    raw = teacher["click_region_mask"]
    steps = int(target_id.shape[0])
    if raw.shape != (steps, FRAME_SIZE, FRAME_SIZE):
        raise ValueError(f"click_region_mask must be [{steps},64,64], got {raw.shape}")
    if raw.dtype == np.bool_:
        region = raw.copy()
    elif np.issubdtype(raw.dtype, np.integer):
        if np.any((raw < 0) | (raw > 1)):
            raise ValueError("click_region_mask values must be 0 or 1")
        region = raw.astype(np.bool_)
    else:
        raise ValueError("click_region_mask must be bool or integer 0/1")
    click = target_valid & (target_id == CLICK_ACTION)
    if np.any(region[~click]):
        raise ValueError("click_region_mask must be empty on non-click rows")
    rows = np.flatnonzero(click)
    if not np.all(region[rows, target_y[rows], target_x[rows]]):
        raise ValueError("every click region must contain its exact teacher target pixel")
    if "click_region_size" in teacher:
        size = teacher["click_region_size"]
        _require_integer(size, "click_region_size")
        if size.shape != (steps,) or not np.array_equal(size.astype(np.int64), region.sum(axis=(1, 2))):
            raise ValueError("click_region_size disagrees with click_region_mask")
    return region


class MultiGameSequenceDataset(Dataset[GameSequence]):
    """Dataset of already validated whole games; it never splices games."""

    def __init__(self, games: Sequence[GameSequence]):
        if not games:
            raise ValueError("at least one whole game is required")
        modes = {game.supervised for game in games}
        if len(modes) != 1:
            raise ValueError("a dataset cannot mix supervised and public-only games")
        self.games = tuple(games)

    @classmethod
    def from_paths(
        cls,
        paths: Sequence[tuple[str | Path, str | Path] | str | Path],
        *,
        supervised: bool = True,
    ) -> "MultiGameSequenceDataset":
        games = []
        for item in paths:
            if supervised:
                if not isinstance(item, tuple) or len(item) != 2:
                    raise ValueError("supervised paths must be (public_npz, teacher_npz) pairs")
                games.append(load_supervised_game(*item))
            else:
                if isinstance(item, tuple):
                    raise ValueError("public-only paths must not include teacher files")
                games.append(load_public_game(item))
        return cls(games)

    def __len__(self) -> int:
        return len(self.games)

    def __getitem__(self, index: int) -> GameSequence:
        return self.games[index]


_SEQUENCE_FIELDS = (
    "frames", "previous_action_id", "previous_action_x", "previous_action_y",
    "previous_level_boundary", "terminal", "won", "legal_action_mask",
    "executed_action_id", "executed_action_x", "executed_action_y", "next_frames",
    "next_level_boundary", "next_terminal", "next_won",
)
_TARGET_FIELDS = (
    "target_action_id", "target_action_x", "target_action_y", "target_valid", "action_source",
)
_CLICK_REGION_FIELD = "target_click_region"


def collate_game_sequences(games: Sequence[GameSequence]) -> dict[str, Tensor]:
    """Right-pad complete games without joining causal histories."""
    if not games:
        raise ValueError("cannot collate an empty batch")
    modes = {game.supervised for game in games}
    if len(modes) != 1:
        raise ValueError("cannot mix supervised and public-only games")
    batch_size = len(games)
    max_steps = max(map(len, games))
    padding_mask = torch.ones((batch_size, max_steps), dtype=torch.bool)
    result: dict[str, Tensor] = {"padding_mask": padding_mask}
    fields = _SEQUENCE_FIELDS + (_TARGET_FIELDS if games[0].supervised else ())
    for field in fields:
        exemplar = getattr(games[0], field)
        assert exemplar is not None
        shape = (batch_size, max_steps, *exemplar.shape[1:])
        if exemplar.dtype == np.bool_:
            tensor = torch.zeros(shape, dtype=torch.bool)
        elif exemplar.dtype == np.uint8:
            tensor = torch.zeros(shape, dtype=torch.uint8)
        else:
            fill = COORDINATE_NONE if "action" in field and field != "legal_action_mask" else 0
            tensor = torch.full(shape, fill, dtype=torch.long)
        result[field] = tensor
    supervised = games[0].supervised
    if supervised:
        result[_CLICK_REGION_FIELD] = torch.zeros(
            (batch_size, max_steps, FRAME_SIZE, FRAME_SIZE), dtype=torch.bool,
        )
    for batch_index, game in enumerate(games):
        length = len(game)
        padding_mask[batch_index, :length] = False
        for field in fields:
            value = getattr(game, field)
            assert value is not None
            result[field][batch_index, :length] = torch.from_numpy(value)
        if supervised:
            # Old teacher files without regions collate to the exact pixel, so
            # the set-valued click loss reduces to plain cross-entropy for them.
            result[_CLICK_REGION_FIELD][batch_index, :length] = torch.from_numpy(
                click_region_or_exact(game)
            )
    return result


ARCHITECTURES = ("v1", "v2")


@dataclass(frozen=True)
class MultiGameModelConfig:
    """Default-sized initial GPU model; use ``cpu_test`` for unit smokes.

    ``architecture`` selects the network layout.  ``"v1"`` is the original
    globally pooled action path with the query/bias click head; ``"v2"`` keeps
    the 8x8 feature grid on the action path and decodes clicks with a
    contextual convolutional head.  ``history_dropout`` is provenance only:
    the model never samples it, callers supply ``history_keep`` per game.
    """

    palette_dim: int = 24
    spatial_channels: int = 32
    conv_channels: int = 96
    hidden_dim: int = 256
    action_dim: int = 32
    coordinate_dim: int = 24
    dropout: float = 0.1
    architecture: str = "v1"
    history_dropout: float = 0.0

    def __post_init__(self) -> None:
        for key in ("palette_dim", "spatial_channels", "conv_channels", "hidden_dim",
                    "action_dim", "coordinate_dim"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}, got {self.architecture!r}")
        if not 0 <= self.history_dropout <= 1:
            raise ValueError("history_dropout must be in [0,1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MultiGameModelConfig":
        """Build a config; keys absent from old checkpoints take their defaults."""
        return cls(**dict(data))

    @classmethod
    def cpu_test(cls, **overrides: Any) -> "MultiGameModelConfig":
        values: dict[str, Any] = dict(
            palette_dim=4,
            spatial_channels=6,
            conv_channels=12,
            hidden_dim=24,
            action_dim=6,
            coordinate_dim=4,
            dropout=0.0,
        )
        values.update(overrides)
        return cls(**values)


class VisualEncoder(nn.Module):
    """Palette-aware encoder retaining one feature vector per display pixel.

    ``forward`` returns ``(spatial, features, context)``: full-resolution
    ``[B,T,S,64,64]`` features, the ``[B,T,H]`` per-frame vector fed to the
    recurrent token, and for ``v2`` the ``[B,T,C,8,8]`` grid the vector was
    read from (``None`` for ``v1``, whose grid is globally pooled away).
    """

    def __init__(self, config: MultiGameModelConfig):
        super().__init__()
        c = config
        self.architecture = c.architecture
        self.palette = nn.Embedding(PALETTE_SIZE, c.palette_dim)
        self.rows = nn.Embedding(FRAME_SIZE, c.palette_dim)
        self.columns = nn.Embedding(FRAME_SIZE, c.palette_dim)
        self.spatial = nn.Sequential(
            nn.Conv2d(c.palette_dim, c.spatial_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c.spatial_channels, c.spatial_channels, 3, padding=1),
            nn.GELU(),
        )
        context_layers = (
            nn.Conv2d(c.spatial_channels, c.conv_channels, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(c.conv_channels, c.conv_channels, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(c.conv_channels, c.conv_channels, 4, stride=2, padding=1),
            nn.GELU(),
        )
        if c.architecture == "v1":
            # Module name and layer order are frozen: v1 checkpoints index
            # ``global_path.<n>``.
            self.global_path = nn.Sequential(
                *context_layers,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c.conv_channels, c.hidden_dim),
                nn.LayerNorm(c.hidden_dim),
            )
        else:
            # v2 keeps the 8x8 grid: every cell reaches the hidden projection
            # with its own weights instead of being averaged into one vector.
            self.context = nn.Sequential(*context_layers)
            self.grid_projection = nn.Sequential(
                nn.Flatten(),
                nn.Linear(c.conv_channels * GRID_SIZE * GRID_SIZE, c.hidden_dim),
                nn.LayerNorm(c.hidden_dim),
            )

    def forward(self, frames: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        if frames.ndim != 4 or frames.shape[-2:] != (FRAME_SIZE, FRAME_SIZE):
            raise ValueError(f"frames must be [B,T,64,64], got {tuple(frames.shape)}")
        batch, steps = frames.shape[:2]
        values = frames.long()
        if torch.any((values < 0) | (values >= PALETTE_SIZE)):
            raise ValueError("frames contain palette values outside 0..15")
        pixels = self.palette(values)
        rows = self.rows(torch.arange(FRAME_SIZE, device=frames.device))[None, None, :, None, :]
        columns = self.columns(torch.arange(FRAME_SIZE, device=frames.device))[None, None, None, :, :]
        pixels = pixels + rows + columns
        pixels = pixels.reshape(batch * steps, FRAME_SIZE, FRAME_SIZE, -1).permute(0, 3, 1, 2)
        spatial = self.spatial(pixels)
        context: Tensor | None = None
        if self.architecture == "v1":
            features = self.global_path(spatial).reshape(batch, steps, -1)
        else:
            grid = self.context(spatial)
            if grid.shape[-2:] != (GRID_SIZE, GRID_SIZE):
                raise RuntimeError(f"unexpected context grid {tuple(grid.shape)}")
            features = self.grid_projection(grid).reshape(batch, steps, -1)
            context = grid.reshape(batch, steps, grid.shape[1], GRID_SIZE, GRID_SIZE)
        spatial = spatial.reshape(batch, steps, spatial.shape[1], FRAME_SIZE, FRAME_SIZE)
        return spatial, features, context


@dataclass(frozen=True)
class EncodedHistory:
    spatial: Tensor
    hidden: Tensor
    final_memory: Tensor
    padding_mask: Tensor
    # v2 only: the causal [B,T,C,8,8] context grid used by the click decoder.
    context: Tensor | None = None


@dataclass(frozen=True)
class PolicyOutput:
    action_logits: Tensor
    click_logits: Tensor


@dataclass(frozen=True)
class TransitionOutput:
    next_frame_logits: Tensor
    event_logits: Tensor
    indices: Tensor


class MultiGameModel(nn.Module):
    """Visual whole-game policy with a separate action-conditioned predictor."""

    def __init__(self, config: MultiGameModelConfig | None = None):
        super().__init__()
        self.config = config or MultiGameModelConfig()
        c = self.config
        self.visual = VisualEncoder(c)

        # Previous-action embeddings belong to the policy's causal history.
        self.previous_action = nn.Embedding(ACTION_COUNT + 1, c.action_dim)
        self.previous_x = nn.Embedding(FRAME_SIZE + 1, c.coordinate_dim)
        self.previous_y = nn.Embedding(FRAME_SIZE + 1, c.coordinate_dim)
        token_width = c.hidden_dim + c.action_dim + 2 * c.coordinate_dim + 3 + ACTION_COUNT
        self.token = nn.Sequential(
            nn.Linear(token_width, c.hidden_dim), nn.GELU(), nn.Dropout(c.dropout),
        )
        self.memory = nn.GRUCell(c.hidden_dim, c.hidden_dim)
        self.memory_norm = nn.LayerNorm(c.hidden_dim)

        self.action_head = nn.Linear(c.hidden_dim, ACTION_COUNT)
        if c.architecture == "v1":
            self.click_query = nn.Linear(c.hidden_dim, c.spatial_channels)
            self.click_bias = nn.Conv2d(c.spatial_channels, 1, 1)
        else:
            # v2 click decoder inputs, all at 64x64: full-resolution spatial
            # features, the 8x8 context grid (1x1-projected, then upsampled
            # x8) and the recurrent state broadcast to every pixel.
            self.click_context = nn.Conv2d(c.conv_channels, c.spatial_channels, 1)
            self.click_state = nn.Linear(c.hidden_dim, c.spatial_channels)
            self.click_decoder = nn.Sequential(
                nn.Conv2d(3 * c.spatial_channels, c.spatial_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(c.spatial_channels, 1, 3, padding=1),
            )

        # Current-action embeddings are reachable only through
        # predict_transitions, after causal history has already been encoded.
        self.prediction_action = nn.Embedding(ACTION_COUNT, c.action_dim)
        self.prediction_x = nn.Embedding(FRAME_SIZE + 1, c.coordinate_dim)
        self.prediction_y = nn.Embedding(FRAME_SIZE + 1, c.coordinate_dim)
        prediction_width = c.hidden_dim + c.action_dim + 2 * c.coordinate_dim
        self.prediction_condition = nn.Linear(prediction_width, c.spatial_channels)
        self.frame_predictor = nn.Sequential(
            nn.Conv2d(c.spatial_channels, c.spatial_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c.spatial_channels, PALETTE_SIZE, 1),
        )
        self.event_predictor = nn.Sequential(
            nn.LayerNorm(prediction_width),
            nn.Linear(prediction_width, c.hidden_dim),
            nn.GELU(),
            nn.Linear(c.hidden_dim, len(EVENT_NAMES)),
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def initial_memory(self, batch_size: int, *, device: torch.device | str | None = None) -> Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        parameter = next(self.parameters())
        return torch.zeros(
            batch_size, self.config.hidden_dim,
            device=parameter.device if device is None else device,
            dtype=parameter.dtype,
        )

    @staticmethod
    def _embedding_index(values: Tensor, *, upper: int, name: str) -> Tensor:
        values = values.long()
        if torch.any((values < COORDINATE_NONE) | (values >= upper)):
            raise ValueError(f"{name} values must be -1..{upper - 1}")
        return values + 1

    def encode_history(
        self,
        batch: Mapping[str, Tensor],
        *,
        initial_memory: Tensor | None = None,
        history_keep: Tensor | None = None,
    ) -> EncodedHistory:
        """Encode the causal history of every game in ``batch``.

        ``history_keep`` is an optional ``[B]`` bool mask.  Where it is False
        the previous-action triple (id, x, y) is replaced by the BOS values
        ``(-1, -1, -1)`` at every step of that game, so the policy sees only
        frames, flags and legal masks.  Targets and auxiliary executed-action
        labels are not inputs and are therefore never touched by it.
        """
        missing = [key for key in MODEL_INPUT_KEYS if key not in batch]
        if missing:
            raise ValueError(f"model batch is missing: {', '.join(missing)}")
        frames = batch["frames"]
        padding = batch["padding_mask"].bool()
        if padding.shape != frames.shape[:2]:
            raise ValueError("padding_mask must have shape [B,T]")
        if torch.any(padding[:, :-1] & ~padding[:, 1:]):
            raise ValueError("whole-game padding must be a right-padded suffix")
        batch_size, steps = frames.shape[:2]
        expected = (batch_size, steps)
        for key in ("previous_action_id", "previous_action_x", "previous_action_y",
                    "previous_level_boundary", "terminal", "won"):
            if batch[key].shape != expected:
                raise ValueError(f"{key} must have shape {expected}")
        legal = batch["legal_action_mask"].bool()
        if legal.shape != (*expected, ACTION_COUNT):
            raise ValueError(f"legal_action_mask must have shape {(*expected, ACTION_COUNT)}")

        previous_id = batch["previous_action_id"].long()
        previous_x = batch["previous_action_x"].long()
        previous_y = batch["previous_action_y"].long()
        if torch.any((previous_id < COORDINATE_NONE) | (previous_id >= ACTION_COUNT)):
            raise ValueError("previous_action_id values must be -1..7")
        click = previous_id == CLICK_ACTION
        if torch.any(click & ((previous_x < 0) | (previous_x >= FRAME_SIZE) |
                              (previous_y < 0) | (previous_y >= FRAME_SIZE))):
            raise ValueError("previous click coordinates must be in 0..63")
        if torch.any(~click & ((previous_x != COORDINATE_NONE) | (previous_y != COORDINATE_NONE))):
            raise ValueError("previous non-click/BOS coordinates must be -1,-1")
        if history_keep is not None:
            keep = history_keep.to(device=previous_id.device).bool()
            if keep.shape != (batch_size,):
                raise ValueError(f"history_keep must have shape {(batch_size,)}")
            drop = ~keep[:, None]
            previous_id = previous_id.masked_fill(drop, COORDINATE_NONE)
            previous_x = previous_x.masked_fill(drop, COORDINATE_NONE)
            previous_y = previous_y.masked_fill(drop, COORDINATE_NONE)

        spatial, visual, context = self.visual(frames)
        flags = torch.stack((
            batch["previous_level_boundary"].bool(),
            batch["terminal"].bool(),
            batch["won"].bool(),
        ), dim=-1).to(visual.dtype)
        parts = (
            visual,
            self.previous_action(previous_id + 1),
            self.previous_x(self._embedding_index(previous_x, upper=FRAME_SIZE, name="previous_action_x")),
            self.previous_y(self._embedding_index(previous_y, upper=FRAME_SIZE, name="previous_action_y")),
            flags,
            legal.to(visual.dtype),
        )
        tokens = self.token(torch.cat(parts, dim=-1))
        memory = self.initial_memory(batch_size, device=frames.device) if initial_memory is None else initial_memory
        if memory.shape != (batch_size, self.config.hidden_dim):
            raise ValueError("initial_memory has the wrong shape")
        hidden = []
        for step in range(steps):
            updated = self.memory(tokens[:, step], memory)
            active = ~padding[:, step, None]
            memory = torch.where(active, updated, memory)
            hidden.append(torch.where(active, self.memory_norm(memory), torch.zeros_like(memory)))
        history = torch.stack(hidden, dim=1)
        return EncodedHistory(spatial, history, memory, padding, context)

    def policy_from_history(self, encoded: EncodedHistory, legal_action_mask: Tensor) -> PolicyOutput:
        legal = legal_action_mask.bool()
        if legal.shape != (*encoded.hidden.shape[:2], ACTION_COUNT):
            raise ValueError("legal_action_mask does not align with encoded history")
        legal = legal.clone()
        legal[..., 0] = False
        active = ~encoded.padding_mask
        if torch.any(active & ~legal[..., 1:].any(dim=-1)):
            raise ValueError("each unpadded state needs at least one legal action in 1..7")
        raw_actions = self.action_head(encoded.hidden)
        action_logits = raw_actions.masked_fill(~legal, float("-inf"))
        action_logits = torch.where(active[..., None], action_logits, torch.zeros_like(action_logits))

        if self.config.architecture == "v1":
            click_logits = self._click_logits_v1(encoded)
        else:
            click_logits = self._click_logits_v2(encoded)
        click_logits = torch.where(active[..., None, None], click_logits, torch.zeros_like(click_logits))
        return PolicyOutput(action_logits, click_logits)

    def _click_logits_v1(self, encoded: EncodedHistory) -> Tensor:
        query = self.click_query(encoded.hidden)[..., None, None]
        click_logits = (encoded.spatial * query).sum(dim=2) / math.sqrt(self.config.spatial_channels)
        flat_spatial = encoded.spatial.flatten(0, 1)
        bias = self.click_bias(flat_spatial).reshape(*encoded.hidden.shape[:2], FRAME_SIZE, FRAME_SIZE)
        return click_logits + bias

    def _click_logits_v2(self, encoded: EncodedHistory) -> Tensor:
        """Decode one 64x64 logit map per step; row index is y, column is x."""
        if encoded.context is None:
            raise ValueError("v2 click decoding needs the encoder's context grid")
        batch, steps = encoded.hidden.shape[:2]
        spatial = encoded.spatial.flatten(0, 1)
        context = self.click_context(encoded.context.flatten(0, 1))
        context = F.interpolate(context, scale_factor=FRAME_SIZE // GRID_SIZE, mode="nearest")
        state = self.click_state(encoded.hidden).flatten(0, 1)[..., None, None]
        state = state.expand(-1, -1, FRAME_SIZE, FRAME_SIZE)
        features = torch.cat((spatial, context, state), dim=1)
        return self.click_decoder(features).reshape(batch, steps, FRAME_SIZE, FRAME_SIZE)

    def policy(
        self,
        batch: Mapping[str, Tensor],
        *,
        initial_memory: Tensor | None = None,
        history_keep: Tensor | None = None,
    ) -> PolicyOutput:
        encoded = self.encode_history(batch, initial_memory=initial_memory, history_keep=history_keep)
        return self.policy_from_history(encoded, batch["legal_action_mask"])

    def forward(self, batch: Mapping[str, Tensor], *, history_keep: Tensor | None = None) -> PolicyOutput:
        return self.policy(batch, history_keep=history_keep)

    @staticmethod
    def decode_click(click_logits: Tensor) -> tuple[Tensor, Tensor]:
        if click_logits.shape[-2:] != (FRAME_SIZE, FRAME_SIZE):
            raise ValueError("click logits must end in [64,64]")
        index = click_logits.flatten(-2).argmax(dim=-1)
        return index % FRAME_SIZE, index // FRAME_SIZE

    def predict_transitions(
        self,
        encoded: EncodedHistory,
        action_id: Tensor,
        action_x: Tensor,
        action_y: Tensor,
        *,
        indices: Tensor | None = None,
    ) -> TransitionOutput:
        """Predict successors after explicitly supplied executed actions.

        ``indices`` is optional ``[N,2]`` (batch, time) selection.  It lets a
        trainer subsample expensive 64x64 auxiliary targets without changing
        the complete causal policy history.
        """
        shape = encoded.hidden.shape[:2]
        if action_id.shape != shape or action_x.shape != shape or action_y.shape != shape:
            raise ValueError("executed action arrays must align with encoded [B,T] history")
        if indices is None:
            indices = (~encoded.padding_mask).nonzero(as_tuple=False)
        if indices.ndim != 2 or indices.shape[1] != 2:
            raise ValueError("transition indices must be [N,2] batch/time pairs")
        indices = indices.to(device=encoded.hidden.device, dtype=torch.long)
        if len(indices) == 0:
            raise ValueError("at least one transition must be selected")
        if torch.any(indices[:, 0] < 0) or torch.any(indices[:, 0] >= shape[0]) or \
           torch.any(indices[:, 1] < 0) or torch.any(indices[:, 1] >= shape[1]):
            raise ValueError("transition indices are out of bounds")
        if torch.any(encoded.padding_mask[indices[:, 0], indices[:, 1]]):
            raise ValueError("transition indices cannot select padding")

        batch_index, time_index = indices[:, 0], indices[:, 1]
        chosen_id = action_id[batch_index, time_index].long()
        chosen_x = action_x[batch_index, time_index].long()
        chosen_y = action_y[batch_index, time_index].long()
        if torch.any((chosen_id < 1) | (chosen_id >= ACTION_COUNT)):
            raise ValueError("executed action IDs must be in 1..7")
        click = chosen_id == CLICK_ACTION
        if torch.any(click & ((chosen_x < 0) | (chosen_x >= FRAME_SIZE) |
                              (chosen_y < 0) | (chosen_y >= FRAME_SIZE))):
            raise ValueError("executed click coordinates must be in 0..63")
        if torch.any(~click & ((chosen_x != COORDINATE_NONE) | (chosen_y != COORDINATE_NONE))):
            raise ValueError("executed non-click coordinates must be -1,-1")

        hidden = encoded.hidden[batch_index, time_index]
        action_features = torch.cat((
            hidden,
            self.prediction_action(chosen_id),
            self.prediction_x(self._embedding_index(chosen_x, upper=FRAME_SIZE, name="action_x")),
            self.prediction_y(self._embedding_index(chosen_y, upper=FRAME_SIZE, name="action_y")),
        ), dim=-1)
        condition = self.prediction_condition(action_features)[..., None, None]
        spatial = encoded.spatial[batch_index, time_index]
        frame_logits = self.frame_predictor(spatial + condition)
        event_logits = self.event_predictor(action_features)
        return TransitionOutput(frame_logits, event_logits, indices)

    def policy_step(
        self,
        *,
        frame: Tensor,
        previous_action_id: Tensor,
        previous_action_x: Tensor,
        previous_action_y: Tensor,
        previous_level_boundary: Tensor,
        terminal: Tensor,
        won: Tensor,
        legal_action_mask: Tensor,
        memory: Tensor | None = None,
        history_keep: Tensor | None = None,
    ) -> tuple[PolicyOutput, Tensor]:
        """One online decision; reuse returned memory until the game ends.

        ``history_keep`` (``[B]`` bool) lets inference run history-free for
        diagnostics: False rows see BOS instead of their previous action.
        """
        if frame.ndim != 3 or frame.shape[-2:] != (FRAME_SIZE, FRAME_SIZE):
            raise ValueError("frame must be [B,64,64]")
        batch_size = frame.shape[0]
        batch = {
            "frames": frame[:, None],
            "previous_action_id": previous_action_id[:, None],
            "previous_action_x": previous_action_x[:, None],
            "previous_action_y": previous_action_y[:, None],
            "previous_level_boundary": previous_level_boundary[:, None],
            "terminal": terminal[:, None],
            "won": won[:, None],
            "legal_action_mask": legal_action_mask[:, None],
            "padding_mask": torch.zeros((batch_size, 1), dtype=torch.bool, device=frame.device),
        }
        encoded = self.encode_history(batch, initial_memory=memory, history_keep=history_keep)
        output = self.policy_from_history(encoded, batch["legal_action_mask"])
        return PolicyOutput(output.action_logits[:, 0], output.click_logits[:, 0]), encoded.final_memory


@dataclass(frozen=True)
class LossWeights:
    action: float = 1.0
    click: float = 1.0
    next_frame: float = 0.25
    events: float = 0.1
    # Explicit objective multipliers, preserved in the training recipe.  Mean
    # reductions still divide by the number of pixels/events, so rare changes
    # contribute more gradient instead of being hidden by static backgrounds.
    changed_pixel_weight: float = 1.0
    event_positive_weight: float = 1.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.changed_pixel_weight == 0 or self.event_positive_weight == 0:
            raise ValueError("changed-pixel and positive-event weights must be positive")


@dataclass(frozen=True)
class MultiGameLoss:
    """Loss terms plus the counts they were (or can be) normalised by.

    With ``reduction="mean"`` every term is already averaged over its own
    targets.  With ``reduction="sum"`` every term is the plain sum and the
    ``*_targets`` counts are the divisors that recover the mean:
    ``action_targets`` valid policy rows, ``click_targets`` click rows,
    ``frame_targets`` predicted pixels (transitions x 64 x 64) and
    ``event_targets`` event logits (transitions x 3).  ``total`` applies the
    weights to the terms as returned, so a trainer using sums must divide
    the individual terms itself.
    """

    total: Tensor
    action: Tensor
    click: Tensor
    next_frame: Tensor
    events: Tensor
    click_targets: int
    policy_targets: int
    auxiliary_targets: int
    final_memory: Tensor | None = None
    # Argmax-pixel hit counts over the click targets of this batch.
    click_exact_correct: int = 0
    click_region_correct: int = 0
    reduction: str = "mean"
    action_targets: int = 0
    frame_targets: int = 0
    event_targets: int = 0


def _zero(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def click_region_nll(click_logits: Tensor, region: Tensor) -> Tensor:
    """Per-row negative log of the probability mass inside ``region``.

    ``click_logits`` is ``[..., 64, 64]`` and ``region`` a matching bool mask.
    The value is ``logsumexp(all) - logsumexp(region)``; for a one-pixel region
    this is exactly the pixel cross-entropy.  Every row must have a non-empty
    region.
    """
    if click_logits.shape[-2:] != (FRAME_SIZE, FRAME_SIZE) or region.shape != click_logits.shape:
        raise ValueError("click logits and region must share a [..., 64, 64] shape")
    flat = click_logits.flatten(-2)
    mask = region.bool().flatten(-2)
    if flat.numel() and not bool(mask.any(dim=-1).all()):
        raise ValueError("every click region must contain at least one pixel")
    inside = torch.logsumexp(flat.masked_fill(~mask, float("-inf")), dim=-1)
    return torch.logsumexp(flat, dim=-1) - inside


def exact_click_region(target_x: Tensor, target_y: Tensor) -> Tensor:
    """One-hot bool ``[N,64,64]`` regions from exact click coordinates."""
    region = torch.zeros((*target_x.shape, FRAME_SIZE, FRAME_SIZE), dtype=torch.bool, device=target_x.device)
    flat = region.flatten(-2)
    flat.scatter_(-1, (target_y.long() * FRAME_SIZE + target_x.long()).unsqueeze(-1), True)
    return flat.view_as(region)


def compute_multigame_loss(
    model: MultiGameModel,
    batch: Mapping[str, Tensor],
    *,
    weights: LossWeights = LossWeights(),
    transition_indices: Tensor | None = None,
    initial_memory: Tensor | None = None,
    history_keep: Tensor | None = None,
    reduction: str = "mean",
) -> MultiGameLoss:
    """Compute causal policy and action-conditioned world-model losses.

    ``reduction="sum"`` returns unnormalised per-term sums with their target
    counts so a trainer can normalise per whole game across chunks.  When
    both auxiliary weights are zero the transition head is not run and the
    frame/event terms are zero with zero counts.
    """
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be 'mean' or 'sum'")
    for key in ("target_action_id", "target_action_x", "target_action_y", "target_valid",
                "executed_action_id", "executed_action_x", "executed_action_y", "next_frames",
                "next_level_boundary", "next_terminal", "next_won"):
        if key not in batch:
            raise ValueError(f"loss batch is missing {key}")
    encoded = model.encode_history(batch, initial_memory=initial_memory, history_keep=history_keep)
    policy = model.policy_from_history(encoded, batch["legal_action_mask"])
    padding = batch["padding_mask"].bool()
    target_valid = batch["target_valid"].bool() & ~padding
    if torch.any(target_valid):
        action_loss = F.cross_entropy(
            policy.action_logits[target_valid], batch["target_action_id"][target_valid].long(),
            reduction=reduction,
        )
    else:
        action_loss = _zero(encoded.hidden)
    click_mask = target_valid & (batch["target_action_id"] == CLICK_ACTION)
    click_exact_correct = click_region_correct = 0
    if torch.any(click_mask):
        target_x = batch["target_action_x"][click_mask].long()
        target_y = batch["target_action_y"][click_mask].long()
        if "target_click_region" in batch:
            region = batch["target_click_region"][click_mask].bool()
        else:
            region = exact_click_region(target_x, target_y)
        click_logits = policy.click_logits[click_mask]
        # Set-valued supervision: maximise the mass on every engine-equivalent
        # pixel rather than the single teacher pixel.
        per_click = click_region_nll(click_logits, region)
        click_loss = per_click.mean() if reduction == "mean" else per_click.sum()
        with torch.no_grad():
            guess_x, guess_y = MultiGameModel.decode_click(click_logits)
            rows = torch.arange(len(guess_x), device=guess_x.device)
            click_exact_correct = int(((guess_x == target_x) & (guess_y == target_y)).sum())
            click_region_correct = int(region[rows, guess_y, guess_x].sum())
    else:
        click_loss = _zero(encoded.hidden)

    auxiliary_targets = frame_targets = event_targets = 0
    if (weights.next_frame == 0 and weights.events == 0) or (
        transition_indices is not None and transition_indices.numel() == 0
    ):
        # Both auxiliary heads are disabled: skip the transition head entirely.
        frame_loss = _zero(encoded.hidden)
        event_loss = _zero(encoded.hidden)
    else:
        transition = model.predict_transitions(
            encoded,
            batch["executed_action_id"], batch["executed_action_x"], batch["executed_action_y"],
            indices=transition_indices,
        )
        selected_batch, selected_time = transition.indices[:, 0], transition.indices[:, 1]
        next_frames = batch["next_frames"][selected_batch, selected_time].long()
        per_pixel = F.cross_entropy(transition.next_frame_logits, next_frames, reduction="none")
        current_frames = batch["frames"][selected_batch, selected_time]
        pixel_weights = torch.where(
            next_frames != current_frames, weights.changed_pixel_weight, 1.0,
        )
        weighted_pixels = per_pixel * pixel_weights
        frame_loss = weighted_pixels.mean() if reduction == "mean" else weighted_pixels.sum()
        event_labels = torch.stack((
            batch["next_level_boundary"][selected_batch, selected_time],
            batch["next_terminal"][selected_batch, selected_time],
            batch["next_won"][selected_batch, selected_time],
        ), dim=-1).to(transition.event_logits.dtype)
        event_loss = F.binary_cross_entropy_with_logits(
            transition.event_logits, event_labels, reduction=reduction,
            pos_weight=transition.event_logits.new_tensor(weights.event_positive_weight),
        )
        auxiliary_targets = len(transition.indices)
        frame_targets = auxiliary_targets * FRAME_SIZE * FRAME_SIZE
        event_targets = auxiliary_targets * len(EVENT_NAMES)
    total = (
        weights.action * action_loss
        + weights.click * click_loss
        + weights.next_frame * frame_loss
        + weights.events * event_loss
    )
    policy_targets = int(target_valid.sum().item())
    return MultiGameLoss(
        total, action_loss, click_loss, frame_loss, event_loss,
        click_targets=int(click_mask.sum().item()),
        policy_targets=policy_targets,
        auxiliary_targets=auxiliary_targets,
        final_memory=encoded.final_memory,
        click_exact_correct=click_exact_correct,
        click_region_correct=click_region_correct,
        reduction=reduction,
        action_targets=policy_targets,
        frame_targets=frame_targets,
        event_targets=event_targets,
    )


def save_multigame_checkpoint(
    model: MultiGameModel,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save model weights/config plus caller-owned provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save({
        "format": FORMAT,
        "config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "metadata": dict(metadata or {}),
    }, temporary)
    os.replace(temporary, path)
    return path


def load_multigame_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[MultiGameModel, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError(f"unsupported multigame checkpoint format {payload.get('format')!r}")
    model = MultiGameModel(MultiGameModelConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    return model, dict(payload.get("metadata") or {})


__all__ = [
    "ACTION_COUNT", "ARCHITECTURES", "CLICK_ACTION", "COORDINATE_NONE", "EVENT_NAMES", "FORMAT",
    "FRAME_SIZE", "GRID_SIZE", "GameSequence", "LossWeights", "MODEL_INPUT_KEYS", "MultiGameLoss",
    "MultiGameModel", "MultiGameModelConfig", "MultiGameSequenceDataset", "PolicyOutput",
    "TransitionOutput", "click_region_nll", "click_region_or_exact", "collate_game_sequences",
    "compute_multigame_loss", "exact_click_region", "load_multigame_checkpoint",
    "load_public_game", "load_supervised_game", "save_multigame_checkpoint",
]
