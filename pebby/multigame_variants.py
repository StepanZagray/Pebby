"""Fixed public control/appearance transforms for generated whole games.

Variants never mutate the real engine or planner.  They form a bijection
between raw engine observations/actions and the public trajectory, sampled
once per game and retained across all of its levels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from numbers import Integral
from typing import Any, Iterable, Sequence

import numpy as np


FRAME_SIZE = 64
ACTION_COUNT = 8
CLICK_ACTION = 6
PALETTE_SIZE = 16
NONCLICK_IDS = (1, 2, 3, 4, 5, 7)
D4_NAMES = (
    "identity", "rot90", "rot180", "rot270",
    "flip_x", "flip_y", "transpose", "anti_transpose",
)


@dataclass(frozen=True)
class VariantOptions:
    """Explicit opt-in controls for sampling one variant per whole game."""

    enabled: bool = False
    mix_probability: float = 1.0
    seed: int = 0
    controls: bool = True
    spatial: bool = True
    palette: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.mix_probability <= 1.0:
            raise ValueError("variant mix_probability must be in [0,1]")
        if self.enabled and not (self.controls or self.spatial or self.palette):
            raise ValueError("an enabled variant needs at least one component")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _permutation(values: Sequence[int], name: str) -> tuple[int, ...]:
    raw = tuple(values)
    if any(not isinstance(value, Integral) or isinstance(value, bool) for value in raw):
        raise ValueError(f"{name} entries must be integers")
    result = tuple(int(value) for value in raw)
    if sorted(result) != list(range(len(result))):
        raise ValueError(f"{name} must be a permutation of 0..{len(result) - 1}")
    return result


def _inverse_permutation(values: Sequence[int]) -> tuple[int, ...]:
    inverse = [0] * len(values)
    for source, target in enumerate(values):
        inverse[int(target)] = source
    return tuple(inverse)


def transform_xy(name: str, x: int, y: int, *, size: int = FRAME_SIZE) -> tuple[int, int]:
    """Map a raw coordinate to the coordinate of the named transformed frame."""
    if name not in D4_NAMES:
        raise ValueError(f"unknown D4 transform {name!r}")
    if not all(isinstance(value, Integral) and not isinstance(value, bool) for value in (x, y)):
        raise ValueError("coordinates must be integers")
    x, y = int(x), int(y)
    if not (0 <= x < size and 0 <= y < size):
        raise ValueError(f"coordinates must be in 0..{size - 1}")
    edge = size - 1
    if name == "identity":
        return x, y
    if name == "rot90":
        return y, edge - x
    if name == "rot180":
        return edge - x, edge - y
    if name == "rot270":
        return edge - y, x
    if name == "flip_x":
        return edge - x, y
    if name == "flip_y":
        return x, edge - y
    if name == "transpose":
        return y, x
    return edge - y, edge - x  # anti_transpose


def transform_frame(frame: np.ndarray, name: str) -> np.ndarray:
    """Apply the same D4 mapping as ``transform_xy`` to the last two axes."""
    frame = np.asarray(frame)
    if frame.shape[-2:] != (FRAME_SIZE, FRAME_SIZE):
        raise ValueError(f"frame must end in [64,64], got {frame.shape}")
    if name == "identity":
        result = frame
    elif name == "rot90":
        result = np.rot90(frame, 1, axes=(-2, -1))
    elif name == "rot180":
        result = np.rot90(frame, 2, axes=(-2, -1))
    elif name == "rot270":
        result = np.rot90(frame, 3, axes=(-2, -1))
    elif name == "flip_x":
        result = np.flip(frame, axis=-1)
    elif name == "flip_y":
        result = np.flip(frame, axis=-2)
    elif name == "transpose":
        result = np.swapaxes(frame, -2, -1)
    elif name == "anti_transpose":
        result = np.flip(np.swapaxes(frame, -2, -1), axis=(-2, -1))
    else:
        raise ValueError(f"unknown D4 transform {name!r}")
    return np.ascontiguousarray(result)


_D4_INVERSE = {
    "identity": "identity",
    "rot90": "rot270",
    "rot180": "rot180",
    "rot270": "rot90",
    "flip_x": "flip_x",
    "flip_y": "flip_y",
    "transpose": "transpose",
    "anti_transpose": "anti_transpose",
}


@dataclass(frozen=True)
class WholeGameVariant:
    """A complete immutable raw-to-public bijection for one game."""

    selected: bool
    seed: int
    control_raw_to_public: tuple[int, ...]
    spatial: str
    palette_raw_to_public: tuple[int, ...]

    def __post_init__(self) -> None:
        controls = _permutation(self.control_raw_to_public, "control_raw_to_public")
        palette = _permutation(self.palette_raw_to_public, "palette_raw_to_public")
        if len(controls) != ACTION_COUNT:
            raise ValueError("control permutation must contain IDs 0..7")
        if len(palette) != PALETTE_SIZE:
            raise ValueError("palette permutation must contain values 0..15")
        if controls[0] != 0 or controls[CLICK_ACTION] != CLICK_ACTION:
            raise ValueError("RESET and click action IDs must remain fixed")
        if self.spatial not in D4_NAMES:
            raise ValueError(f"unknown D4 transform {self.spatial!r}")
        # Frozen dataclasses do not recursively freeze a caller-owned list.
        # Normalize after validating the original values so this game-wide
        # bijection cannot be mutated externally or accept lossy casts.
        object.__setattr__(self, "control_raw_to_public", controls)
        object.__setattr__(self, "palette_raw_to_public", palette)

    @classmethod
    def identity(cls, seed: int = 0) -> "WholeGameVariant":
        return cls(False, int(seed), tuple(range(ACTION_COUNT)), "identity", tuple(range(PALETTE_SIZE)))

    @property
    def control_public_to_raw(self) -> tuple[int, ...]:
        return _inverse_permutation(self.control_raw_to_public)

    @property
    def palette_public_to_raw(self) -> tuple[int, ...]:
        return _inverse_permutation(self.palette_raw_to_public)

    @property
    def is_identity(self) -> bool:
        return (
            self.control_raw_to_public == tuple(range(ACTION_COUNT))
            and self.spatial == "identity"
            and self.palette_raw_to_public == tuple(range(PALETTE_SIZE))
        )

    def public_frame(self, raw_frame: np.ndarray) -> np.ndarray:
        spatial = transform_frame(raw_frame, self.spatial)
        return np.asarray(self.palette_raw_to_public, dtype=np.uint8)[spatial]

    def raw_frame(self, public_frame: np.ndarray) -> np.ndarray:
        uncolored = np.asarray(self.palette_public_to_raw, dtype=np.uint8)[np.asarray(public_frame)]
        return transform_frame(uncolored, _D4_INVERSE[self.spatial])

    @staticmethod
    def _validate_action(action_id: int, x: int | None, y: int | None) -> tuple[int, int | None, int | None]:
        if not isinstance(action_id, Integral) or isinstance(action_id, bool):
            raise ValueError("action_id must be an integer")
        action_id = int(action_id)
        if not 0 <= action_id < ACTION_COUNT:
            raise ValueError("action_id must be in 0..7")
        if action_id == CLICK_ACTION:
            if x is None or y is None:
                raise ValueError("click actions require coordinates")
            transform_xy("identity", x, y)
            return action_id, int(x), int(y)
        if x is not None or y is not None:
            raise ValueError("non-click actions cannot have coordinates")
        return action_id, None, None

    def public_action(self, raw_id: int, raw_x: int | None, raw_y: int | None) -> tuple[int, int | None, int | None]:
        raw_id, raw_x, raw_y = self._validate_action(raw_id, raw_x, raw_y)
        public_id = self.control_raw_to_public[raw_id]
        if raw_id == CLICK_ACTION:
            public_x, public_y = transform_xy(self.spatial, raw_x, raw_y)
            return public_id, public_x, public_y
        return public_id, None, None

    def raw_action(self, public_id: int, public_x: int | None, public_y: int | None) -> tuple[int, int | None, int | None]:
        public_id, public_x, public_y = self._validate_action(public_id, public_x, public_y)
        raw_id = self.control_public_to_raw[public_id]
        if public_id == CLICK_ACTION:
            raw_x, raw_y = transform_xy(_D4_INVERSE[self.spatial], public_x, public_y)
            return raw_id, raw_x, raw_y
        return raw_id, None, None

    def public_legal_mask(self, raw_mask: np.ndarray) -> np.ndarray:
        raw_mask = np.asarray(raw_mask)
        if raw_mask.shape != (ACTION_COUNT,) or raw_mask.dtype != np.bool_:
            raise ValueError("raw legal mask must be bool [8]")
        public = np.zeros(ACTION_COUNT, dtype=np.bool_)
        public[np.asarray(self.control_raw_to_public)] = raw_mask
        return public

    def private_metadata(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "seed": self.seed,
            "identity": self.is_identity,
            "control_raw_to_public": list(self.control_raw_to_public),
            "spatial": self.spatial,
            "palette_raw_to_public": list(self.palette_raw_to_public),
        }


def deterministic_variant_seed(base_seed: int, source_id: str, game_index: int) -> int:
    material = f"variant:{int(base_seed)}:{source_id}:{int(game_index)}".encode()
    return int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def sample_whole_game_variant(
    options: VariantOptions,
    *,
    source_id: str,
    game_index: int,
    legal_action_ids: Iterable[int],
) -> WholeGameVariant:
    """Deterministically sample one fixed variant without inspecting a game ID at inference."""
    seed = deterministic_variant_seed(options.seed, source_id, game_index)
    if not options.enabled:
        return WholeGameVariant.identity(seed)
    rng = np.random.default_rng(seed)
    selected = bool(rng.random() < options.mix_probability)
    if not selected:
        return WholeGameVariant.identity(seed)

    controls = list(range(ACTION_COUNT))
    legal_nonclick = sorted({int(value) for value in legal_action_ids} & set(NONCLICK_IDS))
    if options.controls and len(legal_nonclick) > 1:
        shuffled = rng.permutation(legal_nonclick).tolist()
        for raw, public in zip(legal_nonclick, shuffled):
            controls[raw] = int(public)
    spatial = str(rng.choice(D4_NAMES)) if options.spatial else "identity"
    palette = tuple(int(value) for value in (
        rng.permutation(PALETTE_SIZE) if options.palette else np.arange(PALETTE_SIZE)
    ))
    return WholeGameVariant(selected, seed, tuple(controls), spatial, palette)


__all__ = [
    "ACTION_COUNT", "CLICK_ACTION", "D4_NAMES", "FRAME_SIZE", "NONCLICK_IDS",
    "PALETTE_SIZE", "VariantOptions", "WholeGameVariant", "deterministic_variant_seed",
    "sample_whole_game_variant", "transform_frame", "transform_xy",
]
