"""Level-first difficulty sampling for generated world-model transitions.

Rows from one level are near duplicates.  This sampler therefore allocates a
batch across difficulty stages, chooses distinct level seeds within each stage,
and chooses one row uniformly from each selected level.  The metadata is the
source of the seed-to-difficulty mapping; rows whose seed is absent from
``meta['levels']`` are deliberately ignored.
"""

import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral

import torch


from ..ls20.provenance import (LEGACY_DIFFICULTIES, difficulty_stages, difficulty_version,
                               validate_difficulty)

DIFFICULTIES = LEGACY_DIFFICULTIES  # Public historical constant; use sampler.difficulties.
DEFAULT_START = (0.55, 0.25, 0.12, 0.06, 0.02)
DEFAULT_END = (0.05, 0.10, 0.20, 0.25, 0.40)
CALIBRATED_START = (.40, .25, .15, .09, .05, .04, .02)
CALIBRATED_END = (.02, .04, .06, .10, .18, .25, .35)
MAX_BATCH_SIZE = 1024


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _metadata(value):
    """Turn an NPZ scalar/string or a mapping into metadata."""
    while not isinstance(value, (str, bytes, Mapping)) and hasattr(value, "size"):
        if value.size != 1:
            break
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("meta must contain valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("meta must be a mapping or JSON object")
    return value


def _weights(values, name, count):
    count_label = "five" if count == 5 else "seven"
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must contain {count_label} nonnegative weights")
    try:
        weights = torch.as_tensor(values, dtype=torch.float64, device="cpu")
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name} must contain {count_label} nonnegative weights") from error
    if weights.ndim != 1 or weights.numel() != count:
        raise ValueError(f"{name} must contain {count_label} weights")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError(f"{name} must contain finite nonnegative weights")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError(f"{name} must have a positive sum")
    return weights / total


class CurriculumSampler:
    """Sample distinct level seeds according to an easy-to-hard schedule.

    ``data`` is the loaded shard mapping and must provide a one-dimensional
    ``seeds`` array plus ``meta['levels']`` entries containing ``seed`` and
    ``difficulty``.  A level can contribute at most one row to a batch.
    """

    def __init__(self, data, *, start=None, end=None):
        if not isinstance(data, Mapping):
            raise ValueError("data must be a mapping containing seeds and meta")
        if "seeds" not in data or "meta" not in data:
            raise ValueError("data must contain seeds and meta")
        raw_seeds = data["seeds"]
        try:
            seeds = torch.as_tensor(raw_seeds, device="cpu")
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("seeds must be a one-dimensional integer array") from error
        if (seeds.ndim != 1 or seeds.dtype == torch.bool or
                torch.is_floating_point(seeds) or torch.is_complex(seeds)):
            raise ValueError("seeds must be a one-dimensional integer array")
        seeds = seeds.to(dtype=torch.int64)

        meta = _metadata(data["meta"])
        levels = meta.get("levels")
        if not isinstance(levels, Sequence) or isinstance(levels, (str, bytes)):
            raise ValueError("meta['levels'] must be a sequence")

        by_seed = {}
        row_seed_set = set(seeds.tolist())
        included = [level for level in levels if isinstance(level, Mapping)
                    and isinstance(level.get("seed"), Integral)
                    and level.get("seed") in row_seed_set and "excluded" not in level]
        versions = {difficulty_version(level) for level in included}
        if len(versions) > 1:
            raise ValueError("mixed legacy and calibrated difficulty versions require separate samplers")
        self.difficulty_version = next(iter(versions), None)
        self.difficulties = difficulty_stages({'difficulty_version': self.difficulty_version})
        if start is None:
            start = CALIBRATED_START if self.difficulty_version else DEFAULT_START
        if end is None:
            end = CALIBRATED_END if self.difficulty_version else DEFAULT_END
        for level in levels:
            if not isinstance(level, Mapping) or "seed" not in level:
                raise ValueError("each meta['levels'] entry needs seed")
            seed = level["seed"]
            if isinstance(seed, bool) or not isinstance(seed, Integral):
                raise ValueError("level seed must be an integer")
            if seed not in row_seed_set or "excluded" in level:
                continue
            if "difficulty" not in level:
                raise ValueError("each sampled level needs difficulty")
            difficulty = validate_difficulty(level)
            seed = int(seed)
            previous = by_seed.get(seed)
            if previous is not None and previous != difficulty:
                raise ValueError(f"inconsistent difficulty metadata for seed {seed}")
            by_seed[seed] = difficulty

        rows_by_seed = {}
        for index, seed in enumerate(seeds.tolist()):
            seed = int(seed)
            if seed in by_seed:
                rows_by_seed.setdefault(seed, []).append(index)
        if not rows_by_seed:
            raise ValueError("meta['levels'] contains no rows present in seeds")

        self.seeds = torch.tensor(sorted(rows_by_seed), dtype=torch.int64)
        self._rows_by_seed = {seed: torch.tensor(rows, dtype=torch.long)
                              for seed, rows in rows_by_seed.items()}
        self._levels_by_difficulty = {difficulty: [] for difficulty in self.difficulties}
        for seed in self.seeds.tolist():
            self._levels_by_difficulty[by_seed[seed]].append(seed)
        self._start = _weights(start, "start", len(self.difficulties))
        self._end = _weights(end, "end", len(self.difficulties))
        self.last_difficulty_counts = {difficulty: 0 for difficulty in self.difficulties}
        self.last_distinct_levels = 0
        self.last_level_seeds = ()

    def ratios(self, progress):
        """Return normalized stage weights at normalized progress ``[0, 1]``."""
        if isinstance(progress, bool):
            raise ValueError("progress must be finite and in [0, 1]")
        try:
            progress = float(progress)
        except (TypeError, ValueError) as error:
            raise ValueError("progress must be finite and in [0, 1]") from error
        if not math.isfinite(progress) or not 0. <= progress <= 1.:
            raise ValueError("progress must be finite and in [0, 1]")
        return self._start + progress * (self._end - self._start)

    def check_coverage(self, batch_size):
        """Conservative bound covers quota rounding at every schedule position."""
        batch_size = _positive_integer(batch_size, "batch_size")
        if batch_size > MAX_BATCH_SIZE:
            raise ValueError(f"batch_size cannot exceed {MAX_BATCH_SIZE}")
        required = torch.ceil(torch.maximum(self._start, self._end) * batch_size).long()
        for difficulty in self.difficulties:
            need = int(required[difficulty - 1])
            have = len(self._levels_by_difficulty[difficulty])
            if have < need:
                raise ValueError(f"difficulty {difficulty} needs {need} distinct levels across the "
                                 f"schedule, has {have}")

    @staticmethod
    def _quotas(ratios, batch_size):
        exact = ratios * batch_size
        quotas = torch.floor(exact).to(dtype=torch.long)
        remainder = batch_size - int(quotas.sum())
        fractions = exact - quotas.to(dtype=exact.dtype)
        # Stable stage-index tie breaking makes quota rounding reproducible.
        order = sorted(range(len(ratios)), key=lambda i: (-float(fractions[i]), i))
        for index in order[:remainder]:
            quotas[index] += 1
        return quotas

    def indices(self, batch_size, progress, generator):
        """Return one uniformly chosen row index for each selected level."""
        batch_size = _positive_integer(batch_size, "batch_size")
        if batch_size > MAX_BATCH_SIZE:
            raise ValueError(f"batch_size cannot exceed {MAX_BATCH_SIZE}")
        if not isinstance(generator, torch.Generator):
            raise ValueError("generator must be a CPU torch.Generator")
        if getattr(generator, "device", torch.device("cpu")).type != "cpu":
            raise ValueError("generator must be a CPU torch.Generator")

        quotas = self._quotas(self.ratios(progress), batch_size)
        available = {difficulty: len(self._levels_by_difficulty[difficulty])
                     for difficulty in self.difficulties}
        insufficient = [(difficulty, int(quotas[difficulty - 1]), available[difficulty])
                       for difficulty in self.difficulties
                       if int(quotas[difficulty - 1]) > available[difficulty]]
        if insufficient:
            details = ", ".join(f"difficulty {d} needs {need} distinct levels, has {have}"
                                for d, need, have in insufficient)
            raise ValueError(f"insufficient distinct level coverage: {details}")

        chosen_rows, chosen_seeds = [], []
        for difficulty in self.difficulties:
            quota = int(quotas[difficulty - 1])
            if not quota:
                continue
            candidates = self._levels_by_difficulty[difficulty]
            selected = torch.randperm(len(candidates), generator=generator)[:quota]
            for candidate_index in selected.tolist():
                seed = candidates[candidate_index]
                rows = self._rows_by_seed[seed]
                row_index = int(torch.randint(len(rows), (), generator=generator).item())
                chosen_rows.append(int(rows[row_index]))
                chosen_seeds.append(seed)

        # Keep batches mixed while retaining deterministic generator behavior.
        order = torch.randperm(batch_size, generator=generator)
        result = torch.tensor(chosen_rows, dtype=torch.long)[order]
        self.last_difficulty_counts = {difficulty: int(quotas[difficulty - 1])
                                       for difficulty in self.difficulties}
        self.last_distinct_levels = len(chosen_seeds)
        self.last_level_seeds = tuple(chosen_seeds[index] for index in order.tolist())
        return result
