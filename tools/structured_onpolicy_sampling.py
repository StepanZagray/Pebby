"""Paired level-balanced batches for an on-policy state-distribution experiment.

Both arms see the same distinct levels and legacy view draws. The treatment
replaces a fixed subset with a uniformly sampled visited state from that level.
Bound auxiliary expert/failure rows can occupy a configured fraction of the
same-level treatment replacements. Their original on-policy flags stay false;
the training reports distinguish the two sources. Legacy caches are unchanged.
"""
from dataclasses import dataclass

import numpy as np
from pebby.ls20.provenance import validate_row_difficulties


@dataclass(frozen=True)
class PairedRows:
    base_rows: np.ndarray
    base_views: np.ndarray
    trajectory_rows: np.ndarray


class PairedStateSampler:
    def __init__(self, base_seeds, difficulties, trajectory_seeds, on_policy, *, auxiliary=None, auxiliary_fraction=.25, difficulty_metadata=None):
        self.seeds = np.asarray(base_seeds)
        self.difficulty = np.asarray(difficulties)
        visited = np.asarray(trajectory_seeds)
        flags = np.asarray(on_policy)
        if (self.seeds.ndim != 1 or not len(self.seeds)
                or not np.issubdtype(self.seeds.dtype, np.integer)
                or len(np.unique(self.seeds)) != len(self.seeds)
                or np.any((self.seeds < 0) | (self.seeds >= 1_000_000))):
            raise ValueError('distinct generated TRAIN base levels required')
        if (self.difficulty.shape != self.seeds.shape
                or not np.issubdtype(self.difficulty.dtype, np.integer)):
            raise ValueError('base difficulties must be integer row labels')
        self.difficulties = validate_row_difficulties(self.seeds, self.difficulty, difficulty_metadata or {})
        if (visited.ndim != 1 or not np.issubdtype(visited.dtype, np.integer)
                or flags.shape != visited.shape or flags.dtype != np.bool_
                or not flags.any()):
            raise ValueError('visited TRAIN seeds and nonempty boolean on-policy mask required')
        lookup = {int(seed): row for row, seed in enumerate(self.seeds)}
        if any(int(seed) not in lookup for seed in visited):
            raise ValueError('every trajectory level must have a paired base level')
        grouped = {}
        for row in np.flatnonzero(flags):
            base_row = lookup[int(visited[row])]
            grouped.setdefault(base_row, []).append(int(row))
        self.groups = {key: np.asarray(rows, dtype=np.int64) for key, rows in grouped.items()}
        self.eligible = np.asarray(sorted(self.groups), dtype=np.int64)
        if not np.isfinite(auxiliary_fraction) or not 0 <= auxiliary_fraction <= 1:
            raise ValueError('auxiliary fraction must be0..1')
        self.auxiliary_fraction = auxiliary_fraction
        self.auxiliary_groups = {}
        if auxiliary is not None:
            auxiliary = np.asarray(auxiliary)
            if auxiliary.shape != flags.shape or auxiliary.dtype != np.bool_ or np.any(auxiliary & flags):
                raise ValueError('auxiliary rows must be a disjoint boolean mask')
            for row in np.flatnonzero(auxiliary):
                self.auxiliary_groups.setdefault(lookup[int(visited[row])], []).append(int(row))

    def draw(self, batch_size, replacements, progress, rng):
        if (type(batch_size) is not int or batch_size < 1 or batch_size > 1024
                or batch_size & (batch_size - 1) or batch_size > len(self.seeds)):
            raise ValueError('batch must be a feasible power of two at most1024')
        if (type(replacements) is not int or not 1 <= replacements <= batch_size
                or replacements > len(self.eligible)):
            raise ValueError('replacement count exceeds distinct eligible levels')
        if not np.isfinite(progress) or not 0 <= progress <= 1:
            raise ValueError('curriculum progress must be0..1')
        weights = np.exp((2 * progress - 1) * (self.difficulty.astype(float) - 3) * .5)
        eligible_weights = weights[self.eligible]
        selected = rng.choice(self.eligible, replacements, replace=False,
                              p=eligible_weights / eligible_weights.sum())
        remaining_weights = weights.copy()
        remaining_weights[selected] = 0
        rest = rng.choice(len(self.seeds), batch_size - replacements, replace=False,
                          p=remaining_weights / remaining_weights.sum()) if batch_size > replacements else np.empty(0, np.int64)
        base = np.concatenate((selected, rest)).astype(np.int64)
        trajectory = np.full(batch_size, -1, dtype=np.int64)
        for index, row in enumerate(selected):
            auxiliary = self.auxiliary_groups.get(int(row))
            pool = (auxiliary if auxiliary and rng.random() < self.auxiliary_fraction else self.groups[int(row)])
            trajectory[index] = rng.choice(pool)
        # Balanced legacy views for the paired control, independently shuffled.
        views = (np.arange(batch_size) % 2).astype(np.int8)
        rng.shuffle(views)
        order = rng.permutation(batch_size)
        return PairedRows(base[order], views[order], trajectory[order])
