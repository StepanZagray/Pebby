"""Matched level-first replay with independent dynamics and policy eligibility."""
import numpy as np


# D7 has only 100 original TRAIN levels. Every batch keeps distinct levels.
TIER_COUNTS = (176, 176, 176, 176, 160, 80, 80)


def grouped_rows(seeds, allowed=None):
    if allowed is None:
        allowed = np.arange(len(seeds))
    allowed = np.asarray(allowed)
    if (allowed.ndim != 1 or not np.issubdtype(allowed.dtype, np.integer)
            or len(np.unique(allowed)) != len(allowed) or np.any((allowed < 0) | (allowed >= len(seeds)))):
        raise ValueError('approved rows must be unique in-range integer indices')
    order = allowed[np.argsort(np.asarray(seeds)[allowed], kind='stable')]
    sorted_seeds = np.asarray(seeds)[order]
    cuts = np.flatnonzero(np.diff(sorted_seeds)) + 1
    return {int(sorted_seeds[rows[0]]): order[rows] for rows in np.split(np.arange(len(order)), cuts) if len(rows)}


class RouteReplaySampler:
    """All verified roots retain dynamics labels, even when optimal is zero.

    A fixed per-tier fraction of distinct levels receives recent-policy replay.
    Recent replay first chooses a trajectory kind, then a row, avoiding long
    refusal trajectories taking all the mass. Original rows stay level-uniform.
    """

    def __init__(self, base, recent, level_tiers, *, tier_counts=TIER_COUNTS,
                 recent_fraction=.25, kind_weights=(.5, .375, .125), base_rows=None, recent_rows=None):
        if (len(tier_counts) != 7 or any(type(n) is not int or n < 0 for n in tier_counts)
                or sum(tier_counts) < 1 or not 0 <= recent_fraction <= 1):
            raise ValueError('seven nonnegative tier counts and a replay fraction in [0,1] required')
        weights = np.asarray(kind_weights, dtype=float)
        if weights.shape != (3,) or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError('three finite nonnegative trajectory-kind weights required')
        self.kind_weights = weights / weights.sum()
        self.base = grouped_rows(base['seeds'], base_rows)
        self.recent = {} if recent is None else grouped_rows(recent['seeds'], recent_rows)
        if not set(self.recent) <= set(self.base):
            raise ValueError('recent replay must belong to the original TRAIN levels')
        if set(self.base) != set(level_tiers) or any(t not in range(1, 8) for t in level_tiers.values()):
            raise ValueError('complete original TRAIN tier mapping required')
        if recent is None and recent_fraction:
            raise ValueError('nonzero replay fraction requires a recent cache')
        if recent is not None:
            if not np.isin(recent['row_kind'], [0, 1, 2]).all():
                raise ValueError('invalid recent trajectory kind')
            if 'dynamics_valid' in recent and not np.asarray(recent['dynamics_valid']).all():
                raise ValueError('unverified dynamics rows cannot enter replay')
            if 'policy_valid' in recent and not np.array_equal(recent['policy_valid'], recent['optimal'] != 0):
                raise ValueError('policy eligibility must exactly follow the nonzero optimal mask')
        self.kinds = {}
        for seed, rows in self.recent.items():
            self.kinds[seed] = {kind: rows[np.asarray(recent['row_kind'])[rows] == kind]
                                for kind in range(3)}
        self.tiers = []
        for tier, count in enumerate(tier_counts, 1):
            seeds = np.array(sorted(s for s in self.base if level_tiers[s] == tier), dtype=np.int64)
            eligible = np.array([s for s in seeds if int(s) in self.recent], dtype=np.int64)
            reserved = round(count * recent_fraction)
            if len(seeds) < count or len(eligible) < reserved:
                raise ValueError(f'tier {tier} cannot supply {count} distinct levels / {reserved} recent levels')
            self.tiers.append((seeds, eligible, count, reserved))
        self.batch_size = sum(tier_counts)

    def sample(self, rng):
        seeds, base_rows, recent_rows, kinds = [], [], [], []
        for available, eligible, count, reserved in self.tiers:
            replay = rng.choice(eligible, reserved, replace=False)
            remaining = available[~np.isin(available, replay)]
            chosen = np.concatenate((replay, rng.choice(remaining, count - reserved, replace=False)))
            for index, value in enumerate(chosen):
                seed = int(value)
                seeds.append(seed)
                base_rows.append(int(rng.choice(self.base[seed])))
                row, kind = -1, -1
                if index < reserved:
                    available_kinds = np.array([k for k, rows in self.kinds[seed].items()
                                                if len(rows) and self.kind_weights[k] > 0])
                    if not len(available_kinds):
                        raise ValueError('recent level has no positively weighted trajectory kind')
                    weights = self.kind_weights[available_kinds]
                    kind = int(rng.choice(available_kinds, p=weights / weights.sum()))
                    row = int(rng.choice(self.kinds[seed][kind]))
                recent_rows.append(row)
                kinds.append(kind)
        order = rng.permutation(len(seeds))
        result = {name: np.asarray(values, dtype=np.int64)[order] for name, values in
                  [('seeds', seeds), ('base_rows', base_rows), ('recent_rows', recent_rows), ('row_kinds', kinds)]}
        if len(np.unique(result['seeds'])) != self.batch_size:
            raise ValueError('replay batch repeats a level')
        return result

    def config(self):
        return dict(batch_size=self.batch_size, tier_counts=[item[2] for item in self.tiers],
                    recent_counts=[item[3] for item in self.tiers], kind_weights=self.kind_weights.tolist(),
                    base_levels=len(self.base), recent_levels=len(self.recent),
                    original_zero_policy_rows_retained=True)
