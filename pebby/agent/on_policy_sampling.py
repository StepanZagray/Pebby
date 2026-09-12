"""Mix model-visited rows into exact difficulty quotas without duplicate levels."""
import math
from numbers import Integral

import torch

from .curriculum_sampling import CurriculumSampler


class OnPolicySampler(CurriculumSampler):
    """Return base indices, supplemental indices and their final batch permutation.

    Only ``meta.on_policy_rows`` enter the reserved model-visited fraction.
    New ``meta.auxiliary_rows`` (expert/failure trajectories) may replace an
    otherwise-base draw for the same level with ``auxiliary_fraction`` chance.
    Legacy supplemental archives without that metadata keep their old behavior.
    """
    def __init__(self, data, supplemental, *, fraction, auxiliary_fraction=.25, **kwargs):
        super().__init__(data, **kwargs)
        if isinstance(fraction, bool) or not math.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError('on-policy fraction must be finite and between 0 and 1')
        self.fraction = float(fraction)
        if isinstance(auxiliary_fraction, bool) or not math.isfinite(auxiliary_fraction) or not 0 <= auxiliary_fraction <= 1:
            raise ValueError('auxiliary fraction must be finite and between 0 and 1')
        self.auxiliary_fraction = float(auxiliary_fraction)
        indices = supplemental['meta'].get('on_policy_rows')
        if not isinstance(indices, list) or not indices:
            raise ValueError('supplemental metadata needs nonempty on_policy_rows')
        if any(isinstance(i, bool) or not isinstance(i, Integral) or
               not 0 <= i < len(supplemental['seeds']) for i in indices):
            raise ValueError('on_policy_rows contains invalid indices')
        if len(set(indices)) != len(indices):
            raise ValueError('on_policy_rows contains duplicate indices')
        policy = CurriculumSampler(supplemental, **kwargs)
        if policy.difficulty_version != self.difficulty_version:
            raise ValueError("base and supplemental difficulty versions differ")
        base_stage = {seed: stage for stage, seeds in self._levels_by_difficulty.items() for seed in seeds}
        for stage, seeds in policy._levels_by_difficulty.items():
            if any(base_stage.get(seed) != stage for seed in seeds):
                raise ValueError('supplemental levels must match base training seeds and difficulties')
        self._on_rows = {}
        for index in indices:
            seed = int(supplemental['seeds'][index])
            if seed not in base_stage:
                raise ValueError('on-policy row is not a base training level')
            self._on_rows.setdefault(seed, []).append(index)
        self._on_levels = {stage: [s for s in self._levels_by_difficulty[stage] if s in self._on_rows]
                           for stage in self.difficulties}
        auxiliary = supplemental['meta'].get('auxiliary_rows', [])
        if not isinstance(auxiliary, list) or any(type(i) is not int or not 0 <= i < len(supplemental['seeds']) for i in auxiliary):
            raise ValueError('auxiliary_rows contains invalid indices')
        if len(set(auxiliary)) != len(auxiliary) or set(auxiliary) & set(indices):
            raise ValueError('auxiliary_rows must be distinct from on_policy_rows')
        self.auxiliary_indices = tuple(auxiliary)
        self._auxiliary_rows = {}
        for index in auxiliary:
            seed = int(supplemental['seeds'][index])
            if seed not in base_stage:
                raise ValueError('auxiliary row is not a base training level')
            self._auxiliary_rows.setdefault(seed, []).append(index)
        self.last_on_policy_count = 0
        self.last_auxiliary_count = 0

    def reserved_count(self, batch_size):
        count = round(batch_size * self.fraction)
        if not 0 < count < batch_size:
            raise ValueError('batch must reserve at least one row for each data source')
        return count

    def check_coverage(self, batch_size):
        super().check_coverage(batch_size)
        count = self.reserved_count(batch_size)
        required = torch.ceil(torch.maximum(self._start, self._end) * count).long()
        for stage in self.difficulties:
            if len(self._on_levels[stage]) < int(required[stage - 1]):
                raise ValueError(f'on-policy difficulty {stage} needs {int(required[stage-1])} distinct levels, '
                                 f'has {len(self._on_levels[stage])}')

    def indices(self, batch_size, progress, generator):
        self.check_coverage(batch_size)
        if not isinstance(generator, torch.Generator) or generator.device.type != 'cpu':
            raise ValueError('generator must be a CPU torch.Generator')
        ratios = self.ratios(progress)
        total = self._quotas(ratios, batch_size)
        exact = ratios * self.reserved_count(batch_size)
        reserved = torch.minimum(torch.floor(exact).long(), total)
        # Capped largest remainder prevents pathological custom schedules from
        # assigning a reserved row to a stage with zero total rows.
        for _ in range(self.reserved_count(batch_size) - int(reserved.sum())):
            eligible = [i for i in range(len(total)) if reserved[i] < total[i]]
            best = min(eligible, key=lambda i: (-float(exact[i] - reserved[i]), i))
            reserved[best] += 1
        base_rows, policy_rows, base_seeds, policy_seeds = [], [], [], []
        auxiliary_count = 0
        for stage in self.difficulties:
            need = int(reserved[stage - 1])
            candidates = self._on_levels[stage]
            selected = [candidates[i] for i in torch.randperm(len(candidates), generator=generator)[:need].tolist()]
            selected_set = set(selected)
            for seed in selected:
                rows = self._on_rows[seed]
                policy_rows.append(rows[int(torch.randint(len(rows), (), generator=generator))])
                policy_seeds.append(seed)
            remaining = [s for s in self._levels_by_difficulty[stage] if s not in selected_set]
            need = int(total[stage - 1]) - need
            for i in torch.randperm(len(remaining), generator=generator)[:need].tolist():
                seed = remaining[i]
                auxiliary = self._auxiliary_rows.get(seed)
                if auxiliary and float(torch.rand((), generator=generator)) < self.auxiliary_fraction:
                    policy_rows.append(auxiliary[int(torch.randint(len(auxiliary), (), generator=generator))])
                    policy_seeds.append(seed)
                    auxiliary_count += 1
                    continue
                rows = self._rows_by_seed[seed]
                base_rows.append(int(rows[int(torch.randint(len(rows), (), generator=generator))]))
                base_seeds.append(seed)
        chosen = base_seeds + policy_seeds
        if len(chosen) != batch_size or len(set(chosen)) != batch_size:
            raise RuntimeError('on-policy sampling violated distinct-level batch contract')
        order = torch.randperm(batch_size, generator=generator)
        self.last_level_seeds = tuple(chosen[i] for i in order.tolist())
        self.last_distinct_levels = len(chosen)
        self.last_difficulty_counts = {stage: int(total[stage-1]) for stage in self.difficulties}
        self.last_on_policy_count = len(policy_rows) - auxiliary_count
        self.last_auxiliary_count = auxiliary_count
        return torch.tensor(base_rows, dtype=torch.long), torch.tensor(policy_rows, dtype=torch.long), order


def mixed_batch(base, supplemental, indices):
    """Materialize only this batch; both complete banks may remain memory mapped."""
    if base.keys() != supplemental.keys():
        raise ValueError('on-policy and base tensor fields differ')
    if any(value.dtype != supplemental[name].dtype or value.shape[1:] != supplemental[name].shape[1:]
           for name, value in base.items()):
        raise ValueError('on-policy and base tensor shapes or dtypes differ')
    base_indices, policy_indices, order = indices
    return {name: torch.cat((value[base_indices], supplemental[name][policy_indices]))[order]
            for name, value in base.items()}
