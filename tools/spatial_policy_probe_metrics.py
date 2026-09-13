"""Level-weighted decision readouts for the prospective policy-learning pilot."""
from collections import defaultdict

import numpy as np
import torch

from tools.audit_spatial_semantic_decisions import DecisionMetrics
from tools.train_reference_outcomes import batch, guard
from tools.train_reference_spatial_outcomes import player_probabilities


def _counter():
    return dict(roots=0, defined=0, correct=0, set_nll_sum=0., uniform_ce_sum=0.)


def _finish(count):
    n = count['defined']
    return {**count, 'set_nll': count['set_nll_sum'] / n if n else None,
            'uniform_ce': count['uniform_ce_sum'] / n if n else None,
            'accuracy': count['correct'] / n if n else None}


class PanelMetrics:
    """Never treat a long visited trajectory as multiple independent levels."""

    def __init__(self):
        self.micro = DecisionMetrics()
        self.levels = defaultdict(_counter)
        self.level_tiers = {}
        self.cardinality = {n: _counter() for n in range(5)}

    def update(self, scores, items, seeds, tiers):
        seeds, tiers = np.asarray(seeds), np.asarray(tiers)
        if seeds.shape != (len(scores),) or not np.issubdtype(seeds.dtype, np.integer):
            raise ValueError('one integer level seed per decision required')
        values = self.micro.update(scores, items, tiers)
        values = {name: value.detach().cpu().numpy() for name, value in values.items()}
        masks = items['optimal'].detach().cpu().numpy()
        cardinality = np.array([int(mask).bit_count() for mask in masks])
        for index, seed in enumerate(seeds):
            seed, tier = int(seed), int(tiers[index])
            if seed in self.level_tiers and self.level_tiers[seed] != tier:
                raise ValueError('a level cannot belong to multiple tiers')
            self.level_tiers[seed] = tier
            for count in (self.levels[seed], self.cardinality[int(cardinality[index])]):
                count['roots'] += 1
                if values['valid'][index]:
                    count['defined'] += 1
                    count['correct'] += int(values['correct'][index])
                    count['set_nll_sum'] += max(0., float(values['set_nll'][index]))
                    count['uniform_ce_sum'] += float(values['uniform_ce'][index])

    def result(self):
        levels = {str(seed): dict(tier=self.level_tiers[seed], **_finish(count))
                  for seed, count in sorted(self.levels.items())}
        tiers = {}
        for tier in range(1, 8):
            all_levels = [row for row in levels.values() if row['tier'] == tier]
            supported = [row for row in all_levels if row['defined']]
            tiers[str(tier)] = dict(levels=len(all_levels), supported_levels=len(supported),
                **{name: sum(row[name] for row in supported) / len(supported) if supported else None
                   for name in ('set_nll', 'uniform_ce', 'accuracy')})
        eligible = [tier for tier in tiers if tiers[tier]['supported_levels']]
        macro = {name: sum(tiers[tier][name] for tier in eligible) / len(eligible) if eligible else None
                 for name in ('set_nll', 'uniform_ce', 'accuracy')}
        macro.update(included_tiers=eligible, excluded_tiers=[tier for tier in tiers if tier not in eligible])
        return dict(rows=sum(count['roots'] for count in self.levels.values()), levels=len(levels),
            macro=macro, micro=self.micro.result(), per_level=levels, per_tier=tiers,
            cardinality={str(n): _finish(count) for n, count in self.cardinality.items()},
            definition='Average defined roots within each level, then supported levels within each tier, then supported tiers equally; cardinality and micro rows are descriptive.',
            limits='Continuation-held-out levels were seen in parent pretraining. No independence or fresh-confirmation claim for repeated roots or reused validation.')


def evaluate_panel(model, arrays, indices, level_tiers, encoder, device='cuda', batch_size=128):
    """Evaluate each specified root once with only the five public model inputs."""
    indices = np.asarray(indices)
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or len(np.unique(indices)) != len(indices)
            or np.any((indices < 0) | (indices >= len(arrays['seeds'])))):
        raise ValueError('panel indices must be unique in-range integer rows')
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('positive batch size required')
    seeds = np.asarray(arrays['seeds'])[indices]
    if any(int(seed) not in level_tiers or level_tiers[int(seed)] not in range(1, 8) for seed in seeds):
        raise ValueError('panel level tier mapping is incomplete or invalid')
    metrics, was_training = PanelMetrics(), model.training
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(indices), batch_size):
                guard()
                chosen = indices[start:start + batch_size]
                current_seeds = np.asarray(arrays['seeds'])[chosen]
                items = batch(arrays, chosen, device)
                predicted = model(items['raw'], items['state'], items['glyph'],
                    player_probabilities(items, encoder), items['semantic'])
                metrics.update(predicted['action_logits'], items, current_seeds,
                               [level_tiers[int(seed)] for seed in current_seeds])
    finally:
        model.train(was_training)
    return metrics.result()
