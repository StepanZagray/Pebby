"""Normalization folding must preserve the actual projector and grounding heads."""
import copy
import unittest

import torch

from pebby.agent.world_position_recall import PositionRecallPolicy, initialize_from_base
from tests.test_world_model import make_model
from tools.warmup_reference_position import (NAMES, fold_projector, gate,
                                            normalize_projector, unchanged_nonheads)


class WarmupPositionTests(unittest.TestCase):
    def test_normalized_training_weights_fold_into_identical_raw_input_functions(self):
        torch.manual_seed(13)
        parent = make_model(grounding=True, state_recall=True)
        model = PositionRecallPolicy(parent.cfg)
        initialize_from_base(model, parent)
        raw = torch.randn(47, model.projector[0].in_features) * 7 + 4
        # Exercise a constant feature and the newly appended position columns.
        raw[:, 3] = .5
        raw[:, -24:] = torch.rand(47, 24)
        mean, scale = raw.mean(0), raw.std(0, correction=0).clamp_min(1e-6)
        original = model.projector(raw)
        normalize_projector(model.projector[0], mean, scale)
        torch.testing.assert_close(model.projector((raw - mean) / scale), original, atol=2e-5, rtol=2e-5)
        # Simulate learned normalized-space weights, then compare every output
        # on fresh examples rather than simply restoring the original weights.
        with torch.no_grad():
            update = torch.randn_like(model.projector[0].weight) * .003
            update[:, scale == 1e-6] = 0  # Constant normalized inputs receive zero data gradient.
            model.projector[0].weight.add_(update)
        heldout = torch.randn_like(raw) * 3 + 2
        heldout[:, 3] = .5
        expected = model.grounding_head(model.projector((heldout - mean) / scale))
        fold_projector(model.projector[0], mean, scale)
        for a, b in zip(model.grounding_head(model.projector(heldout)), expected):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
        unchanged_nonheads(model, parent)

    def test_frozen_player_or_glyph_must_not_change_during_warmup(self):
        parent = make_model(grounding=True)
        model = PositionRecallPolicy(parent.cfg)
        initialize_from_base(model, parent)
        unchanged_nonheads(model, parent)
        with torch.no_grad():
            model.player_head.weight[0, 0] += .01
        with self.assertRaisesRegex(ValueError, 'player_head'):
            unchanged_nonheads(model, parent)

    def test_gate_requires_both_current_and_actual_nonterminal_fields(self):
        population = {'count': 25, 'fields': {name: {'accuracy': 1.} for name in NAMES}}
        result = {'current': copy.deepcopy(population), 'actual_nonterminal': copy.deepcopy(population)}
        self.assertTrue(gate(result)['passed'])
        result['actual_nonterminal']['fields']['player']['accuracy'] = .5
        self.assertFalse(gate(result)['passed'])
        result['actual_nonterminal'] = copy.deepcopy(population)
        result['current']['count'] = 0
        self.assertFalse(gate(result)['passed'])


if __name__ == '__main__':
    unittest.main()
