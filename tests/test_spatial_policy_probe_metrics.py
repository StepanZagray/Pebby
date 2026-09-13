import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tests.test_train_spatial_semantic_repair import arrays
from tools.spatial_policy_probe_metrics import PanelMetrics, evaluate_panel


def items(masks):
    values = arrays(np.arange(len(masks)))
    values['optimal'][:] = masks
    return {key: torch.from_numpy(value) for key, value in values.items()}


class PanelMetricTests(unittest.TestCase):
    def test_long_trajectories_do_not_dominate_level_or_tier_macro(self):
        scores = torch.tensor([[.1, .3, .3, .3]] * 9 + [[.9, .03, .03, .04], [.8, .1, .05, .05]]).log()
        metric = PanelMetrics()
        metric.update(scores, items([1] * 11), [11] * 9 + [22, 33], [1] * 10 + [2])
        report = metric.result()
        self.assertAlmostEqual(report['macro']['accuracy'], .75)
        self.assertAlmostEqual(report['micro']['all']['optimal_set_accuracy'], 2 / 11)
        expected = ((-math.log(.1) - math.log(.9)) / 2 - math.log(.8)) / 2
        self.assertAlmostEqual(report['macro']['set_nll'], expected, places=6)
        self.assertEqual(report['macro']['included_tiers'], ['1', '2'])

    def test_undefined_and_alloptimal_cardinalities_are_reported_separately(self):
        metric = PanelMetrics()
        metric.update(torch.zeros(4, 4), items([0, 1, 3, 15]), [11, 11, 22, 33], [1, 1, 1, 2])
        report = metric.result()
        self.assertIsNone(report['cardinality']['0']['set_nll'])
        self.assertEqual(report['cardinality']['0']['roots'], 1)
        self.assertEqual(report['cardinality']['4']['accuracy'], 1.)
        self.assertAlmostEqual(report['cardinality']['4']['set_nll'], 0.)
        self.assertEqual(report['per_level']['11']['defined'], 1)
        self.assertEqual(report['per_level']['11']['roots'], 2)

    def test_allundefined_panel_has_no_macro_estimand(self):
        metric = PanelMetrics()
        metric.update(torch.zeros(2, 4), items([0, 0]), [11, 22], [1, 2])
        report = metric.result()
        self.assertIsNone(report['macro']['set_nll'])
        self.assertIsNone(report['macro']['accuracy'])
        self.assertEqual(report['levels'], 2)

    def test_repeated_batches_aggregate_one_level_and_reject_conflicting_tier(self):
        metric = PanelMetrics()
        for _ in range(2):
            metric.update(torch.zeros(1, 4), items([1]), [11], [1])
        self.assertEqual(metric.result()['levels'], 1)
        self.assertEqual(metric.result()['per_level']['11']['defined'], 2)
        with self.assertRaisesRegex(ValueError, 'multiple tiers'):
            metric.update(torch.zeros(1, 4), items([1]), [11], [2])

    def test_evaluation_preserves_mode_rng_and_exact_row_panel(self):
        data = arrays([11, 22, 33, 44]); data['glyph'][:, 0] = np.arange(4)
        seen = []
        class Model(torch.nn.Module):
            def forward(self, raw, state, glyph, player, semantic):
                if self.training or torch.is_grad_enabled():
                    raise AssertionError('evaluation used training or gradient mode')
                seen.extend(glyph[:, 0].long().tolist())
                return {'action_logits': torch.zeros(len(raw), 4)}
        model = Model().train(); rng = torch.get_rng_state().clone()
        with patch('tools.spatial_policy_probe_metrics.player_probabilities',
                   side_effect=lambda x, _: torch.ones(len(x['raw']), 144) / 144):
            result = evaluate_panel(model, data, np.array([3, 1]), {22: 2, 44: 7}, {}, 'cpu', 1)
        self.assertEqual(seen, [3, 1]); self.assertEqual(result['rows'], 2)
        self.assertTrue(model.training); self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for rows in (np.array([1, 1]), np.array([-1]), np.array([4]), np.array([1.])):
            with self.assertRaises(ValueError):
                evaluate_panel(model, data, rows, {}, {}, 'cpu')
