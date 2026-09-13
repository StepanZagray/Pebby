"""CPU tests for exact optimal masks, interventions and resource denominators."""
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pebby.agent.spatial_outcome_planner import SpatialOutcomePlanner
from pebby.agent.spatial_semantic_outcome_planner import SpatialSemanticOutcomePlanner
from tools import audit_spatial_semantic_decisions as audit


def items(count=4):
    return dict(optimal=torch.tensor([3, 2, 0, 15], dtype=torch.uint8)[:count],
        current_steps=torch.full((count,), 8, dtype=torch.int16),
        next_steps=torch.full((count, 4), 7, dtype=torch.int16),
        next_lives=torch.full((count, 4), 3, dtype=torch.int16),
        lost_life=torch.zeros(count, 4, dtype=torch.bool), terminal=torch.zeros(count, 4, dtype=torch.bool),
        won=torch.zeros(count, 4, dtype=torch.bool), distances=torch.ones(count, 4, dtype=torch.int16))


class SemanticDecisionAuditTests(unittest.TestCase):
    def test_policy_undefined_masks_and_set_nll_differ_from_uniform_ce(self):
        labels = items(); scores = torch.zeros(4, 4)
        metric = audit.DecisionMetrics(); values = metric.update(scores, labels, [1, 2, 3, 7])
        result = metric.result()['all']
        self.assertEqual(result['policy_defined'], 3)
        self.assertEqual(result['policy_undefined'], 1)
        self.assertEqual(result['correct'], 2)
        self.assertAlmostEqual(result['uniform_optimal_ce'], math.log(4), places=6)
        self.assertAlmostEqual(result['optimal_set_nll'], (math.log(2) + math.log(4)) / 3, places=6)
        self.assertFalse(values['valid'][2])
        self.assertIsNone(metric.result()['3']['uniform_optimal_ce'])
        self.assertEqual(metric.result()['3']['roots'], 1)

    def test_batch_aggregation_matches_single_panel_including_empty_policy_batch(self):
        labels = items(); scores = torch.tensor([[1., 0., 2., 3.], [0., 2., 0., 0.], [3., 1., 0., 2.], [0., 0., 0., 0.]])
        whole = audit.DecisionMetrics(); split = audit.DecisionMetrics()
        whole.update(scores, labels, [1, 2, 3, 7])
        for row, tier in enumerate([1, 2, 3, 7]):
            split.update(scores[row:row + 1], {k: v[row:row + 1] for k, v in labels.items()}, [tier])
        self.assertEqual(whole.result(), split.result())

    def test_optional_refill_is_separate_from_required_refill(self):
        labels = items(2)
        # Row0: both actions0/1 optimal, only1 refills. Row1: action1 alone optimal/refills.
        labels['next_steps'][:, 1] = 12
        scores = torch.tensor([[3., 0., 0., 0.], [3., 0., 0., 0.]])
        metric = audit.DecisionMetrics(); metric.update(scores, labels, [1, 2]); result = metric.result()['all']
        self.assertEqual(result['optimal_live_refill_available'], 2)
        self.assertEqual(result['missed_available_optimal_live_refill'], 2)
        self.assertEqual(result['optimal_live_refill_required'], 1)
        self.assertEqual(result['missed_required_optimal_live_refill'], 1)
        self.assertEqual(result['correct'], 1)
        # Budget reset at death must never count as a live refill.
        labels['lost_life'][1, 1] = True
        changed = audit.decision_values(scores, labels)
        self.assertEqual(changed['optimal_refill_available'].tolist(), [True, False])

    def test_unreachable_choice_requires_a_reachable_alternative(self):
        labels = items(); labels['distances'][:] = -1
        labels['distances'][0, 2] = 4
        values = audit.decision_values(torch.zeros(4, 4), labels)
        self.assertEqual(values['has_reachable'].tolist(), [True, False, False, False])
        metric = audit.DecisionMetrics(); metric.update(torch.zeros(4, 4), labels, [1, 1, 1, 1])
        result = metric.result()['all']
        self.assertEqual(result['reachable_roots'], 1)
        self.assertEqual(result['reachable_to_unreachable'], 1)

    def test_comparison_counts_help_harm_and_excludes_undefined_targets(self):
        labels = items(); left = torch.tensor([[3., 0., 0., 0.], [3., 0., 0., 0.], [3., 0., 0., 0.], [3., 0., 0., 0.]])
        right = torch.tensor([[0., 0., 3., 0.], [0., 3., 0., 0.], [0., 0., 0., 3.], [0., 0., 3., 0.]])
        result = audit.compare_decisions(audit.decision_values(left, labels), audit.decision_values(right, labels))
        self.assertEqual(result, dict(roots=4, policy_defined=3, action_changes=4, policy_defined_action_changes=3, helped=1, hurt=1))

    def test_components_match_native_and_actor_removed_can_change_choice(self):
        torch.set_num_threads(1); torch.manual_seed(19)
        parent = SpatialOutcomePlanner().eval()
        model = SpatialSemanticOutcomePlanner.from_parent(parent, actor=True).eval()
        with torch.no_grad():
            model.actor_readout.output_projection.weight.normal_(std=.4)
        raw = torch.randn(2, 160, 64); state = torch.randn_like(raw); glyph = torch.randn(2, 14)
        player = torch.randn(2, 144).softmax(-1); semantic = torch.rand(2, 144, 22)
        for start, stop in ((8, 14), (14, 18), (18, 22)):
            semantic[..., start:stop] = semantic[..., start:stop].softmax(-1)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        with torch.inference_mode():
            pred = model(raw, state, glyph, player, semantic, return_components=True)
            normal = model(raw, state, glyph, player, semantic)
            shuffled = model(raw, state, glyph, player, semantic.roll(37, dims=1))
        torch.testing.assert_close(pred['action_logits'], pred['outcome_action_logits'] + pred['actor_action_logits'], atol=0, rtol=0)
        torch.testing.assert_close(pred['action_logits'], normal['action_logits'], atol=0, rtol=0)
        self.assertTrue(bool((pred['actor_action_logits'] != 0).any()))
        self.assertEqual(shuffled['action_logits'].shape, (2, 4))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_alignment_checks_rows_and_seeds_and_native_score_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp); arrays = dict(rows=np.array([2, 7]), seeds=np.array([1, 2]))
            np.save(path / 'rows.npy', arrays['rows']); np.save(path / 'seeds.npy', arrays['seeds'])
            np.save(path / 'semantic.npy', np.zeros((2, 144, 22), np.float32))
            value = audit.validate_alignment(arrays, path); audit.release({'semantic': value}, close=True)
            np.save(path / 'seeds.npy', np.array([2, 1]))
            with self.assertRaisesRegex(ValueError, 'alignment'):
                audit.validate_alignment(arrays, path)
        self.assertEqual(audit.native_score_weight(dict(score_weights=dict(direct=0., planner=2.))), 2.)
        for weights in [dict(direct=.1, planner=1.), dict(direct=0., planner=0.), dict(direct=0., planner=float('nan'))]:
            with self.assertRaises(ValueError):
                audit.native_score_weight(dict(score_weights=weights))

    def test_nonfinite_scores_or_invalid_optimal_bits_fail(self):
        labels = items(); labels['optimal'][0] = 16
        with self.assertRaises(ValueError): audit.decision_values(torch.zeros(4, 4), labels)
        labels['optimal'][0] = 3; scores = torch.zeros(4, 4); scores[0, 0] = float('nan')
        with self.assertRaises(ValueError): audit.decision_values(scores, labels)


if __name__ == '__main__':
    unittest.main()
