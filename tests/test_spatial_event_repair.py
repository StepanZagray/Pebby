"""CPU contracts for separate dynamics eligibility and frozen-summary repair."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from pebby.agent.spatial_outcome_planner import SpatialOutcomePlanner
from tools import train_spatial_event_repair as repair


def targets():
    return dict(current_steps=torch.tensor([8, 8]), current_lives=torch.tensor([3, 3]),
                next_steps=torch.tensor([[7, 8, 42, -3], [7, 7, 7, 7]]),
                next_lives=torch.tensor([[3, 3, 3, 2], [3, 3, 3, 3]]),
                lost_life=torch.tensor([[False, False, False, True], [False] * 4]),
                terminal=torch.tensor([[False] * 4, [False, False, False, True]]),
                won=torch.tensor([[False] * 4, [False, False, False, True]]),
                optimal=torch.zeros(2, dtype=torch.uint8))


def model():
    torch.manual_seed(42)
    return SpatialOutcomePlanner(channels=4, width=8, hud_width=8, summary=8, comparator_hidden=8).eval()


class SpatialEventRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        assert not torch.cuda.is_initialized()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        assert not torch.cuda.is_initialized()

    def weights(self, items):
        return repair.cohort_weights({key: value.numpy() for key, value in items.items()})

    def test_cohorts_preserve_terminal_and_canonical_exhausted_budget(self):
        value = repair.repair_targets(targets())
        self.assertEqual(value['budget_group'].tolist(), [[0, 1, 2, 3], [0, 0, 0, 4]])
        self.assertEqual(value['steps'][0].tolist(), [8, 9, 43, 0])
        # A life-loss branch stays in its own cohort even when also terminal.
        items = targets()
        items['terminal'][0, 3] = True
        self.assertEqual(repair.repair_targets(items)['budget_group'][0, 3], 3)

    def test_fixed_train_weights_equalize_cohorts_without_dropping_terminal(self):
        weights = self.weights(targets())
        counts, values = np.array(weights['budget_counts']), np.array(weights['budget_weights'])
        np.testing.assert_allclose((counts * values)[:4], np.full(4, counts[:4].sum() / 4))
        self.assertEqual(values[4], 1)
        self.assertAlmostEqual(float((counts * values).sum()), float(counts.sum()))
        lives = np.array(weights['lives_counts']) * weights['lives_weights']
        np.testing.assert_allclose(lives, [4, 4])

    def test_sampler_is_matched_level_uniform_and_keeps_zero_action_roots(self):
        seeds = np.repeat(np.arange(1100), 8)
        sampler = repair.LevelSampler(seeds, np.arange(2000, 2010))
        left, right = np.random.default_rng(42), np.random.default_rng(42)
        visited = set()
        for _ in range(20):
            rows = sampler.sample(left)
            np.testing.assert_array_equal(rows, sampler.sample(right))
            self.assertEqual(len(np.unique(seeds[rows])), 1024)
            visited.update((rows % 8).tolist())
        self.assertEqual(visited, set(range(8)))
        # Eligibility is independent of optimal masks; this sampler accepts only seeds.
        with self.assertRaisesRegex(ValueError, 'overlap'):
            repair.LevelSampler(seeds, np.array([1]))
        with self.assertRaisesRegex(ValueError, 'eight roots'):
            repair.LevelSampler(seeds[:-1], np.array([2000]))

    def test_zero_action_masks_train_exactly_three_heads_and_events_match_arms(self):
        network, items = model(), targets()
        names = repair.freeze_except_repair_heads(network)
        self.assertEqual(len(names), 6)
        before = repair.frozen_digest(network)
        summary = torch.randn(2, 4, 8)
        weights = self.weights(items)
        natural = repair.repair_loss(network, summary, items, weights, False)
        balanced = repair.repair_loss(network, summary, items, weights, True)
        torch.testing.assert_close(natural['losses']['events'], balanced['losses']['events'], atol=0, rtol=0)
        self.assertNotEqual(float(natural['losses']['steps'].detach()), float(balanced['losses']['steps'].detach()))
        optimizer = repair.optimizer_for(network, .01)
        repair.fit_step(network, optimizer, summary, items, weights, True)
        self.assertEqual(repair.frozen_digest(network), before)
        for name, parameter in network.named_parameters():
            if name in names:
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0)
            else:
                self.assertIsNone(parameter.grad)

    def test_natural_loss_has_no_original_positive_weights_or_policy_dependency(self):
        network, items = model(), targets()
        summary = torch.randn(2, 4, 8)
        result = repair.repair_loss(network, summary, items, {}, False)
        pred, target = result['prediction'], result['target']
        expected = F.cross_entropy(pred['steps'].flatten(0, 1), target['steps'].flatten())
        expected += F.cross_entropy(pred['lives'].flatten(0, 1), target['lives'].flatten())
        expected += F.binary_cross_entropy_with_logits(pred['events'], target['events'])
        torch.testing.assert_close(result['total'], expected, atol=1e-6, rtol=1e-6)
        items['optimal'].fill_(15)
        torch.testing.assert_close(result['total'], repair.repair_loss(network, summary, items, {}, False)['total'], atol=0, rtol=0)

    def test_missing_batch_cohorts_still_have_finite_fixed_weight_loss(self):
        network, items = model(), targets()
        weights = self.weights(items)
        for key in ('lost_life', 'terminal', 'won'):
            items[key].zero_()
        items['next_steps'].fill_(7)
        result = repair.repair_loss(network, torch.randn(2, 4, 8), items, weights, True)
        self.assertTrue(torch.isfinite(result['total']))
        with self.assertRaisesRegex(ValueError, 'each of four budget'):
            self.weights(items)

    def test_summary_hook_reproduces_heads_and_unaffected_native_outputs(self):
        network, items = model(), targets()
        items.update(raw=torch.randn(2, 160, 4), state=torch.randn(2, 160, 4), glyph=torch.randn(2, 14))
        player = {'player_head.weight': torch.randn(1, 4), 'player_head.bias': torch.randn(1)}
        with torch.inference_mode():
            before = network(items['raw'], items['state'], items['glyph'], repair.player_probabilities(items, player))
        arrays = {key: value.numpy() for key, value in items.items()}
        arrays['seeds'] = np.array([1, 2])
        with tempfile.TemporaryDirectory() as directory, patch.object(repair, 'guard', return_value=10 * 2**30):
            cached = repair.cache_summaries(network, arrays, player, Path(directory) / 'summary.npy', device='cpu', size=2)
            summary = torch.from_numpy(np.array(cached))
            pred = repair.head_predictions(network, summary)
            torch.testing.assert_close(pred['steps'], before['field_logits'][4], atol=0, rtol=0)
            repair.release({'summary': cached}, close=True)
        self.assertFalse(network.summary_head._forward_hooks)
        repair.freeze_except_repair_heads(network)
        repair.fit_step(network, repair.optimizer_for(network, .1), summary, items, self.weights(items), True)
        check = repair.verify_frozen_predictions(network, items, player, before)
        self.assertTrue(check['unaffected_fields_and_value_bitwise_unchanged'])
        self.assertTrue(check['fixed_input_comparator_bitwise_unchanged'])

    def test_tiny_separable_summaries_can_fit_all_three_heads(self):
        # A synthetic plumbing prerequisite, not a claim about real frozen features.
        network, items = model(), targets()
        repair.freeze_except_repair_heads(network)
        summary = torch.eye(8).reshape(2, 4, 8)
        optimizer = torch.optim.Adam([p for p in network.parameters() if p.requires_grad], lr=.15)
        first = float(repair.repair_loss(network, summary, items, {}, False)['total'].detach())
        for _ in range(120):
            repair.fit_step(network, optimizer, summary, items, {}, False)
        result = repair.repair_loss(network, summary, items, {}, False)
        self.assertLess(float(result['total'].detach()), first * .05)
        self.assertTrue(torch.equal(result['prediction']['steps'].argmax(-1), result['target']['steps']))
        self.assertTrue(torch.equal(result['prediction']['lives'].argmax(-1), result['target']['lives']))
        self.assertTrue(torch.equal(result['prediction']['events'] >= 0, result['target']['events'].bool()))

    def test_checkpoint_retains_inference_contract_and_replaces_stale_training_metadata(self):
        network = model()
        parent = dict(planner_weights=copy.deepcopy(network.state_dict()), planner_config=network.config(),
                      encoder_weights={'test': torch.tensor(1)}, score_weights={'planner': 1., 'direct': 0.},
                      format='unchanged', encoder_frozen=True, learned_voluntary_reset=False,
                      persistent_game_memory=False, quality_filter={'required_optimal_nonzero': True},
                      supplemental_cache='old-supplement', objective_weights={'old': True},
                      actual_outcome_comparator_auxiliary_training=True)
        args = SimpleNamespace(checkpoint_sha256='parent', steps=5, lr=.001, seed=42, cache=Path('base'))
        report = dict(initial_planner_weights_sha256='initial', source_sha256={'source': 'hash'},
                      train_levels=10000, cache_manifest_sha256='cache', sampling='all original roots',
                      cohort_weights=self.weights(targets()), trainable_names=repair.freeze_except_repair_heads(network),
                      frozen_weights_sha256=repair.frozen_digest(network))
        output = repair.checkpoint_for(parent, network, 'balanced', args, report)
        self.assertNotIn('quality_filter', output)
        self.assertNotIn('supplemental_cache', output)
        self.assertEqual(output['parent_training_metadata']['quality_filter'], parent['quality_filter'])
        self.assertEqual(output['objective_weights']['policy'], 0)
        self.assertEqual(output['objective_weights']['value'], 0)
        self.assertTrue(output['event_repair_continuation']['zero_optimal_roots_retained'])
        self.assertFalse(output['event_repair_continuation']['gameplay_gain_established'])
        self.assertFalse(output['actual_outcome_comparator_auxiliary_training'])
        for key in ('planner_config', 'score_weights', 'format', 'encoder_frozen', 'learned_voluntary_reset', 'persistent_game_memory'):
            self.assertEqual(output[key], parent[key])
        self.assertIn('quality_filter', parent)

    def test_evaluation_counts_outcomes_independently_of_optimal_mask(self):
        items = targets()
        target = repair.repair_targets(items)
        perfect = dict(steps=F.one_hot(target['steps'], 44).float() * 40 - 20,
                       lives=F.one_hot(target['lives'], 4).float() * 40 - 20,
                       events=target['events'] * 40 - 20)
        arrays = {key: value.numpy() for key, value in items.items()}
        with patch.object(repair, 'head_predictions', return_value=perfect), patch.object(repair, 'guard', return_value=10 * 2**30):
            report = repair.evaluate_heads(model(), arrays, np.zeros((2, 4, 8), dtype=np.float32), device='cpu')
        self.assertEqual(report['branches'], 8)
        self.assertEqual(report['cohorts']['life_loss'], dict(correct=1, support=1, rate=1.))
        self.assertEqual(report['cohorts']['refill_increase'], dict(correct=1, support=1, rate=1.))
        self.assertEqual(report['cohorts']['lives'], dict(correct=8, support=8, rate=1.))
        self.assertEqual(report['contradictions_at_zero_logit'], dict(life_loss_without_lives_decrease=0, won_without_terminal=0))

    def test_wrong_checkpoint_pin_fails_before_model_or_data_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'parent.pt'
            checkpoint.write_bytes(b'not the pinned checkpoint')
            out = root / 'out'
            with patch.object(repair, 'guard', return_value=10 * 2**30), patch.object(repair, 'gpu_available'), \
                    patch.object(torch.cuda, 'is_available', return_value=True), patch.object(repair, 'load_data') as load:
                with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                    repair.main(['--checkpoint', str(checkpoint), '--checkpoint-sha256', '0' * 64, '--out', str(out)])
            load.assert_not_called()
            self.assertEqual(json.loads((out / 'report.json').read_text())['status'], 'failed')
            self.assertFalse(list(out.glob('*.pt')))


if __name__ == '__main__':
    unittest.main()
