"""Small CPU value-head/objective checks; no shipped levels or GPU execution."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from pebby.agent.spatial_outcome_planner import SpatialOutcomePlanner
from pebby.agent.spatial_value_objective import finite_cdf_mse, spatial_value_loss, value_targets
from tools import train_spatial_event_repair as shared
from tools import train_spatial_value_repair as repair


def model():
    torch.manual_seed(42)
    return SpatialOutcomePlanner(channels=4, width=8, hud_width=8, summary=8, comparator_hidden=8).eval()


class SpatialValueRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        assert not torch.cuda.is_initialized()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        assert not torch.cuda.is_initialized()

    def test_full_ce_includes_unreachable_and_zero_optimal_roots(self):
        logits = torch.randn(2, 4, 130, requires_grad=True)
        distances = torch.tensor([[2, -1, 4, 8], [10, 15, -2, 3]])
        # The loss accepts distances only: optimal masks cannot filter value CE.
        arrays = dict(distances=distances.numpy(), optimal=np.zeros(2, dtype=np.uint8))
        _, selected = repair.selected_items(arrays, np.zeros((2, 4, 8), dtype=np.float32), np.arange(2), 'cpu')
        result = spatial_value_loss(logits, selected)
        expected = F.cross_entropy(logits.flatten(0, 1), torch.tensor([2, 129, 4, 8, 10, 15, 129, 3]))
        torch.testing.assert_close(result['total'], expected)
        result['total'].backward()
        self.assertTrue((logits.grad.abs().sum(-1) > 0).all())
        self.assertGreater(logits.grad[0, 1, :129].sum().item(), 0)
        self.assertLess(logits.grad[0, 1, 129].item(), 0)

    def test_finite_conditional_distribution_ignores_unreachable_mass(self):
        logits = torch.randn(1, 4, 130, requires_grad=True)
        distances = torch.tensor([[5, -1, 20, -2]])
        first = finite_cdf_mse(logits, distances)
        changed = logits.detach().clone()
        changed[..., 129] += 100
        torch.testing.assert_close(first, finite_cdf_mse(changed, distances), atol=0, rtol=0)
        first.backward()
        torch.testing.assert_close(logits.grad[..., 129], torch.zeros(1, 4))
        torch.testing.assert_close(logits.grad[:, [1, 3]], torch.zeros(1, 2, 130))
        self.assertGreater(float(logits.grad[:, [0, 2], :129].abs().sum()), 0)

    def test_cdf_normalization_and_farther_wrong_distance_penalty(self):
        def sharp(distance):
            logits = torch.full((1, 4, 130), -50.)
            logits[..., distance] = 50.
            return logits
        target = torch.full((1, 4), 10, dtype=torch.long)
        close = finite_cdf_mse(sharp(12), target)
        far = finite_cdf_mse(sharp(18), target)
        torch.testing.assert_close(close, torch.tensor(2 / 128))
        torch.testing.assert_close(far, torch.tensor(8 / 128))
        self.assertLess(float(close), float(far))
        self.assertLess(float(finite_cdf_mse(sharp(10), target)), 1e-12)

    def test_reachable_denominator_does_not_include_unreachable_rows(self):
        logits = torch.randn(2, 4, 130)
        distances = torch.tensor([[1, 3, 5, 7], [-1, -1, -1, -1]])
        torch.testing.assert_close(finite_cdf_mse(logits, distances), finite_cdf_mse(logits[:1], distances[:1]))

    def test_all_unreachable_ordinal_loss_has_finite_graph_connected_zero(self):
        logits = torch.randn(2, 4, 130, requires_grad=True)
        distances = torch.full((2, 4), -1, dtype=torch.long)
        ordinal = finite_cdf_mse(logits, distances)
        self.assertEqual(ordinal.item(), 0)
        ordinal.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
        self.assertGreater(float(spatial_value_loss(logits, distances, ordinal_weight=10)['total'].detach()), 0)

    def test_reachable_129_is_overflow_and_not_unreachable(self):
        logits = torch.zeros(1, 4, 130)
        distances = torch.tensor([[129, 1000, 128, -1]])
        self.assertEqual(value_targets(logits, distances).tolist(), [[128, 128, 128, 129]])
        torch.testing.assert_close(finite_cdf_mse(logits, distances), finite_cdf_mse(logits, torch.tensor([[128, 128, 128, -1]])))
        target_zero = torch.zeros(1, 4, dtype=torch.long)
        expected = (torch.arange(1, 129).float() / 129 - 1).square().mean()
        torch.testing.assert_close(finite_cdf_mse(logits, target_zero), expected)

    def test_ordinal_arm_changes_only_added_loss_and_preserves_action_permutation(self):
        logits = torch.randn(2, 4, 130)
        distance = torch.tensor([[2, 5, -1, 3], [0, 1, 5, 20]])
        control = spatial_value_loss(logits, distance)
        ordinal = spatial_value_loss(logits, distance, ordinal_weight=10)
        torch.testing.assert_close(ordinal['cross_entropy'], control['cross_entropy'], atol=0, rtol=0)
        torch.testing.assert_close(ordinal['total'], control['total'] + 10 * control['cdf_mse'])
        permutation = torch.tensor([3, 1, 0, 2])
        permuted = spatial_value_loss(logits[:, permutation], distance[:, permutation], ordinal_weight=10)
        for key in ordinal:
            torch.testing.assert_close(ordinal[key], permuted[key])

    def test_contract_failures(self):
        logits, distances = torch.zeros(1, 4, 130), torch.zeros(1, 4, dtype=torch.long)
        for scores, target in [(logits[..., :129], distances), (logits, distances.float()),
                               (logits, distances[:, :3]), (logits.long(), distances),
                               (torch.full_like(logits, float('nan')), distances)]:
            with self.assertRaises(ValueError):
                spatial_value_loss(scores, target)
        for weight in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                spatial_value_loss(logits, distances, ordinal_weight=weight)

    def test_shared_sampler_keeps_all_roots_and_matched_distinct_levels(self):
        seeds = np.repeat(np.arange(1100), 8)
        sampler = repair.LevelSampler(seeds, np.arange(2000, 2005))
        left, right = np.random.default_rng(42), np.random.default_rng(42)
        visited = set()
        for _ in range(10):
            rows = sampler.sample(left)
            np.testing.assert_array_equal(rows, sampler.sample(right))
            self.assertEqual(len(np.unique(seeds[rows])), 1024)
            visited.update((rows % 8).tolist())
        self.assertEqual(visited, set(range(8)))

    def test_cached_summary_value_equivalence_and_frozen_native_outputs_after_step(self):
        network = model()
        items = dict(raw=torch.randn(2, 160, 4), state=torch.randn(2, 160, 4), glyph=torch.randn(2, 14),
                     distances=torch.tensor([[1, 2, -1, 4], [5, 6, 8, -1]]), optimal=torch.zeros(2, dtype=torch.uint8))
        player = {'player_head.weight': torch.randn(1, 4), 'player_head.bias': torch.randn(1)}
        with torch.inference_mode():
            before = network(items['raw'], items['state'], items['glyph'], repair.player_probabilities(items, player))
        arrays = {key: value.numpy() for key, value in items.items()}
        arrays['seeds'] = np.array([1, 2])
        with tempfile.TemporaryDirectory() as directory, patch.object(shared, 'guard', return_value=10 * 2**30):
            cached = repair.cache_summaries(network, arrays, player, Path(directory) / 'summary.npy', device='cpu', size=2)
            summary = torch.from_numpy(np.array(cached))
            torch.testing.assert_close(network.value_head(summary), before['value_logits'], atol=0, rtol=0)
            repair.release({'summary': cached}, close=True)
        self.assertFalse(network.summary_head._forward_hooks)
        names = repair.freeze_except_value_head(network)
        self.assertEqual(set(names), {'value_head.weight', 'value_head.bias'})
        frozen = repair.frozen_digest(network)
        record, _ = repair.fit_step(network, repair.optimizer_for(network, .01), summary, items['distances'], 10.)
        self.assertTrue(torch.isfinite(record['total']))
        self.assertEqual(repair.frozen_digest(network), frozen)
        for name, parameter in network.named_parameters():
            if name in names:
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0)
            else:
                self.assertIsNone(parameter.grad)
        check = repair.verify_frozen_predictions(network, items, player, before)
        self.assertTrue(check['physical_and_event_predictions_bitwise_unchanged'])
        self.assertTrue(check['fixed_input_comparator_bitwise_unchanged'])
        self.assertTrue(check['native_value_logits_changed'])

    def test_checkpoint_replaces_latest_stage_metadata_and_keeps_inference_format(self):
        network = model()
        parent = dict(planner_weights=copy.deepcopy(network.state_dict()), planner_config=network.config(),
                      encoder_weights={'test': torch.tensor(1)}, score_weights={'planner': 1., 'direct': 0.},
                      format='unchanged', encoder_frozen=True, learned_voluntary_reset=False,
                      persistent_game_memory=False, quality_filter={'required_optimal_nonzero': True},
                      supplemental_cache='old', event_repair_continuation={'old': True}, objective_weights={'old': True})
        args = SimpleNamespace(checkpoint_sha256='parent', steps=5, lr=.001, seed=42, cache=Path('base'))
        report = dict(initial_planner_weights_sha256='initial', source_sha256={'source': 'hash'},
                      train_levels=10000, cache_manifest_sha256='cache', sampling='all original roots',
                      trainable_names=repair.freeze_except_value_head(network), frozen_weights_sha256=repair.frozen_digest(network))
        output = repair.checkpoint_for(parent, network, 'ordinal', args, report)
        for key in ('quality_filter', 'supplemental_cache', 'event_repair_continuation'):
            self.assertNotIn(key, output)
            self.assertEqual(output['parent_training_metadata'][key], parent[key])
        self.assertEqual(output['objective_weights']['finite_conditional_cdf_mse'], 10)
        self.assertEqual(output['objective_weights']['policy'], 0)
        self.assertEqual(output['optimizer_steps'], 5)
        self.assertTrue(output['value_repair_continuation']['zero_optimal_roots_retained'])
        self.assertFalse(output['value_repair_continuation']['gameplay_gain_established'])
        for key in ('planner_config', 'score_weights', 'format', 'encoder_frozen', 'learned_voluntary_reset', 'persistent_game_memory'):
            self.assertEqual(output[key], parent[key])


if __name__ == '__main__':
    unittest.main()
