"""CPU semantic checks for generated-only spatial outcome supervision."""
import math
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import unittest
from unittest.mock import patch
import torch

from pebby.agent.neural_outcome_planner import EVENT_NAMES, neural_outcome_losses
from pebby.agent.spatial_outcome_objective import actual_outcomes, spatial_outcome_losses, targets, training_weights
from pebby.agent.spatial_outcome_planner import SpatialOutcomePlanner
from pebby.agent.world_grounding import SIZES
from pebby.agent.outcome_ordering import pairwise_safe_ordering_loss
from tools.train_reference_spatial_outcomes import forward



def fixture():
    torch.manual_seed(42)
    model = SpatialOutcomePlanner(channels=4, width=8, hud_width=8, summary=8, comparator_hidden=8)
    items = dict(raw=torch.randn(2, 160, 4), state=torch.randn(2, 160, 4), glyph=torch.randn(2, 14),
                 player_cell=torch.tensor([[2, 3], [4, 5]]), current_triple=torch.zeros(2, 3, dtype=torch.long),
                 current_steps=torch.tensor([8, 9]), current_lives=torch.tensor([3, 3]),
                 next_player_cell=torch.tensor([[[2, 2], [3, 3], [2, 4], [1, 3]], [[4, 4], [5, 5], [4, 6], [3, 5]]]),
                 next_triple=torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]] * 2),
                 next_steps=torch.tensor([[-3, -1, 0, 42], [8, 8, 8, 8]]), next_lives=torch.tensor([[2, 3, 3, 3]] * 2),
                 distances=torch.tensor([[7, -1, 0, 150]] * 2), lost_life=torch.tensor([[1, 0, 0, 0]] * 2),
                 terminal=torch.tensor([[0, 1, 1, 0]] * 2), won=torch.tensor([[0, 0, 1, 0]] * 2), optimal=torch.tensor([4, 5]))
    weights = training_weights({k: v.numpy() for k, v in items.items()})
    return model, items, weights


def predictions(model, items):
    return model(items['raw'], items['state'], items['glyph'], torch.full((2, 144), 1 / 144))


class SpatialOutcomeObjectiveTests(unittest.TestCase):
    def setUp(self):
        assert not torch.cuda.is_initialized(), "run with CUDA_VISIBLE_DEVICES=''"
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)
        assert not torch.cuda.is_initialized()

    def test_training_weights_equal_mass_and_event_cap(self):
        n, branches = 16, 64
        changed_counts = (1, 3, 8)
        positive_counts = (1, 4, 16)
        arrays = dict(current_triple=np.zeros((n, 3), np.int8), next_triple=np.zeros((n, 4, 3), np.int8))
        for index, count in enumerate(changed_counts):
            arrays['next_triple'].reshape(-1, 3)[:count, index] = 1
        for name, count in zip(EVENT_NAMES, positive_counts):
            arrays[name] = np.zeros((n, 4), np.uint8)
            arrays[name].flat[:count] = 1
        weights = training_weights(arrays)
        for pair, count in zip(weights['glyph_change_weights'], changed_counts):
            assert np.isclose(pair[0] * (branches - count), branches / 2)
            assert np.isclose(pair[1] * count, branches / 2)
        # One positive among64 would yield63: cap50 applies; other ratios stay natural.
        np.testing.assert_allclose(weights['event_positive_weights'], [50., 15., 3.])


    def test_steps_and_reachable_reset_semantics(self):
        _, items, _ = fixture()
        fields, value, events = actual_outcomes(items)
        assert fields[4].argmax(-1)[0].tolist() == [0, 0, 1, 43]
        assert value.argmax(-1)[0].tolist() == [7, 129, 0, 128]
        assert events[0, 0, 0] == 20  # A life-loss successor can still be reachable.
        assert fields[0].argmax(-1)[0, 1] == 3 * 12 + 3
        assert set(torch.unique(value).tolist()) == {-20., 20.}


    def test_empty_changed_strata_are_none_not_zero(self):
        model, items, weights = fixture()
        for future, current in [('next_player_cell', 'player_cell'), ('next_triple', 'current_triple'),
                                ('next_steps', 'current_steps'), ('next_lives', 'current_lives')]:
            items[future] = items[current][:, None].expand_as(items[future]).clone()
        items['optimal'].zero_()
        record = spatial_outcome_losses(model, predictions(model, items), items, weights)
        for name in ('player', 'shape', 'color', 'rotation', 'steps', 'lives'):
            for suffix in ('changed', 'changed_no_life_loss'):
                key = f'{name}_{suffix}_accuracy'
                assert record['diagnostics'][key] is None
                assert record['diagnostic_weights'][key] == 0
        assert record['diagnostics']['teacher_set_accuracy'] is None
        assert record['losses']['teacher_policy'] == 0 and torch.isfinite(record['total'])


    def test_teacher_policy_gradient_reaches_only_comparator(self):
        model, items, weights = fixture()
        record = spatial_outcome_losses(model, predictions(model, items), items, weights)
        record['losses']['teacher_policy'].backward()
        groups = {'outcome_projection': 0., 'comparator': 0.}
        for name, parameter in model.named_parameters():
            prefix = name.split('.')[0]
            if prefix in groups:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                groups[prefix] += float(parameter.grad.square().sum())
            else:
                assert parameter.grad is None, name
        assert all(norm > 0 for norm in groups.values())


    def test_trainer_public_predictions_unchanged_when_targets_are_perturbed(self):
        model, items, weights = fixture(); model.eval()
        player_weights = {'player_head.weight': torch.randn(1, 4), 'player_head.bias': torch.randn(1)}
        captures = []
        handle = model.register_forward_hook(lambda module, args, output: captures.append(output))
        try:
            first = forward(model, items, player_weights, weights, 'float32')
            altered = {k: v.clone() for k, v in items.items()}
            for key in ('next_player_cell', 'next_triple', 'next_steps', 'next_lives', 'distances', 'lost_life', 'terminal', 'won', 'optimal'):
                altered[key].zero_()
            altered['player_cell'].zero_(); altered['current_triple'].fill_(1)
            second = forward(model, altered, player_weights, weights, 'float32')
        finally:
            handle.remove()
        assert len(captures) == 2
        for key in ('action_logits', 'value_logits', 'event_logits'):
            torch.testing.assert_close(captures[0][key], captures[1][key], atol=0, rtol=0)
        for left, right in zip(captures[0]['field_logits'], captures[1]['field_logits']):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        assert first['losses']['teacher_policy'] > 0 and second['losses']['teacher_policy'] == 0


    def test_weighted_physical_and_event_losses_have_declared_scale(self):
        model, items, weights = fixture()
        uniform = dict(field_logits=tuple(torch.zeros(2, 4, size) for size in SIZES),
                       value_logits=torch.zeros(2, 4, 130), event_logits=torch.zeros(2, 4, 3), action_logits=torch.zeros(2, 4))
        record = spatial_outcome_losses(model, uniform, items, weights)
        # Equal TRAIN changed/unchanged mass normalizes each glyph's expected weight to one.
        expected = sum(math.log(size) * weight for size, weight in zip(SIZES, [1, 1/3, 1/3, 1/3, .5, .5]))
        assert np.isclose(float(record['losses']['physical']), expected, rtol=1e-6)
        expected_events = sum((float(items[name].sum()) * weight + 8 - float(items[name].sum())) * math.log(2)
                              for name, weight in zip(EVENT_NAMES, weights['event_positive_weights'])) / 24
        assert np.isclose(float(record['losses']['events']), expected_events, rtol=1e-6)

    def test_distance_ordering_zero_preserves_original_losses_exactly(self):
        model, items, weights = fixture()
        predicted = predictions(model, items)
        default = spatial_outcome_losses(model, predicted, items, weights)
        zero = spatial_outcome_losses(model, predicted, items, weights, distance_ordering=0.)
        self.assertEqual(set(default['losses']), set(zero['losses']))
        self.assertNotIn('distance_ordering', zero['losses'])
        for name in default['losses']:
            torch.testing.assert_close(default['losses'][name], zero['losses'][name], atol=0, rtol=0)
        torch.testing.assert_close(default['total'], zero['total'], atol=0, rtol=0)

    def test_ordering_uses_safe_siblings_and_reaches_predictor_not_comparator(self):
        model, items, weights = fixture()
        items['optimal'] = torch.tensor([4, 1])
        items['distances'][1] = torch.tensor([2, 3, 4, 5])
        items['lost_life'][1].zero_()
        predicted = predictions(model, items)
        record = spatial_outcome_losses(model, predicted, items, weights, distance_ordering=.25)
        expected_distance = (predicted['value_logits'].softmax(-1) * torch.arange(130)).sum(-1)
        expected_loss = .25 * pairwise_safe_ordering_loss(-expected_distance, items['distances'],
                                                         items['lost_life'], items['optimal'])
        torch.testing.assert_close(record['losses']['distance_ordering'], expected_loss, atol=0, rtol=0)
        record['losses']['distance_ordering'].backward()
        for parameter in (model.value_head.weight, model.context_projection[0].weight,
                          model.blocks[0].conv1.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in model.comparator.parameters()))
        metrics = record['diagnostics']
        self.assertEqual(metrics['distance_all_supervised_roots_count'], 2)
        self.assertEqual(metrics['distance_all_optimal_boundary_pairs_count'], 6)
        self.assertEqual(metrics['distance_all_safe_supervised_roots_count'], 1)
        self.assertEqual(metrics['distance_safe_gap1_supervised_roots_count'], 1)

    def test_distance_ordering_no_pairs_is_graph_connected_zero(self):
        model, items, weights = fixture()
        items['optimal'].zero_()
        record = spatial_outcome_losses(model, predictions(model, items), items, weights, distance_ordering=1.)
        self.assertEqual(record['losses']['distance_ordering'], 0)
        record['losses']['distance_ordering'].backward()
        self.assertIsNotNone(model.value_head.weight.grad)
        self.assertEqual(float(model.value_head.weight.grad.abs().sum()), 0)
        self.assertIsNone(record['diagnostics']['distance_all_optimal_boundary_accuracy'])

    def test_invalid_ordering_weights_fail_before_forward_or_gpu(self):
        model, items, weights = fixture()
        for value in (-1., float('inf'), float('nan'), True, 'bad'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
                spatial_outcome_losses(model, {}, items, weights, distance_ordering=value)
        from tools import train_reference_spatial_outcomes, train_spatial_recovery_comparison
        for trainer in (train_reference_spatial_outcomes, train_spatial_recovery_comparison):
            required = ['--out-dir', '/unused', '--qualify']
            if trainer is train_spatial_recovery_comparison:
                required += ['--supplement', '/unused', '--quality-manifest', '/unused']
            with patch.object(trainer, 'gpu_available', side_effect=AssertionError('GPU must not be consulted')):
                for value in ('-1', 'nan', 'inf'):
                    with self.subTest(trainer=trainer.__name__, value=value), self.assertRaises(SystemExit) as raised:
                        trainer.main(required + ['--distance-ordering=' + value])
                    self.assertEqual(raised.exception.code, 2)

    def test_skipping_replaced_losses_preserves_spatial_total_and_gradients(self):
        model, items, weights = fixture()
        original = copy.deepcopy(model)
        # Reproduce the prior wrapper, which computed both base families before
        # replacing them with the balanced spatial versions.
        with patch('pebby.agent.spatial_outcome_objective.neural_outcome_losses',
                   side_effect=lambda predicted, batch, **kwargs: neural_outcome_losses(predicted, batch)):
            before = spatial_outcome_losses(original, predictions(original, items), items, weights)
        after = spatial_outcome_losses(model, predictions(model, items), items, weights)
        torch.testing.assert_close(before['total'], after['total'], atol=0, rtol=0)
        self.assertEqual(list(before['losses']), list(after['losses']))
        before['total'].backward(); after['total'].backward()
        for left, right in zip(original.parameters(), model.parameters()):
            torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)

    def test_skipping_base_families_avoids_computation_but_keeps_validation(self):
        from torch.nn import functional as F
        model, items, _ = fixture()
        predicted = predictions(model, items)
        baseline = neural_outcome_losses(predicted, items)
        with patch.object(F, 'cross_entropy', wraps=F.cross_entropy) as ce, \
             patch.object(F, 'binary_cross_entropy_with_logits', wraps=F.binary_cross_entropy_with_logits) as bce:
            skipped = neural_outcome_losses(predicted, items, skip_losses=('physical', 'events'))
        self.assertEqual(ce.call_count, 1)
        self.assertEqual(bce.call_count, 0)
        self.assertEqual(set(skipped['losses']), {'value', 'policy'})
        for key, value in baseline['diagnostics'].items():
            if value is None:
                self.assertIsNone(skipped['diagnostics'][key])
            else:
                torch.testing.assert_close(value, skipped['diagnostics'][key], atol=0, rtol=0)
        for key in ('next_lives', 'won'):
            bad = {**items, key: torch.full_like(items[key], 99)}
            with self.assertRaises(ValueError):
                neural_outcome_losses(predicted, bad, skip_losses=('physical', 'events'))
        with self.assertRaisesRegex(ValueError, 'skipped loss family'):
            neural_outcome_losses(predicted, items, skip_losses=('unknown',))

    def test_reference_qualification_binds_ordering_before_gpu_work(self):
        from tools import train_reference_spatial_outcomes as trainer
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'qualification.json'
            qualification = dict(status='complete', qualification_passed=True, selected_batch_size=1024,
                                 precision='float32', learning_rate=.001, cache_manifest_sha256='cache',
                                 objective_weights={}, distance_ordering=.25)
            args = SimpleNamespace(qualification=path, batch_size=1024, precision='float32', lr=.001,
                                   distance_ordering=.5)
            for bound in (.25, None):
                if bound is None:
                    qualification.pop('distance_ordering')
                path.write_text(json.dumps(qualification))
                with self.assertRaisesRegex(ValueError, 'differs from completed qualification'):
                    trainer.train(args, {}, None, {}, {}, {}, {'cache_manifest_sha256': 'cache'})


if __name__ == '__main__':
    unittest.main()


class OrderingPairMetricsTests(unittest.TestCase):
    def test_bad_pair_changes_are_separate_from_optimal_boundary(self):
        from pebby.agent.spatial_outcome_objective import _ordering_diagnostics
        items = dict(distances=torch.tensor([[0, 1, 2, 3]]), lost_life=torch.zeros(1, 4), optimal=torch.tensor([1]))
        records = []
        for scores in (torch.tensor([[4., 3., 2., 1.]]), torch.tensor([[4., 1., 2., 3.]])):
            record = dict(diagnostics={}, diagnostic_weights={})
            _ordering_diagnostics(record, {'action_logits': scores}, items, -scores)
            records.append(record['diagnostics'])
        self.assertEqual(int(records[0]['comparator_safe_pair_count']), 6)
        self.assertEqual(int(records[0]['comparator_safe_gap1_pair_count']), 3)
        self.assertEqual(int(records[0]['comparator_safe_gap1_pair_correct_count']), 3)
        self.assertEqual(int(records[1]['comparator_safe_gap1_pair_correct_count']), 1)
        for r in records:
            self.assertEqual(int(r['comparator_all_optimal_boundary_correct_count']), 3)
