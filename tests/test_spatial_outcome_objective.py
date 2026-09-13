"""CPU semantic checks for generated-only spatial outcome supervision."""
import math

import numpy as np
import unittest
import torch

from pebby.agent.neural_outcome_planner import EVENT_NAMES
from pebby.agent.spatial_outcome_objective import actual_outcomes, spatial_outcome_losses, targets, training_weights
from pebby.agent.spatial_outcome_planner import SpatialOutcomePlanner
from pebby.agent.world_grounding import SIZES
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


if __name__ == '__main__':
    unittest.main()
