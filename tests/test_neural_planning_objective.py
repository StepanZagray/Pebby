import unittest

import torch

from pebby.agent.neural_planning_objective import planning_sequence_loss
from pebby.agent.structured_local_glyph import LocalGlobalGlyphTransition


class ZeroFieldDynamics(torch.nn.Module):
    """A fixed predicted board isolates the loss's region weighting."""
    def __init__(self):
        super().__init__()
        self.field = torch.nn.Parameter(torch.zeros(148, 96))

    def forward(self, field, actions):
        size = len(field)
        predicted = self.field[None].expand(size, -1, -1)
        readout = {key: torch.zeros(size, width) for key, width in
                   [('player_logits', 144), ('steps_logits', 44), ('lives_logits', 4),
                    ('carried_shape_logits', 6), ('carried_color_logits', 4), ('carried_rotation_logits', 4)]}
        return dict(field=predicted, readout=readout,
                    glyph_logits={key: torch.zeros(size, width) for key, width in [('shape', 6), ('color', 4), ('rotation', 4)]},
                    events={key + '_logits': torch.zeros(size) for key in ('lost_life', 'terminal', 'won')})


def appearance_case(future, observed=None):
    batch, horizon = future.shape[:2]
    valid = torch.ones(batch, horizon, dtype=torch.bool)
    labels = dict(next_player_cell=torch.zeros(batch, horizon, 2, dtype=torch.long),
                  next_triple=torch.zeros(batch, horizon, 3, dtype=torch.long),
                  next_steps=torch.zeros(batch, horizon, dtype=torch.long),
                  next_lives=torch.zeros(batch, horizon, dtype=torch.long),
                  **{key: torch.zeros(batch, horizon, dtype=torch.bool) for key in ('lost_life', 'terminal', 'won')})
    model = ZeroFieldDynamics()
    result = planning_sequence_loss(model, torch.zeros(batch, 148, 96), torch.zeros(batch, horizon, dtype=torch.long),
                                    future, labels, valid, valid if observed is None else observed)
    return model, result


class PlanningObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(13)
        self.model = LocalGlobalGlyphTransition(loops=1)
        self.fields = torch.rand(2, 148, 96)
        self.actions = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
        self.future = torch.rand(2, 4, 148, 96)
        self.valid = torch.tensor([[True, True, False, False], [True] * 4])
        self.observed = self.valid.clone()
        self.observed[0, 1] = False
        self.labels = dict(next_player_cell=torch.zeros(2, 4, 2, dtype=torch.long),
                           next_triple=torch.zeros(2, 4, 3, dtype=torch.long),
                           next_steps=torch.full((2, 4), 20, dtype=torch.long),
                           next_lives=torch.ones(2, 4, dtype=torch.long),
                           lost_life=torch.zeros(2, 4, dtype=torch.bool),
                           terminal=torch.zeros(2, 4, dtype=torch.bool),
                           won=torch.zeros(2, 4, dtype=torch.bool))
        self.labels['terminal'][0, 1] = True
        self.labels['lost_life'][:, 1] = True

    def loss(self):
        return planning_sequence_loss(self.model, self.fields, self.actions, self.future,
                                      self.labels, self.valid, self.observed)

    def test_reset_transition_kept_terminal_tail_excluded(self):
        result = self.loss()
        self.assertEqual(result['transition_count'], 6)
        self.assertEqual(result['field_count'], 5)
        self.assertEqual(result['change_comparison_count'], 5)
        self.assertEqual(result['changed_transition_count'], 5)
        self.assertEqual(result['event_positive_counts']['lost_life'], 2)
        result['total'].backward()
        self.assertGreater(float(self.model.glyph_head[-1].weight.grad.abs().sum()), 0.)

    def test_unobserved_and_unexecuted_targets_cannot_change_loss(self):
        first = self.loss()['total'].detach()
        self.future[~self.observed] = 999
        for name in ('next_player_cell', 'next_triple', 'next_steps', 'next_lives'):
            self.labels[name][~self.valid] = 999
        torch.testing.assert_close(first, self.loss()['total'], rtol=0, atol=0)

    def test_each_future_uses_prediction_not_teacher_target(self):
        inputs, outputs = [], []
        before = self.model.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach().clone()))
        def retain(module, args, result):
            result['field'].retain_grad()
            outputs.append(result['field'])
        after = self.model.register_forward_hook(retain)
        try:
            result = self.loss()
            result['total'].backward()
        finally:
            before.remove()
            after.remove()
        for step in range(1, 4):
            torch.testing.assert_close(inputs[step], outputs[step - 1])
        self.assertGreater(float(outputs[0].grad.abs().sum()), 0.)

    def test_invalid_tail_and_resurrection_rejected(self):
        self.valid[0, 2] = True
        with self.assertRaises(ValueError):
            self.loss()
        self.valid[0, 2] = False
        self.observed[0, 3] = True
        with self.assertRaises(ValueError):
            self.loss()

    def test_small_changed_region_is_not_diluted_by_static_board(self):
        future = torch.zeros(1, 3, 148, 96)
        future[:, :, 0, 48:70] = 1.
        model, result = appearance_case(future)
        self.assertEqual(result['change_comparison_count'], 3)
        # Only the first transition changes; comparison against the original
        # root at every horizon would incorrectly count all three.
        self.assertEqual(result['changed_transition_count'], 1)
        self.assertEqual(result['changed_cell_count'], 1)
        torch.testing.assert_close(result['losses']['field_changed'], torch.tensor(1.))
        self.assertGreater(float(result['losses']['field_changed'].detach()), 100 * float(result['losses']['field'].detach()))
        result['losses']['field_changed'].backward()
        self.assertGreater(float(model.field.grad[0, 48:70].abs().sum()), 0.)
        self.assertEqual(float(model.field.grad[1:].abs().sum()), 0.)

    def test_each_changed_transition_has_equal_weight_across_region_sizes(self):
        future = torch.zeros(2, 1, 148, 96)
        future[0, 0, 0, 48:70] = 1.
        future[1, 0, :144, 48:70] = .5
        _, result = appearance_case(future)
        self.assertEqual(result['changed_cell_count'], 145)
        self.assertEqual(result['changed_transition_count'], 2)
        torch.testing.assert_close(result['losses']['field_changed'], torch.tensor((1. + .25) / 2))

    def test_missing_previous_field_cannot_create_a_change_target(self):
        future = torch.zeros(1, 3, 148, 96)
        future[0, 2, 0, 48:70] = 1.
        observed = torch.tensor([[True, False, True]])
        _, first = appearance_case(future, observed)
        future[0, 1] = 999.
        _, second = appearance_case(future, observed)
        torch.testing.assert_close(first['total'], second['total'], rtol=0, atol=0)
        self.assertEqual(second['change_comparison_count'], 1)
        self.assertEqual(second['changed_transition_count'], 0)
        self.assertEqual(float(second['losses']['field_changed'].detach()), 0.)
        second['losses']['field_changed'].backward()  # graph-connected zero

    def test_actual_change_targets_have_no_gradient_path(self):
        self.fields.requires_grad_(True)
        self.future.requires_grad_(True)
        self.loss()['losses']['field_changed'].backward()
        self.assertIsNone(self.fields.grad)
        self.assertIsNone(self.future.grad)
        self.assertGreater(float(self.model.block.local_conv.weight.grad.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
