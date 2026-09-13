import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import unittest
import torch
from pebby.agent.neural_outcome_planner import neural_outcome_losses
from pebby.agent.spatial_outcome_planner import SpatialOutcomePlanner, SpatialOutcomePlannerConfig
from pebby.agent.world_grounding import SIZES

torch.set_num_threads(1)


class SpatialOutcomePlannerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        self.model = SpatialOutcomePlanner(channels=8, width=8, hud_width=8, summary=16, comparator_hidden=16)
        self.raw = torch.randn(2, 160, 8, requires_grad=True)
        self.state = torch.randn(2, 160, 8, requires_grad=True)
        self.glyph = torch.randn(2, 14, requires_grad=True)
        self.player_source = torch.randn(2, 144, requires_grad=True)
        self.player = self.player_source.softmax(-1)

    def forward_model(self, **kwargs):
        return self.model(self.raw, self.state, self.glyph, self.player, **kwargs)

    def test_typed_contract_config_and_distinct_blocks(self):
        output = self.forward_model()
        self.assertEqual(output['action_logits'].shape, (2, 4))
        self.assertEqual([x.shape for x in output['field_logits']], [(2, 4, n) for n in SIZES])
        self.assertEqual(output['value_logits'].shape, (2, 4, 130))
        self.assertEqual(output['event_logits'].shape, (2, 4, 3))
        self.assertEqual(SpatialOutcomePlanner(self.model.config()).config(), self.model.config())
        self.assertEqual([b.conv1.dilation for b in self.model.blocks], [(1, 1), (2, 2), (4, 4)])
        self.assertEqual(len({id(b.conv1.weight) for b in self.model.blocks}), 3)
        self.assertEqual(len({id(b.condition.weight) for b in self.model.blocks}), 3)
        default = SpatialOutcomePlanner()
        self.assertLessEqual(default.parameter_count(), 300000)
        self.assertGreaterEqual(default.parameter_count(), 100000)
        self.assertFalse(torch.cuda.is_initialized())

    def test_policy_gradient_reaches_all_outcomes_dynamics_and_public_inputs(self):
        output = self.forward_model()
        all_logits = [*output['field_logits'], output['value_logits'], output['event_logits']]
        for tensor in all_logits: tensor.retain_grad()
        torch.nn.functional.cross_entropy(output['action_logits'], torch.tensor([0, 3])).backward()
        for tensor in all_logits:
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(float(tensor.grad.abs().sum()), 0.)
        for tensor in (self.raw, self.state, self.glyph, self.player_source):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(float(tensor.grad.abs().sum()), 0.)
        for tensor in (self.raw, self.state):
            self.assertGreater(float(tensor.grad[:, :144].abs().sum()), 0.)
            self.assertGreater(float(tensor.grad[:, 144:].abs().sum()), 0.)
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0., name)

    def test_explicit_action_rows_permute_all_branches(self):
        original = self.forward_model()
        order = torch.tensor([2, 0, 3, 1])
        permuted = self.forward_model(action_rows=torch.eye(4)[order])
        for before, after in zip(original['field_logits'], permuted['field_logits']):
            torch.testing.assert_close(after, before[:, order], atol=2e-6, rtol=2e-5)
        for name in ('value_logits', 'event_logits', 'action_logits'):
            torch.testing.assert_close(permuted[name], original[name][:, order], atol=2e-6, rtol=2e-5)
        # Conditioned branches must have an actual numerical path from action identity.
        self.assertGreater(float((original['field_logits'][0][:, 0]-original['field_logits'][0][:, 1]).detach().abs().max()), 1e-6)
        actions = torch.eye(4).requires_grad_(True)
        self.forward_model(action_rows=actions)['field_logits'][0].square().mean().backward()
        self.assertGreater(float(actions.grad.abs().sum()), 0.)

    def test_comparator_no_bypass_and_permutation(self):
        output = self.forward_model()
        reconstructed = self.model.score_outcomes(output['field_logits'], output['value_logits'], output['event_logits'])
        torch.testing.assert_close(reconstructed, output['action_logits'], rtol=0, atol=0)
        order = torch.tensor([3, 0, 2, 1])
        shuffled = self.model.score_outcomes(tuple(x[:, order] for x in output['field_logits']),
                                             output['value_logits'][:, order], output['event_logits'][:, order])
        torch.testing.assert_close(shuffled, reconstructed[:, order], atol=1e-6, rtol=1e-5)
        identical = [x[:, :1].expand_as(x) for x in output['field_logits']]
        same = self.model.score_outcomes(identical, output['value_logits'][:, :1].expand_as(output['value_logits']),
                                        output['event_logits'][:, :1].expand_as(output['event_logits']))
        torch.testing.assert_close(same, same[:, :1].expand_as(same), rtol=1e-6, atol=1e-7)
        # Changing spatial weights cannot alter scores for a fixed supplied outcome.
        with torch.no_grad(): self.model.context_projection[0].weight.add_(2)
        again = self.model.score_outcomes(output['field_logits'], output['value_logits'], output['event_logits'])
        torch.testing.assert_close(again, reconstructed, rtol=0, atol=0)

    def test_existing_loss_compatibility_and_reachable_reset(self):
        targets = dict(next_player_cell=torch.zeros(2, 4, 2, dtype=torch.long),
                       next_triple=torch.zeros(2, 4, 3, dtype=torch.long), next_steps=torch.zeros(2, 4, dtype=torch.long),
                       next_lives=torch.ones(2, 4, dtype=torch.long), distances=torch.full((2, 4), 7, dtype=torch.long),
                       lost_life=torch.ones(2, 4, dtype=torch.bool), terminal=torch.zeros(2, 4, dtype=torch.bool),
                       won=torch.zeros(2, 4, dtype=torch.bool), optimal=torch.tensor([0, 3]))
        output = self.forward_model()
        record = neural_outcome_losses(output, targets)
        self.assertTrue(torch.isfinite(record['total']))
        self.assertEqual(float(record['diagnostics']['policy_valid_count']), 1.)
        expected = torch.nn.functional.cross_entropy(output['value_logits'].flatten(0, 1), torch.full((8,), 7))
        torch.testing.assert_close(record['losses']['value'], expected)

    def test_invalid_public_probability_or_action_contract(self):
        with self.assertRaises(ValueError):
            self.model(self.raw, self.state, self.glyph, torch.ones(2, 144))
        with self.assertRaises(ValueError):
            self.forward_model(action_rows=torch.eye(4)[torch.tensor([0, 0, 2, 3])])
        with self.assertRaises(ValueError):
            SpatialOutcomePlannerConfig(width=0)


if __name__ == '__main__': unittest.main()
