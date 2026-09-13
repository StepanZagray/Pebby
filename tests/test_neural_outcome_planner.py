"""CPU contracts for the fixed-depth, one-step neural outcome planner."""
import unittest

import torch
from torch.nn import functional as F

from pebby.agent.neural_outcome_planner import (NeuralOutcomePlanner, NeuralOutcomePlannerConfig,
                                               SIZES, neural_outcome_losses)


def make_model():
    return NeuralOutcomePlanner(channels=16, heads=4, context_tokens=12, comparator_hidden=16).eval()


def targets(batch=2):
    return dict(next_player_cell=torch.zeros(batch, 4, 2, dtype=torch.long),
                next_triple=torch.zeros(batch, 4, 3, dtype=torch.long),
                next_steps=torch.full((batch, 4), 20, dtype=torch.long),
                next_lives=torch.full((batch, 4), 2, dtype=torch.long),
                distances=torch.arange(4)[None].expand(batch, -1).clone(),
                lost_life=torch.zeros(batch, 4, dtype=torch.bool),
                terminal=torch.zeros(batch, 4, dtype=torch.bool),
                won=torch.zeros(batch, 4, dtype=torch.bool),
                optimal=torch.ones(batch, dtype=torch.long))


class NeuralOutcomePlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(12)

    def test_shapes_fixed_depth_config_and_parameter_budget(self):
        model = NeuralOutcomePlanner()
        self.assertLessEqual(model.parameter_count(), 350_000)
        self.assertEqual(len(model.blocks), 2)
        self.assertIsNot(model.blocks[0], model.blocks[1])
        self.assertEqual(model.config()['horizon'], 1)
        self.assertEqual(NeuralOutcomePlanner(model.config()).config(), model.config())
        output = make_model()(torch.randn(2, 12, 16), torch.randn(2, 12, 16))
        self.assertEqual(output['action_logits'].shape, (2, 4))
        self.assertEqual([tuple(x.shape) for x in output['field_logits']], [(2, 4, n) for n in SIZES])
        self.assertEqual(output['value_logits'].shape, (2, 4, 130))
        self.assertEqual(output['event_logits'].shape, (2, 4, 3))
        with self.assertRaises(ValueError):
            NeuralOutcomePlannerConfig(channels=15, heads=4)
        with self.assertRaises(ValueError):
            NeuralOutcomePlanner({**model.config(), 'horizon': 2})

    def test_public_inputs_and_every_outcome_family_are_differentiably_used(self):
        model = make_model()
        raw = torch.randn(2, 12, 16, requires_grad=True)
        state = torch.randn(2, 12, 16, requires_grad=True)
        glyph = torch.randn(2, 14, requires_grad=True)
        result = model(raw, state, glyph)
        for tensor in (*result['field_logits'], result['value_logits'], result['event_logits']):
            tensor.retain_grad()
        neural_outcome_losses(result, targets(), {'physical': 0., 'value': 0., 'events': 0.})['total'].backward()
        for tensor in (raw, state, glyph, *result['field_logits'], result['value_logits'], result['event_logits']):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0.)
        for block in model.blocks:
            self.assertGreater(block.cross_attention.in_proj_weight.grad.abs().sum().item(), 0.)
        with torch.no_grad():
            changed = model(raw.flip(1) * 2, state, glyph)
            self.assertGreater((changed['action_logits']-result['action_logits']).abs().max().item(), 1e-9)

    def test_comparator_has_no_context_bypass_and_respects_branch_permutations(self):
        model = make_model()
        output = model(torch.randn(2, 12, 16), torch.randn(2, 12, 16), torch.randn(2, 14))
        fields, values, events = output['field_logits'], output['value_logits'], output['event_logits']
        expected = model.score_outcomes(fields, values, events)
        torch.testing.assert_close(expected, output['action_logits'])
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = model.score_outcomes(tuple(x[:, permutation] for x in fields), values[:, permutation], events[:, permutation])
        torch.testing.assert_close(permuted, expected[:, permutation], atol=1e-7, rtol=1e-6)
        # Changing all context-side parameters cannot change scores of fixed outcomes.
        with torch.no_grad():
            model.context_projection.weight.add_(100)
            model.action_embedding.add_(100)
            model.glyph_projection.weight.add_(100)
        torch.testing.assert_close(model.score_outcomes(fields, values, events), expected, rtol=0, atol=0)
        # Equal outcomes have equal scores, regardless of which action produced them.
        equal = model.score_outcomes(tuple(x[:, :1].expand_as(x) for x in fields),
                                     values[:, :1].expand_as(values), events[:, :1].expand_as(events))
        torch.testing.assert_close(equal, equal[:, :1].expand_as(equal), rtol=0, atol=0)

    def test_reachable_reset_value_is_separate_from_life_loss_and_overflow(self):
        model = make_model()
        output = model(torch.randn(1, 12, 16), torch.randn(1, 12, 16))
        batch = targets(1)
        batch['distances'][0] = torch.tensor([7, -1, 128, 300])
        baseline = neural_outcome_losses(output, batch)
        batch['lost_life'][0, 0] = True
        reset = neural_outcome_losses(output, batch)
        torch.testing.assert_close(reset['losses']['value'], baseline['losses']['value'], rtol=0, atol=0)
        expected = F.cross_entropy(output['value_logits'].flatten(0, 1), torch.tensor([7, 129, 128, 128]))
        torch.testing.assert_close(reset['losses']['value'], expected)
        self.assertNotEqual(reset['losses']['events'].item(), baseline['losses']['events'].item())
        self.assertEqual(reset['diagnostics']['life_loss_fraction'].item(), .25)
        self.assertEqual(reset['diagnostics']['value_overflow_fraction'].item(), .25)
        self.assertEqual(reset['diagnostics']['lost_life_support'].item(), 1)

    def test_zero_policy_support_is_finite_excluded_and_has_no_fake_accuracy(self):
        model = make_model()
        output = model(torch.randn(2, 12, 16), torch.randn(2, 12, 16))
        output['action_logits'].retain_grad()
        batch = targets()
        batch['optimal'].zero_()
        result = neural_outcome_losses(output, batch)
        self.assertTrue(torch.isfinite(result['total']))
        self.assertEqual(result['losses']['policy'].item(), 0.)
        self.assertIsNone(result['diagnostics']['set_accuracy'])
        self.assertIsNone(result['diagnostics']['optimal_probability'])
        self.assertIsNone(result['diagnostics']['lost_life_recall'])
        self.assertEqual(result['diagnostic_weights']['set_accuracy'].item(), 0)
        result['total'].backward()
        self.assertEqual(output['action_logits'].grad.abs().sum().item(), 0.)
        self.assertGreater(model.field_heads[0].weight.grad.abs().sum().item(), 0.)
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()))

    def test_policy_averages_valid_rows_and_multi_action_targets(self):
        model = make_model()
        output = model(torch.randn(2, 12, 16), torch.randn(2, 12, 16))
        batch = targets()
        batch['optimal'] = torch.tensor([3, 0])
        result = neural_outcome_losses(output, batch)
        expected = -output['action_logits'][0].log_softmax(-1)[:2].mean()
        torch.testing.assert_close(result['losses']['policy'], expected)
        self.assertEqual(result['diagnostics']['policy_valid_count'].item(), 1)
        self.assertEqual(result['diagnostics']['policy_valid_fraction'].item(), .5)
        with self.assertRaisesRegex(ValueError, 'missing target'):
            neural_outcome_losses(output, {k: v for k, v in batch.items() if k != 'lost_life'})
        with self.assertRaisesRegex(ValueError, 'unknown loss'):
            neural_outcome_losses(output, batch, {'prediction': 1.})


if __name__ == '__main__':
    unittest.main()
