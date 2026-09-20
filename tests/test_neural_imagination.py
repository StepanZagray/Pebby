import unittest

import numpy as np

import torch
from torch import nn

from pebby.agent.neural_imagination import NeuralImagination


class TrackingDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(.2))
        self.inputs = []
        self.outputs = []

    def forward(self, field, actions):
        self.inputs.append(field.detach().clone())
        # A nonuniform feature change survives LayerNorm and distinguishes roots.
        delta = torch.zeros_like(field)
        delta[:, :, 0] = actions[:, None].float() + 1
        following = field + self.gain * delta
        if following.requires_grad:
            following.retain_grad()
        self.outputs.append(following)
        event = following[:, 0, 0]
        return dict(field=following, events={name + '_logits': event for name in
                    ('lost_life', 'terminal', 'won')})


class Continuation(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(96, 4)

    def forward(self, field):
        return self.head(field.mean(1))


class ImaginationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.dynamics = TrackingDynamics()
        self.model = NeuralImagination(self.dynamics, dict(horizon=4, hidden=24),
                                       continuation=Continuation())
        self.fields = torch.randn(2, 148, 96)

    def test_recurrence_consumes_own_predictions_and_late_gradients(self):
        result = self.model.imagine(self.fields)
        self.assertEqual(result['imagined_actions'].shape, (2, 4, 4))
        self.assertEqual(result['transition_count_per_root'], 4)
        self.assertEqual(result['transition_count_per_decision'], 16)
        for step in range(1, 4):
            torch.testing.assert_close(self.dynamics.inputs[step], self.dynamics.outputs[step - 1])
        result['action_logits'][:, 0].sum().backward()
        for predicted in self.dynamics.outputs:
            self.assertGreater(float(predicted.grad.abs().sum()), 0.)
        self.assertGreater(float(self.dynamics.gain.grad.abs()), 0.)
        self.assertTrue(all(p.grad is None for p in self.model.continuation.parameters()))
        self.model.continuation_logits(self.fields).sum().backward()
        self.assertTrue(all(p.grad is not None for p in self.model.continuation.parameters()))

    def test_permuting_roots_preserves_canonical_action_logits(self):
        self.model.eval()
        with torch.no_grad():
            native = self.model.imagine(self.fields)
            permuted = self.model.imagine(self.fields, root_actions=torch.tensor([[3, 1, 0, 2], [2, 0, 3, 1]]))
        torch.testing.assert_close(native['action_logits'], permuted['action_logits'], atol=1e-6, rtol=1e-5)

    def test_later_imagined_states_change_scores(self):
        with torch.no_grad():
            one = self.model.imagine(self.fields, horizon=1)['action_logits']
            four = self.model.imagine(self.fields, horizon=4)['action_logits']
        self.assertGreater(float((one - four).abs().max()), 1e-6)

    def test_same_depth_tail_intervention_preserves_actions_and_work(self):
        with torch.no_grad():
            native = self.model.imagine(self.fields)
            ablated = self.model.imagine(self.fields, tail_ablation=True)
        torch.testing.assert_close(native['imagined_actions'], ablated['imagined_actions'])
        self.assertEqual(native['transition_count_per_decision'], ablated['transition_count_per_decision'])
        self.assertGreater(float((native['action_logits'] - ablated['action_logits']).abs().max()), 1e-6)
        with torch.no_grad():
            one = self.model.imagine(self.fields, horizon=1)
            one_ablated = self.model.imagine(self.fields, horizon=1, tail_ablation=True)
        torch.testing.assert_close(one['action_logits'], one_ablated['action_logits'])

    def test_real_transition_interoperability(self):
        from pebby.agent.structured_local_glyph import LocalGlobalGlyphTransition
        model = NeuralImagination(LocalGlobalGlyphTransition(loops=1), dict(horizon=2, hidden=24))
        prediction = model(self.fields[:1])
        self.assertEqual(prediction.shape, (1, 4))
        prediction.square().sum().backward()
        self.assertIsNotNone(model.dynamics.output[1].weight.grad)

    def test_validation_selection_covers_ordered_difficulty_bank(self):
        from tools.train_neural_imagination import stratified_rows
        tiers = np.repeat(np.arange(1, 6), 20)
        rows = stratified_rows(tiers, 15, 42)
        self.assertEqual(len(set(rows)), 15)
        self.assertEqual(np.bincount(tiers[rows])[1:].tolist(), [3] * 5)
        np.testing.assert_array_equal(rows, stratified_rows(tiers, 15, 42))

    def test_invalid_horizon_and_root_duplicates(self):
        for depth in (0, True, 17):
            with self.assertRaises(ValueError):
                self.model.imagine(self.fields, horizon=depth)
        with self.assertRaises(ValueError):
            self.model.imagine(self.fields, root_actions=torch.zeros(2, 4, dtype=torch.long))


if __name__ == '__main__':
    unittest.main()
