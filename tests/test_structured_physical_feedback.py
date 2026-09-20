import unittest

import torch

from pebby.agent.neural_imagination import NeuralImagination
from pebby.agent.structured_local_glyph import LocalGlobalGlyphTransition
from pebby.agent.structured_physical_feedback import (
    PHYSICAL_FEEDBACK_FORMAT, PhysicalFeedbackConfig, PhysicalFeedbackTransition,
)


class PhysicalFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def pair(self):
        torch.manual_seed(42)
        base = LocalGlobalGlyphTransition(loops=1)
        torch.manual_seed(42)
        feedback = PhysicalFeedbackTransition(loops=1)
        return base, feedback

    def observed(self, batch=2):
        field = torch.randn(batch, 148, 96)
        field[..., 85:] = 0
        return field

    def test_shared_initialization_and_strict_warmstart(self):
        base, model = self.pair()
        self.assertEqual(model.parameter_count() - base.parameter_count(), 264)
        for key, value in base.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
        new_weight = model.budget_feedback.weight.detach().clone()
        with torch.no_grad():
            base.output[1].bias.add_(.1)
        self.assertEqual(model.warmstart_from_local_state_dict(base.state_dict()),
                         ['budget_feedback.weight'])
        for key, value in base.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
            self.assertNotEqual(model.state_dict()[key].data_ptr(), value.data_ptr())
        torch.testing.assert_close(model.budget_feedback.weight, new_weight, atol=0, rtol=0)

        before = {key: value.clone() for key, value in model.state_dict().items()}
        missing = dict(base.state_dict())
        missing.pop('position')
        bad_shape = dict(base.state_dict())
        bad_shape['position'] = torch.zeros(1)
        for bad in (missing, bad_shape, model.state_dict()):
            with self.assertRaises(ValueError):
                model.warmstart_from_local_state_dict(bad)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    def test_channels_encode_predictions_and_readout_sees_final_field(self):
        base, model = self.pair()
        field = self.observed()
        before = field.clone()
        actions = torch.tensor([0, 3])
        ordinary = base(field, actions)
        result = model(field, actions)
        predicted = result['field']
        torch.testing.assert_close(field, before, atol=0, rtol=0)
        torch.testing.assert_close(predicted[..., :85], ordinary['field'][..., :85], atol=0, rtol=0)
        for name, logits in result['feedback_logits'].items():
            torch.testing.assert_close(logits, ordinary['readout'][name + '_logits'], atol=0, rtol=0)
        torch.testing.assert_close(predicted[:, :144, 85],
                                   result['feedback_logits']['player'].softmax(-1))
        self.assertEqual(float(predicted[:, 144:, 85].abs().sum().detach()), 0)
        budget = model.budget_feedback(result['feedback_logits']['steps'].softmax(-1))
        torch.testing.assert_close(predicted[..., 86:92], budget[:, None].expand(-1, 148, -1))
        torch.testing.assert_close(predicted[..., 92:96],
                                   result['feedback_logits']['lives'].softmax(-1)[:, None].expand(-1, 148, -1))
        for key, value in model.readout(predicted).items():
            torch.testing.assert_close(result['readout'][key], value, atol=0, rtol=0)
        events = model.event_head(torch.cat((model.readout.summary(field),
                                            model.readout.summary(predicted),
                                            model.action_embedding(actions)), -1))
        for index, name in enumerate(('lost_life', 'terminal', 'won')):
            torch.testing.assert_close(result['events'][name + '_logits'], events[:, index], atol=0, rtol=0)

    def test_recurrence_consumes_previous_feedback_and_keeps_gradients(self):
        _, model = self.pair()
        field = self.observed()
        actions = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
        inputs, outputs = [], []

        def capture(_module, args, output):
            inputs.append(args[0])
            output['field'].retain_grad()
            outputs.append(output)

        hook = model.register_forward_hook(capture)
        result = model.rollout(field, actions)
        hook.remove()
        self.assertIs(inputs[0], field)
        for step in range(1, 4):
            self.assertIs(inputs[step], outputs[step - 1]['field'])
        result['fields'][:, -1, :, :85].square().mean().backward()
        for output in outputs[:-1]:
            self.assertGreater(float(output['field'].grad[..., 85:].abs().sum()), 0)
        self.assertGreater(float(model.budget_feedback.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.readout.player.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.readout.global_head[-1].weight.grad.abs().sum()), 0)
        with torch.no_grad():
            first = outputs[0]['field']
            cleared = torch.cat((first[..., :85], torch.zeros_like(first[..., 85:])), -1)
            cleared_next = model.predict(cleared, actions[:, 1])
            self.assertGreater(float((cleared_next[..., :85] - result['fields'][:, 1, :, :85]).abs().max()), 1e-6)

    def test_predict_rollout_and_roundtrip_agree(self):
        _, model = self.pair()
        field = self.observed()
        actions = torch.tensor([[0, 1], [2, 3]])
        with torch.no_grad():
            result = model.rollout(field, actions)
            current = field
            for step in range(2):
                output = model(current, actions[:, step])
                torch.testing.assert_close(model.predict(current, actions[:, step]), output['field'], atol=0, rtol=0)
                torch.testing.assert_close(result['fields'][:, step], output['field'], atol=0, rtol=0)
                for group in ('readout', 'events', 'glyph_logits', 'feedback_logits'):
                    for key, value in output[group].items():
                        torch.testing.assert_close(result[group][key][:, step], value, atol=0, rtol=0)
                current = output['field']
            other = PhysicalFeedbackTransition(model.config())
            other.load_state_dict(model.state_dict(), strict=True)
            torch.testing.assert_close(other.predict(field, actions[:, 0]), result['fields'][:, 0], atol=0, rtol=0)
        self.assertEqual(model.checkpoint_format, PHYSICAL_FEEDBACK_FORMAT)
        self.assertEqual(model.config()['variant'], 'physical_feedback')

    def test_neural_imagination_direct_calls_preserve_feedback(self):
        _, model = self.pair()
        planner = NeuralImagination(model, dict(horizon=4, hidden=16))
        inputs, outputs = [], []

        def capture(_module, args, output):
            inputs.append(args[0])
            outputs.append(output['field'])

        hook = model.register_forward_hook(capture)
        with torch.no_grad():
            result = planner.imagine(self.observed(1))
        hook.remove()
        self.assertEqual(result['action_logits'].shape, (1, 4))
        self.assertTrue(torch.isfinite(result['action_logits']).all())
        self.assertEqual(len(inputs), 4)
        for step in range(1, 4):
            self.assertIs(inputs[step], outputs[step - 1])

    def test_rejects_invalid_inputs_and_future_labels(self):
        _, model = self.pair()
        field = self.observed()
        with self.assertRaises(ValueError):
            PhysicalFeedbackConfig(variant='local_global_glyph')
        with self.assertRaises(ValueError):
            model.rollout(field, torch.empty(2, 0, dtype=torch.long))
        with self.assertRaises(ValueError):
            model(field, torch.tensor([0, 4]))
        with self.assertRaises(ValueError):
            model.predict(field, torch.tensor([0, 1]), loops=0)
        with self.assertRaises(TypeError):
            model(field, torch.tensor([0, 1]), next_fields=field)
