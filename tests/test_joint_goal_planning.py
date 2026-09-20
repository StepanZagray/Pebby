import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from pebby.agent.joint_goal_planning import (
    FORMAT, JointGoalPlanning, hud_patches, load_checkpoint, save_checkpoint,
)


class JointGoalPlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, **overrides):
        torch.manual_seed(42)
        return JointGoalPlanning(dict(horizon=2, encoder_loops=1, dynamics_loops=1, **overrides))

    def public(self, batch=1):
        frames = torch.randint(0, 16, (batch, 8, 64, 64), dtype=torch.uint8)
        valid = torch.tensor([[False] * 5 + [True] * 3]).expand(batch, -1).clone()
        actions = torch.tensor([[-1] * 6 + [0, 3]]).expand(batch, -1).clone()
        return frames, valid, actions

    def test_fresh_construction_interfaces_and_parameter_accounting(self):
        with patch('torch.load', side_effect=AssertionError('construction must not load weights')):
            model = self.model()
        public = self.public()
        details = model.encode_details(*public)
        field = details['field']
        self.assertEqual(field.shape, (1, 148, 96))
        self.assertEqual(details['role_logits'].shape, (1, 144, 8))
        self.assertEqual(details['goal_shape_logits'].shape, (1, 144, 6))
        self.assertEqual(details['carried_shape_logits'].shape, (1, 6))
        self.assertEqual(model.relation(field)['hidden'].shape, (1, 144, 64))
        self.assertEqual(model.relation(field)['solved_logits'].shape, (1, 144))
        self.assertEqual(model.value_logits(field).shape, (1, 130))
        self.assertEqual(model.continuation_logits(field).shape, (1, 4))
        pixels = model.pixel_logits(field)
        self.assertEqual(pixels['board'].shape, (1, 144, 7, 7, 16))
        self.assertEqual(pixels['hud'].shape, (1, 4, 12, 16, 16))
        self.assertTrue(torch.equal(field[..., 85:], torch.zeros_like(field[..., 85:])))
        self.assertTrue(torch.equal(field[:, 144:, 48:70], torch.zeros_like(field[:, 144:, 48:70])))
        torch.testing.assert_close(details['visibility_logits'], model.visibility_logits(field))
        for start, stop in ((70, 76), (76, 80), (80, 84)):
            torch.testing.assert_close(field[..., start:stop].sum(-1), torch.ones(1, 148))
        counts = model.parameter_counts()
        self.assertEqual(counts['total'], counts['trainable'])
        self.assertEqual(counts['total'], counts['deployed'] + counts['decoder_only'])
        self.assertEqual(counts['decoder_only'], sum(p.numel() for p in model.pixel_decoder.parameters()))
        self.assertEqual(model.config()['architecture'], 'structured')
        self.assertEqual(model.config()['history'], 8)
        self.assertIsNone(model.continuation.scorer.bias)

    def test_padding_is_inert_but_valid_history_is_consumed(self):
        model = self.model()
        frames, valid, actions = self.public()
        with torch.no_grad():
            expected = model.encode(frames, valid, actions)
            padded = frames.clone()
            padded[:, :5] = (padded[:, :5] + 3) % 16
            torch.testing.assert_close(model.encode(padded, valid, actions), expected, atol=0, rtol=0)
            changed = frames.clone()
            changed[:, 5:7] = (changed[:, 5:7] + 7) % 16
            self.assertGreater(float((model.encode(changed, valid, actions)[..., :48] - expected[..., :48]).abs().max()), 1e-5)
        with self.assertRaises(ValueError):
            model.encode(frames, torch.tensor([[True, False] + [True] * 6]), actions)
        bad_actions = actions.clone()
        bad_actions[:, 0] = 0
        with self.assertRaises(ValueError):
            model.encode(frames, valid, bad_actions)
        with self.assertRaises(TypeError):
            model(frames, valid, actions, goal_triple=torch.zeros(1, 3))

    def test_goal_feedback_ablation_preserves_initial_parameters_and_changes_readouts(self):
        active = self.model(relation_feedback=True)
        control = self.model(relation_feedback=False)
        for key, value in active.state_dict().items():
            torch.testing.assert_close(value, control.state_dict()[key], atol=0, rtol=0)
        field = active.encode(*self.public())
        torch.testing.assert_close(control.conditioned_field(field), field, atol=0, rtol=0)
        with torch.no_grad():
            self.assertGreater(float((active.value_logits(field) - control.value_logits(field)).abs().max()), 1e-5)
            self.assertGreater(float((active.continuation_logits(field) - control.continuation_logits(field)).abs().max()), 1e-5)
        # Control still exposes the supervised relation head; it is not removed.
        relation = control.relation(field.detach())
        relation['compatibility_logits'].square().mean().backward()
        self.assertGreater(float(control.compatibility_head.weight.grad.abs().sum()), 0)

    def test_policy_gradients_cross_imagined_fields_but_not_hard_continuation(self):
        model = self.model()
        field = model.encode(*self.public())
        inputs, outputs = [], []

        def capture(_module, args, output):
            inputs.append(args[0])
            output['field'].retain_grad()
            outputs.append(output['field'])

        hook = model.dynamics.register_forward_hook(capture)
        result = model.imagine(field)
        hook.remove()
        self.assertIs(inputs[1], outputs[0])
        self.assertEqual(result['imagined_fields'].shape, (1, 4, 2, 148, 96))
        self.assertEqual(result['imagined_actions'].shape, (1, 4, 2))
        self.assertEqual(result['imagined_value_logits'].shape, (1, 4, 2, 130))
        self.assertEqual(result['transition_count_per_decision'], 8)
        F.cross_entropy(result['action_logits'], torch.tensor([3])).backward()
        for parameter in (model.encoder.cell_projection.weight, model.encoder.appearance.network[-1].weight,
                          model.encoder.glyph.mlp[-1].weight, model.dynamics.action_embedding.weight,
                          model.dynamics.budget_feedback.weight, model.relation_projection.weight,
                          model.compatibility_head.weight, model.value_head.weight, model.trajectory.weight_ih):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertGreater(float(outputs[0].grad.abs().sum()), 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.continuation.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.pixel_decoder.parameters()))

    def test_auxiliary_losses_reach_continuation_decoder_and_shared_visibility(self):
        model = self.model()
        field = model.encode(*self.public())
        prediction = model.dynamics(field, torch.tensor([1]))
        following = prediction['field']
        pixels = model.pixel_logits(following)
        loss = (F.cross_entropy(model.continuation_logits(following), torch.tensor([3]))
                + pixels['board'].square().mean() + pixels['hud'].square().mean()
                + model.visibility_logits(following).square().mean()
                + prediction['readout']['role_logits'].square().mean())
        loss.backward()
        for parameter in (model.continuation.scorer.weight, model.continuation.attention.in_proj_weight,
                          model.pixel_decoder.board[-1].weight, model.pixel_decoder.hud[-1].weight,
                          model.encoder.visibility.weight, model.encoder.cell_projection.weight,
                          model.dynamics.output[-1].weight, model.dynamics.readout.cell.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_root_permutation_and_k4_forward_without_persistent_state(self):
        model = self.model()
        public = self.public()
        with torch.no_grad():
            field = model.encode(*public)
            canonical = model.imagine(field, horizon=4)
            permuted = model.imagine(field, horizon=4, root_actions=torch.tensor([[3, 1, 0, 2]]))
            torch.testing.assert_close(canonical['action_logits'], permuted['action_logits'], atol=1e-6, rtol=1e-5)
            first = model(*public)
            model(*self.public())
            torch.testing.assert_close(first, model(*public), atol=0, rtol=0)
        self.assertEqual(canonical['transition_count_per_decision'], 16)
        self.assertTrue(torch.isfinite(canonical['action_logits']).all())

    def test_hud_patch_order_and_checkpoint_roundtrip_are_exact(self):
        frames = torch.zeros(1, 64, 64, dtype=torch.long)
        for strip in range(4):
            frames[:, 52:64, strip * 16:(strip + 1) * 16] = strip
        patches = hud_patches(frames)
        for strip in range(4):
            self.assertTrue((patches[:, strip] == strip).all())
        model = self.model()
        public = self.public()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'candidate.pt'
            saved = save_checkpoint(path, model, {'promoted': False, 'initialization': 'fresh'})
            before = torch.random.get_rng_state().clone()
            restored, payload = load_checkpoint(path)
            torch.testing.assert_close(torch.random.get_rng_state(), before)
            self.assertEqual(payload['format'], FORMAT)
            self.assertEqual(payload['config'], model.config())
            self.assertFalse(payload['promoted'])
            model.eval()  # Compare the same attention execution mode as the loader.
            with torch.no_grad():
                torch.testing.assert_close(restored(*public), model(*public), atol=0, rtol=0)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, model)
            self.assertEqual(path.read_bytes(), original)
            broken = copy.deepcopy(saved)
            broken['weights'].pop('summary_query')
            bad_path = Path(directory) / 'broken.pt'
            torch.save(broken, bad_path)
            with self.assertRaises(RuntimeError):
                load_checkpoint(bad_path)
            with self.assertRaises(ValueError):
                JointGoalPlanning({**model.config(), 'history': 9})
