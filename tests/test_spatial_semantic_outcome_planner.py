"""CPU structural tests, not evidence of learned game competence.

Synthetic checkpoint provenance is a format fixture. The protected perceptor
weight digest is patched to the synthetic teacher only within those tests.
"""
import copy
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from pebby.agent.cell_appearance import CellAppearance, FORMAT as CELL_FORMAT
from pebby.agent.neural_outcome_policy import weights_sha256
from pebby.agent.spatial_outcome_planner import SpatialOutcomePlanner
from pebby.agent.spatial_outcome_policy import load_checkpoint as load_original
from pebby.agent.spatial_semantic_outcome_planner import SpatialSemanticOutcomePlanner
from pebby.agent import spatial_semantic_outcome_policy as policy_module
from tests.test_spatial_outcome_policy import checkpoint, make_policy, public_inputs


def semantics(batch=2):
    return torch.cat((torch.randn(batch, 144, 8).sigmoid(),
                      *(torch.randn(batch, 144, size).softmax(-1) for size in (6, 4, 4))), -1)


def migrate(parent, actor=False):
    return SpatialSemanticOutcomePlanner.from_parent(parent, actor=actor, route_channels=16,
                                                     actor_channels=16, actor_heads=4)


class SpatialSemanticOutcomePlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        assert not torch.cuda.is_initialized()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        assert not torch.cuda.is_initialized()

    def setUp(self):
        torch.manual_seed(29)
        self.parent = SpatialOutcomePlanner(channels=8, width=8, hud_width=8,
                                             summary=16, comparator_hidden=16).eval()
        self.inputs = (torch.randn(2, 160, 8), torch.randn(2, 160, 8),
                       torch.randn(2, 14), torch.randn(2, 144).softmax(-1), semantics())

    def assert_outputs_equal(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in ('action_logits', 'value_logits', 'event_logits'):
            torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)
        for a, b in zip(left['field_logits'], right['field_logits']):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def train_twice(self, model):
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        gradients = []
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            result = model(*self.inputs)
            loss = F.cross_entropy(result['action_logits'], torch.tensor([0, 3]))
            loss = loss + F.cross_entropy(result['value_logits'].flatten(0, 1), torch.arange(8))
            loss.backward()
            gradients.append({name: float(p.grad.abs().sum()) for name, p in model.named_parameters()
                              if p.grad is not None})
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            optimizer.step()
        return gradients

    def test_both_arms_preserve_parent_bitwise_and_share_initial_weights_without_aliases(self):
        self.parent.requires_grad_(False)
        torch.manual_seed(42)
        control = migrate(self.parent)
        torch.manual_seed(42)
        actor = migrate(self.parent, actor=True)
        with torch.no_grad():
            expected = self.parent(*self.inputs[:4])
            self.assert_outputs_equal(control(*self.inputs), expected)
            self.assert_outputs_equal(actor(*self.inputs), expected)
        for name, value in control.state_dict().items():
            torch.testing.assert_close(value, actor.state_dict()[name], atol=0, rtol=0)
            self.assertNotEqual(value.data_ptr(), actor.state_dict()[name].data_ptr())
        for model in (control, actor):
            self.assertTrue(all(p.requires_grad for p in model.parameters()))
            self.assertFalse(model.training)
            for name, value in self.parent.state_dict().items():
                self.assertNotEqual(value.data_ptr(), model.state_dict()[name].data_ptr())
        self.assertFalse(any(name.startswith('actor_readout.') for name, _ in control.named_parameters()))
        self.assertEqual(SpatialSemanticOutcomePlanner(actor.config()).config(), actor.config())

    def test_semantics_and_route_interiors_receive_gradients_by_second_update(self):
        model = migrate(self.parent)
        gradients = self.train_twice(model)
        self.assertGreater(gradients[0]['semantic_grid_projection.weight'], 0)
        for suffix in ('semantic_projection.weight', 'public_projection.weight', 'attention.in_proj_weight'):
            name = 'route_readout.' + suffix
            self.assertEqual(gradients[0][name], 0)
            self.assertGreater(gradients[1][name], 0)
        semantic = self.inputs[-1].detach().requires_grad_()
        result = model(*self.inputs[:4], semantic)
        result['value_logits'].square().mean().backward()
        self.assertTrue((semantic.grad.abs().sum(-1) > 0).all())

    def test_actor_interiors_receive_gradients_and_bypass_typed_logits(self):
        model = migrate(self.parent, actor=True)
        gradients = self.train_twice(model)
        self.assertGreater(gradients[0]['actor_readout.output_projection.weight'], 0)
        for suffix in ('query_projection.weight', 'public_projection.weight',
                       'blocks.0.attention.in_proj_weight', 'blocks.2.ffn.0.weight'):
            name = 'actor_readout.' + suffix
            self.assertEqual(gradients[0][name], 0)
            self.assertGreater(gradients[1][name], 0)
        with torch.no_grad():
            result = model(*self.inputs, return_components=True)
            torch.testing.assert_close(result['action_logits'], result['outcome_action_logits'] +
                                       result['actor_action_logits'], atol=0, rtol=0)
            with patch.object(model, 'score_outcomes', return_value=torch.zeros(2, 4)):
                bypass = model(*self.inputs, return_components=True)
            torch.testing.assert_close(bypass['action_logits'], result['actor_action_logits'], atol=0, rtol=0)
            self.assertGreater(float(bypass['action_logits'].std(-1).sum()), 0)

    def test_action_permutations_commute_after_updates(self):
        order = torch.tensor([2, 0, 3, 1])
        for actor in (False, True):
            model = migrate(self.parent, actor)
            self.train_twice(model)
            with torch.no_grad():
                normal = model(*self.inputs, return_components=True)
                permuted = model(*self.inputs, action_rows=torch.eye(4)[order], return_components=True)
            for key in normal:
                if key == 'field_logits':
                    for a, b in zip(permuted[key], normal[key]):
                        torch.testing.assert_close(a, b[:, order], atol=2e-6, rtol=2e-5)
                else:
                    torch.testing.assert_close(permuted[key], normal[key][:, order], atol=2e-6, rtol=2e-5)
            if not actor:
                self.assertEqual(int(normal['actor_action_logits'].count_nonzero()), 0)

    def test_public_contract_rejects_invalid_probabilities_or_privileged_arguments(self):
        model = migrate(self.parent)
        invalid = [torch.zeros(2, 143, 22), torch.zeros(2, 144, 22),
                   self.inputs[-1].clone(), self.inputs[-1].clone()]
        invalid[2][0, 0, 0] = float('nan')
        invalid[3][0, 0, 0] = 1.1
        for semantic in invalid:
            with self.assertRaises(ValueError):
                model(*self.inputs[:4], semantic)
        for change in (dict(actor=1), dict(actor_heads=3), dict(horizon=2), dict(semantic_channels=23)):
            with self.assertRaises(ValueError):
                SpatialSemanticOutcomePlanner({**model.config(), **change})
        with self.assertRaises(TypeError):
            model(*self.inputs, distances=torch.zeros(2, 4))

    def test_public_history_roundtrip_and_current_frame_only_frozen_perceptor(self):
        original = make_policy()
        inputs = public_inputs()
        for actor in (False, True):
            planner, perceptor = migrate(original.planner, actor), CellAppearance()
            digest = weights_sha256(perceptor.state_dict())
            wrapper = policy_module.SpatialSemanticOutcomePolicy(copy.deepcopy(original.encoder), planner, perceptor).eval()
            captured = []
            hook = perceptor.register_forward_pre_hook(lambda module, args: captured.append(args[0].clone()))
            with torch.no_grad():
                expected = wrapper(*inputs)
                torch.testing.assert_close(expected, original(*inputs), atol=0, rtol=0)
            hook.remove()
            torch.testing.assert_close(captured[0], inputs[0][:, -1], atol=0, rtol=0)
            self.assertEqual(wrapper.config()['architecture'], 'world')
            self.assertEqual(wrapper.config()['history'], 8)
            self.assertEqual(wrapper.parameter_count(), sum(p.numel() for p in wrapper.parameters()))
            with patch.object(policy_module, 'TEACHER_WEIGHTS_SHA256', digest):
                base = checkpoint(original)
                migrated = policy_module.checkpoint_from_parent(base, planner, perceptor)
                for key, original_weights in (('encoder_weights', base['encoder_weights']),
                                               ('planner_weights', planner.state_dict()),
                                               ('perceptor_weights', perceptor.state_dict())):
                    for name, value in original_weights.items():
                        self.assertNotEqual(value.data_ptr(), migrated[key][name].data_ptr())
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'fixture.pt'
                    torch.save(migrated, path)
                    loaded, _ = policy_module.load_checkpoint(path)
                    with torch.no_grad():
                        torch.testing.assert_close(loaded(*inputs), expected, atol=0, rtol=0)
                    with self.assertRaises(ValueError):
                        load_original(path)
                    torch.save(base, path)
                    with self.assertRaises(ValueError):
                        policy_module.load_checkpoint(path)
            wrapper.train()
            self.assertFalse(wrapper.encoder.training)
            self.assertFalse(wrapper.perceptor.training)
            wrapper(*inputs).square().sum().backward()
            self.assertTrue(all(p.grad is None and not p.requires_grad for module in
                                (wrapper.encoder, wrapper.perceptor) for p in module.parameters()))
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in planner.parameters()))

    def test_loader_protects_teacher_even_when_corrupt_digest_is_recomputed(self):
        original, teacher = make_policy(), CellAppearance()
        digest = weights_sha256(teacher.state_dict())
        with patch.object(policy_module, 'TEACHER_WEIGHTS_SHA256', digest), tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.pt'
            base = policy_module.checkpoint_from_parent(checkpoint(original), migrate(original.planner), teacher)
            changes = [dict(perceptor_checkpoint_sha256='0' * 64), dict(perceptor_frozen=False),
                       dict(encoder_runtime={}), dict(official_training_inputs=True),
                       dict(source_checkpoint_sha256='0' * 64), dict(privileged_inference_inputs=True),
                       dict(semantic_architecture={**base['semantic_architecture'], 'actor': True})]
            for change in changes:
                torch.save({**base, **change}, path)
                with self.assertRaises(ValueError):
                    policy_module.load_checkpoint(path)
            broken = copy.deepcopy(base)
            next(iter(broken['perceptor_weights'].values())).add_(1)
            broken['perceptor_weights_sha256'] = weights_sha256(broken['perceptor_weights'])
            torch.save(broken, path)
            with self.assertRaises(ValueError):
                policy_module.load_checkpoint(path)

    def test_fresh_loader_binds_file_and_weights_and_probability_channel_contract(self):
        teacher = CellAppearance()
        buffer = io.BytesIO()
        torch.save(dict(format=CELL_FORMAT, weights=teacher.state_dict()), buffer)
        payload = buffer.getvalue()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.pt'
            path.write_bytes(payload)
            with patch.object(policy_module, 'TEACHER_SHA256', hashlib.sha256(payload).hexdigest()), \
                    patch.object(policy_module, 'TEACHER_WEIGHTS_SHA256', weights_sha256(teacher.state_dict())):
                loaded = policy_module.load_perceptor(path)
                frames = public_inputs()[0][:, -1]
                actual = policy_module.semantic_probabilities(loaded, frames)
                expected = torch.cat((loaded(frames)[0].sigmoid(),
                                      *(x.softmax(-1) for x in loaded(frames)[1:])), -1)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                self.assertEqual(actual.shape, (2, 144, 22))
                self.assertFalse(actual.requires_grad)
                self.assertEqual(actual.dtype, torch.float32)
                path.write_bytes(payload + b'changed')
                with self.assertRaises(ValueError):
                    policy_module.load_perceptor(path)


if __name__ == '__main__':
    unittest.main()
