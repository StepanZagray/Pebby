"""CPU public-input/versioning tests; no game, official data, or route-fit claims.

Synthetic checkpoint lineage fields below are format fixtures, not assertions
that these randomly initialized encoders are the protected trained checkpoint.
"""
import copy
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from pebby.agent.spatial_outcome_planner import SpatialOutcomePlanner
from pebby.agent.spatial_outcome_policy import load_checkpoint as load_original
from pebby.agent.spatial_route_outcome_planner import SpatialRouteOutcomePlanner, SpatialRouteOutcomePlannerConfig
from pebby.agent.spatial_route_outcome_policy import (FORMAT, SpatialRouteOutcomePolicy,
                                                    checkpoint_from_parent, load_checkpoint)
from tests.test_spatial_outcome_policy import checkpoint, make_policy, public_inputs


class SpatialRouteOutcomePlannerTests(unittest.TestCase):
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
        torch.manual_seed(19)
        self.parent = SpatialOutcomePlanner(channels=8, width=8, hud_width=8,
                                             summary=16, comparator_hidden=16).eval()
        self.model = SpatialRouteOutcomePlanner.from_parent(self.parent, route_channels=16, route_heads=4)
        self.inputs = (torch.randn(2, 160, 8), torch.randn(2, 160, 8),
                       torch.randn(2, 14), torch.randn(2, 144).softmax(-1))

    def assert_output_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key in ('action_logits', 'value_logits', 'event_logits'):
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        for left, right in zip(actual['field_logits'], expected['field_logits']):
            torch.testing.assert_close(left, right, atol=0, rtol=0)

    def train_route_twice(self):
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(name.startswith('route_readout.'))
        optimizer = torch.optim.SGD([p for p in self.model.parameters() if p.requires_grad], lr=.2)
        target = torch.tensor([0, 3, 7, 9, 2, 8, 4, 1])
        norms = []
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            result = self.model(*self.inputs)
            loss = F.cross_entropy(result['value_logits'].flatten(0, 1), target)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            norms.append({name: float(parameter.grad.abs().sum()) for name, parameter in self.model.named_parameters()
                          if name.startswith('route_readout.') and parameter.grad is not None})
            self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                                for parameter in self.model.parameters()))
            optimizer.step()
        return norms

    def test_parent_transfer_preserves_every_typed_output_bitwise_without_aliasing(self):
        with torch.no_grad():
            self.assert_output_equal(self.model(*self.inputs), self.parent(*self.inputs))
        for name, parameter in self.parent.named_parameters():
            transferred = dict(self.model.named_parameters())[name]
            torch.testing.assert_close(parameter, transferred, atol=0, rtol=0)
            self.assertNotEqual(parameter.data_ptr(), transferred.data_ptr())
        self.assertTrue(all(p.requires_grad for p in self.model.parameters()))
        self.assertFalse(self.model.training)
        self.assertEqual(self.model.config()['route_version'], 1)
        self.assertEqual(SpatialRouteOutcomePlanner(self.model.config()).config(), self.model.config())
        with self.assertRaises(ValueError):
            SpatialRouteOutcomePlanner.from_parent(self.model)

    def test_parent_freezing_is_not_silently_copied(self):
        self.parent.requires_grad_(False).train()
        migrated = SpatialRouteOutcomePlanner.from_parent(self.parent)
        self.assertTrue(migrated.training)
        self.assertTrue(all(p.requires_grad for p in migrated.parameters()))

    def test_route_interiors_receive_gradients_on_second_update(self):
        norms = self.train_route_twice()
        self.assertGreater(norms[0]['route_readout.output_projection.weight'], 0)
        for name in ('public_projection.weight', 'spatial_projection.weight', 'query_projection.weight',
                     'attention.in_proj_weight', 'ffn.0.weight'):
            self.assertEqual(norms[0]['route_readout.' + name], 0)
            self.assertGreater(norms[1]['route_readout.' + name], 0)
        with torch.no_grad():
            after, parent = self.model(*self.inputs), self.parent(*self.inputs)
        self.assertFalse(torch.equal(after['value_logits'], parent['value_logits']))
        self.assertFalse(torch.equal(after['action_logits'], parent['action_logits']))
        torch.testing.assert_close(after['event_logits'], parent['event_logits'], atol=0, rtol=0)
        for left, right in zip(after['field_logits'], parent['field_logits']):
            torch.testing.assert_close(left, right, atol=0, rtol=0)

    def test_trained_route_reads_all_cells_with_old_summary_held_fixed(self):
        # A structural sensitivity check: not evidence of a learned routing skill.
        self.train_route_twice()
        cfg = self.model.cfg
        summary = torch.randn(2, 4, cfg.summary)
        condition = torch.randn(2, 4, cfg.hud_width + 14 + 4)
        cells = torch.randn(2, 4, cfg.width, 144, requires_grad=True)
        raw = torch.randn(2, 144, cfg.channels, requires_grad=True)
        state = torch.randn(2, 144, cfg.channels, requires_grad=True)
        before = self.model.route_readout(summary, condition, cells, raw, state)
        before.square().mean().backward()
        # Every public cell can affect the route, including distant cells that
        # were not selected by either original player-weighted pool.
        self.assertTrue((raw.grad.abs().sum(-1) > 0).all())
        self.assertTrue((state.grad.abs().sum(-1) > 0).all())
        self.assertTrue((cells.grad.abs().sum(2) > 0).all())
        changed = raw.detach().clone()
        changed[:, 143, 0] += 5
        after = self.model.route_readout(summary, condition, cells.detach(), changed, state.detach())
        self.assertFalse(torch.equal(before, after))

    def test_comparator_has_no_new_bypass_and_branch_permutations_commute(self):
        self.train_route_twice()
        with torch.no_grad():
            output = self.model(*self.inputs)
            reconstructed = self.model.score_outcomes(output['field_logits'], output['value_logits'], output['event_logits'])
            torch.testing.assert_close(reconstructed, output['action_logits'], atol=0, rtol=0)
            order = torch.tensor([2, 0, 3, 1])
            permuted = self.model(*self.inputs, action_rows=torch.eye(4)[order])
        for key in ('action_logits', 'value_logits', 'event_logits'):
            torch.testing.assert_close(permuted[key], output[key][:, order], atol=2e-6, rtol=2e-5)
        for left, right in zip(permuted['field_logits'], output['field_logits']):
            torch.testing.assert_close(left, right[:, order], atol=2e-6, rtol=2e-5)

    def test_configuration_and_public_input_contracts_reject_invalid_inputs(self):
        for change in (dict(route_heads=3), dict(route_channels=0), dict(route_version=2), dict(horizon=2)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                SpatialRouteOutcomePlanner({**self.model.config(), **change})
        with self.assertRaises(ValueError):
            self.model(*self.inputs[:3], torch.ones(2, 144))
        with self.assertRaises(TypeError):
            self.model(*self.inputs, distances=torch.zeros(2, 4))

    def test_versioned_public_policy_roundtrip_and_original_loader_is_unchanged(self):
        original = make_policy()
        route = SpatialRouteOutcomePlanner.from_parent(original.planner, route_channels=16, route_heads=4)
        policy = SpatialRouteOutcomePolicy(copy.deepcopy(original.encoder), route).eval()
        inputs = public_inputs()
        with torch.no_grad():
            torch.testing.assert_close(policy(*inputs), original(*inputs), atol=0, rtol=0)
        base = checkpoint(original)
        migrated = checkpoint_from_parent(base, route, parent_checkpoint_sha256='a' * 64)
        self.assertNotEqual(base['format'], migrated['format'])
        self.assertEqual(migrated['format'], FORMAT)
        self.assertEqual(migrated['source_checkpoint_sha256'], 'a' * 64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic-route.pt'
            torch.save(migrated, path)
            loaded, metadata = load_checkpoint(path, 'cpu')
            with torch.no_grad():
                torch.testing.assert_close(loaded(*inputs), policy(*inputs), atol=0, rtol=0)
            self.assertEqual(metadata['format'], FORMAT)
            self.assertEqual(loaded.config()['decision_architecture'], 'spatial_route_outcomes')
            self.assertEqual(loaded.config()['history'], 8)
            self.assertTrue(all(not p.requires_grad for p in loaded.encoder.parameters()))
            self.assertFalse(loaded.encoder.training)
            for name, value in policy.planner.state_dict().items():
                torch.testing.assert_close(loaded.planner.state_dict()[name], value, atol=0, rtol=0)
            with self.assertRaises(ValueError):
                load_original(path, 'cpu')
            torch.save(base, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path, 'cpu')

    def test_loader_rejects_wrong_runtime_privilege_and_missing_route_weights(self):
        original = make_policy()
        route = SpatialRouteOutcomePlanner.from_parent(original.planner, route_channels=16, route_heads=4)
        migrated = checkpoint_from_parent(checkpoint(original), route, parent_checkpoint_sha256='a' * 64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic-route.pt'
            for change in (dict(encoder_frozen=False), dict(official_training_inputs=True),
                           dict(privileged_inference_inputs=True), dict(encoder_runtime={}),
                           dict(encoder_weights_sha256='0' * 64)):
                torch.save({**migrated, **change}, path)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    load_checkpoint(path, 'cpu')
            broken = copy.deepcopy(migrated)
            broken['planner_weights'].pop('route_readout.output_projection.weight')
            torch.save(broken, path)
            with self.assertRaises(RuntimeError):
                load_checkpoint(path, 'cpu')


if __name__ == '__main__':
    unittest.main()
