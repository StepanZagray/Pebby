"""Tiny CPU pilot contracts, no GPU, heldout fitting, or game execution."""
from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pebby.agent.cell_appearance import CellAppearance
from pebby.agent.neural_outcome_policy import weights_sha256
from pebby.agent.optimal_set_objective import optimal_action_loss
from pebby.agent.spatial_outcome_objective import spatial_outcome_losses
from pebby.agent.spatial_outcome_planner import SpatialOutcomePlanner
from pebby.agent.spatial_semantic_outcome_planner import SpatialSemanticOutcomePlanner
from tools import train_spatial_policy_probe as pilot
from tests.test_spatial_outcome_policy import checkpoint as parent_checkpoint, make_policy


class PolicyProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        assert not torch.cuda.is_initialized()

    def setUp(self):
        torch.manual_seed(37)
        original = SpatialOutcomePlanner(channels=8, width=8, hud_width=8, summary=16, comparator_hidden=16)
        self.model = SpatialSemanticOutcomePlanner.from_parent(original, actor=True, actor_channels=16, route_channels=16)
        semantic = torch.cat((torch.rand(4, 144, 8), *(torch.randn(4, 144, n).softmax(-1) for n in (6, 4, 4))), -1)
        self.items = dict(raw=torch.randn(4, 160, 8), state=torch.randn(4, 160, 8), glyph=torch.randn(4, 14),
            semantic=semantic, optimal=torch.tensor([1, 3, 0, 12]),
            next_player_cell=torch.zeros(4, 4, 2, dtype=torch.long), next_triple=torch.zeros(4, 4, 3, dtype=torch.long),
            next_steps=torch.full((4, 4), 10, dtype=torch.long), next_lives=torch.full((4, 4), 3, dtype=torch.long),
            distances=torch.arange(16).reshape(4, 4), current_triple=torch.zeros(4, 3, dtype=torch.long),
            current_steps=torch.full((4,), 11, dtype=torch.long), current_lives=torch.full((4,), 3, dtype=torch.long),
            player_cell=torch.zeros(4, 2, dtype=torch.long),
            lost_life=torch.zeros(4, 4, dtype=torch.bool), terminal=torch.zeros(4, 4, dtype=torch.bool),
            won=torch.zeros(4, 4, dtype=torch.bool))
        self.encoder = {'player_head.weight': torch.randn(1, 8), 'player_head.bias': torch.zeros(1)}
        self.weights = dict(glyph_change_weights=[[1., 1.]] * 3, physical_weights=[1., 1/3, 1/3, 1/3, .5, .5],
                            event_positive_weights=[1., 1., 1.])

    def test_configuring_arms_preserves_identical_initial_state_and_exact_scope(self):
        before = weights_sha256(self.model.state_dict())
        for arm in pilot.ARMS:
            pilot.configure_arm(self.model, arm)
            self.assertEqual(weights_sha256(self.model.state_dict()), before)
            for name, parameter in self.model.named_parameters():
                self.assertEqual(parameter.requires_grad, arm.startswith('joint-') or name.startswith('actor_readout.'))
        with self.assertRaises(ValueError):
            pilot.configure_arm(self.model, 'unknown')

    def test_joint_replaces_only_native_policy_and_actor_uses_deployed_sum(self):
        player = pilot.player_probabilities(self.items, self.encoder)
        predicted = self.model(self.items['raw'], self.items['state'], self.items['glyph'], player, self.items['semantic'])
        existing = spatial_outcome_losses(self.model, predicted, self.items, self.weights)
        for mode in ('uniform', 'set'):
            joint = pilot.loss_record(self.model, self.items, self.encoder, self.weights, 'joint-' + mode)
            actor = pilot.loss_record(self.model, self.items, self.encoder, self.weights, 'actor-' + mode)
            self.assertEqual(set(actor['losses']), {'policy'})
            expected = optimal_action_loss(predicted['action_logits'], self.items['optimal'], mode)
            torch.testing.assert_close(actor['total'], expected, atol=0, rtol=0)
            torch.testing.assert_close(joint['losses']['policy'], expected, atol=0, rtol=0)
            for name in ('physical', 'value', 'events', 'teacher_policy'):
                torch.testing.assert_close(joint['losses'][name], existing['losses'][name], atol=0, rtol=0)

    def test_two_actor_updates_keep_nonactor_weights_frozen_and_expose_pre_post_clip_norms(self):
        pilot.configure_arm(self.model, 'actor-set')
        frozen = pilot.frozen_planner_digest(self.model)
        before = self.model.actor_readout.output_projection.weight.detach().clone()
        optimizer = pilot.optimizer_for(self.model, .001)
        for step in range(2):
            _, norms = pilot.fit_step(self.model, optimizer, self.items, self.encoder, self.weights, 'actor-set')
            self.assertGreater(norms['actor_preclip'], 0)
            self.assertAlmostEqual(norms['global_preclip'], norms['actor_preclip'], places=5)
            self.assertLessEqual(norms['actor_postclip'], 1.00001)
            self.assertIsNone(norms['route_interior_preclip'])
            if step:
                self.assertGreater(norms['actor_interior_preclip'], 0)
                self.assertGreater(norms['actor_output_relative_update_l2'], 0)
            else:
                self.assertIsNone(norms['actor_output_relative_update_l2'])
            self.assertGreater(norms['actor_update_l2'], 0)
            self.assertGreater(norms['actor_relative_update_l2'], 0)
            self.assertEqual(pilot.frozen_planner_digest(self.model), frozen)
        self.assertFalse(torch.equal(before, self.model.actor_readout.output_projection.weight))

    def test_zero_optimal_keeps_joint_outcome_supervision_but_no_actor_policy_gradient(self):
        items = {**self.items, 'optimal': torch.zeros(4, dtype=torch.long)}
        pilot.configure_arm(self.model, 'actor-uniform')
        actor = pilot.loss_record(self.model, items, self.encoder, self.weights, 'actor-uniform')
        actor['total'].backward()
        self.assertEqual(float(actor['total'].detach()), 0)
        self.assertTrue(all(p.grad is None or p.grad.count_nonzero() == 0 for p in self.model.parameters()))
        self.model.zero_grad(set_to_none=True)
        pilot.configure_arm(self.model, 'joint-set')
        joint = pilot.loss_record(self.model, items, self.encoder, self.weights, 'joint-set')
        joint['total'].backward()
        self.assertGreater(float(self.model.value_head.weight.grad.abs().sum()), 0)
        self.assertEqual(float(joint['losses']['policy'].detach()), 0)

    def test_continuation_filter_excludes_same_level_from_both_sources_and_rejects_invalid_rows(self):
        base = {'seeds': np.array([11, 11, 22, 22, 33, 33])}
        recent = {'seeds': np.array([22, 33, 11, 22])}
        allowed, tiers = pilot.continuation_rows(base, recent, np.arange(6), np.arange(4), [22], {11: 1, 22: 2, 33: 3})
        self.assertEqual(allowed['base_rows'].tolist(), [0, 1, 4, 5])
        self.assertEqual(allowed['recent_rows'].tolist(), [1, 2])
        self.assertEqual(tiers, {11: 1, 33: 3})
        for bad in (np.array([0, 0]), np.array([-1]), np.array([6])):
            with self.assertRaises(ValueError):
                pilot.continuation_rows(base, recent, bad, np.arange(4), [22], {11: 1, 22: 2, 33: 3})
        with self.assertRaises(ValueError):
            pilot.continuation_rows(base, recent, np.arange(6), np.arange(4), [999], {11: 1, 22: 2, 33: 3})

    def test_current_checkpoint_metadata_discards_old_top_level_training_and_qualification(self):
        original, teacher = make_policy(), CellAppearance()
        model = SpatialSemanticOutcomePlanner.from_parent(original.planner, actor=True, actor_channels=16, route_channels=16)
        parent = parent_checkpoint(original)
        parent.update(qualification_passed=False, optimizer_steps=780, quality_filter='old',
                      actual_outcome_comparator_auxiliary_training=True)
        report = dict(continuation=dict(train_levels=9934, heldout_levels=66), sampling={'batch_size': 256},
            source_sha256={}, base_manifest_sha256='base', recent_manifest_sha256='recent',
            semantic_manifest_sha256='semantic', protocol_sha256='protocol', split_sha256='split', objective_weights=self.weights)
        args = SimpleNamespace(seed=42, cache='base', recent='recent', semantics='semantic')
        with patch('pebby.agent.spatial_semantic_outcome_policy.TEACHER_WEIGHTS_SHA256', weights_sha256(teacher.state_dict())):
            result = pilot.checkpoint(parent, model, teacher, 'actor-set', args, report)
        self.assertEqual(result['optimizer_steps'], 400)
        self.assertEqual(result['train_levels'], 9934)
        self.assertFalse(result['actual_outcome_comparator_auxiliary_training'])
        self.assertNotIn('qualification_passed', result)
        self.assertNotIn('quality_filter', result)
        self.assertEqual(result['parent_training_metadata']['quality_filter'], 'old')
        self.assertEqual(result['native_policy_objective'], 'set')

    def test_fixed_protocol_rejects_drift_and_frozen_split_excludes_all_heldout_base_rows(self):
        protocol = json.loads(pilot.PROTOCOL.read_text())
        pilot.validate_protocol(protocol)
        protocol['training']['steps'] = 401
        with self.assertRaises(ValueError):
            pilot.validate_protocol(protocol)
        seeds = np.arange(320)
        tiers = {int(seed): min(7, int(seed) // 46 + 1) for seed in seeds}
        fit, heldout = [], []
        for tier in range(1, 8):
            levels = [seed for seed in range(320) if tiers[seed] == tier]
            cut = round(.8 * len(levels))
            fit.extend(levels[:cut]); heldout.extend(levels[cut:])
        base, recent = {'seeds': np.repeat(seeds, 2)}, {'seeds': seeds}
        split = dict(format='pebby.policy-probe-split.v1', seed=20260913, source_sha256={},
                     fit_seeds=sorted(fit), heldout_seeds=sorted(heldout), heldout_recent_rows=sorted(heldout),
                     heldout_base_rows=np.flatnonzero(np.isin(base['seeds'], heldout)).tolist())
        quality = {'supplement_rows': seeds}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'split.json'
            path.write_text(json.dumps(split))
            result = pilot.load_split(path, base, recent, quality, tiers, {}, pilot.Bindings())
            self.assertEqual(result['heldout_rows'].tolist(), sorted(heldout))
            split['heldout_base_rows'].pop()
            path.write_text(json.dumps(split))
            with self.assertRaisesRegex(ValueError, 'omitted'):
                pilot.load_split(path, base, recent, quality, tiers, {}, pilot.Bindings())


if __name__ == '__main__':
    unittest.main()
