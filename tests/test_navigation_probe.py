"""CPU experimental contracts on real public-history shapes; no training campaign."""
import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from pebby.agent.navigation_probe import NavigationProbe, make_checkpoint, load_checkpoint, stage_loss
from pebby.agent.neural_outcome_policy import weights_sha256
from pebby.agent.spatial_outcome_planner import SpatialOutcomePlanner
from pebby.agent.world_model import WorldModelConfig, WorldPolicy


def probe(**options):
    encoder = WorldPolicy(WorldModelConfig(channels=16, blocks=1, heads=4, expansion=2, loops=1,
        history=8, temporal_layers=1, hud_channels=8, latent=16, reduce=2,
        predictor_blocks=1, predictor_hidden=16, value_hidden=16, max_distance=16,
        summary=4, readout_hidden=8, ranker_hidden=8, sigreg_projections=8,
        sigreg_knots=3, state_recall=True, glyph_recall=True, query_readout=True))
    planner = SpatialOutcomePlanner(channels=16, width=8, hud_width=8, summary=8, comparator_hidden=8)
    return NavigationProbe(encoder, planner, **options)


def public_inputs():
    return (torch.randint(0, 16, (2, 8, 64, 64)),
            torch.tensor([[False] * 4 + [True] * 4, [True] * 8]),
            torch.tensor([[-1] * 4 + [0, 1, 2, 3], [-1, 0, 1, 2, 3, 0, 1, 2]]))


def primitive_labels():
    return dict(next_player_cell=torch.zeros(2, 4, 2, dtype=torch.long),
                next_triple=torch.zeros(2, 4, 3, dtype=torch.long),
                next_steps=torch.full((2, 4), 10, dtype=torch.long),
                next_lives=torch.full((2, 4), 3, dtype=torch.long),
                distances=torch.tensor([[3, 4, 5, 4], [4, 3, 4, 5]]),
                optimal=torch.tensor([1, 2]),
                lost_life=torch.zeros(2, 4, dtype=torch.bool),
                terminal=torch.zeros(2, 4, dtype=torch.bool),
                won=torch.zeros(2, 4, dtype=torch.bool))


class NavigationProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(81)

    def test_frozen_and_finetune_identical_initial_logits_and_real_updates(self):
        for readout in ('outcomes', 'direct'):
            with self.subTest(readout=readout):
                frozen = probe(readout=readout).train()
                # Reset RNG so the fresh direct head is exactly the same too.
                tuned = NavigationProbe(copy.deepcopy(frozen.encoder), copy.deepcopy(frozen.planner),
                                        encoder_mode='finetune', readout=readout).train()
                if readout == 'direct':
                    tuned.direct_head.load_state_dict(frozen.direct_head.state_dict())
                inputs = public_inputs()
                with torch.no_grad():
                    torch.testing.assert_close(tuned(*inputs), frozen(*inputs), rtol=1e-5, atol=1e-7)
                # Autograd eligibility can change low-level float32 reduction kernels.
                torch.testing.assert_close(tuned(*inputs), frozen(*inputs), rtol=1e-5, atol=1e-7)
                for model in (frozen, tuned):
                    self.assertFalse(model.encoder.training)
                    before = {n: p.detach().clone() for n, p in model.encoder.named_parameters()}
                    optimizer = torch.optim.SGD(model.parameter_groups(.1, .1))
                    optimizer.zero_grad(set_to_none=True)
                    result = stage_loss(model.predict(*inputs), primitive_labels())
                    result['total'].backward()
                    active = dict(model.encoder.named_parameters())
                    if model.encoder_mode == 'finetune':
                        for prefix in ('stem.0.weight', 'core.0', 'player_head.weight', 'glyph_encoder.'):
                            gradients = [p.grad for n, p in active.items() if n.startswith(prefix)]
                            self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in gradients), prefix)
                        self.assertTrue(all(p.grad is None for n, p in active.items() if not p.requires_grad))
                    else:
                        self.assertTrue(all(p.grad is None for p in active.values()))
                    optimizer.step()
                    changed = [n for n, p in active.items() if not torch.equal(before[n], p)]
                    self.assertEqual(bool(changed), model.encoder_mode == 'finetune')
                    if model.encoder_mode == 'finetune':
                        self.assertIn('stem.0.weight', changed)
                        self.assertIn('player_head.weight', changed)
                    self.assertTrue(all(torch.equal(before[n], p) for n, p in active.items() if not p.requires_grad))

    def test_primitive_only_outcome_objective_and_common_policy_ce(self):
        model = probe()
        prediction = model.predict(*public_inputs())
        labels = primitive_labels()
        full = stage_loss(prediction, labels, objective='outcomes')
        self.assertTrue(torch.isfinite(full['total']))
        self.assertEqual(set(full['losses']), {'physical', 'value', 'events', 'policy'})
        self.assertIsNone(full['diagnostics']['won_recall'])
        policy = stage_loss(prediction, labels, objective='policy')
        torch.testing.assert_close(policy['total'], full['losses']['policy'], atol=0, rtol=0)
        full['total'].backward()
        with self.assertRaisesRegex(ValueError, 'requires the outcome'):
            stage_loss(probe(readout='direct').predict(*public_inputs()), labels, objective='outcomes')

    def test_uniform_optimal_masks_and_no_target_rows(self):
        scores = torch.tensor([[1., 2., 3., 4.], [8., 4., 2., 1.]], requires_grad=True)
        result = stage_loss({'action_logits': scores}, {'optimal': torch.tensor([5, 0])})
        expected = -(scores[0].log_softmax(-1)[0] + scores[0].log_softmax(-1)[2]) / 2
        torch.testing.assert_close(result['total'], expected)
        result['total'].backward()
        self.assertEqual(scores.grad[1].abs().sum().item(), 0.)
        result = stage_loss({'action_logits': scores}, {'optimal': torch.tensor([0, 0])})
        self.assertEqual(result['total'].item(), 0.)
        self.assertIsNone(result['diagnostics']['set_accuracy'])
        for mask in (torch.tensor([16, 0]), torch.tensor([1., 0.]), torch.tensor([1])):
            with self.assertRaises(ValueError):
                stage_loss({'action_logits': scores}, {'optimal': mask})

    def test_direct_uses_same_features_and_exact_preoutcome_trunk(self):
        outcome = probe()
        direct = NavigationProbe(copy.deepcopy(outcome.encoder), copy.deepcopy(outcome.planner), readout='direct')
        inputs = public_inputs()
        features = outcome.public_features(*inputs)
        for left, right in zip(features, direct.public_features(*inputs)):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        summaries = []
        hook = outcome.planner.summary_head.register_forward_hook(lambda module, args, output: summaries.append(output.detach()))
        outcome.planner(*features)
        hook.remove()
        captured = []
        hook = direct.planner.summary_head.register_forward_hook(lambda module, args, output: captured.append(output.detach()))
        direct._direct(*features)
        hook.remove()
        torch.testing.assert_close(summaries[0], captured[0], atol=0, rtol=0)
        with patch.object(direct.planner, 'score_outcomes', side_effect=AssertionError('decoded outcomes used')):
            self.assertEqual(direct(*inputs).shape, (2, 4))

    def test_forward_has_only_public_inputs_and_no_legacy_policy_path(self):
        model = probe(encoder_mode='finetune')
        self.assertEqual(list(inspect.signature(model.forward).parameters),
                         ['frames', 'history_valid', 'previous_actions'])
        self.assertEqual(list(inspect.signature(model.predict).parameters),
                         ['frames', 'history_valid', 'previous_actions'])
        with patch.object(model.encoder, 'direct_logits', side_effect=AssertionError('legacy policy')):
            model(*public_inputs())
        with self.assertRaises(TypeError):
            model(*public_inputs(), distances=torch.zeros(2, 4))

    def test_parameter_groups_audit_and_parent_copy(self):
        parent = probe()
        tuned = NavigationProbe.from_parent(parent, encoder_mode='finetune', readout='direct')
        self.assertTrue(all(not p.requires_grad for p in parent.encoder.parameters()))
        groups = tuned.parameter_groups(.001, .01)
        self.assertEqual([g['name'] for g in groups], ['encoder', 'controller'])
        grouped = [id(p) for group in groups for p in group['params']]
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(set(grouped), {id(p) for p in tuned.parameters() if p.requires_grad})
        audit = tuned.parameter_audit()
        json.dumps(audit, allow_nan=False)
        for prefix in ('encoder.projector.', 'encoder.reduce.', 'encoder.predictor.',
                       'encoder.move_head.', 'encoder.query_head.', 'planner.field_heads.',
                       'planner.value_head.', 'planner.event_head.', 'planner.comparator.'):
            self.assertFalse(any(n.startswith(prefix) for n in audit['trainable']))
        self.assertIn('encoder.player_head.bias', audit['frozen'])

    def test_exact_checkpoint_roundtrip_every_mode(self):
        inputs = public_inputs()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'probe.pt'
            for mode in ('frozen', 'finetune'):
                for readout in ('outcomes', 'direct'):
                    with self.subTest(mode=mode, readout=readout):
                        model = probe(encoder_mode=mode, readout=readout).eval()
                        data = make_checkpoint(model, {'seed': 81, 'objective': 'policy'})
                        torch.save(data, path)
                        loaded, metadata = load_checkpoint(path)
                        self.assertEqual(metadata, {'seed': 81, 'objective': 'policy'})
                        self.assertEqual((loaded.encoder_mode, loaded.readout), (mode, readout))
                        self.assertEqual(loaded.parameter_audit(), model.parameter_audit())
                        with torch.no_grad():
                            torch.testing.assert_close(loaded(*inputs), model(*inputs), atol=0, rtol=0)
                        # Snapshot storage is detached from the living model.
                        name, weight = next(iter(model.named_parameters()))
                        old = data['weights'][name].clone()
                        with torch.no_grad():
                            weight.add_(1)
                        torch.testing.assert_close(old, data['weights'][name], atol=0, rtol=0)

    def test_checkpoint_malformed_schema_digest_config_and_modes_rejected(self):
        base = make_checkpoint(probe(), {})
        changes = [dict(format='production'), dict(extra=True), dict(encoder_mode='unknown'),
                   dict(readout='oracle'), dict(weights_sha256='0' * 64), dict(encoder_runtime={}),
                   dict(metadata=[]), dict(encoder_config={**base['encoder_config'], 'channels': 8}),
                   dict(planner_config={**base['planner_config'], 'width': 0})]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.pt'
            for change in changes:
                with self.subTest(change=change), self.assertRaises(ValueError):
                    torch.save({**base, **change}, path)
                    load_checkpoint(path)
            for recompute_digest in (False, True):
                damaged = copy.deepcopy(base)
                damaged['weights'].pop(next(iter(damaged['weights'])))
                if recompute_digest:
                    damaged['weights_sha256'] = weights_sha256(damaged['weights'])
                torch.save(damaged, path)
                with self.assertRaises(ValueError):
                    load_checkpoint(path)


if __name__ == '__main__':
    unittest.main()
