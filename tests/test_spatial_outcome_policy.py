"""CPU wrapper contracts with synthetic frames and randomly initialized weights.

Checkpoint lineage strings below are format-test fixtures, not claims that the
synthetic encoder is the protected trained parent. No game environments are used.
"""

import copy
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from pebby.agent.neural_outcome_policy import ENCODER_RUNTIME, PARENT_SHA, weights_sha256
from pebby.agent.spatial_outcome_planner import SpatialOutcomePlanner
from pebby.agent.spatial_outcome_policy import FORMAT, SpatialOutcomePolicy, load_checkpoint
from pebby.agent.world_model import WorldModelConfig, WorldPolicy


def make_policy(*, direct_weight=0., planner_weight=1.):
    encoder = WorldPolicy(WorldModelConfig(channels=16, blocks=1, heads=4, expansion=2, loops=1,
        history=8, temporal_layers=1, hud_channels=8, hud_tokens=16, latent=16, reduce=2,
        predictor_blocks=1, predictor_hidden=16, value_hidden=16, max_distance=16,
        lookahead_depth=1, summary=4, readout_hidden=8, ranker_hidden=8,
        sigreg_projections=8, sigreg_knots=3, state_recall=True, glyph_recall=True, query_readout=True))
    planner = SpatialOutcomePlanner(channels=16, width=8, hud_width=8, summary=8, comparator_hidden=8)
    return SpatialOutcomePolicy(encoder, planner, direct_weight=direct_weight, planner_weight=planner_weight).eval()


def public_inputs():
    return (torch.randint(0, 16, (2, 8, 64, 64)), torch.tensor([[False] * 4 + [True] * 4, [True] * 8]),
            torch.tensor([[-1] * 4 + [0, 1, 2, 3], [-1, 0, 1, 2, 3, 0, 1, 2]]))


def checkpoint(policy):
    weights = {key: value.clone() for key, value in policy.encoder.state_dict().items()}
    return dict(format=FORMAT, encoder_parent_sha256=PARENT_SHA, encoder_frozen=True,
                official_training_inputs=False, encoder_runtime=dict(ENCODER_RUNTIME),
                encoder_weights=weights, encoder_weights_sha256=weights_sha256(weights),
                encoder_config=policy.encoder.config(), planner_config=policy.planner.config(),
                planner_weights=policy.planner.state_dict(),
                score_weights=dict(direct=policy.direct_weight, planner=policy.planner_weight))


class SpatialOutcomePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)
        if torch.cuda.is_initialized():
            raise AssertionError('CPU wrapper tests must not initialize CUDA')

    def setUp(self):
        torch.manual_seed(73)

    def test_cached_current_player_linear_probabilities_equal_encoder_head(self):
        policy = make_policy()
        with torch.no_grad():
            encoded = policy.encoder.encode(*public_inputs())
            # Clone simulates the ordinary copied float32 state cache batch.
            current_cells = encoded['state'][:, :144].clone()
            actual = policy.encoder.player_weights(encoded['cells'])[1]
            cached = F.linear(current_cells, policy.encoder.player_head.weight,
                              policy.encoder.player_head.bias).squeeze(-1).softmax(-1)
        torch.testing.assert_close(cached, policy.encoder.player_weights(current_cells)[1], atol=0, rtol=0)
        # Cloning the strided cell view selects a different CPU matrix kernel;
        # its FP32 reduction differs by a few ULPs (observed max abs 3.73e-9).
        torch.testing.assert_close(cached, actual, atol=1e-8, rtol=1e-6)
        torch.testing.assert_close(cached.sum(-1), torch.ones(2))

    def test_public_forward_equals_manual_current_encoding_and_planner(self):
        for direct_weight, planner_weight in ((0., 1.), (.25, .75)):
            with self.subTest(direct_weight=direct_weight):
                policy, inputs = make_policy(direct_weight=direct_weight, planner_weight=planner_weight), public_inputs()
                with torch.no_grad():
                    encoding = policy.encoder.encode(*inputs)
                    probabilities = policy.encoder.player_weights(encoding['cells'])[1]
                    expected = planner_weight * policy.planner(encoding['raw'], encoding['state'],
                                                               encoding['glyph'], probabilities)['action_logits']
                    if direct_weight:
                        direct = policy.encoder.direct_logits(encoding['cells'])[0]
                        direct = direct + policy.encoder.query_logits(encoding, probabilities)
                        expected = expected + direct_weight * direct
                    actual = policy(*inputs)
                self.assertEqual(actual.shape, (2, 4))
                self.assertEqual(actual.dtype, torch.float32)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                self.assertEqual(policy.config()['history'], 8)
                self.assertEqual(policy.config()['decision_architecture'], 'spatial_outcomes')

    def test_training_mode_keeps_encoder_eval_and_without_gradients(self):
        policy = make_policy().train()
        self.assertTrue(policy.planner.training)
        self.assertFalse(policy.encoder.training)
        self.assertTrue(all(not p.requires_grad for p in policy.encoder.parameters()))
        values = policy(*public_inputs())
        values.square().sum().backward()
        self.assertTrue(all(p.grad is None for p in policy.encoder.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.planner.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.parameters()))

    def test_strict_spatial_round_trip_preserves_scores_and_frozen_encoder(self):
        policy, inputs = make_policy(direct_weight=.2, planner_weight=.8), public_inputs()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.pt'
            torch.save(checkpoint(policy), path)
            loaded, metadata = load_checkpoint(path, 'cpu')
            with torch.no_grad():
                torch.testing.assert_close(loaded(*inputs), policy(*inputs), atol=0, rtol=0)
            self.assertEqual(metadata['format'], FORMAT)
            self.assertEqual(loaded.direct_weight, .2)
            self.assertEqual(loaded.planner_weight, .8)
            self.assertTrue(all(not p.requires_grad for p in loaded.encoder.parameters()))
            self.assertFalse(loaded.encoder.training)

    def test_loader_rejects_wrong_format_lineage_runtime_and_encoder_digest(self):
        base = checkpoint(make_policy())
        changes = (dict(format='pebby.ls20-neural-outcome-policy.v1'),
                   dict(encoder_parent_sha256='0' * 64), dict(encoder_frozen=False),
                   dict(official_training_inputs=True), dict(encoder_weights_sha256='0' * 64),
                   dict(encoder_runtime={**ENCODER_RUNTIME, 'temporal_backend': 'math'}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.pt'
            for change in changes:
                with self.subTest(change=change), self.assertRaises(ValueError):
                    torch.save({**base, **change}, path)
                    load_checkpoint(path, 'cpu')
            corrupt = copy.deepcopy(base)
            next(iter(corrupt['encoder_weights'].values())).add_(1)
            torch.save(corrupt, path)
            with self.assertRaisesRegex(ValueError, 'digest'):
                load_checkpoint(path, 'cpu')
            missing = copy.deepcopy(base)
            missing['planner_weights'].pop(next(iter(missing['planner_weights'])))
            torch.save(missing, path)
            with self.assertRaises(RuntimeError):
                load_checkpoint(path, 'cpu')

    def test_evaluator_exact_hash_gate_precedes_any_model_or_game(self):
        from tools.evaluate_reference_spatial_outcomes import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'model').write_bytes(b'not a checkpoint')
            with patch('pebby.agent.spatial_outcome_policy.load_checkpoint') as loader:
                with self.assertRaisesRegex(ValueError, 'hash differs'):
                    main(['--checkpoint', str(root / 'model'), '--checkpoint-sha256', '0' * 64,
                          '--bank', str(root / 'bank'), '--bank-sha256', '1' * 64,
                          '--report-out', str(root / 'report.json'), '--device', 'cpu'])
                loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
