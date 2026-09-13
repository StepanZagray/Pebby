"""Focused CPU checks for outcome training admission and policy serialization."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pebby.agent.neural_outcome_planner import NeuralOutcomePlanner
from pebby.agent.neural_outcome_policy import (NeuralOutcomePolicy, FORMAT, PARENT_SHA,
                                               ENCODER_RUNTIME, weights_sha256, load_checkpoint)
from pebby.agent.world_model import WorldPolicy, WorldModelConfig
from tools import train_reference_outcomes as trainer


class OutcomeTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_perfect_policy_does_not_admit_bad_outcomes(self):
        metrics = {name + '_accuracy': torch.tensor(.95) for name in
                   ('player', 'shape', 'color', 'rotation', 'steps', 'lives')}
        metrics.update(set_accuracy=torch.tensor(1.), value_accuracy=torch.tensor(.9))
        record = dict(diagnostics=metrics, losses={'events': torch.tensor(.1)}, total=torch.tensor(1.))
        self.assertTrue(trainer.tiny_gate(record))
        for name in ('player_accuracy', 'rotation_accuracy', 'value_accuracy'):
            bad = {**record, 'diagnostics': {**metrics, name: torch.tensor(.25)}}
            self.assertFalse(trainer.tiny_gate(bad))
        self.assertFalse(trainer.tiny_gate({**record, 'losses': {'events': torch.tensor(.8)}}))
        rare_missed = {**metrics, 'terminal_support': torch.tensor(4), 'terminal_recall': torch.tensor(0.)}
        self.assertFalse(trainer.tiny_gate({**record, 'diagnostics': rare_missed}))
        rare_learned = {**rare_missed, 'terminal_recall': torch.tensor(1.)}
        self.assertTrue(trainer.tiny_gate({**record, 'diagnostics': rare_learned}))

    def test_count_aggregation_and_eval_mode_are_preserved(self):
        model = torch.nn.Linear(1, 1).eval()
        records = [dict(diagnostics={'policy_valid_count': torch.tensor(n),
                                    'lost_life_support': torch.tensor(n),
                                    'set_accuracy': torch.tensor(acc)},
                        diagnostic_weights={'policy_valid_count': 1, 'lost_life_support': 1,
                                            'set_accuracy': n}) for n, acc in [(2, .5), (1, 1.)]]
        with patch.object(trainer, 'batch', return_value={}), patch.object(trainer, 'forward', side_effect=records):
            metrics, support = trainer.evaluate(model, {'seeds': np.arange(3)}, 'cpu', 2)
        self.assertEqual(metrics['policy_valid_count'], 3)
        self.assertEqual(metrics['lost_life_support'], 3)
        self.assertAlmostEqual(metrics['set_accuracy'], 2/3)
        self.assertFalse(model.training)

    def test_cache_loader_uses_single_owning_validator_with_dtype_strings(self):
        from tools import cache_reference_outcome_inputs as cache
        settings = dict(precision='float32', execution='native_eager', temporal_backend='auto',
                        matmul_tf32=False, cudnn_tf32=True, encoder_chunk_size=0)
        manifest = dict(settings=settings, validated_output_hashes={'x': 'digest'}, validated_output_stats={'x': [1]*5})
        with patch.object(cache, 'validate_published', return_value=manifest) as validate, \
             patch.object(trainer.np, 'load', return_value=np.zeros(1, dtype='<f4')), \
             patch.object(trainer, 'sha', side_effect=AssertionError('double hash')):
            arrays, hashes, stats = trainer.load_data(Path('/unused'))
        validate.assert_called_once_with(Path('/unused'))
        self.assertEqual(arrays['train']['raw'].dtype.str, '<f4')
        self.assertEqual(hashes, {'x': 'digest'})

    def test_wrapper_roundtrip_frozen_encoder_and_weight_integrity(self):
        encoder = WorldPolicy(WorldModelConfig(channels=16, heads=2, hud_channels=16,
            hud_tokens=4, latent=16, reduce=2, blocks=1, loops=1, history=1))
        planner = NeuralOutcomePlanner(channels=16, heads=2, context_tokens=148, glyph_inputs=0, comparator_hidden=16)
        policy = NeuralOutcomePolicy(encoder, planner).train()
        self.assertFalse(policy.encoder.training)
        frames = torch.zeros(2, 1, 64, 64, dtype=torch.long)
        scores = policy(frames)
        scores.square().sum().backward()
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in encoder.parameters()))
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in planner.parameters()))
        checkpoint = dict(format=FORMAT, encoder_config=encoder.config(), encoder_weights=encoder.state_dict(),
            planner_config=planner.config(), planner_weights=planner.state_dict(), score_weights={'planner':1., 'direct':0.},
            encoder_parent_sha256=PARENT_SHA, encoder_frozen=True, encoder_runtime=ENCODER_RUNTIME,
            encoder_weights_sha256=weights_sha256(encoder.state_dict()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            torch.save(checkpoint, path)
            loaded, _ = load_checkpoint(path)
            torch.testing.assert_close(loaded(frames), policy.eval()(frames), rtol=1e-5, atol=1e-6)
            checkpoint['encoder_weights_sha256'] = 'wrong'
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                load_checkpoint(path)
        opt = trainer.optimizer_for(planner, .001)
        covered = [id(p) for group in opt.param_groups for p in group['params']]
        self.assertEqual(len(covered), len(set(covered)))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
