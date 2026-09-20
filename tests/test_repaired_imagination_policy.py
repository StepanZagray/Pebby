"""CPU checkpoint-boundary tests with real D and a small public-input encoder."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from pebby.agent import repaired_imagination_policy as repaired
from pebby.agent.neural_imagination_policy import NeuralImaginationPolicy
from pebby.agent.structured_local_glyph import LocalGlobalGlyphTransition


class PublicEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.field = nn.Parameter(torch.randn(148, 96) * .1)

    def metadata(self):
        return dict(history=8, test_encoder=True)

    def forward(self, frames, history_valid=None, previous_actions=None):
        if frames.shape[1:] != (8, 3, 4, 4):
            raise ValueError('public H8 frames required')
        return self.field[None].expand(len(frames), -1, -1) + frames.mean((1, 2, 3, 4))[:, None, None]


class Parent(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PublicEncoder()
        self.dynamics = LocalGlobalGlyphTransition(loops=1, glyph_hidden=8, event_hidden=8)


class RepairedImaginationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.original_path = self.root / 'original.pt'
        self.original_path.write_bytes(b'temporary original checkpoint boundary')
        self.original = NeuralImaginationPolicy(Parent(), dict(horizon=1, hidden=8))
        self.loader = patch.object(repaired, 'load_original_checkpoint',
                                   side_effect=lambda *args: (copy.deepcopy(self.original), {}))
        self.loader.start()
        self.addCleanup(self.loader.stop)
        weights = {key: value.clone() for key, value in self.original.planner.dynamics.state_dict().items()}
        weights['event_head.2.bias'] += .5
        sources = {path: repaired.digest(path) for path in repaired._dynamics_source_paths()}
        sources[str(self.original_path)] = repaired.digest(self.original_path)
        self.source = self.root / 'sequences.npz'
        self.source.write_bytes(b'temporary sequence artifact')
        sources[str(self.source)] = repaired.digest(self.source)
        self.envelope = dict(
            format=repaired.DYNAMICS_FORMAT, config=self.original.planner.dynamics.config(),
            weights=weights, weights_sha256=repaired.state_digest(weights),
            parent=str(self.original_path), parent_sha256=repaired.digest(self.original_path),
            encoder_state_sha256=repaired.state_digest(self.original.encoder.state_dict()),
            official_training_inputs=False, updates=1,
            train_events=dict(lost_life=1, terminal=1, won=1), sources=sources)
        self.dynamics_path = self.root / 'dynamics.pt'
        torch.save(self.envelope, self.dynamics_path)

    def make_policy(self, cls=repaired.RepairedImaginationPolicy):
        parent, metadata = repaired.load_dynamics_parent(
            self.dynamics_path, expected_sha256=repaired.digest(self.dynamics_path))
        json.dumps(metadata)
        return cls(parent, dict(horizon=2, hidden=8))

    def save(self, policy=None):
        path = self.root / 'selector.pt'
        repaired.save_checkpoint(path, policy or self.make_policy(), self.dynamics_path,
                                 dict(official_training_inputs=False, source='generated_only'))
        return path

    def test_repaired_weights_and_public_interface_freeze(self):
        policy = self.make_policy()
        self.assertEqual(repaired.state_digest(policy.planner.dynamics.state_dict()), self.envelope['weights_sha256'])
        self.assertNotEqual(self.envelope['weights_sha256'], repaired.state_digest(self.original.planner.dynamics.state_dict()))
        policy.train()
        self.assertFalse(policy.encoder.training)
        self.assertFalse(policy.planner.dynamics.training)
        self.assertTrue(policy.planner.training)
        frames = torch.randn(2, 8, 3, 4, 4)
        logits = policy(frames, torch.ones(2, 8, dtype=torch.bool), torch.zeros(2, 8, dtype=torch.long))
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertTrue(torch.isfinite(logits).all())
        logits.square().sum().backward()
        self.assertGreater(float(policy.planner.scorer[-1].weight.grad.abs().sum()), 0)
        for module in (policy.encoder, policy.planner.dynamics):
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in module.parameters()))
        with self.assertRaises(TypeError):
            policy(frames, next_fields=frames)

    def test_exact_roundtrip_and_legacy_constructor_compatibility(self):
        policy = self.make_policy(NeuralImaginationPolicy)
        frames = torch.randn(1, 8, 3, 4, 4)
        path = self.save(policy)
        restored, metadata = repaired.load_checkpoint(path)
        self.assertFalse(metadata['official_training_inputs'])
        self.assertEqual(restored.config(), dict(history=8, architecture='structured',
                         decision_architecture='repaired_neural_imagination', horizon=2, hidden=8, heads=4))
        with torch.no_grad():
            torch.testing.assert_close(policy(frames), restored(frames), atol=0, rtol=0)
        with self.assertRaises(FileExistsError):
            repaired.save_checkpoint(path, policy, self.dynamics_path, metadata)

    def test_frozen_weight_and_configuration_mutations_rejected(self):
        for target in ('encoder', 'dynamics', 'dynamics_config', 'trainable'):
            with self.subTest(target=target):
                policy = self.make_policy()
                if target == 'dynamics_config':
                    from dataclasses import replace
                    policy.planner.dynamics.cfg = replace(policy.planner.dynamics.cfg, loops=2)
                elif target == 'trainable':
                    policy.planner.dynamics.requires_grad_(True)
                else:
                    module = policy.encoder if target == 'encoder' else policy.planner.dynamics
                    with torch.no_grad():
                        next(module.parameters()).add_(1)
                with self.assertRaisesRegex(ValueError, 'frozen'):
                    self.save(policy)

    def test_dynamics_provenance_tampering_rejected(self):
        for target in ('encoder', 'config', 'weights', 'missing_weight', 'nonfinite',
                       'sources', 'parent', 'attestation', 'events'):
            with self.subTest(target=target):
                bad = copy.deepcopy(self.envelope)
                if target == 'encoder':
                    bad['encoder_state_sha256'] = '0' * 64
                elif target == 'config':
                    bad['config']['loops'] += 1
                elif target == 'weights':
                    bad['weights']['position'] += 1
                elif target == 'missing_weight':
                    bad['weights'].pop('position')
                    bad['weights_sha256'] = repaired.state_digest(bad['weights'])
                elif target == 'nonfinite':
                    bad['weights']['position'].fill_(float('nan'))
                    bad['weights_sha256'] = repaired.state_digest(bad['weights'])
                elif target == 'sources':
                    bad['sources'].pop(next(iter(repaired._dynamics_source_paths())))
                elif target == 'parent':
                    bad['parent_sha256'] = '0' * 64
                elif target == 'attestation':
                    bad['official_training_inputs'] = True
                else:
                    bad['train_events']['won'] = 0
                torch.save(bad, self.dynamics_path)
                with self.assertRaises((ValueError, RuntimeError)):
                    repaired.load_dynamics_parent(self.dynamics_path)

    def test_bound_sources_and_artifact_checksum_rejected(self):
        with self.assertRaisesRegex(ValueError, 'checksum'):
            repaired.load_dynamics_parent(self.dynamics_path, expected_sha256='0' * 64)
        self.source.write_bytes(b'changed data')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            repaired.load_dynamics_parent(self.dynamics_path)

    def test_selector_provenance_tampering_rejected(self):
        path = self.save()
        original = torch.load(path, weights_only=True)
        for target in ('weights', 'missing_weight', 'extra_dynamics', 'sources', 'encoder',
                       'dynamics', 'original_parent', 'parent', 'attestation'):
            with self.subTest(target=target):
                bad = copy.deepcopy(original)
                if target == 'weights':
                    bad['weights']['position'] += 1
                elif target == 'missing_weight':
                    bad['weights'].pop('position')
                    bad['weights_sha256'] = repaired.state_digest(bad['weights'])
                elif target == 'extra_dynamics':
                    bad['weights']['dynamics.position'] = torch.zeros(148, 96)
                    bad['weights_sha256'] = repaired.state_digest(bad['weights'])
                elif target == 'sources':
                    bad['sources'].pop(str(Path(repaired.__file__).resolve()))
                elif target in ('encoder', 'dynamics'):
                    bad[target + '_state_sha256'] = '0' * 64
                elif target == 'original_parent':
                    bad['original_parent']['sha256'] = '0' * 64
                elif target == 'parent':
                    bad['parent_sha256'] = '0' * 64
                else:
                    bad['metadata']['official_training_inputs'] = True
                torch.save(bad, path)
                with self.assertRaises(ValueError):
                    repaired.load_checkpoint(path)


if __name__ == '__main__':
    unittest.main()
