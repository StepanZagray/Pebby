"""CPU-only contracts for the bounded planning dynamics runner."""
import copy
import hashlib
import json
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools import train_neural_planning_dynamics as trainer


def fixture(count=3, split='train'):
    data = {key:np.zeros((count, *tail), dtype=dtype) for key, (dtype, tail) in trainer.SCHEMA.items()}
    data['seeds'][:] = np.arange(count) + (1_000_000 if split == 'validation' else 0)
    data['difficulties'][:] = 1
    data['actions'][:, :, 0] = np.arange(4)
    data['transition_valid'][:] = True
    data['next_field_valid'][:] = True
    return data


class IdentityModel:
    def eval(self):
        return self

    def rollout(self, field, actions):
        n, horizon = actions.shape
        def logits(classes):
            return torch.zeros(n, horizon, classes)
        return dict(fields=field[:, None].expand(-1, horizon, -1, -1) + 1,
                    readout=dict(player_logits=logits(144), steps_logits=logits(44), lives_logits=logits(4)),
                    glyph_logits={name:logits(classes) for name, classes in [('shape', 6), ('color', 4), ('rotation', 4)]},
                    events={name + '_logits':torch.full((n, horizon), -1.) for name in trainer.EVENTS})


class PlanningRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        dynamics = torch.nn.Linear(1, 1)
        dynamics.config = lambda: dict(loops=1)
        self.policy = SimpleNamespace(encoder=torch.nn.Linear(1, 1), planner=SimpleNamespace(
            dynamics=dynamics, continuation=torch.nn.Linear(1, 1), cfg=SimpleNamespace(horizon=4)))

    def publication(self, data=None):
        data = fixture() if data is None else data
        directory = self.root / 'cache'
        directory.mkdir(exist_ok=True)
        np.savez(directory / 'sequences.npz', **data)
        manifest = dict(format='pebby.neural-planning-sequences.v1', status='complete',
                        source='generated_only', split='train', horizon=4,
                        official_inputs_used=False, sources_unchanged=True,
                        encoder_state_sha256=trainer.state_digest(self.policy.encoder.state_dict()),
                        dynamics_state_sha256=trainer.state_digest(self.policy.planner.dynamics.state_dict()),
                        continuation_state_sha256=trainer.state_digest(self.policy.planner.continuation.state_dict()),
                        dynamics_config=self.policy.planner.dynamics.config(), data_file='sequences.npz',
                        data_sha256=trainer.digest(directory / 'sequences.npz'),
                        arrays={key:dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in data.items()})
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        return directory, manifest

    def test_fixed_shapes_canonical_branches_and_mask_contract(self):
        data = fixture()
        trainer.validate_sequences(data, 'train')
        for kind in ('horizon', 'branches', 'actions', 'resurrection', 'missing_frame', 'unexecuted_event', 'split'):
            with self.subTest(kind=kind):
                bad = copy.deepcopy(data)
                if kind == 'horizon':
                    bad['actions'] = bad['actions'][:, :, :3]
                elif kind == 'branches':
                    bad['actions'] = bad['actions'][:, :3]
                elif kind == 'actions':
                    bad['actions'][0, 0, 0] = 3
                elif kind == 'resurrection':
                    bad['terminal'][0, 0, 1] = True
                elif kind == 'missing_frame':
                    bad['next_field_valid'][0, 0, 1] = False
                elif kind == 'unexecuted_event':
                    bad['transition_valid'][0, 0, -1] = False
                    bad['next_field_valid'][0, 0, -1] = False
                    bad['lost_life'][0, 0, -1] = True
                else:
                    bad['seeds'][0] = 1_000_000
                with self.assertRaises(ValueError):
                    trainer.validate_sequences(bad, 'train')

    def test_terminal_missing_frame_allowed_and_masked_labels_ignored(self):
        data = fixture()
        data['terminal'][0, 0, 1] = True
        data['transition_valid'][0, 0, 2:] = False
        data['next_field_valid'][0, 0, 1:] = False
        data['next_steps'][0, 0, 2:] = 999
        trainer.validate_sequences(data, 'train')
        trainer.evaluate(IdentityModel(), data, 4, 'cpu', lambda:None)

    def test_loading_checks_continuation_binding_duplicate_paths_and_data_hash(self):
        directory, manifest = self.publication()
        loaded, sources = trainer.load_sequences([directory], 'train', self.policy)
        np.testing.assert_array_equal(loaded['actions'], fixture()['actions'])
        self.assertEqual(len(sources), 2)
        with self.assertRaisesRegex(ValueError, 'distinct'):
            trainer.load_sequences([directory, directory], 'train', self.policy)
        manifest['continuation_state_sha256'] = 'wrong'
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'matching'):
            trainer.load_sequences([directory], 'train', self.policy)
        directory, manifest = self.publication()
        with (directory / 'sequences.npz').open('ab') as handle:
            handle.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            trainer.load_sequences([directory], 'train', self.policy)

    def test_manifest_self_consistency_cannot_authorize_wrong_horizon(self):
        data = fixture()
        data['actions'] = data['actions'][:, :, :3]
        directory, _ = self.publication(data)
        with patch.object(trainer.np, 'load') as load:
            with self.assertRaisesRegex(ValueError, 'K4'):
                trainer.load_sequences([directory], 'train', self.policy)
            load.assert_not_called()

    def test_validation_macro_tiers_event_strata_and_identity_baseline(self):
        data = fixture(3, 'validation')
        data['seeds'][:] = [1_000_000, 1_000_000, 1_000_001]
        data['difficulties'][:] = [1, 1, 7]
        data['next_player_cell'][2, ..., 0] = 1
        data['next_fields'][:] = 2
        data['lost_life'][2, :, 0] = True
        report = trainer.evaluate(IdentityModel(), data, 3, 'cpu', lambda:None)
        self.assertAlmostEqual(report['micro_by_horizon'][0]['player_accuracy'], 2/3)
        self.assertEqual(report['level_macro_by_horizon'][0]['player_accuracy'], .5)
        self.assertEqual(report['tier_level_macro_by_horizon']['7'][0]['player_accuracy'], 0)
        strata = report['positive_event_strata']
        self.assertEqual(strata['lost_life'][0]['count'], 4)
        self.assertEqual(strata['lost_life'][0]['player_accuracy'], 0)
        self.assertEqual(strata['after_life_loss'][1]['count'], 4)
        self.assertIsNone(strata['won'][0]['player_accuracy'])
        errors = report['noncarried_field_error_by_horizon'][0]
        self.assertEqual(errors['prediction_mse'], 1)
        self.assertEqual(errors['identity_mse'], 4)
        json.dumps(report, allow_nan=False)

    def test_h1_batch_is_bounded_copied_and_k1(self):
        data = fixture()
        h1 = dict(fields=data['fields'], next_fields=data['next_fields'][:, :, 0],
                  **{key:data[key][:, :, 0] for key in trainer.LABELS})
        h1['fields'][2] = 7
        h1['next_fields'][2, 3] = 9
        fields, items = trainer.h1_tensor_batch(h1, np.array([2]), np.array([3]), 'cpu')
        self.assertEqual(tuple(items['actions'].shape), (1, 1))
        self.assertEqual(tuple(items['next_fields'].shape), (1, 1, 148, 96))
        self.assertEqual(float(fields[0, 0, 0]), 7)
        self.assertEqual(float(items['next_fields'][0, 0, 0, 0]), 9)
        fields.fill_(0)
        self.assertEqual(float(h1['fields'][2, 0, 0]), 7)
        self.assertTrue(items['transition_valid'].all())

    def test_guard_interrupts_hash_before_consuming_entire_file(self):
        path = self.root / 'input'
        path.write_bytes(b'a' * (2 * 1024 * 1024))
        with self.assertRaisesRegex(TimeoutError, 'bounded'):
            trainer.guarded_digest(path, lambda: (_ for _ in ()).throw(TimeoutError('bounded')))
        self.assertEqual(trainer.guarded_digest(path, lambda:None), hashlib.sha256(path.read_bytes()).hexdigest())

    def test_failure_restores_alarm_and_records_complete_settings_before_loading(self):
        parent = self.root / 'parent.pt'
        parent.write_bytes(b'not loaded')
        out = self.root / 'run'
        before = signal.getsignal(signal.SIGALRM)
        with patch.object(trainer, 'load_checkpoint') as load:
            with self.assertRaisesRegex(ValueError, 'checksum'):
                trainer.main(['--parent', str(parent), '--parent-sha256', 'wrong',
                              '--train', str(self.root), '--validation', str(self.root),
                              '--out', str(out), '--device', 'cpu', '--lr', '.002', '--seed', '71'])
            load.assert_not_called()
        report = json.loads((out / 'report.json').read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['settings']['lr'], .002)
        self.assertEqual(report['settings']['seed'], 71)
        self.assertIs(signal.getsignal(signal.SIGALRM), before)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0., 0.))


if __name__ == '__main__':
    unittest.main()
