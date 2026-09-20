"""Training wrapper isolation, deployment compatibility, and fail-closed gates."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from pebby.agent import world_train
from pebby.agent.world_model import load_world_checkpoint
from tests.test_world_goal_objective import PixelTeacher
from tests.test_world_model import make_model
from tools import train_reference_goals as runner


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_optimizer_clipping_sidecar_and_public_forward_roundtrip(self):
        model = make_model(channels=64).eval()
        original_keys = list(model.state_dict())
        frames = torch.zeros(1, 4, 64, 64, dtype=torch.long)
        expected = model(frames)
        observed = {}

        def epoch(model, tensors, device, weights, batch_size, optimizer, *args, **kwargs):
            optimizer.zero_grad()
            # Strong head gradients exercise the one joint clipping bound.
            loss = sum(p.sum() for group in optimizer.param_groups for p in group['params'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            all_params = [p for group in optimizer.param_groups for p in group['params']]
            observed['joint_norm'] = torch.stack([p.grad.norm() ** 2 for p in all_params]).sum().sqrt().item()
            observed['head_gradient'] = aux.head[0].weight.grad.abs().sum().item()
            return {'complete': True}

        with tempfile.TemporaryDirectory() as directory, patch.object(world_train, 'run_epoch', epoch):
            with runner.objective_context(PixelTeacher(), {42: False}, {}) as aux:
                with patch.object(model, 'parameter_count', return_value=1198165):
                    groups = world_train.parameter_groups(model, .01)
                optimizer = torch.optim.AdamW(groups, lr=.0001)
                self.assertFalse(optimizer.state)
                params = [p for group in groups for p in group['params']]
                self.assertEqual(len(params), len({id(p) for p in params}))
                self.assertTrue(all(any(p is h for p in params) for h in aux.head.parameters()))
                self.assertEqual(list(model.state_dict()), original_keys)
                self.assertTrue(torch.equal(model(frames), expected))
                world_train.run_epoch(model, {}, torch.device('cpu'), {}, 1, optimizer)
                self.assertLessEqual(observed['joint_norm'], 1.00001)
                self.assertGreater(observed['head_gradient'], 0)
                path = Path(directory) / 'model.pt'
                world_train.save_world_checkpoint(path, model)
                restored, checkpoint = load_world_checkpoint(path)
                self.assertTrue(torch.equal(restored.eval()(frames), expected))
                sidecar = checkpoint['goal_head_sidecar']
                self.assertEqual(sidecar['sha256'], runner.repair.digest(sidecar['path']))
                self.assertEqual(sidecar['parameters'], 10126)
                saved = torch.load(sidecar['path'], weights_only=True)
                for key, value in aux.head.state_dict().items():
                    self.assertTrue(torch.equal(saved['weights'][key], value))
                self.assertFalse(any('goal' in key for key in checkpoint['weights']))

    def test_all_patches_restore_after_failure(self):
        names = ('world_losses', 'training_objective_source', 'parameter_groups', 'as_tensors',
                 'run_epoch', 'save_world_checkpoint')
        before = {name: getattr(world_train, name) for name in names}
        guard, write = runner.repair.source_guard, runner.repair.write
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with runner.objective_context(PixelTeacher(), {}, {}):
                raise RuntimeError('injected')
        for name, value in before.items():
            self.assertIs(getattr(world_train, name), value)
        self.assertIs(runner.repair.source_guard, guard)
        self.assertIs(runner.repair.write, write)

    def test_annotation_added_after_control_batch_hash_without_rng_change(self):
        model = make_model(channels=64)
        observed = []
        sampler = type('Sampler', (), {'last_level_seeds': (7, 42)})()
        base_batch = {'frames': torch.zeros(2, 4, 64, 64, dtype=torch.long)}

        def recorded(*args, **kwargs):
            observed.append(tuple(base_batch))
            yield base_batch

        def epoch(*args, **kwargs):
            before = torch.random.get_rng_state().clone()
            batch = next(world_train.curriculum_batches({}, sampler))
            self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
            self.assertEqual(batch['goal_seeds'].tolist(), [7, 42])
            self.assertEqual(tuple(base_batch), ('frames',))

        with patch.object(world_train, 'run_epoch', epoch), \
                patch.object(world_train, 'curriculum_batches', recorded):
            with runner.objective_context(PixelTeacher(), {7: False, 42: True}, {}) as aux:
                with patch.object(model, 'parameter_count', return_value=1198165):
                    groups = world_train.parameter_groups(model, .01)
                world_train.run_epoch(model, {}, torch.device('cpu'), {}, 2,
                                      torch.optim.AdamW(groups, lr=.0001))
        self.assertEqual(observed, [('frames',)])

    def test_missing_approval_changed_teacher_and_parent_fail_before_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher, audit, approval, parent = [root / name for name in ('teacher.pt', 'audit.json', 'approval.json', 'parent.pt')]
            teacher.write_bytes(b'not a checkpoint')
            audit.write_text(json.dumps({'status': 'complete'}))
            parent.write_bytes(b'wrong parent')
            approval.write_text('{}')
            with patch.object(torch, 'load') as load:
                with self.assertRaisesRegex(ValueError, 'root-approved'):
                    runner.validate_gate(teacher, audit, approval)
                accepted = dict(format='pebby.goal-teacher-approval.v1', approved=True,
                                scope='goal_preservation_training', dynamic_audit_accepted=True,
                                audit_sha256=runner.repair.digest(audit), teacher_sha256=runner.repair.digest(teacher),
                                parent_sha256=runner.repair.PARENT_SHA)
                approval.write_text(json.dumps(accepted))
                teacher.write_bytes(b'changed teacher')
                with self.assertRaisesRegex(ValueError, 'root-approved'):
                    runner.validate_gate(teacher, audit, approval)
                accepted['teacher_sha256'] = runner.repair.digest(teacher)
                approval.write_text(json.dumps(accepted))
                with patch.object(runner.repair, 'PARENT', parent):
                    with self.assertRaisesRegex(ValueError, 'parent checkpoint'):
                        runner.validate_gate(teacher, audit, approval)
                load.assert_not_called()
