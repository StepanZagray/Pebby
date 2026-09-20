from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pebby.agent import gameplay_gate, world_train as trainer, world_model
from pebby.agent.competition import CompetitionSession
from tests.test_gameplay_gate import FixedPolicy, generated_level
from tests.test_world_model import TINY, make_model, make_synthetic


class WorldGameplaySelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.gameplay = {}
        for count in (1, 2):
            session = CompetitionSession([generated_level() for _ in range(7)])
            with patch.object(gameplay_gate.competition, 'CompetitionSession', return_value=session):
                cls.gameplay[count] = gameplay_gate.evaluate_sequential(FixedPolicy(count), 'cpu', per_level_cap=3)

    def flags(self, root, epochs=3):
        np.savez(root / 'train.npz', **make_synthetic(seed=21, levels=2, steps=2, history=4))
        flags = ['--train', str(root / 'train.npz'), '--epochs', str(epochs), '--batch-size', '2',
                 '--device', 'cpu', '--checkpoint-out', str(root / 'candidate.pt'),
                 '--temporal-backend', 'math']
        for key, value in TINY.items():
            flags.extend(['--' + key.replace('_', '-'), str(value)])
        return flags

    def fake_epoch(self):
        calls = []

        def run(model, *args, **kwargs):
            calls.append(True)
            with torch.no_grad():
                next(model.parameters()).fill_(len(calls))
            return dict(total=1. / len(calls), policy=1. / len(calls),
                        set_accuracy=len(calls) / 3, prediction=.1, copy_mse=.2,
                        counterfactual_top1=.25, copy_top1=.25, sigreg=.1,
                        target_variance_mean=.1, target_variance_min=.01,
                        value=.1, imagined_value=.1)
        return run

    def test_default_actual_gameplay_selects_weights_and_keeps_earlier_ties(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            flags = self.flags(root)
            reports = [copy.deepcopy(self.gameplay[count]) for count in (2, 1, 2)]
            with patch.object(trainer, 'run_epoch', side_effect=self.fake_epoch()):
                with patch.object(trainer, 'evaluate_sequential', side_effect=reports) as evaluate:
                    with redirect_stdout(io.StringIO()):
                        self.assertEqual(trainer.main(flags), 0)
            self.assertEqual(evaluate.call_count, 3)
            self.assertTrue(all(call.kwargs['per_level_cap'] == 300 for call in evaluate.call_args_list))
            report = json.loads((root / 'candidate.training.json').read_text())
            self.assertEqual(report['best']['epoch'], 1)
            self.assertEqual(report['best']['criterion'], 'gameplay')
            self.assertEqual(report['best']['selected_on'], 'actual_sequential_gameplay')
            self.assertEqual(report['history'][1]['gameplay_assessment']['status'], 'reject')
            self.assertEqual(report['history'][2]['gameplay_assessment']['status'], 'tied')
            self.assertTrue(report['shipped_gameplay_used_for_checkpoint_selection'])
            self.assertTrue(report['repeated_target_exposure'])
            self.assertFalse(report['untouched_test_or_generalization_claim'])
            selected, checkpoint = world_model.load_world_checkpoint(root / 'candidate.pt')
            final, last_metadata = world_model.load_world_checkpoint(root / 'candidate.last.pt')
            self.assertTrue(torch.equal(next(selected.parameters()), torch.ones_like(next(selected.parameters()))))
            self.assertTrue(torch.equal(next(final.parameters()), torch.full_like(next(final.parameters()), 3)))
            self.assertEqual(checkpoint['gameplay']['policy_weights_sha256'], trainer.initial_state_sha256(selected))
            for metadata in (report, checkpoint, last_metadata):
                self.assertFalse(metadata['promoted'])
                self.assertTrue(metadata['candidate_only'])
            self.assertFalse(last_metadata['shipped_gameplay_used_for_checkpoint_selection'])

    def test_explicit_last_artifact_skips_gameplay_and_reports_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(trainer, 'run_epoch', side_effect=self.fake_epoch()):
                with patch.object(trainer, 'evaluate_sequential') as evaluate:
                    with redirect_stdout(io.StringIO()):
                        self.assertEqual(trainer.main(self.flags(root, 2) + ['--select-on', 'last']), 0)
            evaluate.assert_not_called()
            for path in (root / 'candidate.pt', root / 'candidate.last.pt', root / 'candidate.running.pt'):
                _, metadata = world_model.load_world_checkpoint(path)
                self.assertEqual(metadata['gameplay_status'], 'gameplay_not_evaluated')
                self.assertIsNone(metadata['gameplay'])
                self.assertFalse(metadata['promoted'])
                self.assertFalse(metadata['shipped_gameplay_used_for_checkpoint_selection'])
            report = json.loads((root / 'candidate.training.json').read_text())
            self.assertEqual(report['best']['epoch'], 2)
            self.assertEqual(report['gameplay_status'], 'gameplay_not_evaluated')

    def test_offline_flags_and_existing_outputs_fail_before_compute(self):
        for criterion in ('set_accuracy', 'policy_cross_entropy', 'total'):
            with patch.object(trainer, 'resolve_device') as device, patch.object(trainer, 'load_dataset') as data:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    trainer.main(['--train', 'missing.npz', '--select-on', criterion])
                device.assert_not_called()
                data.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'offline'):
                trainer.selection_score({'set_accuracy': 1., 'policy': 0.}, criterion)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / 'candidate.pt'
            candidate.write_bytes(b'keep existing checkpoint')
            with patch.object(trainer, 'resolve_device') as device, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    trainer.main(['--train', 'missing.npz', '--checkpoint-out', str(candidate)])
                device.assert_not_called()
            model = make_model()
            rng_before = torch.random.get_rng_state().clone()
            with self.assertRaises(FileExistsError):
                trainer.save_candidate(candidate, model, promoted=False)
            torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
            self.assertEqual(candidate.read_bytes(), b'keep existing checkpoint')
            self.assertEqual(list(root.iterdir()), [candidate])

    def test_protocol_mismatch_and_interrupted_gameplay_cannot_select(self):
        first = dict(epoch=1, gameplay=copy.deepcopy(self.gameplay[1]))
        best = trainer.candidate_selection(first, None, 'gameplay')
        invalid = copy.deepcopy(self.gameplay[2])
        invalid['ending'] = 'interrupted'
        with self.assertRaisesRegex(ValueError, 'evidence'):
            trainer.candidate_selection(dict(epoch=2, gameplay=invalid), best, 'gameplay')
        changed = copy.deepcopy(self.gameplay[2])
        changed['per_level_caps'] = [4] * 7
        changed['protocol_identity'] = gameplay_gate._identity(changed, changed['source_bindings'])
        with self.assertRaisesRegex(ValueError, 'protocols differ'):
            trainer.candidate_selection(dict(epoch=2, gameplay=changed), best, 'gameplay')
