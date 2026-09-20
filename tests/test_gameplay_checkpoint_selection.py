"""Gameplay selects candidates; offline diagnostics cannot select or promote."""
import argparse
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from pebby.agent import data, gameplay_gate, train as trainer
from pebby.agent.competition import CompetitionSession
from tests.test_agent import fake_shard
from tests.test_gameplay_gate import FixedPolicy, generated_level


class GameplayCheckpointSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.reports = {}
        # Real engine reports over generated fixtures, never shipped gameplay.
        for progress in (0, 1, 2, 7):
            session = CompetitionSession([generated_level() for _ in range(7)])
            with patch.object(gameplay_gate.competition, 'CompetitionSession', return_value=session):
                cls.reports[progress] = gameplay_gate.evaluate_sequential(FixedPolicy(progress), 'cpu', per_level_cap=300)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def args(self, epochs, selection='gameplay'):
        return argparse.Namespace(select_on=selection, seed=17, batch_size=2, loader_workers=0,
            architecture='cnn', channels=4, blocks=1, hidden=8, reduce_channels=2,
            broadcast_hud=False, condition_channels=4, lr=.001, weight_decay=.01,
            epochs=epochs, train_min_loops=1, loop_loss='final', loops=1,
            validation_fraction=.2)

    def run_selection(self, progress, diagnostics, selection='gameplay'):
        counter = [0]
        def epoch(model, loader, device, optimizer=None, scheduler=None, **kwargs):
            if optimizer is not None:
                counter[0] += 1
                model.train()
                with torch.no_grad():
                    next(model.parameters()).fill_(counter[0])
            else:
                model.eval()
            ce, accuracy = diagnostics[counter[0] - 1]
            return dict(cross_entropy=ce, accuracy=accuracy, set_accuracy=accuracy, samples=2)
        def gameplay(model, device, *, per_level_cap):
            self.assertFalse(model.training)
            self.assertEqual(device, torch.device('cpu'))
            self.assertEqual(per_level_cap, 300)
            return copy.deepcopy(self.reports[progress[counter[0] - 1]])
        with patch.object(trainer, 'run_epoch', side_effect=epoch), \
                patch.object(trainer, 'evaluate_sequential', side_effect=gameplay) as evaluate, \
                redirect_stdout(io.StringIO()):
            result = trainer.train(fake_shard([1, 2], per_level=2),
                                   fake_shard([1_000_000], per_level=2),
                                   self.args(len(progress), selection), torch.device('cpu'))
        self.assertEqual(evaluate.call_count, len(progress))
        return result

    def test_better_offline_metrics_cannot_select_worse_gameplay(self):
        model, history, baseline, best, *_ = self.run_selection([2, 1], [(2., .1), (.01, 1.)])
        self.assertEqual(best['epoch'], 1)
        self.assertEqual(best['levels_completed'], 2)
        self.assertEqual(float(next(model.parameters()).detach().flatten()[0]), 1.)
        self.assertEqual(best['selected_on'], 'actual_sequential_gameplay')
        self.assertTrue(best['offline_diagnostics_only'])
        self.assertTrue(best['shipped_gameplay_used_for_checkpoint_selection'])
        self.assertFalse(best['promoted'])
        self.assertFalse(best['objective_complete'])

    def test_better_gameplay_wins_despite_worse_offline_scores(self):
        model, _, _, best, *_ = self.run_selection([1, 2], [(.01, 1.), (2., .1)])
        self.assertEqual(best['epoch'], 2)
        self.assertEqual(float(next(model.parameters()).detach().flatten()[0]), 2.)

    def test_tied_gameplay_keeps_earliest_weights_without_offline_tiebreak(self):
        _, _, _, best, *_ = self.run_selection([1, 1], [(2., .1), (.01, 1.)])
        self.assertEqual(best['epoch'], 1)

    def test_zero_gameplay_is_not_success_even_when_offline_prior_is_beaten(self):
        _, history, baseline, best, *_ = self.run_selection([0, 0], [(.01, 1.), (.001, 1.)])
        self.assertEqual(best['epoch'], 1)
        self.assertFalse(best['objective_complete'])
        self.assertFalse(best['promoted'])
        verdict = trainer.verdict(history, baseline, best)
        self.assertTrue(verdict['beats_prior'])
        self.assertTrue(verdict['offline_diagnostic'])
        self.assertFalse(verdict['objective_complete'])
        self.assertFalse(verdict['promoted'])

    def test_last_is_explicit_final_candidate_even_when_gameplay_regresses(self):
        _, _, _, best, *_ = self.run_selection([2, 0], [(.01, 1.), (2., .1)], 'last')
        self.assertEqual(best['epoch'], 2)
        self.assertEqual(best['selected_on'], 'last_epoch')
        self.assertFalse(best['shipped_gameplay_used_for_checkpoint_selection'])
        self.assertFalse(best['objective_complete'])
        self.assertFalse(best['promoted'])

    def test_full_win_is_recorded_without_automatic_promotion(self):
        _, _, _, best, *_ = self.run_selection([7], [(2., .1)])
        self.assertTrue(best['objective_complete'])
        self.assertFalse(best['promoted'])
        self.assertTrue(best['candidate_only'])

    def test_offline_choices_rejected_before_device_or_data_access(self):
        for value in ('cross_entropy', 'accuracy', 'set_accuracy'):
            with self.subTest(value=value), patch.object(sys, 'argv', ['train', '--shards', 'absent', '--select-on', value]), \
                    patch.object(trainer, 'resolve_device') as resolve, \
                    patch.object(data, 'load_shards') as load, redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit):
                    trainer.main()
                self.assertIn('offline checkpoint selection is disabled', stderr.getvalue())
                resolve.assert_not_called()
                load.assert_not_called()
            with self.assertRaises(argparse.ArgumentTypeError):
                trainer.train(None, None, self.args(1, value), torch.device('cpu'))

    def test_incomplete_gameplay_evidence_cannot_select_a_checkpoint(self):
        with patch.object(trainer, 'assess_gameplay', return_value=dict(status='no_evidence', reason='incomplete session')):
            with self.assertRaisesRegex(ValueError, 'complete gameplay evidence'):
                self.run_selection([7], [(.001, 1.)])

    def test_existing_output_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard, output = root / 'shard.npz', root / 'existing.pt'
            shard.write_bytes(b'fixture')
            output.write_bytes(b'previous model')
            with patch.object(sys, 'argv', ['train', '--shards', str(shard), '--checkpoint-out', str(output)]), \
                    patch.object(trainer, 'train') as fit, redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit):
                    trainer.main()
            self.assertIn('refusing to overwrite', stderr.getvalue())
            fit.assert_not_called()
            self.assertEqual(output.read_bytes(), b'previous model')

    def test_cli_default_is_unique_candidate_with_gameplay_metadata(self):
        result = self.run_selection([0], [(.01, 1.)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / 'shard.npz'
            shard.write_bytes(b'fixture')
            fake = fake_shard([1, 2], per_level=2)
            fake['meta'] = [fake['meta']]
            captured = {}
            def fit(shard, held, args, device):
                self.assertEqual(args.select_on, 'gameplay')
                self.assertEqual(args.checkpoint_out.parent, Path('checkpoints/candidates'))
                self.assertTrue(args.checkpoint_out.name.startswith('ls20-looped-'))
                args.checkpoint_out = root / 'candidate.pt'
                return result
            def save(path, model, **metadata):
                captured.update(metadata)
                return dict(format='fixture', config=model.config(), parameters=model.parameter_count(), **metadata)
            report = root / 'report.json'
            with patch.object(sys, 'argv', ['train', '--shards', str(shard), '--device', 'cpu', '--report-out', str(report)]), \
                    patch.object(data, 'load_shards', return_value=fake), patch.object(trainer, 'train', side_effect=fit), \
                    patch.object(trainer, 'save_candidate', side_effect=save), redirect_stdout(io.StringIO()):
                trainer.main()
            self.assertFalse(captured['promoted'])
            self.assertFalse(captured['objective_complete'])
            self.assertTrue(captured['candidate_only'])
            self.assertEqual(captured['gameplay']['levels_completed'], 0)
            self.assertTrue(captured['offline_diagnostics_only'])
            self.assertTrue(captured['shipped_gameplay_used_for_checkpoint_selection'])
            self.assertFalse(json.loads(report.read_text())['promoted'])

    def test_atomic_candidate_publication_cannot_replace_existing_file(self):
        model, *_ = self.run_selection([0], [(.01, 1.)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'existing.pt'
            output.write_bytes(b'previous model')
            with self.assertRaises(FileExistsError):
                trainer.save_candidate(output, model, promoted=False, candidate_only=True)
            self.assertEqual(output.read_bytes(), b'previous model')
            self.assertEqual(list(root.glob('.candidate-*')), [])
            fresh = root / 'new.pt'
            trainer.save_candidate(fresh, model, promoted=False, candidate_only=True)
            restored, metadata = trainer.load_checkpoint(fresh)
            self.assertFalse(metadata['promoted'])
            self.assertEqual(restored.config(), model.config())


if __name__ == '__main__':
    unittest.main()
