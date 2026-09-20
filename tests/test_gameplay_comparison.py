from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import torch

from pebby.agent.competition import CompetitionSession
from tools import evaluate_gameplay_comparison as comparison
from tests.test_gameplay_gate import generated_level


def spec(seed=1_000_000, tier=1):
    return dict(seed=seed, source='generated_only', difficulty=tier,
                difficulty_version='ls20-reference-v1', context_index=tier - 1,
                geometry_split='validation', walls=[(1, 3), (0, 2), (1, 1)], start=(1, 2),
                start_triple=(0, 0, 0), goals=[dict(cell=(2, 2), triple=(0, 0, 0))],
                cyclers=[], refills=[], launchers=[], step_counter=2, step_cost=1, fog=False,
                context_solution=[1])  # Deliberately wrong stored route must never be used.


class Policy(torch.nn.Module):
    def __init__(self, direction=3):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(4))
        with torch.no_grad():
            self.logits[direction] = 1

    def config(self):
        return dict(architecture='test_pixels')

    def forward(self, frames):
        return self.logits[None].expand(len(frames), -1)


class GameplayComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def bank(self, root):
        path = root / 'bank.jsonl'
        specs = [spec(1_000_000 + (tier - 1) * 10 + index, tier)
                 for tier in range(1, 8) for index in range(10)]
        path.write_text(''.join(json.dumps(row) + '\n' for row in specs))
        return path

    def args(self, root, bank=None):
        checkpoint = root / 'fixture.pt'
        checkpoint.write_bytes(b'fixture file: loader is patched in tests')
        args = ['--checkpoint', 'baseline', str(checkpoint), comparison.digest(checkpoint),
                '--checkpoint', 'candidate', str(checkpoint), comparison.digest(checkpoint),
                '--report-out', str(root / 'report.json'), '--max-seconds', '30']
        return args + (['--sequential-only'] if bank is None else
                       ['--bank', str(bank), '--bank-sha256', comparison.digest(bank)])

    def test_generated_actual_engine_strict_argmax_and_no_route_inputs(self):
        win = comparison.generated_rollout(Policy(3), spec(), lambda: None)
        loss = comparison.generated_rollout(Policy(1), spec(), lambda: None)
        self.assertTrue(win['completed'])
        self.assertEqual((win['actions'], win['lives_left']), (1, 3))
        self.assertEqual((loss['ending'], loss['actions'], loss['lives_left']), ('game_over', 9, 0))
        self.assertEqual(loss['on_stall'], 'repeat')
        self.assertEqual(loss['temperature'], 0.)
        self.assertIsNone(win['optimal'])

    def test_bank_rejects_wrong_context_training_or_unbalanced_levels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.bank(root)
            rows = comparison.checked_specs(path)
            self.assertEqual([comparison.generated_context(row) for row in rows[::10]], list(range(7)))
            for update in (dict(context_index=6), dict(geometry_split='train'), dict(source='shipped')):
                changed = copy.deepcopy(rows)
                changed[0].update(update)
                path.write_text(''.join(json.dumps(row) + '\n' for row in changed))
                with self.assertRaises(ValueError):
                    comparison.checked_specs(path)
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows[:-1]))
            with self.assertRaises(ValueError):
                comparison.checked_specs(path)

    def test_full_comparison_plays_same70_generated_games_and_pairs_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.args(root, self.bank(root))
            sessions = [CompetitionSession([generated_level() for _ in range(7)]) for _ in range(2)]
            # Both sequential sessions and generated episodes execute the actual engine,
            # but only synthetic fixture levels enter tests.
            with patch.object(comparison, 'load_policy', side_effect=[(Policy(3), {'format': 'test'}),
                                                                     (Policy(1), {'format': 'test'})]):
                with patch.object(comparison.gameplay_gate.competition, 'CompetitionSession', side_effect=sessions):
                    with patch.object(comparison, 'memory_guard', return_value=9 * 2**30):
                        with redirect_stdout(io.StringIO()):
                            report = comparison.main(args)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['arms']['baseline']['sequential']['levels_completed'], 7)
            self.assertEqual(report['arms']['candidate']['sequential']['levels_completed'], 0)
            self.assertEqual(report['arms']['baseline']['generated_summary']['completed'], 70)
            self.assertEqual(report['arms']['candidate']['generated_summary']['failures'], 70)
            self.assertEqual(report['paired']['candidate']['paired_levels'], 70)
            self.assertEqual(report['paired']['candidate']['net_wins'], -70)
            self.assertEqual(report['gameplay_gate']['candidate']['status'], 'reject')
            self.assertFalse(report['promoted'])
            self.assertFalse(report['cuda_initialized'])
            self.assertEqual(len(report['arms']['candidate']['per_tier']['7']['failure_seeds']), 10)
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0)

    def test_interrupted_generated_seed_is_not_a_loss_or_paired_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.args(root, self.bank(root))
            session = CompetitionSession([generated_level() for _ in range(7)])
            real_rollout = comparison.generated_rollout
            calls = []

            def interrupt(policy, record, guard):
                calls.append(record['seed'])
                if len(calls) == 2:
                    raise TimeoutError('incomplete game')
                return real_rollout(policy, record, guard)

            with patch.object(comparison, 'load_policy', return_value=(Policy(), {'format': 'test'})):
                with patch.object(comparison.gameplay_gate.competition, 'CompetitionSession', return_value=session):
                    with patch.object(comparison, 'memory_guard', return_value=9 * 2**30):
                        with patch.object(comparison, 'generated_rollout', side_effect=interrupt):
                            with redirect_stdout(io.StringIO()), self.assertRaises(TimeoutError):
                                comparison.main(args)
            report = json.loads((root / 'report.json').read_text())
            arm = report['arms']['baseline']
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(arm['status'], 'partial_failed')
            self.assertEqual(arm['active_seed'], 1_000_001)
            self.assertEqual(arm['generated_summary']['levels'], 1)
            self.assertEqual(arm['generated_summary']['failures'], 0)
            self.assertEqual(arm['generated_summary']['unrecorded_levels'], 69)
            self.assertEqual(report['paired'], {})
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0)

    def test_sequential_only_skips_bank_and_hash_mismatch_precedes_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.args(root)
            args[3] = '0' * 64
            with patch.object(comparison, 'load_policy') as load:
                with patch.object(comparison, 'memory_guard', return_value=9 * 2**30):
                    with self.assertRaisesRegex(ValueError, 'SHA256'):
                        comparison.main(args)
                load.assert_not_called()
            self.assertEqual(json.loads((root / 'report.json').read_text())['status'], 'failed')
            (root / 'report.json').unlink()
            args = self.args(root)
            sessions = [CompetitionSession([generated_level() for _ in range(7)]) for _ in range(2)]
            with patch.object(comparison, 'load_policy', side_effect=[(Policy(), {'format': 'test'}) for _ in range(2)]):
                with patch.object(comparison.gameplay_gate.competition, 'CompetitionSession', side_effect=sessions):
                    with patch.object(comparison, 'memory_guard', return_value=9 * 2**30):
                        with patch.object(comparison, 'checked_specs') as bank:
                            with redirect_stdout(io.StringIO()):
                                report = comparison.main(args)
                            bank.assert_not_called()
            self.assertEqual(report['bank_levels'], 0)
            self.assertEqual(report['gameplay_gate']['candidate']['status'], 'tied')

    def test_generated_guard_after_winning_action_propagates(self):
        env = comparison.Ls20Scenario(generated_level(), 0)

        def guard():
            if env.levels_completed == 1:
                raise TimeoutError('budget')

        with patch.object(comparison, 'Ls20Scenario', return_value=env):
            with self.assertRaises(TimeoutError):
                comparison.generated_rollout(Policy(), spec(), guard)
        self.assertEqual(env.levels_completed, 1)
