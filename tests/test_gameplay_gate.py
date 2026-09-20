import copy
import unittest
from unittest.mock import patch

import torch

from pebby.agent import gameplay_gate
from pebby.agent.competition import CompetitionSession
from pebby.ls20.generate import build_level


def generated_level():
    return build_level(dict(walls=[(1, 3), (0, 2), (1, 1)], start=(1, 2),
                            start_triple=(0, 0, 0), goals=[dict(cell=(2, 2), triple=(0, 0, 0))],
                            cyclers=[], refills=[], launchers=[], step_counter=2,
                            step_cost=1, fog=False))


class FixedPolicy:
    """Choose RIGHT for a fixed number of calls, then repeatedly hit DOWN wall."""
    def __init__(self, right_count):
        self.right_count = right_count
        self.calls = 0

    def __call__(self, frames):
        scores = torch.zeros(1, 4)
        scores[0, 3 if self.calls < self.right_count else 1] = 1
        self.calls += 1
        return scores


class GameplayGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        # Only generated fixtures execute in tests; no shipped policy evaluation.
        cls.reports = {}
        for progress in (0, 1, 2, 7):
            session = CompetitionSession([generated_level() for _ in range(7)])
            with patch.object(gameplay_gate.competition, 'CompetitionSession', return_value=session):
                cls.reports[progress] = gameplay_gate.evaluate_sequential(FixedPolicy(progress), 'cpu',
                                                                          per_level_cap=3)

    def test_real_engine_strict_argmax_game_over_reset_and_accounting(self):
        session = CompetitionSession([generated_level() for _ in range(7)])
        policy = FixedPolicy(0)
        with patch.object(gameplay_gate.competition, 'CompetitionSession', return_value=session):
            report = gameplay_gate.evaluate_sequential(policy, 'cpu', per_level_cap=11)
        self.assertEqual([row['action'] for row in report['ledger']], [2] * 9 + [0, 2])
        self.assertEqual(policy.calls, 10)  # RESET controller does not call four-logit policy.
        self.assertEqual(report['actions'], 11)
        self.assertEqual(report['resets'], 1)
        self.assertEqual(report['lives_left'], 3)
        self.assertEqual(report['levels_completed'], 0)
        self.assertFalse(report['completed'])
        self.assertEqual(report['decision']['action_selection'], 'strict_argmax')
        self.assertEqual(report['decision']['stall_controller'], 'none')
        self.assertEqual(report['full_game_initializations'], 1)
        self.assertEqual(report['full_game_resets_after_initialization'], 0)
        self.assertIsNone(gameplay_gate._evidence_problem(report))

    def test_more_gameplay_progress_only_qualifies_for_confirmation(self):
        result = gameplay_gate.assess_gameplay(self.reports[1], self.reports[2])
        self.assertEqual(result['status'], 'eligible_for_gameplay_confirmation')
        self.assertFalse(result['promoted'])
        self.assertTrue(result['confirmation_required'])
        self.assertFalse(result['objective_evidence'])
        self.assertEqual((result['baseline_levels_completed'], result['candidate_levels_completed']), (1, 2))

    def test_regression_and_tie_cannot_be_rescued_by_offline_or_action_scores(self):
        candidate = copy.deepcopy(self.reports[1])
        candidate['offline_accuracy'] = 1.
        candidate['training_loss'] = 0.
        self.assertEqual(gameplay_gate.assess_gameplay(self.reports[2], candidate)['status'], 'reject')
        result = gameplay_gate.assess_gameplay(self.reports[1], candidate)
        self.assertEqual(result['status'], 'tied')
        self.assertFalse(result['promoted'])
        self.assertFalse(result['offline_metrics_used'])

    def test_seven_level_win_is_evidence_only_with_matching_protocol(self):
        result = gameplay_gate.assess_gameplay(self.reports[1], self.reports[7])
        self.assertTrue(result['objective_evidence'])
        self.assertFalse(result['promoted'])
        self.assertEqual(result['status'], 'eligible_for_gameplay_confirmation')
        tied = gameplay_gate.assess_gameplay(self.reports[7], self.reports[7])
        self.assertEqual(tied['status'], 'tied')
        self.assertTrue(tied['objective_evidence'])
        changed_cap = copy.deepcopy(self.reports[7])
        changed_cap['per_level_caps'] = [4] * 7
        changed_cap['protocol_identity'] = gameplay_gate._identity(changed_cap, changed_cap['source_bindings'])
        mismatch = gameplay_gate.assess_gameplay(self.reports[1], changed_cap)
        self.assertEqual(mismatch['status'], 'no_evidence')
        self.assertFalse(mismatch['objective_evidence'])

    def test_missing_identity_tampered_completion_and_partial_reports_fail_closed(self):
        bad_reports = [None, {'offline_accuracy': 1.}]
        missing_identity = copy.deepcopy(self.reports[7])
        missing_identity.pop('protocol_identity')
        bad_reports.append(missing_identity)
        missing_sources = copy.deepcopy(self.reports[7])
        missing_sources['source_bindings'] = {}
        missing_sources['protocol_identity'] = gameplay_gate._identity(missing_sources, {})
        bad_reports.append(missing_sources)
        incomplete = copy.deepcopy(self.reports[7])
        incomplete['ending'] = 'budget_interrupted'
        bad_reports.append(incomplete)
        false_win = copy.deepcopy(self.reports[1])
        false_win['completed'] = True
        false_win['levels_completed'] = 7
        false_win['ending'] = 'win'
        false_win['final_state'] = gameplay_gate.competition.GameState.WIN.value
        bad_reports.append(false_win)
        discontinuous = copy.deepcopy(self.reports[7])
        discontinuous['ledger'][1]['frame_before_sha256'] = 'wrong'
        bad_reports.append(discontinuous)
        for report in bad_reports:
            with self.subTest(report=report and report.get('ending')):
                result = gameplay_gate.assess_gameplay(self.reports[1], report)
                self.assertEqual(result['status'], 'no_evidence')
                self.assertFalse(result['objective_evidence'])
                self.assertFalse(result['promoted'])

    def test_budget_interrupt_before_initialization_and_after_winning_observation_propagates(self):
        class BudgetExpired(RuntimeError):
            pass

        def immediately():
            raise BudgetExpired('stop before initialization')

        with patch.object(gameplay_gate.competition, 'CompetitionSession') as constructor:
            with self.assertRaises(BudgetExpired):
                gameplay_gate.evaluate_sequential(FixedPolicy(7), 'cpu', guard=immediately)
            constructor.assert_not_called()
        session = CompetitionSession([generated_level() for _ in range(7)])

        def after_win():
            if session.levels_completed == 7:
                raise BudgetExpired('stop after winning observation')

        with patch.object(gameplay_gate.competition, 'CompetitionSession', return_value=session):
            with self.assertRaisesRegex(BudgetExpired, 'winning observation'):
                gameplay_gate.evaluate_sequential(FixedPolicy(7), 'cpu', guard=after_win, per_level_cap=3)
        self.assertEqual(session.actions, 7)

    def test_invalid_logits_caps_and_source_changes_do_not_return_evidence(self):
        for cap in (0, True, 1.5):
            with patch.object(gameplay_gate.competition, 'CompetitionSession') as constructor:
                with self.assertRaises(ValueError):
                    gameplay_gate.evaluate_sequential(FixedPolicy(0), 'cpu', per_level_cap=cap)
                constructor.assert_not_called()
        session = CompetitionSession([generated_level() for _ in range(7)])
        with patch.object(gameplay_gate.competition, 'CompetitionSession', return_value=session):
            with self.assertRaisesRegex(ValueError, 'finite'):
                gameplay_gate.evaluate_sequential(lambda frames: torch.full((1, 4), float('nan')), 'cpu')
        self.assertEqual(session.actions, 0)
        session = CompetitionSession([generated_level() for _ in range(7)])
        real_digest = gameplay_gate._digest

        def changed_source(path):
            return 'changed' if session.actions else real_digest(path)

        with patch.object(gameplay_gate.competition, 'CompetitionSession', return_value=session):
            with patch.object(gameplay_gate, '_digest', side_effect=changed_source):
                with self.assertRaisesRegex(RuntimeError, 'sources changed'):
                    gameplay_gate.evaluate_sequential(FixedPolicy(0), 'cpu', per_level_cap=1)
