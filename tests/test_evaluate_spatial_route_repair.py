"""CPU-only protocol tests for the bounded route-repair evaluator."""

import copy
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = ''

from arcengine import GameState
import torch
from torch import nn

from pebby.agent import evaluate
from pebby.agent.competition import DecisionContext, run_competition
from pebby.ls20.env import Ls20Env
from tools import evaluate_spatial_route_repair as evaluator


class FixedPlanner(nn.Module):
    def forward(self, *args, **kwargs):
        return {'event_logits': torch.zeros(1, 4, 3)}


class FixedPolicy(nn.Module):
    """Raw argmax is UP (index 0); public next-best can select DOWN (index 1)."""

    def __init__(self, architecture='world'):
        super().__init__()
        self.planner = FixedPlanner()
        self.architecture = architecture

    def config(self):
        return {'architecture': self.architecture, 'history': 2}

    def forward(self, frames, history_valid=None, previous_actions=None):
        if self.architecture == 'world':
            self.planner()
        return torch.tensor([[4., 3., 0., -1.]], device=frames.device)


class GeneratedEnvironment:
    level_count = 1
    level_index = 0

    def reset(self):
        self.steps = 4
        self._lives = 3
        self._frame = [[1]]
        self._goals = [False]
        self.levels_completed = 0
        self._state = GameState.NOT_FINISHED
        self.actions = []
        return copy.deepcopy(self._frame)

    def set_level(self, index):
        if index != 0:
            raise ValueError(index)

    def goal_triples(self):
        return [(0, 0, 0)]

    def lives(self):
        return self._lives

    def goals_solved(self):
        return list(self._goals)

    def player_cell(self):
        return (0, 0)

    def triple(self):
        return (0, 0, 0)

    def steps_left(self):
        return self.steps

    @property
    def state(self):
        return self._state

    def perform(self, action):
        self.actions.append(action)
        self.steps -= 1
        if action == 1:
            # A charged stationary move changes only the budget, so it is both
            # an exact-frame refusal and the new blocked diagnostic.
            frame = copy.deepcopy(self._frame)
            return SimpleNamespace(frame=frame, finished=False, won=False)
        if action != 2:
            raise AssertionError(f'next-best should choose action 2, got {action}')
        self._frame = [[2]]
        self._goals = [True]
        self.levels_completed = 1
        self._state = GameState.WIN
        return SimpleNamespace(frame=copy.deepcopy(self._frame), finished=True, won=True)


class FakeEngine:
    def __init__(self):
        self.level_index = 0
        self._cell = (0, 0)
        self._triple = (0, 0, 0)
        self._goals = [False]
        self._steps = 10
        self._lives = 3
        self._state = GameState.NOT_FINISHED

    @property
    def state(self):
        return self._state

    def player_cell(self):
        return self._cell

    def triple(self):
        return self._triple

    def goals_solved(self):
        return list(self._goals)

    def steps_left(self):
        return self._steps

    def lives(self):
        return self._lives


class FakeCompetitionSession:
    """Small session with a terminal, charged RESET, and level transition."""

    def __init__(self):
        self._env = FakeEngine()
        self.frame = [[0]]
        self.initial_frame_sha256 = evaluator.competition.frame_digest(self.frame)
        self.actions = 0
        self.resets = 0
        self.per_level_actions = [0, 0]
        self.ledger = []
        self._run_started = False
        self.level_count = 2
        self._levels_completed = 0

    @property
    def level_index(self):
        return self._env.level_index

    @property
    def state(self):
        return self._env.state

    @property
    def lives(self):
        return self._env.lives()

    @property
    def levels_completed(self):
        return self._levels_completed

    def context(self):
        legal = () if self.state == GameState.WIN else ((0,) if self.state == GameState.GAME_OVER
                                                         else (0, 1, 2, 3, 4))
        return DecisionContext(self.frame, self.state, self.levels_completed,
                               self.level_count, legal)

    def step(self, action):
        if action not in self.context().available_actions:
            raise ValueError(action)
        before_level = self.level_index
        before_progress = self.levels_completed
        before_lives = self.lives
        before_state = self.state.value
        before_frame = evaluator.competition.frame_digest(self.frame)
        if action == 0:
            self._env._state = GameState.NOT_FINISHED
            self._env._steps = 10
            # Deliberately unchanged: RESET is a charged controller command,
            # not a four-logit policy transition for refusal accounting.
            self.frame = [[1]]
            self.resets += 1
        elif self.actions == 0:
            self._env._state = GameState.GAME_OVER
            self.frame = [[1]]
        elif self.actions == 2:
            self._env.level_index = 1
            self._levels_completed = 1
            self._env._goals = [False, False]
            self.frame = [[3]]
        elif self.actions == 3:
            self._levels_completed = 2
            self._env._goals = [True, True]
            self._env._state = GameState.WIN
            self.frame = [[4]]
        else:
            raise AssertionError(f'unexpected action {action} at count {self.actions}')
        self.actions += 1
        self.per_level_actions[before_level] += 1
        event = dict(action=action, charged_actions=self.actions,
                     level_before=before_level + 1, level_after=self.level_index + 1,
                     level_actions=self.per_level_actions[before_level],
                     levels_completed_before=before_progress,
                     levels_completed=self.levels_completed, lives_before=before_lives,
                     lives_after=self.lives, state_before=before_state,
                     state_after=self.state.value,
                     frame_before_sha256=before_frame,
                     frame_after_sha256=evaluator.competition.frame_digest(self.frame),
                     reset=action == 0)
        self.ledger.append(event)
        return SimpleNamespace(frame=copy.deepcopy(self.frame),
                               finished=self.state in (GameState.WIN, GameState.GAME_OVER),
                               won=self.state == GameState.WIN), event


class SpatialRouteRepairEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.census = unittest.mock.patch.object(evaluate, 'level_goal_counts',
                                                 return_value=[1])
        self.census.start()

    def tearDown(self):
        self.census.stop()
        self.assertFalse(torch.cuda.is_initialized())

    def test_provenance_binds_paired_statistics_and_distinguishes_official_evaluation(self):
        dependency = evaluator.ROOT / 'tools/spatial_paired_statistics.py'
        for mode in ('generated', 'shipped', 'shipped-isolated'):
            with self.subTest(mode=mode):
                self.assertIn(dependency, evaluator._source_paths(mode))
                args = SimpleNamespace(mode=mode, max_seconds=10, bank_sha256='a' * 64,
                    device='cpu', max_actions=5, on_stall='repeat',
                    foundation_human_baseline_caps=False, per_level_max_actions=5)
                result = evaluator._new_report(args, [{'seed': 1}], [{'seed': 1}], {})
                self.assertEqual(result['official_inputs_used'], mode != 'generated')
                self.assertFalse(result['official_training_inputs'])
                self.assertFalse(result['training'])

    def test_generated_next_best_records_executed_native_second_action(self):
        env = GeneratedEnvironment()
        run = evaluator.rollout(FixedPolicy(), env, max_actions=5, on_stall='next-best')
        self.assertTrue(run['completed'])
        self.assertEqual(run['native_action_sequence'], [1, 2])
        self.assertEqual(run['native_policy_action_indices'], [0, 1])
        self.assertEqual(run['recovery']['refusal_actions'], 1)
        self.assertEqual(run['recovery']['blocked_no_player_motion_budget_spending'], 1)
        self.assertEqual(run['recovery']['blocked_no_player_motion_budget_units'], 1)

    def test_generated_repeat_matches_unmodified_greedy_action_sequence(self):
        baseline_env = GeneratedEnvironment()
        measured_env = GeneratedEnvironment()
        baseline = evaluate.rollout(FixedPolicy(), baseline_env, 3, on_stall='repeat')
        measured = evaluator.rollout(FixedPolicy(), measured_env, max_actions=3, on_stall='repeat')
        recovery = measured.pop('recovery')
        measured.pop('native_action_sequence')
        measured.pop('native_policy_action_indices')
        self.assertEqual(measured, baseline)
        self.assertEqual(measured_env.actions, baseline_env.actions)
        self.assertEqual(recovery['blocked_no_player_motion_budget_spending'], 3)

    def test_shipped_adapter_handles_movement_terminal_reset_and_level_transition(self):
        session = FakeCompetitionSession()
        diagnostics = evaluator.CompetitionDiagnostics()
        decision = evaluator.CompetitionTraceDecision(
            FixedPolicy(architecture='cnn'), 'cpu', 'repeat', session, lambda: None, diagnostics)
        result = run_competition(decision, session, per_level_caps=[10, 10])
        self.assertTrue(result['completed'])
        self.assertEqual(decision.native_action_sequence, [1, 0, 1, 1])
        self.assertEqual(result['resets'], 1)
        self.assertEqual(result['levels_completed'], 2)
        report = diagnostics.report()
        self.assertEqual(report['blocked_no_player_motion_budget_spending'], 0)
        self.assertEqual(report['unchanged_actions'], 0)
        self.assertEqual(report['blocked_no_player_motion_exclusions']['terminal'], 2)
        self.assertEqual(report['blocked_no_player_motion_exclusions']['level_transition'], 1)
        self.assertEqual(diagnostics.goal_level, 1)
        self.assertEqual(diagnostics.ever_goals, [True, True])

    def test_blocked_budget_count_is_separate_from_exact_frame_refusal(self):
        diagnostics = evaluator.CompetitionDiagnostics()
        diagnostics.observe_level_goals(0, [False])
        before = dict(level=0, cell=(0, 0), triple=(0, 0, 0), goals=(False,),
                      steps=5, lives=3)
        after = {**before, 'steps': 4}
        diagnostics.transition(0, [[0]], SimpleNamespace(frame=[[1]], finished=False, won=False),
                               3, 3, [False], before_state=before, after_state=after,
                               fingerprint='a')
        after_refusal = {**before, 'steps': 5}
        diagnostics.transition(0, [[1]], SimpleNamespace(frame=[[1]], finished=False, won=False),
                               3, 3, [False], before_state=before, after_state=after_refusal,
                               fingerprint='b')
        report = diagnostics.report()
        self.assertEqual(report['blocked_no_player_motion_budget_spending'], 1)
        self.assertEqual(report['refusal_actions'], 1)
        self.assertEqual(report['blocked_no_player_motion_exclusions']['budget_not_decreased'], 1)

    def test_checkpoint_format_dispatch_keeps_retained_and_route_loaders_versioned(self):
        from pebby.agent import (spatial_outcome_policy, spatial_route_outcome_policy,
                                 spatial_semantic_outcome_policy)
        self.assertIs(evaluator._loader_for_metadata({'format': evaluator.SPATIAL_FORMAT}),
                      spatial_outcome_policy.load_checkpoint)
        self.assertIs(evaluator._loader_for_metadata({'format': evaluator.ROUTE_FORMAT}),
                      spatial_route_outcome_policy.load_checkpoint)
        self.assertIs(evaluator._loader_for_metadata({'format': evaluator.SEMANTIC_FORMAT}),
                      spatial_semantic_outcome_policy.load_checkpoint)
        with self.assertRaises(ValueError):
            evaluator._loader_for_metadata({'format': 'legacy-oracle-format'})

    def test_transient_game_over_row_is_capped_when_reset_commands_reach_cap(self):
        report = dict(levels_completed=0, per_level_actions=[3], per_level_caps=[3],
                      ledger=[{'level_before': 1, 'state_after': 'GAME_OVER',
                               'lives_after': 0}])
        rows = evaluator._shipped_level_rows(report)
        self.assertEqual(rows[0]['ending'], 'capped')
        self.assertTrue(rows[0]['game_over_seen'])

    def test_isolated_shipped_arm_keeps_real_per_level_goal_denominators(self):
        class Policy(nn.Module):
            def config(self):
                return {'architecture': 'cnn'}

        fake_runs = [dict(completed=index == 0, levels_completed=int(index == 0),
                          levels_total=1, goals_cleared=int(index == 0),
                          goals_total=(2 if index == 5 else 1), actions=3,
                          ending=('win' if index == 0 else 'capped'), stalls=0,
                          on_stall='repeat', temperature=0., optimal=index + 1,
                          actions_vs_optimal=3 / (index + 1), final_level=index,
                          lives_left=3)
                     for index in range(7)]
        with tempfile.TemporaryDirectory() as directory, \
                unittest.mock.patch.object(evaluator, 'load_policy',
                                           return_value=(Policy(), {'format': 'test',
                                                                     'encoder_frozen': True,
                                                                     'official_training_inputs': False})), \
                unittest.mock.patch.object(evaluator.evaluate, 'shipped_levels',
                                           return_value=list(range(7))), \
                unittest.mock.patch.object(evaluator.evaluate, 'level_optimum',
                                           side_effect=lambda index: (index + 1, 'test')), \
                unittest.mock.patch.object(evaluator, 'Ls20Scenario',
                                           side_effect=lambda level, index: (level, index)), \
                unittest.mock.patch.object(evaluator.evaluate, 'rollout',
                                           side_effect=fake_runs):
            args = SimpleNamespace(device='cpu', foundation_human_baseline_caps=False,
                                   per_level_max_actions=9, on_stall='repeat',
                                   report_out=Path(directory) / 'report.json')
            report = {'arms': {}, '_started_monotonic': time.monotonic()}
            evaluator._run_isolated_shipped_arm('baseline', Path('unused.pt'), 'a' * 64,
                                                args, report, lambda: None)
        rows = report['arms']['baseline']['per_level']
        self.assertEqual([row['goals_total'] for row in rows], [1, 1, 1, 1, 1, 2, 1])
        self.assertEqual(report['arms']['baseline']['summary']['goals_total'], 8)
        self.assertEqual(report['arms']['baseline']['summary']['resets'], 0)


class RealShippedGoalCensusTests(unittest.TestCase):
    def test_real_shipped_goal_counts_are_per_level(self):
        self.assertEqual(evaluate.level_goal_counts(Ls20Env()), [1, 1, 1, 1, 1, 2, 1])


if __name__ == '__main__':
    unittest.main()
