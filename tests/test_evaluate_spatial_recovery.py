"""CPU checks that passive recovery measurements preserve strict gameplay."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from pebby.agent import evaluate
from tools.evaluate_spatial_recovery import input_digest, paired, rollout, validate_specs
from tools import evaluate_spatial_recovery as recovery_tool


class Planner(nn.Module):
    def forward(self):
        return {'event_logits': torch.tensor([[[0., 0., 4.]] * 4])}


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.planner = Planner()

    def config(self):
        return dict(architecture='world', history=8)

    def forward(self, frames, history_valid=None, previous_actions=None):
        self.planner()
        return torch.tensor([[1., 0., 0., 0.]])


class Environment:
    level_count = 1
    level_index = 0

    def __init__(self, events=()):
        self.events = events
        self.reset()

    def reset(self):
        self.steps = 0
        self.levels_completed = 0
        self.remaining = 3
        self.goals = [False, False]
        self.frame = [[0]]
        self.actions = []
        return copy.deepcopy(self.frame)

    def lives(self):
        return self.remaining

    def goals_solved(self):
        return list(self.goals)

    def perform(self, action):
        self.actions.append(action)
        event = self.events[self.steps] if self.steps < len(self.events) else 'refusal'
        self.steps += 1
        won = event == 'win'
        if event == 'goal':
            self.goals[0] = True
            self.frame = [[1]]
        elif event == 'death':
            self.remaining -= 1
            self.goals = [False, False]
            self.frame = [[2]]
        elif won:
            self.goals = [True, True]
            self.levels_completed = 1
            self.frame = [[3]]
        return SimpleNamespace(frame=copy.deepcopy(self.frame), finished=won, won=won)


class RecoveryEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.census = patch.object(evaluate, 'level_goal_counts', return_value=[2])
        self.census.start()

    def tearDown(self):
        self.census.stop()
        self.assertFalse(torch.cuda.is_initialized())

    def test_diagnostics_preserve_actions_and_native_metrics_at_fixedpoint(self):
        baseline_env, measured_env = Environment(), Environment()
        baseline = evaluate.rollout(Policy(), baseline_env, 20, on_stall='repeat')
        policy = Policy()
        measured = rollout(policy, measured_env, max_actions=20)
        diagnostics = measured.pop('recovery')
        self.assertEqual(measured, baseline)
        self.assertEqual(measured_env.actions, baseline_env.actions)
        self.assertEqual(diagnostics['refusal_actions'], 20)
        self.assertEqual(diagnostics['max_consecutive_refusals'], 20)
        self.assertGreater(diagnostics['exact_fixedpoint_steps'], 0)
        self.assertEqual(diagnostics['refusal_predicted_win_ge_0_5'], 20)
        self.assertEqual(diagnostics['refusal_predicted_no_win_lt_0_5'], 0)
        self.assertFalse(diagnostics['all_goals_ever_observed_solved'])
        self.assertFalse(policy._forward_hooks)
        self.assertFalse(policy._forward_pre_hooks)
        self.assertFalse(policy.planner._forward_hooks)

    def test_goal_progress_is_retained_in_diagnostic_after_native_life_loss(self):
        run = rollout(Policy(), Environment(['goal', 'death']), max_actions=2)
        self.assertEqual(run['goals_cleared'], 0)
        self.assertEqual(run['recovery']['goals_ever_observed_solved'], 1)
        self.assertEqual(run['recovery']['lives_lost'], 1)
        self.assertFalse(run['recovery']['all_goals_simultaneously_observed'])

    def test_success_and_refusal_exit_remain_distinct(self):
        run = rollout(Policy(), Environment(['refusal', 'goal', 'win']), max_actions=10)
        self.assertTrue(run['completed'])
        self.assertEqual(run['actions'], 3)
        self.assertEqual(run['recovery']['refusal_episodes_followed_by_change'], 1)
        self.assertTrue(run['recovery']['all_goals_simultaneously_observed'])

    def test_no_refusal_support_returns_null_and_failure_removes_hooks(self):
        run = rollout(Policy(), Environment(['win']), max_actions=1)
        self.assertIsNone(run['recovery']['refusal_win_output_mean'])
        policy = Policy()
        def fail():
            raise TimeoutError('test deadline')
        with self.assertRaises(TimeoutError):
            rollout(policy, Environment(), check=fail)
        self.assertFalse(policy._forward_hooks)
        self.assertFalse(policy.planner._forward_hooks)

    def test_public_input_hash_includes_validity_and_actions(self):
        frame = torch.zeros(1, 8, 1, 1, dtype=torch.int64)
        kwargs = dict(history_valid=torch.ones(1, 8, dtype=torch.bool), previous_actions=torch.zeros(1, 8, dtype=torch.int64))
        original = input_digest((frame,), kwargs)
        kwargs['previous_actions'][0, 0] = 1
        self.assertNotEqual(original, input_digest((frame,), kwargs))
        kwargs['previous_actions'][0, 0] = 0
        kwargs['history_valid'][0, 0] = False
        self.assertNotEqual(original, input_digest((frame,), kwargs))

    def test_partial_pairing_never_counts_unplayed_levels_as_losses(self):
        reference = [dict(seed=1, completed=True), dict(seed=2, completed=False), dict(seed=3, completed=True)]
        candidate = [dict(seed=1, completed=False), dict(seed=2, completed=True)]
        self.assertEqual(paired(reference, candidate), dict(paired_levels=2, win_gains=[2], win_losses=[1],
                                                          net_wins=0, reference_only=1, candidate_only=0))
        with self.assertRaises(ValueError):
            paired(reference + [reference[0]], candidate)

    def test_only_distinct_generated_validation_levels_are_accepted(self):
        good = dict(seed=1_000_000, source='generated_only', difficulty=1)
        validate_specs([good])
        for bad in [[], [good, good], [{**good, 'seed': 1}], [{**good, 'source': 'official'}]]:
            with self.assertRaises(ValueError):
                validate_specs(bad)

    def test_deadline_failure_publishes_honest_prefix_and_does_not_count_unplayed_arm(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            bank, checkpoint, output = folder/'bank.jsonl', folder/'model.pt', folder/'report.json'
            spec = dict(seed=1_000_000, source='generated_only', difficulty=1)
            bank.write_text(json.dumps(spec)+'\n')
            checkpoint.write_text('fixture; loader mocked')
            policy = Policy()
            policy.direct_weight, policy.planner_weight = 0., 1.
            metadata = dict(planner_horizon=1, planner_refinement_loops=1, encoder_weights_sha256='same')
            with (patch.object(recovery_tool, 'guard', return_value=10*2**30),
                  patch.object(recovery_tool, 'load_checkpoint', side_effect=[(policy, metadata), TimeoutError('second arm')]),
                  patch.object(evaluate, 'bank_levels', return_value=([None], [1], [spec])),
                  patch.object(recovery_tool, 'Ls20Scenario', return_value=Environment(['win']))):
                args = ['--bank', str(bank), '--bank-sha256', recovery_tool.sha(bank), '--device', 'cpu',
                        '--report-out', str(output), '--max-seconds', '10']
                for name in ('current', 'recovery'):
                    args += ['--checkpoint', name, str(checkpoint), recovery_tool.sha(checkpoint)]
                with self.assertRaises(TimeoutError):
                    recovery_tool.main(args)
            result = json.loads(output.read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['arms']['current']['status'], 'complete')
            self.assertEqual(result['arms']['current']['summary']['levels'], 1)
            self.assertNotIn('recovery', result['arms'])
            self.assertEqual(result['paired'], {})
            self.assertNotIn('sources_unchanged', result)


if __name__ == '__main__':
    unittest.main()


class StallProtocolTests(unittest.TestCase):
    def test_requested_protocol_is_forwarded_and_validated(self):
        for mode in ('repeat', 'next-best'):
            with patch.object(evaluate, 'rollout', return_value={}) as run:
                rollout(Policy(), Environment(), on_stall=mode)
            self.assertEqual(run.call_args.kwargs['on_stall'], mode)
        with self.assertRaises(ValueError):
            rollout(Policy(), Environment(), on_stall='unrecorded')
