import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from tools.collect_neural_planning_sequences import checked_specs, collect_level, collect_root, collect_teacher_endings, observe, summarize


class ToyBranch:
    def __init__(self, root=None):
        self.root = root
        self.count = 0
        self.remaining = 3
        self.calls = []

    def lives(self):
        return self.remaining

    def player_cell(self):
        return (self.count, 0)

    def triple(self):
        return (0, 1, 2)

    def steps_left(self):
        return 20 - self.count

    def perform(self, action):
        self.calls.append(action)
        self.count += 1
        if self.count == 1:
            self.remaining -= 1
        terminal = self.count == (1 if self.root == 0 else 3)
        frame = None if terminal else np.full((64, 64), self.count + 3, np.uint8)
        return SimpleNamespace(frame=frame, finished=terminal, won=terminal)


class PlanningSequenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_private_histories_reset_and_labels_mask_only_actual_terminal_tail(self):
        initial = np.zeros((64, 64), np.uint8)
        recent = np.ones((64, 64), np.uint8)
        history = ([initial, recent], [-1, 2])
        before = copy.deepcopy(history)
        encodings = []
        branches = [ToyBranch(root) for root in range(4)]

        def encoder(frames, valid, actions):
            encodings.append((frames.clone(), valid.clone(), actions.clone()))
            return torch.full((1, 148, 96), float(frames[0, -1, 0, 0]))

        def imagine(field, horizon):
            self.assertTrue(all(not branch.calls for branch in branches))
            self.assertEqual(len(encodings), 1)
            return dict(root_actions=torch.tensor([[3, 0, 2, 1]]),
                        imagined_actions=torch.tensor([[[3, 2, 1, 0], [0, 1, 2, 3], [2, 3, 0, 1], [1, 0, 3, 2]]]))

        policy = SimpleNamespace(encoder=encoder, planner=SimpleNamespace(imagine=imagine))
        with patch('tools.collect_neural_planning_sequences.clone_env', side_effect=branches):
            row = collect_root(policy, ToyBranch(), history)
        np.testing.assert_array_equal(history[0], before[0])
        self.assertEqual(history[1], before[1])
        np.testing.assert_array_equal(row['transition_valid'].sum(1), [1, 3, 3, 3])
        np.testing.assert_array_equal(row['next_field_valid'].sum(1), [0, 2, 2, 2])
        self.assertTrue(row['terminal'][0, 0])
        self.assertTrue(row['lost_life'][:, 0].all())
        self.assertFalse(row['lost_life'][:, 1:].any())
        self.assertEqual(int(row['next_lives'][1, 1]), 2)
        # For each live branch first observed target is reset: only last slot
        # valid, producing action -1. Next target appends its fixed second action.
        for index in (1, 3, 5):
            self.assertEqual(int(encodings[index][1].sum()), 1)
            self.assertTrue(bool((encodings[index][2] == -1).all()))
            self.assertEqual(int(encodings[index + 1][1].sum()), 2)
        self.assertEqual([branch.calls for branch in branches], [[1], [2, 1, 4], [3, 4, 1], [4, 3, 2]])
        self.assertTrue(bool((row['next_fields'][~row['next_field_valid']] == 0).all()))

    def test_observe_keeps_parent_buffers_private(self):
        original = ([0, 1], [-1, 2])
        left = observe(original, 2, 3)
        right = observe(original, 9, 0, reset=True)
        self.assertEqual(original, ([0, 1], [-1, 2]))
        self.assertEqual(left, ([0, 1, 2], [-1, 2, 3]))
        self.assertEqual(right, ([9], [-1]))

    def test_coverage_uses_transition_masks_for_padded_events(self):
        arrays = dict(transition_valid=np.zeros((1, 4, 4), bool), next_field_valid=np.zeros((1, 4, 4), bool),
                      seeds=np.array([5]), root_step=np.array([0]),
                      **{key: np.ones((1, 4, 4), bool) for key in ('terminal', 'won', 'lost_life')})
        arrays['transition_valid'][0, 0, 0] = True
        coverage = summarize(arrays)
        self.assertEqual(coverage['transitions_by_horizon'], [1, 0, 0, 0])
        self.assertEqual(coverage['events']['won'], [1, 0, 0, 0])

    def test_train_bank_rejects_wrong_split_duplicate_seed_and_context(self):
        specs = [dict(seed=tier, source='generated_only', split='train', generator_version=3,
                      geometry_split='train', difficulty_version='ls20-reference-v1', difficulty=tier,
                      context_engine_verified=True, search_truncated=False,
                      training_context_index=tier - 1, context_index=tier - 1) for tier in range(1, 8)]
        with patch('tools.collect_neural_planning_sequences.load_bank', return_value=specs):
            self.assertEqual(len(checked_specs('unused')), 7)
        for field, value in [('split', 'validation'), ('seed', 1_000_000), ('training_context_index', 6)]:
            invalid = copy.deepcopy(specs)
            invalid[0][field] = value
            with patch('tools.collect_neural_planning_sequences.load_bank', return_value=invalid):
                with self.assertRaises(ValueError):
                    checked_specs('unused')
        with patch('tools.collect_neural_planning_sequences.load_bank', return_value=specs + [specs[0]]):
            with self.assertRaises(ValueError):
                checked_specs('unused')
        validation = [{**spec, 'seed': spec['seed'] + 1_000_000, 'split': 'validation', 'geometry_split': 'validation'} for spec in specs]
        with patch('tools.collect_neural_planning_sequences.load_bank', return_value=validation):
            self.assertEqual(len(checked_specs('unused', 'validation')), 7)
            with self.assertRaises(ValueError):
                checked_specs('unused', 'train')

    def test_budget_roots_continue_into_later_lives_with_bounded_root_count(self):
        class BudgetEnv:
            step = 0

            def reset(self):
                return np.zeros((64, 64), np.uint8)

            def lives(self):
                return 3 - self.step // 6

            def steps_left(self):
                return 6 - self.step % 6

            def perform(self, action):
                self.step += 1
                return SimpleNamespace(frame=np.zeros((64, 64), np.uint8), finished=False, won=False)

        spec = dict(seed=5, difficulty=1, difficulty_version='ls20-reference-v1')
        policy = SimpleNamespace(encoder=None, planner=SimpleNamespace(continuation_logits=lambda _: torch.zeros(1, 4)))
        for limit, expected in [(8, [0, 2, 8]), (2, [0, 2])]:
            env = BudgetEnv()
            with patch('tools.collect_neural_planning_sequences.build_level', return_value=None), \
                 patch('tools.collect_neural_planning_sequences.Ls20Scenario', return_value=env), \
                 patch('tools.collect_neural_planning_sequences.encode_history', return_value=torch.zeros(1, 148, 96)), \
                 patch('tools.collect_neural_planning_sequences.collect_root', side_effect=lambda *_: {}):
                _, record = collect_level(policy, spec, [0], 0., np.random.default_rng(42), behavior_actions=10, max_roots=limit)
            self.assertEqual(env.step, 10)
            self.assertEqual(record['root_steps'], expected)
            self.assertEqual(record['root_reasons'][1]['reasons'], ['near_budget'])
            if limit == 8:
                self.assertEqual(record['root_reasons'][2]['lives'], 2)

    def test_teacher_prefix_replay_keeps_imagination_model_selected_and_requires_win(self):
        route = [4, 1, 2, 3, 4]

        class TeacherEnv:
            def __init__(self, allow_win=True):
                self.actions = []
                self.allow_win = allow_win

            def reset(self):
                return np.zeros((64, 64), np.uint8)

            def lives(self):
                return 3

            def steps_left(self):
                return 42 - len(self.actions)

            def player_cell(self):
                return (len(self.actions), 0)

            def triple(self):
                return (0, 1, 2)

            def perform(self, action):
                self.actions.append(action)
                won = self.allow_win and self.actions == route
                return SimpleNamespace(frame=None if won else np.full((64, 64), len(self.actions), np.uint8),
                                       finished=won, won=won)

        seen = []

        def imagine(field, horizon):
            seen.append(int(field[0, 0, 0]))
            return dict(root_actions=torch.arange(4)[None],
                        imagined_actions=torch.arange(4)[None, :, None].expand(1, 4, 4))

        def encoder(frames, valid, actions):
            return torch.full((1, 148, 96), float(frames[0, -1, 0, 0]))

        policy = SimpleNamespace(encoder=encoder, planner=SimpleNamespace(imagine=imagine))
        spec = dict(seed=5, difficulty=1, difficulty_version='ls20-reference-v1',
                    context_solution=route, context_optimal_actions=len(route))
        for allow_win in (True, False):
            env = TeacherEnv(allow_win)

            def clone(source):
                branch = copy.deepcopy(source)
                # Keep branch targets successful in the negative control so it
                # specifically checks complete real prefix verification too.
                branch.allow_win = True
                return branch

            with patch('tools.collect_neural_planning_sequences.build_level', return_value=None), \
                 patch('tools.collect_neural_planning_sequences.Ls20Scenario', return_value=env), \
                 patch('tools.collect_neural_planning_sequences.clone_env', side_effect=clone):
                if not allow_win:
                    with self.assertRaisesRegex(ValueError, 'failed to win'):
                        collect_teacher_endings(policy, spec)
                    continue
                rows, record = collect_teacher_endings(policy, spec)
            self.assertEqual(env.actions, route)
            self.assertEqual(record['root_steps'], [1, 4])
            self.assertEqual(seen, [1, 4])
            self.assertTrue(record['teacher_route_verified_won'])
            for row in rows:
                np.testing.assert_array_equal(row['actions'], np.repeat(np.arange(4)[:, None], 4, axis=1))
            self.assertTrue(rows[-1]['won'][3, 0])
            self.assertFalse(rows[-1]['transition_valid'][3, 1:].any())

    def test_teacher_route_ids_and_declared_length_checked_before_engine(self):
        with patch('tools.collect_neural_planning_sequences.Ls20Scenario') as constructor:
            for route, length in [([0], 1), ([True], 1), ([1, 2], 1), ([], 0)]:
                with self.assertRaises(ValueError):
                    collect_teacher_endings(None, dict(context_solution=route, context_optimal_actions=length))
        constructor.assert_not_called()


if __name__ == '__main__':
    unittest.main()
