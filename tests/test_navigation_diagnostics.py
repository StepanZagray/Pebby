"""Real-engine checks for the bounded generated navigation curriculum."""

from collections import Counter, defaultdict
import copy
import unittest

import numpy as np

from pebby.agent import navigation_diagnostics as nav
from pebby.ls20 import names
from pebby.ls20.env import Ls20Scenario
from pebby.ls20.generate import build_level


class NavigationDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = nav.make_cases(seed=17, groups_per_split=1)
        cls.cases = cls.splits['train']
        cls.rows = nav.collect_examples(cls.cases, history=3)

    def test_every_stage_teacher_replays_and_wins_at_optimum(self):
        for cases in self.splits.values():
            for case in cases:
                with self.subTest(case=case['id']):
                    spec = case['spec']
                    env = Ls20Scenario(build_level(spec), 0)
                    expected = (1 if case['stage'] == 'adjacent' else 5 if case['stage'] == 'detour'
                                else 4 if '-' in case['direction'] else 2)
                    self.assertEqual(case['optimal_length'], expected)
                    for step in range(expected):
                        mask = nav.optimal_actions(spec, env.player_cell())
                        index = next(i for i in range(4) if mask & (1 << i))
                        observation = env.perform(names.ACTION_IDS[index])
                        self.assertEqual(observation.finished, step == expected - 1)
                        self.assertEqual(nav.distance_to_goal(spec, env.player_cell()), expected - step - 1)
                    self.assertTrue(observation.won)
                    self.assertEqual(env.lives(), 3)
                    self.assertEqual(nav.optimal_actions(spec, env.player_cell()), 0)

    def test_detour_requires_non_manhattan_first_action(self):
        for case in self.cases:
            if case['stage'] != 'detour':
                continue
            spec = case['spec']
            goal, start = spec['goals'][0]['cell'], spec['start']
            mask = nav.optimal_actions(spec, start)
            manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
            self.assertGreater(case['optimal_length'], manhattan)
            for index, (dx, dy) in enumerate(names.ACTION_DELTAS):
                if mask & (1 << index):
                    self.assertGreater(abs(start[0] + dx - goal[0]) + abs(start[1] + dy - goal[1]), manhattan)

    def test_whole_groups_pairs_and_public_frames_are_disjoint(self):
        seen = set()
        for split, cases in self.splits.items():
            groups = {case['group_id'] for case in cases}
            self.assertFalse(groups & seen)
            seen |= groups
            pairs = defaultdict(list)
            for case in cases:
                self.assertEqual(case['split'], split)
                pairs[case['pair_id']].append(case)
            for pair in pairs.values():
                self.assertEqual(len(pair), 2)
                a, b = (case['spec'] for case in pair)
                self.assertEqual(a['start'], b['start'])
                self.assertEqual(a['start_triple'], b['start_triple'])
                self.assertEqual(a['walls'], b['walls'])
                self.assertEqual([u + v for u, v in zip(a['goals'][0]['cell'], b['goals'][0]['cell'])],
                                 [2 * u for u in a['start']])
            self.assertEqual(Counter(case['stage'] for case in cases), {'adjacent': 4, 'open': 8, 'detour': 4})
        self.assertTrue(all(not overlaps for overlaps in nav.public_frame_overlap(self.splits).values()))
        copied = copy.deepcopy(self.splits)
        copied['confirmation'][0] = copy.deepcopy(copied['train'][0])
        copied['confirmation'][0]['initial_frame_sha256'] = 'untrusted'
        self.assertTrue(nav.public_frame_overlap(copied)['train/confirmation'])
        self.assertEqual(self.splits, nav.make_cases(seed=17, groups_per_split=1))

    def test_causal_history_and_each_branch_match_independent_engine_replay(self):
        batch = self.rows
        for case_index, case in enumerate(self.cases):
            env = Ls20Scenario(build_level(case['spec']), 0)
            frames, actions = [env.render()], [-1]
            for row_index in np.flatnonzero(batch['case_index'] == case_index):
                count = min(3, len(frames))
                np.testing.assert_array_equal(batch['frames'][row_index], [frames[-count]] * (3 - count) + frames[-count:])
                np.testing.assert_array_equal(batch['history_valid'][row_index], [False] * (3 - count) + [True] * count)
                np.testing.assert_array_equal(batch['previous_actions'][row_index], [-1] * (3 - count) + actions[-count:])
                self.assertGreater(nav.distance_to_goal(case['spec'], env.player_cell()), 0)
                mask = 0
                before = nav.distance_to_goal(case['spec'], env.player_cell())
                for index, action in enumerate(names.ACTION_IDS):
                    branch = Ls20Scenario(build_level(case['spec']), 0)
                    for previous_index in actions[1:]:
                        branch.perform(names.ACTION_IDS[previous_index])
                    observation = branch.perform(action)
                    np.testing.assert_array_equal(batch['next_player_cell'][row_index, index], branch.player_cell())
                    np.testing.assert_array_equal(batch['next_triple'][row_index, index], branch.triple())
                    self.assertEqual(batch['next_steps'][row_index, index], branch.steps_left())
                    self.assertEqual(batch['next_lives'][row_index, index], branch.lives())
                    self.assertEqual(batch['won'][row_index, index], observation.won)
                    self.assertEqual(batch['terminal'][row_index, index], observation.finished)
                    distance = nav.distance_to_goal(case['spec'], branch.player_cell())
                    self.assertEqual(batch['distances'][row_index, index], distance)
                    self.assertNotIn(list(branch.player_cell()), case['spec']['walls'])
                    if distance == before - 1:
                        mask |= 1 << index
                self.assertEqual(batch['optimal'][row_index], mask)
                index = next(i for i in range(4) if mask & (1 << i))
                observation = env.perform(names.ACTION_IDS[index])
                if not observation.finished:
                    frames.append(observation.frame)
                    actions.append(index)
        self.assertFalse(batch['lost_life'].any())
        self.assertGreater(batch['won'].sum(), 0)
        self.assertEqual(batch['frames'].dtype, np.uint8)
        self.assertEqual(batch['previous_actions'].dtype, np.int64)
        self.assertEqual(batch['optimal'].dtype, np.uint8)
        self.assertEqual(nav.MODEL_INPUT_KEYS, ('frames', 'history_valid', 'previous_actions'))

    def test_learner_gets_only_public_history_and_labels_follow_visited_state(self):
        case = next(c for c in self.cases if c['stage'] == 'adjacent' and c['direction'] == 'right')
        seen = []
        def wrong_action(**inputs):
            self.assertEqual(set(inputs), set(nav.MODEL_INPUT_KEYS))
            seen.append({key: value.copy() for key, value in inputs.items()})
            inputs['frames'][:] = 255  # Cannot corrupt the collected history.
            return 0
        batch = nav.collect_examples([case], history=3, choose_action=wrong_action, max_actions=47)
        self.assertEqual(len(batch['optimal']), 47)
        self.assertEqual(batch['root_step'].tolist(), list(range(47)))
        self.assertEqual(batch['previous_actions'][1, -1], 0)
        self.assertNotEqual(batch['player_cell'][1].tolist(), case['spec']['start'])
        self.assertTrue((batch['frames'] < 16).all())
        env = Ls20Scenario(build_level(case['spec']), 0)
        for index in range(47):
            np.testing.assert_array_equal(batch['player_cell'][index], env.player_cell())
            np.testing.assert_array_equal(batch['frames'][index, -1], env.render())
            if index == 43:
                self.assertEqual(batch['current_lives'][index], 2)
                self.assertEqual(batch['history_valid'][index].tolist(), [False, False, True])
                self.assertEqual(batch['previous_actions'][index].tolist(), [-1, -1, -1])
            env.perform(1)
        self.assertTrue(batch['lost_life'].any())
        final = nav.collect_examples([case], choose_action=lambda **_: 0, max_actions=256)
        self.assertLess(len(final['optimal']), 256)
        self.assertEqual(final['current_lives'][-1], 1)
        self.assertTrue(final['terminal'][-1].all())
        self.assertFalse(final['won'][-1].any())
        self.assertTrue((final['distances'][-1] == -1).all())
        self.assertEqual(final['optimal'][-1], 0)

    def test_invalid_bounds_and_nonstatic_teacher_are_rejected(self):
        for groups in (0, 33, True):
            with self.assertRaises(ValueError):
                nav.make_cases(groups_per_split=groups)
        for stages in ((), ('adjacent', 'adjacent'), ('unknown',)):
            with self.assertRaises(ValueError):
                nav.make_cases(stages=stages)
        for cap in (0, 257, True):
            with self.assertRaises(ValueError):
                nav.collect_examples(self.cases, max_actions=cap)
        with self.assertRaises(ValueError):
            nav.collect_examples(self.cases, choose_action=lambda **_: 4)
        spec = copy.deepcopy(self.cases[0]['spec'])
        spec['cyclers'] = [{'cell': [3, 3], 'kind': 'shape'}]
        with self.assertRaises(ValueError):
            nav.optimal_actions(spec, spec['start'])


if __name__ == '__main__':
    unittest.main()
