"""Real-engine checks for the single-cycler spoil probe."""

from collections import Counter
import unittest

import numpy as np
import torch

from pebby.agent import cycler_probe as cp
from pebby.agent.world_data import history_arrays
from pebby.ls20 import names
from pebby.ls20.env import Ls20Scenario
from pebby.ls20.generate import build_level


def _key(frames, valid, previous):
    return (np.asarray(frames, dtype=np.uint8).tobytes() + np.asarray(valid, dtype=bool).tobytes()
            + np.asarray(previous, dtype=np.int64).tobytes())


class LookupPolicy(torch.nn.Module):
    """Public-input keyed stub: answers only from what it was handed."""

    def __init__(self, cases, choose, history=cp.HISTORY):
        super().__init__()
        self.length = history
        self.table = {_key(*cp.public_arrays(case, history)): choose(case) for case in cases}

    def config(self):
        return {'architecture': 'world', 'history': self.length}

    def forward(self, frames, history_valid=None, previous_actions=None):
        logits = torch.zeros(frames.shape[0], 4)
        for row in range(frames.shape[0]):
            key = _key(frames[row].to(torch.uint8).numpy(), history_valid[row].numpy(), previous_actions[row].numpy())
            logits[row, self.table[key]] = 4.
        return logits


class InputSumPolicy(torch.nn.Module):
    """Scores that depend on every public input, to compare invocation paths."""

    def __init__(self, history=cp.HISTORY):
        super().__init__()
        self.length = history

    def config(self):
        return {'architecture': 'world', 'history': self.length}

    def forward(self, frames, history_valid=None, previous_actions=None):
        frames = frames.float()
        return torch.stack([frames.sum((1, 2, 3)) % 7, history_valid.float().sum(1),
                            previous_actions.float().sum(1), frames[:, -1].mean((1, 2))], -1)


class CyclerProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.cases = cp.build_cases(3, 8)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_cases_are_engine_reached_with_exact_oracle_sets(self):
        self.assertEqual(len(self.cases), 24)
        self.assertEqual([c['type'] for c in self.cases], list(cp.TYPES) * 8)
        for case in self.cases:
            with self.subTest(case=case['id']):
                self.assertNotEqual(case['optimal'], 0)
                bit = 1 << case['cycler_direction']
                if case['type'] == 'approach':
                    self.assertTrue(case['optimal'] & bit)
                    self.assertNotEqual(case['triple'], case['goal_triple'])
                else:
                    self.assertFalse(case['optimal'] & bit)
                    self.assertEqual(case['triple'], case['goal_triple'])
                # Independent replay: the stored actions drive a fresh engine to
                # the stored state and render exactly the stored public frames.
                spec = case['spec']
                env = Ls20Scenario(build_level(spec), case['context_index'])
                frames = [np.asarray(env.render(), dtype=np.uint8)]
                for action in case['raw_actions'][1:]:
                    result = env.perform(names.ACTION_IDS[action])
                    self.assertFalse(result.finished)
                    frames.append(np.asarray(result.frame, dtype=np.uint8))
                self.assertEqual(list(env.player_cell()), case['player_cell'])
                self.assertEqual(list(env.triple()), case['triple'])
                self.assertEqual(env.lives(), 3)
                self.assertEqual(env.steps_left(), case['steps_left'])
                cycler = tuple(spec['cyclers'][0]['cell'])
                dx, dy = names.ACTION_DELTAS[case['cycler_direction']]
                expected = cycler if case['type'] != 'leave' else (cycler[0] + dx, cycler[1] + dy)
                self.assertEqual((env.player_cell()[0] + dx, env.player_cell()[1] + dy), expected)
                for stored, rendered in zip(case['raw_frames'], frames):
                    np.testing.assert_array_equal(stored, rendered)
                for stored, built in zip((case['frames'], case['history_valid'], case['previous_actions']),
                                         history_arrays(frames, case['raw_actions'], cp.HISTORY)):
                    np.testing.assert_array_equal(stored, built)
                self.assertEqual(case['frames'].shape, (cp.HISTORY, 64, 64))
                self.assertEqual(int(case['history_valid'].sum()), min(len(frames), cp.HISTORY))

    def test_spoiling_actions_break_the_match_in_the_engine(self):
        for case in self.cases:
            if case['type'] == 'approach':
                continue
            with self.subTest(case=case['id']):
                env = Ls20Scenario(build_level(case['spec']), case['context_index'])
                for action in case['raw_actions'][1:]:
                    env.perform(names.ACTION_IDS[action])
                env.perform(names.ACTION_IDS[case['cycler_direction']])
                if case['type'] == 'leave':
                    self.assertEqual(list(env.triple()), case['goal_triple'])
                    env.perform(names.ACTION_IDS[cp.OPPOSITE[case['cycler_direction']]])
                self.assertNotEqual(list(env.triple()), case['goal_triple'])

    def test_directions_kinds_and_contexts_are_balanced(self):
        for case_type in cp.TYPES:
            directions = Counter(c['direction'] for c in self.cases if c['type'] == case_type)
            self.assertEqual(directions, Counter({name: 2 for name in names.ACTION_NAMES}))
            contexts = Counter(c['context_index'] for c in self.cases if c['type'] == case_type)
            self.assertEqual(contexts, Counter({0: 4, 1: 4}))
        kinds = Counter(c['kind'] for c in self.cases if c['type'] == 'approach')
        self.assertEqual(set(kinds), set(cp.KINDS))
        self.assertLessEqual(max(kinds.values()) - min(kinds.values()), 1)
        self.assertEqual(cp.build_cases(3, 8)[5]['id'], self.cases[5]['id'])

    def test_cycler_chasing_stub_spoils_every_leave_and_avoid(self):
        policy = LookupPolicy(self.cases, lambda case: case['cycler_direction'])
        report = cp.evaluate(policy, self.cases, batch_size=5)
        for case_type in cp.SPOIL_TYPES:
            self.assertEqual(report['per_type'][case_type]['spoil_rate'], 1.0)
            self.assertEqual(report['per_type'][case_type]['accuracy'], 0.0)
        self.assertEqual(report['per_type']['approach']['accuracy'], 1.0)
        self.assertEqual(report['overall']['spoil_rate'], 1.0)
        self.assertGreater(report['overall']['mean_spoil_probability'], 0.9)
        for name in names.ACTION_NAMES:
            self.assertEqual(report['per_direction'][name]['avoid']['spoil_rate'], 1.0)

    def test_oracle_stub_is_perfect(self):
        policy = LookupPolicy(self.cases, lambda case: next(i for i in range(4) if case['optimal'] & (1 << i)))
        report = cp.evaluate(policy, self.cases, batch_size=7)
        self.assertEqual(report['overall']['accuracy'], 1.0)
        self.assertEqual(report['overall']['spoil_rate'], 0.0)
        for case_type in cp.TYPES:
            self.assertEqual(report['per_type'][case_type]['accuracy'], 1.0)
        self.assertEqual(len(report['rows']), 24)
        self.assertIn('leave', cp.format_table(report))

    def test_batched_inputs_match_policy_history_at_shorter_history(self):
        policy = InputSumPolicy(history=3)
        batched = cp._score_cases(policy, self.cases, 'cpu', 4)
        for case, scores in zip(self.cases, batched):
            torch.testing.assert_close(cp.history_scores(policy, case), scores)

    def test_rejects_invalid_arguments(self):
        with self.assertRaises(ValueError):
            cp.build_cases(0, 0)
        with self.assertRaises(ValueError):
            cp.build_cases(0, 2, kinds=('shape', 'size'))
        with self.assertRaises(ValueError):
            cp.evaluate(InputSumPolicy(), [])


if __name__ == '__main__':
    unittest.main()
