"""Real generated-engine tests for fresh public inputs and separate targets."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pebby.agent import joint_goal_data as data
from pebby.agent.world_data import clone_env, verified_context
from pebby.ls20 import names
from tools.collect_joint_goal_qualification import main


class JointGoalDataTests(unittest.TestCase):
    def fixture(self, index):
        spec = data.qualification_specs()[index]
        env, oracle, proof = verified_context(spec, search_limit=50_000)
        self.assertIsNotNone(env, proof)
        return spec, env, oracle, ([env.render()], [-1])

    def move(self, env, history, action):
        old_lives = env.lives()
        observation = env.perform(names.ACTION_IDS[action])
        return data.observe(history, observation.frame, action, env.lives() < old_lives)

    def test_actual_matched_cycler_exit_and_destructive_return_labels(self):
        spec, env, oracle, history = self.fixture(3)
        history = self.move(env, history, 3)  # Generated RIGHT activates the rotation cycler.
        self.assertEqual(env.triple(), (5, 1, 0))
        covered = data.state_targets(spec, env, history[0][-1])
        self.assertFalse(covered['semantic_valid'][4 * 12 + 3])
        history = self.move(env, history, 0)  # Leave the cycler toward the generated goal.
        before = copy.deepcopy(history)
        row = data.collect_root(spec, env, oracle, history)
        self.assertEqual(row['next_triple'][1].tolist(), [5, 1, 1])
        self.assertEqual(row['next_triple'][3].tolist(), [5, 1, 0])
        self.assertEqual(int(row['optimal']), 1 << 3)
        self.assertEqual(set(data.public_inputs(row)), set(data.PUBLIC_KEYS))
        for action in range(4):
            branch = clone_env(env)
            observation = branch.perform(names.ACTION_IDS[action])
            np.testing.assert_array_equal(row['next_frames'][action], observation.frame)
            self.assertEqual(row['next_player_cell'][action].tolist(), list(branch.player_cell()))
            public = data.successor_history(row, action)
            self.assertEqual(set(public), set(data.PUBLIC_KEYS))
            np.testing.assert_array_equal(public['frames'][-1], observation.frame)
            self.assertEqual(int(public['previous_actions'][-1]), action)
        np.testing.assert_array_equal(history[0], before[0])
        self.assertEqual(history[1], before[1])

    def test_refill_consumption_and_solved_goal_are_current_dynamic_labels(self):
        spec, env, _, history = self.fixture(6)
        initial = data.state_targets(spec, env, history[0][-1])
        self.assertTrue(initial['roles'][7 * 12 + 4, 6])
        history = self.move(env, history, 3)
        history = self.move(env, history, 3)
        following = data.state_targets(spec, env, history[0][-1])
        self.assertFalse(following['roles'][7 * 12 + 4, 6])
        self.assertEqual(int(following['steps']), 3)
        spec, env, _, history = self.fixture(2)
        history = self.move(env, history, 3)
        history = self.move(env, history, 3)
        following = data.state_targets(spec, env, history[0][-1])
        solved = 6 * 12 + 4
        self.assertTrue(following['goal_presence'][solved])
        self.assertTrue(following['goal_solved'][solved])
        self.assertFalse(following['roles'][solved, 1])
        self.assertFalse(following['goal_attribute_valid'][solved])
        self.assertTrue(following['goal_presence'][6 * 12 + 7])
        self.assertFalse(following['goal_solved'][6 * 12 + 7])

    def test_real_life_loss_clears_successor_history_and_unknown_targets_stay_masked(self):
        spec, env, oracle, history = self.fixture(6)
        for _ in range(3):
            history = self.move(env, history, 0)  # Wall bumps deplete this three-fuel level.
        row = data.collect_root(spec, env, oracle, history)
        self.assertTrue(row['current_distance_valid'])
        self.assertEqual(int(row['current_distance']), -1)
        self.assertEqual(int(row['optimal']), 0)
        self.assertTrue(row['lost_life'][0])
        public = data.successor_history(row, 0)
        self.assertEqual(int(public['history_valid'].sum()), 1)
        self.assertTrue((public['previous_actions'] == -1).all())
        np.testing.assert_array_equal(public['frames'][-1], row['next_frames'][0])
        self.assertEqual(int(row['next_lives'][0]), 2)
        no_frame = data.state_targets(spec, env, None)
        for key in ('visible', 'support', 'semantic_valid', 'goal_attribute_valid'):
            self.assertFalse(no_frame[key].any())
        masked = dict(row, next_frame_valid=row['next_frame_valid'].copy())
        masked['next_frame_valid'][0] = False
        self.assertIsNone(data.successor_history(masked, 0))

    def test_train_and_complete_solver_boundary_rejects_invalid_inputs(self):
        spec, env, oracle, history = self.fixture(0)
        for changed in ({'split': 'validation'}, {'source': 'shipped'}, {'seed': 1_000_000},
                        {'search_truncated': True}, {'search_limit': 60_000}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                data.check_spec(spec | changed)
        oracle.truncated = True
        with self.assertRaisesRegex(ValueError, 'complete oracle'):
            data.collect_root(spec, env, oracle, history)

    def test_eight_level_roundtrip_exact_sources_masks_and_no_cached_features(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / 'qualification'
            manifest = main(['--out', str(out), '--max-seconds', '180'])
            arrays, specs, loaded = data.load_dataset(out)
            self.assertEqual(loaded['data_sha256'], manifest['data_sha256'])
            self.assertEqual(len(specs), 8)
            self.assertEqual(len({data.spec_digest(spec) for spec in specs}), 8)
            self.assertFalse(loaded['full_seven_tier_coverage'])
            self.assertTrue(all(proof['teacher_replay_won'] for proof in loaded['levels']))
            self.assertTrue(all(proof['refill_necessity_verified'] for proof in loaded['levels'][-2:]))
            self.assertEqual(set(arrays['collection'].tolist()), set(data.COLLECTIONS.values()))
            self.assertTrue(arrays['lost_life'].any())
            self.assertTrue(arrays['won'].any())
            self.assertTrue((arrays['terminal'] & ~arrays['won']).any())
            self.assertTrue(arrays['goal_solved'].any())
            self.assertTrue((arrays['visible'] & ~arrays['support']).any())
            self.assertTrue((~arrays['optimal_valid']).any())
            self.assertFalse(any('field' in key for key in arrays))
            for action in range(4):
                selected = arrays['optimal'] & (1 << action) != 0
                self.assertFalse(arrays['lost_life'][selected, action].any())
                np.testing.assert_array_equal(arrays['distances'][selected, action], arrays['current_distance'][selected] - 1)
            mutated = dict(arrays, next_history_reset=~arrays['next_history_reset'])
            with self.assertRaisesRegex(ValueError, 'observation/event'):
                data.validate_arrays(mutated, specs)
            # A provenance mutation must fail before any dataset reaches fitting.
            bank = out / 'bank.json'
            bank.write_text(bank.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                data.load_dataset(out)
            self.assertEqual(hashlib.sha256((out / 'data.npz').read_bytes()).hexdigest(), manifest['data_sha256'])


if __name__ == '__main__':
    unittest.main()
