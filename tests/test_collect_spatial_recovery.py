import unittest

import numpy as np
import torch

from tests.test_policy_history import RecordingPolicy, corridor
from tools.collect_spatial_recovery import collect_level


class RecoveryCollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_refusal_history_recovery_and_exact_labels(self):
        spec = corridor()
        spec.update(seed=8, difficulty=1, start=(5, 3),
                    goals=[{'cell': (6, 3), 'triple': [1, 0, 0]}],
                    cyclers=[{'cell': (4, 3), 'kind': 'shape'}])
        policy = RecordingPolicy(history=8)
        rows, proof = collect_level(spec, policy)
        policy_rows = [r for r in rows if r['row_kind'] == 0]
        self.assertEqual(len(policy_rows), 9)
        self.assertEqual(len(policy.calls), 9)
        self.assertEqual(proof['stop'], 'exact_attractor_repeat')
        self.assertEqual(proof['stats']['rows_after_refusals_8'], 1)
        recoveries = [t for t in proof['trajectories'] if t['kind'] == 'recovery']
        self.assertGreaterEqual(len(recoveries), 2)
        self.assertTrue(all(t['stop'] == 'won' for t in recoveries))
        for trajectory in recoveries:
            samples = [r for r in rows if r['trajectory_id'] == trajectory['trajectory_id']]
            actual = policy_rows[trajectory['policy_step']]
            for key in ('frames', 'history_valid', 'previous_actions'):
                np.testing.assert_array_equal(samples[0][key], actual[key])
            self.assertEqual(int(samples[0]['chosen_action']), 2)
            for sample in samples:
                choice = int(sample['chosen_action'])
                self.assertTrue(int(sample['optimal']) & (1 << choice))
                self.assertFalse(sample['lost_life'][choice])
        self.assertEqual(proof['branch_checks']['branches'], 4 * len(rows))
        self.assertTrue(any(row['won'].any() for row in rows))

    def test_successful_policy_preserved_without_artificial_recovery(self):
        spec = corridor()
        spec.update(seed=8, difficulty=1)
        rows, proof = collect_level(spec, RecordingPolicy(history=8))
        self.assertEqual(proof['stop'], 'won')
        self.assertEqual(proof['policy_rows'], 3)
        self.assertEqual(proof['recovery_rows'], 0)
        self.assertEqual(proof['stats']['optimal_choices'], 3)
        self.assertEqual(proof['branch_checks']['branches'], 4 * len(rows))

    def test_validation_seed_rejected_before_execution(self):
        spec = corridor()
        spec.update(seed=1000001, difficulty=1)
        with self.assertRaisesRegex(ValueError, 'TRAIN'):
            collect_level(spec, RecordingPolicy(history=8))


if __name__ == '__main__':
    unittest.main()
