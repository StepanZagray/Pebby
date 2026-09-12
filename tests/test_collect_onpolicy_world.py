import unittest
from unittest.mock import patch
import torch
from tests.test_policy_history import corridor, RecordingPolicy
from tools.collect_onpolicy_world import collect_level


class OnPolicyCollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_greedy_public_history_and_expert_coverage(self):
        spec = corridor()
        spec.update(seed=8, difficulty=1)
        policy = RecordingPolicy(history=8)
        rows, proof, count = collect_level(spec, policy)
        self.assertEqual(proof['stop'], 'won')
        self.assertEqual(count, 3)
        self.assertEqual(len(policy.calls), 3)
        self.assertTrue(any(r['won'].any() for r in rows))
        self.assertEqual(proof['branch_checks']['branches'], 4*len(rows))
        self.assertEqual(policy.calls[1][2][0, -1].item(), 3)

    def test_eight_refusals_retained_then_exact_attractor_stops(self):
        spec = corridor()
        spec.update(seed=8, difficulty=1)
        # Right is a mismatching goal; left reaches the needed shape cycler.
        spec['start'] = (5, 3)
        spec['goals'][0]['triple'] = [1, 0, 0]
        spec['cyclers'] = [{'cell': (4, 3), 'kind': 'shape'}]
        policy = RecordingPolicy(history=8)
        rows, proof, count = collect_level(spec, policy)
        self.assertEqual(proof['stop'], 'exact_attractor_repeat')
        self.assertEqual(count, 9)
        self.assertEqual(proof['stats']['rows_after_refusals_8'], 1)
        self.assertEqual(proof['stats']['optimal_choices'], 0)
        self.assertEqual(len(policy.calls), count)
        self.assertTrue(all(int(row['optimal']) & 4 for row in rows[:count]))
        self.assertTrue(any(r['won'].any() for r in rows[count:]))
        self.assertTrue((rows[8]['previous_actions'] == 3).all())
        self.assertEqual(proof['branch_checks']['branches'], 4*len(rows))

    def test_unreachable_states_are_retained_and_contract_errors_are_not_silenced(self):
        spec = corridor()
        spec.update(seed=8, difficulty=1)
        spec['start'] = (8, 3)
        rows, proof, count = collect_level(spec, RecordingPolicy(history=8))
        self.assertEqual(proof['stop'], 'action_limit')
        self.assertEqual(count, 48)
        self.assertTrue(any(int(row['optimal']) == 0 for row in rows[:count]))
        self.assertGreater(proof['stats']['life_losses'], 0)
        self.assertTrue(any(row['won'].any() for row in rows))
        with patch('tools.collect_onpolicy_world.checked_expansion', side_effect=ValueError('contract mismatch')):
            with self.assertRaisesRegex(ValueError, 'contract mismatch'):
                collect_level(spec, RecordingPolicy(history=8))

    def test_validation_seed_and_history_rejected(self):
        spec = corridor()
        spec.update(seed=1000001, difficulty=1)
        with self.assertRaisesRegex(ValueError, 'training seeds'):
            collect_level(spec, RecordingPolicy(history=8))
        spec['seed'] = 8
        with self.assertRaisesRegex(ValueError, 'H8'):
            collect_level(spec, RecordingPolicy(history=3))

if __name__ == '__main__':
    unittest.main()
