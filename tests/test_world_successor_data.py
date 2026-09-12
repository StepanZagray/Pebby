"""Successor action labels agree with a second round of actual engine actions."""
import unittest
from types import SimpleNamespace

import numpy as np

from pebby.agent import world_data
from pebby.ls20 import generate, names


class SuccessorDataTests(unittest.TestCase):
    def test_masks_match_real_second_actions_in_generated_contexts(self):
        spec = generate.generate_legacy_level(0, 1)
        checked = 0
        for context in (0, 3, 6):
            env, oracle, proof = world_data.verified_context(spec, context)
            self.assertIsNotNone(env, proof)
            for _ in range(4):
                before = oracle.distance_for(oracle.state_of(env))
                targets, branches, results, _ = world_data._expand(env, oracle, before, 0, 0)
                self.assertEqual(targets['next_optimal'].dtype, np.uint8)
                for index, (branch, result) in enumerate(zip(branches, results)):
                    distance = oracle.distance_for(oracle.state_of(branch))
                    expected = 0
                    if not result.finished and distance is not None and distance > 0:
                        for action, game_action in enumerate(names.ACTION_IDS):
                            second = world_data.clone_env(branch)
                            outcome = second.perform(game_action)
                            remaining = 0 if outcome.won else oracle.distance_for(oracle.state_of(second))
                            if remaining == distance - 1 and second.lives() == branch.lives():
                                expected |= 1 << action
                        self.assertNotEqual(expected, 0)
                    self.assertEqual(int(targets['next_optimal'][index]), expected)
                    checked += 1
                action = oracle.action_for(oracle.state_of(env))
                env = branches[action]
                if results[action].finished:
                    break
        self.assertGreaterEqual(checked, 24)

    def test_no_labels_for_finished_or_unreachable_states_and_truncation_refused(self):
        incomplete = SimpleNamespace(truncated=True)
        with self.assertRaisesRegex(ValueError, 'complete oracle'):
            world_data.successor_optimal_mask(incomplete, None, terminal=True)
        complete = SimpleNamespace(truncated=False, distance_for=lambda _: None)
        self.assertEqual(world_data.successor_optimal_mask(complete, None), 0)
        self.assertEqual(world_data.successor_optimal_mask(complete, None, terminal=True), 0)

    def test_alive_life_loss_reset_gets_its_own_action_target(self):
        spec = generate.generate_legacy_level(0, 1)
        free = {(3, 3), (4, 3), (5, 3)}
        spec = {**spec, 'start': (3, 3), 'start_triple': [0, 0, 0],
                'walls': sorted({(x, y) for x in range(12) for y in range(12)} - free),
                'goals': [{'cell': (5, 3), 'triple': [0, 0, 0]}],
                'cyclers': [], 'launchers': [], 'refills': [],
                'step_counter': 1, 'step_cost': 1}
        rows, _ = world_data.collect_level(spec, samples=2, epsilon=0., context_index=3)
        row = rows[1]
        reset = row['lost_life'] & ~row['terminal']
        self.assertTrue(reset.any())
        np.testing.assert_array_equal(row['next_optimal'][reset], 8)  # reset needs right
        self.assertTrue(np.all(row['next_optimal'][row['terminal']] == 0))


if __name__ == '__main__':
    unittest.main()
