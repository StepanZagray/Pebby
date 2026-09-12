"""Failure coverage must survive the real engine -> arrays -> training seam."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pebby.agent import world_data as wd
from pebby.agent.world_training_objectives import _value_loss, world_losses
from pebby.agent.world_train import load_dataset
from pebby.ls20 import names
from pebby.ls20.env import Ls20Scenario
from pebby.ls20.generate import build_level
from tests.test_policy_history import corridor, RecordingPolicy, tiny_world
from tools.collect_onpolicy_world import collect_level


class FailureCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def spec(self):
        return {**corridor(budget=4), 'seed': 8, 'difficulty': 1}

    def test_failure_rows_are_three_actual_life_losses_with_causal_histories(self):
        arrays = wd.build([self.spec()], history=3, samples=8, coverage='mixed_failure')
        proof = arrays['meta']['levels'][0]
        self.assertTrue(proof['context_engine_verified'])
        self.assertEqual(proof['failure_samples'], 3)
        self.assertGreater(int((arrays['terminal'] & ~arrays['won']).sum()), 0)
        failure = np.flatnonzero(arrays['lost_life'].any(axis=1))
        self.assertEqual(set(arrays['current_lives'][failure]), {1, 2, 3})
        # The exhaustion trajectory is recorded as actions for independent replay.
        actions, indices = proof['failure_actions'], proof['failure_indices']
        for row_index, step in zip(range(len(arrays['seeds']) - 3, len(arrays['seeds'])), indices):
            env = Ls20Scenario(build_level(self.spec()), 1)
            frames, previous = [env.render()], [-1]
            for action in actions[:step]:
                lives = env.lives()
                result = env.perform(names.ACTION_IDS[action])
                if env.lives() < lives:
                    frames, previous = [result.frame], [-1]
                else:
                    frames.append(result.frame)
                    previous.append(action)
            expected = wd.history_arrays(frames, previous, 3)
            for field, value in zip(('frames', 'history_valid', 'previous_actions'), expected):
                np.testing.assert_array_equal(arrays[field][row_index], value)
            for action in range(4):
                branch = wd.clone_env(env)
                result = branch.perform(names.ACTION_IDS[action])
                np.testing.assert_array_equal(arrays['next_frames'][row_index, action], result.frame)
                self.assertEqual(arrays['terminal'][row_index, action], result.finished)
                self.assertEqual(arrays['lost_life'][row_index, action], branch.lives() < env.lives())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'failure.npz'
            wd.save(path, arrays)
            loaded = load_dataset(path)
        result = world_losses(tiny_world(), loaded, weights={'sigreg': 0.})
        self.assertTrue(torch.isfinite(result['total']))
        result['total'].backward()
        self.assertGreater(result['diagnostics']['unsafe_fraction'].item(), 0.)
        self.assertLess(result['diagnostics']['policy_valid_fraction'].item(), 1.)

    def test_onpolicy_keeps_failure_coverage_even_when_policy_wins(self):
        rows, proof, count = collect_level(self.spec(), RecordingPolicy(history=8))
        self.assertEqual(proof['stop'], 'won')
        self.assertEqual(count, 3)
        self.assertEqual(proof['failure_samples'], 3)
        self.assertTrue(any((row['terminal'] & ~row['won']).any() for row in rows[count:]))

    def test_life_loss_supervises_existing_unsafe_bin_without_faking_terminal(self):
        model = tiny_world()
        latent = torch.zeros(2, model.cfg.latent)
        distances = torch.tensor([3, 3])
        terminal = won = torch.zeros(2, dtype=torch.bool)
        logits = torch.full((2, model.bins), -20.)
        logits[0, 3], logits[1, -1] = 20., 20.
        from unittest.mock import patch
        with patch.object(model, 'value', return_value=(logits, torch.full((2,), -20.), torch.full((2,), -20.))):
            loss, accuracy = _value_loss(model, latent, distances, terminal, won,
                                         torch.tensor([False, True]))
        self.assertLess(loss.item(), 1e-6)
        self.assertEqual(accuracy.item(), 1.)

    def test_actual_reset_distance_is_preserved_but_imagined_loss_sees_event(self):
        from unittest.mock import patch
        from pebby.agent import world_training_objectives as wm
        arrays = wd.build([self.spec()], history=3, samples=4, coverage='mixed_failure')
        with patch.object(wm, '_value_loss', wraps=wm._value_loss) as loss:
            result = world_losses(tiny_world(), arrays, weights={'sigreg': 0.})
        actual, imagined = loss.call_args_list
        self.assertEqual(len(actual.args), 5)
        self.assertEqual(len(imagined.args), 6)
        reset = arrays['lost_life'] & ~arrays['terminal']
        self.assertTrue(reset.any())
        self.assertTrue((actual.args[2].numpy().reshape(-1, 4)[reset] > 0).all())
        np.testing.assert_array_equal(imagined.args[5].numpy(), arrays['lost_life'].flatten())
        self.assertEqual(int(result['diagnostic_weights']['policy']), int((arrays['optimal'] != 0).sum()))

    def test_real_long_route_has_middle_anchors_and_proportional_rows(self):
        path = []
        for index, row in enumerate(range(1, 10, 2)):
            columns = list(range(1, 11))[::(-1 if index % 2 else 1)]
            path.extend((column, row) for column in columns)
            if row < 9:
                path.append((columns[-1], row + 1))
        spec = {**self.spec(), 'start': path[0], 'step_counter': 42,
                'walls': sorted({(x, y) for x in range(12) for y in range(12)} - set(path)),
                'goals': [{'cell': path[-1], 'triple': [0, 0, 0]}],
                'refills': [path[18], path[36]]}
        rows, proof, count = collect_level(spec, RecordingPolicy(history=8), max_actions=1)
        length = proof['context_optimal_actions']
        self.assertGreater(length, 48)
        self.assertGreaterEqual(proof['expert_samples'], (length + 3) // 4)
        self.assertTrue(any(length // 3 < index < 2 * length // 3 for index in proof['expert_indices']))
        self.assertTrue(rows[count + proof['expert_samples'] - 1]['won'].any())
        env, oracle, _ = wd.verified_context(spec)
        solution = oracle.solution(seed=spec['seed'])
        frames, actions = [env.render()], [-1]
        anchor_rows = dict(zip(proof['expert_indices'], rows[count:count + proof['expert_samples']]))
        for step, action in enumerate(solution):
            if step in anchor_rows:
                expected = wd.history_arrays(frames, actions, 8)
                for field, value in zip(('frames', 'history_valid', 'previous_actions'), expected):
                    np.testing.assert_array_equal(anchor_rows[step][field], value)
            result = env.perform(action)
            frames.append(result.frame)
            actions.append(names.ACTION_IDS.index(action))

    def test_disk_row_store_grows_for_long_route_anchors_without_losing_rows(self):
        from tools.collect_onpolicy_world import RowStore
        with tempfile.TemporaryDirectory() as directory:
            store = RowStore(directory, 1)
            store.append([{'value': np.array([1, 2], dtype=np.int16)}])
            store.append([{'value': np.array([3, 4], dtype=np.int16)},
                          {'value': np.array([5, 6], dtype=np.int16)}])
            np.testing.assert_array_equal(store.arrays()['value'], [[1, 2], [3, 4], [5, 6]])

    def test_route_proportional_anchors_keep_middle_and_end(self):
        from tools.collect_onpolicy_world import expert_budget
        for length in (3, 12, 40, 79):
            count = expert_budget(length)
            indices = wd.spread_indices(list(range(length)), count)
            self.assertEqual(indices[-1], length - 1)
            self.assertTrue(any(0 < index < length - 1 for index in indices))
            self.assertGreaterEqual(count, (length + 3) // 4)


if __name__ == '__main__':
    unittest.main()
