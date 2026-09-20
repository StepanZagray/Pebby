import copy
from pathlib import Path
import tempfile
import unittest

import torch

from tools.train_joint_goal_replay import gameplay_key, save_training_state, verify_specs


def sequential(completed):
    return dict(evaluation_kind='actual_sequential_gameplay', source_unchanged=True,
                levels_completed=completed)


def generated(completed, levels=7):
    return dict(completed=completed, levels=levels,
                runs=[dict(completed=i < completed) for i in range(levels)])


class GameplaySelectionTests(unittest.TestCase):
    def test_shipped_progress_precedes_generated_wins(self):
        self.assertGreater(gameplay_key(sequential(2), generated(0)),
                           gameplay_key(sequential(1), generated(7)))

    def test_cached_accuracy_does_not_break_gameplay_tie(self):
        first, second = generated(2), generated(2)
        first.update(accuracy=.1, loss=9., actions=1000)
        second.update(accuracy=1., loss=0., actions=1)
        self.assertEqual(gameplay_key(sequential(1), first), gameplay_key(sequential(1), second))

    def test_partial_or_unverified_gameplay_is_not_ranked(self):
        partial = generated(3)
        partial['runs'].pop()
        for seq, val in [(sequential(1), partial),
                         (dict(sequential(1), source_unchanged=False), generated(3)),
                         (dict(sequential(1), evaluation_kind='offline'), generated(3))]:
            with self.subTest(seq=seq, val=val), self.assertRaises(ValueError):
                gameplay_key(seq, val)

    def test_seven_tier_split_and_layout_separation(self):
        train = [dict(seed=i, difficulty=i, split='train', source='generated_only',
                      geometry_split='train', start=[i, 1]) for i in range(1, 8)]
        validation = [dict(seed=100+i, difficulty=i, split='validation', source='generated_only',
                           geometry_split='validation', start=[i, 2]) for i in range(1, 8)]
        verify_specs(train, validation)
        overlap = copy.deepcopy(validation)
        overlap[0]['start'] = train[0]['start']
        for bad in (validation[:-1], overlap, [dict(s, split='train') for s in validation]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                verify_specs(train, bad)


class TrainingStateSaveTests(unittest.TestCase):
    def test_optimizer_progress_round_trips_through_a_hidden_temporary(self):
        # torch.save names the archive after the temporary file's stem, so a
        # dot-leading name without a suffix used to abort the run at its first
        # snapshot, after the gameplay evaluation had already been paid for.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'step-000500.training.pt'
            state = dict(format='pebby.joint-goal-training-state.v1', step=500,
                         torch_rng=torch.get_rng_state(), optimizer=dict(state={}, param_groups=[]))
            save_training_state(path, state)
            restored = torch.load(path, weights_only=False)
            self.assertEqual(restored['step'], 500)
            self.assertTrue(torch.equal(restored['torch_rng'], state['torch_rng']))
            self.assertEqual([entry.name for entry in Path(directory).iterdir()], [path.name])


if __name__ == '__main__':
    unittest.main()
