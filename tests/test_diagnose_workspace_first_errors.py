import numpy as np
import unittest
from pathlib import Path
import torch

from tools.diagnose_workspace_first_errors import (
    aggregate_levels,
    CHECKPOINT,
    CHECKPOINT_SHA,
    resolve_checkpoint,
    score_successor_views,
    matched_successor_scores,
    successor_histories,
)


class FirstErrorHelpersTest(unittest.TestCase):
  def test_default_and_custom_checkpoint_binding(self):
    self.assertEqual(resolve_checkpoint(None, None), (CHECKPOINT, CHECKPOINT_SHA))
    self.assertEqual(resolve_checkpoint(None, CHECKPOINT_SHA), (CHECKPOINT, CHECKPOINT_SHA))
    custom = 'checkpoints/custom-evolving.pt'
    self.assertEqual(resolve_checkpoint(custom, 'A' * 64), (Path(custom), 'a' * 64))
    with self.assertRaisesRegex(ValueError, 'required'):
      resolve_checkpoint(custom, None)
    with self.assertRaisesRegex(ValueError, '64-character'):
      resolve_checkpoint(custom, 'abc')

  def test_successor_histories_append_and_reset_are_public_contract(self):
    observed = np.arange(8 * 64 * 64, dtype=np.uint8).reshape(8, 64, 64)
    valid = np.ones(8, dtype=bool)
    previous = np.array([-1, 0, 1, 2, 3, 0, 1, 2], dtype=np.int64)
    futures = np.stack([np.full((64, 64), value, dtype=np.uint8) for value in (10, 11, 12, 13)])
    lost = np.array([False, True, False, True], dtype=bool)

    histories, validity, actions = successor_histories(observed, valid, previous, futures, lost)

    self.assertEqual(histories.shape, (4, 8, 64, 64))
    self.assertEqual(validity.shape, (4, 8))
    self.assertEqual(actions.shape, (4, 8))
    # Non-loss branches shift the public H8 and append the producing action.
    np.testing.assert_array_equal(histories[0, -1], futures[0])
    np.testing.assert_array_equal(histories[0, 0], observed[1])
    np.testing.assert_array_equal(actions[0], np.array([0, 1, 2, 3, 0, 1, 2, 0]))
    self.assertTrue(validity[0].all())
    # A real life loss resets public memory: only the post-loss frame is valid.
    np.testing.assert_array_equal(histories[1, -1], futures[1])
    self.assertEqual(validity[1].tolist(), [False] * 7 + [True])
    self.assertEqual(actions[1].tolist(), [-1] * 8)


  def test_successor_score_comparison_marks_actual_and_imagined_separately(self):
    result = score_successor_views(
        actual_logits=[0.1, 0.9, 0.2, 0.3],
        imagined_logits=[0.8, 0.2, 0.1, 0.0],
        optimal_mask=0b0010,
        unsafe=[True, False, False, True],
    )
    self.assertEqual(result['actual']['choice'], 1)
    self.assertIs(result['actual']['optimal'], True)
    self.assertIs(result['actual']['unsafe'], False)
    self.assertEqual(result['imagined']['choice'], 0)
    self.assertIs(result['imagined']['optimal'], False)
    self.assertIs(result['imagined']['unsafe'], True)
    self.assertIs(result['choice_disagrees'], True)

  def test_matched_readout_must_match_the_committed_public_scores(self):
    class FakePolicy:
      def encoder(self, frames, valid, actions):
        return torch.zeros((len(frames), 148, 96))

      def successor_fields(self, fields):
        return torch.zeros((len(fields), 4, 148, 96))

      def readout(self, fields):
        return torch.arange(len(fields) * 4, dtype=torch.float32).reshape(len(fields), 4)

    targets = {'next_frames': np.zeros((4, 64, 64), dtype=np.uint8),
               'lost_life': np.zeros(4, dtype=bool),
               'distances': np.ones(4, dtype=np.int16),
               'terminal': np.zeros(4, dtype=bool), 'won': np.zeros(4, dtype=bool),
               'optimal': 1}
    observed = np.zeros((8, 64, 64), dtype=np.uint8)
    valid = np.ones(8, dtype=bool)
    previous = np.full(8, -1, dtype=np.int64)
    result, _ = matched_successor_scores(FakePolicy(), observed, valid, previous,
                                         targets, public_logits=[0., 1., 2., 3.])
    self.assertTrue(result['imagined_matches_public'])
    with self.assertRaisesRegex(ValueError, 'disagrees'):
      matched_successor_scores(FakePolicy(), observed, valid, previous, targets,
                               public_logits=[3., 2., 1., 0.])


  def test_aggregate_levels_preserves_unknown_and_mechanic_subgroups(self):
    rows = [
        {'outcome': 'first_error', 'mechanics': {'fog': True, 'launchers': False}},
        {'outcome': 'no_error', 'mechanics': {'fog': True, 'launchers': True}},
        {'outcome': 'unknown', 'mechanics': {'fog': False, 'launchers': True}},
    ]
    result = aggregate_levels(rows)
    self.assertEqual(result['first_error'], 1)
    self.assertEqual(result['no_error'], 1)
    self.assertEqual(result['unknown'], 1)
    self.assertEqual(result['mechanics']['fog'], {'levels': 2, 'first_error': 1, 'no_error': 1, 'unknown': 0})
    self.assertEqual(result['mechanics']['launchers'], {'levels': 2, 'first_error': 0, 'no_error': 1, 'unknown': 1})


  def test_successor_history_rejects_nonboolean_loss_labels(self):
    base = np.zeros((8, 64, 64), dtype=np.uint8)
    with self.assertRaisesRegex(ValueError, 'boolean'):
      successor_histories(base, np.ones(8, dtype=bool), np.full(8, -1, dtype=np.int64),
                          np.zeros((4, 64, 64), dtype=np.uint8), np.zeros(4, dtype=np.int8))
