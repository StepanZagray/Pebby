"""Structured policies must receive the same causal H8 as feature training."""
import unittest

import numpy as np
import torch

from pebby.agent.history import for_policy


class PublicHistoryPolicy:
    def config(self):
        return {'architecture': 'structured', 'history': 8}

    def __call__(self, frames, history_valid, previous_actions):
        self.inputs = (frames, history_valid, previous_actions)
        return torch.tensor([[0., 1., 2., 3.]])


class StructuredHistoryTests(unittest.TestCase):
    def test_h8_padding_actions_and_reset_are_causal(self):
        policy = PublicHistoryPolicy()
        first = np.zeros((64,64), dtype=np.uint8).tolist()
        second = np.ones((64,64), dtype=np.uint8).tolist()
        history = for_policy(policy, first, 'cpu')
        history.observe(second, 2)
        self.assertEqual(history.scores().tolist(), [0.,1.,2.,3.])
        frames, valid, actions = policy.inputs
        self.assertEqual(tuple(frames.shape), (1,8,64,64))
        self.assertEqual(valid.tolist(), [[False]*6+[True,True]])
        self.assertEqual(actions.tolist(), [[-1]*7+[2]])
        self.assertTrue((frames[0,:7]==0).all())
        self.assertTrue((frames[0,7]==1).all())
        history.observe(first, 1, reset=True)
        history.scores()
        frames, valid, actions = policy.inputs
        self.assertTrue((frames==0).all())
        self.assertEqual(valid.tolist(), [[False]*7+[True]])
        self.assertEqual(actions.tolist(), [[-1]*8])

    def test_non_temporal_policy_keeps_single_frame_route(self):
        class SingleFrame:
            def config(self): return {'architecture':'cnn'}
        self.assertIsNone(for_policy(SingleFrame(), [[0]]))


if __name__ == '__main__': unittest.main()
