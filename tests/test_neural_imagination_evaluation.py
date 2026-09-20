import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from pebby.ls20 import names
from tools.evaluate_neural_imagination import fidelity_metrics, initial_fidelity, replay_fixed_actions


class Branch:
    def __init__(self, finish=3):
        self.calls = []
        self.remaining = 3
        self.finish = finish

    def lives(self):
        return self.remaining

    def player_cell(self):
        return (len(self.calls), 2)

    def triple(self):
        return (1, 2, 3)

    def steps_left(self):
        return 10 - len(self.calls)

    def perform(self, action):
        self.calls.append(action)
        if len(self.calls) == 1:
            self.remaining -= 1
        done = len(self.calls) == self.finish
        return SimpleNamespace(finished=done, won=done)


class FidelityEvaluationTests(unittest.TestCase):
    def test_imagination_and_dynamics_receive_fixed_traces_before_real_branches(self):
        branches = [Branch(1) for _ in range(4)]
        actions = torch.tensor([[[0, 3], [1, 2], [2, 1], [3, 0]]])
        calls = []

        def encoder(frames, valid, previous):
            self.assertEqual(frames.shape, (1, 8, 64, 64))
            self.assertEqual(valid.sum(), 1)
            self.assertTrue((previous == -1).all())
            calls.append('encoder')
            return torch.zeros(1, 148, 96)

        def imagine(field):
            self.assertTrue(all(not branch.calls for branch in branches))
            calls.append('imagine')
            return dict(imagined_actions=actions, root_actions=torch.arange(4)[None], action_logits=torch.zeros(1, 4))

        def dynamics(field, trace):
            self.assertTrue(all(not branch.calls for branch in branches))
            torch.testing.assert_close(trace, actions[0])
            calls.append('dynamics')
            return dict(readout={key: torch.zeros(4, 2, count) for key, count in
                                 [('player_logits', 144), ('steps_logits', 44), ('lives_logits', 4)]},
                        glyph_logits={key: torch.zeros(4, 2, count) for key, count in [('shape', 6), ('color', 4), ('rotation', 4)]},
                        events={key + '_logits': torch.zeros(4, 2) for key in ('lost_life', 'terminal', 'won')})

        policy = SimpleNamespace(encoder=encoder, planner=SimpleNamespace(imagine=imagine, dynamics=SimpleNamespace(rollout=dynamics)))
        env = SimpleNamespace(reset=lambda: [[0] * 64 for _ in range(64)])
        with patch('tools.evaluate_neural_imagination.clone_env', side_effect=branches):
            result = initial_fidelity(policy, env)
        self.assertEqual(calls, ['encoder', 'imagine', 'dynamics'])
        self.assertEqual([branch.calls for branch in branches], [[action] for action in names.ACTION_IDS])
        self.assertTrue(all(not branch['steps'][1]['valid'] for branch in result['branches']))

    def test_fixed_actions_survive_life_loss_and_terminal_masks_tail(self):
        original = object()
        branch = Branch()
        with patch('tools.evaluate_neural_imagination.clone_env', return_value=branch) as clone:
            actual = replay_fixed_actions(original, [3, 0, 2, 1])
        clone.assert_called_once_with(original)
        self.assertEqual(branch.calls, [names.ACTION_IDS[i] for i in (3, 0, 2)])
        self.assertTrue(actual[0]['lost_life'])
        self.assertFalse(actual[1]['lost_life'])
        self.assertEqual(actual[1]['lives'], 2)
        self.assertTrue(actual[2]['terminal'])
        self.assertTrue(actual[2]['won'])
        self.assertIsNone(actual[3])
        self.assertEqual(actual[0]['player'], 25)

    def test_metric_denominators_exclude_unexecuted_terminal_tails(self):
        with patch('tools.evaluate_neural_imagination.clone_env', side_effect=[Branch(1), Branch(3)]):
            traces = [replay_fixed_actions(object(), [0, 1, 2, 3]) for _ in range(2)]
        branches = []
        for trace in traces:
            rows = []
            for truth in trace:
                rows.append(dict(valid=truth is not None, actual=truth,
                                 predicted={key: truth[key] for key in ('player', 'glyph', 'budget_class', 'lives')} if truth else {},
                                 event_probability={key: float(truth[key]) for key in ('lost_life', 'terminal', 'won')} if truth else {}))
            branches.append(dict(steps=rows))
        result = fidelity_metrics([dict(branches=branches)])
        self.assertEqual([row['state']['player']['count'] for row in result], [2, 1, 1, 0])
        self.assertEqual([row['events']['won']['positives'] for row in result], [1, 0, 1, 0])
        self.assertEqual(result[1]['events']['lost_life']['positives'], 0)
        self.assertEqual(result[0]['state']['glyph']['accuracy'], 1.)
        self.assertIsNone(result[3]['state']['glyph']['accuracy'])

    def test_invalid_actions_are_rejected_before_engine_clone(self):
        with patch('tools.evaluate_neural_imagination.clone_env') as clone:
            for actions in ([], [4], [-1], [True], [1.5]):
                with self.assertRaises(ValueError):
                    replay_fixed_actions(object(), actions)
        clone.assert_not_called()


if __name__ == '__main__':
    unittest.main()
