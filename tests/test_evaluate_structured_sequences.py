import unittest
import numpy as np
import torch
from tools.evaluate_structured_sequences import score


class CountingTransition(torch.nn.Module):
    """Transparent accumulated dynamics exposes horizon and teacher-forcing mistakes."""
    def __init__(self):
        super().__init__()
        self.inputs = []

    def readout(self, field):
        value = field[:, 0, 0].long()
        output = {}
        for key, classes in [('player', 144), ('carried_shape', 6),
                             ('carried_color', 4), ('carried_rotation', 4), ('steps', 44)]:
            target = value if key in ('player', 'steps') else torch.zeros_like(value)
            output[key + '_logits'] = torch.nn.functional.one_hot(target, classes).float() * 20
        output['role_logits'] = torch.zeros(len(field), 144, 8)
        return output

    def rollout(self, initial, actions):
        self.inputs.append((initial.clone(), actions.clone()))
        fields = initial[:, None] + actions.cumsum(1)[:, :, None, None]
        return {'fields': fields,
                'readout': {k: v.reshape(len(initial), 4, *v.shape[1:])
                            for k, v in self.readout(fields.flatten(0, 1)).items()}}


class SequenceScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def data(self):
        return {'seeds': np.array([1000001, 1000002]),
                'fields': np.zeros((2, 148, 96), np.float16),
                'next_fields': np.broadcast_to(np.arange(1, 5)[None, :, None, None], (2, 4, 148, 96)).copy(),
                'actions': np.ones((2, 4), np.int64),
                'player_cell': np.zeros((2, 2), np.int16),
                'next_player_cell': np.tile(np.stack([np.arange(1, 5), np.zeros(4)], -1), (2, 1, 1)).astype(np.int16),
                'triple': np.zeros((2, 3), np.int16), 'next_triple': np.zeros((2, 4, 3), np.int16),
                'steps': np.zeros(2, np.int16), 'next_steps': np.tile(np.arange(1, 5), (2, 1))}

    def test_horizon_labels_copy_and_accumulation(self):
        model = CountingTransition()
        result = score(model, self.data(), torch.ones(48), 1)
        self.assertEqual(result['readout']['predicted']['player']['accuracy'], [1.] * 4)
        self.assertEqual(result['readout']['actual']['steps']['accuracy'], [1.] * 4)
        self.assertEqual(result['readout']['initial_field_copy']['player']['accuracy'], [0.] * 4)
        self.assertEqual(result['group_mse']['core']['prediction_mse'], [0.] * 4)
        self.assertEqual(result['group_mse']['core']['initial_copy_mse'], [1., 4., 9., 16.])
        self.assertEqual(len(model.inputs), 2)

    def test_future_targets_and_labels_never_change_rollout_inputs(self):
        data = self.data(); first = CountingTransition(); second = CountingTransition()
        score(first, data, torch.ones(48), 2)
        data['next_fields'][:] = 3
        data['next_steps'][:] = 3
        data['next_player_cell'][:, :, 0] = 3
        score(second, data, torch.ones(48), 2)
        for x, y in zip(first.inputs[0], second.inputs[0]):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
        self.assertEqual(len(first.inputs[0]), 2)


if __name__ == '__main__':
    unittest.main()
