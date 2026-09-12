import unittest

import numpy as np
import torch

from tools.diagnose_workspace_first_error_fields_v2 import field_group_errors, raw_field_semantics, replace_actual_group


class FirstErrorFieldHelpersTest(unittest.TestCase):
    def test_intervention_replaces_channels_for_all_actions_and_tokens(self):
        imagined = torch.zeros((4, 148, 96))
        actual = torch.arange(4 * 148 * 96, dtype=torch.float32).reshape(4, 148, 96)
        hybrid = replace_actual_group(imagined, actual, 48, 70)
        self.assertEqual(tuple(hybrid.shape), (4, 148, 96))
        torch.testing.assert_close(hybrid[..., 48:70], actual[..., 48:70])
        self.assertTrue(torch.equal(hybrid[..., :48], imagined[..., :48]))
        self.assertTrue(torch.equal(hybrid[..., 70:], imagined[..., 70:]))

    def test_raw_player_and_broadcast_glyph_probe_uses_expected_channels(self):
        fields = torch.zeros((1, 148, 96))
        fields[:, 12, 55] = 2
        fields[:, :, 70 + 2] = 3
        fields[:, :, 76 + 1] = 3
        fields[:, :, 80 + 3] = 3
        result = raw_field_semantics(fields, [[0, 1]], [[2, 1, 3]])
        self.assertEqual(result['player_prediction'], [12])
        self.assertEqual(result['glyph_prediction'], [[2, 1, 3]])
        self.assertEqual(result['glyph_joint_correct'], [True])
        self.assertEqual(result['glyph_token_consistency'], [1.0])

    def test_field_groups_cover_width_and_report_per_branch(self):
        actual = np.zeros((4, 148, 96), dtype=np.float32)
        imagined = np.zeros_like(actual)
        imagined[:, :, 70:84] = 2.0
        result = field_group_errors(actual, imagined)
        self.assertEqual(set(result), {'encoder_state', 'appearance', 'carried_glyph',
                                       'visibility', 'reserved'})
        self.assertEqual(result['carried_glyph']['per_branch_mse'], [4.0] * 4)
        self.assertEqual(result['encoder_state']['mse'], 0.0)

    def test_field_groups_reject_wrong_shape(self):
        with self.assertRaisesRegex(ValueError, 'four actual'):
            field_group_errors(np.zeros((4, 148, 95), dtype=np.float32),
                               np.zeros((4, 148, 95), dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
