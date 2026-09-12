import unittest

import numpy as np

from tools.diagnose_workspace_first_error_fields import field_group_errors


class FirstErrorFieldHelpersTest(unittest.TestCase):
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
