import unittest

import torch

from tools.evaluate_workspace_reserved_zero import ReservedZero, zero_reserved


class ReservedZeroTests(unittest.TestCase):
    def test_only_reserved_channels_change_for_every_action_and_token(self):
        fields = torch.arange(2 * 4 * 148 * 96).reshape(2, 4, 148, 96).float()
        original = fields.clone()
        result = zero_reserved(fields)
        self.assertTrue(torch.equal(result[..., :85], original[..., :85]))
        self.assertEqual(int(torch.count_nonzero(result[..., 85:])), 0)
        self.assertTrue(torch.equal(fields, original))
        self.assertTrue(torch.equal(zero_reserved(result), result))

    def test_hook_scope_and_removal(self):
        module = torch.nn.Identity()
        hook = ReservedZero()
        fields = torch.ones(1, 4, 148, 96)
        handle = module.register_forward_pre_hook(hook)
        self.assertTrue(torch.equal(module(fields), zero_reserved(fields)))
        handle.remove()
        self.assertTrue(torch.equal(module(fields), fields))
        self.assertEqual(hook.calls, 1)
        self.assertEqual(hook.nonzero_values, 4 * 148 * 11)
        with self.assertRaises(ValueError):
            zero_reserved(fields[:, 0])


if __name__ == '__main__':
    unittest.main()
