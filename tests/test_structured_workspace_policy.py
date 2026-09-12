import unittest
import torch
from pebby.agent.structured_policy import StructuredPolicyReadout
from pebby.agent.structured_workspace_policy import StructuredWorkspaceReadout


class WorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def old(self, mode='successors', loops=2):
        old = StructuredPolicyReadout({'mode': mode, 'loops': loops})
        with torch.no_grad():
            old.scorer.weight.normal_(std=.1)
        return old

    def test_neutral_exact_legacy_both_modes_all_depths(self):
        for mode in ('direct', 'successors'):
            field = torch.randn((2, 4, 148, 96) if mode == 'successors' else (2, 148, 96))
            for depth in (1, 2, 4):
                old = self.old(mode, depth)
                for evolving in (False, True):
                    new = StructuredWorkspaceReadout.from_readout(old, evolving=evolving)
                    torch.testing.assert_close(old(field), new(field), atol=0, rtol=0)

    def test_gates_then_interior_gradients_and_immutable_source(self):
        new = StructuredWorkspaceReadout.from_readout(self.old())
        fields = torch.randn(2, 4, 148, 96, requires_grad=True)
        original = fields.detach().clone()
        new(fields).square().sum().backward()
        self.assertGreater(abs(float(new.workspace.attention_gate.grad)), 0)
        self.assertEqual(float(new.workspace.attention.in_proj_weight.grad.abs().sum()), 0)
        new.zero_grad()
        with torch.no_grad():
            new.workspace.attention_gate.fill_(.2)
            new.workspace.mlp_gate.fill_(.2)
        new(fields, loops=4).square().sum().backward()
        for name, p in new.workspace.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(float(p.grad.abs().sum()), 0, name)
        torch.testing.assert_close(fields.detach(), original, atol=0, rtol=0)

    def test_board_propagation_and_original_recall_all_loops(self):
        new = StructuredWorkspaceReadout.from_readout(self.old('direct'))
        with torch.no_grad():
            new.workspace.attention_gate.fill_(1)
        source = torch.randn(1, 148, 96, requires_grad=True)
        # A distant token affects token0 through the workspace itself, not pooling.
        changed = new.workspace(source, source)
        gradient = torch.autograd.grad(changed[0, 0, 0], source)[0]
        self.assertGreater(float(gradient[0, 143].abs().sum()), 0)
        seen = []
        hook = new.workspace.register_forward_pre_hook(lambda m, args: seen.append(args))
        new(source, loops=4)
        hook.remove()
        self.assertEqual(len(seen), 4)
        for i in range(1, 4):
            self.assertIs(seen[i][1], seen[0][1])
            self.assertIsNot(seen[i][0], seen[0][0])

    def test_counts_roundtrip_permutation_and_validation(self):
        new = StructuredWorkspaceReadout.from_readout(self.old())
        self.assertEqual(new.parameter_count(), 201315)
        self.assertEqual(new.trainable_parameter_count(), 201315)
        static = StructuredWorkspaceReadout.from_readout(self.old(), evolving=False)
        self.assertEqual(static.parameter_count(), new.parameter_count())
        self.assertEqual(static.trainable_parameter_count(), 108097)
        field = torch.randn(2, 4, 148, 96)
        permutation = torch.tensor([2, 0, 3, 1])
        reference = new(field)
        torch.testing.assert_close(new(field[:, permutation], permutation[None].expand(2, -1)), reference[:, permutation])
        restored = StructuredWorkspaceReadout(new.config())
        restored.load_state_dict(new.state_dict())
        torch.testing.assert_close(restored(field), reference, atol=0, rtol=0)
        count = new.parameter_count()
        for depth in (1, 2, 4):
            new(field, loops=depth)
            self.assertEqual(new.parameter_count(), count)
        for depth in (0, True, 1.5):
            with self.assertRaises(ValueError): new(field, loops=depth)
        with self.assertRaises(ValueError): new(field, torch.zeros(2, 4, dtype=torch.long))
        with self.assertRaises(ValueError): new(field[:, :3])
        with self.assertRaises(TypeError): new(field, targets=field)
        with self.assertRaises(ValueError): new.load_from(self.old('direct'))

    def test_checkpoint_output_and_gradient_parity(self):
        a = StructuredWorkspaceReadout.from_readout(self.old())
        with torch.no_grad():
            a.workspace.attention_gate.fill_(.2)
            a.workspace.mlp_gate.fill_(.1)
        b = StructuredWorkspaceReadout({**a.config(), 'checkpoint_workspace': True})
        b.load_state_dict(a.state_dict())
        x = torch.randn(2, 4, 148, 96)
        ya, yb = a(x, loops=4), b(x, loops=4)
        torch.testing.assert_close(ya, yb, atol=0, rtol=0)
        ya.square().sum().backward(); yb.square().sum().backward()
        for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0, msg=name)


if __name__ == '__main__': unittest.main()
