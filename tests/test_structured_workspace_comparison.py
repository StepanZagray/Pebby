import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from pebby.agent.structured_policy import StructuredPolicyReadout
from pebby.agent.structured_workspace_policy import StructuredWorkspaceReadout
from tools.train_structured_workspace_comparison import draws, selection_bytes, save_head, load_head, verify_sources, evaluate
from tools.preflight_structured_workspace import backward
from tools.structured_policy_batch import prepared_policy_inputs
from tools.train_structured_policy import policy_terms
from tools.train_structured_transition import digest


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)
    def setUp(self): torch.manual_seed(42)

    def test_paired_stream_distinct_deterministic_independent_depth(self):
        data = {'seeds': np.arange(2048), 'difficulties': np.arange(2048) % 5 + 1}
        a, b = list(draws(data, 1024, 20)), list(draws(data, 1024, 20))
        for (rows, view, depth), (other, ov, od) in zip(a, b):
            np.testing.assert_array_equal(rows, other)
            self.assertEqual((view, depth), (ov, od))
            self.assertEqual(len(np.unique(data['seeds'][rows])), 1024)
        self.assertEqual(set(v[2] for v in a), {1, 2, 4})
        self.assertNotEqual(selection_bytes(a[0][0], np.zeros(1024), 1), selection_bytes(a[0][0], np.zeros(1024), 2))

    def test_two_backward_equal_joint_loss_gradients(self):
        old = StructuredPolicyReadout({'mode': 'successors'})
        with torch.no_grad(): old.scorer.weight.normal_(std=.1)
        a = StructuredWorkspaceReadout.from_readout(old)
        with torch.no_grad():
            a.workspace.attention_gate.fill_(.1); a.workspace.mlp_gate.fill_(.1)
        b = StructuredWorkspaceReadout(a.config()); b.load_state_dict(a.state_dict())
        rng = np.random.default_rng(4)
        batch = {key: rng.normal(size=(2, 4, 148, 96)).astype('float32') for key in ('next_fields', 'imagined_fields')}
        batch['optimal'] = np.array([3, 4], np.uint8)
        backward(a, batch, 'cpu', 2)
        inputs, masks = prepared_policy_inputs(batch, 'successors', 'cpu')
        loss = sum(.5 * policy_terms(b(x, loops=2), masks)['ce'].mean() for x in inputs.values())
        loss.backward()
        for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=1e-5, msg=name)

    def test_strict_roundtrip_and_source_drift(self):
        head = StructuredWorkspaceReadout.from_readout(StructuredPolicyReadout({'mode': 'successors'}))
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source'; source.write_text('frozen')
            sources = {str(source): digest(source)}
            path = Path(folder) / 'head.pt'
            save_head(path, head, sources, {'smoke': True}, actor_checkpoint=str(source))
            loaded, saved = load_head(path)
            self.assertEqual(loaded.config(), head.config())
            saved['weights']['scorer.bias'] += 1
            torch.save(saved, path)
            with self.assertRaisesRegex(ValueError, 'weight digest'): load_head(path)
            source.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'source changed'): verify_sources(sources)

    def test_evaluation_counts_and_chunk_invariance(self):
        head = StructuredWorkspaceReadout.from_readout(StructuredPolicyReadout({'mode': 'successors'}), evolving=False)
        rng = np.random.default_rng(7)
        data = {key: rng.normal(size=(3, 4, 148, 96)).astype('float32') for key in ('next_fields', 'imagined_fields')}
        data.update(optimal=np.array([1, 1, 1], np.uint8), distances=np.array([[1, 2, 3, 4], [16, 17, 18, 19], [20, 21, 22, 23]]))
        for key in ('terminal', 'won', 'lost_life'): data[key] = np.zeros((3, 4), bool)
        a, b = evaluate(head, data, np.arange(3), 1), evaluate(head, data, np.arange(3), 3)
        for depth in ('1', '2', '4'):
            for kind in ('actual', 'imagined'):
                for group, count in [('all', 3), ('distance_ge17', 2)]:
                    self.assertEqual(a[depth][kind][group]['count'], count)
                    self.assertEqual(a[depth][kind][group]['correct'], count)
                    self.assertAlmostEqual(a[depth][kind][group]['ce'], b[depth][kind][group]['ce'], places=6)

if __name__ == '__main__': unittest.main()
