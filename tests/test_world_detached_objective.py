"""CPU gradient and provenance contracts for the isolated prediction experiment."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from pebby.agent import world_detached_objective as detached
from pebby.agent import world_train, world_training_objectives as base
from pebby.agent.world_model import load_world_checkpoint, save_world_checkpoint
from tests.test_world_model import make_model, make_synthetic, to_batch
from tests.test_world_glyph import wake
from tools import train_reference_detached as wrapper


class DetachedObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(7)
        self.model = wake(make_model(grounding=True)).eval()
        self.batch = to_batch(make_synthetic(seed=3, levels=1, steps=3))
        b = len(self.batch['frames'])
        self.batch.update(next_player_cell=self.batch['player_cell'][:, None].expand(-1, 4, -1),
                          current_triple=torch.zeros(b, 3, dtype=torch.long),
                          next_triple=torch.zeros(b, 4, 3, dtype=torch.long),
                          current_steps=torch.full((b,), 20), next_steps=torch.full((b, 4), 19),
                          current_lives=torch.full((b,), 3), next_lives=torch.full((b, 4), 3),
                          next_optimal=(~self.batch['terminal']).long())
        self.weights = {'successor_policy': .7, 'prediction': 1.3, 'grounding': .9}

    def run_loss(self, fn):
        return fn(self.model, self.batch, self.weights, loops=1,
                  sigreg_generator=torch.Generator().manual_seed(33))

    def test_exact_forward_values_and_order_for_every_output(self):
        expected = self.run_loss(base.world_losses)
        actual = self.run_loss(detached.world_losses)

        def equal(a, b):
            if isinstance(a, dict):
                self.assertEqual(list(a), list(b))
                for key in a:
                    equal(a[key], b[key])
            elif isinstance(a, torch.Tensor):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            else:
                self.assertEqual(a, b)
        equal(expected, actual)

    def test_only_prediction_target_gradient_is_removed(self):
        for fn, attached in ((base.world_losses, True), (detached.world_losses, False)):
            with self.subTest(objective=fn.__module__):
                out = self.run_loss(fn)
                prediction = out['losses']['prediction']
                target_grad, predicted_grad = torch.autograd.grad(
                    prediction, (out['targets'], out['predicted']), allow_unused=True, retain_graph=True)
                if attached:
                    self.assertGreater(target_grad.abs().sum().item(), 0)
                else:
                    self.assertIsNone(target_grad)
                self.assertGreater(predicted_grad.abs().sum().item(), 0)
                self.model.zero_grad(set_to_none=True)
                prediction.backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for p in self.model.predictor.parameters()))

    def test_each_actual_state_objective_retains_identical_nonzero_gradient(self):
        gradients = []
        for fn in (base.world_losses, detached.world_losses):
            encodings = []
            original = self.model.assemble

            def assemble(*args, **kwargs):
                out = original(*args, **kwargs)
                encodings.append(out['latent'])
                return out

            with patch.object(self.model, 'assemble', side_effect=assemble):
                out = self.run_loss(fn)
            terms = {}
            for name in ('grounding', 'value', 'sigreg', 'successor_policy'):
                grad, = torch.autograd.grad(out['losses'][name], encodings[1], retain_graph=True)
                self.assertGreater(grad.abs().sum().item(), 0, name)
                terms[name] = grad
            gradients.append(terms)
        for name in gradients[0]:
            torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=0, atol=0)

    def test_scoped_hooks_provenance_roundtrip_guard_and_failure_cleanup(self):
        original_loss = world_train.world_losses
        original_source = world_train.training_objective_source
        original_guard = wrapper.repair.source_guard
        original_write = wrapper.repair.write
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            guard = wrapper.repair.SourceGuard([], [])
            with patch.object(wrapper.repair, 'source_guard', return_value=guard):
                with self.assertRaisesRegex(RuntimeError, 'test exit'):
                    with wrapper.objective_context():
                        self.assertIs(world_train.world_losses, detached.world_losses)
                        record = world_train.training_objective_source()
                        self.assertEqual(record['module'], detached.__name__)
                        self.assertEqual(record['base_objective']['module'], base.__name__)
                        for binding in (record, record['base_objective'], record['wrapper']):
                            self.assertEqual(binding['sha256'], wrapper.repair.digest(binding['path']))
                        guarded = wrapper.repair.source_guard()
                        for binding in (record, record['base_objective'], record['wrapper']):
                            self.assertEqual(guarded.hashes[binding['path']], binding['sha256'])
                        guarded.verify()
                        wrapper.repair.write(path / 'provenance.json', {'optimizer_state': 'new'})
                        self.assertEqual(json.loads((path / 'provenance.json').read_text())[
                            'training_objective_source'], record)
                        save_world_checkpoint(path / 'model.pt', self.model, training_objective_source=record)
                        _, checkpoint = load_world_checkpoint(path / 'model.pt')
                        self.assertEqual(checkpoint['training_objective_source'], record)
                        raise RuntimeError('test exit')
        self.assertIs(world_train.world_losses, original_loss)
        self.assertIs(world_train.training_objective_source, original_source)
        self.assertIs(wrapper.repair.source_guard, original_guard)
        self.assertIs(wrapper.repair.write, original_write)

    def test_wrapper_preserves_matched_onpolicy_training_arguments(self):
        base_arguments = wrapper.repair.training_arguments
        parent = {'weight_decay': .01, 'config': {},
                  'loss_weights': dict(detached.DEFAULT_WEIGHTS)}
        observed = []
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'train.npz', Path(directory) / 'out'
            options = SimpleNamespace(out_dir=output, seed=42, lr=.0001,
                                      grounding_weight=1., sigreg_weight=.1)
            expected = base_arguments(options, parent) + [
                '--on-policy-data', str(source), '--on-policy-fraction', '.25',
                '--on-policy-auxiliary-fraction', '0']

            def main(argv):
                self.assertEqual(argv, ['--out-dir', str(output), '--grounding-weight', '1',
                                       '--sigreg-weight', '.1', '--lr', '.0001', '--seed', '42'])
                self.assertEqual(wrapper.repair.training_arguments(options, parent), expected)
                self.assertIs(world_train.world_losses, detached.world_losses)
                self.assertEqual(wrapper.repair.STEPS, 318)
                observed.append(True)

            with patch.object(wrapper.repair, 'memory_check'), \
                    patch.object(wrapper.onpolicy, 'validate_supplement', return_value=(source, 'synthetic')), \
                    patch.object(wrapper.repair, 'main', side_effect=main):
                wrapper.main(['--data-dir', directory, '--out-dir', str(output)])
        self.assertEqual(observed, [True])
        self.assertIs(wrapper.repair.training_arguments, base_arguments)


if __name__ == '__main__':
    unittest.main()
