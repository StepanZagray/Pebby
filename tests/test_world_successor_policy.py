"""Optional actual-successor policy supervision: synthetic fixtures, CPU only."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from pebby.agent import world_model as wm, world_train as trainer
from tests.test_world_model import TINY, make_model
from tests.test_world_glyph import make_glyph_synthetic, wake


class SuccessorPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def data(self):
        data = make_glyph_synthetic(seed=41, levels=1, steps=4, history=4)
        masks = np.tile(np.array([1, 3, 0, 8], dtype=np.uint8), (len(data['frames']), 1))
        masks[data['terminal']] = 0
        data['next_optimal'] = masks
        return data

    def run_loss(self, model, batch, weight=1.):
        return wm.world_losses(model, batch, {'successor_policy': weight},
                               sigreg_generator=torch.Generator().manual_seed(71))

    def test_reuses_encoding_matches_mask_ce_and_backpropagates(self):
        model = wake(make_model()).train()
        batch = trainer.as_tensors(self.data())
        encodings, logits = [], []
        original = model.logits_from
        def capture(encoding, *args, **kwargs):
            result = original(encoding, *args, **kwargs)
            encodings.append(encoding)
            logits.append(result[0])
            return result
        with patch.object(model, 'logits_from', side_effect=capture), \
                patch.object(model, 'frame_tokens', wraps=model.frame_tokens) as tokens:
            out = self.run_loss(model, batch)
        self.assertEqual(tokens.call_count, 2)
        self.assertEqual(len(encodings), 2)
        torch.testing.assert_close(encodings[1]['latent'], out['targets'].flatten(0, 1))
        masks = batch['next_optimal'].flatten()
        valid = masks != 0
        bits = wm.optimal_bits(masks[valid])
        target = bits / bits.sum(-1, keepdim=True)
        expected = -(target * F.log_softmax(logits[1][valid], dim=-1)).sum(-1).mean()
        torch.testing.assert_close(out['losses']['successor_policy'], expected)
        accuracy = bits.gather(1, logits[1][valid].argmax(-1)[:, None]).mean()
        torch.testing.assert_close(out['diagnostics']['successor_policy_set_accuracy'], accuracy)
        out['losses']['successor_policy'].backward()
        for parameter in (model.stem[0].weight, model.ranker[0].weight,
                          model.projector[0].weight, model.move_head[0].weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0.)

    def test_fail_closed_and_all_invalid_differentiable_zero(self):
        model = make_model()
        batch = trainer.as_tensors(self.data())
        missing = {k: v for k, v in batch.items() if k != 'next_optimal'}
        cases = [missing, {**batch, 'next_optimal': batch['next_optimal'][:, :3]},
                 {**batch, 'next_optimal': batch['next_optimal'].float()},
                 {**batch, 'next_optimal': batch['next_optimal'].bool()},
                 {**batch, 'next_optimal': torch.full_like(batch['next_optimal'], 16)},
                 {**batch, 'next_optimal': torch.full_like(batch['next_optimal'].long(), -1)},
                 {**batch, 'terminal': torch.ones_like(batch['terminal']),
                  'next_optimal': torch.ones_like(batch['next_optimal'])}]
        for bad in cases:
            with self.subTest(keys=bad.keys()), self.assertRaisesRegex(ValueError, 'next_optimal'):
                self.run_loss(model, bad)
        out = self.run_loss(model, {**batch, 'next_optimal': torch.zeros_like(batch['next_optimal'])})
        loss = out['losses']['successor_policy']
        self.assertTrue(loss.requires_grad)
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(out['diagnostics']['successor_policy_set_accuracy'].item(), 0.)
        self.assertEqual(out['diagnostics']['successor_policy_valid_fraction'].item(), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(model.stem[0].weight.grad).all())
        self.assertEqual(model.stem[0].weight.grad.abs().sum().item(), 0.)

    def test_default_parity_and_current_teacher_future_isolation(self):
        model = make_model(glyph_recall=True, state_recall=True).eval()
        batch = trainer.as_tensors(self.data())
        with torch.no_grad():
            plain = self.run_loss(model, {k: v for k, v in batch.items() if k != 'next_optimal'}, 0.)
            disabled = self.run_loss(model, batch, 0.)
            enabled = self.run_loss(model, batch)
            self.assertNotIn('successor_policy', disabled['losses'])
            torch.testing.assert_close(plain['total'], disabled['total'], rtol=0, atol=0)
            for key in ('logits', 'latent', 'targets', 'predicted'):
                torch.testing.assert_close(enabled[key], disabled[key], rtol=0, atol=0)
            for key, loss in disabled['losses'].items():
                torch.testing.assert_close(enabled['losses'][key], loss, rtol=0, atol=0)
            altered = {**batch, 'next_frames': torch.full_like(batch['next_frames'], 6),
                       'next_optimal': torch.zeros_like(batch['next_optimal']),
                       'next_triple': (batch['next_triple'] + 1) % 4,
                       'current_triple': (batch['current_triple'] + 1) % 4}
            changed = self.run_loss(model, altered)
            torch.testing.assert_close(changed['logits'], enabled['logits'], rtol=0, atol=0)
            self.assertFalse(torch.allclose(changed['targets'], enabled['targets']))

    def test_chunk_checkpoint_and_life_reset_parity_with_glyph(self):
        model = wake(make_model(glyph_recall=True, state_recall=True)).train()
        other = copy.deepcopy(model)
        other.checkpoint_encoder = other.checkpoint_loops = True
        other.encoder_chunk_size = 1
        batch = trainer.as_tensors(self.data())
        batch['lost_life'] = torch.zeros_like(batch['terminal'])
        batch['lost_life'][0, 0] = True
        outputs = []
        for policy in (model, other):
            out = self.run_loss(policy, batch)
            out['total'].backward()
            outputs.append(out)
        for key in ('logits', 'targets', 'latent'):
            torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=1e-6, rtol=1e-5)
        for (name, first), (_, second) in zip(model.named_parameters(), other.named_parameters()):
            if first.grad is not None:
                torch.testing.assert_close(first.grad, second.grad, atol=1e-6, rtol=1e-5, msg=name)
        reset_frame = batch['next_frames'][:1, 0:1].expand(-1, 4, -1, -1)
        expected = model.encode(reset_frame, torch.tensor([[False, False, False, True]]),
                                torch.full((1, 4), -1))['latent']
        torch.testing.assert_close(outputs[0]['targets'][0, 0], expected[0], atol=1e-6, rtol=1e-5)

    def test_grounding_glyph_successor_metrics_survive_and_aggregate(self):
        model = wake(make_model(grounding=True, glyph_recall=True, state_recall=True)).eval()
        data = self.data()
        count = len(data['frames'])
        data.update(next_player_cell=np.repeat(data['player_cell'][:, None], 4, axis=1),
                    current_steps=np.full(count, 20), next_steps=np.full((count, 4), 19),
                    current_lives=np.full(count, 2), next_lives=np.full((count, 4), 2))
        data['next_optimal'][:2] = 0
        batch = trainer.as_tensors(data)
        with torch.no_grad():
            out = self.run_loss(model, batch)
            for key in ('successor_policy_set_accuracy', 'successor_policy_valid_fraction',
                        'latent_player_accuracy', 'actual_steps_accuracy', 'imagined_shape_accuracy',
                        'glyph_current_shape_accuracy', 'glyph_actual_rotation_accuracy'):
                self.assertIn(key, out['diagnostics'])
            records = []
            for part in trainer.batches(batch, 2):
                result = self.run_loss(model, part)
                records.append((int((part['next_optimal'] != 0).sum()), result))
            stats = trainer.run_epoch(model, batch, torch.device('cpu'), {'successor_policy': 1.}, 2)
        valid = sum(n for n, _ in records)
        expected_ce = sum(n * result['losses']['successor_policy'].item() for n, result in records) / valid
        expected_accuracy = sum(n * result['diagnostics']['successor_policy_set_accuracy'].item()
                                for n, result in records) / valid
        self.assertAlmostEqual(stats['successor_policy'], expected_ce)
        self.assertAlmostEqual(stats['successor_policy_set_accuracy'], expected_accuracy)
        self.assertAlmostEqual(stats['successor_policy_valid_fraction'], valid / (count * 4))
        self.assertIn('latent_player_accuracy', stats)
        self.assertIn('glyph_current_shape_accuracy', stats)

    def test_epoch_successor_metrics_use_valid_counts(self):
        tensors = {'frames': torch.zeros(2, 1),
                   'next_optimal': torch.tensor([[1, 0, 0, 0], [1, 2, 4, 8]])}
        def fake_loss(model, batch, *args, **kwargs):
            sparse = int((batch['next_optimal'] != 0).sum()) == 1
            loss = torch.tensor(2. if sparse else 4.)
            return {'total': loss, 'losses': {'successor_policy': loss},
                    'diagnostics': {'successor_policy_set_accuracy': torch.tensor(1. if sparse else 0.),
                                    'successor_policy_valid_fraction': torch.tensor(.25 if sparse else 1.)}}
        with patch.object(trainer, 'world_losses', side_effect=fake_loss):
            stats = trainer.run_epoch(make_model(), tensors, torch.device('cpu'),
                                      {'successor_policy': 1.}, 1)
        self.assertAlmostEqual(stats['successor_policy'], 3.6)
        self.assertAlmostEqual(stats['successor_policy_set_accuracy'], .2)
        self.assertAlmostEqual(stats['successor_policy_valid_fraction'], .625)

    def test_sidecar_valid_attachment_and_fail_closed_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = self.data()
            # Distinct row seeds make order mismatches observable.
            data['seeds'] = np.arange(len(data['frames']), dtype=np.int32)
            masks = data.pop('next_optimal')
            source = root / 'source.npz'
            np.savez(source, **data)
            loaded = trainer.load_dataset(source)
            meta = {'format': 'pebby.ls20-successor-labels.v1', 'source': 'generated_only',
                    'source_sha256': trainer.file_digest(source), 'rows': len(masks),
                    'levels': len(np.unique(data['seeds']))}
            sidecar = root / 'labels.npz'
            def write(metadata=meta, seeds=None, labels=masks):
                np.savez(sidecar, next_optimal=labels,
                         seeds=data['seeds'] if seeds is None else seeds, meta=np.array(json.dumps(metadata)))
            write()
            attached = trainer.attach_successor_labels(loaded, source, sidecar)
            np.testing.assert_array_equal(attached['next_optimal'], masks)
            self.assertIs(attached['frames'], loaded['frames'])
            self.assertIsNone(loaded['next_optimal'])
            provenance = attached['meta']['successor_labels']
            self.assertEqual(provenance['sha256'], trainer.file_digest(sidecar))
            self.assertEqual(provenance['source_sha256'], meta['source_sha256'])
            with self.assertRaisesRegex(ValueError, 'replace existing'):
                trainer.attach_successor_labels(attached, source, sidecar)
            for changed in ({'format': 'wrong'}, {'source': 'official'}, {'source_sha256': 'wrong'},
                            {'rows': len(masks) + 1}, {'levels': 0}, {'rows': True}):
                write({**meta, **changed})
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    trainer.attach_successor_labels(loaded, source, sidecar)
            write(seeds=data['seeds'][::-1])
            with self.assertRaisesRegex(ValueError, 'exact order'):
                trainer.attach_successor_labels(loaded, source, sidecar)
            for labels in (masks.astype(float), masks[:, :3], np.full_like(masks, 16)):
                write(labels=labels)
                with self.assertRaisesRegex(ValueError, 'next_optimal'):
                    trainer.attach_successor_labels(loaded, source, sidecar)
            write(labels=np.ones_like(masks))
            with self.assertRaisesRegex(ValueError, 'terminal'):
                trainer.attach_successor_labels({**loaded, 'terminal': np.ones_like(data['terminal'])}, source, sidecar)

    def test_trainer_one_step_and_both_split_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = self.data()
            validation = {**data, 'seeds': data['seeds'] + 100}
            for name, payload in [('train', data), ('validation', validation)]:
                np.savez(root / f'{name}.npz', **payload)
                np.savez(root / f'{name}-bare.npz', **{k: v for k, v in payload.items() if k != 'next_optimal'})
            flags = ['--train', str(root / 'train.npz'), '--validation', str(root / 'validation.npz'),
                     '--device', 'cpu', '--epochs', '1', '--batch-size', '4',
                     '--select-on', 'last',
                     '--successor-policy-weight', '0.5', '--glyph-recall', '--state-recall',
                     '--checkpoint-out', str(root / 'model.pt')]
            for key, value in TINY.items():
                flags += [f"--{key.replace('_', '-')}", str(value)]
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(trainer.main(flags), 0)
            report = json.loads((root / 'model.training.json').read_text())
            for split in ('train', 'validation'):
                for key in ('successor_policy', 'successor_policy_set_accuracy', 'successor_policy_valid_fraction'):
                    self.assertTrue(np.isfinite(report['history'][0][split][key]))
            # Sidecars bind full files before row subsampling and survive in checkpoint/report provenance.
            sidecar_flags = []
            for split in ('train', 'validation'):
                source = root / f'{split}-bare.npz'
                source_data = trainer.load_dataset(source)
                labels = root / f'{split}-labels.npz'
                meta = {'format': 'pebby.ls20-successor-labels.v1', 'source': 'generated_only',
                        'source_sha256': trainer.file_digest(source), 'rows': len(data['frames']),
                        'levels': len(np.unique(source_data['seeds']))}
                np.savez(labels, next_optimal=data['next_optimal'], seeds=source_data['seeds'],
                         meta=np.array(json.dumps(meta)))
                sidecar_flags += [f'--{split}', str(source), f'--{split}-successor-labels', str(labels)]
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(trainer.main(flags + sidecar_flags + ['--max-states', '2',
                                 '--max-validation-states', '2', '--checkpoint-out', str(root / 'attached.pt')]), 0)
            _, checkpoint = wm.load_world_checkpoint(root / 'attached.pt')
            self.assertEqual(checkpoint['samples'], 2)
            self.assertEqual(set(checkpoint['successor_labels']), {'train', 'validation'})
            attached_report = json.loads((root / 'attached.training.json').read_text())
            self.assertEqual(attached_report['successor_labels'], checkpoint['successor_labels'])
            self.assertEqual(checkpoint['data_meta']['successor_labels'], checkpoint['successor_labels']['train'])
            loaded = trainer.load_dataset(root / 'train.npz')
            self.assertEqual(trainer.as_tensors(loaded)['next_optimal'].dtype, torch.uint8)
            for split in ('train', 'validation'):
                output = root / f'bad-{split}.pt'
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
                    with self.assertRaises(SystemExit):
                        trainer.main(flags + [f'--{split}', str(root / f'{split}-bare.npz'),
                                              '--checkpoint-out', str(output)])
                self.assertIn('next_optimal', errors.getvalue())
                self.assertFalse(output.exists())
            for masks in (data['next_optimal'].astype(float), np.full_like(data['next_optimal'], 16),
                          data['next_optimal'][:, :3]):
                np.savez(root / 'malformed.npz', **{**data, 'next_optimal': masks})
                with self.assertRaisesRegex(ValueError, 'next_optimal'):
                    trainer.load_dataset(root / 'malformed.npz')


if __name__ == '__main__':
    unittest.main()
