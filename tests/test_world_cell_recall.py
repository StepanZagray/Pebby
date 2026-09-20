"""Public-pixel appearance recall: migration, gradients, provenance and IO."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
import numpy as np

from pebby.agent.cell_appearance import CellAppearance, FORMAT
from pebby.agent.world_cell_recall import initialize_cell_encoder, load_cell_source, verify_cell_source
from pebby.agent.world_model import (WorldModelConfig, WorldPolicy, initialize_from_checkpoint,
                                    load_world_checkpoint, parameter_groups, save_world_checkpoint)
from tests.test_world_model import make_model, make_synthetic, TINY


class CellRecallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_default_off_and_boolean_validation(self):
        self.assertFalse(WorldModelConfig().cell_recall)
        self.assertFalse(any(key.startswith('cell_') for key in make_model().state_dict()))
        with self.assertRaisesRegex(ValueError, 'cell_recall must be boolean'):
            WorldModelConfig(cell_recall=1)

    def test_migration_preserves_logits_with_or_without_existing_readouts(self):
        for extra in ({}, dict(glyph_recall=True, query_readout=True, grounding=True)):
            with self.subTest(extra=extra):
                torch.manual_seed(72)
                original = make_model(**extra).eval()
                target = make_model(**extra, cell_recall=True).eval()
                initialize_cell_encoder(target, CellAppearance())
                target.cell_context.weight.data.normal_()
                self.assertEqual(initialize_from_checkpoint(target, original), [])
                frames = torch.randint(16, (2, 4, 64, 64), dtype=torch.uint8)
                valid = torch.tensor([[False, False, True, True], [True, True, True, True]])
                actions = torch.tensor([[-1, -1, -1, 2], [-1, 0, 1, 3]])
                with torch.no_grad():
                    torch.testing.assert_close(original(frames, valid, actions),
                                               target(frames, valid, actions), atol=0, rtol=0)
                with self.assertRaisesRegex(ValueError, 'without cell_recall'):
                    initialize_from_checkpoint(original, target)

    def test_frozen_decoder_and_trainable_context_reach_policy(self):
        torch.manual_seed(51)
        model = make_model(cell_recall=True).train()
        frames = torch.randint(16, (2, 4, 64, 64), dtype=torch.uint8)
        model(frames).square().sum().backward()
        self.assertGreater(float(model.cell_context.weight.grad.abs().sum()), 0)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.cell_appearance.parameters()))
        optim_ids = {id(p) for group in parameter_groups(model, .01) for p in group['params']}
        self.assertTrue(all(id(p) not in optim_ids for p in model.cell_appearance.parameters()))
        self.assertIn(id(model.cell_context.weight), optim_ids)
        self.assertGreater(float(model.stem[0].weight.grad.abs().sum()), 0)

    def test_chunk_checkpoint_and_roundtrip_with_active_recall(self):
        torch.manual_seed(22)
        model = make_model(cell_recall=True, glyph_recall=True, query_readout=True).eval()
        model.cell_context.weight.data.normal_(std=.2)
        frames = torch.randint(16, (2, 4, 64, 64), dtype=torch.uint8)
        reference = model(frames)
        model.encoder_chunk_size = 2
        model.checkpoint_encoder = model.checkpoint_loops = True
        actual = model(frames)
        torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)
        actual.sum().backward()
        self.assertGreater(float(model.cell_context.weight.grad.abs().sum()), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            save_world_checkpoint(path, model)
            restored, _ = load_world_checkpoint(path)
            torch.testing.assert_close(restored(frames), reference, atol=0, rtol=0)
            self.assertTrue(all(not p.requires_grad for p in restored.cell_appearance.parameters()))
            self.assertEqual(restored.parameter_count() - make_model(glyph_recall=True, query_readout=True).parameter_count(),
                             110166 + 22 * model.cfg.channels)

    def test_provenance_rejects_validation_training_or_invalid_seeds(self):
        source = dict(format=FORMAT, train_seeds=[1, 2], validation_used_for_training_or_selection=False,
                      frozen=True, parameters=110166, path='decoder.pt', bank='bank.npz', proof='proof.json',
                      **{key: 'a' * 64 for key in ('sha256', 'bank_sha256', 'proof_sha256', 'embedded_weights_sha256')})
        self.assertIs(verify_cell_source(source, [1_000_001]), source)
        for altered, validation in ((dict(source, train_seeds=[1, 1]), []),
                                    (dict(source, train_seeds=[1_000_001]), []),
                                    (dict(source, validation_used_for_training_or_selection=True), []),
                                    (source, [2]), (dict(source, bank_sha256=None), []),
                                    (dict(source, frozen=False), [])):
            with self.assertRaises(ValueError):
                verify_cell_source(altered, validation)

    def test_actual_generated_decoder_provenance_and_import(self):
        path = Path('checkpoints/cell-appearance-2k-400.pt')
        if not path.exists():
            self.skipTest('local generated decoder checkpoint not present')
        encoder, source = load_cell_source(path, 'data/ls20-visible-cell-labels-2k.npz',
                                          'artifacts/world-visible-cell-labels-2k-proof.json', [1_000_001])
        self.assertEqual(len(source['train_seeds']), 2000)
        model = make_model(cell_recall=True)
        initialize_cell_encoder(model, encoder)
        verify_cell_source(source, [], model.cell_appearance)
        with torch.no_grad():
            model.cell_appearance.network[0].weight[0, 0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'embedded cell decoder weights differ'):
            verify_cell_source(source, [], model.cell_appearance)
        initialize_cell_encoder(model, encoder)
        frames = torch.randint(16, (2, 64, 64), dtype=torch.uint8)
        with torch.no_grad():
            torch.testing.assert_close(torch.cat(model.cell_appearance(frames), -1),
                                       torch.cat(encoder(frames), -1), atol=3e-4, rtol=1e-4)
        with tempfile.TemporaryDirectory() as directory:
            changed = torch.load(path, weights_only=True)
            changed['source_hashes'] = dict(changed['source_hashes'], **{'data/ls20-visible-cell-labels-2k.npz': '0' * 64})
            altered = Path(directory) / 'bad.pt'
            torch.save(changed, altered)
            with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
                load_cell_source(altered, 'data/ls20-visible-cell-labels-2k.npz',
                                 'artifacts/world-visible-cell-labels-2k-proof.json', [])

    def test_trainer_import_resume_and_missing_decoder_guard(self):
        from pebby.agent import world_train as trainer
        if not Path('checkpoints/cell-appearance-2k-400.pt').exists():
            self.skipTest('local generated decoder checkpoint not present')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = make_synthetic(seed=51, levels=2, steps=4)
            validation = make_synthetic(seed=52, levels=1, steps=4)
            validation['seeds'] += 1_000_000
            for split, arrays in (('train', train), ('validation', validation)):
                np.savez(root / f'{split}.npz', meta=np.array(json.dumps({'source': 'synthetic'})), **arrays)
            flags = ['--train', str(root / 'train.npz'), '--validation', str(root / 'validation.npz'),
                     '--epochs', '1', '--batch-size', '8', '--device', 'cpu', '--cell-recall',
                     '--select-on', 'last']
            for key, value in TINY.items():
                flags += [f"--{key.replace('_', '-')}", str(value)]
            imported = root / 'imported.pt'
            resumed = root / 'resumed.pt'
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    trainer.main(flags + ['--checkpoint-out', str(root / 'missing.pt')])
                self.assertEqual(trainer.main(flags + ['--checkpoint-out', str(imported),
                    '--initialize-cell-checkpoint', 'checkpoints/cell-appearance-2k-400.pt',
                    '--cell-appearance-data', 'data/ls20-visible-cell-labels-2k.npz',
                    '--cell-appearance-proof', 'artifacts/world-visible-cell-labels-2k-proof.json']), 0)
                self.assertEqual(trainer.main(flags + ['--checkpoint-out', str(resumed),
                                                       '--initialize-checkpoint', str(imported)]), 0)
            first, first_metadata = load_world_checkpoint(imported)
            second, second_metadata = load_world_checkpoint(resumed)
            self.assertEqual(first_metadata['cell_source'], second_metadata['cell_source'])
            _, running = load_world_checkpoint(root / 'resumed.running.pt')
            self.assertEqual(running['cell_source'], first_metadata['cell_source'])
            for name, value in first.cell_appearance.state_dict().items():
                torch.testing.assert_close(value, second.cell_appearance.state_dict()[name], atol=0, rtol=0)
            self.assertGreater(float(second.cell_context.weight.detach().abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
