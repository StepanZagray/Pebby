"""Independent regression tests for byte-bound decoder import provenance."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pebby.agent.cell_appearance import CellAppearance, FORMAT
from pebby.agent.cell_appearance_dense import DenseCellAppearance
from pebby.agent.world_cell_recall import load_cell_source, verify_cell_source


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CellProvenanceReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, directory):
        root = Path(directory)
        bank, proof, checkpoint = (root / name for name in ('bank.npz', 'proof.json', 'cell.pt'))
        np.savez(bank, seeds=np.array([1, 1000001]), split=np.array(['train', 'validation']))
        proof.write_text('{}')
        encoder = CellAppearance()
        payload = dict(format=FORMAT, weights=encoder.state_dict(), parameters=110166,
                       validation_used_for_training_or_selection=False, initial_state_only=True,
                       source_hashes={str(p): digest(p) for p in
                                      (bank, proof, Path('pebby/agent/cell_appearance.py'))})
        torch.save(payload, checkpoint)
        return checkpoint, bank, proof, encoder

    def test_checkpoint_replaced_during_load_remains_bound_to_consumed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, bank, proof, expected = self.fixture(directory)
            consumed_hash = digest(checkpoint)
            real_load = torch.load

            def replace_then_load(buffer, **kwargs):
                self.assertFalse(isinstance(buffer, (str, Path)))
                checkpoint.write_bytes(b'replacement that is not the loaded checkpoint')
                return real_load(buffer, **kwargs)

            # Pixel-label validation is deliberately isolated here; its production
            # contract has separate tests. This test attacks checkpoint byte identity.
            with patch('pebby.agent.world_cell_recall.torch.load', side_effect=replace_then_load), \
                    patch('tools.train_cell_appearance.load_examples'):
                encoder, source = load_cell_source(checkpoint, bank, proof, [1000001])
            self.assertEqual(source['sha256'], consumed_hash)
            self.assertNotEqual(source['sha256'], digest(checkpoint))
            for name, value in expected.state_dict().items():
                torch.testing.assert_close(encoder.state_dict()[name], value, atol=0, rtol=0)
            verify_cell_source(source, [1000001], DenseCellAppearance(encoder))

    def test_inherited_weights_and_complete_schema_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, bank, proof, _ = self.fixture(directory)
            with patch('tools.train_cell_appearance.load_examples'):
                encoder, source = load_cell_source(checkpoint, bank, proof, [])
            embedded = DenseCellAppearance(encoder)
            verify_cell_source(source, [], embedded)
            for field in ('sha256', 'bank_sha256', 'proof_sha256', 'embedded_weights_sha256', 'path', 'bank', 'proof', 'frozen', 'parameters'):
                altered = dict(source); altered.pop(field)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    verify_cell_source(altered, [], embedded)
            with torch.no_grad():
                embedded.network[0].weight[0, 0, 0, 0].add_(1)
            with self.assertRaisesRegex(ValueError, 'weights differ'):
                verify_cell_source(source, [], embedded)

    def test_source_proof_replaced_during_label_load_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, bank, proof, _ = self.fixture(directory)
            with patch('tools.train_cell_appearance.load_examples', side_effect=lambda *_: proof.write_text('{"changed":true}')):
                with self.assertRaisesRegex(ValueError, 'inputs changed'):
                    load_cell_source(checkpoint, bank, proof, [])


if __name__ == '__main__':
    unittest.main()
