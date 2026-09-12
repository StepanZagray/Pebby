import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pebby.agent import world_data
from pebby.ls20 import generate
from tools.annotate_world_successors import annotate, row_tag, source_fingerprints


class AnnotateSuccessorsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.spec = generate.generate_legacy_level(0, 1)
        self.bank = self.root / 'bank.jsonl'
        self.bank.write_text(json.dumps(self.spec) + '\n')
        self.data = world_data.build([self.spec], history=4, samples=4, coverage='mixed', workers=1)
        self.expected = self.data.pop('next_optimal')
        self.source = self.root / 'source.npz'
        world_data.save(self.source, self.data)

    def test_streamed_hashes_equal_direct_rows_and_replay_emits_only_labels(self):
        seeds, hashes, schema, _ = source_fingerprints(self.source, chunk_rows=1)
        for index in range(len(seeds)):
            direct = hashlib.sha256()
            for name, dtype, shape in schema:
                direct.update(row_tag(name, dtype, shape))
                direct.update(self.data[name][index].tobytes())
            self.assertEqual(bytes(hashes[index]), direct.digest())
        output = self.root / 'labels.npz'
        result = annotate(self.source, self.bank, output, workers=1, samples=4)
        self.assertEqual(result['exact_replay_rows'], len(seeds))
        with np.load(output) as archive:
            self.assertEqual(set(archive.files), {'seeds', 'next_optimal', 'meta'})
            np.testing.assert_array_equal(archive['next_optimal'], self.expected)
            np.testing.assert_array_equal(archive['seeds'], seeds)
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            annotate(self.source, self.bank, output, workers=1, samples=4)

    def test_changed_pixel_fails_replay_and_publishes_nothing(self):
        self.data['frames'][0, -1, 20, 20] ^= 1
        world_data.save(self.source, self.data)
        output = self.root / 'labels.npz'
        with self.assertRaisesRegex(ValueError, 'differs from original'):
            annotate(self.source, self.bank, output, workers=1, samples=4)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix('.json').exists())

    def test_pilot_never_publishes_partial_training_labels(self):
        output = self.root / 'pilot.npz'
        result = annotate(self.source, self.bank, output, workers=1, samples=4, limit_levels=1)
        self.assertEqual(result['status'], 'pilot')
        self.assertFalse(output.exists())
        self.assertTrue(output.with_suffix('.json').exists())


if __name__ == '__main__':
    unittest.main()
