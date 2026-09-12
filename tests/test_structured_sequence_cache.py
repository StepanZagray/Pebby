"""Chronological field targets checked against independently collected game histories."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pebby.agent import world_data as wd
from pebby.agent.world_sequences import build_sidecar, load_sidecar
from pebby.agent.world_train import as_tensors
from tests.test_policy_history import corridor
from tests.test_structured_field_cache import RecordingAssembler
from tools.build_structured_sequence_cache import (
    CURRENT, FUTURE, FORMAT, digest, encode_sequence, sequence_batch, write_cache,
)


class StructuredSequenceCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        specs = []
        for difficulty, seed in enumerate((1000001, 1000002), start=1):
            spec = corridor()
            spec.update(seed=seed, difficulty=difficulty)
            spec['goals'][0]['cell'] = (8, 3)
            specs.append(spec)
        self.data = wd.build(specs, workers=1, history=8, samples=16,
                             coverage='mixed', epsilon=0)
        self.source = self.root / 'validation.npz'
        self.path = self.root / 'index.npz'
        wd.save(self.source, self.data)
        build_sidecar(self.data, self.source, self.path, 'explore_validation')
        self.index = load_sidecar(self.path, self.source, self.data,
                                  mode='explore_validation')
        self.rows = self.index.anchor_row

    def test_four_real_histories_actions_and_exact_labels(self):
        tensors = as_tensors(self.data)
        assembler = RecordingAssembler()
        fields, targets, labels, future, _ = encode_sequence(
            assembler, self.data, tensors, self.index, self.rows)
        np.testing.assert_array_equal(labels['actions'], np.full((2, 4), 3))
        np.testing.assert_array_equal(labels['distances'], [[4, 3, 2, 1]] * 2)
        independent = RecordingAssembler()
        for horizon in range(4):
            actual = independent(*(tensors[key][future[:, horizon]] for key in
                ('frames', 'history_valid', 'previous_actions')))
            np.testing.assert_array_equal(targets[:, horizon], actual.numpy().astype(np.float16))
        for output, original in CURRENT.items():
            np.testing.assert_array_equal(labels[output], self.data[original][self.rows])
            self.assertEqual(labels[output].dtype, self.data[original].dtype)
        for output, original in FUTURE.items():
            np.testing.assert_array_equal(labels[output], self.data[original][future])
            self.assertEqual(labels[output].dtype, self.data[original].dtype)
        self.assertEqual(fields.shape, (2, 148, 96))
        self.assertTrue(all(not labels[key].any() for key in ('terminal', 'won', 'lost_life')))

    def test_teacher_values_do_not_enter_current_or_future_encoder_inputs(self):
        changed = copy.deepcopy(self.data)
        changed['current_steps'][self.index.future_rows.ravel()] += 7
        a, b = RecordingAssembler(), RecordingAssembler()
        first = encode_sequence(a, self.data, as_tensors(self.data), self.index, self.rows)
        second = encode_sequence(b, changed, as_tensors(changed), self.index, self.rows)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertFalse(np.array_equal(first[2]['next_steps'], second[2]['next_steps']))
        self.assertEqual(len(a.calls), 2)
        for first_call, second_call in zip(a.calls, b.calls):
            self.assertEqual(len(first_call), 3)
            for x, y in zip(first_call, second_call):
                torch.testing.assert_close(x, y, atol=0, rtol=0)

    def test_corrupt_connectivity_missing_labels_and_reset_fail_closed(self):
        changed = copy.deepcopy(self.data)
        changed['frames'][self.index.future_rows[0, 0], 0, 0, 0] ^= 1
        with self.assertRaisesRegex(ValueError, 'chronological frames'):
            sequence_batch(changed, as_tensors(changed), self.index, self.rows)
        changed = copy.deepcopy(self.data)
        changed['lost_life'][self.rows[0], 3] = True
        with self.assertRaisesRegex(ValueError, 'terminal/life reset'):
            sequence_batch(changed, as_tensors(changed), self.index, self.rows)
        changed = dict(self.data)
        del changed['next_player_cell']
        with self.assertRaisesRegex(ValueError, 'necessary source array missing'):
            sequence_batch(changed, as_tensors(changed), self.index, self.rows)

    def test_published_cache_has_chronology_source_binding_and_exact_arrays(self):
        out = self.root / 'heldout'
        manifest = write_cache(self.data, self.index, self.source, self.path,
            out, RecordingAssembler(), {'test': 'public-only fixture'}, levels=2)
        self.assertEqual(manifest['format'], FORMAT)
        self.assertEqual(manifest['split'], 'validation')
        self.assertTrue(manifest['history_verified_against_actual_source_rows'])
        self.assertEqual(manifest['event_coverage']['transitions'], 8)
        self.assertEqual(manifest['source_hashes'][str(self.source)], digest(self.source))
        for name, record in manifest['arrays'].items():
            self.assertEqual(record['sha256'], digest(out / f'{name}.npy'))
        future = np.load(out / 'future_rows.npy')
        np.testing.assert_array_equal(np.load(out / 'next_steps.npy'), self.data['current_steps'][future])
        np.testing.assert_array_equal(np.load(out / 'actions.npy'), np.full((2, 4), 3))
        self.assertEqual(len(np.unique(np.load(out / 'seeds.npy'))), 2)
        with self.assertRaises(FileExistsError):
            write_cache(self.data, self.index, self.source, self.path,
                        out, RecordingAssembler(), {}, levels=2)


if __name__ == '__main__':
    unittest.main()
