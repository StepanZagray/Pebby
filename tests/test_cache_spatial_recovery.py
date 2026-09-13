"""CPU-only contract tests; never launch extraction or initialize CUDA."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools import cache_spatial_recovery as cache


def source(n=4):
    arrays = {key: np.zeros((n, *tail), dtype=dtype) for key, (dtype, tail) in cache.INPUT_SCHEMA.items()}
    arrays['seeds'] = np.full(n, 7, dtype=np.int32)
    arrays['history_valid'][:] = True
    arrays['row_kind'][:] = [0, 1, 2, 1][:n]
    arrays['trajectory_id'][:] = np.arange(n) + 11
    arrays['step'][:] = np.arange(n) * 2
    arrays['current_steps'][:] = np.arange(n) + 20
    arrays['frames'][:, -1, 0, 0] = np.arange(n)
    return arrays


class PublicEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, *args):
        # All teacher labels are deliberately absent from this call boundary.
        assert len(args) == 3
        assert args[0].shape[1:] == (8, 64, 64)
        assert args[1].shape[1:] == args[2].shape[1:] == (8,)
        self.calls.append(args)
        return {key: args[0][:, -1, 0, 0].float().reshape(-1, *([1] * len(tail))).expand(-1, *tail).clone()
                for key, (_, tail) in cache.SCHEMA.items() if key in ('raw', 'state', 'glyph')}


class RecoveryCacheTests(unittest.TestCase):
    def test_source_kind_indices_and_targets_preserve_order(self):
        arrays = source()
        self.assertEqual(cache.validate_rows(arrays, 7), 4)
        encoder = PublicEncoder()
        chosen = np.array([3, 0, 2])
        result = cache.payload(encoder, arrays, chosen, 25, 'cpu')
        np.testing.assert_array_equal(result['rows'], [28, 25, 27])
        for key in ('row_kind', 'trajectory_id', 'step', 'current_steps'):
            np.testing.assert_array_equal(result[key], arrays[key][chosen])
        np.testing.assert_array_equal(result['raw'][:, 0, 0], chosen)
        self.assertEqual(set(result), set(cache.SCHEMA))
        self.assertFalse(set(cache.base.PUBLIC) & set(result))

    def test_privileged_fields_do_not_enter_encoder(self):
        arrays = source()
        encoder = PublicEncoder()
        first = cache.payload(encoder, arrays, np.arange(4), 0, 'cpu')
        arrays['next_frames'] = np.full((4, 4, 64, 64), 255, dtype=np.uint8)
        arrays['next_optimal'][:] = 15
        arrays['current_triple'][:] = 9
        second = cache.payload(encoder, arrays, np.arange(4), 0, 'cpu')
        for key in ('raw', 'state', 'glyph'):
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(len(encoder.calls), 2)

    def test_rejects_malformed_source(self):
        for key, bad in [('row_kind', np.full(4, 3, dtype=np.uint8)),
                         ('trajectory_id', np.full(4, -1, dtype=np.int64)),
                         ('frames', np.zeros((4, 1, 64, 64), dtype=np.uint8)),
                         ('previous_actions', np.full((4, 8), 4, dtype=np.int64)),
                         ('current_steps', np.full(4, np.nan))]:
            with self.subTest(key=key):
                arrays = source()
                arrays[key] = bad
                with self.assertRaises(ValueError):
                    cache.validate_rows(arrays, 7)
        with self.assertRaises(ValueError):
            cache.validate_rows(source(), 8)

    def test_manifest_refuses_incomplete_before_reading_arrays(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            for status in ('running', 'failed', 'complete'):
                (path / 'manifest.json').write_text(json.dumps({'status': status}))
                with self.assertRaisesRegex(ValueError, 'incomplete or invalid'):
                    cache.validate_published(path)

    def test_source_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                cache.child(Path(folder), '../other.npz')

    def test_published_features_match_native_public_call(self):
        arrays, encoder = source(), PublicEncoder()
        chosen = np.arange(4)
        features = cache.payload(encoder, arrays, chosen, 0, 'cpu')
        self.assertEqual(cache.feature_parity(encoder, arrays, chosen, features, 0, 'cpu')['same_batch_max_abs'],
                         dict(raw=0., state=0., glyph=0.))
        features['glyph'][2, 0] += 1
        with self.assertRaises(AssertionError):
            cache.feature_parity(encoder, arrays, chosen, features, 0, 'cpu')

    def test_parity_reconstructs_original_batches_and_final_single_row_tail(self):
        class BatchSensitiveEncoder(PublicEncoder):
            def encode(self, *args):
                encoded = super().encode(*args)
                encoded['state'] += len(args[0]) * .001
                return encoded
        arrays = {key: np.repeat(value, 17, axis=0)[:65] for key, value in source().items()}
        encoder = BatchSensitiveEncoder()
        cached = {key: np.empty((65, *tail), dtype=dtype) for key, (dtype, tail) in cache.SCHEMA.items()}
        for begin, end in [(0, 64), (64, 65)]:
            values = cache.payload(encoder, arrays, np.arange(begin, end), 0, 'cpu')
            for key, value in values.items():
                cached[key][begin:end] = value
        encoder.calls.clear()
        result = cache.feature_parity(encoder, arrays, np.array([0, 21, 42, 64]), cached, 0, 'cpu', batch_size=64)
        self.assertEqual([len(call[0]) for call in encoder.calls[:2]], [64, 1])
        self.assertEqual([b['batch_rows'] for b in result['original_batches']], [64, 1])
        self.assertEqual(result['same_batch_max_abs']['state'], 0.)
        self.assertGreater(result['batch_size_sensitivity']['grouped_vs_original']['state']['outside_tolerance'], 0)
        self.assertGreater(result['batch_size_sensitivity']['single_vs_original']['state']['max_abs'], .05)
        # The original tail must still be validated even though B4 differs legitimately.
        cached['state'][64, 0, 0] += .01
        with self.assertRaises(AssertionError):
            cache.feature_parity(encoder, arrays, np.array([0, 21, 42, 64]), cached, 0, 'cpu', batch_size=64)

    def test_collection_proofs_and_original_split_are_enforced(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / 'checkpoint.pt'
            checkpoint.write_bytes(b'CPU fixture; never loaded')
            digest = cache.base.sha(checkpoint)
            directory, base_dir = root / 'collection', root / 'base'
            (directory / 'levels').mkdir(parents=True)
            manifest = dict(status='complete', parent_sha256=cache.base.PARENT_SHA,
                            sources_unchanged=True, validation_disjoint=True, arrays={})
            for split, values in [('train', np.arange(10000)), ('validation', np.arange(10000, 10500))]:
                (base_dir / split).mkdir(parents=True)
                path = base_dir / split / 'seeds.npy'
                np.save(path, values)
                manifest['arrays'][split] = {'seeds': dict(sha256=cache.base.sha(path),
                    shape=list(values.shape), dtype=values.dtype.str)}
            (base_dir / 'manifest.json').write_text(json.dumps(manifest))
            (directory / 'train.jsonl').write_text(json.dumps({'seed': 7}) + '\n')
            array_path = directory / 'levels/7.npz'
            np.savez(array_path, **source())
            array_sha = cache.base.sha(array_path)
            proof_path = directory / 'levels/7.json'
            proof = dict(status='complete', seed=7, source_checkpoint_sha256=digest,
                         array_sha256=array_sha, rows=4)
            proof_path.write_text(json.dumps(proof))
            report = dict(status='complete', source_checkpoint_sha256=digest,
                official_inputs_used=False, public_inputs=list(cache.base.PUBLIC), levels=[dict(seed=7,
                array_path='levels/7.npz', sha256=array_sha, proof_path='levels/7.json',
                proof_sha256=cache.base.sha(proof_path))])
            (directory / 'report.json').write_text(json.dumps(report))
            with patch.object(cache, 'SOURCE_SHA', digest):
                levels, count = cache.inspect_collection(directory, checkpoint, base_dir, cache.base.Bindings())
                self.assertEqual(count, 4)
                self.assertEqual(levels[0]['offset'], 0)
                (directory / 'train.jsonl').write_text(json.dumps({'seed': 10001}) + '\n')
                with self.assertRaisesRegex(ValueError, 'original TRAIN/VAL split'):
                    cache.inspect_collection(directory, checkpoint, base_dir, cache.base.Bindings())
                (directory / 'train.jsonl').write_text(json.dumps({'seed': 7}) + '\n')
                proof_path.write_text(json.dumps({**proof, 'status': 'running'}))
                with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                    cache.inspect_collection(directory, checkpoint, base_dir, cache.base.Bindings())

    def test_existing_output_rejected_before_gpu_or_source_reads(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(cache.base, 'require_no_foreign_cuda') as gpu:
            with self.assertRaises(FileExistsError):
                cache.main(['--source-dir', 'absent', '--checkpoint', 'absent', '--out-dir', folder])
            gpu.assert_not_called()


if __name__ == '__main__':
    unittest.main()
