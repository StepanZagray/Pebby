"""CPU-only semantic-addon contracts: alignment, pixels, parity and publication."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from tools import cache_spatial_semantics as cache


class PixelPerceptor(nn.Module):
    def forward(self, frames):
        value = frames[:, 0, 0].float()[:, None, None]
        return tuple(value.expand(-1, 144, n) + torch.arange(n) for n in (8, 6, 4, 4))


def publication(path):
    report = dict(format=cache.FORMAT, status='complete', teacher_sha256=cache.TEACHER_SHA,
        channels=cache.CHANNELS, runtime=cache.RUNTIME, sources_unchanged=True, weights_unchanged=True,
        all_rows_retained=True, official_inputs_used=False, validation_disjoint=True,
        counts=dict(train=2, validation=1, recent=1), files={}, aligned_source_files={}, parity={})
    semantic = np.full((2, 144, 22), .5, np.float32)
    for start, stop in ((8, 14), (14, 18), (18, 22)):
        semantic[..., start:stop] = 1 / (stop - start)
    for split, count in report['counts'].items():
        folder = path / split; folder.mkdir()
        arrays = dict(semantic=semantic[:count], rows=np.arange(count, dtype=np.int64),
                      seeds=np.array([1, 2] if split == 'train' else [3] if split == 'validation' else [1], dtype=np.int64))
        report['files'][split] = {}; report['aligned_source_files'][split] = {}
        for key, array in arrays.items():
            target = folder / f'{key}.npy'; np.save(target, array)
            report['files'][split][key] = dict(path=f'{split}/{key}.npy', sha256=cache.base.sha(target), shape=list(array.shape), dtype=array.dtype.str)
            if key != 'semantic':
                report['aligned_source_files'][split][key] = dict(path=f'/original/{split}/{key}.npy', sha256=cache.base.sha(target))
        report['parity'][split] = dict(samples=1, maximum_absolute_error=0.)
    teacher = path / 'fixture-teacher.pt'; teacher.write_bytes(b'fixture')
    report['teacher_path'] = str(teacher)
    report['runtime_source_bindings'] = {str(p.resolve()): cache.base.sha(p) for p in cache.RUNTIME_SOURCES}
    report['runtime_source_bindings'][str(teacher)] = cache.base.sha(teacher)
    (path / 'manifest.json').write_text(json.dumps(report))
    return report


class SemanticCacheTests(unittest.TestCase):
    def setUp(self):
        self.teacher_pin = patch.object(cache, "TEACHER_SHA", hashlib.sha256(b"fixture").hexdigest())
        self.teacher_pin.start()
        self.addCleanup(self.teacher_pin.stop)

    def test_current_selection_uses_last_frame_and_keeps_requested_order(self):
        frames = np.zeros((4, 8, 64, 64), dtype=np.uint8)
        frames[:, 0] = 15
        frames[:, -1] = np.arange(4)[:, None, None]
        result = cache.current_frames(frames, np.array([3, 0, 2], dtype=np.int64))
        np.testing.assert_array_equal(result[:, 0, 0], [3, 0, 2])
        self.assertEqual(result.shape, (3, 64, 64))
        self.assertFalse(np.shares_memory(result, frames))

    def test_bad_frame_contracts_and_palette_fail(self):
        frames = np.zeros((2, 8, 64, 64), dtype=np.uint8)
        for source, rows in [(frames[:, -1], np.array([0])), (frames.astype(np.float32), np.array([0])),
                             (frames, np.array([-1])), (frames, np.array([2])), (frames, np.array([0.]))]:
            with self.subTest(shape=source.shape, rows=rows), self.assertRaises(ValueError):
                cache.current_frames(source, rows)
        frames[0, -1, 0, 0] = 16
        with self.assertRaises(ValueError):
            cache.current_frames(frames, np.array([0], dtype=np.int64))

    def test_alignment_rejects_permuted_seeds_duplicates_and_missing_rows(self):
        source = np.array([11, 22, 33], dtype=np.int64)
        cache.check_alignment(np.array([2, 0]), np.array([33, 11]), source)
        for rows, seeds in [(np.array([2, 0]), np.array([11, 33])),
                            (np.array([0, 0]), np.array([11, 11])), (np.array([3]), np.array([33]))]:
            with self.assertRaises(ValueError):
                cache.check_alignment(rows, seeds, source)

    def test_probability_api_exact_role_sigmoid_and_attribute_softmax(self):
        frames = np.zeros((2, 64, 64), dtype=np.uint8)
        frames[1, 0, 0] = 3
        actual = cache.encode_frames(PixelPerceptor(), frames, 'cpu')
        expected = np.concatenate([torch.arange(8).float().sigmoid().numpy(),
            *[torch.arange(n).float().softmax(-1).numpy() for n in (6, 4, 4)]])
        np.testing.assert_array_equal(actual[0, 0], expected)
        # Role probabilities change with absolute logits; each separate attribute
        # softmax is invariant to the shared pixel-dependent constant.
        self.assertGreater(actual[1, 0, 0], actual[0, 0, 0])
        np.testing.assert_allclose(actual[1, :, 8:], actual[0, :, 8:], atol=1e-7, rtol=0)
        np.testing.assert_array_equal(actual[:1], cache.encode_frames(PixelPerceptor(), frames[:1], 'cpu'))
        self.assertFalse(torch.cuda.is_initialized())

    def test_probability_validation_rejects_nan_range_shape_dtype_and_bad_simplex(self):
        good = cache.encode_frames(PixelPerceptor(), np.zeros((1, 64, 64), np.uint8), 'cpu')
        bad = [good.astype(np.float16), good[:, :143]]
        for value in [float('nan'), -1., 2.]:
            changed = good.copy(); changed[0, 0, 0] = value; bad.append(changed)
        changed = good.copy(); changed[..., 8:14] *= .5; bad.append(changed)
        for array in bad:
            with self.assertRaises(ValueError):
                cache.validate_probabilities(array, 1)

    def test_recent_archive_reads_frames_once_and_aligns_cached_labels(self):
        arrays = dict(frames=np.zeros((3, 8, 64, 64), np.uint8), seeds=np.full(3, 7, np.int64),
                      optimal=np.array([1, 0, 0], np.uint8), row_kind=np.array([0, 1, 2], np.int8))
        arrays['frames'][:, -1] = np.arange(3)[:, None, None]
        calls = {}
        class Archive:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def __getitem__(self, key):
                calls[key] = calls.get(key, 0) + 1
                return arrays[key].copy()
        cached = {k: v for k, v in arrays.items() if k != 'frames'}
        cached['rows'] = np.arange(3)
        with patch.object(cache.np, 'load', return_value=Archive()):
            actual = cache.read_recent_level(Path('unused.npz'), dict(offset=0, rows=3, seed=7), cached)
        self.assertTrue(all(count == 1 for count in calls.values()))
        np.testing.assert_array_equal(actual[:, 0, 0], [0, 1, 2])
        self.assertTrue(np.array_equal(cached['optimal'], [1, 0, 0]))
        cached['row_kind'] = np.array([2, 1, 0], np.int8)
        with patch.object(cache.np, 'load', return_value=Archive()), self.assertRaisesRegex(ValueError, 'alignment'):
            cache.read_recent_level(Path('unused.npz'), dict(offset=0, rows=3, seed=7), cached)

    def test_publication_validates_hashes_split_disjointness_and_parity(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp); report = publication(path)
            verified = cache.validate_published(path)
            self.assertEqual(len(verified['validated_output_hashes']), 14)
            report['parity']['recent']['maximum_absolute_error'] = .01
            (path / 'manifest.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'parity'):
                cache.validate_published(path)
            report['parity']['recent']['maximum_absolute_error'] = 0.
            (path / 'manifest.json').write_text(json.dumps(report))
            with (path / 'train/semantic.npy').open('ab') as f:
                f.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                cache.validate_published(path)

    def test_publication_rejects_identity_mismatch_and_source_aliasing(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp); report = publication(path)
            report['aligned_source_files']['recent']['rows']['sha256'] = '0' * 64
            (path / 'manifest.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'row identity'):
                cache.validate_published(path)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp); report = publication(path)
            np.save(path / 'validation/seeds.npy', np.array([1], np.int64))
            digest = cache.base.sha(path / 'validation/seeds.npy')
            report['files']['validation']['seeds']['sha256'] = digest
            report['aligned_source_files']['validation']['seeds']['sha256'] = digest
            (path / 'manifest.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'leakage'):
                cache.validate_published(path)

    def test_runtime_and_teacher_hashes_are_rechecked_without_frame_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp); report = publication(path)
            report['runtime_source_bindings'][str(cache.RUNTIME_SOURCES[0])] = '0' * 64
            (path / 'manifest.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                cache.validate_published(path)
            report['runtime_source_bindings'][str(cache.RUNTIME_SOURCES[0])] = cache.base.sha(cache.RUNTIME_SOURCES[0])
            (path / 'manifest.json').write_text(json.dumps(report))
            (path / 'fixture-teacher.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                cache.validate_published(path)

    def test_atomic_publication_cannot_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); stage = root / 'stage'; stage.mkdir(); out = root / 'out'; out.mkdir()
            (out / 'sentinel').write_text('preserve')
            with self.assertRaises(OSError):
                cache.base.publish_directory(stage, out)
            self.assertEqual((out / 'sentinel').read_text(), 'preserve')
            self.assertTrue(stage.exists())


if __name__ == '__main__':
    unittest.main()
