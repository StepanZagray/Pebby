import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pebby.agent.world_cache import cached_arrays
from pebby.agent.world_train import as_tensors, load_dataset
from tests.test_world_glyph import make_glyph_synthetic


class WorldCacheTests(unittest.TestCase):
    def test_training_loader_equal_and_batch_tensors_share_disk_backing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'data.npz'
            data = make_glyph_synthetic(seed=41, levels=1, steps=4, history=4)
            np.savez_compressed(path, **data, meta=np.array(json.dumps({'source': 'synthetic'})))
            original = load_dataset(path)
            mapped = load_dataset(path, cache_dir=root / 'cache')
            for name, value in original.items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(value, mapped[name])
                else:
                    self.assertEqual(value, mapped[name])
            self.assertIsInstance(mapped['frames'], np.memmap)
            tensors = as_tensors(mapped)
            self.assertEqual(tensors['frames'].data_ptr(), mapped['frames'].ctypes.data)
            # Copy-on-write protects persistent cache if a consumer mutates a tensor.
            expected = original['frames'][0, 0, 0, 0]
            tensors['frames'][0, 0, 0, 0] = (int(expected) + 1) % 16
            with patch('zipfile.ZipFile', side_effect=AssertionError('cache hit decompressed NPZ')):
                again = load_dataset(path, cache_dir=root / 'cache')
            self.assertEqual(again['frames'][0, 0, 0, 0], expected)
            shortened = load_dataset(path, history=2, cache_dir=root / 'cache')
            tensor = as_tensors(shortened)['frames']
            self.assertEqual(tensor.data_ptr(), shortened['frames'].ctypes.data)
            np.testing.assert_array_equal(tensor.numpy(), original['frames'][:, -2:])
            self.assertEqual(tensor.stride(0), original['frames'].strides[0])

    def test_corrupt_cache_fails_and_new_source_gets_new_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'data.npz'
            np.savez_compressed(path, x=np.arange(9))
            with cached_arrays(path, root / 'cache', ['x']) as archive:
                np.testing.assert_array_equal(archive['x'], np.arange(9))
            cached = next((root / 'cache').glob('*/x.npy'))
            with cached.open('r+b') as stream:
                stream.seek(-1, 2)
                stream.write(b'\xff')
            with self.assertRaisesRegex(ValueError, 'corrupted'):
                with cached_arrays(path, root / 'cache', ['x']):
                    pass
            np.savez_compressed(path, x=np.arange(10))
            with cached_arrays(path, root / 'cache', ['x']) as archive:
                np.testing.assert_array_equal(archive['x'], np.arange(10))
            for manifest in (root / 'cache').glob('*/manifest.json'):
                manifest.unlink()
            with self.assertRaisesRegex(ValueError, 'manifest is missing'):
                with cached_arrays(path, root / 'cache', ['x']):
                    pass

    def test_object_array_rejected_without_publishing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(root / 'bad.npz', x=np.array([{}], dtype=object))
            with self.assertRaises(ValueError):
                with cached_arrays(root / 'bad.npz', root / 'cache', ['x']):
                    pass
            self.assertFalse(list((root / 'cache').glob('*/manifest.json')))


if __name__ == '__main__':
    unittest.main()
