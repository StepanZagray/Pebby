import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from pebby.agent import world_train
from tools.build_reference_world_cache import main as collect_main


class ReferenceBaseGuards(unittest.TestCase):
    def test_training_rejects_distance_overflow_before_model_allocation(self):
        data = {'seeds': np.array([7]), 'distances': np.array([[128, 1, -1, 3]])}
        with patch.object(world_train, 'load_dataset', return_value=data), \
             patch.object(world_train, 'WorldPolicy') as model:
            with self.assertRaises(SystemExit):
                world_train.main(['--train', 'unused.npz', '--device', 'cpu',
                                  '--max-distance', '128', '--require-exact-distances'])
            model.assert_not_called()

    def test_resume_cannot_adopt_orphaned_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'train.npz').write_bytes(b'unbound output')
            with self.assertRaisesRegex(ValueError, 'original manifest'):
                collect_main(['--bank-dir', 'unused', '--out-dir', str(root), '--resume'])


if __name__ == '__main__':
    unittest.main()
