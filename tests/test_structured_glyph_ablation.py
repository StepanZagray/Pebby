import copy
import unittest

import numpy as np
import torch

from tools.train_structured_glyph_ablation import check_initial_encoder, evaluation_rows, make_model
from pebby.agent.structured_global_glyph import GlobalGlyphTransition, GLOBAL_GLYPH_FORMAT


class GlyphAblationBoundaries(unittest.TestCase):
    def test_initialization_cannot_cross_encoder_provenance(self):
        encoder = {key: {'bound': key} for key in ('config', 'sources', 'parameter_counts', 'code_hashes')}
        initial = {'cache_manifests': {'train': {'field_encoder': copy.deepcopy(encoder)}}}
        check_initial_encoder(initial, encoder | {'checkpoint_hashes': {'redundant': 'hash'}})
        initial['cache_manifests']['train']['field_encoder']['sources'] = {'different': True}
        with self.assertRaisesRegex(ValueError, 'different field encoder'):
            check_initial_encoder(initial, encoder)

    def test_sparse_difficulties_still_produce_exact_distinct_sample(self):
        data = {'seeds': np.arange(12), 'difficulties': np.array([1]*9+[3]*2+[5])}
        rows = evaluation_rows(data, 11, 42)
        self.assertEqual(len(rows), 11)
        self.assertEqual(len(np.unique(rows)), 11)
        self.assertTrue(np.array_equal(rows, evaluation_rows(data, 11, 42)))
        self.assertEqual(set(evaluation_rows(data, 50, 42)), set(range(12)))

    def test_local_comparison_starts_with_identical_global_weights_and_predictions(self):
        torch.set_num_threads(1)
        initial_model = GlobalGlyphTransition()
        initial = {'format': GLOBAL_GLYPH_FORMAT, 'config': initial_model.config(),
                   'weights': initial_model.state_dict()}
        global_model = make_model(initial, 'global-balanced', 42, 'cpu').eval()
        local_model = make_model(initial, 'local-balanced', 42, 'cpu').eval()
        fields = torch.randn(2, 148, 96)
        actions = torch.tensor([0, 3])
        with torch.no_grad():
            self.assertTrue(torch.equal(global_model.predict(fields, actions),
                                        local_model.predict(fields, actions)))
        with self.assertRaisesRegex(ValueError, 'base comparison requires'):
            make_model(initial, 'baseline', 42, 'cpu')


if __name__ == '__main__':
    unittest.main()
