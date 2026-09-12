"""Trainer boundaries: distinct generated batches and counterfactual labels."""
import unittest
import numpy as np
import torch

from tools.train_structured_transition import branch_batch, sample_rows, training_scale, event_counts


class StructuredTrainerTests(unittest.TestCase):
    def test_curriculum_changes_ratio_without_repeating_levels_within_batch(self):
        data = {'seeds': np.arange(2048), 'difficulties': np.arange(2048) % 5 + 1}
        averages = []
        for progress in (0., 1.):
            rng = np.random.default_rng(42)
            ids = sample_rows(data, 1024, progress, rng)
            self.assertEqual(len(np.unique(ids)), 1024)
            averages.append(data['difficulties'][ids].mean())
        self.assertGreater(averages[1]-averages[0], .7)
        with self.assertRaises(ValueError): sample_rows(data, 4096, .5, np.random.default_rng(42))

    def test_selected_action_cannot_change_current_inputs_but_selects_matching_labels(self):
        data = {'fields': np.arange(2*148*96).reshape(2,148,96).astype(np.float16),
                'next_fields': np.zeros((2,4,148,96), np.float16)}
        for key, shape in [('player_cell',(2,2)),('triple',(2,3)),('steps',(2,)),('lives',(2,))]:
            data[key] = np.ones(shape,dtype=np.int16)
            data['next_'+key] = np.zeros((2,4,*shape[1:]),dtype=np.int16)
            data['next_'+key][:,3] = 3
        for key in ('lost_life','terminal','won'):
            data[key] = np.zeros((2,4),bool)
        data['lost_life'][0,3] = True
        first = branch_batch(data,np.array([0,1]),np.array([0,0]),'cpu')
        second = branch_batch(data,np.array([0,1]),np.array([3,3]),'cpu')
        torch.testing.assert_close(first[0],second[0],atol=0,rtol=0)
        self.assertTrue((second[3]['next_triple']==3).all())
        self.assertEqual(second[3]['lost_life'].tolist(),[True,False])
        with self.assertRaises(ValueError):branch_batch(data,[0],[4],'cpu')

    def test_feature_scale_finite_on_constant_training_features(self):
        data = {'seeds': np.arange(2), 'fields': np.ones((2,148,96),np.float16)}
        scale = training_scale(data)
        np.testing.assert_allclose(scale,.1)
        self.assertEqual(scale.shape,(48,))

    def test_event_support_counts_levels_separately_from_correlated_branches(self):
        data = {'lost_life':np.array([[1,1,0,0],[0,0,0,0]],bool),
                'terminal':np.array([[1,1,1,0],[0,0,0,0]],bool),
                'won':np.array([[0,0,1,0],[0,0,0,0]],bool)}
        counts = event_counts(data)
        self.assertEqual(counts['lost_life']['positive_branches'],2)
        self.assertEqual(counts['lost_life']['positive_levels'],1)
        self.assertEqual(counts['terminal_failure']['positive_branches'],2)
        self.assertEqual(counts['terminal_failure']['positive_levels'],1)


if __name__ == '__main__': unittest.main()
