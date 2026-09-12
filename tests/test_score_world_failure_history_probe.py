import unittest
import numpy as np
import torch
from tests.test_policy_history import RecordingPolicy
from tools.score_world_failure_history_probe import metrics, predict, validate_probe

class HistoryScoreTests(unittest.TestCase):
    def test_zero_masks_excluded_and_set_ties_use_uniform_ce(self):
        logits=torch.tensor([[0.,0.,0.,0.],[100.,-100.,0.,0.],[0.,2.,0.,0.]])
        result=metrics(logits,np.array([3,0,2]),np.array([0,0,0]))
        self.assertEqual(result['valid_rows'],2)
        self.assertEqual(result['excluded_zero_masks'],1)
        self.assertEqual(result['optimal_set_accuracy'],1)
        self.assertEqual(result['corrected_vs_actor'],1)
        self.assertEqual(result['regressed_vs_actor'],0)
        expected=(np.log(4)+np.log(1+3*np.exp(-2)))/2
        self.assertAlmostEqual(result['uniform_optimal_target_ce'],expected,places=6)
        empty=metrics(logits[:1],np.array([0]),np.array([0]))
        self.assertIsNone(empty['uniform_optimal_target_ce'])
        self.assertIsNone(empty['optimal_set_accuracy'])

    def test_checkpoint_history_and_batch_validation(self):
        with self.assertRaisesRegex(ValueError,'H8'):
            predict(RecordingPolicy(history=3),{},1)
        with self.assertRaisesRegex(ValueError,'batch'):
            predict(RecordingPolicy(history=8),{},33)

    def test_training_schema_rejected(self):
        with self.assertRaisesRegex(ValueError,'validation history probe'):
            validate_probe({'meta':{'format':'pebby.ls20-world.v1','source':'generated_only','split':'train'}})

if __name__=='__main__': unittest.main()
