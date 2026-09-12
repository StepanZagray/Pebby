import unittest
from pathlib import Path
import numpy as np
import torch
from pebby.agent.structured_distance import StructuredDistanceReadout,current_distance_labels,next_distance_labels,distance_targets,distance_loss,discounted_value

class DistanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_public_forward_gradients_shared_loops_roundtrip(self):
        torch.manual_seed(42);model=StructuredDistanceReadout(max_distance=93);field=torch.randn(3,148,96)
        logits=model(field);self.assertEqual(logits.shape,(3,95));r=distance_loss(logits,torch.tensor([0,93,-1]),93);r['loss'].backward()
        self.assertGreater(float(model.attention.in_proj_weight.grad.abs().sum()),0);self.assertGreater(float(model.position.grad.abs().sum()),0)
        other=StructuredDistanceReadout(model.config());other.load_state_dict(model.state_dict());torch.testing.assert_close(other(field),logits)
        self.assertEqual(model.parameter_count(),StructuredDistanceReadout(max_distance=93,loops=4).parameter_count())
        with self.assertRaises(TypeError):model(field,distances=torch.zeros(3))
    def test_semantics_masks_resets_and_no_clipping(self):
        d=torch.tensor([[0,2,-1,5],[4,4,6,2]]);terminal=torch.tensor([[1,0,1,0],[0,0,0,0]],dtype=torch.bool);won=torch.tensor([[1,0,0,0],[0,0,0,0]],dtype=torch.bool);lost=torch.tensor([[0,0,0,1],[0,0,0,1]],dtype=torch.bool)
        self.assertEqual(current_distance_labels(torch.tensor([1,3]),d,terminal,won,lost).tolist(),[1,5]);self.assertEqual(next_distance_labels(d,terminal,won)[1,3],2)
        for mask in ([1,1],[1,0],[2,3]):
            with self.assertRaises(ValueError):current_distance_labels(torch.tensor(mask),d,terminal,won,lost)
        with self.assertRaises(ValueError):distance_targets(torch.tensor([94]),93)
        with self.assertRaises(ValueError):distance_targets(torch.tensor([1.]),93)
        self.assertEqual(distance_targets(torch.tensor([-1,0,93]),93).tolist(),[94,0,93])
        bad=d.clone();bad[0,2]=3
        with self.assertRaises(ValueError):next_distance_labels(bad,terminal,won)
        with self.assertRaises(ValueError):next_distance_labels(torch.tensor([0]),torch.tensor([False]),torch.tensor([False]))
    def test_value_and_metric_denominators(self):
        logits=torch.full((3,5),-100.);logits[0,0]=100;logits[1,2]=100;logits[2,4]=100
        torch.testing.assert_close(discounted_value(logits,.5),torch.tensor([1.,.25,0.]))
        r=distance_loss(logits,torch.tensor([0,2,-1]),3);self.assertEqual(r['metrics']['finite_count'],2);self.assertEqual(r['metrics']['finite_within1_count'],2);self.assertEqual(r['metrics']['unreachable_tp'],1)
        r=distance_loss(logits[2:],torch.tensor([-1]),3);self.assertEqual(r['metrics']['finite_count'],0);self.assertEqual(r['metrics']['finite_mae_sum'],0)
        with self.assertRaises(ValueError):discounted_value(logits,1.1)
    @unittest.skipUnless(Path('data/structured-field-16384/train/distances.npy').exists(),'requires local verified H1 cache')
    def test_real_train_validation_labels_and_live_resets(self):
        for split in ('train','validation'):
            p=Path('data/structured-field-16384')/split
            arrays={k:np.load(p/(k+'.npy')) for k in ('distances','optimal','terminal','won','lost_life')}
            # Include actual reset cases and ordinary rows without frame loading.
            selected=np.unique(np.r_[np.arange(16),np.flatnonzero(arrays['lost_life'].any(1))[:16]])
            values={k:torch.tensor(v[selected]) for k,v in arrays.items()}
            current=current_distance_labels(values['optimal'],values['distances'],values['terminal'],values['won'],values['lost_life']);self.assertTrue((current>0).all())
            following=next_distance_labels(values['distances'],values['terminal'],values['won']);self.assertTrue((following[values['lost_life']]>=0).all())
