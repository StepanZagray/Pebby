import unittest
from types import SimpleNamespace
import numpy as np
import torch
from tools.train_structured_distance_ranking import ranking_loss, objective, preflight, make_head
from pebby.agent.structured_distance import StructuredDistanceReadout

class RankingTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(42)
    def test_uniform_optimal_and_loss_exclusion(self):
        z=torch.zeros(1,4,5,requires_grad=True)
        loss,n=ranking_loss(z,torch.tensor([3]),torch.tensor([[False,False,False,True]]))
        self.assertAlmostEqual(float(loss),np.log(3),places=5);self.assertEqual(int(n),1)
        loss.backward();self.assertEqual(float(z.grad[0,3].abs().sum()),0.);self.assertGreater(float(z.grad[0,:3].abs().sum()),0.)
    def test_all_optimal_finite_zero_and_bad_mask(self):
        z=torch.randn(2,4,5,requires_grad=True)
        loss,n=ranking_loss(z,torch.tensor([15,7]),torch.tensor([[False]*4,[False,False,False,True]]))
        self.assertEqual(float(loss),0.);self.assertEqual(int(n),0);loss.backward();self.assertTrue(torch.isfinite(z.grad).all())
        with self.assertRaises(ValueError):ranking_loss(z,torch.tensor([0,7]),torch.zeros(2,4,dtype=torch.bool))
        with self.assertRaises(ValueError):ranking_loss(z,torch.tensor([15,7]),torch.ones(2,4,dtype=torch.bool))
    def test_unreachable_negative_and_stability(self):
        z=torch.full((1,4,5),-10000.);z[0,0,0]=10000.;z[0,1:,4]=10000.;z.requires_grad_()
        loss,n=ranking_loss(z,torch.tensor([1]),torch.zeros(1,4,dtype=torch.bool))
        self.assertTrue(torch.isfinite(loss));self.assertLess(float(loss),1e-5);loss.backward();self.assertTrue(torch.isfinite(z.grad).all())
    def test_real_optimizer_fallback_fresh_initial(self):
        h=StructuredDistanceReadout(max_distance=3);initial={'config':h.config(),'weights':h.state_dict()}
        rng=np.random.default_rng(1);d={'seeds':np.arange(3),'difficulties':np.array([1,2,3]),'fields':rng.normal(size=(3,148,96)).astype('f'),'next_fields':rng.normal(size=(3,4,148,96)).astype('f'),'imagined_fields':rng.normal(size=(3,4,148,96)).astype('f'),'current_targets':np.ones(3,dtype='int64'),'next_targets':np.ones((3,4),dtype='int64'),'optimal':np.ones(3,dtype='uint8'),'lost_life':np.zeros((3,4),bool)}
        result=preflight(d,initial,SimpleNamespace(max_batch=8,device='cpu'));self.assertEqual(result['batch_size'],2)
        self.assertGreater(result['attempts'][0]['gradient_norm'],0)
        fresh=make_head(initial,'cpu')
        for k,v in h.state_dict().items():self.assertTrue(torch.equal(v,fresh.state_dict()[k]))
        with self.assertRaises(ValueError):preflight(d,initial,SimpleNamespace(max_batch=3,device='cpu'))
        a,_=objective(fresh,d,np.array([0,1]),'cpu',0);b,info=objective(fresh,d,np.array([0,1]),'cpu',1)
        self.assertAlmostEqual(float(b-a),float(info['ranking_loss']),places=5)

if __name__=='__main__':unittest.main()
