import unittest
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from pebby.agent.structured_distance import StructuredDistanceReadout,distance_targets
from tools.train_structured_distance import batch_loss,ranking,support_from_train,new_head,preflight,validate_batch_size,diagnostic_rows,evaluate

class DistanceTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_equal_three_losses_and_unique_levels(self):
        head=StructuredDistanceReadout(max_distance=3);data={'seeds':np.arange(2),'fields':np.random.default_rng(1).normal(size=(2,148,96)).astype('float32'),'next_fields':np.zeros((2,4,148,96),np.float32),'imagined_fields':np.ones((2,4,148,96),np.float32),'current_targets':np.array([1,2]),'next_targets':np.tile(np.array([0,1,2,4]),(2,1))};rows=np.arange(2);actions=np.array([0,3]);loss=batch_loss(head,data,rows,actions,'cpu');terms=[]
        for k,t in [('fields','current_targets'),('next_fields','next_targets'),('imagined_fields','next_targets')]:
            a=data[k][rows] if k=='fields' else data[k][rows,actions];b=data[t][rows] if k=='fields' else data[t][rows,actions];terms.append(F.cross_entropy(head(torch.tensor(a)),torch.tensor(b)))
        torch.testing.assert_close(loss,sum(terms)/3);loss.backward();self.assertGreater(float(head.output.weight.grad.abs().sum()),0)
        with self.assertRaises(ValueError):batch_loss(head,data,np.array([0,0]),actions,'cpu')
    def test_train_support_no_validation_expansion_and_seed_reset(self):
        D=support_from_train(np.array([2,3]),np.array([-1,4]));self.assertEqual(D,4)
        with self.assertRaises(ValueError):distance_targets(torch.tensor([5]),D)
        a=new_head(D,'cpu');torch.randn(100);b=new_head(D,'cpu')
        for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key],atol=0,rtol=0)
    def test_ties_and_reset_state_ranking_are_explicit(self):
        data={'optimal':np.array([2,1]),'lost_life':np.array([[1,0,0,0],[0,0,0,0]],bool),'distances':np.array([[1,2,3,-1],[0,3,4,5]]),'won':np.array([[0,0,0,0],[1,0,0,0]],bool)}
        r=ranking(np.array([[.9,.9,.2,.1],[.9,.2,.1,.1]]),data,np.arange(2));self.assertEqual(r['tie_rows'],1);self.assertEqual(r['argmax_optimal'],1);self.assertEqual(r['tie_has_optimal'],2);self.assertEqual(r['argmax_lost_life'],1)

    def test_real_cpu_preflight_rounds_down_small_bank_and_resets(self):
        n=3;data={'seeds':np.arange(n),'difficulties':np.array([1,2,3]),'fields':np.zeros((n,148,96),np.float32),'next_fields':np.zeros((n,4,148,96),np.float32),'imagined_fields':np.ones((n,4,148,96),np.float32),'current_targets':np.ones(n,np.int64),'next_targets':np.ones((n,4),np.int64)}
        args=SimpleNamespace(max_batch=8,device='cpu',lr=.001)
        fresh=new_head(3,'cpu');result=preflight(data,3,args)
        self.assertEqual(result['batch_size'],2);self.assertEqual(result['unique_levels'],2);self.assertEqual(result['attempts'][0]['status'],'fits')
        self.assertGreater(result['gradient_norm'],0)
        reset=new_head(3,'cpu')
        for key,value in fresh.state_dict().items():torch.testing.assert_close(value,reset.state_dict()[key],atol=0,rtol=0)
        for size in (0,3,6,1025):
            with self.assertRaises(ValueError):validate_batch_size(size)
    def test_fixed_train_selection_distinct_balanced(self):
        data={'seeds':np.arange(1100),'difficulties':np.arange(1100)%5+1}
        a=diagnostic_rows(data);b=diagnostic_rows(data)
        np.testing.assert_array_equal(a,b);self.assertEqual(len(np.unique(a)),1024)
        self.assertEqual(np.bincount(data['difficulties'][a],minlength=6)[1:].tolist(),[205,205,205,205,204])

    def test_cli_rejects_non_power_of_two_before_loading(self):
        import subprocess,sys
        command=[sys.executable,'-m','tools.train_structured_distance','--cache','missing','--imagined-cache','missing','--dynamics','missing','--checkpoint','missing','--report','missing','--max-batch','3']
        result=subprocess.run(command,text=True,capture_output=True,timeout=10)
        self.assertEqual(result.returncode,2);self.assertIn('power of two',result.stderr)
