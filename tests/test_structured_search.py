import unittest
import torch
from pebby.agent.structured_search import search

class SearchTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);self.field=torch.zeros(1,148,96)
    def head(self,f):
        z=torch.full((len(f),4),-torch.inf);z[:,1]=0;return z
    def test_formula_absorption_and_roots(self):
        def head(f):return torch.tensor([[0.,-1000.,-1000.]]).expand(len(f),-1)
        def world(f,a):
            p=torch.zeros(len(f),3);p[:,2]=1;p[a==0]=torch.tensor([0.,1.,0.]);p[a==1]=torch.tensor([1.,0.,0.]);return f+1,p
        r=search(self.field,world,head,lambda p:p,depth=1,gamma=.9)
        self.assertEqual(r['transition_counts'],[4]);self.assertEqual(r['actions'].tolist(),[0])
        self.assertAlmostEqual(r['results'][0]['roots'][0]['score'],.9)
        self.assertEqual(r['results'][0]['roots'][1]['score'],0.)
        r=search(self.field,world,head,lambda p:p,depth=4,gamma=.9)
        self.assertAlmostEqual(r['results'][0]['roots'][0]['score'],.9)
        self.assertEqual(r['results'][0]['roots'][1]['score'],0.)
    def test_depth1_fractional_formula(self):
        def world(f,a):return f,torch.tensor([[.2,.3,.5]]).expand(len(f),-1)
        def head(f):return torch.tensor([[-1000.,0.,-1000.]]).expand(len(f),-1)
        r=search(self.field,world,head,lambda p:p,depth=1,gamma=.8)
        self.assertAlmostEqual(r['results'][0]['roots'][0]['score'],.8*.3+.8*.5*.8,places=6)
    def test_chronological_counts_chunk_and_ties(self):
        seen=[]
        def world(f,a):
            seen.extend(f[:,0,0].tolist());return f+1,torch.tensor([[0.,0.,1.]]).expand(len(f),-1)
        def head(f):return torch.zeros(len(f),3)
        r=search(self.field.expand(2,-1,-1),world,head,lambda p:p,depth=4,chunk_size=3)
        self.assertEqual(r['transition_counts'],[148,148]);self.assertEqual(r['actions'].tolist(),[0,0])
        self.assertEqual(seen.count(0.),8);self.assertEqual(seen.count(1.),32);self.assertEqual(seen.count(2.),128);self.assertEqual(seen.count(3.),128)
        self.assertEqual(r['results'][0]['roots'][2]['actions'],[2,0,0,0])
    def test_known_value_choice_no_grad_and_gamma_one(self):
        def world(f,a):
            self.assertFalse(torch.is_grad_enabled())
            out=f.clone();out[:,0,0]=a
            return out,torch.tensor([[0.,0.,1.]]).expand(len(f),-1)
        def head(f):
            self.assertFalse(torch.is_grad_enabled())
            z=torch.full((len(f),5),-1000.)
            d=3-f[:,0,0].long();z[torch.arange(len(f)),d]=0
            return z
        r=search(self.field.requires_grad_(),world,head,lambda p:p,depth=1,gamma=.9)
        self.assertEqual(r['actions'].tolist(),[3])
        self.assertAlmostEqual(r['results'][0]['roots'][0]['score'],.9**4,places=6)
        r=search(self.field,world,head,lambda p:p,depth=1,gamma=1.)
        self.assertEqual(r['actions'].tolist(),[0])
        self.assertIsNone(self.field.grad)

    def test_fail_closed_no_teacher(self):
        world=lambda f,a:(f,torch.ones(len(f),3))
        with self.assertRaises(ValueError):search(self.field,world,self.head,lambda p:p)
        with self.assertRaises(ValueError):search(self.field,world,self.head,lambda p:p,gamma=0)
        with self.assertRaises(TypeError):search(self.field,world,self.head,lambda p:p,teacher=self.field)
        with self.assertRaises(ValueError):search(self.field,world,self.head,lambda p:p,depth=5)

if __name__=='__main__':unittest.main()
