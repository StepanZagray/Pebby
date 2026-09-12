import unittest
import numpy as np
import torch
from tools.preflight_reference_world import check_distance_support, candidates, choose_capacity, fresh_config

class ReferencePreflightTests(unittest.TestCase):
    def test_exact_distance_boundary_and_unreachable(self):
        result=check_distance_support({'distances':np.array([[-1,0,126,127]],dtype=np.int16)},128)
        self.assertEqual(result['maximum_finite_distance'],127)
        self.assertEqual(result['unreachable_branches'],1)
        with self.assertRaisesRegex(ValueError,'overflow'):
            check_distance_support({'distances':np.array([[0,128,1,-1]])},128)
        for bad in (np.ones((1,4),dtype=float),np.array([[0,-2,0,0]]),np.ones((4,1),dtype=int)):
            with self.assertRaises(ValueError):check_distance_support({'distances':bad})

    def test_only_cuda_oom_allows_descending_fallback(self):
        seen=[];records=[]
        def attempt(size):
            seen.append(size)
            if size>4:raise torch.cuda.OutOfMemoryError('synthetic capacity boundary')
            return {'batch_size':size,'status':'complete'}
        self.assertEqual(choose_capacity(16,attempt,records.append),4)
        self.assertEqual(seen,[16,8,4])
        self.assertEqual([r['status'] for r in records],['cuda_oom','cuda_oom','complete'])
        for error in (ValueError('bad labels'),RuntimeError('nonfinite')):
            def broken(size):raise error
            with self.assertRaises(type(error)):choose_capacity(16,broken,records.append)

    def test_power_two_and_fresh_config(self):
        self.assertEqual(candidates(1024),[1024,512,256,128,64,32,16,8,4,2,1])
        for bad in (0,3,1025,2048,True):
            with self.assertRaises(ValueError):candidates(bad)
        cfg=fresh_config()
        self.assertEqual((cfg.history,cfg.loops,cfg.max_distance),(8,6,128))
        self.assertTrue(cfg.grounding and cfg.glyph_recall)
        self.assertFalse(cfg.cell_recall)

    def test_production_parser_exposes_exact_distance_guard(self):
        from pebby.agent.world_train import build_parser
        args=build_parser().parse_args(['--train','train.npz','--max-distance','128','--require-exact-distances'])
        self.assertTrue(args.require_exact_distances)
        self.assertEqual(args.max_distance,128)

if __name__=='__main__':unittest.main()
