import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from pebby.agent import world_data as wd
from pebby.agent.world_exploratory_sequences import build_index_arrays,build_sidecar,load_sidecar
from pebby.agent.world_sequences import FourStepSampler
from tests.test_policy_history import corridor

class ExploratoryTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        specs=[]
        for seed in (8,9):
            spec=corridor();spec.update(seed=seed,difficulty=1);spec['goals'][0]['cell']=(8,3);specs.append(spec)
        cls.data=wd.build(specs,workers=1,history=8,samples=16,coverage='mixed',epsilon=0)

    def test_real_complete_live_clips_and_expert_exclusion(self):
        a,f,c=build_index_arrays(self.data)
        self.assertGreater(len(a),0)
        np.testing.assert_array_equal(f,a[:,None]+np.arange(1,5))
        for anchor,future in zip(a,f):
            seed=self.data['seeds'][anchor]
            level=next(x for x in self.data['meta']['levels'] if x['seed']==seed)
            rows=np.flatnonzero(self.data['seeds']==seed)[:level['explore_samples']]
            self.assertTrue(set([anchor,*future])<=set(rows))
            for i,j in zip([anchor,*future[:-1]],future):
                action=self.data['previous_actions'][j,-1]
                np.testing.assert_array_equal(self.data['next_frames'][i,action],self.data['frames'][j,-1])
                self.assertFalse(self.data['terminal'][i,action])

    def test_split_and_policy_provenance_rejected(self):
        for change in ({'split':'validation'},{'on_policy_rows':[]}):
            d=copy.deepcopy(self.data);d['meta'].update(change)
            with self.assertRaises(ValueError):build_index_arrays(d)
        d=copy.deepcopy(self.data);d['seeds']+=1000000
        with self.assertRaisesRegex(ValueError,'namespace'):build_index_arrays(d)

    def test_corruption_and_reset_exclusion(self):
        a,f,_=build_index_arrays(self.data);i=int(a[0]);j=int(f[0,0]);act=int(self.data['previous_actions'][j,-1])
        d=copy.deepcopy(self.data);d['next_frames'][i,act,0,0]^=1
        with self.assertRaisesRegex(ValueError,'continuity'):build_index_arrays(d)
        d=copy.deepcopy(self.data);d['distances'][i,act]+=1
        with self.assertRaisesRegex(ValueError,'distance'):build_index_arrays(d)
        d=copy.deepcopy(self.data);d['lost_life'][i,act]=True
        aa,_,_=build_index_arrays(d);self.assertNotIn(i,aa)

    def test_binding_and_old_sampler_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);source=p/'source.npz';wd.save(source,self.data)
            index=build_sidecar(self.data,source,p/'index.npz')
            loaded=load_sidecar(p/'index.npz',source,self.data)
            np.testing.assert_array_equal(index.anchor_row,loaded.anchor_row)
            with self.assertRaises(ValueError):FourStepSampler(self.data,self.data,index)
            source.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'binding'):load_sidecar(p/'index.npz',source,self.data)
