import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from tests.test_world_exploratory_sequences import ExploratoryTrainingTests
from pebby.agent.world_exploratory_sequences import build_index_arrays,FourStepIndex,FORMAT,MODE
from pebby.agent.world_train import as_tensors
from tools.build_structured_exploratory_cache import select_anchors,sequence_batch,encode_sequence,write_cache
from pebby.agent import world_data as wd

class PublicEncoder(torch.nn.Module):
    def __init__(self):super().__init__();self.dummy=torch.nn.Parameter(torch.zeros(()),requires_grad=False);self.calls=[]
    def forward(self,frames,valid,actions):
        self.calls.append((frames.clone(),valid.clone(),actions.clone()))
        return frames[:,-1,0,0].float()[:,None,None].expand(-1,148,96)/16

class ExploratoryCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):ExploratoryTrainingTests.setUpClass();cls.data=ExploratoryTrainingTests.data
    def index(self):
        a,f,_=build_index_arrays(self.data)
        return FourStepIndex(a,f,dict(format=FORMAT,mode=MODE,split='train'))
    def test_real_public_history_alignment_and_label_isolation(self):
        ix=self.index();rows=select_anchors(self.data,ix,1);t=as_tensors(self.data);encoder=PublicEncoder()
        current,nextf,labels,future,_=encode_sequence(encoder,self.data,t,ix,rows)
        torch.testing.assert_close(encoder.calls[0][0],t['frames'][rows])
        torch.testing.assert_close(encoder.calls[1][0],t['frames'][future.reshape(-1)])
        np.testing.assert_array_equal(labels['next_triple'],self.data['current_triple'][future])
        self.assertEqual(nextf.shape,(1,4,148,96));self.assertEqual(current.dtype,np.float16)
        d=copy.deepcopy(self.data);d['frames'][future[0,0],0,0,0]^=1
        with self.assertRaisesRegex(ValueError,'histories|frames'):sequence_batch(d,as_tensors(d),ix,rows)
    def test_selection_balance_determinism_and_wrong_mode(self):
        ix=self.index()
        np.testing.assert_array_equal(select_anchors(self.data,ix,1),select_anchors(self.data,ix,1))
        with self.assertRaisesRegex(ValueError,'insufficient'):select_anchors(self.data,ix,8)
        synthetic={'seeds':np.arange(10),'meta':{'levels':[{'seed':i,'difficulty':i//2+1} for i in range(10)]}}
        syn=FourStepIndex(np.arange(10),np.zeros((10,4),dtype=np.int64),ix.meta)
        selected=select_anchors(synthetic,syn,8)
        self.assertEqual(len(np.unique(selected)),8)
        self.assertEqual(np.bincount(selected//2+1,minlength=6)[1:].tolist(),[2,2,2,1,1])
        bad=FourStepIndex(ix.anchor_row,ix.future_rows,{**ix.meta,'split':'validation'})
        with self.assertRaises(ValueError):select_anchors(self.data,bad,1)
    def test_atomic_cache_initial_rows_and_source_guard(self):
        from pebby.agent.world_exploratory_sequences import build_sidecar
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);source=p/'source.npz';wd.save(source,self.data)
            ix=build_sidecar(self.data,source,p/'index.npz')
            manifest=write_cache(self.data,ix,source,p/'index.npz',p/'cache',PublicEncoder(),{},levels=1)
            self.assertEqual(manifest['split'],'train');self.assertTrue(manifest['history_verified_against_actual_source_rows'])
            row=np.load(p/'cache/source_initial_rows.npy');self.assertEqual(self.data['history_valid'][row].sum(),1)
            with self.assertRaisesRegex(ValueError,'checksum mismatch|changed'):
                write_cache(self.data,ix,source,p/'index.npz',p/'bad',PublicEncoder(),{},levels=1,extra_hashes={str(source):'bad'})
            self.assertFalse((p/'bad').exists())
