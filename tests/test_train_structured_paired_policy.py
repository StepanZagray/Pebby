import copy
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from tools.train_structured_paired_policy import paired_batch,validate_pair
from tools.train_structured_policy import load_policy_cache,loss_for_rows,new_head
from tools.train_structured_transition import sample_rows


class PairedPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def views(self,n=1200):
        first={'seeds':np.arange(n),'difficulties':np.arange(n)%5+1,'source_rows':np.arange(n)*2,
               'fields':np.arange(n,dtype=np.float32)[:,None],'optimal':np.ones(n,np.int64)}
        second={k:v.copy() for k,v in first.items()};second['fields']+=10000;second['source_rows']+=1;second['optimal'][:]=8
        return first,second

    def test_1024_unique_balanced_views_keep_baseline_level_rng_and_order(self):
        views=self.views();baseline=np.random.default_rng(42);paired=np.random.default_rng(42);vrng=np.random.default_rng(43)
        for progress in (0,.5,1):
            wanted=sample_rows(views[0],1024,progress,baseline);rows=sample_rows(views[0],1024,progress,paired)
            batch,which=paired_batch(views,rows,vrng,'direct')
            np.testing.assert_array_equal(rows,wanted);np.testing.assert_array_equal(batch['seeds'],rows)
            self.assertEqual(len(np.unique(batch['seeds'])),1024);self.assertEqual(np.bincount(which).tolist(),[512,512])
            np.testing.assert_array_equal(batch['fields'][:,0],rows+10000*which.astype(np.int64))
            np.testing.assert_array_equal(batch['optimal'],np.where(which,8,1))
            np.testing.assert_array_equal(batch['source_rows'],2*rows+which)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            paired_batch(views,np.array([0,0]),vrng,'direct')
        with self.assertRaisesRegex(ValueError,'both views'):
            paired_batch(views,np.arange(4),vrng,'successors')

    def test_pair_binding_rejects_permutation_source_encoder_and_repeated_row(self):
        original,first=load_policy_cache('data/structured-field-cache-smoke8/train','train')
        additional,second=load_policy_cache('data/structured-field-additional-state-smoke4/train','train')
        validate_pair(original,additional,first,second,'data/structured-field-cache-smoke8/train')
        for kind in ('order','source','encoder','row','binding'):
            data=dict(additional);meta=copy.deepcopy(second)
            if kind=='order':data['seeds']=data['seeds'][::-1]
            if kind=='source':meta['source_sha256']='bad'
            if kind=='encoder':meta['field_encoder']={}
            if kind=='row':data['source_rows']=original['source_rows']
            if kind=='binding':meta['paired_source']['manifest_sha256']='bad'
            with self.assertRaises(ValueError):validate_pair(original,data,first,meta,'data/structured-field-cache-smoke8/train')

    def test_full_batch_loss_and_gradients_equal_mean_of_same_paired_views(self):
        rng=np.random.default_rng(12);views=[]
        for view in (0,1):
            views.append({'seeds':np.arange(4),'difficulties':np.ones(4,np.int64),
                'source_rows':np.arange(4)*2+view,'fields':rng.normal(size=(4,148,96)).astype(np.float32),
                'optimal':np.array([1,2,4,8] if view==0 else [3,12,5,10]),
                'next_fields':rng.normal(size=(4,4,148,96)).astype(np.float32),
                'imagined_fields':rng.normal(size=(4,4,148,96)).astype(np.float32)})
        batch,which=paired_batch(views,np.arange(4),np.random.default_rng(43),'successors')
        head=new_head({'mode':'successors'},42,'cpu');other=copy.deepcopy(head)
        # Nonneutral scorer exercises attention gradients, not only the final bias.
        with torch.no_grad():
            for p in head.parameters():p.add_(.001*torch.randn_like(p))
        other.load_state_dict(head.state_dict())
        whole,_=loss_for_rows(head,batch,np.arange(4),'cpu');whole.backward()
        separate=sum(loss_for_rows(other,views[v],np.flatnonzero(which==v),'cpu')[0]/2 for v in (0,1));separate.backward()
        torch.testing.assert_close(whole,separate,atol=1e-6,rtol=1e-6)
        for x,y in zip(head.parameters(),other.parameters()):torch.testing.assert_close(x.grad,y.grad,atol=2e-6,rtol=2e-5)


if __name__=='__main__':unittest.main()
