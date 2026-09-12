import unittest,tempfile,json
from pathlib import Path
import numpy as np
import torch
from tools.train_structured_refusal_correction import counts,sample,actions_for,backward_groups,make
from tools import train_structured_factored_sequences as h4
from tools import train_structured_transition as h1
from tools.train_structured_glyph_ablation import losses as h1_losses

class RefusalTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_ratios_and_crossbank_distinct_paired_sampling(self):
        data={'seeds':np.arange(3000),'difficulties':np.arange(3000)%5+1};live={'seeds':np.arange(500,5500),'difficulties':np.arange(5000)%5+1};close={'seeds':np.arange(1000,3000),'difficulties':np.arange(2000)%5+1};candidate={'cache_rows':np.arange(1100),'actions':np.arange(1100)%4}
        self.assertEqual(counts(1024),(128,672,224));self.assertEqual(counts(8,True),(1,5,2))
        with self.assertRaises(ValueError):counts(8)
        with self.assertRaises(ValueError):counts(48)
        for progress in [0.,.5,1.]:
            a=sample(data,live,close,candidate,1024,progress,np.random.default_rng(42));b=sample(data,live,close,candidate,1024,progress,np.random.default_rng(42))
            for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
            self.assertEqual(len(np.unique(np.r_[data['seeds'][a[0]],live['seeds'][a[1]],close['seeds'][a[2]]])),1024)
            act=actions_for(a[0],candidate,np.random.default_rng(43),True);np.testing.assert_array_equal(act,a[0]%4)
    def test_multiple_refusal_actions_share_one_eligible_level(self):
        candidates={'cache_rows':np.array([0,0,1]),'actions':np.array([1,3,2])}
        chosen=actions_for(np.zeros(100,dtype=np.int64),candidates,np.random.default_rng(43),True)
        self.assertEqual(set(chosen.tolist()),{1,3})
        chosen=actions_for(np.ones(10,dtype=np.int64),candidates,np.random.default_rng(43),True)
        np.testing.assert_array_equal(chosen,np.full(10,2))

    def test_index_wrong_split_or_manifest_rejected_before_data_access(self):
        from tools.train_structured_refusal_correction import load_candidates
        with tempfile.TemporaryDirectory() as directory:
            manifest=Path(directory)/'manifest.json';manifest.write_text('{}')
            index=Path(directory)/'index.npz'
            for meta in [{'status':'complete','source':'generated_only','split':'validation'}, {'status':'complete','source':'generated_only','split':'train','h1_manifest_sha256':'0'*64}]:
                np.savez(index,meta=np.array(json.dumps(meta)))
                with self.assertRaisesRegex(ValueError,'split/manifest'):load_candidates(index,{},manifest)

    @unittest.skipUnless(Path('checkpoints/ls20-factored-local-h4-400.pt').exists(),'real generated cache fixture absent')
    def test_real_weighted_subgroup_gradients_match_single_total(self):
        saved=torch.load('checkpoints/ls20-factored-local-h4-400.pt',weights_only=True,map_location='cpu');first=make(saved,'cpu');second=make(saved,'cpu')
        def arrays(root):return {p.stem:np.load(p,mmap_mode='r') for p in Path(root).glob('*.npy')}
        one=arrays('data/structured-field-16384/train');four=arrays('data/structured-field-h4-exploratory-train-16384')
        b1=h1.branch_batch(one,np.array([0]),np.array([0]),'cpu')
        from tools.train_structured_sequences import sequence_batch
        b4=sequence_batch(four,np.array([0,1]),'cpu');scale=saved['feature_scale'];positive=saved['event_positive_weights']
        result=backward_groups(first,b1,b4,scale,positive,'cpu',True)
        r=h4.sequence_objective(h4.ObjectiveView(second),*b4,scale,pos_weight=positive,checkpoint_steps=True);aux=h4.h4_glyph_auxiliary_losses(r,b4[3],direct_weight=1,predicted_readout_weight=1);l,_,_=h1_losses(second,b1,scale,positive,'local-balanced',direct_glyph_weight=1)
        (.875*(r['total']+aux['total'])+.125*l).backward()
        self.assertGreater(result['weighted_h4_gradient_l2'],0);self.assertGreater(result['weighted_h1_gradient_l2'],0)
        for a,b in zip(first.parameters(),second.parameters()):
            if a.grad is None:self.assertIsNone(b.grad)
            else:torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-5)
if __name__=='__main__':unittest.main()
