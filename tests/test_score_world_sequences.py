import copy
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from tools import score_world_sequences as scorer
from pebby.agent.world_sequences import FourStepIndex
from pebby.agent.world_train import as_tensors
from tests import test_world_sequences as sequence_fixture
from tests.test_world_rollout_draft import draft
from tests.test_world_model import make_model
from tests.test_world_glyph import wake

class SequenceScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_split_and_balanced_distinct_selection(self):
        seeds=np.repeat(np.arange(1_000_000,1_000_020),2)
        data={'seeds':seeds,'meta':{'levels':[{'seed':int(s),'difficulty':i%5+1} for i,s in enumerate(np.unique(seeds))]}}
        index=FourStepIndex(np.arange(40),np.zeros((40,4),np.int64),{'mode':'explore_validation','split':'validation'})
        rows=scorer.select_anchors(data,index,12)
        selected=seeds[rows];self.assertEqual(len(set(selected)),12)
        self.assertEqual([sum((selected-1_000_000)%5==i) for i in range(5)],[3,3,2,2,2])
        np.testing.assert_array_equal(rows,scorer.select_anchors(data,index,12))
        for meta in ({'mode':'on_policy_train','split':'train'},{'mode':'explore_validation','split':'train'}):
            with self.assertRaisesRegex(ValueError,'validation'):scorer.select_anchors(data,replace(index,meta=meta),12)
        data['seeds']-=1_000_000
        with self.assertRaisesRegex(ValueError,'training seeds'):scorer.select_anchors(data,index,12)

    def test_chunk_invariant_horizons_full_population_sig_and_policy(self):
        fixture=sequence_fixture.WorldSequenceTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        raw=fixture.index();index=replace(raw,meta={**raw.meta,'mode':'explore_validation','split':'validation'})
        tensors=as_tensors(fixture.data);torch.manual_seed(731)
        model=wake(make_model(history=8,grounding=True,glyph_recall=True,query_readout=True)).eval()
        with patch.object(scorer,'world_losses',draft.world_losses):
            one=scorer.score(model,tensors,index,index.anchor_row,chunk=1)
            two=scorer.score(model,tensors,index,index.anchor_row,chunk=2)
        for key in one:
            np.testing.assert_allclose(one[key],two[key],atol=2e-5,rtol=2e-5,err_msg=key)
        self.assertEqual(one['sigreg_population'],[5,2,model.cfg.latent])
        with torch.no_grad():
            rows=torch.from_numpy(index.anchor_row)
            enc=model.encode(tensors['frames'][rows],tensors['history_valid'][rows],tensors['previous_actions'][rows])
            logits=model.logits_from(enc)[0];bits=(tensors['optimal'][rows,None].long()&(1<<torch.arange(4)))!=0
            accuracy=bits.gather(1,logits.argmax(-1)[:,None]).float().mean()
            ce=-(bits.float()/bits.sum(-1,keepdim=True)*logits.log_softmax(-1)).sum(-1).mean()
        self.assertAlmostEqual(one['policy_set_accuracy'],float(accuracy),places=6)
        self.assertAlmostEqual(one['policy_cross_entropy'],float(ce),places=5)
        with self.assertRaisesRegex(ValueError,'validation'):scorer.score(model,tensors,raw,raw.anchor_row)
        with self.assertRaisesRegex(ValueError,'distinct'):scorer.score(model,tensors,index,np.repeat(index.anchor_row[0],2))

if __name__=='__main__':unittest.main()
