"""Closing-only indices against real generated winning and unsafe engine branches."""
import copy
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from pebby.agent import world_closing_sequences as closing, world_data as wd
from pebby.agent.world_train import as_tensors
from pebby.agent.on_policy_provenance import file_digest
from tests import test_world_sequences as fixtures
from tests.test_policy_history import corridor,RecordingPolicy
from tools.collect_onpolicy_world import collect_level


class PublicHistoryDeadendPolicy(RecordingPolicy):
    def forward(self,frames,history_valid=None,previous_actions=None):
        # Four rights, then left, based solely on visible history length.
        logits=torch.zeros((len(frames),4))
        logits[:,3]=4
        for i,n in enumerate(history_valid.sum(-1)):
            if n>=5:logits[i,3]=0;logits[i,2]=4
        return logits


class ClosingSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        self.fixture=fixtures.WorldSequenceTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root;self.data=self.fixture.data;self.source=self.fixture.source

    def attestation(self,last_action=3):
        rows=[];cursor=0
        for level in self.data['meta']['levels']:
            last=cursor+level['on_policy_samples']-1
            rows.append({'seed':level['seed'],'row':last,'action':last_action,'behavior_index':0,
                         'public_history_sha256':closing.public_history_digest(self.data,last)})
            cursor+=level['samples']
        proof={'format':closing.ATTESTATION,'source':'generated_only','split':'train',
               'inference':'greedy_argmax_batch1','batch_size':1,'official_inputs':False,'oracle_calls':0,'engine_calls':0,
               'source_path':str(self.source),'source_sha256':file_digest(self.source),
               'behavior_checkpoints':[self.data['meta']['behavior_checkpoint']],'source_ranges':[], 'records':rows}
        path=self.root/'attestation.json';path.write_text(json.dumps(proof));return path,file_digest(path)

    def index(self,action=3):
        path,sha=self.attestation(action)
        index=closing.build_sidecar(self.data,self.source,self.root/'closing.npz',attestation=path,attestation_sha256=sha)
        return index,path,sha

    def test_real_winning_closings_use_actual_branch_targets_and_preserve_ordinary(self):
        index,path,sha=self.index()
        loaded=closing.load_sidecar(self.root/'closing.npz',self.source,self.data,attestation=path,attestation_sha256=sha)
        offset = self.data['meta']['levels'][0]['samples']
        np.testing.assert_array_equal(loaded.anchor_row,[1,offset + 1])
        np.testing.assert_array_equal(loaded.branch_rows,np.array([0, offset])[:, None] + np.arange(1, 5))
        tensors=as_tensors(self.data)
        output=closing.closing_mixed_batch(tensors,tensors,(torch.tensor([0]),torch.tensor([1]),torch.tensor([1,0])),index)
        self.assertEqual(output['rollout_actions'].tolist(),[[3,3,3,3],[-1,-1,-1,-1]])
        self.assertEqual(output['distances'][0].tolist(),[3,2,1,0])
        self.assertEqual(output['won'][0].tolist(),[False,False,False,True])
        self.assertEqual(output['next_optimal'][0,-1],0)
        for key in closing.TARGETS:
            torch.testing.assert_close(output[key][0],tensors[key][torch.arange(1,5),3])
            torch.testing.assert_close(output[key][1],tensors[key][0])
        torch.testing.assert_close(output['frames'][0],tensors['frames'][1])
        torch.testing.assert_close(output['optimal'][0],tensors['optimal'][1])

    def test_real_alive_unreachable_closing_has_negative_distance_zero_policy_mask(self):
        spec=corridor(budget=5);spec.update(seed=8,difficulty=1);spec['goals'][0]['cell']=(8,3)
        rows,proof,count=collect_level(spec,PublicHistoryDeadendPolicy(history=8),max_actions=5)
        self.assertEqual(proof['stop'],'action_limit');self.assertEqual(count,5)
        meta=copy.deepcopy(self.data['meta']);meta['levels']=[proof];meta['on_policy_rows']=list(range(count))
        self.data={key:np.stack([r[key] for r in rows]) for key in rows[0]};self.data['meta']=meta
        wd.save(self.source,self.data);index,_,_=self.index(action=2)
        targets=closing.sequence_targets(as_tensors(self.data),index,torch.tensor([1]))
        self.assertEqual(targets['rollout_actions'].tolist(),[[3,3,3,2]])
        self.assertEqual(targets['distances'][0,-1],-1)
        self.assertEqual(targets['next_optimal'][0,-1],0)
        self.assertFalse(targets['terminal'].any());self.assertFalse(targets['lost_life'].any())
        np.testing.assert_array_equal(targets['next_frames'][0,-1],self.data['next_frames'][4,2])

    def test_external_attestation_pin_and_branch_tampering_fail(self):
        index,path,sha=self.index()
        proof=json.loads(path.read_text());proof['records'][0]['action']=2;path.write_text(json.dumps(proof))
        with self.assertRaisesRegex(ValueError,'checksum'):
            closing.load_sidecar(self.root/'closing.npz',self.source,self.data,attestation=path,attestation_sha256=sha)
        path,sha=self.attestation()
        with np.load(self.root/'closing.npz') as z:content={k:z[k] for k in z.files}
        content['branch_actions'][0,-1]=2;np.savez_compressed(self.root/'closing.npz',**content)
        with self.assertRaisesRegex(ValueError,'fully checked'):
            closing.load_sidecar(self.root/'closing.npz',self.source,self.data,attestation=path,attestation_sha256=sha)

    def test_changed_public_history_and_validation_source_rejected(self):
        _,path,sha=self.index();changed=copy.deepcopy(self.data);changed['frames'][4,0,0,0]^=1
        with self.assertRaisesRegex(ValueError,'history bytes'):
            closing.load_sidecar(self.root/'closing.npz',self.source,changed,attestation=path,attestation_sha256=sha)
        changed=copy.deepcopy(self.data);changed['meta']['split']='validation'
        with self.assertRaisesRegex(ValueError,'split'):
            closing.build_index_arrays(changed,self.source,path,sha)

    def test_sampler_reserves_distinct_levels_without_mutating_metadata(self):
        index,_,_=self.index();before=copy.deepcopy(self.data['meta'])
        sampler=closing.ClosingSampler(self.data,self.data,index,fraction=.5,start=[1,0,0,0,0],end=[1,0,0,0,0])
        for progress in (0,.5,1):
            bi,pi,_=sampler.indices(2,progress,torch.Generator().manual_seed(4))
            self.assertEqual((len(bi),len(pi)),(1,1));self.assertEqual(len(set(sampler.last_level_seeds)),2)
            self.assertIn(int(pi[0]),index.anchor_row)
        self.assertEqual(self.data['meta'],before)

if __name__=='__main__':unittest.main()
