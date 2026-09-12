"""Generated real-engine branch alignment and separate field-cache contracts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from pebby.agent import world_data as wd
from pebby.ls20 import generate,names
from pebby.ls20.env import Ls20Scenario
from tests.test_policy_history import corridor
from tools.build_structured_field_cache import actual_histories,encode_fields,select_rows,write_split,LABELS,digest


class RecordingAssembler(torch.nn.Module):
    def __init__(self):super().__init__();self.calls=[]
    def forward(self,frames,valid,previous):
        self.calls.append(tuple(x.clone() for x in (frames,valid,previous)))
        out=frames.new_zeros((len(frames),148,96),dtype=torch.float32)
        out[:,:,0]=frames[:,-1].float().mean((1,2))[:,None]
        out[:,:,1]=valid.sum(1)[:,None];out[:,:,2]=previous.float().sum(1)[:,None]
        return out


class StructuredFieldCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def observed(self,budget=20,steps=2):
        spec=corridor(budget=budget);spec.update(seed=3,difficulty=1)
        env=Ls20Scenario(generate.build_level(spec),1)
        frames=[env.render()];actions=[-1]
        for _ in range(steps):
            result=env.perform(names.ACTION_IDS[3]);frames.append(result.frame);actions.append(3)
        f,v,p=wd.history_arrays(frames,actions,8)
        branches=[];results=[]
        for action in names.ACTION_IDS:
            branch=wd.clone_env(env);results.append(branch.perform(action));branches.append(branch)
        return env,frames,actions,{'frames':f[None],'history_valid':v[None],'previous_actions':p[None],
            'next_frames':np.asarray([r.frame for r in results])[None],
            'lost_life':np.asarray([b.lives()<env.lives() for b in branches])[None]},branches,results

    def test_actual_ordinary_and_winning_histories_match_real_branches(self):
        _,frames,actions,batch,_,results=self.observed()
        self.assertTrue(results[3].won)
        observed=actual_histories(batch)
        for action,result in enumerate(results):
            expected=wd.history_arrays(frames+[result.frame],actions+[action],8)
            for actual,want in zip(observed,expected):
                np.testing.assert_array_equal(actual[0,action].numpy(),want)
        self.assertFalse(batch['lost_life'].any())

    def test_real_life_loss_clears_all_target_history_except_current(self):
        _,_,_,batch,_,_=self.observed(budget=0,steps=0)
        self.assertTrue(batch['lost_life'].all())
        f,v,p=actual_histories(batch)
        for action in range(4):
            expected=wd.history_arrays([batch['next_frames'][0,action]],[-1],8)
            for actual,want in zip((f,v,p),expected):np.testing.assert_array_equal(actual[0,action].numpy(),want)
        self.assertTrue((p==-1).all());self.assertEqual(int(v.sum()),4)

    def test_future_labels_pixels_and_resets_cannot_change_current_assembly(self):
        _,_,_,batch,_,_=self.observed();changed=copy.deepcopy(batch)
        changed['next_frames'][:]=6;changed['lost_life'][:]=True
        changed['next_player_cell']=np.zeros((1,4,2),np.int16)
        a=RecordingAssembler();b=RecordingAssembler()
        fa,na,_=encode_fields(a,batch);fb,nb,_=encode_fields(b,changed)
        np.testing.assert_array_equal(fa,fb);self.assertFalse(np.array_equal(na,nb))
        for x,y in zip(a.calls[0],b.calls[0]):torch.testing.assert_close(x,y,atol=0,rtol=0)
        self.assertEqual(fa.dtype,np.float16)

    def dataset(self):
        env,_,_,batch,branches,results=self.observed(budget=2)
        spec=corridor(budget=2);spec.update(seed=3,difficulty=1)
        _,oracle,proof=wd.verified_context(spec,context_index=1)
        self.assertFalse(oracle.truncated);self.assertTrue(proof['context_engine_verified'])
        distance=oracle.distance_for(oracle.state_of(env))
        targets,_,_,_=wd._expand(env,oracle,distance,3,2)
        data={**batch,**{key:np.asarray(value)[None] for key,value in targets.items()},
              'seeds':np.array([3],np.int32),
              'meta':{'source':'generated_only','split':'train','oracle_search':'complete_only',
                      'levels':[{'seed':3,'difficulty':1}]}}
        return data

    def test_cache_preserves_exact_branch_labels_arrays_and_provenance(self):
        data=self.dataset()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.npz'
            np.savez(source,**{k:v for k,v in data.items() if k!='meta'})
            out=root/'train';manifest=write_split(data,source,out,RecordingAssembler(),1,42,'train',{'test':'public recording assembler'})
            for target,original in LABELS.items():
                actual=np.load(out/f'{target}.npy',allow_pickle=False)
                np.testing.assert_array_equal(actual,data[original]);self.assertEqual(actual.dtype,data[original].dtype)
            self.assertEqual(np.load(out/'fields.npy',mmap_mode='r').shape,(1,148,96))
            self.assertEqual(np.load(out/'next_fields.npy',mmap_mode='r').shape,(1,4,148,96))
            self.assertEqual(manifest['event_coverage']['won'],1)
            self.assertEqual(manifest['event_coverage']['lost_life'],3)
            self.assertEqual(int(np.load(out/'next_steps.npy')[0,3]),-1)
            self.assertEqual(manifest['source_sha256'],digest(source))
            for name,item in manifest['arrays'].items():self.assertEqual(item['sha256'],digest(out/f'{name}.npy'))
            with self.assertRaises(FileExistsError):write_split(data,source,out,RecordingAssembler(),1,42,'train',{})

    def test_explicit_life_loss_enrichment_selects_event_row_and_respects_quota(self):
        data=self.dataset()
        for key,value in list(data.items()):
            if isinstance(value,np.ndarray):data[key]=np.repeat(value,2,axis=0)
        data['lost_life'][0]=False
        rows,_=select_rows(data,1,42,'train',include_life_loss_levels=True)
        self.assertEqual(rows.tolist(),[1])
        data['seeds'][0]=8
        data['meta']['levels'].append({'seed':8,'difficulty':1})
        data['lost_life'][0]=True
        with self.assertRaisesRegex(ValueError,'exceed difficulty1 quota'):
            select_rows(data,1,42,'train',include_life_loss_levels=True)

    def test_missing_successor_labels_and_split_leakage_fail_closed(self):
        data=self.dataset();a=select_rows(data,1,42,'train');b=select_rows(data,1,42,'train')
        np.testing.assert_array_equal(a[0],b[0])
        for broken in ('next_player_cell','next_steps'):
            changed=copy.deepcopy(data);del changed[broken]
            with self.assertRaisesRegex(ValueError,'exact source label missing'):select_rows(changed,1,42,'train')
        with self.assertRaisesRegex(ValueError,'requested split'):select_rows(data,1,42,'validation')
        data['seeds'][0]=1000003
        with self.assertRaisesRegex(ValueError,'namespace'):select_rows(data,1,42,'train')

    def test_grouped_selection_matches_prechange_fixedseed_interleaved_rows(self):
        seeds=np.tile(np.arange(10,30,dtype=np.int32),3)
        data={name:np.zeros(len(seeds),dtype=np.int16) for name in LABELS.values()}
        data.update(seeds=seeds,lost_life=np.zeros((len(seeds),4),bool),
            meta={'source':'generated_only','oracle_search':'complete_only','split':'train',
                  'levels':[{'seed':int(s),'difficulty':int(s%5+1)} for s in np.unique(seeds)]})
        data['lost_life'][[20,47],0]=True
        # Captured from the preserved original implementation before optimization.
        expected={False:[35,20,50,21,31,26,7,12,37,53,8,43,49,14,24],
                  True:[20,25,15,56,46,21,47,22,57,28,33,38,9,19,24]}
        for enriched in (False,True):
            rows,difficulty=select_rows(data,15,42,'train',enriched)
            self.assertEqual(rows.tolist(),expected[enriched])
            self.assertEqual(difficulty.tolist(),[1]*3+[2]*3+[3]*3+[4]*3+[5]*3)

    def test_bounded_encoder_calls_keep_actual_reset_and_win_outputs_identical(self):
        _,_,_,batch,_,_=self.observed(budget=2)
        original=RecordingAssembler();chunked=RecordingAssembler()
        first=encode_fields(original,batch)
        second=encode_fields(chunked,batch,max_encoder_batch=1)
        for i in (0,1):np.testing.assert_array_equal(first[i],second[i])
        self.assertEqual(first[2],second[2])
        self.assertTrue(all(len(call[0])==1 for call in chunked.calls))
        self.assertEqual(len(chunked.calls),5)
        for invalid in (0,1025,-1):
            with self.assertRaisesRegex(ValueError,'max encoder batch'):
                encode_fields(RecordingAssembler(),batch,max_encoder_batch=invalid)


if __name__=='__main__':unittest.main()
