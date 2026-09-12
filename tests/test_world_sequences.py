import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pebby.agent import world_data as wd
from pebby.agent.world_train import as_tensors
from pebby.agent.world_sequences import (FourStepIndex,FourStepSampler,build_index_arrays,build_sidecar,
                                         load_sidecar,four_step_mixed_batch,exact_current_distances)
from pebby.agent.on_policy_provenance import file_digest
from tests.test_policy_history import corridor,RecordingPolicy
from tools.collect_onpolicy_world import collect_level


class WorldSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        checkpoint=self.root/'behavior.pt';checkpoint.write_bytes(b'test public recording policy')
        rows=[];levels=[];marked=[]
        for seed in (8,9):
            spec=corridor();spec.update(seed=seed,difficulty=1);spec['goals'][0]['cell']=(8,3)
            collected,proof,count=collect_level(spec,RecordingPolicy(history=8))
            marked.extend(range(len(rows),len(rows)+count));rows.extend(collected);levels.append(proof)
        self.data={key:np.stack([r[key] for r in rows]) for key in rows[0]}
        self.data['meta']={'format':wd.FORMAT,'source':'generated_only','oracle_search':'complete_only','history':8,
                           'levels':levels,'on_policy_rows':marked,'collection_policy':'model_greedy',
                           'behavior_checkpoint':{'path':str(checkpoint),'sha256':file_digest(checkpoint),'parameters':1,'config':{'history':8}},
                           'on_policy_provenance':{'official_inputs_used':False,'oracle_actions_in_policy_rollout':0}}
        self.source=self.root/'source.npz';wd.save(self.source,self.data)

    def index(self):
        return build_sidecar(self.data,self.source,self.root/'index.npz')

    def test_real_corridor_chronology_labels_and_no_anchor_leakage(self):
        offset = self.data['meta']['levels'][0]['samples']
        index=self.index();np.testing.assert_array_equal(index.anchor_row,[0,offset])
        np.testing.assert_array_equal(index.future_rows, np.array([0, offset])[:, None] + np.arange(1, 5))
        np.testing.assert_array_equal(exact_current_distances(self.data,index.future_rows),[[4,3,2,1],[4,3,2,1]])
        tensors=as_tensors(self.data);before={key:value.clone() for key,value in tensors.items()}
        indices=(torch.tensor([offset + 4]),torch.tensor([0]),torch.tensor([1,0]))
        batch=four_step_mixed_batch(tensors,tensors,indices,index)
        self.assertEqual(batch['rollout_mask'].tolist(),[True,False])
        self.assertEqual(batch['rollout_actions'].tolist(),[[3,3,3,3],[-1,-1,-1,-1]])
        torch.testing.assert_close(batch['frames'][0],tensors['frames'][0])
        torch.testing.assert_close(batch['optimal'][0],tensors['optimal'][0])
        torch.testing.assert_close(batch['next_frames'][0],tensors['frames'][torch.tensor([1,2,3,4]),-1])
        self.assertEqual(batch['distances'][0].tolist(),[4,3,2,1])
        self.assertEqual(batch['next_steps'][0].tolist(),self.data['current_steps'][1:5].tolist())
        for name in ('terminal','won','lost_life'):self.assertFalse(batch[name][0].any())
        self.assertTrue(batch['won'][1].any()) # ordinary final row keeps real winning branch
        for key in tensors:
            torch.testing.assert_close(tensors[key],before[key])
            torch.testing.assert_close(batch[key][1],tensors[key][offset + 4])

    def test_failure_auxiliary_enters_actual_training_batch_as_one_step_beside_sequence(self):
        from pebby.agent.world_train import curriculum_batches
        from pebby.agent.world_training_objectives import world_losses
        from tests.test_world_model import make_model
        index = self.index()
        marked = set(self.data['meta']['on_policy_rows'])
        self.data['meta']['auxiliary_rows'] = [i for i in range(len(self.data['seeds'])) if i not in marked]
        sampler = FourStepSampler(self.data, self.data, index, fraction=.5, auxiliary_fraction=1.,
                                   start=[1,0,0,0,0], end=[1,0,0,0,0])
        tensors = as_tensors(self.data)
        batch = next(curriculum_batches(tensors, sampler, 2, torch.Generator().manual_seed(4),
                                        1, 0, 1, tensors))
        mask = batch['rollout_mask']
        self.assertEqual(mask.sum(), 1)
        self.assertEqual(sampler.last_on_policy_count, 1)
        self.assertEqual(sampler.last_auxiliary_count, 1)
        self.assertTrue(batch['lost_life'][~mask].any())
        self.assertTrue((batch['optimal'][~mask] == 0).all())
        self.assertTrue((batch['rollout_actions'][~mask] == -1).all())
        result = world_losses(make_model(history=8), batch)
        self.assertTrue(torch.isfinite(result['total']))
        self.assertEqual(result['diagnostics']['policy_valid_fraction'], .5)
        result['total'].backward()

    def test_unreachable_policy_visits_are_kept_in_source_but_not_distance_sequences(self):
        spec = corridor(budget=5)
        spec.update(seed=8, difficulty=1, start=(8, 3))
        rows, proof, count = collect_level(spec, RecordingPolicy(history=8))
        data = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
        data['meta'] = {**self.data['meta'], 'levels': [proof], 'on_policy_rows': list(range(count))}
        self.assertTrue((data['optimal'][:count] == 0).any())
        anchors, future, checks = build_index_arrays(data)
        self.assertGreater(checks['unreachable_future'], 0)
        self.assertTrue((data['optimal'][future] != 0).all())

    def test_sampler_excludes_reserved_level_and_keeps_metadata_unchanged(self):
        index=self.index();original=copy.deepcopy(self.data['meta'])
        sampler=FourStepSampler(self.data,self.data,index,start=[1,0,0,0,0],end=[1,0,0,0,0])
        bi,pi,order=sampler.indices(2,.5,torch.Generator().manual_seed(4))
        self.assertEqual((len(bi),len(pi)),(1,1));self.assertEqual(len(set(sampler.last_level_seeds)),2)
        self.assertNotEqual(int(self.data['seeds'][bi[0]]),int(self.data['seeds'][pi[0]]))
        self.assertEqual(self.data['meta'],original)

    def test_source_binding_and_tampered_future_indices_rejected(self):
        index=self.index();path=self.root/'index.npz';loaded=load_sidecar(path,self.source,self.data)
        np.testing.assert_array_equal(loaded.future_rows,index.future_rows)
        with np.load(path) as archive:values={key:archive[key] for key in archive.files}
        values['future_rows'][0,-1]=5
        np.savez_compressed(path,**values)
        with self.assertRaisesRegex(ValueError,'fully checked'):load_sidecar(path,self.source,self.data)
        self.source.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'binding'):load_sidecar(path,self.source,self.data)

    def test_reset_expert_boundary_and_bad_image_excluded_or_rejected(self):
        changed=copy.deepcopy(self.data)
        for j in range(2,5):
            f,v,a=wd.history_arrays([self.data['frames'][t,-1] for t in range(2,j+1)],[-1]+[3]*(j-2),8)
            changed['frames'][j],changed['history_valid'][j],changed['previous_actions'][j]=f,v,a
        anchors,_,checks=build_index_arrays(changed)
        self.assertNotIn(0,anchors);self.assertEqual(checks['reset_or_missing_action'],1)
        changed=copy.deepcopy(self.data);changed['meta']['on_policy_rows'].remove(4);changed['meta']['levels'][0]['on_policy_samples']=4
        anchors,_,_=build_index_arrays(changed);self.assertNotIn(0,anchors)
        changed=copy.deepcopy(self.data);changed['next_frames'][0,3,0,0]^=1
        with self.assertRaisesRegex(ValueError,'continuity'):build_index_arrays(changed)

    def test_validation_index_separate_and_never_enters_training_sampler(self):
        specs=[]
        for seed in (1000001,1000002):
            spec=corridor();spec.update(seed=seed,difficulty=1);spec['goals'][0]['cell']=(8,3);specs.append(spec)
        data=wd.build(specs,workers=1,history=8,samples=16,coverage='mixed',epsilon=0)
        source=self.root/'validation.npz';wd.save(source,data)
        index=build_sidecar(data,source,self.root/'validation-index.npz','explore_validation')
        self.assertGreater(len(index.anchor_row),0)
        with self.assertRaisesRegex(ValueError,'validation sequence'):FourStepSampler(self.data,data,index)
        with self.assertRaisesRegex(ValueError,'split/mode'):load_sidecar(self.root/'validation-index.npz',source,data)

if __name__=='__main__':unittest.main()
