"""Actual generated closing win/deadend cache targets, without engine labels as inputs."""
import copy
import unittest
import numpy as np
import torch
from pebby.agent import world_data as wd
from pebby.agent.world_train import as_tensors
from tests.test_world_closing_sequences import ClosingSequenceTests, PublicHistoryDeadendPolicy
from tests.test_policy_history import corridor
from tests.test_structured_field_cache import RecordingAssembler
from tools.collect_onpolicy_world import collect_level
from tools.build_structured_training_sequences import encode_sequences, sequence_batch, write_cache


class StructuredTrainingSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        self.fixture = ClosingSequenceTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_real_winning_fields_and_exact_branch_labels_persist(self):
        f = self.fixture; index, _, _ = f.index(); tensors = as_tensors(f.data)
        assembler = RecordingAssembler()
        fields, targets, labels, _ = encode_sequences(assembler, f.data, tensors, index, index.anchor_row)
        np.testing.assert_array_equal(labels['won'], [[False,False,False,True]]*2)
        np.testing.assert_array_equal(labels['distances'], [[3,2,1,0]]*2)
        self.assertEqual(labels['next_optimal'][:,-1].tolist(), [0,0])
        for key in labels:
            if key not in ('branch_rows','actions','player_cell','triple','steps','lives','optimal'):
                np.testing.assert_array_equal(labels[key], f.data[key][index.branch_rows,index.branch_actions])
        out = f.root/'cache'
        m = write_cache(f.data,index,out,RecordingAssembler(),{}, {},count=2)
        self.assertEqual(m['event_coverage']['won'],2)
        self.assertEqual(m['current_zero_optimal'],0)
        self.assertEqual(m['next_zero_optimal_by_horizon'],[0,0,0,2])
        np.testing.assert_array_equal(np.load(out/'next_steps.npy'),f.data['next_steps'][index.branch_rows,index.branch_actions])
        self.assertEqual(np.load(out/'next_steps.npy').dtype,f.data['next_steps'].dtype)
        self.assertEqual(fields.shape,(2,148,96));self.assertEqual(targets.shape,(2,4,148,96))
        with self.assertRaises(FileExistsError):write_cache(f.data,index,out,RecordingAssembler(),{}, {},count=2)

    def test_real_alive_deadend_preserves_negative_distance_and_zero_policy_label(self):
        f = self.fixture
        spec=corridor(budget=5);spec.update(seed=8,difficulty=1);spec['goals'][0]['cell']=(8,3)
        rows,proof,count=collect_level(spec,PublicHistoryDeadendPolicy(history=8),max_actions=5)
        meta=copy.deepcopy(f.data['meta']);meta['levels']=[proof];meta['on_policy_rows']=list(range(count))
        f.data={key:np.stack([r[key] for r in rows]) for key in rows[0]};f.data['meta']=meta
        wd.save(f.source,f.data);index,_,_=f.index(action=2)
        _,_,labels,_=encode_sequences(RecordingAssembler(),f.data,as_tensors(f.data),index,index.anchor_row)
        self.assertEqual(labels['actions'].tolist(),[[3,3,3,2]])
        self.assertEqual(labels['distances'][0,-1],-1)
        self.assertEqual(labels['next_optimal'][0,-1],0)
        self.assertFalse(labels['terminal'].any());self.assertFalse(labels['lost_life'].any())

    def test_source_continuity_and_interior_reset_rejected_and_labels_not_encoder_inputs(self):
        f=self.fixture; index,_,_=f.index(); changed=copy.deepcopy(f.data)
        changed['frames'][index.branch_rows[0,1],0,0,0]^=1
        with self.assertRaisesRegex(ValueError,'interior chronological'):
            sequence_batch(changed,as_tensors(changed),index,index.anchor_row)
        changed=copy.deepcopy(f.data);changed['lost_life'][index.branch_rows[0,0],3]=True
        with self.assertRaisesRegex(ValueError,'first three'):
            sequence_batch(changed,as_tensors(changed),index,index.anchor_row)
        changed=copy.deepcopy(f.data);changed['next_steps'][index.branch_rows,index.branch_actions]+=1
        a,b=RecordingAssembler(),RecordingAssembler()
        first=encode_sequences(a,f.data,as_tensors(f.data),index,index.anchor_row)
        second=encode_sequences(b,changed,as_tensors(changed),index,index.anchor_row)
        for i in (0,1):np.testing.assert_array_equal(first[i],second[i])
        self.assertFalse(np.array_equal(first[2]['next_steps'],second[2]['next_steps']))
        self.assertEqual(len(a.calls[0]),3)

    def test_encoder_chunking_and_separate_initial_candidate_indices(self):
        f=self.fixture;index,_,_=f.index();tensors=as_tensors(f.data)
        first=encode_sequences(RecordingAssembler(),f.data,tensors,index,index.anchor_row)
        recorder=RecordingAssembler()
        chunked=encode_sequences(recorder,f.data,tensors,index,index.anchor_row,1)
        for i in (0,1):np.testing.assert_array_equal(first[i],chunked[i])
        self.assertTrue(all(len(call[0])==1 for call in recorder.calls))
        out=f.root/'mapped-cache'
        m=write_cache(f.data,index,out,RecordingAssembler(),{}, {},count=2,
            initial_data=f.data,initial_provenance={'source_path':'separate initial source'})
        seeds=np.load(out/'seeds.npy');initial=np.load(out/'source_initial_rows.npy')
        np.testing.assert_array_equal(f.data['seeds'][initial],seeds)
        self.assertTrue((f.data['history_valid'][initial].sum(1)==1).all())
        self.assertFalse(np.array_equal(initial,np.load(out/'source_rows.npy')))
        self.assertEqual(m['encoding_precision']['device'],'cpu')


if __name__=='__main__':unittest.main()
