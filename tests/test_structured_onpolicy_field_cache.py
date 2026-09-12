"""All-row source alignment, distinct-level indexing and public-only cache flow."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pebby.agent import world_data as wd
from tests.test_policy_history import corridor
from tests.test_structured_field_cache import RecordingAssembler
from tools.build_structured_field_cache import LABELS, actual_histories, digest, encode_fields
from tools.build_structured_onpolicy_field_cache import FORMAT, build, load_source, main, row_metadata


class Dynamics(torch.nn.Module):
    def predict(self, fields, actions):
        return fields + actions[:,None,None].float()/16


class PublicPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=RecordingAssembler()
        self.encoder.metadata=lambda:{'test':'public-only recorder'}
        self.dynamics=Dynamics()
        self.sources={}


class OnpolicyFieldCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        spec=corridor(budget=2);spec.update(seed=3,difficulty=1)
        rows,proof=wd.collect_level(spec,history=8,samples=4,epsilon=0,context_index=3)
        cls.fixture={key:np.stack([row[key] for row in rows]) for key in rows[0]}
        cls.fixture['context_index']=np.full(len(rows),3,dtype=np.int8)
        cls.fixture['meta']={'format':wd.FORMAT,'source':'generated_only','oracle_search':'complete_only',
            'history':8,'alternatives_per_state':4,'split':'train','levels':[proof],
            'on_policy_rows':list(range(len(rows)-1))}
        assert len(rows)>1

    def save_source(self,root):
        data=copy.deepcopy(self.fixture)
        checkpoint=root/'actor.pt';checkpoint.write_bytes(b'fixture actor source')
        data['meta']['behavior_checkpoint']={'path':str(checkpoint),'sha256':digest(checkpoint)}
        source=root/'source.npz';wd.save(source,data)
        bound=root/'collector.py';bound.write_text('fixture collector')
        report=root/'report.json';report.write_text(json.dumps({'status':'complete','source_unchanged':True,
            'output_sha256':digest(source),'source_hashes':{str(bound):digest(bound)}}))
        return data,source,report,checkpoint

    def test_stable_grouping_preserves_repeated_seeds_and_flags(self):
        data=copy.deepcopy(self.fixture)
        n=len(data['seeds']);index=row_metadata(data)
        self.assertEqual(index['level_seeds'].tolist(),[3])
        self.assertEqual(index['level_offsets'].tolist(),[0,n])
        self.assertEqual(index['level_rows'].tolist(),list(range(n)))
        self.assertEqual(index['source_rows'].tolist(),list(range(n)))
        self.assertEqual(index['on_policy'].tolist(),[True]*(n-1)+[False])
        bad=copy.deepcopy(data);bad['optimal'][0]=0
        with self.assertRaisesRegex(ValueError,'optimal'):row_metadata(bad)
        bad=copy.deepcopy(data);bad['meta']['on_policy_rows']=[0,0]
        with self.assertRaisesRegex(ValueError,'on_policy_rows'):row_metadata(bad)
        bad=copy.deepcopy(data);del bad['next_player_cell']
        with self.assertRaisesRegex(ValueError,'next_player_cell'):row_metadata(bad)
        bad=copy.deepcopy(data);bad['context_index'][0]=4
        with self.assertRaisesRegex(ValueError,'context_index'):row_metadata(bad)
        bad=copy.deepcopy(data);bad['next_lives'][0,0]=0
        with self.assertRaisesRegex(ValueError,'life decrement'):row_metadata(bad)
        bad=copy.deepcopy(data);bad['previous_actions']=bad['previous_actions'][:,:7]
        with self.assertRaisesRegex(ValueError,'bad shape'):row_metadata(bad)

    def test_complete_real_source_and_checkpoint_hash_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,source,report,checkpoint=self.save_source(root)
            data,hashes=load_source(source,report,checkpoint)
            np.testing.assert_array_equal(data['optimal'],self.fixture['optimal'])
            self.assertIn(str(source.resolve()),hashes)
            checkpoint.write_bytes(b'wrong actor')
            with self.assertRaisesRegex(ValueError,'behavior checkpoint'):load_source(source,report,checkpoint)
            value=json.loads(report.read_text());value['status']='failed';report.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'completed'):load_source(source,report,checkpoint)

    def test_interleaved_real_levels_have_stable_group_rows(self):
        spec=corridor(budget=2);spec.update(seed=5,difficulty=2)
        rows,proof=wd.collect_level(spec,history=8,samples=4,epsilon=0,context_index=5)
        second={key:np.stack([row[key] for row in rows]) for key in rows[0]}
        second['context_index']=np.full(len(rows),5,dtype=np.int8)
        first=self.fixture;n=len(first['frames']);self.assertEqual(len(rows),n)
        order=np.array([index for row in range(n) for index in (row,n+row)])
        mixed={key:np.concatenate([first[key],second[key]])[order] for key in second}
        mixed['meta']={**copy.deepcopy(first['meta']),'levels':first['meta']['levels']+[proof],
                       'on_policy_rows':[0,1]}
        index=row_metadata(mixed)
        self.assertEqual(index['level_seeds'].tolist(),[3,5])
        self.assertEqual(index['level_offsets'].tolist(),[0,n,n*2])
        self.assertEqual(index['level_rows'].tolist(),list(range(0,n*2,2))+list(range(1,n*2,2)))
        self.assertEqual(index['source_rows'].tolist(),list(range(n*2)))

    def test_cache_all_labels_actual_histories_and_imagined_fp32_quantization(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,source,report,checkpoint=self.save_source(root)
            data,hashes=load_source(source,report,checkpoint);policy=PublicPolicy()
            output=root/'cache';result=build(data,output,policy,hashes,batch_size=2,max_encoder_batch=3)
            self.assertEqual(result['format'],FORMAT)
            self.assertEqual(result['rows'],len(data['frames']))
            self.assertEqual(result['levels'],1)
            np.testing.assert_array_equal(np.load(output/'context_index.npy'),data['context_index'])
            self.assertFalse(policy.training);self.assertFalse(policy.dynamics.training)
            for name,source_name in LABELS.items():
                np.testing.assert_array_equal(np.load(output/f'{name}.npy'),data[source_name])
            current,future,_=encode_fields(RecordingAssembler(),data,max_encoder_batch=3)
            np.testing.assert_array_equal(np.load(output/'fields.npy'),current)
            np.testing.assert_array_equal(np.load(output/'next_fields.npy'),future)
            expected=(current[:,None].astype(np.float32)+np.arange(4,dtype=np.float32)[None,:,None,None]/16).astype(np.float16)
            np.testing.assert_array_equal(np.load(output/'imagined_fields.npy'),expected)
            histories,valid,actions=actual_histories(data)
            for row,action in np.argwhere(data['lost_life']):
                wanted=wd.history_arrays([data['next_frames'][row,action]],[-1],8)
                for actual,expected in zip((histories,valid,actions),wanted):
                    np.testing.assert_array_equal(actual[row,action].numpy(),expected)
            self.assertTrue(data['lost_life'].any())
            for name,entry in result['arrays'].items():self.assertEqual(digest(output/f'{name}.npy'),entry['sha256'])
            with self.assertRaises(FileExistsError):build(data,output,policy,hashes)

    def test_no_future_leak_and_atomic_source_drift_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,source,report,checkpoint=self.save_source(root)
            data,hashes=load_source(source,report,checkpoint)
            build(data,root/'a',PublicPolicy(),hashes)
            changed=copy.deepcopy(data);changed['next_frames'][:]=6;changed['next_triple'][:]=1
            build(changed,root/'b',PublicPolicy(),hashes)
            for name in ('fields','imagined_fields'):
                np.testing.assert_array_equal(np.load(root/'a'/f'{name}.npy'),np.load(root/'b'/f'{name}.npy'))
            self.assertFalse(np.array_equal(np.load(root/'a/next_fields.npy'),np.load(root/'b/next_fields.npy')))
            policy=PublicPolicy();original=policy.dynamics.predict
            def mutate(*args):
                checkpoint.write_bytes(b'changed during prediction');return original(*args)
            policy.dynamics.predict=mutate
            with self.assertRaisesRegex(ValueError,'source changed'):build(data,root/'failed',policy,hashes)
            self.assertFalse((root/'failed').exists())
            self.assertFalse(list(root.glob('.failed-*')))

    def test_cli_runs_validated_source_through_public_factory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,source,report,checkpoint=self.save_source(root)
            with patch('pebby.agent.model.load_checkpoint',return_value=(PublicPolicy(),{})) as factory:
                self.assertEqual(main(['--source',str(source),'--report',str(report),'--checkpoint',str(checkpoint),
                    '--out',str(root/'cli'),'--seconds','30']),0)
            factory.assert_called_once_with(checkpoint,device='cpu')
            self.assertEqual(json.loads((root/'cli/manifest.json').read_text())['rows'],len(self.fixture['frames']))


if __name__=='__main__':unittest.main()
