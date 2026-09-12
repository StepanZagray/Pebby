"""Initial public row identity, immutable ordering and encoder input boundary."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools.build_structured_initial_cache import (select_initial_rows,parent_metadata,
                                                 encode_public_initial,build_split,digest)


def data_fixture():
    seeds=np.array([1,2,1,3,2],dtype=np.int32)
    valid=np.ones((5,8),dtype=bool);valid[[0,1,3],:-1]=False
    previous=np.zeros((5,8),dtype=np.int64);previous[[0,1,3]]=-1
    frames=np.repeat(np.arange(5,dtype=np.uint8)[:,None,None,None],8*64*64,axis=1).reshape(5,8,64,64)
    return dict(seeds=seeds,history_valid=valid,previous_actions=previous,current_lives=np.full(5,3,dtype=np.int16),frames=frames,
                next_frames=np.full((5,4,64,64),15,dtype=np.uint8),player_cell=np.full((5,2),11))


class PublicEncoder(torch.nn.Module):
    def __init__(self,callback=None):super().__init__();self.calls=[];self.callback=callback
    def forward(self,frames,valid,previous):
        self.calls.append((frames.clone(),valid.clone(),previous.clone()))
        if self.callback:self.callback()
        value=torch.zeros(len(frames),148,96);value[...,0]=frames[:,-1,0,0,None].float()
        return value


class StructuredInitialCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_exact_unique_initial_selection_preserves_parent_order(self):
        data=data_fixture();rows=select_initial_rows(data,np.array([3,1,2]),'train')
        np.testing.assert_array_equal(rows,[3,0,1])
        data['current_lives'][2]=2;data['history_valid'][2,:-1]=False;data['previous_actions'][2]=-1
        np.testing.assert_array_equal(select_initial_rows(data,np.array([1]),'train'),[0])

    def test_missing_ambiguous_split_and_duplicate_seeds_fail_closed(self):
        data=data_fixture();data['current_lives'][0]=2
        with self.assertRaisesRegex(ValueError,'0 initial rows'):select_initial_rows(data,np.array([1]),'train')
        data=data_fixture();data['history_valid'][2,:-1]=False;data['previous_actions'][2]=-1
        with self.assertRaisesRegex(ValueError,'2 initial rows'):select_initial_rows(data,np.array([1]),'train')
        with self.assertRaises(ValueError):select_initial_rows(data_fixture(),np.array([1]),'validation')
        with self.assertRaises(ValueError):select_initial_rows(data_fixture(),np.array([1,1]),'train')
        with self.assertRaises(ValueError):select_initial_rows(data_fixture(),np.array([1.]),'train')

    def test_only_stored_public_history_crosses_encoder_boundary(self):
        data=data_fixture();encoder=PublicEncoder();rows=np.array([3,0,1])
        encoded,error=encode_public_initial(encoder,data,rows)
        self.assertEqual(encoded.shape,(3,148,96));self.assertEqual(encoded.dtype,np.float16);self.assertEqual(error,0.)
        np.testing.assert_array_equal(encoded[:,0,0],[3,0,1])
        frames,valid,previous=encoder.calls[0]
        np.testing.assert_array_equal(frames.numpy(),data['frames'][rows])
        self.assertEqual(len(encoder.calls[0]),3);self.assertTrue(valid[:,-1].all());self.assertTrue((previous==-1).all())
        # Contradictory hidden labels/future images cannot alter this boundary.
        data['next_frames'][:]=7;data['player_cell'][:]=0
        np.testing.assert_array_equal(encode_public_initial(encoder,data,rows)[0],encoded)
        with self.assertRaises(ValueError):encode_public_initial(encoder,data,np.array([2]))

    def parent_fixture(self,root):
        cache=root/'parent';cache.mkdir();source=root/'source.npz';source.write_bytes(b'hashed-source-placeholder')
        np.save(cache/'seeds.npy',np.array([3,1,2],dtype=np.int32));np.save(cache/'source_rows.npy',np.array([3,2,4],dtype=np.int64))
        manifest=dict(format='pebby.structured-field-cache.v1',status='complete',source='generated_only',split='train',
                      source_path=str(source),source_sha256=digest(source),field_encoder={'test':'stub'},
                      arrays={name:{'sha256':digest(cache/(name+'.npy'))} for name in ('seeds','source_rows')})
        (cache/'manifest.json').write_text(json.dumps(manifest))
        return cache,source

    def test_parent_split_and_seed_hash_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            cache,_=self.parent_fixture(Path(directory))
            _,values,_=parent_metadata(cache,'train');np.testing.assert_array_equal(values['seeds'],[3,1,2])
            with self.assertRaises(ValueError):parent_metadata(cache,'validation')
            np.save(cache/'seeds.npy',np.array([1,3,2]))
            with self.assertRaisesRegex(ValueError,'hash mismatch'):parent_metadata(cache,'train')

    def test_atomic_sidecar_exact_order_and_partial_alignment_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cache,source=self.parent_fixture(root);data=data_fixture()
            watched={str(source):digest(source)}
            with patch('tools.build_structured_initial_cache.source_arrays',return_value=(data,watched)), \
                    patch('tools.build_structured_initial_cache.encoder_for',return_value=(PublicEncoder(),{})):
                report=build_split(cache,root/'out','train',limit=2,batch_size=1)
            self.assertFalse(report['complete_parent_alignment']);self.assertEqual(report['parent_levels'],3)
            np.testing.assert_array_equal(np.load(root/'out/seeds.npy'),[3,1])
            np.testing.assert_array_equal(np.load(root/'out/source_initial_rows.npy'),[3,0])
            np.testing.assert_array_equal(np.load(root/'out/initial_fields.npy')[:,0,0],[3,0])
            for name,info in report['arrays'].items():self.assertEqual(info['sha256'],digest(root/'out'/f'{name}.npy'))
            with self.assertRaises(FileExistsError):build_split(cache,root/'out','train')

    def test_source_drift_and_wrong_source_seed_alignment_never_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cache,source=self.parent_fixture(root);data=data_fixture();watched={str(source):digest(source)}
            encoder=PublicEncoder(lambda:source.write_bytes(b'changed'))
            with patch('tools.build_structured_initial_cache.source_arrays',return_value=(data,watched)), \
                    patch('tools.build_structured_initial_cache.encoder_for',return_value=(encoder,{})):
                with self.assertRaisesRegex(ValueError,'drift'):build_split(cache,root/'out','train')
            self.assertFalse((root/'out').exists());self.assertFalse(list(root.glob('.out-*')))
            data['seeds'][4]=1
            with patch('tools.build_structured_initial_cache.source_arrays',return_value=(data,{})):
                with self.assertRaisesRegex(ValueError,'ordering'):build_split(cache,root/'out','train')

    def closing_fixture(self,root):
        cache,source=self.parent_fixture(root)
        branch=root/'branch.npz';np.savez(branch,seeds=np.array([2,3,1]))
        np.save(cache/'source_rows.npy',np.array([1,2,0]))
        np.save(cache/'source_initial_rows.npy',np.array([3,0,1]))
        path=cache/'manifest.json';m=json.loads(path.read_text())
        m.update(format='pebby.structured-closing-sequence-cache.v1',mode='closing_only_chronological_K4',
                 history_verified_against_actual_source_rows=True,final_history_verified_independent_append_reset=True,
                 source_hashes={str(branch):digest(branch),str(source):digest(source)},
                 source_index_metadata=dict(format='pebby.ls20-closing-four-step-index.v1',source='generated_only',split='train',
                                            mode='closing_only_train',K=4,history=8,source_path=str(branch),source_sha256=digest(branch)),
                 initial_source_mapping=dict(source_path=str(source),source_sha256=digest(source),array='source_initial_rows'))
        for name in ('source_rows','source_initial_rows'):m['arrays'][name]={'sha256':digest(cache/(name+'.npy'))}
        m.pop('source_path');m.pop('source_sha256');path.write_text(json.dumps(m))
        return cache,source,branch

    def test_closing_uses_combined_initial_rows_not_aggregate_branch_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cache,source,branch=self.closing_fixture(root)
            with patch('tools.build_structured_initial_cache.source_arrays',return_value=(data_fixture(),{})), \
                    patch('tools.build_structured_initial_cache.encoder_for',return_value=(PublicEncoder(),{})):
                result=build_split(cache,root/'out','train')
            np.testing.assert_array_equal(np.load(root/'out/source_initial_rows.npy'),[3,0,1])
            self.assertEqual(result['source_path'],str(source));self.assertEqual(result['parent_branch_source_path'],str(branch))
            self.assertTrue(result['complete_parent_alignment'])

    def test_closing_rejects_wrong_explicit_initial_row_even_when_seed_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cache,_,_=self.closing_fixture(root)
            np.save(cache/'source_initial_rows.npy',np.array([3,2,1]))
            p=cache/'manifest.json';m=json.loads(p.read_text());m['arrays']['source_initial_rows']['sha256']=digest(cache/'source_initial_rows.npy');p.write_text(json.dumps(m))
            with patch('tools.build_structured_initial_cache.source_arrays',return_value=(data_fixture(),{})):
                with self.assertRaisesRegex(ValueError,'unique verified initial'):build_split(cache,root/'out','train')
            self.assertFalse((root/'out').exists())

    def test_closing_rejects_branch_seed_and_mapping_hash_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cache,_,branch=self.closing_fixture(root)
            np.savez(branch,seeds=np.array([3,2,1]))
            with self.assertRaisesRegex(ValueError,'hash mismatch'):parent_metadata(cache,'train')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cache,_,_=self.closing_fixture(root);p=cache/'manifest.json';m=json.loads(p.read_text())
            m['initial_source_mapping']['source_sha256']='0'*64;p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'hash-bound'):parent_metadata(cache,'train')

    def test_live_validation_format_is_strict_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cache,source=self.parent_fixture(root)
            np.save(cache/'seeds.npy',np.array([1000003,1000001,1000002]))
            p=cache/'manifest.json';m=json.loads(p.read_text())
            m.update(format='pebby.structured-sequence-cache.v1',split='validation',mode='heldout_chronological_K4',
                     history_verified_against_actual_source_rows=True,source_hashes={str(source):digest(source)},
                     source_index_metadata=dict(format='pebby.ls20-four-step-index.v1',source='generated_only',split='validation',
                                                mode='explore_validation',K=4,history=8,source_path=str(source),source_sha256=digest(source)))
            m['arrays']['seeds']['sha256']=digest(cache/'seeds.npy');m.pop('source_path');m.pop('source_sha256');p.write_text(json.dumps(m))
            normalized,values,_=parent_metadata(cache,'validation')
            self.assertEqual(normalized['source_path'],str(source));np.testing.assert_array_equal(values['seeds'],[1000003,1000001,1000002])
            with self.assertRaises(ValueError):parent_metadata(cache,'train')
            m['source_index_metadata']['K']=3;p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'H4'):parent_metadata(cache,'validation')


if __name__=='__main__':unittest.main()
