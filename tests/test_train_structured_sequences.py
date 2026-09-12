import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.train_structured_sequences import load_cache, sequence_batch
from tools.train_structured_transition import digest


class SequenceTrainerTests(unittest.TestCase):
    def fixture(self, directory):
        data = {'fields':np.zeros((2,148,96),np.float16),
                'next_fields':np.zeros((2,4,148,96),np.float16),
                'seeds':np.array([10,11]),'difficulties':np.array([1,5]),
                'actions':np.array([[3,2,0,1],[1,0,2,3]])}
        for name,width in [('player_cell',2),('triple',3),('steps',None),('lives',None)]:
            data[name]=np.zeros((2,width) if width else (2,),np.int16)
            data['next_'+name]=np.zeros((2,4,width) if width else (2,4),np.int16)
        for name in ('lost_life','terminal','won'): data[name]=np.zeros((2,4),bool)
        data['next_steps'][:]=np.arange(4)
        data['won'][0,3]=True;data['terminal'][0,3]=True
        manifest={'format':'pebby.structured-closing-sequence-cache.v1','status':'complete',
            'source':'generated_only','split':'train','field_encoder':{'test':True},
            'history_verified_against_actual_source_rows':True,
            'final_history_verified_independent_append_reset':True,'arrays':{}}
        for name,array in data.items():
            path=directory/(name+'.npy');np.save(path,array)
            manifest['arrays'][name]={'shape':list(array.shape),'dtype':str(array.dtype),'sha256':digest(path)}
        (directory/'manifest.json').write_text(json.dumps(manifest))
        return data,manifest

    def test_chronological_batch_keeps_all_four_actions_and_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            self.fixture(Path(temp));data,_=load_cache(temp,'train')
            fields,targets,actions,labels=sequence_batch(data,np.array([1,0]),'cpu')
            self.assertEqual(actions.tolist(),[[1,0,2,3],[3,2,0,1]])
            self.assertEqual(labels['next_steps'].tolist(),[[0,1,2,3],[0,1,2,3]])
            self.assertEqual(labels['won'].tolist(),[[False]*4,[False,False,False,True]])
            self.assertEqual(tuple(targets.shape),(2,4,148,96))

    def test_counterfactual_cache_cannot_be_silently_used_as_chronology(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);_,manifest=self.fixture(path)
            manifest['format']='pebby.structured-field-cache.v1'
            (path/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'chronological'):load_cache(path,'train')

    def test_tampered_targets_fail_before_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);data,_=self.fixture(path)
            data['next_steps'][0,0]=42;np.save(path/'next_steps.npy',data['next_steps'])
            with self.assertRaisesRegex(ValueError,'checksum'):load_cache(path,'train')

    def test_interior_ending_rejected_even_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);data,manifest=self.fixture(path)
            data['terminal'][0,1]=True;np.save(path/'terminal.npy',data['terminal'])
            manifest['arrays']['terminal']['sha256']=digest(path/'terminal.npy')
            (path/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'interior ending'):load_cache(path,'train')


if __name__=='__main__':unittest.main()
