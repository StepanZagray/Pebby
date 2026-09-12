import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from pebby.agent.structured_transition import StructuredTransition
from tools.train_structured_mixed_sequences import (
    EXPLORATORY_FORMAT, InsufficientMixedLevels, event_weights,
    load_exploratory_cache, mixed_batch, mixed_rows, source_counts,
)
from tools.train_structured_transition import digest


def fixture(path, seeds, kind):
    path.mkdir();n=len(seeds)
    data={'seeds':np.array(seeds,np.int32),'difficulties':np.arange(n,dtype=np.int8)%5+1,
          'fields':np.zeros((n,148,96),np.float16),'next_fields':np.zeros((n,4,148,96),np.float16),
          'actions':np.tile(np.arange(4,dtype=np.int64),(n,1)),
          'source_rows':np.arange(n,dtype=np.int64)*5,
          'optimal':np.full(n,8,np.int8),'next_optimal':np.full((n,4),8,np.int8),
          'distances':np.ones((n,4),np.int16)}
    data['future_rows']=data['source_rows'][:,None]+np.arange(1,5)
    for name,width in [('player_cell',2),('triple',3),('steps',None),('lives',None)]:
        data[name]=np.zeros((n,width) if width else (n,),np.int16)
        data['next_'+name]=np.zeros((n,4,width) if width else (n,4),np.int16)
    data['lives'][:]=3;data['next_lives'][:]=3
    data['next_steps'][:]=np.arange(4)
    for name in ('lost_life','terminal','won'):data[name]=np.zeros((n,4),bool)
    if kind=='closing':
        data['won'][:,-1]=True;data['terminal'][:,-1]=True
        data['next_optimal'][:,-1]=0;data['distances'][:,-1]=0
    index={'format':'pebby.ls20-exploratory-train-four-step-index.v1','mode':'explore_train',
           'source':'generated_only','split':'train','K':4,'history':8,'source_sha256':'a'*64}
    manifest={'format':EXPLORATORY_FORMAT if kind=='live' else ('pebby.structured-closing-sequence-cache.v1' if kind=='closing' else 'pebby.structured-sequence-cache.v1'),
        'mode':'exploratory_chronological_K4','status':'complete','source':'generated_only',
        'split':'validation' if kind=='validation' else 'train','field_encoder':{'test':'synthetic fixtures'},
        'history_verified_against_actual_source_rows':True,'no_future_inputs_to_current_field':True,
        'final_history_verified_independent_append_reset':True,'source_index_metadata':index,
        'source_hashes':{'synthetic_test_source':'a'*64},'arrays':{}}
    for name,value in data.items():
        f=path/(name+'.npy');np.save(f,value)
        manifest['arrays'][name]={'shape':list(value.shape),'dtype':str(value.dtype),'sha256':digest(f)}
    (path/'manifest.json').write_text(json.dumps(manifest))
    return data,manifest


class MixedSequenceTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_crossbank_overlap_excluded_and_full_batch_labels_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);live,_=fixture(root/'live',range(12),'live');closing,_=fixture(root/'closing',range(8),'closing')
            for progress in (0,.5,1):
                rows=mixed_rows(live,closing,8,progress,np.random.default_rng(42))
                self.assertEqual(tuple(map(len,rows)),(6,2))
                self.assertEqual(len(set(live['seeds'][rows[0]])|set(closing['seeds'][rows[1]])),8)
                fields,future,actions,labels=mixed_batch(live,closing,rows,'cpu')
                self.assertEqual(tuple(fields.shape),(8,148,96));self.assertEqual(tuple(future.shape),(8,4,148,96))
                np.testing.assert_array_equal(actions,np.concatenate((live['actions'][rows[0]],closing['actions'][rows[1]])))
                np.testing.assert_array_equal(labels['won'],np.concatenate((live['won'][rows[0]],closing['won'][rows[1]])))
            self.assertEqual(source_counts(1024),(768,256))
            with self.assertRaises(InsufficientMixedLevels):mixed_rows(live,closing,32,0,np.random.default_rng(42))

    def test_train_only_fixed_mixture_event_weights(self):
        live={k:np.zeros((8,4),bool) for k in ('lost_life','terminal','won')}
        closing={k:np.zeros((2,4),bool) for k in live}
        closing['terminal'][:,-1]=True;closing['won'][:,-1]=True
        rates,weights=event_weights(live,closing)
        np.testing.assert_array_equal(rates,[0,.0625,.0625]);np.testing.assert_array_equal(weights,[20,15,15])

    def test_explicit_exploratory_provenance_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'live';data,manifest=fixture(path,range(4),'live')
            loaded,_=load_exploratory_cache(path);self.assertEqual(len(loaded['seeds']),4)
            manifest['split']='validation';(path/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'TRAIN'):load_exploratory_cache(path)
            manifest['split']='train';(path/'manifest.json').write_text(json.dumps(manifest))
            data['actions'][0,0]=2;np.save(path/'actions.npy',data['actions'])
            with self.assertRaisesRegex(ValueError,'checksum'):load_exploratory_cache(path)

    def test_real_unmocked_cli_checkpoint_and_exposure(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            _,lm=fixture(root/'live',range(8),'live')
            fixture(root/'closing',[0,1,8,9],'closing');fixture(root/'val',[1000000,1000001],'validation')
            torch.manual_seed(42);model=StructuredTransition(loops=1)
            warm=root/'warm.pt';torch.save({'format':'pebby.structured-transition.v1','config':model.config(),
                'weights':model.state_dict(),'cache_manifests':{'train':lm}},warm)
            command=[sys.executable,'-m','tools.train_structured_mixed_sequences','--live-cache',str(root/'live'),
                '--closing-cache',str(root/'closing'),'--validation-cache',str(root/'val'),
                '--initialize',str(warm),'--checkpoint',str(root/'model.pt'),'--report',str(root/'report.json'),
                '--device','cpu','--max-batch','4','--eval-batch','2','--updates','2','--loops','1','--seconds','30','--checkpoint-steps']
            result=subprocess.run(command,text=True,capture_output=True,timeout=40)
            if result.stdout:print('CLI subprocess '+result.stdout.splitlines()[0],flush=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            report=json.loads((root/'report.json').read_text())
            self.assertEqual(report['status'],'complete');self.assertEqual(report['batch_source_counts'],{'live':3,'closing':1})
            self.assertEqual(report['completed_updates'],2)
            self.assertGreater(report['training'][-1]['elapsed_seconds'],0)
            self.assertEqual(report['distinct_train_levels_seen'],len(report['seen_train_seeds']))
            self.assertEqual(set(report['seen_train_seeds']),set(report['seen_seeds_by_bank']['live'])|set(report['seen_seeds_by_bank']['closing']))
            self.assertEqual(sum(report['difficulty_draws']['live']),6)
            self.assertEqual(sum(report['difficulty_draws']['closing']),2)
            self.assertEqual(report['validation_evaluation']['levels'],2)
            saved=torch.load(root/'model.pt',map_location='cpu',weights_only=False)
            self.assertEqual(saved['objective'],'mixed_autoregressive_H4')
            self.assertEqual(saved['exposure']['distinct_train_levels_seen'],report['distinct_train_levels_seen'])
            self.assertTrue(any(not torch.equal(value,model.state_dict()[key]) for key,value in saved['weights'].items()))


if __name__=='__main__':unittest.main()
