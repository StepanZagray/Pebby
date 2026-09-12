"""Second-view selection, source binding, and real generated branch preservation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pebby.agent import world_data
from tests.test_policy_history import corridor
from tests.test_structured_field_cache import RecordingAssembler
from tools.build_structured_field_cache import LABELS, write_split, digest, encode_fields
from tools.build_structured_additional_state_cache import select_additional_rows, build, load_parent


class AdditionalStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def data(self):
        spec = corridor(budget=20);spec.update(seed=3,difficulty=1)
        rows, proof = world_data.collect_level(spec,history=8,samples=6,epsilon=0,context_index=1)
        self.assertTrue(proof['context_engine_verified'])
        self.assertGreater(len(rows),1)
        result = {key:np.asarray([row[key] for row in rows]) for key in rows[0]}
        result['meta'] = {'source':'generated_only','oracle_search':'complete_only','split':'train','levels':[proof]}
        return result

    def test_selection_matches_independent_uniform_exclusion_and_singletons(self):
        data = self.data()
        old=np.array([1]);levels=np.array([3]);tiers=np.array([1])
        expected=int(np.random.default_rng(43).choice([i for i in range(len(data['seeds'])) if i!=1]))
        selected,singleton=select_additional_rows(data,old,levels,tiers)
        self.assertEqual(selected.tolist(),[expected]);self.assertEqual(singleton.tolist(),[False])
        again=select_additional_rows(data,old,levels,tiers)
        np.testing.assert_array_equal(selected,again[0])
        for key,value in list(data.items()):
            if isinstance(value,np.ndarray):data[key]=value[:1]
        selected,singleton=select_additional_rows(data,np.array([0]),levels,tiers)
        self.assertEqual(selected.tolist(),[0]);self.assertEqual(singleton.tolist(),[True])

    def test_selection_rejects_split_row_order_and_missing_labels(self):
        data=self.data()
        for changed,pattern in [('split','TRAIN'),('row','order'),('label','missing'),('tier','difficulty')]:
            value=copy.deepcopy(data);old=np.array([0]);levels=np.array([3]);tiers=np.array([1])
            if changed=='split':value['meta']['split']='validation'
            if changed=='row':levels[0]=4
            if changed=='label':del value['next_player_cell']
            if changed=='tier':tiers[0]=2
            with self.assertRaisesRegex(ValueError,pattern):select_additional_rows(value,old,levels,tiers)

    def test_real_generated_cache_is_paired_exact_and_fails_closed(self):
        data=self.data()
        provenance=json.loads(Path('data/structured-field-cache-smoke8/train/manifest.json').read_text())['field_encoder']
        assembler=RecordingAssembler();assembler.metadata=lambda:copy.deepcopy(provenance)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.npz'
            np.savez(source,**{k:v for k,v in data.items() if k!='meta'},meta=json.dumps(data['meta']))
            parent=root/'original'
            write_split(data,source,parent,assembler,1,42,'train',provenance)
            result=build(data,source,parent,root/'additional',assembler,max_encoder_batch=1)
            newrow=int(np.load(root/'additional/source_rows.npy')[0]);oldrow=int(np.load(parent/'source_rows.npy')[0])
            self.assertNotEqual(newrow,oldrow)
            self.assertTrue(result['paired_source']['identical_ordered_seeds'])
            self.assertEqual(result['paired_source']['singleton_seeds'],[])
            for output,label in LABELS.items():
                np.testing.assert_array_equal(np.load(root/'additional'/f'{output}.npy'),data[label][[newrow]])
            batch={k:data[k][[newrow]] for k in ('frames','history_valid','previous_actions','next_frames','lost_life')}
            current,future,_=encode_fields(RecordingAssembler(),batch,max_encoder_batch=1)
            np.testing.assert_array_equal(np.load(root/'additional/fields.npy'),current)
            np.testing.assert_array_equal(np.load(root/'additional/next_fields.npy'),future)
            load_parent(root/'additional')
            with self.assertRaises(FileExistsError):build(data,source,parent,root/'additional',assembler)
            altered=copy.deepcopy(data);altered['current_steps'][oldrow]+=1
            with self.assertRaisesRegex(ValueError,'exact source label'):
                build(altered,source,parent,root/'badlabels',assembler)
            bad=RecordingAssembler();bad.metadata=lambda:{**provenance,'config':{}}
            with self.assertRaisesRegex(ValueError,'encoder mismatch'):
                build(data,source,parent,root/'badencoder',bad)
            path=parent/'seeds.npy';np.save(path,np.array([4]))
            with self.assertRaisesRegex(ValueError,'hash mismatch'):load_parent(parent)
            self.assertFalse((root/'badlabels').exists())
            self.assertFalse((root/'badencoder').exists())


if __name__=='__main__':unittest.main()
