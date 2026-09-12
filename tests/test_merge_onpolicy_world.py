import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.merge_onpolicy_world import merge, plan_sources, MergeError
from tools.merge_world_data import _sha256
from pebby.agent.on_policy_provenance import validate_on_policy_provenance


class OnPolicyMergeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def source(self,name,seed,marked=(0,2),behavior=None,change=None):
        checkpoint=behavior or self.root/(name+'.pt')
        if not checkpoint.exists():checkpoint.write_bytes(name.encode())
        count=3
        proof={'seed':seed,'difficulty':1,'context_index':seed%7,'context_engine_verified':True,
               'search_truncated':False,'samples':count,'on_policy_samples':len(marked),'expert_samples':count-len(marked)}
        meta={'format':'pebby.ls20-world-transitions.v1','source':'generated_only','oracle_search':'complete_only',
              'history':8,'levels':[proof],'collection_policy':'model_greedy','on_policy_rows':list(marked),
              'behavior_checkpoint':{'path':str(checkpoint),'sha256':_sha256(checkpoint),'parameters':1,'config':{'history':8}},
              'on_policy_provenance':{'official_inputs_used':False,'oracle_actions_in_policy_rollout':0}}
        if change:change(meta)
        won=np.zeros((count,4),bool);won[-1,0]=True
        next_optimal=np.ones((count,4),np.uint8);next_optimal[-1,0]=0
        arrays={'frames':np.full((count,8,64,64),seed%16,np.uint8),'next_frames':np.full((count,4,64,64),(seed+1)%16,np.uint8),
                'history_valid':np.ones((count,8),bool),'previous_actions':np.full((count,8),-1,np.int64),
                'seeds':np.full(count,seed,np.int32),'optimal':np.ones(count,np.uint8),'distances':np.ones((count,4),np.int16),
                'terminal':won,'won':won,'next_optimal':next_optimal,'context_index':np.full(count,seed%7,np.int8)}
        path=self.root/(name+'.npz');np.savez_compressed(path,**arrays,meta=np.array(json.dumps(meta)))
        return path

    def test_failure_masks_and_auxiliary_provenance_survive_merge(self):
        paths = [self.source(name, seed, change=lambda meta: meta.update(auxiliary_rows=[1]))
                 for name, seed in (('failure-a', 8), ('failure-b', 9))]
        for path in paths:
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            arrays['optimal'][1] = 0
            arrays['distances'][1] = -1
            arrays['next_optimal'][1] = 0
            arrays['terminal'][1] = True
            np.savez_compressed(path, **arrays)
        out = self.root / 'failures.npz'
        meta = merge(paths, out)
        self.assertEqual(meta['auxiliary_rows'], [1, 4])
        with np.load(out, allow_pickle=False) as archive:
            self.assertEqual(archive['optimal'][[1, 4]].tolist(), [0, 0])
            data = {'meta': meta, 'seeds': archive['seeds']}
        validate_on_policy_provenance(data)
        meta['auxiliary_rows'] = [1]
        with self.assertRaisesRegex(ValueError, 'coverage'):
            validate_on_policy_provenance(data)

    def test_offsets_images_and_multiple_behaviors_preserved(self):
        a=self.source('a',8);b=self.source('b',9,marked=(1,2));out=self.root/'merged.npz'
        meta=merge([a,b],out)
        self.assertEqual(meta['on_policy_rows'],[0,2,4,5]);self.assertNotIn('behavior_checkpoint',meta)
        self.assertEqual([(s['row_start'],s['row_stop'],s['behavior_index']) for s in meta['on_policy_sources']],[(0,3,0),(3,6,1)])
        self.assertEqual(meta['on_policy_sources'][1]['on_policy_rows'],[4,5])
        with np.load(out,allow_pickle=False) as result:
            for key in ('frames','next_frames','next_optimal','seeds'):
                with np.load(a) as aa,np.load(b) as bb:np.testing.assert_array_equal(result[key],np.concatenate([aa[key],bb[key]]))
            validate_on_policy_provenance({'meta':meta,'seeds':result['seeds']})
        self.assertFalse(list(self.root.glob('.merge-onpolicy-*')))
        with self.assertRaisesRegex(MergeError,'overwrite'):merge([a,b],out)

    def test_false_provenance_and_behavior_hash_rejected(self):
        a=self.source('a',8)
        for name,change in [('false',lambda m:m['on_policy_provenance'].update(official_inputs_used=True)),
                            ('badsha',lambda m:m['behavior_checkpoint'].update(sha256='0'*64))]:
            b=self.source(name,9,change=change)
            with self.assertRaisesRegex(MergeError,'provenance|hash mismatch'):plan_sources([a,b])

    def test_duplicate_seeds_nested_and_validation_rejected(self):
        a=self.source('a',8)
        b=self.source('same',8)
        with self.assertRaisesRegex(MergeError,'duplicate seeds'):plan_sources([a,b])
        b=self.source('validation',1000001)
        with self.assertRaisesRegex(MergeError,'training seeds'):plan_sources([a,b])
        b=self.source('nested',9,change=lambda m:m.update(behavior_checkpoints=[]))
        with self.assertRaisesRegex(MergeError,'nested'):plan_sources([a,b])

    def test_winning_coverage_and_array_schema_required(self):
        a=self.source('a',8);b=self.source('b',9)
        with np.load(b,allow_pickle=False) as archive:
            values={key:archive[key] for key in archive.files}
        values['won']=np.zeros((3,4),bool)
        np.savez_compressed(b,**values)
        with self.assertRaisesRegex(MergeError,'winning coverage'):plan_sources([a,b])
        b=self.source('other',9)
        with np.load(b,allow_pickle=False) as archive:
            values={key:archive[key] for key in archive.files}
        values['distances']=values['distances'].astype(np.int32)
        np.savez_compressed(b,**values)
        with self.assertRaisesRegex(MergeError,'schema'):plan_sources([a,b])

    def test_same_behavior_deduplicated_and_changed_source_not_published(self):
        checkpoint=self.root/'shared.pt';checkpoint.write_bytes(b'shared')
        a=self.source('a',8,behavior=checkpoint);b=self.source('b',9,behavior=checkpoint)
        _,meta=plan_sources([a,b]);self.assertEqual(len(meta['behavior_checkpoints']),1)
        self.assertEqual([s['behavior_index'] for s in meta['on_policy_sources']],[0,0])
        from tools import merge_world_data as stream
        original=stream._copy_selected_rows;changed=False
        def copy_then_mutate(*args,**kwargs):
            nonlocal changed
            original(*args,**kwargs)
            if not changed:
                checkpoint.write_bytes(b'changed');changed=True
        out=self.root/'bad.npz'
        with patch.object(stream,'_copy_selected_rows',side_effect=copy_then_mutate):
            with self.assertRaisesRegex(MergeError,'hash mismatch'):merge([a,b],out)
        self.assertFalse(out.exists());self.assertFalse(list(self.root.glob('.merge-onpolicy-*')))

if __name__=='__main__':unittest.main()
