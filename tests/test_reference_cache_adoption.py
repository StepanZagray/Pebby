"""External receipt validation uses tiny files; existing validators own array math."""
from contextlib import ExitStack
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from tools import run_reference_base_pipeline as p


class CacheAdoptionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.bank=self.root/'bank';self.cache=self.root/'cache'
        self.bank.mkdir();self.cache.mkdir();(self.cache/'.build.lock').touch()
        self.snap=self.root/'frozen';(self.snap/'tools').mkdir(parents=True)
        self.code=self.snap/'tools/build_reference_world_cache.py';self.code.write_text('frozen code')
        self.old=self.root/'old-runtime';self.old.mkdir()
        self.put(self.snap/'snapshot.json',dict(snapshot_root=str(self.snap),source_root=str(self.old),
            sources={'tools/build_reference_world_cache.py':p.digest(self.code)}))
        inventory=self.root/'inventory';inventory.write_text('generated inventory')
        gen=dict(status='complete',workers_stopped=True,config={},banks={},
            source_hashes={str(inventory):p.digest(inventory)},
            code_hashes={str(self.old/'tools/build_reference_world_cache.py'):p.digest(self.code)})
        self.specs={}
        for split,offset in [('train',100),('validation',1000)]:
            self.specs[split]=[dict(seed=offset+i,difficulty=i+1,split=split) for i in range(7)]
            path=self.bank/(split+'.jsonl');path.write_text(''.join(json.dumps(s)+'\n' for s in self.specs[split]))
            gen['banks'][split]=dict(sha256=p.digest(path),levels=7)
        self.put(self.bank/'manifest.json',{k:gen[k] for k in ('config','source_hashes','code_hashes')})
        generation_manifest=json.loads((self.bank/'manifest.json').read_text())
        gen['manifest_sha256']=hashlib.sha256(json.dumps(generation_manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.put(self.bank/'generation-report.json',gen)
        inputs={str(self.bank/name):p.digest(self.bank/name) for name in ('train.jsonl','validation.jsonl','generation-report.json')}
        self.audit=self.root/'audit.json';self.put(self.audit,dict(format='pebby.generated-bank-audit.v1',
            status='complete',errors=[],coverage_policy='full',required_validation_levels=500,input_hashes_unchanged=True,
            generation_evidence=dict(binding_verified=True),input_sha256=inputs,
            audit_code_sha256={str(self.code):p.digest(self.code)},spotchecks={split:[dict(seed=s['seed'],
                same_planner_search_agrees=True,engine_win_three_lives=True) for s in specs] for split,specs in self.specs.items()}))
        sources=inputs|{str(self.code):p.digest(self.code)}
        self.manifest=dict(sources=sources,config=dict(history=8,samples=32,epsilon=.15,coverage='mixed_failure',shard_levels=100))
        self.put(self.cache/'manifest.json',self.manifest)
        self.report=dict(status='complete',sources_unchanged=True,workers_stopped=True,
            source='generated_only',official_frames_or_routes_used=False,**self.manifest,splits={})
        self.arrays={}
        for split,specs in self.specs.items():
            directory=self.cache/split;directory.mkdir();path=directory/'shard-000000.npz';path.write_bytes(split.encode())
            proofs=[dict(seed=s['seed'],search_truncated=False) for s in specs]
            record=dict(path=str(path),sha256=p.digest(path),seeds=[s['seed'] for s in specs],rows=7)
            self.put(path.with_suffix('.json'),record)
            validity=directory/'validity.json';self.put(validity,dict(status='complete',levels=proofs))
            merged=self.cache/(split+'.npz');merged.write_bytes((split+' merged').encode())
            self.put(directory/'merged.json',dict(sha256=p.digest(merged)))
            self.report['splits'][split]=dict(status='complete',requested_levels=7,collected_levels=7,
                shards=[record],sha256=p.digest(merged),output=str(merged),rows=7,max_distance=8)
            data=dict(seeds=np.array([s['seed'] for s in specs]),distances=np.full((7,4),8),
                meta=dict(split=split,bank_sha256=inputs[str(self.bank/(split+'.jsonl'))],levels=proofs))
            self.arrays[str(path)]=data;data=copy.deepcopy(data)
            data['meta'].update(source_sha256={str(path):p.digest(path)},audit_sha256=p.digest(validity),
                levels=[dict(proof,accepted=True,proof=dict(proof)) for proof in proofs])
            self.arrays[str(merged)]=data
            from pebby.agent.world_train import REQUIRED_ARRAYS,OPTIONAL_ARRAYS
            names=sorted(set((*REQUIRED_ARRAYS,*OPTIONAL_ARRAYS,'context_index','meta')))
            schema=hashlib.sha256(json.dumps(names).encode()).hexdigest()[:16]
            mmap_root=self.cache/'array-cache'/f'{p.digest(merged)}-{schema}';mmap_root.mkdir(parents=True)
            self.put(mmap_root/'manifest.json',dict(source_sha256=p.digest(merged),arrays={}))
        self.put(self.cache/'build-report.json',self.report)
        stack=self.enterContext(ExitStack())
        stack.enter_context(patch.object(p,'check_bank_counts',return_value={'train':7,'validation':7}))
        stack.enter_context(patch('pebby.agent.world_train.load_dataset',side_effect=lambda path,**kw:self.arrays[str(path)]))
        stack.enter_context(patch('tools.stream_extended_collection.validate_arrays',side_effect=lambda data,specs:dict(rows=len(data['seeds']))))
        for name in ('require_verified_data','require_winning_coverage','validate_successor_labels'):
            stack.enter_context(patch('pebby.agent.world_train.'+name))

    def put(self,path,value):path.write_text(json.dumps(value))
    def adopt(self):return p.validate_adopted_cache(self.bank,self.cache,self.audit)

    def test_adopts_external_completed_receipts_without_current_source_comparison(self):
        # The former runtime no longer exists; its exact snapshot is sufficient.
        report,receipt=self.adopt();self.assertEqual(report['status'],'complete')
        self.assertEqual(receipt['origin'],'adopted');self.assertNotIn('stages',receipt)
        self.assertIn(str(self.snap/'snapshot.json'),receipt['inputs'])
        self.assertIn(str(self.cache/'train.npz'),receipt['inputs'])
        p.verify(receipt['inputs'])

    def test_rejects_running_cache_and_holds_builder_lock(self):
        self.report['status']='running';self.put(self.cache/'build-report.json',self.report)
        with self.assertRaisesRegex(ValueError,'complete stopped'):self.adopt()
        self.report['status']='complete';self.put(self.cache/'build-report.json',self.report)
        with (self.cache/'.build.lock').open() as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError,'still active'):self.adopt()
        self.adopt() # rejected attempt released its own descriptor

    def test_generation_binding_uses_content_hash_and_guard_binds_exact_bytes(self):
        path=self.bank/'manifest.json';manifest=json.loads(path.read_text())
        path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
        _,receipt=self.adopt()
        path.write_text(json.dumps(manifest,indent=4)+'\n')
        with self.assertRaisesRegex(ValueError,'changed'):p.verify(receipt['inputs'])
        manifest['config']['changed']=True;self.put(path,manifest)
        with self.assertRaisesRegex(ValueError,'content hash mismatch'):self.adopt()

    def test_rejects_modified_frozen_source_and_false_snapshot_manifest(self):
        self.code.write_text('changed')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.adopt()
        self.code.write_text('frozen code')
        snap=json.loads((self.snap/'snapshot.json').read_text());snap['sources']['tools/build_reference_world_cache.py']='0'*64
        self.put(self.snap/'snapshot.json',snap)
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.adopt()

    def test_rejects_audit_weakening_and_wrong_bank_binding(self):
        old=json.loads(self.audit.read_text())
        for key,value in [('coverage_policy','sample'),('required_validation_levels',499),('generation_evidence',{})]:
            self.put(self.audit,dict(old,**{key:value}))
            with self.assertRaisesRegex(ValueError,'full bank audit'):self.adopt()
        bad=copy.deepcopy(old);bad['input_sha256'][str(self.bank/'train.jsonl')]='0'*64;self.put(self.audit,bad)
        with self.assertRaisesRegex(ValueError,'input bindings'):self.adopt()
        bad=copy.deepcopy(old);bad['spotchecks']['train'][0]['engine_win_three_lives']=False;self.put(self.audit,bad)
        with self.assertRaisesRegex(ValueError,'spotchecks'):self.adopt()

    def test_rejects_missing_shards_merged_corruption_and_proof_drift(self):
        self.report['splits']['train']['shards']=[];self.put(self.cache/'build-report.json',self.report)
        with self.assertRaisesRegex(ValueError,'cover complete bank'):self.adopt()
        record=json.loads((self.cache/'train/shard-000000.json').read_text())
        self.report['splits']['train']['shards']=[record];self.put(self.cache/'build-report.json',self.report)
        self.arrays[str(self.cache/'train.npz')]['meta']['levels'][0]['search_truncated']=True
        with self.assertRaisesRegex(ValueError,'proof differs'):self.adopt()
        self.arrays[str(self.cache/'train.npz')]['meta']['levels'][0]['search_truncated']=False
        (self.cache/'train.npz').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.adopt()

    def test_supervisor_adoption_enters_shared_tail_without_fake_child_stages(self):
        out=self.root/'run';out.mkdir()
        args=SimpleNamespace(out_dir=out,resume=False,adopt_cache_audit=self.audit,bank_dir=self.bank,cache_dir=self.cache,
            generation_pid=None,generation_start_ticks=None)
        with patch.object(p,'code_sources',return_value={}),patch.object(p.Supervisor,'run_training') as tail:
            runner=p.Supervisor(args);runner.run()
            tail.assert_called_once();self.assertEqual(runner.report['stages'],{})
            self.assertIsNone(runner.report['generation_identity']);self.assertIn('adoption',runner.report)
            (self.cache/'train.npz').write_bytes(b'late change')
            with self.assertRaisesRegex(ValueError,'changed'):runner.guard()


if __name__=='__main__':unittest.main()
