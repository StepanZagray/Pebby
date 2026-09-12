import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import regenerate_mechanism_banks as r
from tools import upgrade_fastplan_storage as m

PY_OLD='''ABI_VERSION = 1
class Tables:
    safe = 7
class Params:
    safe = 9
class PackedDistances:
    pass
def _load():
    return 1
def search():
    return 1
'''
PY_NEW=PY_OLD.replace('ABI_VERSION = 1','ABI_VERSION = 2').replace('class PackedDistances:\n    pass','class PackedDistances:\n    compact = True')
C_OLD='''int ls20_step(void) { return 7; }
int ls20_search(void) { return 9; }
int ls20_abi_version(void) {
    return 1;
}
'''
C_NEW=C_OLD.replace('return 1;','return 2;')+'''/* storage only */
int32_t *ls20_index_build(const uint64_t *keys, size_t count, size_t *capacity) { return 0; }
int64_t ls20_index_lookup(const int32_t *index, size_t capacity, const uint64_t *keys, size_t count, uint64_t key) { return -1; }
'''

class StorageMigrationTests(unittest.TestCase):
    def fixture(self,root):
        bank=root/'bank';(bank/'jobs').mkdir(parents=True)
        oldpy=root/'old.py';oldc=root/'old.c';runner=root/'runner.py';oldrunner=root/'old-runner.py'
        newpy=root/'fastplan.py';newc=root/'fastplan.c'
        oldpy.write_text(PY_OLD);newpy.write_text(PY_NEW);oldc.write_text(C_OLD);newc.write_text(C_NEW)
        text='\n'.join(f'def {name}(*args, **kwargs):\n    return 1\n' for name in m.PROTECTED)
        oldrunner.write_text(text+'\nCAP=2\n');runner.write_text(text+'\nCAP=4\n')
        oldsources={str(newpy):m.sha(oldpy),str(newc):m.sha(oldc),str(runner):m.sha(oldrunner)}
        newsources={str(p):m.sha(p) for p in (newpy,newc,runner)}
        row=json.loads(Path('tests/fixtures/ls20_reference/tier1.json').read_text());seed=row['seed']
        cfg=dict(profile='ls20-reference-v1',train_count=1,validation_count=1,attempts=400,limit=None,
                 train_seed_range=[seed,seed+64],validation_seed_range=[1000000,1000064])
        manifest=dict(config=cfg,code_hashes={**oldsources,'unchanged.py':'a'*64},source_hashes={})
        r._write(bank/'manifest.json',manifest)
        r._write(bank/'generation-report.json',dict(status='failed_closed',pid=2147483647,worker_pids=[],
                 workers_stopped=True,accepted={'train':1,'validation':0},manifest_sha256=r._hash(manifest),**manifest))
        job=r.jobs_for(1,'train',set(),seed_range=cfg['train_seed_range'])[0]
        state=dict(status='accepted',next_seed=seed+1,failures=[],row=row)
        r._checkpoint(bank/'jobs/train-000000.json',job,r._hash(manifest),state)
        receipt=root/'validation.json';r._write(receipt,dict(format=m.RECEIPT_FORMAT,status='complete',
            old_sources=oldsources,new_sources=newsources,generated_only=True,tests_passed=True,
            solver_unchanged=True,equality_verified=True))
        args=(bank,oldpy,oldc,oldrunner,m.sha(oldpy),m.sha(oldc),m.sha(oldrunner),receipt)
        patches=[patch.object(m,'LIVE_FASTPLAN',newpy),patch.object(m,'LIVE_C',newc),patch.object(m,'LIVE_RUNNER',runner),
                 patch.object(r,'_code_hashes',return_value={**newsources,'unchanged.py':'a'*64}),
                 patch.object(r,'_inventory',return_value=(set(),set(),{}))]
        from contextlib import ExitStack
        stack=ExitStack()
        for value in patches:stack.enter_context(value)
        return args,stack,row

    def test_real_accepted_row_preserved_and_idempotent(self):
        with tempfile.TemporaryDirectory() as root:
            args,stack,row=self.fixture(Path(root))
            with stack:
                before=(args[0]/'jobs/train-000000.json').read_bytes()
                result=m.upgrade(*args)
                self.assertTrue(result['accepted_rows_unchanged'])
                self.assertEqual((args[0]/m.ARCHIVE/'jobs/train-000000.json').read_bytes(),before)
                self.assertEqual(json.loads((args[0]/'jobs/train-000000.json').read_text())['payload']['state']['row'],row)
                self.assertEqual(m.upgrade(*args),result)

    def test_interruption_after_first_rebind_resumes_identical_transaction(self):
        with tempfile.TemporaryDirectory() as root:
            args,stack,row=self.fixture(Path(root))
            with stack:
                original=m.durable_json
                def interrupt(path,value):
                    original(path,value)
                    if path.parent==args[0]/'jobs':raise InterruptedError('after rebind')
                with patch.object(m,'durable_json',side_effect=interrupt),self.assertRaises(InterruptedError):m.upgrade(*args)
                self.assertEqual(json.loads((args[0]/m.ARCHIVE/'upgrade.json').read_text())['status'],'prepared')
                self.assertEqual(m.upgrade(*args)['status'],'complete')

    def test_prepared_archive_publication_interruption_resumes(self):
        with tempfile.TemporaryDirectory() as root:
            args,stack,row=self.fixture(Path(root))
            with stack:
                original=m.os.replace
                def interrupt(source,target):
                    if Path(target)==args[0]/m.ARCHIVE:raise InterruptedError('before archive publication')
                    return original(source,target)
                with patch.object(m.os,'replace',side_effect=interrupt),self.assertRaises(InterruptedError):m.upgrade(*args)
                self.assertTrue((args[0]/(m.ARCHIVE+'.preparing')/'upgrade.json').exists())
                self.assertEqual(m.upgrade(*args)['status'],'complete')

    def test_running_process_blocks_before_archive(self):
        import os
        with tempfile.TemporaryDirectory() as root:
            args,stack,row=self.fixture(Path(root))
            with stack:
                path=args[0]/'generation-report.json';report=json.loads(path.read_text())
                report['pid']=os.getpid();r._write(path,report)
                with self.assertRaisesRegex(ValueError,'PID exists'):m.upgrade(*args)
                self.assertFalse((args[0]/m.ARCHIVE).exists())

    def test_rejects_solver_python_and_c_edits(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);old=root/'old';new=root/'new'
            old.write_text(PY_OLD);new.write_text(PY_NEW.replace('safe = 7','safe = 8'))
            with self.assertRaisesRegex(ValueError,'non-storage Python'):m.verify_python(old,new)
            old.write_text(C_OLD)
            for text in (C_NEW.replace('return 7;','return 8;'),C_NEW+'\nint surprise(void) {return 0;}\n',
                         C_NEW+'\n#define ls20_search evil\n'):
                new.write_text(text)
                with self.assertRaises(ValueError):m.verify_c(old,new)
            new.write_text(C_NEW);m.verify_c(old,new)

    def test_receipt_source_and_archive_corruption_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            args,stack,row=self.fixture(Path(root))
            with stack:
                receipt=json.loads(args[-1].read_text());receipt['equality_verified']=False;r._write(args[-1],receipt)
                with self.assertRaisesRegex(ValueError,'receipt required'):m.upgrade(*args)
                self.assertFalse((args[0]/m.ARCHIVE).exists())
                receipt['equality_verified']=True;r._write(args[-1],receipt);m.upgrade(*args)
                (args[0]/m.ARCHIVE/'jobs/train-000000.json').write_text('{}')
                with self.assertRaisesRegex(ValueError,'archived source/job corruption'):m.upgrade(*args)

    def test_live_row_and_nonallowed_source_changes_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            args,stack,row=self.fixture(Path(root))
            with stack:
                codes=dict(r._code_hashes('x'));codes['unchanged.py']='b'*64
                with patch.object(r,'_code_hashes',return_value=codes),self.assertRaisesRegex(ValueError,'unapproved'):m.upgrade(*args)
                m.upgrade(*args)
                path=args[0]/'jobs/train-000000.json';envelope=json.loads(path.read_text())
                envelope['payload']['state']['row']['optimal_actions']+=1
                envelope['sha256']=r._hash(envelope['payload']);r._write(path,envelope)
                with self.assertRaisesRegex(ValueError,'live job state'):m.upgrade(*args)

if __name__=='__main__':unittest.main()
