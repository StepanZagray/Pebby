import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import regenerate_mechanism_banks as r
from tools import upgrade_reference_scheduler as migration


class SchedulerUpgradeTests(unittest.TestCase):
    def fixture(self,root):
        bank=root/'bank';(bank/'jobs').mkdir(parents=True)
        old_runner=Path('artifacts/reference-scheduler-pre16/regenerate_mechanism_banks.py').resolve()
        runner=str(Path(r.__file__).resolve());old_sha=r.digest(old_runner)
        row=json.loads(Path('tests/fixtures/ls20_reference/tier1.json').read_text())
        self.assertEqual(row['split'],'train');seed=row['seed']
        config=dict(profile='ls20-reference-v1',train_count=1,validation_count=1,attempts=400,limit=None,
                    train_seed_range=[seed,seed+64],validation_seed_range=[1000000,1000064])
        old=dict(config=config,source_hashes={},code_hashes={runner:old_sha,'other.py':'a'*64})
        new_codes={**old['code_hashes'],runner:r.digest(runner)}
        r._write(bank/'manifest.json',old)
        report=dict(pid=2147483647,worker_pids=[],workers_stopped=True,accepted={'train':1,'validation':0},status='failed_closed')
        r._write(bank/'generation-report.json',report)
        job=r.jobs_for(1,'train',set(),seed_range=config['train_seed_range'])[0]
        state=dict(status='accepted',next_seed=seed+1,failures=[],row=row)
        r._checkpoint(bank/'jobs'/(job['id']+'.json'),job,r._hash(old),state)
        return bank,old_runner,old_sha,new_codes,row

    def test_preserves_real_verified_row_and_archive_idempotently(self):
        with tempfile.TemporaryDirectory() as directory:
            bank,old_runner,sha,codes,row=self.fixture(Path(directory))
            before=(bank/'jobs/train-000000.json').read_bytes()
            with patch.object(r,'_inventory',return_value=(set(),set(),{})),patch.object(r,'_code_hashes',return_value=codes):
                receipt=migration.upgrade(bank,old_runner,sha)
                self.assertEqual(receipt['preserved_accepted'],{'train':1,'validation':0})
                self.assertEqual((bank/'scheduler-upgrade-v1/jobs/train-000000.json').read_bytes(),before)
                current=json.loads((bank/'jobs/train-000000.json').read_text())
                self.assertEqual(current['payload']['state']['row'],row)
                self.assertEqual(receipt,migration.upgrade(bank,old_runner,sha))

    def test_named_second_upgrade_preserves_old_archive_and_rejects_unsafe_names(self):
        with tempfile.TemporaryDirectory() as directory:
            bank,old_runner,sha,codes,row=self.fixture(Path(directory))
            prior=bank/'scheduler-upgrade-v1';prior.mkdir();(prior/'preserve.txt').write_text('original')
            with patch.object(r,'_inventory',return_value=(set(),set(),{})),patch.object(r,'_code_hashes',return_value=codes):
                result=migration.upgrade(bank,old_runner,sha,archive_name='scheduler-upgrade-v2')
                self.assertEqual(result['status'],'complete')
                self.assertEqual((prior/'preserve.txt').read_text(),'original')
                self.assertTrue((bank/'scheduler-upgrade-v2/jobs/train-000000.json').is_file())
            for invalid in ('../outside','/tmp/archive','jobs','scheduler-upgrade-v2/child'):
                with self.subTest(name=invalid),self.assertRaises(ValueError):
                    migration.upgrade(bank,old_runner,sha,archive_name=invalid)

    def test_non_scheduler_source_drift_is_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            bank,old_runner,sha,codes,row=self.fixture(Path(directory))
            codes['other.py']='b'*64
            before=(bank/'manifest.json').read_bytes()
            with patch.object(r,'_code_hashes',return_value=codes),self.assertRaisesRegex(ValueError,'non-scheduler code'):
                migration.upgrade(bank,old_runner,sha)
            self.assertEqual((bank/'manifest.json').read_bytes(),before)
            self.assertFalse((bank/'scheduler-upgrade-v1').exists())

    def test_interrupted_rebinding_recovers_from_bound_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            bank,old_runner,sha,codes,row=self.fixture(Path(directory))
            original=r._checkpoint
            def interrupted(*args):original(*args);raise InterruptedError('after envelope write')
            with patch.object(r,'_inventory',return_value=(set(),set(),{})),patch.object(r,'_code_hashes',return_value=codes):
                with patch.object(r,'_checkpoint',side_effect=interrupted),self.assertRaises(InterruptedError):
                    migration.upgrade(bank,old_runner,sha,archive_name='scheduler-upgrade-v2')
                self.assertEqual(json.loads((bank/'scheduler-upgrade-v2/upgrade.json').read_text())['status'],'prepared')
                result=migration.upgrade(bank,old_runner,sha,archive_name='scheduler-upgrade-v2')
                self.assertTrue(result['accepted_rows_unchanged'])
                self.assertEqual(result['status'],'complete')


if __name__=='__main__':unittest.main()
