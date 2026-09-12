"""Explicit scheduler-only checkpoint migration; never regenerates accepted rows.

The old runner snapshot must match the live bank's original source fingerprint.
All other code, source inventory, job assignments and acceptance checks remain
bound. An archived transaction permits safe completion after interrupted writes.
"""
import argparse
import ast
import copy
import fcntl
import json
import os
import re
from pathlib import Path
import shutil

from tools import regenerate_mechanism_banks as r

PROTECTED=('_generate','jobs_for','_accept','_restore','_checkpoint','_geometry',
           '_inventory','_code_hashes','shuffled_rows')


def functions(path):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(path).read_text()).body
            if isinstance(n,ast.FunctionDef)}


def upgrade(bank,old_runner,expected_old_sha,*,archive_name='scheduler-upgrade-v1'):
    bank=Path(bank).resolve();old_runner=Path(old_runner).resolve()
    if not isinstance(archive_name,str) or not re.fullmatch(r'scheduler-upgrade-[A-Za-z0-9][A-Za-z0-9._-]*',archive_name):
        raise ValueError('archive name must be one safe scheduler-upgrade-* directory name')
    runner=Path(r.__file__).resolve();archive=bank/archive_name
    if archive.is_symlink():raise ValueError('migration archive must not be a symlink')
    with (bank/'.regeneration.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt_path=archive/'upgrade.json'
        if archive.exists():
            if not receipt_path.exists():raise ValueError('incomplete migration archive without transaction receipt; inspect manually')
            receipt=json.loads(receipt_path.read_text())
            if receipt['old_runner_sha256']!=expected_old_sha or receipt['new_runner_sha256']!=r.digest(runner):raise ValueError('migration source drift')
            old=json.loads((archive/'manifest.json').read_text())
            report=json.loads((archive/'generation-report.json').read_text())
        else:
            old=json.loads((bank/'manifest.json').read_text());report=json.loads((bank/'generation-report.json').read_text())
        if old['code_hashes'].get(str(runner))!=expected_old_sha or r.digest(old_runner)!=expected_old_sha:
            raise ValueError('old source snapshot fingerprint mismatch')
        before,after=functions(old_runner),functions(runner)
        if any(before.get(name)!=after.get(name) for name in PROTECTED):raise ValueError('non-scheduler gameplay/proof/job code changed')
        if report.get('workers_stopped') is not True:raise ValueError('old workers not confirmed stopped')
        for pid in [report['pid'],*report.get('worker_pids',[])]:
            if Path(f'/proc/{pid}').exists():raise ValueError('old parent/worker PID still exists; inspect before migration')
        actual=r._code_hashes(old['config']['profile'])
        if set(actual)!=set(old['code_hashes']) or any(actual[p]!=sha for p,sha in old['code_hashes'].items() if p!=str(runner)):
            raise ValueError('non-scheduler code source changed')
        seeds,fingerprints,inventory=r._inventory(bank)
        seeds,fingerprints=set(seeds),set(fingerprints)
        if inventory!=old['source_hashes']:raise ValueError('source inventory changed')
        new=copy.deepcopy(old);new['code_hashes']=actual
        old_binding,new_binding=r._hash(old),r._hash(new)
        config=old['config'];jobs=[]
        for split in ('train','validation'):
            jobs+=r.jobs_for(config[split+'_count'],split,seeds,attempts=config['attempts'],limit=config['limit'],
                profile=config['profile'],quotas=config.get(split+'_quotas'),seed_range=config.get(split+'_seed_range'))
        expected={job['id']:job for job in jobs};original_jobs=archive/'jobs' if archive.exists() else bank/'jobs'
        states={};accepted={'train':0,'validation':0};geometries={'train':set(),'validation':set()};row_hashes={}
        for path in sorted(original_jobs.glob('*.json')):
            if path.stem not in expected:raise ValueError('unknown checkpoint job')
            job=expected[path.stem];state=r._restore(path,job,old_binding)
            states[path.stem]=state
            if state['status']=='accepted':
                if r._accept(state['row'],job,seeds,fingerprints,geometries):raise ValueError('duplicate accepted checkpoint')
                accepted[job['split']]+=1;row_hashes[path.stem]=r._hash(state['row'])
        if accepted!=report['accepted']:raise ValueError('checkpoint count differs from stopped report')
        if not archive.exists():
            archive.mkdir();shutil.copyfile(bank/'manifest.json',archive/'manifest.json')
            shutil.copyfile(bank/'generation-report.json',archive/'generation-report.json')
            shutil.copyfile(old_runner,archive/'old-runner.py');shutil.copytree(bank/'jobs',archive/'jobs')
            receipt=dict(status='prepared',old_runner_sha256=expected_old_sha,new_runner_sha256=r.digest(runner),
                old_binding=old_binding,new_binding=new_binding,migration_source_sha256=r.digest(__file__),
                preserved_accepted=accepted,accepted_row_sha256=row_hashes)
            r._write(receipt_path,receipt)
        elif (receipt['old_binding']!=old_binding or receipt['new_binding']!=new_binding
                or receipt['accepted_row_sha256']!=row_hashes or receipt['migration_source_sha256']!=r.digest(__file__)):
            raise ValueError('migration archive/config/row drift')
        if receipt['status']=='complete':
            if json.loads((bank/'manifest.json').read_text())!=new:raise ValueError('completed migration manifest changed')
            for name,sha in row_hashes.items():
                state=r._restore(bank/'jobs'/(name+'.json'),expected[name],new_binding)
                if state.get('status')!='accepted' or r._hash(state['row'])!=sha:raise ValueError('accepted row changed after migration')
            return receipt
        # Only replace the envelope binding and its checksum. Every job/state,
        # including accepted row bytes in canonical JSON, remains identical.
        for name,state in states.items():
            path=bank/'jobs'/(name+'.json')
            current=json.loads(path.read_text());payload=current['payload']
            if (current['sha256']!=r._hash(payload) or payload['job']!=expected[name]
                    or payload['state']!=state or payload['binding'] not in (old_binding,new_binding)):
                raise ValueError('live checkpoint differs from archived transaction')
            r._checkpoint(path,expected[name],new_binding,state)
        r._write(bank/'manifest.json',new)
        updated=copy.deepcopy(report);updated.update(manifest_sha256=new_binding,**new,
            scheduler_upgrade=dict(receipt=str(receipt_path),old_binding=old_binding,new_binding=new_binding))
        r._write(bank/'generation-report.json',updated)
        for name,expected_sha in row_hashes.items():
            state=r._restore(bank/'jobs'/(name+'.json'),expected[name],new_binding)
            if r._hash(state['row'])!=expected_sha:raise ValueError('accepted row changed')
        if r._inventory(bank)[2]!=inventory or r._code_hashes(config['profile'])!=actual:raise ValueError('source changed during migration')
        receipt.update(status='complete',accepted_rows_unchanged=True,checkpoint_count=len(states))
        r._write(receipt_path,receipt)
        return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank-dir',type=Path,required=True);p.add_argument('--old-runner-snapshot',type=Path,required=True)
    p.add_argument('--expected-old-runner-sha256',required=True);p.add_argument('--upgrade-scheduler',action='store_true',required=True)
    p.add_argument('--archive-name',default='scheduler-upgrade-v1',help='new named archive for each explicit scheduler upgrade')
    args=p.parse_args(argv)
    result=upgrade(args.bank_dir,args.old_runner_snapshot,args.expected_old_runner_sha256,archive_name=args.archive_name)
    print(json.dumps({k:result[k] for k in ('status','preserved_accepted','accepted_rows_unchanged')}))


if __name__=='__main__':main()
