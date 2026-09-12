"""Explicit, receipt-gated storage-only fastplan checkpoint migration.

Only fastplan.py's result Mapping/search/load/ABI and C's appended index helpers
may change, alongside protected-AST-checked scheduler code. No search is run.
"""
import argparse
import ast
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from tools import regenerate_mechanism_banks as r
from tools.upgrade_reference_scheduler import PROTECTED, functions

ARCHIVE='fastplan-storage-upgrade-v1'
FORMAT='pebby.fastplan-storage-upgrade.v1'
RECEIPT_FORMAT='pebby.fastplan-storage-validation.v1'
LIVE_FASTPLAN=Path(r.REPO_ROOT)/'pebby/ls20/fastplan.py'
LIVE_C=Path(r.REPO_ROOT)/'pebby/ls20/_fastplan.c'
LIVE_RUNNER=Path(r.__file__).resolve()


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def durable_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,sort_keys=True,separators=(',',':'));stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def verify_python(old,new):
    """All top-level declarations except the four named changes stay exact."""
    allowed={'PackedDistances','search','_load'}
    def protected(path,version):
        text=Path(path).read_text();tree=ast.parse(text);result=[];found=set()
        for node in tree.body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in allowed:
                if node.name in found:raise ValueError('duplicate allowed Python declaration')
                found.add(node.name);continue
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='ABI_VERSION':
                if 'ABI_VERSION' in found or not isinstance(node.value,ast.Constant) or type(node.value.value) is not int or node.value.value!=version:
                    raise ValueError('expected exact Python ABI1 to ABI2 change')
                found.add('ABI_VERSION');continue
            result.append((ast.dump(node,include_attributes=False),ast.get_source_segment(text,node)))
        if found!=allowed|{'ABI_VERSION'}:raise ValueError('missing expected Python storage declarations')
        return result
    if protected(old,1)!=protected(new,2):raise ValueError('non-storage Python code changed')


def verify_c(old,new):
    """Keep the entire old translation unit byte-identical except ABI1→2."""
    before=Path(old).read_bytes();after=Path(new).read_bytes()
    pattern=rb'int ls20_abi_version\(void\) \{\s*return 1;\s*\}'
    matches=list(re.finditer(pattern,before))
    if len(matches)!=1 or before[matches[0].end():].strip():
        raise ValueError('old C must end with the recognized ABI1 function')
    match=matches[0];normalized=before[:match.start()]+match.group().replace(b'return 1;',b'return 2;')
    if not after.startswith(normalized):raise ValueError('existing C code changed outside exact ABI return')
    tail=after[len(normalized):].decode('utf-8')
    # Mask literals/comments before brace matching. No appended declarations,
    # preprocessor directives, extra functions or altered old functions allowed.
    tokens=re.compile(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',re.S)
    masked=tokens.sub(lambda m:' '*len(m.group()),tail)
    position=0
    for return_type,name in ((r'int32_t\s*\*','ls20_index_build'),(r'int64_t','ls20_index_lookup')):
        header=re.compile(r'\s*'+return_type+r'\s*'+name+r'\s*\([^{};#]*\)\s*\{')
        found=header.match(masked,position)
        if not found:raise ValueError('only appended index build/lookup C helpers are allowed')
        position=found.end();depth=1
        while position<len(masked) and depth:
            if masked[position]=='#':raise ValueError('preprocessor directives forbidden in appended helpers')
            depth+=(masked[position]=='{')-(masked[position]=='}');position+=1
        if depth:raise ValueError('unbalanced appended C helper')
    if masked[position:].strip():raise ValueError('extra appended C code')


def verify_receipt(path,old_sources,new_sources):
    receipt=json.loads(Path(path).read_text())
    if (receipt.get('format')!=RECEIPT_FORMAT or receipt.get('status')!='complete'
        or any(receipt.get(k) is not True for k in ('generated_only','tests_passed','solver_unchanged','equality_verified'))
        or receipt.get('old_sources')!=old_sources or receipt.get('new_sources')!=new_sources):
        raise ValueError('complete exact-source storage equality receipt required')
    return receipt


def snapshot_tree(source,target):
    shutil.copytree(source,target)
    for path in target.rglob('*'):
        if path.is_file():
            with path.open('rb') as stream:os.fsync(stream.fileno())
    for path in [*sorted((p for p in target.rglob('*') if p.is_dir()),reverse=True),target]:
        fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)


def upgrade(bank,old_fastplan,old_c,old_runner,expected_old_fastplan,expected_old_c,
            expected_old_runner,validation_receipt):
    bank=Path(bank).resolve();archive=bank/ARCHIVE
    current=[LIVE_FASTPLAN.resolve(),LIVE_C.resolve(),LIVE_RUNNER.resolve()]
    snapshots=list(map(lambda p:Path(p).resolve(),(old_fastplan,old_c,old_runner)))
    expected=[expected_old_fastplan,expected_old_c,expected_old_runner]
    old_sources=dict(zip(map(str,current),expected));new_sources={str(p):sha(p) for p in current}
    validation_receipt=Path(validation_receipt).resolve()
    if archive.is_symlink():raise ValueError('archive symlink forbidden')
    if any(sha(p)!=h for p,h in zip(snapshots,expected)):raise ValueError('old snapshot SHA mismatch')
    verify_python(snapshots[0],current[0]);verify_c(snapshots[1],current[1])
    before,after=functions(snapshots[2]),functions(current[2])
    if any(before.get(name)!=after.get(name) or before.get(name) is None for name in PROTECTED):
        raise ValueError('protected runner gameplay/proof/job code changed')
    verify_receipt(validation_receipt,old_sources,new_sources)
    validation_sha=sha(validation_receipt);tool_sha=sha(__file__)
    validator_path=Path(__file__).with_name('upgrade_reference_scheduler.py')
    validator_sha=sha(validator_path)
    with (bank/'.regeneration.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt_path=archive/'upgrade.json'
        staging=bank/(ARCHIVE+'.preparing')
        if not archive.exists() and (staging/'upgrade.json').is_file():
            staged=json.loads((staging/'upgrade.json').read_text())
            if (staged.get('format')!=FORMAT or staged.get('status')!='prepared'
                or staged.get('old_sources')!=old_sources or staged.get('new_sources')!=new_sources
                or staged.get('validation_sha256')!=validation_sha or staged.get('migration_source_sha256')!=tool_sha
                or staged.get('validator_source_sha256')!=validator_sha):
                raise ValueError('prepared staging transaction source drift')
            for relative,expected_sha in staged['archive_sha256'].items():
                if sha(staging/relative)!=expected_sha:raise ValueError('prepared staging archive corruption')
            os.replace(staging,archive)
            fd=os.open(bank,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
        if archive.exists():
            if not receipt_path.is_file():raise ValueError('incomplete archive without transaction; manual inspection required')
            receipt=json.loads(receipt_path.read_text())
            if (receipt.get('format')!=FORMAT or receipt['old_sources']!=old_sources or receipt['new_sources']!=new_sources
                or receipt['validation_sha256']!=validation_sha or receipt['migration_source_sha256']!=tool_sha
                or receipt.get('validator_source_sha256')!=validator_sha):
                raise ValueError('transaction source/validation receipt drift')
            for relative,expected_sha in receipt['archive_sha256'].items():
                if sha(archive/relative)!=expected_sha:raise ValueError('archived source/job corruption')
            origin=archive
        else:origin=bank
        old=json.loads((origin/'manifest.json').read_text());report=json.loads((origin/'generation-report.json').read_text())
        if any(old['code_hashes'].get(p)!=h for p,h in old_sources.items()):raise ValueError('old snapshots not bound by old manifest')
        if (report.get('manifest_sha256')!=r._hash(old)
            or any(report.get(k)!=old[k] for k in ('config','code_hashes','source_hashes'))):
            raise ValueError('stopped report does not match old manifest binding')
        if report.get('workers_stopped') is not True:raise ValueError('old workers not confirmed stopped')
        pids=[report['pid'],*report.get('worker_pids',[])]
        for pid in pids:
            if type(pid) is not int or pid<=0 or Path(f'/proc/{pid}').exists():raise ValueError('old parent/worker PID exists or invalid')
        actual=r._code_hashes(old['config']['profile'])
        if (set(actual)!=set(old['code_hashes']) or any(actual[p]!=h for p,h in old['code_hashes'].items() if p not in old_sources)
            or any(actual.get(p)!=h for p,h in new_sources.items())):
            raise ValueError('unapproved code source changed')
        seeds,fingerprints,inventory=r._inventory(bank);seeds=set(seeds);fingerprints=set(fingerprints)
        if inventory!=old['source_hashes']:raise ValueError('source inventory changed')
        new=copy.deepcopy(old);new['code_hashes']=actual
        old_binding,new_binding=r._hash(old),r._hash(new)
        config=old['config'];jobs=[]
        for split in ('train','validation'):
            jobs+=r.jobs_for(config[split+'_count'],split,seeds,attempts=config['attempts'],limit=config['limit'],
                profile=config['profile'],quotas=config.get(split+'_quotas'),seed_range=config.get(split+'_seed_range'))
        expected_jobs={j['id']:j for j in jobs};states={};rows={};geometries={'train':set(),'validation':set()}
        accepted={'train':0,'validation':0}
        for path in sorted((origin/'jobs').glob('*.json')):
            if path.stem not in expected_jobs:raise ValueError('unknown checkpoint job')
            job=expected_jobs[path.stem];state=r._restore(path,job,old_binding);states[path.stem]=state
            if state['status']=='accepted':
                if r._accept(state['row'],job,seeds,fingerprints,geometries):raise ValueError('duplicate accepted checkpoint')
                accepted[job['split']]+=1;rows[path.stem]=r._hash(state['row'])
        if accepted!=report['accepted']:raise ValueError('stopped report accepted count differs from checkpoints')
        updated=copy.deepcopy(report);updated.update(manifest_sha256=new_binding,**new,
            fastplan_storage_upgrade=dict(receipt=str(receipt_path),validation_sha256=validation_sha,
                                         old_binding=old_binding,new_binding=new_binding))
        # Validate *every* live state before changing any file, including retries.
        if {p.stem for p in (bank/'jobs').glob('*.json')}!=set(states):raise ValueError('live job inventory differs from transaction')
        if json.loads((bank/'manifest.json').read_text()) not in (old,new):raise ValueError('live manifest differs from transaction')
        if json.loads((bank/'generation-report.json').read_text()) not in (report,updated):raise ValueError('live report differs from transaction')
        for name,state in states.items():
            envelope=json.loads((bank/'jobs'/(name+'.json')).read_text());payload=envelope['payload']
            if (envelope['sha256']!=r._hash(payload) or payload['format']!=r.CHECKPOINT_FORMAT
                or payload['job']!=expected_jobs[name] or payload['state']!=state or payload['binding'] not in (old_binding,new_binding)):
                raise ValueError('live job state differs from archived transaction')
        if not archive.exists():
            staging=bank/(ARCHIVE+'.preparing')
            if staging.exists():raise ValueError('unfinished archive preparation; inspect before retry')
            staging.mkdir();shutil.copyfile(bank/'manifest.json',staging/'manifest.json')
            shutil.copyfile(bank/'generation-report.json',staging/'generation-report.json')
            snapshot_tree(bank/'jobs',staging/'jobs')
            for source,name in zip(snapshots,('old-fastplan.py','old-fastplan.c','old-runner.py')):shutil.copyfile(source,staging/name)
            shutil.copyfile(__file__,staging/'upgrade_fastplan_storage.py')
            shutil.copyfile(validator_path,staging/'upgrade_reference_scheduler.py')
            shutil.copyfile(validation_receipt,staging/'validation.json')
            hashes={str(p.relative_to(staging)):sha(p) for p in staging.rglob('*') if p.is_file()}
            receipt=dict(format=FORMAT,status='prepared',old_sources=old_sources,new_sources=new_sources,
                validation_sha256=validation_sha,migration_source_sha256=tool_sha,validator_source_sha256=validator_sha,
                old_binding=old_binding,new_binding=new_binding,preserved_accepted=accepted,
                accepted_row_sha256=rows,archive_sha256=hashes)
            for path in staging.rglob('*'):
                if path.is_file():
                    with path.open('rb') as stream:os.fsync(stream.fileno())
            durable_json(staging/'upgrade.json',receipt)
            os.replace(staging,archive)
            fd=os.open(bank,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
        elif receipt['old_binding']!=old_binding or receipt['new_binding']!=new_binding or receipt['accepted_row_sha256']!=rows:
            raise ValueError('transaction manifest/row drift')
        if receipt['status'] not in ('prepared','complete'):raise ValueError('invalid transaction status')
        if receipt['status']=='complete':
            if json.loads((bank/'manifest.json').read_text())!=new or json.loads((bank/'generation-report.json').read_text())!=updated:
                raise ValueError('completed transaction changed')
            for name in states:r._restore(bank/'jobs'/(name+'.json'),expected_jobs[name],new_binding)
            return receipt
        # Binding/checksum are the only envelope fields rewritten.
        for name,state in states.items():
            payload=dict(format=r.CHECKPOINT_FORMAT,binding=new_binding,job=expected_jobs[name],state=state)
            durable_json(bank/'jobs'/(name+'.json'),dict(payload=payload,sha256=r._hash(payload)))
        durable_json(bank/'manifest.json',new);durable_json(bank/'generation-report.json',updated)
        for name,state in states.items():
            if r._restore(bank/'jobs'/(name+'.json'),expected_jobs[name],new_binding)!=state:raise ValueError('state changed after rebinding')
        if (r._inventory(bank)[2]!=inventory or r._code_hashes(config['profile'])!=actual
            or sha(validation_receipt)!=validation_sha or sha(__file__)!=tool_sha or sha(validator_path)!=validator_sha
            or any(sha(p)!=h for p,h in new_sources.items()) or any(sha(p)!=h for p,h in zip(snapshots,expected))):
            raise ValueError('source changed during transaction')
        receipt.update(status='complete',accepted_rows_unchanged=True,checkpoint_count=len(states))
        durable_json(receipt_path,receipt)
        return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upgrade-fastplan-storage',action='store_true',required=True)
    p.add_argument('--bank-dir',type=Path,required=True)
    for name in ('fastplan','c','runner'):
        p.add_argument('--old-'+name+'-snapshot',type=Path,required=True)
        p.add_argument('--expected-old-'+name+'-sha256',required=True)
    p.add_argument('--validation-receipt',type=Path,required=True)
    a=p.parse_args(argv)
    receipt=upgrade(a.bank_dir,a.old_fastplan_snapshot,a.old_c_snapshot,a.old_runner_snapshot,
        a.expected_old_fastplan_sha256,a.expected_old_c_sha256,a.expected_old_runner_sha256,a.validation_receipt)
    print(json.dumps({k:receipt[k] for k in ('status','preserved_accepted','accepted_rows_unchanged')}))

if __name__=='__main__':main()
