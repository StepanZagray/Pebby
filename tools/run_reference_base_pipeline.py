"""Durable sequential supervisor for an already-running reference-bank generation.

Never restarts generation or failed child stages. Only public generated validation
is played automatically. Fresh counterfactual training is not an H4/search model;
reset-distance and unsafe predictions still share the existing value head.
"""
import argparse
from contextlib import contextmanager
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
START=[.40,.25,.15,.09,.05,.04,.02]
END=[.04,.08,.14,.22,.23,.21,.08]
TRAIN=[3000,2500,1800,1300,1000,300,100]
VALIDATION=[100,100,80,70,60,50,40]


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def identity(pid):
    try:
        text=Path(f'/proc/{pid}/stat').read_text();values=text[text.rfind(')')+2:].split()
        return dict(pid=pid,start_ticks=int(values[19]),state=values[0])
    except (FileNotFoundError,ProcessLookupError):return None


def same_process(record):
    current=identity(record['pid'])
    return current is not None and current['start_ticks']==record['start_ticks'] and current['state']!='Z'


def code_sources():
    paths=[Path(__file__),*[ROOT/'tools'/name for name in ('audit_generated_banks.py',
        'build_reference_world_cache.py','preflight_reference_world.py','stream_extended_collection.py',
        'validate_extended_collector.py','merge_world_data.py')],ROOT/'third_party/ls20/ls20.py']
    paths+=list((ROOT/'pebby/agent').glob('*.py'))+list((ROOT/'pebby/ls20').glob('*.py'))+list((ROOT/'pebby/ls20').glob('*.c'))
    return {str(p.resolve()):digest(p) for p in paths}


def verify(hashes):
    for path,sha in hashes.items():
        if digest(path)!=sha:raise ValueError(f'bound source/output changed: {path}')



@contextmanager
def cache_read_lock(cache_dir):
    """Reject a live builder; hold its existing lock throughout adoption."""
    with (cache_dir/'.build.lock').open('r') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError as error:raise ValueError('cache builder is still active') from error
        yield


def validate_adopted_cache(bank_dir,cache_dir,audit_path):
    """Verify external completed receipts without inventing executed stages.

    Historical cache code is verified at its frozen snapshot paths, independently
    of the current training implementation. This performs disk/CPU checks only.
    """
    bindings={}
    def bind(path,expected=None):
        path=Path(path).resolve();key=str(path)
        actual=bindings.get(key)
        if actual is None:actual=digest(path);bindings[key]=actual
        if expected is not None and actual!=expected:raise ValueError(f'adopted hash mismatch: {path}')
        return actual
    def read(path):
        bind(path);return json.loads(Path(path).read_text())
    def bound_map(mapping):
        if not isinstance(mapping,dict) or not mapping:raise ValueError('missing adopted source hashes')
        for path,sha in mapping.items():
            if not Path(path).is_absolute() or not isinstance(sha,str) or len(sha)!=64:
                raise ValueError('malformed adopted source hash')
            bind(path,sha)
    with cache_read_lock(cache_dir):
        report=read(cache_dir/'build-report.json')
        if (report.get('status')!='complete' or report.get('sources_unchanged') is not True
                or report.get('workers_stopped') is not True):raise ValueError('complete stopped cache required')
        if report.get('source')!='generated_only' or report.get('official_frames_or_routes_used') is not False:
            raise ValueError('generated-only public cache required')
        manifest=read(cache_dir/'manifest.json')
        if manifest!={k:report[k] for k in ('sources','config')}:raise ValueError('cache manifest/report drift')
        config=manifest['config']
        if any(config.get(k)!=v for k,v in dict(history=8,samples=32,epsilon=.15,coverage='mixed_failure').items()):
            raise ValueError('cache collection configuration differs from fresh-fit contract')
        bound_map(manifest['sources'])
        builders=[Path(p) for p in manifest['sources'] if p.endswith('/tools/build_reference_world_cache.py')]
        if len(builders)!=1:raise ValueError('one frozen cache builder binding required')
        snapshot_root=builders[0].parents[1]
        snapshot=read(snapshot_root/'snapshot.json')
        if Path(snapshot['snapshot_root']).resolve()!=snapshot_root.resolve():raise ValueError('snapshot root mismatch')
        snapshot_sources={}
        for relative,sha in snapshot['sources'].items():
            path=(snapshot_root/relative).resolve()
            if not path.is_relative_to(snapshot_root.resolve()):raise ValueError('snapshot path escapes root')
            snapshot_sources[str(path)]=sha
        bound_map(snapshot_sources)
        for path,sha in manifest['sources'].items():
            if Path(path).is_relative_to(snapshot_root) and snapshot_sources.get(path)!=sha:
                raise ValueError('cache source differs from immutable snapshot manifest')
        generation_path=bank_dir/'generation-report.json';generation=read(generation_path)
        if generation.get('status')!='complete' or generation.get('workers_stopped') is not True:
            raise ValueError('complete stopped generation required')
        generation_manifest=read(bank_dir/'manifest.json')
        # The generator binds normalized JSON content, not the pretty-printed
        # file bytes. read() separately records the exact file hash for guard().
        manifest_binding=hashlib.sha256(json.dumps(generation_manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if manifest_binding!=generation['manifest_sha256']:
            raise ValueError('generation manifest content hash mismatch')
        if any(generation_manifest.get(k)!=generation.get(k) for k in ('config','source_hashes','code_hashes')):
            raise ValueError('generation manifest/report drift')
        bound_map(generation['source_hashes'])
        # Generation preceded runtime optimization. Check its code at the frozen
        # source-root-relative copy, never require the live training tree to be old.
        for path,sha in generation['code_hashes'].items():
            relative=Path(path).relative_to(snapshot['source_root'])
            frozen=str((snapshot_root/relative).resolve())
            if snapshot_sources.get(frozen)!=sha:raise ValueError('generation code absent/different in frozen snapshot')
        counts=check_bank_counts(bank_dir);banks={};bank_bindings={str(generation_path.resolve()):bind(generation_path)}
        for split in ('train','validation'):
            bank=bank_dir/(split+'.jsonl');sha=bind(bank,generation['banks'][split]['sha256'])
            bank_bindings[str(bank.resolve())]=sha
            banks[split]=[json.loads(line) for line in bank.read_text().splitlines() if line.strip()]
            if generation['banks'][split]['levels']!=len(banks[split]):raise ValueError('generation count mismatch')
        if any(manifest['sources'].get(path)!=sha for path,sha in bank_bindings.items()):
            raise ValueError('cache is bound to different banks/generation')
        audit=read(audit_path)
        if (audit.get('format')!='pebby.generated-bank-audit.v1' or audit.get('status')!='complete'
                or audit.get('errors')!=[] or audit.get('coverage_policy')!='full'
                or audit.get('required_validation_levels',0)<500 or audit.get('input_hashes_unchanged') is not True
                or audit.get('generation_evidence',{}).get('binding_verified') is not True):
            raise ValueError('completed full bank audit with generation binding required')
        if {str(Path(p).resolve()):sha for p,sha in audit['input_sha256'].items()}!=bank_bindings:
            raise ValueError('audit input bindings differ from adopted banks')
        bound_map(audit['audit_code_sha256'])
        for split in ('train','validation'):
            checks=audit.get('spotchecks',{}).get(split,[])
            if (len(checks)<7 or len({r['seed'] for r in checks})!=len(checks)
                    or not {r['seed'] for r in checks}<={r['seed'] for r in banks[split]}
                    or any(r.get('same_planner_search_agrees') is not True or r.get('engine_win_three_lives') is not True for r in checks)):
                raise ValueError('seven successful distinct generated spotchecks per split required')
        from pebby.agent.world_train import (load_dataset,disjoint_seeds,require_verified_data,
                                            require_winning_coverage,validate_successor_labels,
                                            REQUIRED_ARRAYS,OPTIONAL_ARRAYS)
        from tools.stream_extended_collection import validate_arrays
        datasets={}
        for split,specs in banks.items():
            entry=report['splits'][split];directory=cache_dir/split
            if entry.get('status')!='complete' or entry.get('requested_levels')!=len(specs) or entry.get('collected_levels')!=len(specs):
                raise ValueError('incomplete cache split coverage')
            proofs=[];shard_hashes={};first=0
            for record in entry['shards']:
                path=directory/f'shard-{first:06d}.npz';receipt=read(path.with_suffix('.json'))
                if record!=receipt or Path(record['path']).resolve()!=path.resolve():raise ValueError('shard receipt/report mismatch')
                selected=specs[first:first+config['shard_levels']]
                if not selected or record['seeds']!=[s['seed'] for s in selected]:raise ValueError('shard seed/order mismatch')
                sha=bind(path,record['sha256']);shard_hashes[str(path)]=sha
                data=load_dataset(path,history=8);observed=validate_arrays(data,selected)
                if any(record.get(k)!=v for k,v in observed.items()):raise ValueError('shard proof/count mismatch')
                if data['meta'].get('split')!=split or data['meta'].get('bank_sha256')!=bank_bindings[str((bank_dir/(split+'.jsonl')).resolve())]:
                    raise ValueError('shard bank binding mismatch')
                proofs.extend(data['meta']['levels']);first+=len(selected);del data
            if first!=len(specs):raise ValueError('shards do not cover complete bank')
            validity_path=directory/'validity.json';validity=read(validity_path)
            if validity!={'status':'complete','levels':proofs}:raise ValueError('validity receipt differs from shard proofs')
            merged=cache_dir/(split+'.npz');receipt=read(directory/'merged.json')
            sha=bind(merged,entry['sha256'])
            if receipt.get('sha256')!=sha or Path(entry['output']).resolve()!=merged.resolve():raise ValueError('merged publication mismatch')
            # load_dataset checks every extracted NPY hash and its NPZ binding;
            # a completed builder has already populated this bounded mmap cache.
            names=sorted(set((*REQUIRED_ARRAYS,*OPTIONAL_ARRAYS,'context_index','meta')))
            schema=hashlib.sha256(json.dumps(names).encode()).hexdigest()[:16]
            mmap_root=cache_dir/'array-cache'/f'{sha}-{schema}'
            mmap_manifest=read(mmap_root/'manifest.json')
            if mmap_manifest.get('source_sha256')!=sha:raise ValueError('mmap source binding mismatch')
            data=load_dataset(merged,history=8,cache_dir=cache_dir/'array-cache')
            for name,info in mmap_manifest['arrays'].items():
                if name not in names:raise ValueError('unexpected mmap array')
                # load_dataset has just verified these exact NPY hashes.
                bindings[str((mmap_root/(name+'.npy')).resolve())]=info['sha256']
            if (data['meta'].get('source_sha256')!=shard_hashes
                    or data['meta'].get('audit_sha256')!=bind(validity_path)
                    or data['meta'].get('split')!=split):raise ValueError('merged shard/proof lineage mismatch')
            merged_proofs=data['meta'].get('levels',[])
            if [p['seed'] for p in merged_proofs]!=sorted(p['seed'] for p in proofs):
                raise ValueError('merged proof coverage/order mismatch')
            originals={p['seed']:p for p in proofs}
            if any(any(p.get(k)!=v for k,v in originals[p['seed']].items() if k!='samples') for p in merged_proofs):
                raise ValueError('merged proof differs from shard source')
            require_verified_data(data);require_winning_coverage(data,split);validate_successor_labels(data,split)
            if set(map(int,data['seeds']))!={s['seed'] for s in specs}:raise ValueError('merged seed coverage differs from bank')
            if len(data['seeds'])!=entry['rows'] or int(data['distances'].max())!=entry['max_distance']:
                raise ValueError('full cache row/distance summary mismatch')
            datasets[split]=data
        disjoint_seeds(datasets['train'],datasets['validation'])
        return report,dict(origin='adopted',inputs=bindings,bank_counts=counts,
                           audit=str(audit_path),cache_report=str(cache_dir/'build-report.json'))


def distance_capacity(report):
    if report.get('status')!='complete' or report.get('sources_unchanged') is not True:raise ValueError('complete verified cache report required')
    high=max(report['splits'][s]['max_distance'] for s in ('train','validation'))
    if type(high) is not int or high<0:raise ValueError('full cache finite distance maximum required')
    return max(128,1 << high.bit_length())


def check_bank_counts(bank_dir):
    counts={};seeds=set()
    for split,expected in [('train',TRAIN),('validation',VALIDATION)]:
        rows=[json.loads(line) for line in (bank_dir/(split+'.jsonl')).read_text().splitlines() if line.strip()]
        if any(r.get('split')!=split or r.get('difficulty_version')!='ls20-reference-v1' for r in rows):raise ValueError('bank split/version mismatch')
        values=[r['seed'] for r in rows]
        if len(set(values))!=len(values) or seeds.intersection(values):raise ValueError('bank duplicate/overlapping seeds')
        seeds.update(values);by=Counter(r['difficulty'] for r in rows)
        if [by[t] for t in range(1,8)]!=expected:raise ValueError('final bank differs from authorized unequal quotas')
        counts[split]=dict(by)
    # Largest-remainder rounding is <=ceil(quota); checking endpoint ceilings
    # conservatively covers every intermediate linear curriculum allocation.
    import math
    for tier,available in enumerate(TRAIN):
        need=math.ceil(max(START[tier],END[tier])*1024)
        if available<need:raise ValueError(f'tier{tier+1} lacks B1024 distinct levels')
    return counts


def train_arguments(probe,cache,checkpoint,report):
    if (probe.get('status')!='complete' or probe.get('smoke') is not False
            or probe.get('fresh_initialization') is not True or probe.get('source_unchanged') is not True
            or probe.get('checkpoint_saved') is not False
            or probe.get('requires_fresh_fit_initialization') is not True):raise ValueError('verified disposable CUDA probe required')
    batch=probe['selected_batch_size']
    if type(batch) is not int or not 1<=batch<=1024 or batch&(batch-1):raise ValueError('invalid selected power-of-two batch')
    success=[a for a in probe['attempts'] if a['status']=='complete']
    if len(success)!=1 or success[0]['batch_size']!=batch:raise ValueError('probe selection witness mismatch')
    initial_hash=success[0].get('initial_weights_sha256')
    if not isinstance(initial_hash,str) or len(initial_hash)!=64 or any(c not in '0123456789abcdef' for c in initial_hash):
        raise ValueError('verified random initialization fingerprint required')
    if probe['precision']!='bf16' or any(type(probe[k]) is not bool for k in ('checkpoint_encoder','checkpoint_loops')):
        raise ValueError('probe memory/precision mismatch')
    if any(len(probe['curriculum'].get(k,[]))!=7 or any(not math.isclose(a,b,rel_tol=0,abs_tol=1e-12)
            for a,b in zip(probe['curriculum'][k],values)) for k,values in [('start',START),('end',END)]):
        raise ValueError('probe curriculum differs from fixed schedule')
    config=probe['config']
    if config.get('cell_recall') or config['history']!=8:raise ValueError('fresh H8 base configuration required')
    args=['--train',str(cache/'train.npz'),'--validation',str(cache/'validation.npz'),
        '--data-cache-dir',str(cache/'array-cache'),'--checkpoint-out',str(checkpoint),'--report-out',str(report),
        '--epochs','10','--batch-size',str(batch),'--device','cuda','--precision',probe['precision'],
        '--seed',str(probe['seed']),'--lr',str(probe['optimizer']['lr']),
        '--weight-decay',str(probe['optimizer']['weight_decay']), '--select-on','last',
        '--require-verified-data','--require-winning-coverage','--require-exact-distances',
        '--min-train-levels','10000','--drop-last','--curriculum','--require-fresh-initialization',
        '--expected-initial-state-sha256',initial_hash,
        '--encoder-chunk-size',str(probe['encoder_chunk_size']),
        '--curriculum-start',*map(str,START),'--curriculum-end',*map(str,END)]
    for name in ('checkpoint_encoder','checkpoint_loops'):
        if probe[name]:args.append('--'+name.replace('_','-'))
    execution=probe.get('execution',dict(compile_core=False,temporal_backend='auto'))
    if type(execution['compile_core']) is not bool or execution['temporal_backend'] not in ('auto','math','cudnn','flash'):
        raise ValueError('invalid probed execution options')
    if execution['compile_core']:args.append('--compile-core')
    args.extend(('--temporal-backend',execution['temporal_backend']))
    for key,value in config.items():
        if key=='architecture':continue
        flag='--'+key.replace('_','-')
        if isinstance(value,bool):
            if value:args.append(flag)
        else:args.extend((flag,str(value)))
    for key,value in probe['weights'].items():args.extend(('--'+key.replace('_','-')+'-weight',str(value)))
    return args


def gpu_idle():
    check=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True,timeout=15)
    if check.stdout.strip():raise RuntimeError('GPU has an active compute process; refusing concurrent stage')


def child_environment(args):
    environment=os.environ.copy()
    environment.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    allocator=getattr(args,'allocator_conf',None)
    if allocator is not None:
        environment['PYTORCH_ALLOC_CONF']=allocator
        environment.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    threads=getattr(args,'compile_threads',None)
    if threads is not None:environment['TORCHINDUCTOR_COMPILE_THREADS']=str(threads)
    return environment


class Supervisor:
    def __init__(self,args):
        self.args=args;self.path=args.out_dir/'pipeline.json';self.child=None
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='resume' and not (k=='adopt_cache_audit' and v is None)}
        hashes=code_sources()
        if self.path.exists():
            if not args.resume:raise ValueError('existing supervisor requires explicit --resume')
            self.report=json.loads(self.path.read_text())
            if self.report['config']!=config or self.report['sources']!=hashes:raise ValueError('supervisor resume config/source drift')
            if any(s['status']!='complete' for s in self.report['stages'].values()):raise ValueError('incomplete child stage requires inspection, never blind restart')
        else:
            adopting=getattr(args,'adopt_cache_audit',None) is not None
            expected=None if adopting else dict(pid=args.generation_pid,start_ticks=args.generation_start_ticks)
            if not adopting and not same_process(expected):raise ValueError('exact running generation PID/start-ticks required for first launch')
            self.report=dict(status='adopting_cache' if adopting else 'waiting_generation',pid=os.getpid(),config=config,sources=hashes,
                generation_identity=expected,stages={},official_evaluation=False,
                limits=['One-step counterfactual dynamics; no H4 sequence supervision or search.',
                        'Actual-reset distance and imagined unsafe labels share the existing value head.',
                        'Fresh training is authorized; success on all seven official levels remains unestablished.'])
        self.persist()
    def persist(self):write(self.path,self.report)
    def guard(self):
        verify(self.report['sources'])
        verify(self.report.get('generation_outputs',{}))
        verify(self.report.get('adoption',{}).get('inputs',{}))
        for stage in self.report['stages'].values():
            if stage['status']=='complete':verify(stage.get('outputs',{}))
    def stop_child(self):
        if self.child is None:return
        child=self.child
        if child.poll() is None:
            # Dedicated session/process group contains only this stage and its workers.
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        self.child=None
    def stage(self,name,module,arguments,timeout,outputs,gpu=False):
        if name in self.report['stages']:return
        self.guard()
        if gpu:gpu_idle()
        if any(Path(p).exists() for p in outputs):raise ValueError(f'{name}: output already exists without completed stage receipt')
        command=[sys.executable,'-m',module,*arguments];log=self.args.out_dir/(name+'.log')
        environment=child_environment(self.args)
        tick=time.monotonic()
        with log.open('x') as stream:
            self.child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            record=dict(status='running',command=command,identity=identity(self.child.pid),log=str(log),timeout_seconds=timeout)
            self.report['stages'][name]=record;self.report['status']=name;self.persist()
            try:
                code=self.child.wait(timeout=timeout)
                if code:raise RuntimeError(f'{name} exited {code}; see {log}')
                verify(self.report['sources'])
                record.update(status='complete',exit_code=code,outputs={str(p):digest(p) for p in outputs},
                              seconds=time.monotonic()-tick,child_reaped=True)
            except BaseException as error:
                record.update(status='failed',error=str(error),seconds=time.monotonic()-tick);raise
            finally:
                self.stop_child()
                record['child_reaped']=not same_process(record['identity'])
                self.persist()
    def wait_generation(self):
        path=self.args.bank_dir/'generation-report.json';deadline=time.monotonic()+self.args.generation_wait_seconds
        while True:
            record=json.loads(path.read_text())
            if record.get('pid')!=self.args.generation_pid:raise ValueError('generation report belongs to another PID')
            alive=same_process(self.report['generation_identity'])
            if not alive:
                if record.get('status')!='complete' or record.get('workers_stopped') is not True:raise ValueError('generation ended without complete verified banks')
                break
            if record.get('status')=='failed_closed':raise RuntimeError('generation failed; supervisor will not restart it')
            if time.monotonic()>=deadline:raise TimeoutError('bounded generation wait expired; generation left untouched')
            self.report['generation_progress']=record.get('accepted');self.persist();time.sleep(30)
        self.guard();counts=check_bank_counts(self.args.bank_dir)
        for split in ('train','validation'):
            bank=self.args.bank_dir/(split+'.jsonl')
            if digest(bank)!=record['banks'][split]['sha256']:raise ValueError('completed bank hash mismatch')
        bindings={str(p):digest(p) for p in [path,*[self.args.bank_dir/(s+'.jsonl') for s in ('train','validation')]]}
        if self.report.get('generation_outputs',bindings)!=bindings:raise ValueError('generation changed since prior supervisor receipt')
        self.report.update(generation_report_sha256=digest(path),generation_outputs=bindings,bank_counts=counts);self.persist()
    def run(self):
        a=self.args
        if getattr(a,'adopt_cache_audit',None) is not None:
            if getattr(a,'cache_wait_seconds',0):
                deadline=time.monotonic()+a.cache_wait_seconds
                expected=dict(pid=a.cache_pid,start_ticks=a.cache_start_ticks)
                while True:
                    self.guard()
                    cache=json.loads((a.cache_dir/'build-report.json').read_text())
                    if cache.get('pid')!=a.cache_pid:raise ValueError('cache report belongs to another process')
                    if not same_process(expected):
                        if cache.get('status')!='complete' or cache.get('workers_stopped') is not True:
                            raise ValueError('cache process ended without complete verified output')
                        break
                    if cache.get('status')=='failed':raise RuntimeError('cache failed; no automatic restart')
                    if time.monotonic()>=deadline:raise TimeoutError('cache wait expired; builder left untouched')
                    self.report.update(status='waiting_cache',cache_identity=expected,
                        cache_progress={k:v.get('collected_levels',0) for k,v in cache.get('splits',{}).items()})
                    self.persist();time.sleep(30)
            self.guard()
            cache,adoption=validate_adopted_cache(a.bank_dir,a.cache_dir,a.adopt_cache_audit)
            if self.report.get('adoption',adoption)!=adoption:raise ValueError('adopted inputs changed since receipt')
            self.report['adoption']=adoption;self.persist()
            return self.run_training(cache)
        self.wait_generation();generation=a.bank_dir/'generation-report.json'
        audit=a.out_dir/'bank-audit.json'
        self.stage('audit','tools.audit_generated_banks',['--train',str(a.bank_dir/'train.jsonl'),
            '--validation',str(a.bank_dir/'validation.jsonl'),'--generation-report',str(generation),
            '--min-validation','500','--coverage-policy','full','--spotcheck','7','--report',str(audit)],a.audit_seconds,[audit])
        if json.loads(audit.read_text())['status']!='complete':raise ValueError('bank audit failed')
        cache_report=a.cache_dir/'build-report.json'
        self.stage('cache','tools.build_reference_world_cache',['--bank-dir',str(a.bank_dir),'--out-dir',str(a.cache_dir),
            '--workers','16','--shard-levels','100','--samples-per-level','32','--seconds',str(a.cache_seconds)],
            a.cache_seconds+30,[cache_report,a.cache_dir/'train.npz',a.cache_dir/'validation.npz'])
        return self.run_training(json.loads(cache_report.read_text()))

    def run_training(self,cache):
        a=self.args;maximum=distance_capacity(cache)
        from tools.preflight_reference_world import fresh_config
        config=fresh_config().__dict__;config['max_distance']=maximum
        config_path=a.out_dir/'fresh-config.json'
        if config_path.exists() and json.loads(config_path.read_text())!=config:raise ValueError('fresh configuration drift')
        write(config_path,config)
        probe_path=a.out_dir/'preflight.json'
        runtime_flags=['--temporal-backend',a.temporal_backend,
                       '--checkpoint-encoder' if a.checkpoint_encoder else '--no-checkpoint-encoder',
                       '--checkpoint-loops' if a.checkpoint_loops else '--no-checkpoint-loops']
        if a.compile_core:runtime_flags.append('--compile-core')
        self.stage('preflight','tools.preflight_reference_world',['--train',str(a.cache_dir/'train.npz'),
            '--validation',str(a.cache_dir/'validation.npz'),'--data-cache-dir',str(a.cache_dir/'array-cache'),
            '--config',str(config_path),'--out',str(probe_path),'--max-batch','1024','--device','cuda',
            '--seed','42','--encoder-chunk-size',str(a.encoder_chunk_size),'--seconds','1800',
            '--curriculum-start',*map(str,START),'--curriculum-end',*map(str,END),*runtime_flags],1830,[probe_path],gpu=True)
        probe=json.loads(probe_path.read_text());training=a.out_dir/'training.json'
        environment=child_environment(a);execution=probe['execution']
        if (execution['compile_core']!=a.compile_core or execution['temporal_backend']!=a.temporal_backend
                or execution['allocator_environment']!={key:environment.get(key) for key in
                    ('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF')}
                or execution['compile_threads']!=environment.get('TORCHINDUCTOR_COMPILE_THREADS')
                or any(probe[key]!=getattr(a,key) for key in ('encoder_chunk_size','checkpoint_encoder','checkpoint_loops'))):
            raise ValueError('preflight runtime differs from selected execution configuration')
        arguments=train_arguments(probe,a.cache_dir,a.checkpoint,training)
        last=a.checkpoint.with_name(a.checkpoint.stem+'.last.pt')
        self.stage('train','pebby.agent.world_train',arguments,a.train_seconds,[training,a.checkpoint,last],gpu=True)
        train=json.loads(training.read_text())
        if len(train['history'])!=10 or train['best']['epoch']!=10 or {k:v for k,v in train['config'].items() if k!='architecture'}!=probe['config'] or train['loss_weights']!=probe['weights'] or train.get('execution')!=execution:
            raise ValueError('fresh fixed-last training differs from preflight contract')
        initial_hash=next(attempt['initial_weights_sha256'] for attempt in probe['attempts'] if attempt['status']=='complete')
        expected_initialization=dict(kind='random',seed=probe['seed'],weights_sha256=initial_hash,optimizer_state='new')
        if train.get('initialization')!=expected_initialization:
            raise ValueError('training did not start from the independently fingerprinted random initialization')
        import torch
        saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
        if (saved.get('initialize_checkpoint') is not None or saved.get('initialize_glyph_checkpoint') is not None
                or saved.get('cell_source') is not None or saved.get('epochs')!=10 or saved.get('best_epoch')!=10
                or saved.get('select_on')!='last' or saved.get('batch_size')!=probe['selected_batch_size']
                or saved.get('encoder_chunk_size')!=probe['encoder_chunk_size']
                or saved.get('precision')!=probe['precision'] or saved.get('loss_weights')!=probe['weights']
                or saved.get('checkpoint_encoder')!=probe['checkpoint_encoder']
                or saved.get('checkpoint_loops')!=probe['checkpoint_loops'] or saved.get('execution')!=execution
                or saved.get('initialization')!=expected_initialization):
            raise ValueError('saved model is not the exact fresh probe configuration')
        del saved
        evaluation=a.out_dir/'generated-validation500.json'
        self.stage('evaluate','pebby.agent.evaluate',['--checkpoint',str(a.checkpoint),'--bank',str(a.bank_dir/'validation.jsonl'),
            '--limit','500','--device','cuda','--max-actions','300','--protocol','strict','--temperature','0',
            '--on-stall','repeat','--report-out',str(evaluation)],a.evaluate_seconds,[evaluation],gpu=True)
        result=json.loads(evaluation.read_text())
        if result.get('levels')!=500 or result.get('protocol')!='strict':raise ValueError('incomplete strict generated evaluation')
        self.guard();self.report.update(status='complete',generated_completion=result['completed']);self.persist()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generation-pid',type=int);p.add_argument('--generation-start-ticks',type=int)
    p.add_argument('--adopt-cache-audit',type=Path,help='Adopt a complete external cache and its completed full bank audit; never restart generation/cache.')
    p.add_argument('--cache-pid',type=int)
    p.add_argument('--cache-start-ticks',type=int)
    p.add_argument('--cache-wait-seconds',type=int,default=0)
    p.add_argument('--checkpoint-encoder',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--checkpoint-loops',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--encoder-chunk-size',type=int,default=128)
    p.add_argument('--compile-core',action='store_true')
    p.add_argument('--temporal-backend',choices=('auto','math','cudnn','flash'),default='auto')
    p.add_argument('--allocator-conf',choices=('expandable_segments:True',))
    p.add_argument('--compile-threads',type=int,default=1)
    p.add_argument('--bank-dir',type=Path,default=ROOT/'data/ls20-reference-unequal-v1')
    p.add_argument('--cache-dir',type=Path,default=ROOT/'data/reference-world-base-v1')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/ls20-reference-base-v1.pt')
    p.add_argument('--out-dir',type=Path,default=ROOT/'artifacts/reference-base-pipeline-v1');p.add_argument('--resume',action='store_true')
    for flag,default in [('generation-wait-seconds',86400),('audit-seconds',7200),('cache-seconds',43200),('train-seconds',86400),('evaluate-seconds',7200)]:
        p.add_argument('--'+flag,type=int,default=default)
    args=p.parse_args(argv)
    if args.encoder_chunk_size<1 or not 1<=args.compile_threads<=4:p.error('positive chunk and compile threads1..4 required')
    if args.adopt_cache_audit is None and (args.generation_pid is None or args.generation_start_ticks is None):
        p.error('live mode requires --generation-pid and --generation-start-ticks')
    if args.adopt_cache_audit is not None:
        if args.generation_pid is not None or args.generation_start_ticks is not None:p.error('adoption cannot also specify a live generation identity')
        args.adopt_cache_audit=args.adopt_cache_audit.resolve()
    if args.cache_wait_seconds:
        if not args.adopt_cache_audit or not args.cache_pid or not args.cache_start_ticks:
            p.error('cache waiting requires adoption and an exact cache PID/start-ticks')
    elif args.cache_pid is not None or args.cache_start_ticks is not None:
        p.error('cache identity requires a positive --cache-wait-seconds')
    if args.cache_wait_seconds<0 or any(v<1 for k,v in vars(args).items() if k.endswith('seconds') and k!='cache_wait_seconds'):
        p.error('positive stage bounds required')
    for key in ('bank_dir','cache_dir','checkpoint','out_dir'):setattr(args,key,getattr(args,key).resolve())
    args.out_dir.mkdir(parents=True,exist_ok=args.resume)
    lock=(args.out_dir/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    runner=None
    def stopped(*_):raise InterruptedError('supervisor interrupted')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stopped)
    try:
        runner=Supervisor(args);runner.run()
    except BaseException as error:
        if runner is not None:runner.stop_child();runner.report.update(status='failed_closed',error=str(error));runner.persist()
        raise
    finally:lock.close()


if __name__=='__main__':main()
