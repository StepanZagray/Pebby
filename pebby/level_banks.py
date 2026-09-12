"""Read-only catalogue of checksum-bound accepted reference levels.

No generation, Oracle search, Torch, or user-supplied filesystem paths. Checks
verify recorded proofs and source integrity; they do not re-prove optimality.
"""
import copy
import hashlib
import json
from pathlib import Path
import re
import threading

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANKS = {'reference-unequal-v1': {'label': 'Seven-reference unequal bank',
                'path': ROOT / 'data/ls20-reference-unequal-v1'}}
JOB_FORMAT = 'pebby.bank-regeneration.checkpoint.v1'
# Derived read cache beside the bank. Deliberately not *.jsonl: the generator
# rglobs data/**/*.jsonl as seed inventory and would ingest this file.
INDEX_NAME = 'row-index.json'
INDEX_FORMAT = 'pebby.bank-read-index.v1'


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def signature(path):
    stat=path.stat()
    return (stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)


def read_json(path):
    if path.stat().st_size > 2*1024*1024: raise ValueError('bank metadata exceeds size limit')
    return json.loads(path.read_text())


def integer(value,minimum,maximum,name):
    if type(value) is not int or not minimum<=value<=maximum:
        raise ValueError(f'{name} must be an integer in {minimum}..{maximum}')


class LevelBanks:
    def __init__(self,banks=None):
        self.banks=copy.deepcopy(DEFAULT_BANKS if banks is None else banks)
        self.lock=threading.RLock(); self.cache={}; self.file_hashes={}; self.job_plans={}

    def _hash(self,path):
        path=Path(path); stamp=signature(path)
        cached=self.file_hashes.get(str(path))
        if cached and cached[0]==stamp:return cached[1]
        with path.open('rb') as stream:value=hashlib.file_digest(stream,'sha256').hexdigest()
        if signature(path)!=stamp:raise ValueError('bank source changed while reading')
        self.file_hashes[str(path)]=(stamp,value)
        return value

    def _job(self,path,manifest,binding):
        # Reuse the generator's cheap proof/geometry checks, not its generation
        # or search entry points. This import does not import Torch.
        from tools.regenerate_mechanism_banks import _accept, jobs_for, FORMAT
        before=signature(path); envelope=read_json(path)
        payload=envelope['payload']
        if envelope['sha256']!=canonical_hash(payload) or payload['format']!=JOB_FORMAT or payload['binding']!=binding:
            raise ValueError('bank job checksum or manifest binding mismatch')
        job=payload['job'];state=payload['state'];cfg=manifest['config']
        match=re.fullmatch(r'(train|validation)-(\d{6})',path.stem)
        if not match or job['id']!=path.stem or job['split']!=match[1]:raise ValueError('bank job identity mismatch')
        split=match[1]; ordinal=int(match[2]); count=cfg[split+'_count']
        integer(ordinal,0,count-1,'job ordinal')
        if binding not in self.job_plans:
            occupied=set()
            for source in manifest['source_hashes']:
                source=Path(source)
                if source.suffix!='.jsonl':continue
                with source.open() as stream:
                    for line in stream:
                        if not line.strip():continue
                        prior=json.loads(line)
                        if isinstance(prior,dict) and prior.get('format')==FORMAT:
                            occupied.add(int(prior['seed']))
            expected={}
            for part in ('train','validation'):
                jobs=jobs_for(cfg[part+'_count'],part,occupied,attempts=cfg['attempts'],
                    limit=cfg['limit'],profile=cfg['profile'],quotas=cfg.get(part+'_quotas'),
                    seed_range=cfg.get(part+'_seed_range'))
                expected.update({value['id']:value for value in jobs})
            self.job_plans[binding]=expected
        if job!=self.job_plans[binding].get(path.stem):
            raise ValueError('bank job differs from exact manifest/inventory assignment')
        if type(state['next_seed']) is not int or not job['seed']<=state['next_seed']<=job['seed']+64:
            raise ValueError('bank job retry cursor invalid')
        if state['status'] not in ('pending','accepted','rejected','quarantined'):
            raise ValueError('invalid bank job status')
        if state['status']!='accepted':return None
        row=state['row']
        reason=_accept(row,job,set(),set(),{'train':set(),'validation':set()})
        if reason:raise ValueError('bank accepted row failed proof validation')
        if signature(path)!=before:raise ValueError('bank job changed while reading')
        return row

    def _index_path(self,directory):return Path(directory)/INDEX_NAME

    def _load_index(self,directory,binding):
        """Restore a previously validated cache, or None when it cannot be trusted.

        Everything in the index was produced by a full validation of this exact
        binding, so restoring it is equivalent to redoing that work. Anything
        that fails to parse, fails its own checksum, or names a different
        binding is discarded and rebuilt rather than repaired.
        """
        path=self._index_path(directory)
        try:
            if not path.is_file() or path.stat().st_size>256*1024*1024:return None
            blob=json.loads(path.read_text())
            if blob.get('format')!=INDEX_FORMAT:return None
            index=blob['index']
            if canonical_hash(index)!=blob['sha256'] or index['binding']!=binding:return None
            jobs={key:tuple(value) for key,value in index['jobs'].items()}
            summaries=index['summaries']
            if not set(summaries)<=set(jobs):return None
            final=index['final']
            if final is not None:
                final=(tuple(final[0]),tuple((split,tuple(stamp)) for split,stamp in final[1]),
                       tuple(sorted(jobs.items())))
            # Restoring the plan matters as much as the summaries: without it the
            # first level() re-derives it by rescanning the whole seed inventory.
            plan=index['job_plan']
            if plan is not None:self.job_plans[binding]=plan
            return {'binding':binding,'jobs':jobs,'summaries':summaries,'final':final}
        except (OSError,KeyError,IndexError,TypeError,ValueError):
            return None

    def _save_index(self,directory,cache):
        """Persist a fully validated cache. Never fatal: serving must not need a writable bank."""
        final=cache['final']
        index={'binding':cache['binding'],'jobs':{key:list(value) for key,value in cache['jobs'].items()},
               'summaries':cache['summaries'],'job_plan':self.job_plans.get(cache['binding']),
               'final':None if final is None else
                       [list(final[0]),[[split,list(stamp)] for split,stamp in final[1]]]}
        blob={'format':INDEX_FORMAT,'sha256':canonical_hash(index),'index':index}
        path=self._index_path(directory);tmp=path.with_name(INDEX_NAME+'.tmp')
        try:
            tmp.write_text(json.dumps(blob,separators=(',',':')));tmp.replace(path)
        except OSError:
            try:tmp.unlink()
            except OSError:pass

    def _refresh(self,bank):
        if not isinstance(bank,str) or bank not in self.banks:raise ValueError('unknown bank')
        spec=self.banks[bank]; directory=Path(spec['path']); manifest_path=directory/'manifest.json'
        if not manifest_path.is_file():return dict(id=bank,label=spec['label'],status='unavailable',
            accepted={'train':0,'validation':0},requested=None),None
        manifest=read_json(manifest_path);binding=canonical_hash(manifest);cfg=manifest['config']
        if cfg.get('profile')!='ls20-reference-v1':raise ValueError('unsupported bank profile')
        for split in ('train','validation'):
            integer(cfg[split+'_count'],1,1000000,'requested levels')
            quotas=cfg.get(split+'_quotas')
            if quotas is not None and (len(quotas)!=7 or any(type(n) is not int or n<1 for n in quotas)
                                       or sum(quotas)!=cfg[split+'_count']):raise ValueError('invalid bank quotas')
        for name in ('code_hashes','source_hashes'):
            if not isinstance(manifest[name],dict) or not manifest[name]:raise ValueError('bank source binding missing')
            for path,sha in manifest[name].items():
                if self._hash(path)!=sha:raise ValueError('bank source integrity check failed')
        cache=self.cache.get(bank)
        if cache is None or cache['binding']!=binding:
            cache=self._load_index(directory,binding);dirty=cache is None
            if cache is None:cache={'binding':binding,'jobs':{},'summaries':{},'final':None}
            self.cache[bank]=cache
        else:dirty=False
        paths=sorted((directory/'jobs').glob('*.json'))
        present={p.stem for p in paths}
        for missing in set(cache['jobs'])-present:
            del cache['jobs'][missing];cache['summaries'].pop(missing,None);dirty=True
        for path in paths:
            stamp=signature(path)
            if cache['jobs'].get(path.stem)==stamp:continue
            row=self._job(path,manifest,binding)
            cache['summaries'].pop(path.stem,None)
            if row is not None:
                from pebby.ls20.extended_curriculum import gameplay_hash
                cache['summaries'][path.stem]=dict(id=path.stem,seed=row['seed'],difficulty=row['difficulty'],
                    split=row['split'],optimal_actions=row['optimal_actions'],fog=row['fog'],goals=len(row['goals']),
                    row_sha256=canonical_hash(row),geometry_sha256=row['geometry_sha256'],gameplay_sha256=gameplay_hash(row))
            cache['jobs'][path.stem]=stamp;dirty=True
        values=list(cache['summaries'].values())
        if len({v['seed'] for v in values})!=len(values) or len({v['gameplay_sha256'] for v in values})!=len(values):
            raise ValueError('duplicate accepted bank levels')
        geometries={split:{v['geometry_sha256'] for v in values if v['split']==split} for split in ('train','validation')}
        if geometries['train'] & geometries['validation']:raise ValueError('cross-split bank geometry overlap')
        accepted={s:sum(v['split']==s for v in values) for s in ('train','validation')}
        requested={s:cfg[s+'_count'] for s in accepted}
        status='partial';report_path=directory/'generation-report.json'
        if report_path.exists():
            report=read_json(report_path)
            if report.get('status')=='complete':
                final_signature=(signature(report_path),tuple((s,signature(directory/(s+'.jsonl'))) for s in accepted),
                                 tuple(sorted(cache['jobs'].items())))
                if cache['final']!=final_signature:
                    if report.get('manifest_sha256')!=binding or accepted!=requested:
                        raise ValueError('completed bank manifest/count mismatch')
                    for s in accepted:
                        info=report['banks'][s];path=directory/(s+'.jsonl')
                        if info['levels']!=accepted[s] or self._hash(path)!=info['sha256']:
                            raise ValueError('completed bank publication hash mismatch')
                        expected={v['row_sha256'] for v in values if v['split']==s}
                        observed=[]
                        with path.open() as stream:
                            for line in stream:
                                if len(line)>2*1024*1024:raise ValueError('bank row exceeds size limit')
                                if line.strip():observed.append(canonical_hash(json.loads(line)))
                        if len(observed)!=len(expected) or set(observed)!=expected:
                            raise ValueError('final bank differs from accepted checkpoints')
                    cache['final']=final_signature;dirty=True
                status='complete'
        if canonical_hash(read_json(manifest_path))!=binding:raise ValueError('bank manifest changed during read')
        if dirty:self._save_index(directory,cache)
        return dict(id=bank,label=spec['label'],status=status,accepted=accepted,requested=requested), (cache,manifest,directory)

    def catalogue(self):
        with self.lock:
            try:return {'banks':[self._refresh(bank)[0] for bank in self.banks]}
            except (OSError,KeyError,TypeError,ValueError) as error:
                raise ValueError('Bank integrity check failed; catalogue unavailable.') from error

    def levels(self,bank,split='train',difficulty=None,offset=0,limit=50):
        if split not in ('train','validation'):raise ValueError('split must be train or validation')
        if difficulty is not None:integer(difficulty,1,7,'difficulty')
        integer(offset,0,1000000,'offset');integer(limit,1,100,'limit')
        with self.lock:
            try:
                info,state=self._refresh(bank)
                values=[] if state is None else [v for v in state[0]['summaries'].values()
                    if v['split']==split and (difficulty is None or v['difficulty']==difficulty)]
                values.sort(key=lambda v:v['id'])
                return dict(bank=bank,bank_status=info['status'],total=len(values),offset=offset,
                            levels=copy.deepcopy(values[offset:offset+limit]))
            except (OSError,KeyError,TypeError,ValueError) as error:
                raise ValueError('Bank integrity check failed or bank is unknown.') from error

    def level(self,bank,row_id):
        if not isinstance(row_id,str) or not re.fullmatch(r'(train|validation)-\d{6}',row_id):
            raise ValueError('invalid bank row id')
        with self.lock:
            try:
                info,state=self._refresh(bank)
                if state is None or row_id not in state[0]['summaries']:raise ValueError('row unavailable')
                cache,manifest,directory=state
                row=self._job(directory/'jobs'/(row_id+'.json'),manifest,cache['binding'])
                if row is None or canonical_hash(row)!=cache['summaries'][row_id]['row_sha256']:
                    raise ValueError('selected bank row changed')
                return copy.deepcopy(row),dict(bank=bank,bank_status=info['status'],id=row_id)
            except (OSError,KeyError,TypeError,ValueError) as error:
                raise ValueError('Bank row unavailable or integrity check failed.') from error
