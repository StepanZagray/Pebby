"""Disposable query-only fit on frozen generated-state features; never saves weights."""
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
SNAPSHOT = ROOT / 'artifacts/world-onpolicy-code'
sys.path.insert(0, str(SNAPSHOT))
import numpy as np
import torch
from torch.nn import functional as F
from pebby.agent.model import load_checkpoint

torch.set_num_threads(1)
torch.manual_seed(2026)
START = time.monotonic()
DEADLINE = START + 480
OUT = Path('artifacts/world-frozen-query-trainability.json')
CHECKPOINT = Path('checkpoints/ls20-world-onpolicy-b1024.epoch1.pt')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


REPORT = dict(status='running', pid=os.getpid(), device='cpu', cpu_threads=1,
              seed=2026, checkpoint=str(CHECKPOINT), checkpoint_sha256=digest(CHECKPOINT),
              source='generated_only', feature_cache='RAM only; source NPY maps opened read-only',
              code_snapshot=str(SNAPSHOT.relative_to(ROOT)),
              code_hashes_at_import=json.loads((SNAPSHOT/'source-hashes.json').read_text()),
              script_sha256=digest(Path(__file__)), splits={}, measurements=[],
              objective='Query-only cross entropy against a uniform target over exact optimal actions',
              optimizer={'name':'Adam', 'lr':.001, 'batch_size':1024, 'maximum_updates':100},
              limits='Frozen-state head trainability diagnostic; no rollouts, official inputs, encoder updates, or saved trained weights.')


def persist():
    REPORT.update(elapsed_seconds=time.monotonic()-START,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    temp=OUT.with_suffix('.tmp')
    temp.write_text(json.dumps(REPORT,indent=2)+'\n');os.replace(temp,OUT)


def select(split, count):
    expected=304365 if split=='train' else 59649
    found=[]
    for manifest in Path('data/world-array-cache').glob('*/manifest.json'):
        meta=json.loads(manifest.read_text())
        if meta['arrays']['frames']['shape'][0]==expected:
            found.append((manifest,meta))
    assert len(found)==1
    manifest,cache_meta=found[0];folder=manifest.parent
    arrays={key:np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False)
            for key in ('frames','history_valid','previous_actions','seeds','optimal')}
    level_meta=json.loads(str(np.load(folder/'meta.npy',allow_pickle=False).item()))
    assert level_meta['source']=='generated_only' and level_meta['oracle_search']=='complete_only'
    assert level_meta['split']==split
    for key in ('seeds','optimal','history_valid','previous_actions','meta'):
        assert digest(folder/(key+'.npy'))==cache_meta['arrays'][key]['sha256']
    levels={int(x['seed']):x for x in level_meta['levels']}
    valid_rows=np.flatnonzero(arrays['optimal']>0)
    rows_by_seed={}
    for row in valid_rows:
        rows_by_seed.setdefault(int(arrays['seeds'][row]),[]).append(int(row))
    rng=np.random.default_rng(2026+(split=='validation'))
    chosen=[];difficulty_counts={}
    for difficulty in range(1,6):
        quota=count//5+(difficulty<=count%5)
        candidates=sorted(seed for seed in rows_by_seed if levels[seed]['difficulty']==difficulty)
        assert len(candidates)>=quota
        picked=rng.choice(candidates,quota,replace=False)
        chosen.extend(int(rng.choice(rows_by_seed[int(seed)])) for seed in picked)
        difficulty_counts[str(difficulty)]=quota
    rng.shuffle(chosen);indices=np.asarray(chosen,dtype=np.int64)
    selected={key:np.array(value[indices],copy=True) for key,value in arrays.items()}
    assert len(np.unique(selected['seeds']))==count
    REPORT['splits'][split]=dict(rows=count,unique_levels=count,difficulty_counts=difficulty_counts,
         row_indices=indices.tolist(),seeds=selected['seeds'].tolist(),cache_manifest=str(manifest),
         cache_manifest_sha256=digest(manifest),recorded_source_npz_sha256=cache_meta['source_sha256'],
         selected_input_sha256={key:array_digest(value) for key,value in selected.items()},
         provenance='Existing verified combined-bank cache; small provenance arrays rehashed, selected public-frame bytes hashed; full frame cache was not rescanned.')
    return selected


def encode(policy, data):
    chunks={key:[] for key in ('raw','state','cells','glyph','weights','base','total','query')}
    with torch.no_grad():
        for start in range(0,len(data['seeds']),16):
            sl=slice(start,start+16)
            enc=policy.encode(torch.from_numpy(data['frames'][sl]).long(),
                    history_valid=torch.from_numpy(data['history_valid'][sl]),
                    previous_actions=torch.from_numpy(data['previous_actions'][sl]))
            total,extra=policy.logits_from(enc)
            for key in ('raw','state','cells','glyph'):
                chunks[key].append(enc[key].detach())
            chunks['weights'].append(extra['player'].softmax(-1))
            chunks['base'].append(extra['direct']+policy.ranker(extra['features']).squeeze(-1))
            chunks['total'].append(total)
            chunks['query'].append(extra['query'])
    return {key:torch.cat(values) for key,values in chunks.items()}


def target(masks):
    bits=(torch.as_tensor(masks).long()[:,None]&(1<<torch.arange(4)))!=0
    assert bits.any(1).all()
    return bits.float()/bits.sum(1,keepdim=True),bits


def metrics(logits,targets,bits):
    return dict(uniform_optimal_ce=float(-(targets*F.log_softmax(logits,dim=-1)).sum(1).mean()),
                optimal_set_accuracy=float(bits.gather(1,logits.argmax(1)[:,None]).float().mean()),
                logit_std_across_states=logits.std(0).tolist())


def predict(head, features):
    return head(features['raw'],features['state'],features['weights'],features['glyph'])


print('PID',os.getpid(),flush=True);persist()
try:
    policy,checkpoint=load_checkpoint(CHECKPOINT,'cpu');policy.requires_grad_(False)
    assert checkpoint['epoch']==1 and policy.cfg.query_readout and policy.cfg.glyph_recall
    train=select('train',1024);validation=select('validation',1024)
    assert not np.intersect1d(train['seeds'],validation['seeds']).size
    REPORT['train_validation_level_disjoint']=True
    inputs={};targets={};bits={}
    for name,data in [('train',train),('validation',validation)]:
        inputs[name]=encode(policy,data);targets[name],bits[name]=target(data['optimal'])
        REPORT['splits'][name]['feature_shapes']={key:list(value.shape) for key,value in inputs[name].items()}
        REPORT['splits'][name]['baselines']={key:metrics(inputs[name][key],targets[name],bits[name]) for key in ('total','base','query')}
        print('FEATURES',name,'elapsed',time.monotonic()-START,flush=True)
        persist()
    REPORT['feature_collection_seconds']=time.monotonic()-START
    head=copy.deepcopy(policy.query_head);head.requires_grad_(True)
    assert all(not p.requires_grad for p in policy.parameters())
    REPORT['trained_parameter_count']=sum(p.numel() for p in head.parameters())
    optimizer=torch.optim.Adam(head.parameters(),lr=.001)
    def measure(step):
        with torch.no_grad():
            measurement={'update':step,'elapsed_seconds':time.monotonic()-START}
            for name in ('train','validation'):
                logits=predict(head,inputs[name])
                measurement[name]={'query_only':metrics(logits,targets[name],bits[name]),
                    'frozen_base_plus_query':metrics(logits+inputs[name]['base'],targets[name],bits[name])}
            REPORT['measurements'].append(measurement);persist()
            print('MEASURE',json.dumps(measurement),flush=True)
    measure(0)
    maximum=50 if REPORT['feature_collection_seconds']>120 else 100
    REPORT['budget_adjusted_maximum_updates']=maximum
    durations=[];completed=0
    for step in range(1,maximum+1):
        # Leave enough wall-clock for final held-out scoring and artifact persistence.
        estimate=max(durations[-3:],default=5.)
        if time.monotonic()+estimate+15>DEADLINE:
            REPORT['stopped_for_wall_budget']=True;break
        tick=time.monotonic();optimizer.zero_grad(set_to_none=True)
        logits=predict(head,inputs['train'])
        assert logits.shape==(1024,4)
        loss=-(targets['train']*F.log_softmax(logits,dim=-1)).sum(1).mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())
        optimizer.step();durations.append(time.monotonic()-tick);completed=step
        if step in (10,50,100):measure(step)
    if REPORT['measurements'][-1]['update']!=completed:measure(completed)
    REPORT.update(status='complete',updates_completed=completed,
                  checkpoint_unchanged=digest(CHECKPOINT)==REPORT['checkpoint_sha256'],
                  updated_weights_saved=False,true_distinct_level_batch_size=1024,
                  mean_update_seconds=sum(durations)/len(durations) if durations else None)
    assert REPORT['checkpoint_unchanged']
    del optimizer,head,policy,inputs
    REPORT['updated_weights_discarded']=True;persist()
    print('COMPLETE',completed,'seconds',REPORT['elapsed_seconds'],'rss_mib',REPORT['peak_rss_mib'],flush=True)
except BaseException as error:
    REPORT.update(status='failed',error=repr(error));persist();raise
