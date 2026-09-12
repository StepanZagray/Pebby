"""Frozen generated-history cell-evidence ablation; no engine or optimization."""
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SNAPSHOT=ROOT/'artifacts/world-cell-recall-code'
sys.path.insert(0,str(SNAPSHOT))
import numpy as np
import torch
from torch.nn import functional as F
from pebby.agent.world_model import load_world_checkpoint
from pebby.agent.world_train import load_dataset

CHECKPOINT=ROOT/'checkpoints/ls20-world-cell-recall-b1024.epoch1.pt'
DATA=ROOT/'data/ls20-world-combined-validation.npz'
OUT=ROOT/'artifacts/world-cell-recall-epoch1-policy-ablation.json'
EXPECTED='66c2637b4cf5f0ac713844224a2f663256bf017fdbc4cec91319a6f21d4074e5'


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    start=time.monotonic();print('PID',os.getpid(),flush=True)
    def timeout(*_):raise TimeoutError('cell policy ablation180second deadline')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(180);torch.set_num_threads(1)
    if OUT.exists():raise FileExistsError(OUT)
    hashes={str(p):digest(p) for p in [CHECKPOINT,DATA,Path(__file__)]}
    assert hashes[str(CHECKPOINT)]==EXPECTED
    code={str(p.relative_to(SNAPSHOT)):digest(p) for p in sorted((SNAPSHOT/'pebby').rglob('*.py'))}
    model,meta=load_world_checkpoint(CHECKPOINT,'cpu');model.eval();model.requires_grad_(False)
    assert model.cfg.cell_recall
    ablated=copy.deepcopy(model)
    with torch.no_grad():ablated.cell_context.weight.zero_()
    altered=[k for k,v in model.state_dict().items() if not torch.equal(v,ablated.state_dict()[k])]
    assert altered==['cell_context.weight'],altered
    weights=model.cell_context.weight.detach()
    norms={}
    for name,part in [('all',weights),('roles8',weights[:,:8]),('attributes14',weights[:,8:]),
                      ('shape6',weights[:,8:14]),('color4',weights[:,14:18]),('rotation4',weights[:,18:22])]:
        norms[name]={'shape':list(part.shape),'frobenius_norm':float(part.norm()),
                     'rms':float(part.square().mean().sqrt()),'per_column_norms':part.norm(dim=0).tolist()}
    data=load_dataset(DATA,history=model.cfg.history,cache_dir=ROOT/'data/world-array-cache')
    assert data['meta'].get('source')=='generated_only' and data['meta'].get('oracle_search')=='complete_only'
    seeds=data['seeds'];assert ((seeds>=1_000_000)&(seeds<2_000_000)).all()
    difficulty={int(level['seed']):int(level['difficulty']) for level in data['meta']['levels']}
    rng=np.random.default_rng(42);chosen=[];quotas={}
    for d in range(1,6):
        quota=1024//5+int(d<=1024%5);quotas[str(d)]=quota
        pool=sorted(s for s in np.unique(seeds) if difficulty[int(s)]==d)
        chosen.extend(map(int,rng.choice(pool,quota,replace=False)))
    rows=np.array([rng.choice(np.flatnonzero(seeds==s)) for s in chosen],np.int64)
    assert len(set(chosen))==1024
    batch={k:torch.from_numpy(np.array(data[k][rows],copy=True)) for k in ['frames','history_valid','previous_actions','optimal']}
    assert batch['frames'].shape==(1024,8,64,64) and (batch['optimal']>0).all()
    print(json.dumps({'loaded_seconds':time.monotonic()-start,'parameters':model.parameter_count(),'levels':len(rows)}),flush=True)
    def run(m,index):
        return m.logits_from(m.encode(batch['frames'][index],batch['history_valid'][index],batch['previous_actions'][index]))[0]
    # Production decisions use B1. This bounded panel is small enough to keep
    # that exact numerical convention throughout both model conditions.
    with torch.inference_mode():
        active=[];zero=[]
        for i in range(1024):
            active.append(run(model,slice(i,i+1)));zero.append(run(ablated,slice(i,i+1)))
            if (i+1)%256==0:print(json.dumps({'scored':i+1,'elapsed_seconds':time.monotonic()-start}),flush=True)
        active=torch.cat(active);zero=torch.cat(zero)
    bits=(batch['optimal'].long()[:,None]&(1<<torch.arange(4))).ne(0)
    target=bits.float()/bits.sum(1,keepdim=True)
    def metrics(logits):
        actions=logits.argmax(-1);correct=bits[torch.arange(1024),actions]
        ce=-(target*F.log_softmax(logits,dim=-1)).sum(-1)
        return {'set_correct':int(correct.sum()),'set_accuracy':float(correct.float().mean()),
                'uniform_optimal_ce':float(ce.mean()),'optimal_probability':float((logits.softmax(-1)*bits).sum(-1).mean())},actions,correct,ce
    ma,aa,ca,cea=metrics(active);mz,az,cz,cez=metrics(zero);delta=active-zero
    centered=delta-delta.mean(-1,keepdim=True)
    paired=[{'seed':s,'row':int(r),'difficulty':difficulty[s],'optimal_mask':int(batch['optimal'][i]),
             'active_action':int(aa[i]),'zero_action':int(az[i]),'active_correct':bool(ca[i]),'zero_correct':bool(cz[i]),
             'active_ce':float(cea[i]),'zero_ce':float(cez[i]),'active_logits':active[i].tolist(),'zero_logits':zero[i].tolist()} for i,(s,r) in enumerate(zip(chosen,rows))]
    assert all(digest(path)==value for path,value in hashes.items())
    assert code=={str(p.relative_to(SNAPSHOT)):digest(p) for p in sorted((SNAPSHOT/'pebby').rglob('*.py'))}
    result={'status':'complete','checkpoint':str(CHECKPOINT),'checkpoint_epoch':meta.get('epoch',meta.get('best_epoch')),
            'source_hashes':hashes,'code_snapshot':str(SNAPSHOT),'code_hashes':code,'parameters':model.parameter_count(),
            'projection_norms':norms,'intervention':'Only cell_context.weight zeroed in a separate deepcopy; all other state_dict tensors exactly equal. Checkpoint file unchanged.',
            'input_contract':'current public H8 frames/history_valid/producing actions only; exact optimal masks are scoring targets only',
            'selection':{'rng':'numpy default_rng42','difficulty_counts':quotas,'levels':1024,'one_row_per_level':'uniform row after stratified uniform level sampling','seed_list':chosen,'rows':rows.tolist()},
            'active':ma,'zeroed':mz,'argmax_flips':int((aa!=az).sum()),
            'paired_correctness':{'active_only_correct':int((ca&~cz).sum()),'zero_only_correct':int((~ca&cz).sum()),'both_correct':int((ca&cz).sum()),'both_wrong':int((~ca&~cz).sum())},
            'logit_differences':{'max_absolute':float(delta.abs().max()),'mean_absolute':float(delta.abs().mean()),'rms':float(delta.square().mean().sqrt()),'action_centered_rms':float(centered.square().mean().sqrt())},
            'paired':paired,'batch_size':1,'cpu_threads':1,'device':'cpu','pid':os.getpid(),'training_performed':False,
            'elapsed_seconds':time.monotonic()-start,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'limitations':['Teacher-state feature uptake only, not rollout completion or architecture selection.','Post-training zeroing measures reliance/ablation, not the causal benefit of training cell recall.','Frozen learned appearance supplies probabilities even on obscured cells; no learned visibility head or engine mask is used.','Balanced1024one-state-per-level panel differs from epoch aggregate validation weighting.','No official inputs, engine execution, oracle calls, optimization or checkpoint edits.']}
    OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');signal.alarm(0)
    print(json.dumps({k:result[k] for k in ['active','zeroed','argmax_flips','paired_correctness','logit_differences','elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
