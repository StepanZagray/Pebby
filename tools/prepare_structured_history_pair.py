"""Paired TRAIN H8/current-only fields; no optimizer or teacher input to encoder."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time
import numpy as np
import torch
from tools.build_structured_field_cache import actual_histories, digest, LABELS
from tools.build_structured_initial_cache import encoder_for
from pebby.agent.world_train import load_dataset,require_verified_data,require_winning_coverage

FORMAT='pebby.structured-history-pair.v1'


def current_only(frames):
    if frames.ndim!=4 or frames.shape[1:]!=(8,64,64):raise ValueError('public H8 required')
    repeated=frames[:,-1:].expand(-1,8,-1,-1).clone()
    valid=torch.zeros((len(frames),8),dtype=torch.bool);valid[:,-1]=True
    return repeated,valid,torch.full((len(frames),8),-1,dtype=torch.long)


def choose_pairs(parent,data,count=64,seed=42):
    seeds=np.asarray(parent['seeds']);rows=np.asarray(parent['source_rows']);difficulty=np.asarray(parent['difficulties'])
    if not 1<=count<=64 or len(np.unique(seeds))!=len(seeds) or not ((seeds>=0)&(seeds<1000000)).all():
        raise ValueError('requires distinct TRAIN seeds and count1..64')
    if not np.array_equal(data['seeds'][rows],seeds):raise ValueError('parent source row/seed alignment mismatch')
    stationary=np.zeros((len(rows),4),bool)
    for start in range(0,len(rows),64):
        rr=rows[start:start+64];stationary[start:start+len(rr)]=(data['frames'][rr,-1,None]==data['next_frames'][rr]).all((-1,-2))
    rng=np.random.default_rng(seed);selected=[];actions=[];forced=[]
    for d in range(1,6):
        quota=count//5+int(d<=count%5);pool=np.flatnonzero(difficulty==d)
        if len(pool)<quota:raise ValueError('insufficient balanced TRAIN levels')
        special=pool[stationary[pool].any(1)];take=min(3,quota,len(special))
        chosen=rng.choice(special,take,replace=False).tolist() if take else []
        remainder=np.setdiff1d(pool,chosen);chosen+=rng.choice(remainder,quota-take,replace=False).tolist()
        for j,ix in enumerate(chosen):
            selected.append(ix);forced.append(j<take)
            actions.append(int(rng.choice(np.flatnonzero(stationary[ix]))) if j<take else int(rng.integers(4)))
    selected=np.asarray(selected,np.int64);actions=np.asarray(actions,np.int64)
    return selected,actions,dict(forced_stationary_rows=int(sum(forced)),actual_stationary_rows=int(stationary[selected,actions].sum()),stationary_available_levels=int(stationary.any(1).sum()),selection='balanced TRAIN difficulty; up to3 byte-identical-frame actions per difficulty, other actions uniform; conditional diagnostic, not population estimate')


def pair_inputs(data,rows,actions):
    rows=np.asarray(rows);actions=np.asarray(actions)
    if rows.ndim!=1 or actions.shape!=rows.shape or not np.issubdtype(actions.dtype,np.integer) or not np.isin(actions,np.arange(4)).all():raise ValueError('one actual action0..3 per row required')
    current=tuple(torch.from_numpy(np.array(data[k][rows],copy=True)) for k in ('frames','history_valid','previous_actions'))
    batch={k:np.array(data[k][rows],copy=True) for k in ('frames','history_valid','previous_actions','next_frames','lost_life')}
    all_next=actual_histories(batch);ix=torch.arange(len(rows));act=torch.from_numpy(actions.copy()).long()
    following=tuple(x[ix,act] for x in all_next)
    return {'h8':(current,following),'current_only':(current_only(current[0]),current_only(following[0]))}


def encode_pairs(encoder,inputs):
    result={};device=next(encoder.parameters()).device
    with torch.inference_mode():
        for mode,(current,following) in inputs.items():
            result[mode]={}
            for key,public in [('fields',current),('next_fields',following)]:
                value=encoder(*(x.to(device) for x in public)).float().cpu().numpy()
                if value.shape!=(len(public[0]),148,96) or not np.isfinite(value).all():raise ValueError('bad frozen field output')
                result[mode][key]=value.astype(np.float16)
                if not np.isfinite(result[mode][key]).all():raise ValueError('float16 overflow')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,default=Path('data/structured-field-16384/train'));p.add_argument('--out',type=Path,required=True)
    p.add_argument('--count',type=int,default=64);p.add_argument('--seed',type=int,default=42);p.add_argument('--device',choices=('cpu','cuda'),default='cpu');args=p.parse_args()
    start=time.monotonic();print('PID',os.getpid(),flush=True);torch.set_num_threads(1)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('paired cache120second deadline')));signal.alarm(120)
    if args.out.exists():raise FileExistsError(args.out)
    manifest=json.loads((args.parent/'manifest.json').read_text())
    if manifest.get('format')!='pebby.structured-field-cache.v1' or manifest.get('source')!='generated_only' or manifest.get('split')!='train' or manifest.get('status')!='complete':raise ValueError('requires completed H1 TRAIN parent')
    guards={str(args.parent/'manifest.json'):digest(args.parent/'manifest.json'),manifest['source_path']:manifest['source_sha256'],__file__:digest(__file__)}
    parent={}
    for key,info in manifest['arrays'].items():
        path=args.parent/(key+'.npy');guards[str(path)]=info['sha256']
        if digest(path)!=info['sha256']:raise ValueError('parent array drift')
        parent[key]=np.load(path,mmap_mode='r')
    if digest(manifest['source_path'])!=manifest['source_sha256']:raise ValueError('source drift')
    data=load_dataset(manifest['source_path'],history=8,cache_dir=Path('data/world-array-cache'));require_verified_data(data);require_winning_coverage(data)
    selected,actions,selection=choose_pairs(parent,data,args.count,args.seed);rows=parent['source_rows'][selected]
    inputs=pair_inputs(data,rows,actions)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    encoder,watched=encoder_for(manifest,args.device);guards.update(watched)
    for helper in ('tools/build_structured_field_cache.py','tools/build_structured_initial_cache.py'):guards[helper]=digest(helper)
    fields=encode_pairs(encoder,inputs)
    labels={}
    for name,source in LABELS.items():
        value=np.array(data[source][rows],copy=True)
        labels[name]=value[np.arange(len(rows)),actions] if name.startswith('next_') or name in ('lost_life','terminal','won','distances') else value
    labels.update(actions=actions,seeds=np.array(parent['seeds'][selected]),source_rows=np.array(rows),parent_rows=selected,difficulties=np.array(parent['difficulties'][selected]))
    args.out.parent.mkdir(parents=True,exist_ok=True);temporary=Path(tempfile.mkdtemp(dir=args.out.parent,prefix='.paired-history-'))
    try:
        for name,value in labels.items():np.save(temporary/(name+'.npy'),value,allow_pickle=False)
        for mode,arrays in fields.items():
            (temporary/mode).mkdir()
            for name,value in arrays.items():np.save(temporary/mode/(name+'.npy'),value,allow_pickle=False)
        inventory={str(path.relative_to(temporary)):{'sha256':digest(path),'shape':list(np.load(path,mmap_mode='r').shape),'dtype':str(np.load(path,mmap_mode='r').dtype)} for path in temporary.rglob('*.npy')}
        if any(digest(path)!=sha for path,sha in guards.items()):raise ValueError('paired sources changed')
        report=dict(format=FORMAT,status='complete',split='train',source='generated_only',pid=os.getpid(),levels=len(rows),seed=args.seed,selection=selection,field_encoder=manifest['field_encoder'],source_hashes=guards,arrays=inventory,device=args.device,precision='FP32, no autocast, both TF32 off, float16 storage',elapsed_seconds=time.monotonic()-start,scope='Matched actual transitions; current-only source/target lastvalid and allactions-1. No training; stationary enrichment is conditional, not population sampling.')
        (temporary/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');temporary.rename(args.out)
    finally:
        if temporary.exists():shutil.rmtree(temporary)
    signal.alarm(0);print(json.dumps({'status':'complete','levels':len(rows),'selection':selection,'elapsed_seconds':report['elapsed_seconds']}),flush=True)

if __name__=='__main__':main()
