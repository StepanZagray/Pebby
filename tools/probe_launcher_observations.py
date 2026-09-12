"""Bounded generated-only launcher patch and frozen-feature direction probe."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import signal
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from pebby.ls20 import names
from pebby.ls20.generate import build_level
from pebby.ls20.env import Ls20Scenario
from pebby.ls20.layout import extract
from pebby.agent.structured_field import load_structured_field_encoder
from tools.prepare_visible_cell_labels import support_patch, support_rejection_reason

DIRECTIONS=((0,-1),(0,1),(-1,0),(1,0))

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def load_bank(path,split):
    specs=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    low,high=(0,1000000) if split=='train' else (1000000,2000000)
    if len({x['seed'] for x in specs})!=len(specs):raise ValueError('duplicate source seeds')
    for s in specs:
        proof=s.get('proof',{});context=s.get('training_context_index')
        if not (low<=s['seed']<high and s.get('search_truncated') is False and s.get('engine_verified') is True
                and s.get('context_engine_verified') is True and proof.get('seed')==s['seed']
                and proof.get('context_index')==context and proof.get('context_engine_verified') is True
                and proof.get('context_optimal_actions')==s.get('context_optimal_actions')
                and s.get('verification_level_index')==context):
            raise ValueError('unverified generated source or context mismatch')
        if s.get('launchers') and context==0:raise ValueError('unsupported context-zero launcher')
    return specs

def select(specs,count,seed):
    candidates=[s for s in specs if s.get('launchers')]
    rng=np.random.default_rng(seed);rng.shuffle(candidates)
    counts=np.zeros(4,np.int64);selected=[]
    while candidates and len(selected)<count:
        vectors=[np.bincount([DIRECTIONS.index(tuple(x['delta'])) for x in s['launchers']],minlength=4) for s in candidates]
        scores=[float(np.square(counts+v-(counts+v).mean()).sum()) for v in vectors]
        index=int(np.argmin(scores));selected.append(candidates.pop(index));counts+=vectors[index]
    return selected

def census(patches,directions,seeds,primary):
    result={}
    for title,mask in [('all',np.ones(len(patches),bool)),('primary',primary)]:
        groups=defaultdict(list)
        for i in np.flatnonzero(mask):groups[hashlib.sha256(patches[i].tobytes()).hexdigest()].append(int(i))
        conflicts=[{'patch_sha256':h,'indices':ids,'directions':[int(directions[i]) for i in ids],
                    'seeds':[int(seeds[i]) for i in ids]} for h,ids in groups.items() if len({directions[i] for i in ids})>1]
        result[title]={'instances':int(mask.sum()),'patterns':len(groups),'contradictory_patterns':len(conflicts),
                       'conflicting_instances':sum(len(x['indices']) for x in conflicts),'conflicts':conflicts}
    return result

def collect(specs):
    frames=[];rows=[]
    for index,spec in enumerate(specs):
        env=Ls20Scenario(build_level(spec),spec['training_context_index']);frame=np.asarray(env.render(),np.uint8)
        if not np.array_equal(frame,np.asarray(env.render(),np.uint8)):raise ValueError('render mismatch')
        if tuple(env.player_cell())!=tuple(spec['start']) or env.fog()!=bool(spec['fog']):raise ValueError('generated state mismatch')
        layout=extract(env);frames.append(frame)
        for launcher_index,entry in enumerate(spec['launchers']):
            cell=tuple(entry['cell']);delta=tuple(entry['delta']);x,y=names.cell_to_pixel(*cell)
            pixels=support_patch(frame,*cell,padding=1)
            if pixels is None:pixels=np.zeros((7,7),np.uint8)
            reason=support_rejection_reason({'frame':frame,'spec':spec,'player_cell':spec['start']},*cell,padding=1)
            dx,dy=delta;origin=(x-dx,y-dy)
            if dx:bar=[(origin[0]+(4 if dx<0 else 0),origin[1]+j) for j in range(5)]
            else:bar=[(origin[0]+j,origin[1]+(4 if dy<0 else 0)) for j in range(5)]
            bar_seen=all(0<=bx<64 and 0<=by<64 and frame[by,bx]==1 for bx,by in bar)
            geometry=next(l for l in layout.launchers if tuple(l['cell'])==cell and tuple(l['delta'])==delta)
            # Diagnostic-only support for all scanned cells through first blocker.
            ray=[(cell[0]+dx*t,cell[1]+dy*t) for t in range(1,geometry['distance']+2)]
            ray_reasons=[support_rejection_reason({'frame':frame,'spec':spec,'player_cell':spec['start']},*c,padding=0) for c in ray]
            rows.append({'frame_index':index,'seed':spec['seed'],'context':spec['training_context_index'],
                         'launcher_index':launcher_index,'cell':cell,'direction':DIRECTIONS.index(delta),
                         'patch':pixels,'support_reason':reason or 'none','bar_visible':bar_seen,
                         'primary':reason is None and bar_seen,'throw_distance':geometry['distance'],
                         'ray_fully_public':all(x is None for x in ray_reasons),
                         'ray_reasons':sorted(set(x for x in ray_reasons if x is not None)),
                         'triggers':geometry['triggers']})
    return np.stack(frames),rows

def fit(features,labels,trainmask,valmask):
    trainclasses=set(labels[trainmask].tolist());valclasses=set(labels[valmask].tolist())
    if trainclasses!={0,1,2,3} or valclasses!={0,1,2,3}:
        return {'status':'not_fit','reason':'all four directions required in primary TRAIN and validation','train_classes':sorted(trainclasses),'validation_classes':sorted(valclasses)}
    x=torch.tensor(features).float();y=torch.tensor(labels).long();ti=torch.tensor(np.flatnonzero(trainmask));vi=torch.tensor(np.flatnonzero(valmask))
    mean=x[ti].mean(0);std=x[ti].std(0,unbiased=False).clamp_min(.01);x=(x-mean)/std
    torch.manual_seed(42);model=torch.nn.Linear(x.shape[1],4);optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x[ti]),y[ti]);loss.backward();optimizer.step()
    with torch.no_grad():
        out={}
        for name,ids in [('train',ti),('validation',vi)]:
            logits=model(x[ids]);predicted=logits.argmax(-1);truth=y[ids]
            out[name]={'instances':len(ids),'correct':int((predicted==truth).sum()),'accuracy':float((predicted==truth).float().mean()),
                       'cross_entropy':float(F.cross_entropy(logits,truth)),
                       'confusion':np.bincount((truth*4+predicted).numpy(),minlength=16).reshape(4,4).tolist()}
    return {'status':'complete','features':x.shape[1],'parameters':sum(p.numel() for p in model.parameters()),
            'updates':200,'optimizer':'Adam','lr':.01,'normalization':'TRAIN only mean/std floor .01','fixed_seed':42,'metrics':out}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',default='data/ls20-mechanism-training-2000.jsonl');p.add_argument('--validation',default='data/ls20-mechanism-validation50.jsonl')
    p.add_argument('--output',default='data/ls20-launcher-observation-probe.npz');p.add_argument('--report',default='artifacts/launcher-observation-probe.json')
    p.add_argument('--seconds',type=int,default=120);args=p.parse_args()
    if not 1<=args.seconds<=120:p.error('seconds must1..120')
    if Path(args.output).exists() or Path(args.report).exists():p.error('outputs must be new')
    torch.set_num_threads(1);start=time.monotonic();print('PID',os.getpid(),flush=True)
    def expired(*_):raise TimeoutError('launcher observation probe deadline')
    signal.signal(signal.SIGALRM,expired);signal.alarm(args.seconds)
    sources={str(x):digest(x) for x in [args.train,args.validation,__file__,'pebby/ls20/generate.py','pebby/ls20/layout.py','pebby/ls20/rails.py','pebby/ls20/env.py','third_party/ls20/ls20.py','tools/prepare_visible_cell_labels.py']}
    report={'status':'running','pid':os.getpid(),'sources':sources,'official_inputs_used':False,'device':'cpu','threads':1}
    try:
        train=load_bank(args.train,'train');val=load_bank(args.validation,'validation')
        if {s['seed'] for s in train}&{s['seed'] for s in val}:raise ValueError('split leakage')
        chosen=[select(train,64,42),select(val,32,43)];allframes=[];allrows=[];offset=0
        with patch('pebby.ls20.plan.Oracle.__init__',side_effect=AssertionError('no new planner calls')):
            for split,specs in enumerate(chosen):
                frames,rows=collect(specs)
                for row in rows:row.update(split=split,frame_index=row['frame_index']+offset)
                allframes.extend(frames);allrows.extend(rows);offset+=len(frames)
        frames=np.stack(allframes);patches=np.stack([x['patch'] for x in allrows]);labels=np.array([x['direction'] for x in allrows],np.int8)
        seeds=np.array([x['seed'] for x in allrows],np.int64);splits=np.array([x['split'] for x in allrows],np.int8);primary=np.array([x['primary'] for x in allrows],bool)
        encoder=load_structured_field_encoder(device='cpu');sources.update({str(encoder.world_checkpoint_path):digest(encoder.world_checkpoint_path),str(encoder.visibility_checkpoint_path):digest(encoder.visibility_checkpoint_path)})
        # Bind the exact implementation used by the fixed cached-field encoder.
        for name in ('structured_field','world_model','world_readout','world_grounding','world_rollout','cell_appearance','cell_appearance_dense','glyph_model','cell_visibility'):
            file='pebby/agent/'+name+'.py';sources[file]=digest(file)
        fields=[]
        for first in range(0,len(frames),8):
            current=torch.tensor(frames[first:first+8]).long();history=current[:,None].expand(-1,8,-1,-1)
            valid=torch.zeros(len(current),8,dtype=torch.bool);valid[:,-1]=True;actions=torch.full((len(current),8),-1)
            fields.append(encoder(history,valid,actions).cpu().numpy())
        fields=np.concatenate(fields);features=np.stack([fields[x['frame_index'],x['cell'][1]*12+x['cell'][0]] for x in allrows])
        raw=F.one_hot(torch.tensor(patches).long(),16).flatten(1).float().numpy()
        classifiers={name:fit(feature,labels,(splits==0)&primary,(splits==1)&primary) for name,feature in [('raw7x7',raw),('core48',features[:,:48]),('field96',features)]}
        summary={}
        for split,name in enumerate(('train','validation')):
            mask=splits==split;selected_rows=[x for x in allrows if x['split']==split]
            summary[name]={'bank_levels':len(train if split==0 else val),'launcher_bearing_bank_levels':sum(bool(s.get('launchers')) for s in (train if split==0 else val)),
                'selected_levels':len(chosen[split]),'selected_seeds':[s['seed'] for s in chosen[split]],'instances':int(mask.sum()),
                'directions':np.bincount(labels[mask],minlength=4).tolist(),'primary_instances':int((mask&primary).sum()),
                'primary_levels':len(set(seeds[mask&primary].tolist())),'primary_directions':np.bincount(labels[mask&primary],minlength=4).tolist(),
                'support_reasons':dict(Counter(x['support_reason'] for x in selected_rows)),'bar_obscured':sum(not x['bar_visible'] for x in selected_rows),
                'fully_public_ray_instances':sum(x['ray_fully_public'] for x in selected_rows),'census':census(patches[mask],labels[mask],seeds[mask],primary[mask])}
        records=[{k:v for k,v in x.items() if k!='patch'} for x in allrows]
        meta={'format':'pebby.launcher-observation-probe.v1','source':'generated_only','direction_deltas':DIRECTIONS,'records':records,'source_hashes':sources,'encoder':encoder.metadata(),
              'input_contract':'public initialframes/patches and frozen public H8 field tokens; labels/support/geometry diagnostic only'}
        output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);temp=output.with_name(output.name+'.tmp')
        with temp.open('wb') as f:np.savez_compressed(f,frames=frames,patches=patches,directions=labels,seeds=seeds,splits=splits,primary=primary,
                                                   fields=features,meta=np.array(json.dumps(meta)))
        if any(digest(file)!=value for file,value in sources.items()):raise ValueError('source changed during probe')
        os.replace(temp,output)
        report.update(status='complete',selection='seed42 train/43 validation greedy direction-count balancing; all launchers retained per selected distinct level',splits=summary,
                      combined_census=census(patches,labels,seeds,primary),classifiers=classifiers,dataset=str(output),dataset_sha256=digest(output),source_unchanged=True,
                      reused_bank_complete_search_and_engine_proofs=True,new_oracle_calls=0,
                      limitations=['Validation bank contains only10 launcher-bearing levels; correlated multiple launchers perlevel are not independent levels.',
                                   'Initial-state direction perception only; no routing accuracy or policy improvement established.',
                                   'Full support and bar-visible primary selection uses diagnostic labels only; it is not an inference gate.',
                                   'Raw patch and core probes fit200fixed CPUupdates; no validation selection, no weights published.','Route support is a conservative diagnostic of observed ray pixels, not proof all blocker identities are recovered.'])
    except BaseException as error:report.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        signal.alarm(0);report['elapsed_seconds']=time.monotonic()-start;Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'status':report['status'],'elapsed_seconds':report['elapsed_seconds']}),flush=True)

if __name__=='__main__':main()
