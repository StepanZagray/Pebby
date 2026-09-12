"""Frozen latent risk probes, generated-only; held-out/fatal rows never fitted."""
import importlib.util,json,os,resource,time
from pathlib import Path
from collections import Counter
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from pebby.agent.world_train import load_dataset
from pebby.agent.on_policy_provenance import file_digest
from tools.goal_attribute_probes import stratified_seeds
from tools import probe_world_fatal_choices as fatal

TRAIN=Path('data/ls20-world-combined-train.npz')
VALID=Path('data/ls20-world-combined-validation.npz')
CHECKPOINT=Path('checkpoints/ls20-world-onpolicy-round2-b1024.epoch1.pt')
SNAPSHOT=Path('artifacts/world_model_fatal_snapshot.py')
FATAL=Path('artifacts/world-round2-fatal-choices.json')
BANK=Path('data/ls20-verified-validation-monitor.jsonl')
OUT=Path('artifacts/world-frozen-binary-risk-probe.json')


def unsafe(data):return (data['distances']<0)|data['lost_life']|(data['terminal']&~data['won'])


def selection(path,count,seed,split):
    with np.load(path,allow_pickle=False) as archive:
        meta=json.loads(str(archive['meta'].item()));seeds=archive['seeds']
        labels={k:archive[k] for k in ('distances','lost_life','terminal','won')}
    if meta.get('source')!='generated_only' or meta.get('oracle_search')!='complete_only':raise ValueError('invalid source')
    if np.any(seeds< (1_000_000 if split=='validation' else 0)) or np.any(seeds>= (2_000_000 if split=='validation' else 1_000_000)):raise ValueError('invalid seed split')
    rng=np.random.default_rng(seed+1);risk_rows=unsafe(labels).any(1)
    risk_levels=np.unique(seeds[risk_rows]);risk_count=min(count//2,len(risk_levels))
    selected=list(map(int,rng.choice(risk_levels,risk_count,replace=False)));risk_selected=set(selected)
    difficulty={int(l['seed']):int(l['difficulty']) for l in meta['levels']}
    for d in range(1,6):
        quota=count//5+int(d<=count%5);remaining=quota-sum(difficulty[s]==d for s in selected)
        if remaining<0:raise ValueError('risk subset exceeds planned difficulty quota')
        pool=[s for s in difficulty if difficulty[s]==d and s not in risk_selected]
        selected.extend(map(int,rng.choice(pool,remaining,replace=False)))
    rows=np.array([rng.choice(np.flatnonzero((seeds==s)&(risk_rows if s in risk_selected else True))) for s in selected],np.int64)
    y=unsafe(labels)[rows];neg=int((~y).sum());pos=int(y.sum())
    return rows,{'levels':len(rows),'risk_enriched_levels':risk_count,'available_unsafe_levels':len(risk_levels),'rows':rows.tolist(),'seeds':seeds[rows].tolist(),'branches':int(y.size),'unsafe':pos,'safe':neg,'unsafe_prevalence':float(y.mean()),'lost_life':int(labels['lost_life'][rows].sum()),'unreachable_distance':int((labels['distances'][rows]<0).sum()),'difficulty_counts':dict(Counter(int(l['difficulty']) for l in meta['levels'] if l['seed'] in set(selected)))}


def features(model,batch):
    f=torch.as_tensor(batch['frames']).long();v=torch.as_tensor(batch['history_valid']).bool();p=torch.as_tensor(batch['previous_actions']).long()
    nxt=torch.as_tensor(batch['next_frames']).long();reset=torch.as_tensor(batch['lost_life']).bool().flatten();b,h=f.shape[:2]
    tokens=model.frame_tokens(f.flatten(0,1)).view(b,h,model.tokens,-1)
    current=model.assemble(tokens,v,p,glyph_logits=model.glyph_logits(f[:,-1]) if model.cfg.glyph_recall else None)
    next_tokens=model.frame_tokens(nxt.flatten(0,1)).view(b,4,1,model.tokens,-1)
    target_tokens=torch.cat((tokens[:,1:][:,None].expand(-1,4,-1,-1,-1),next_tokens),2).flatten(0,1)
    target_v=torch.cat((v[:,1:][:,None].expand(-1,4,-1),torch.ones(b,4,1,dtype=torch.bool)),2).flatten(0,1)
    target_p=torch.cat((p[:,1:][:,None].expand(-1,4,-1),torch.arange(4)[None,:,None].expand(b,-1,-1)),2).flatten(0,1)
    target_tokens=torch.where(reset[:,None,None,None],next_tokens.flatten(0,1).expand(-1,h,-1,-1),target_tokens)
    reset_v=torch.zeros_like(target_v);reset_v[:,-1]=True
    target_v=torch.where(reset[:,None],reset_v,target_v);target_p=torch.where(reset[:,None],-torch.ones_like(target_p),target_p)
    actual=model.assemble(target_tokens,target_v,target_p,glyph_logits=model.glyph_logits(nxt.flatten(0,1)) if model.cfg.glyph_recall else None)['latent']
    imagined=model.predict_successors(current['latent']).flatten(0,1)
    return {k:z.clone() for k,z in [('actual',actual),('imagined',imagined)]}, {k:model.value(z)[0].softmax(-1)[:,-1].clone() for k,z in [('actual',actual),('imagined',imagined)]}


def metrics(y,probability):
    y=np.asarray(y,dtype=bool).ravel();p=np.asarray(probability,dtype=float).ravel()
    if y.shape!=p.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('invalid metric input')
    positive=int(y.sum());negative=len(y)-positive
    order=np.argsort(p,kind='stable');sorted_p=p[order];ranks=np.empty(len(y),float)
    start=0
    while start<len(y):
        end=start+1
        while end<len(y) and sorted_p[end]==sorted_p[start]:end+=1
        ranks[order[start:end]]=(start+1+end)/2;start=end
    auc=float((ranks[y].sum()-positive*(positive+1)/2)/(positive*negative)) if positive and negative else None
    pred=p>=.5;tpr=float(pred[y].mean()) if positive else None;tnr=float((~pred[~y]).mean()) if negative else None
    q=np.clip(p,1e-7,1-1e-7)
    return {'examples':len(y),'positives':positive,'prevalence':float(y.mean()),'auroc':auc,'balanced_accuracy':(tpr+tnr)/2 if positive and negative else None,'recall_unsafe':tpr,'specificity':tnr,'accuracy':float((pred==y).mean()),'bce':float(-(y*np.log(q)+(~y)*np.log(1-q)).mean()),'probability_mean':float(p.mean())}


def train_probe(trainx,trainy,evals,kind):
    mean=trainx.mean(0);scale=trainx.std(0,correction=0).clamp_min(1e-4)
    x=(trainx-mean)/scale;torch.manual_seed(508)
    head=nn.Linear(x.shape[-1],1) if kind=='linear' else nn.Sequential(nn.Linear(x.shape[-1],64),nn.GELU(),nn.Linear(64,1))
    optimizer=torch.optim.Adam(head.parameters(),lr=.02 if kind=='linear' else .003)
    y=trainy.float();curve=[]
    for step in range(200):
        optimizer.zero_grad(set_to_none=True);loss=nn.functional.binary_cross_entropy_with_logits(head(x).flatten(),y);loss.backward();optimizer.step()
        if step in (0,49,99,199):curve.append({'update':step+1,'train_bce':float(loss.detach())})
    with torch.no_grad():
        result={'parameters':sum(p.numel() for p in head.parameters()),'updates':200,'optimizer':'Adam full-batch unweighted BCE','train_only_standardization':True,'training_curve':curve,'train':metrics(y.numpy(),head(x).flatten().sigmoid().numpy())}
        for name,(features_,labels) in evals.items():result[name]=metrics(labels.numpy(),head((features_-mean)/scale).flatten().sigmoid().numpy())
    return result


def main():
    print('PID',os.getpid(),flush=True);torch.set_num_threads(1);start=time.monotonic();deadline=start+600
    if OUT.exists():raise ValueError('refusing report overwrite')
    bound=[TRAIN,VALID,CHECKPOINT,SNAPSHOT,FATAL,BANK,Path(__file__),Path('data/world-round2-fatal-choice-histories.npz')]
    hashes={str(p):file_digest(p) for p in bound}
    report={'status':'running','pid':os.getpid(),'device':'cpu','torch_threads':1,'source_hashes':hashes,'source':'generated_only','train_features':'risk-enriched distinct levels: up to half known-unsafe levels/rows; remaining levels random; all4actual branches','selection':{},'conditions':{},'encoder_frozen':True,'probe_weights_discarded':True,'random_prevalence_gate_report':'artifacts/world-frozen-binary-risk-prevalence-gate.json','limitations':['Selection is explicitly risk-enriched: these metrics are conditional diagnostics, not population calibration or rollout risk.','Diagnostic binary probes do not modify the controller or establish improved completion.','Unsafe combines unreachable successor states with immediate lost-life transitions. Lost-life is not generally a property of a successor state alone, especially after history reset.','Four branches from one level are correlated; train/validation splits are by distinct level.','Frozen value baseline is its unreachable distance-bin probability; it lacks an explicit lost-life head.','No threshold tuning or model selection on validation/fatal sets; both predetermined heads are reported.']}
    def persist():
        report.update(elapsed_seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024);OUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    try:
        selected={}
        for name,path,count,seed in [('train',TRAIN,4096,184),('validation',VALID,1024,185)]:
            selected[name],info=selection(path,count,seed,name);report['selection'][name]=info
            print(name,'prevalence',info['unsafe_prevalence'],'positive',info['unsafe'],'negative',info['safe'],flush=True)
            if min(info['unsafe'],info['safe'])<100:raise ValueError('insufficient label classes before encoding')
        if set(report['selection']['train']['seeds'])&set(report['selection']['validation']['seeds']):raise ValueError('level split overlap')
        persist()
        spec=importlib.util.spec_from_file_location('pebby.agent._binary_risk_frozen',SNAPSHOT);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        model,_=module.load_world_checkpoint(CHECKPOINT);model.requires_grad_(False)
        report.update(model_parameters=model.parameter_count(),latent_dimensions=model.cfg.latent)
        collected={}
        for name,path in [('train',TRAIN),('validation',VALID)]:
            data=load_dataset(path,history=8,cache_dir=Path('data/world-array-cache'));rows=selected[name]
            xs={'actual':[],'imagined':[]};baseline={'actual':[],'imagined':[]}
            with torch.no_grad():
                for offset in range(0,len(rows),16):
                    if time.monotonic()>deadline:raise TimeoutError('risk probe deadline exceeded')
                    batch={k:data[k][rows[offset:offset+16]] for k in ('frames','history_valid','previous_actions','next_frames','lost_life')}
                    x,p=features(model,batch)
                    for k in xs:xs[k].append(x[k]);baseline[k].append(p[k])
                    if offset%1024==0:print(name,'encoded',offset,flush=True)
            collected[name]={'x':{k:torch.cat(v) for k,v in xs.items()},'baseline':{k:torch.cat(v) for k,v in baseline.items()},'y':torch.from_numpy(unsafe(data)[rows].ravel()),'unreachable':torch.from_numpy((data['distances'][rows]<0).ravel())}
            del data;persist()
        # Reproduce fixed diagnostic histories, capture features after public action selection.
        old=json.loads(FATAL.read_text());specs={s['seed']:s for s in map(json.loads,BANK.read_text().splitlines())};captured=[];original=fatal.event_report
        def capture(model,encoding,logits,extra,targets,frames,actions,choice):
            f,v,p=fatal.wd.history_arrays(frames,actions,8)
            batch={'frames':f[None],'history_valid':v[None],'previous_actions':p[None],'next_frames':targets['next_frames'][None],'lost_life':targets['lost_life'][None]}
            x,baseline=features(model,batch);captured.append({'x':x,'baseline':baseline,'y':torch.from_numpy(unsafe(targets).ravel()),'unreachable':torch.from_numpy((targets['distances']<0).ravel())})
            return original(model,encoding,logits,extra,targets,frames,actions,choice)
        saved=[]
        with patch.object(fatal,'event_report',side_effect=capture):
            for previous in old['levels']:
                replay,histories=fatal.probe(specs[previous['seed']],model,deadline);saved.extend(histories)
                if replay['ending']!=previous['ending'] or replay['stats']['checked_actor_transitions']!=previous['stats']['checked_actor_transitions']:raise ValueError('fatal replay diverged')
                if set(replay['events'])!=set(previous['events']):raise ValueError('fatal categories diverged')
                for category,event in replay['events'].items():
                    for key in ('step','actor_action','true_distances','true_lost_life'):
                        if event[key]!=previous['events'][category][key]:raise ValueError('fatal event diverged')
        with np.load('data/world-round2-fatal-choice-histories.npz',allow_pickle=False) as archive:
            for key in ('frames','history_valid','previous_actions','seed','step'):
                if not np.array_equal(np.stack([r[key] for r in saved]),archive[key]):raise ValueError('fatal saved public history mismatch')
        # Capture tensors may be inference tensors; cloning outside inference permits head autograd.
        collected['fixed_fatal']={'x':{k:torch.cat([c['x'][k] for c in captured]).clone() for k in ('actual','imagined')},'baseline':{k:torch.cat([c['baseline'][k] for c in captured]).clone() for k in ('actual','imagined')},'y':torch.cat([c['y'] for c in captured]),'unreachable':torch.cat([c['unreachable'] for c in captured])}
        report['fatal_replay']={'histories':len(captured),'exact_saved_histories_labels_actions_match':True,'no_fatal_training':True}
        train_prevalence=float(collected['train']['y'].float().mean())
        for kind in ('actual','imagined'):
            base={name:{'unsafe':metrics(c['y'],c['baseline'][kind]),'unreachable_only':metrics(c['unreachable'],c['baseline'][kind]),'constant_train_prevalence':metrics(c['y'],np.full(len(c['y']),train_prevalence))} for name,c in collected.items()}
            results={'frozen_value_baseline':base}
            for head in ('linear','mlp64'):
                if time.monotonic()>deadline:raise TimeoutError('risk probe deadline exceeded')
                results[head]=train_probe(collected['train']['x'][kind],collected['train']['y'],{name:(c['x'][kind],c['y']) for name,c in collected.items() if name!='train'},head)
            report['conditions'][kind]=results;persist()
        if any(p.requires_grad for p in model.parameters()):raise ValueError('encoder unexpectedly trainable')
        if any(file_digest(p)!=sha for p,sha in hashes.items()):raise ValueError('source changed')
        report.update(status='complete',source_hashes_unchanged=True,encoder_gradients_none=all(p.grad is None for p in model.parameters()))
    except Exception as error:report.update(status='failed',error=repr(error));raise
    finally:persist()
if __name__=='__main__':main()
