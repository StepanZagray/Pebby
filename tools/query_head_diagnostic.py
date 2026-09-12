"""CPU-only frozen-weight query-head activation and policy-gradient diagnostic."""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch.nn import functional as F

from pebby.agent.world_model import load_world_checkpoint, optimal_bits
from tools.goal_attribute_probes import digest, stratified_seeds

FIELDS=('frames','history_valid','previous_actions','optimal','seeds')


def select_rows(source,policy_only,excluded):
    source=Path(source);sha=digest(source)
    candidates=[]
    for manifest_path in Path('data/world-array-cache').glob('*/manifest.json'):
        manifest=json.loads(manifest_path.read_text())
        if manifest['source_sha256']==sha:
            candidates.append((manifest_path,manifest))
    if len(candidates)!=1:
        raise ValueError('requires one existing source-bound cache; never creates or alters a cache')
    manifest_path,manifest=candidates[0];directory=manifest_path.parent
    arrays={key:np.load(directory/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in (*FIELDS,'meta')}
    meta=json.loads(str(arrays.pop('meta').item()))
    if meta.get('source')!='generated_only' or meta.get('oracle_search')!='complete_only':
        raise ValueError('unverified source')
    eligible=np.asarray(meta['on_policy_rows'],dtype=int) if policy_only else np.arange(len(arrays['seeds']))
    seeds=arrays['seeds'][eligible]
    if np.any((seeds<0)|(seeds>=1_000_000)):
        raise ValueError('training states only')
    eligible_seeds=set(map(int,seeds))
    level_specs={int(level['seed']):level for level in meta['levels']
                 if int(level['seed']) in eligible_seeds and int(level['seed']) not in excluded}
    selected=stratified_seeds(level_specs,32,20260912+int(policy_only));rng=np.random.default_rng(629+int(policy_only))
    indices=[int(rng.choice(eligible[seeds==seed])) for seed in selected]
    data={key:np.array(value[indices]) for key,value in arrays.items()}
    if (data['optimal']==0).any():raise ValueError('unlabelable training state')
    selected_hash=hashlib.sha256()
    for key in FIELDS:selected_hash.update(key.encode());selected_hash.update(data[key].tobytes())
    provenance={'path':str(source),'sha256':sha,'cache_manifest':str(manifest_path),'cache_manifest_sha256':digest(manifest_path),
                'selected_rows':indices,'selected_seeds':selected,'selected_data_sha256':selected_hash.hexdigest(),
                'difficulty_counts':{str(d):sum(level_specs[s]['difficulty']==d for s in selected) for d in range(1,6)},
                'cache_note':'Existing read-only cache tied to exact NPZ source hash; full cache arrays are not rehashed by this diagnostic.'}
    return data,provenance


def stats(t):
    t=t.detach().float().reshape(-1)
    return {'count':t.numel(),'mean':float(t.mean()),'std':float(t.std(unbiased=False)),
            'rms':float(t.square().mean().sqrt()),'min':float(t.min()),'max':float(t.max())}


def gradient_stats(model):
    groups={}
    for name,parameter in model.named_parameters():
        key=name.rsplit('.',1)[0] if name.startswith('query_head.') else name.split('.')[0]
        group=groups.setdefault(key,{'parameters':0,'gradient_elements':0,'l1':0.,'squared_l2':0.,'max_abs':0.,'nonzero':0})
        group['parameters']+=parameter.numel()
        if parameter.grad is not None:
            g=parameter.grad.detach();group['gradient_elements']+=g.numel();group['l1']+=float(g.abs().sum())
            group['squared_l2']+=float(g.square().sum());group['max_abs']=max(group['max_abs'],float(g.abs().max()))
            group['nonzero']+=int((g!=0).sum())
    for g in groups.values():
        g['l2']=math.sqrt(g.pop('squared_l2'));g['rms']=g['l2']/math.sqrt(g['parameters'])
    return groups


def evaluate(path,batches):
    sha=digest(path);model,_=load_world_checkpoint(path,'cpu')
    if not model.cfg.query_readout:raise ValueError('checkpoint lacks query head')
    result={'checkpoint':str(path),'sha256':sha,'parameters':model.parameter_count(),'query_parameters':model.query_head.parameter_count(),'sources':{}}
    for source,data in batches.items():
        captured={};handles=[]
        for name,module in model.query_head.named_modules():
            if isinstance(module,torch.nn.GELU):
                def hook(module,inputs,output,name=name):
                    captured.setdefault(name,[]).append(inputs[0].detach().clone())
                handles.append(module.register_forward_hook(hook))
        values={key:[] for key in ('query','direct','ranker','total')};ce=0.;model.zero_grad(set_to_none=True)
        for start in range(0,32,8):
            sl=slice(start,start+8)
            encoding=model.encode(torch.from_numpy(data['frames'][sl]).long(),torch.from_numpy(data['history_valid'][sl]),torch.from_numpy(data['previous_actions'][sl]))
            logits,extra=model.logits_from(encoding)
            for key,value in [('query',extra['query']),('direct',extra['direct']),('ranker',logits-extra['query']-extra['direct']),('total',logits)]:
                values[key].append(value.detach())
            bits=optimal_bits(torch.from_numpy(data['optimal'][sl]));targets=bits/bits.sum(-1,keepdim=True)
            loss=-(targets*F.log_softmax(logits,dim=-1)).sum()/32
            ce+=float(loss.detach());loss.backward()
        for handle in handles:handle.remove()
        values={key:torch.cat(value) for key,value in values.items()}
        components={}
        for key,value in values.items():
            components[key]={'raw':stats(value),'action_span':stats(value.max(-1).values-value.min(-1).values),
                             'action_centered':stats(value-value.mean(-1,keepdim=True)),
                             'per_action_across_state_std':value.std(0,unbiased=False).tolist()}
        activations={}
        for name,parts in captured.items():
            pre=torch.cat(parts);derivative=.5*(1+torch.erf(pre/math.sqrt(2)))+pre*torch.exp(-pre.square()/2)/math.sqrt(2*math.pi)
            activations[name]={'preactivation':stats(pre),'negative_fraction':float((pre<0).float().mean()),
                              'below_minus_three_fraction':float((pre<-3).float().mean()),
                              'absolute_gelu_derivative_below_1e_minus3_fraction':float((derivative.abs()<1e-3).float().mean()),
                              'gelu_derivative':stats(derivative),'activation':stats(F.gelu(pre))}
        total=values['total'];ablated=total-values['query'];masks=torch.from_numpy(data['optimal']).long()
        correct=lambda logits: ((masks&(1<<logits.argmax(-1)))!=0)
        bits=optimal_bits(masks);targets=bits/bits.sum(-1,keepdim=True)
        ablated_ce=float(-(targets*F.log_softmax(ablated,dim=-1)).sum(-1).mean())
        result['sources'][source]={'policy_only_ce':ce,'components':components,'query_gelu':activations,
             'policy_only_gradients_no_clipping':gradient_stats(model),
             'query_zero_ablation':{'changed_actions':int((total.argmax(-1)!=ablated.argmax(-1)).sum()),
                                   'full_correct':int(correct(total).sum()),'ablated_correct':int(correct(ablated).sum()),
                                   'ablated_policy_ce':ablated_ce,'ce_increase_when_query_removed':ablated_ce-ce}}
    weights={name:p.detach().clone() for name,p in model.query_head.named_parameters()}
    result['query_weight_norms']={name:stats(value) for name,value in weights.items()}
    # A newly introduced head has zero final weight/bias; other initial values
    # depend on an unrecorded RNG state and cannot be reconstructed honestly.
    result['distance_from_fresh_zero_output']={name:float(value.norm()) for name,value in weights.items() if name.startswith('output.2.')}
    if digest(path)!=sha:raise ValueError('checkpoint changed')
    result['checkpoint_unchanged']=True
    return result,weights


def main():
    torch.set_num_threads(1);started=time.monotonic()
    output=Path('artifacts/world-query-head-diagnostic.json')
    if output.exists():raise ValueError('refusing diagnostic overwrite')
    print('PID',os.getpid(),flush=True)
    ordinary,ordinary_meta=select_rows('data/ls20-world-combined-train.npz',False,set())
    policy,policy_meta=select_rows('data/ls20-world-onpolicy-1024-train.npz',True,set(map(int,ordinary['seeds'])))
    report={'status':'running','pid':os.getpid(),'device':'cpu','torch_threads':1,'batch_size':8,'states':64,
            'sources':[ordinary_meta,policy_meta],'models':[],
            'method':'Policy-only mean uniform-optimal-target CE, separate 32-state source means; autograd only, no optimizer or gradient clipping.',
            'caveats':['Different parameter counts make raw gradient L1 totals incomparable as causal importance.',
                       'This is a small stratified generated training-state diagnostic, not rollout success or population risk.',
                       'No SIGReg, dynamics, glyph, grounding or successor losses are included.',
                       'A newly added head final output is exactly zero; its unrecorded random upstream initialization cannot be reconstructed.',
                       'GELU positive large inputs have derivative near one, not saturation; low absolute derivative threshold is reported explicitly.'],
            'code_hashes':{str(path):digest(path) for path in (Path(__file__),Path('pebby/agent/world_model.py'),Path('pebby/agent/world_readout.py'))}}
    def persist():
        report['elapsed_seconds']=time.monotonic()-started;report['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        temp=output.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(output)
    persist();prior=None
    try:
        paths=[Path(f'checkpoints/ls20-world-query-glyph-b1024.epoch{epoch}.pt') for epoch in (1,4,8)]
        paths.append(Path('checkpoints/ls20-world-combined-control-b1024.epoch2.pt'))
        for path in paths:
            result,weights=evaluate(path,{'ordinary':ordinary,'on_policy':policy})
            if prior is not None:
                result['query_change_from_previous_checkpoint']={name:{'l2':float((value-prior[name]).norm()),
                        'relative_l2_to_previous':float((value-prior[name]).norm()/(prior[name].norm()+1e-30))} for name,value in weights.items()}
            prior=weights;report['models'].append(result);print(path,'done',flush=True);persist()
        for source in report['sources']:
            if digest(source['path'])!=source['sha256'] or digest(source['cache_manifest'])!=source['cache_manifest_sha256']:
                raise ValueError('source provenance changed')
        for path,sha in report['code_hashes'].items():
            if digest(path)!=sha:raise ValueError('code changed during diagnostic')
        report.update(status='complete',sources_and_code_unchanged=True)
    except Exception as error:
        report.update(status='failed',error=repr(error));raise
    finally:persist()

if __name__=='__main__':main()
