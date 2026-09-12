"""Frozen CPU glyph-to-latent grounding wiring and sensitivity diagnostic."""
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch.nn import functional as F

from pebby.agent.world_model import load_world_checkpoint
from pebby.agent.glyph_model import GLYPH_SIZES
from pebby.agent.world_grounding import labels as grounding_labels
from tools.goal_attribute_probes import digest
from tools.query_head_diagnostic import stats


def inputs():
    previous=json.loads(Path('artifacts/world-query-head-diagnostic.json').read_text())
    fields=('frames','history_valid','previous_actions','current_triple','player_cell','current_steps','current_lives')
    parts={key:[] for key in fields}
    for source in previous['sources']:
        if digest(source['path'])!=source['sha256'] or digest(source['cache_manifest'])!=source['cache_manifest_sha256']:
            raise ValueError('input provenance changed')
        directory=Path(source['cache_manifest']).parent
        for key in fields:
            array=np.load(directory/(key+'.npy'),mmap_mode='r',allow_pickle=False)
            parts[key].append(np.array(array[source['selected_rows']]))
    return {key:np.concatenate(value) for key,value in parts.items()},previous['sources']


def gradient(value):
    return {'parameters':value.numel(),'l1':float(value.abs().sum()),'l2':float(value.norm()),
            'rms':float(value.square().mean().sqrt()),'max_abs':float(value.abs().max()),
            'nonzero':int((value!=0).sum())}


def attributes(scores,targets):
    return {name:{'accuracy':float((logits.argmax(-1)==targets[:,i]).float().mean()),
                  'ce':float(F.cross_entropy(logits,targets[:,i])),
                  'predicted_class_counts':torch.bincount(logits.argmax(-1),minlength=size).tolist(),
                  'target_class_counts':torch.bincount(targets[:,i],minlength=size).tolist()}
            for i,(name,logits,size) in enumerate(zip(('shape','color','rotation'),scores,GLYPH_SIZES))}


def run(path,data):
    sha=digest(path);model,checkpoint=load_world_checkpoint(path,'cpu')
    if not model.cfg.grounding or not model.cfg.glyph_recall:raise ValueError('required paths disabled')
    n=len(data['frames']);triples=torch.from_numpy(data['current_triple']).long()
    labels=grounding_labels(data)
    assert torch.equal(torch.stack(labels[1:4],-1),triples)
    model.zero_grad(set_to_none=True)
    collected={key:[] for key in ('state','latent','glyph','glyph_logits','glyph_grad','glyph_logits_grad','latent_grad','projector_inputs','projector_pre','grounding_pre')}
    output={key:[] for key in ('glyph','latent')}
    loss_sum=0.
    for start in range(0,n,8):
        sl=slice(start,start+8);frames=torch.from_numpy(data['frames'][sl]).long()
        tokens=model.frame_tokens(frames.flatten(0,1)).view(len(frames),8,model.tokens,-1)
        glyph_logits=model.glyph_logits(frames[:,-1]);glyph_logits.retain_grad()
        encoding=model.assemble(tokens,torch.from_numpy(data['history_valid'][sl]),torch.from_numpy(data['previous_actions'][sl]),glyph_logits=glyph_logits)
        encoding['glyph'].retain_grad();encoding['latent'].retain_grad()
        scores=model.grounding_head(encoding['latent'])[1:4]
        # Exact contribution of CURRENT shape/color/rotation to weight=1 grounding:
        # six fields averaged, then current/actual/imagined averaged. Other losses omitted.
        loss=sum(F.cross_entropy(score,triples[sl,i]) for i,score in enumerate(scores))/18*len(frames)/n
        loss_sum+=float(loss.detach());loss.backward()
        pinputs=model.projector_inputs(encoding['state'],glyph=encoding['glyph']).detach()
        with torch.no_grad():
            for key in ('state','latent','glyph'):collected[key].append(encoding[key].detach())
            collected['glyph_logits'].append(glyph_logits.detach())
            collected['glyph_grad'].append(encoding['glyph'].grad.detach())
            collected['glyph_logits_grad'].append(glyph_logits.grad.detach())
            collected['latent_grad'].append(encoding['latent'].grad.detach())
            collected['projector_inputs'].append(pinputs)
            collected['projector_pre'].append(model.projector[0](pinputs))
            collected['grounding_pre'].append(model.grounding_head.head[0](encoding['latent']))
            output['glyph'].append(glyph_logits.detach())
            output['latent'].append(torch.cat(scores,-1).detach())
    collected={key:torch.cat(value) for key,value in collected.items()}
    output={key:torch.cat(value).split(GLYPH_SIZES,-1) for key,value in output.items()}
    result={'checkpoint':str(path),'sha256':sha,'config':model.config(),'current_attribute_grounding_contribution':loss_sum,
            'glyph_attributes':attributes(output['glyph'],triples),'latent_attributes':attributes(output['latent'],triples),
            'feature_gradients':{key:gradient(collected[key]) for key in ('glyph_grad','glyph_logits_grad','latent_grad')},
            'parameter_gradients':{}}
    for name,parameter in model.named_parameters():
        if name.startswith(('glyph_encoder.','glyph_context.','projector.','grounding_head.')):
            result['parameter_gradients'][name]=gradient(parameter.grad) if parameter.grad is not None else {'no_gradient':True,'parameters':parameter.numel()}
    first=model.projector[0];base=model.base_inputs
    result['projector_blocks']={}
    with torch.no_grad():
        for name,sl in [('reduced_board_and_hud',slice(0,base)),('glyph',slice(base,base+14))]:
            contribution=F.linear(collected['projector_inputs'][:,sl],first.weight[:,sl])
            result['projector_blocks'][name]={'weight':stats(first.weight[:,sl]),'weight_gradient':gradient(first.weight.grad[:,sl]),
                 'input':stats(collected['projector_inputs'][:,sl]),'linear_contribution':stats(contribution),
                 'across_state_centered_linear_contribution':stats(contribution-contribution.mean(0,keepdim=True))}
        result['interventions']={}
        original=collected['latent'];old_scores=model.grounding_head(original)[1:4]
        for mode in ('tail_permutation','tail_uniform','tail_teacher_onehot'):
            glyph=collected['glyph'].clone()
            if mode=='tail_permutation':glyph=glyph.roll(1,0)
            elif mode=='tail_uniform':glyph=torch.cat([torch.full((n,size),1/size) for size in GLYPH_SIZES],dim=-1)
            else:glyph=torch.cat([F.one_hot(triples[:,i],size) for i,size in enumerate(GLYPH_SIZES)],dim=-1).float()
            latent=model.projector(model.projector_inputs(collected['state'],glyph=glyph))
            scores=model.grounding_head(latent)[1:4]
            result['interventions'][mode]={'latent_delta':stats(latent-original),
                   'latent_delta_rms_over_original_rms':float((latent-original).square().mean().sqrt()/original.square().mean().sqrt()),
                   'attributes_against_original_labels':attributes(scores,triples),
                   'changed_predictions':{name:int((a.argmax(-1)!=b.argmax(-1)).sum()) for name,a,b in zip(('shape','color','rotation'),scores,old_scores)},
                   'logit_delta_rms':{name:float((a-b).square().mean().sqrt()) for name,a,b in zip(('shape','color','rotation'),scores,old_scores)}}
        result['gelu_inputs']={}
        for key in ('projector_pre','grounding_pre'):
            pre=collected[key];der=.5*(1+torch.erf(pre/2**.5))+pre*torch.exp(-pre.square()/2)/(2*torch.pi)**.5
            result['gelu_inputs'][key]={'stats':stats(pre),'negative_fraction':float((pre<0).float().mean()),
                  'absolute_derivative_below_1e_minus3':float((der.abs()<1e-3).float().mean())}
        result['glyph_confidence']=[float(scores.softmax(-1).max(-1).values.mean()) for scores in output['glyph']]
        # Full glyph-path permutation additionally changes the broadcast into encoder tokens.
        changed=[]
        for start in range(0,n,8):
            sl=slice(start,start+8);frames=torch.from_numpy(data['frames'][sl]).long()
            tokens=model.frame_tokens(frames.flatten(0,1)).view(len(frames),8,model.tokens,-1)
            enc=model.assemble(tokens,torch.from_numpy(data['history_valid'][sl]),torch.from_numpy(data['previous_actions'][sl]),
                               glyph_logits=collected['glyph_logits'].roll(1,0)[sl])
            changed.append(enc['latent'])
        changed=torch.cat(changed);scores=model.grounding_head(changed)[1:4]
        result['interventions']['full_glyph_path_permutation']={'latent_delta':stats(changed-original),
                   'attributes_against_original_labels':attributes(scores,triples),
                   'changed_predictions':{name:int((a.argmax(-1)!=b.argmax(-1)).sum()) for name,a,b in zip(('shape','color','rotation'),scores,old_scores)}}
    if digest(path)!=sha:raise ValueError('checkpoint changed')
    result['checkpoint_unchanged']=True
    return result


def conflict_diagnostic(sources):
    from pebby.agent.world_model import (REQUIRED_ARRAYS, OPTIONAL_ARRAYS, DEFAULT_WEIGHTS)
    from pebby.agent.world_training_objectives import world_losses
    fields=set(REQUIRED_ARRAYS)|set(OPTIONAL_ARRAYS)
    parts={key:[] for key in fields}
    for source in sources:
        directory=Path(source['cache_manifest']).parent
        for key in fields:
            path=directory/(key+'.npy')
            if path.exists():
                array=np.load(path,mmap_mode='r',allow_pickle=False)
                parts[key].append(np.array(array[source['selected_rows']]))
    batch={key:torch.from_numpy(np.concatenate(value)) for key,value in parts.items() if len(value)==len(sources)}
    weights={**DEFAULT_WEIGHTS,'sigreg':.0125,'grounding':1.,'glyph':1.,'successor_policy':0.}
    results=[]
    for path in ('checkpoints/ls20-world-query-glyph-b1024.epoch8.pt','checkpoints/ls20-world-combined-control-b1024.epoch2.pt'):
        sha=digest(path);model,_=load_world_checkpoint(path,'cpu')
        model.encoder_chunk_size=8;model.checkpoint_encoder=True
        out=world_losses(model,batch,weights=weights,sigreg_generator=torch.Generator().manual_seed(113))
        parameter=model.projector[0].weight
        gradients={}
        for name,loss in out['losses'].items():
            g=torch.autograd.grad(loss*weights[name],parameter,retain_graph=True,allow_unused=True)[0]
            gradients[name]=torch.zeros_like(parameter) if g is None else g.detach()
        total=torch.autograd.grad(out['total'],parameter)[0].detach()
        record={'checkpoint':path,'sha256':sha,'batch_size':64,'weights':weights,
                'losses':{key:float(value.detach()) for key,value in out['losses'].items()},'blocks':{}}
        for name,sl in [('glyph',slice(model.base_inputs,model.base_inputs+14)),('base',slice(0,model.base_inputs))]:
            ground=gradients['grounding'][:,sl].flatten();full=total[:,sl].flatten()
            compare={}
            for term,g in {**gradients,'total':total,'total_without_sigreg':total-gradients['sigreg']}.items():
                v=g[:,sl].flatten();den=ground.norm()*v.norm();dot=float(ground@v)
                compare[term]={'rms':float(v.square().mean().sqrt()),'dot_with_grounding':dot,
                               'cosine_with_grounding':float((ground@v)/den) if float(den)>0 else None}
            record['blocks'][name]=compare
        if digest(path)!=sha:raise ValueError('checkpoint changed')
        results.append(record);print(path,'gradient comparison done',flush=True)
    return {'results':results,'caveat':'Exact configured weighted losses on this fixed64-state FP32 batch. SIGReg depends on batch size and sampled projections; this is not the full1024-state training gradient. Comparison also reports total minus SIGReg.'}


def main():
    torch.set_num_threads(1);started=time.monotonic();output=Path('artifacts/world-glyph-latent-grounding-diagnostic.json')
    if output.exists():raise ValueError('refusing overwrite')
    print('PID',os.getpid(),flush=True);data,sources=inputs()
    report={'status':'running','pid':os.getpid(),'device':'cpu','torch_threads':1,'states':64,'sources':sources,'models':[],
            'objective':'Current shape/color/rotation terms in actual grounding objective: sum of three CE means /18; no auxiliary objectives, optimizer, or clipping.',
            'intervention_note':'Tail interventions hold refined board tokens fixed; full-path permutation also alters glyph broadcast. These create inconsistent diagnostic inputs, not valid game observations.',
            'caveats':['Small generated training-state diagnostic, not a generalization bound or proof that latent information is absent.',
                       'Existing source-hash-bound caches read only; entire large cached arrays not rehashed.',
                       'Softmax confidence can suppress gradients into the already accurate classifier without disconnecting glyph-probability inputs.'],
            'code_hashes':{p:digest(p) for p in (__file__,'pebby/agent/world_model.py','pebby/agent/world_training_objectives.py','pebby/agent/world_grounding.py','pebby/agent/glyph_model.py')}}
    def persist():
        report['elapsed_seconds']=time.monotonic()-started;report['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        temp=output.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(output)
    persist()
    try:
        for path in ('checkpoints/ls20-world-query-glyph-b1024.epoch8.pt','checkpoints/ls20-world-combined-control-b1024.epoch2.pt'):
            report['models'].append(run(Path(path),data));print(path,'done',flush=True);persist()
        for path,sha in report['code_hashes'].items():
            if digest(path)!=sha:raise ValueError('source code changed')
        report['status']='complete'
    except Exception as error:
        report.update(status='failed',error=repr(error));raise
    finally:persist()

if __name__=='__main__':main()
