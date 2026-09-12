"""Generated mixed H4 training: exploratory75% + closing25%, unique levels per batch."""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from pebby.agent.structured_sequence_objective import sequence_objective
from tools.train_structured_sequences import load_cache, new_model, sequence_batch
from tools.train_structured_transition import (LABELS, EVENTS, atomic_json, autocast,
    digest, event_counts, sample_rows, training_scale)

EXPLORATORY_FORMAT = 'pebby.structured-exploratory-sequence-cache.v1'


def load_exploratory_cache(path):
    """Accept TRAIN exploratory chronology explicitly; never relabel it validation."""
    path = Path(path); manifest = json.loads((path/'manifest.json').read_text())
    if (manifest.get('format') != EXPLORATORY_FORMAT or manifest.get('status') != 'complete'
            or manifest.get('source') != 'generated_only' or manifest.get('split') != 'train'
            or manifest.get('mode') != 'exploratory_chronological_K4'
            or not manifest.get('history_verified_against_actual_source_rows')
            or not manifest.get('no_future_inputs_to_current_field') or not manifest.get('field_encoder')):
        raise ValueError('requires verified exploratory TRAIN chronology')
    index = manifest.get('source_index_metadata', {})
    if (index.get('format') != 'pebby.ls20-exploratory-train-four-step-index.v1'
            or index.get('mode') != 'explore_train' or index.get('source') != 'generated_only'
            or index.get('split') != 'train' or index.get('K') != 4 or index.get('history') != 8):
        raise ValueError('exploratory source index provenance mismatch')
    if not manifest.get('source_hashes') or index.get('source_sha256') not in manifest['source_hashes'].values():
        raise ValueError('exploratory index missing source hash binding')
    arrays = {}
    for name, info in manifest['arrays'].items():
        file = path/(name+'.npy')
        if digest(file) != info['sha256']:
            raise ValueError(f'array checksum mismatch: {name}')
        a = np.load(file, mmap_mode='r', allow_pickle=False)
        if list(a.shape) != info['shape'] or str(a.dtype) != info['dtype']:
            raise ValueError(f'array shape/dtype mismatch: {name}')
        for start in range(0, len(a), 64):
            if not np.isfinite(a[start:start+64]).all():
                raise ValueError(f'nonfinite array: {name}')
        arrays[name] = a
    seeds = arrays['seeds']; n = len(seeds)
    if (n < 1 or len(np.unique(seeds)) != n or not np.issubdtype(seeds.dtype,np.integer)
            or not ((seeds >= 0) & (seeds < 1_000_000)).all()):
        raise ValueError('exploratory cache requires distinct generated TRAIN seeds')
    shapes = {'fields':(n,148,96),'next_fields':(n,4,148,96),'actions':(n,4),
              'player_cell':(n,2),'next_player_cell':(n,4,2),'triple':(n,3),'next_triple':(n,4,3),
              'source_rows':(n,),'future_rows':(n,4),'distances':(n,4),'next_optimal':(n,4),'optimal':(n,)}
    for name in ('fields','next_fields','actions','difficulties','source_rows','future_rows','distances','optimal','next_optimal',*LABELS):
        a = arrays[name]; expected = shapes.get(name,(n,4) if name.startswith('next_') or name in EVENTS else (n,))
        if a.shape != expected:
            raise ValueError(f'incorrect chronological shape: {name}')
        if name in ('fields','next_fields'):
            if not np.issubdtype(a.dtype,np.floating):raise ValueError('fields must be floating')
        elif name in EVENTS:
            if not np.isin(a,[0,1]).all():raise ValueError('events must be binary')
        elif not np.issubdtype(a.dtype,np.integer):raise ValueError(f'integer array required: {name}')
    if not np.isin(arrays['actions'],np.arange(4)).all() or not np.isin(arrays['difficulties'],np.arange(1,6)).all():
        raise ValueError('invalid action/difficulty')
    if not np.array_equal(arrays['future_rows'], arrays['source_rows'][:,None]+np.arange(1,5)):
        raise ValueError('exploratory chronological source links mismatch')
    if any(arrays[name].any() for name in EVENTS) or (arrays['distances']<0).any():
        raise ValueError('exploratory clips must remain live/reachable at all horizons')
    return arrays, manifest


class InsufficientMixedLevels(ValueError):
    pass


def source_counts(batch_size):
    if batch_size < 1:raise ValueError('positive batch size required')
    closing = batch_size//4 if batch_size>=4 else int(batch_size==2)
    return batch_size-closing, closing


def mixed_rows(live, closing, batch_size, progress, rng):
    """Draw closing first; exclude its seeds from weighted live selection."""
    live_count, closing_count = source_counts(batch_size)
    if closing_count > len(closing['seeds']):raise InsufficientMixedLevels('insufficient closing levels')
    close_rows = sample_rows(closing,closing_count,progress,rng) if closing_count else np.empty(0,np.int64)
    available = np.flatnonzero(~np.isin(live['seeds'],closing['seeds'][close_rows]))
    if live_count > len(available):raise InsufficientMixedLevels('insufficient live levels after closing exclusions')
    view = {key:live[key][available] for key in ('seeds','difficulties')}
    live_rows = available[sample_rows(view,live_count,progress,rng)]
    seeds = np.concatenate((live['seeds'][live_rows],closing['seeds'][close_rows]))
    if len(np.unique(seeds)) != batch_size:raise RuntimeError('repeated level within mixed batch')
    return live_rows, close_rows


def mixed_batch(live,closing,rows,device):
    """Concatenate chronological sequences, then make ONE full-batch objective call."""
    lr,cr=rows
    batches=[sequence_batch(data,r,device) for data,r in ((live,lr),(closing,cr)) if len(r)]
    return tuple(torch.cat([b[i] for b in batches]) for i in range(3)) + (
        {name:torch.cat([b[3][name] for b in batches]) for name in LABELS},)


def event_weights(live,closing):
    rates=np.asarray([.75*np.asarray(live[name],bool).mean()+.25*np.asarray(closing[name],bool).mean() for name in EVENTS])
    return rates,np.clip((1-rates)/np.maximum(rates,1e-8),1,20).astype(np.float32)


def probe_batch(live,closing,scale,positive,args):
    size=args.max_batch; attempts=[]
    while size:
        model=optimizer=result=batch=None;started=time.monotonic()
        try:
            rows=mixed_rows(live,closing,size,0.,np.random.default_rng(args.seed))
            model=new_model(args).train();optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
            if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
            batch=mixed_batch(live,closing,rows,args.device)
            with autocast(args.device):
                result=sequence_objective(model,*batch,scale,pos_weight=positive,checkpoint_steps=args.checkpoint_steps)
            if not bool(torch.isfinite(result['total'])):raise ValueError('nonfinite mixed preflight loss')
            result['total'].backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
            optimizer.step()
            if args.device=='cuda':torch.cuda.synchronize()
            attempts.append(dict(batch_size=size,live_rows=len(rows[0]),closing_rows=len(rows[1]),
                status='fits',seconds=time.monotonic()-started,gradient_norm=float(norm),
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None))
            return size,attempts
        except (torch.cuda.OutOfMemoryError,InsufficientMixedLevels) as error:
            attempts.append(dict(batch_size=size,status=type(error).__name__,reason=str(error)));size//=2
        finally:
            del model,optimizer,result,batch
            gc.collect()
            if args.device=='cuda':torch.cuda.empty_cache()
    raise RuntimeError('no mixed batch fits')


def exposure(seen):
    union=seen['live']|seen['closing']
    return {'distinct_train_levels_seen':len(union),'distinct_live_levels_seen':len(seen['live']),
            'distinct_closing_levels_seen':len(seen['closing'])}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('live-cache','closing-cache','validation-cache','checkpoint','report'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--initialize',default='checkpoints/ls20-structured-transition-fit2000.pt')
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--updates',type=int,default=400);p.add_argument('--max-batch',type=int,default=1024)
    p.add_argument('--eval-batch',type=int,default=32);p.add_argument('--loops',type=int,default=2)
    p.add_argument('--lr',type=float,default=.0003);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--seconds',type=int,default=1200);p.add_argument('--checkpoint-steps',action='store_true')
    p.add_argument('--preflight-only',action='store_true');args=p.parse_args()
    if not 1<=args.max_batch<=1024 or args.max_batch&(args.max_batch-1):p.error('power-of-two max batch1..1024 required')
    if not 1<=args.updates<=2000 or not 1<=args.seconds<=1800 or args.eval_batch<1:p.error('bounded updates/time and positive evaluation batch required')
    if not np.isfinite(args.lr) or args.lr<=0:p.error('learning rate must be finite positive')
    if Path(args.report).exists() or Path(args.checkpoint).exists():p.error('refusing existing output')
    torch.set_num_threads(1);start=time.monotonic();print('PID',os.getpid(),flush=True)
    report=dict(status='running',pid=os.getpid(),args=vars(args),scope='mixed generated chronological H4 training',
        policy_integrated=False,official_inputs_used=False,training=[],
        limitations=['No policy/search/control integration.','Current live/closing pools have zero life-reset or terminalfailure clips.',
                     'Heldout live-only sequences cannot assess ending/reset risk.',
                     'Event weights use fixed75/25 source transition rates, not validation or changing curriculum prevalence.',
                     '75/25 is a row sampling ratio; changed-cell/visibility/goal-weighted losses normalize over active mass and need not have75/25 per-bank gradient contributions.'])
    atomic_json(args.report,report)
    def expired(*_):raise TimeoutError('bounded mixed H4 deadline')
    signal.signal(signal.SIGALRM,expired);signal.alarm(args.seconds)
    try:
        live,lm=load_exploratory_cache(args.live_cache);closing,cm=load_cache(args.closing_cache,'train')
        val,vm=load_cache(args.validation_cache,'validation')
        if lm['field_encoder']!=cm['field_encoder'] or lm['field_encoder']!=vm['field_encoder']:
            raise ValueError('all three field encoders must match')
        if np.intersect1d(np.union1d(live['seeds'],closing['seeds']),val['seeds']).size:
            raise ValueError('validation level leakage')
        sources={}
        manifests={'live_train':lm,'closing_train':cm,'validation':vm}
        for directory,manifest in ((args.live_cache,lm),(args.closing_cache,cm),(args.validation_cache,vm)):
            sources[str(Path(directory)/'manifest.json')]=digest(Path(directory)/'manifest.json')
            sources.update({str(Path(directory)/(name+'.npy')):info['sha256'] for name,info in manifest['arrays'].items()})
        for file in (__file__,'tools/train_structured_sequences.py','tools/train_structured_transition.py',
                     'tools/structured_sequence_metrics.py','pebby/agent/structured_transition.py',
                     'pebby/agent/structured_sequence_objective.py','pebby/agent/structured_objective.py'):
            sources[str(file)]=digest(file)
        sources[args.initialize]=digest(args.initialize)
        saved=torch.load(args.initialize,map_location='cpu',weights_only=False)
        if saved.get('format')!='pebby.structured-transition.v1' or any(m['field_encoder']!=lm['field_encoder'] for m in saved['cache_manifests'].values()):
            raise ValueError('warmstart format or encoder mismatch')
        scale=torch.as_tensor(training_scale(live),device=args.device)
        rates,weights=event_weights(live,closing);positive=torch.as_tensor(weights,device=args.device)
        report.update(sources=sources,available_levels={name:len(data['seeds']) for name,data in [('live',live),('closing',closing),('validation',val)]},
            available_train_union=int(len(np.union1d(live['seeds'],closing['seeds']))),
            event_coverage={name:event_counts(data) for name,data in [('live',live),('closing',closing),('validation',val)]},
            feature_scale=scale.tolist(),feature_scale_source='live TRAIN current fields only',
            event_mixture_rates=rates.tolist(),event_positive_weights=weights.tolist())
        size,attempts=probe_batch(live,closing,scale,positive,args)
        report.update(batch_size=size,batch_source_counts=dict(zip(('live','closing'),source_counts(size))),batch_probe=attempts)
        atomic_json(args.report,report)
        if args.preflight_only:report['status']='preflight_complete'
        else:
            from tools.structured_sequence_metrics import evaluate
            model=new_model(args).train()  # Fresh exact warmstart after disposable probe.
            optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
            rng=np.random.default_rng(args.seed);seen={'live':set(),'closing':set()}
            draws={'live':np.zeros(5,np.int64),'closing':np.zeros(5,np.int64)}
            report['parameters']=model.parameter_count()
            for step in range(args.updates):
                rows=mixed_rows(live,closing,size,step/max(args.updates-1,1),rng)
                batch=mixed_batch(live,closing,rows,args.device);optimizer.zero_grad(set_to_none=True)
                with autocast(args.device):
                    result=sequence_objective(model,*batch,scale,pos_weight=positive,checkpoint_steps=args.checkpoint_steps)
                if not bool(torch.isfinite(result['total'])):raise ValueError('nonfinite mixed H4 loss')
                result['total'].backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
                optimizer.step()
                for name,data,indices in [('live',live,rows[0]),('closing',closing,rows[1])]:
                    seen[name].update(map(int,data['seeds'][indices]));draws[name]+=np.bincount(data['difficulties'][indices],minlength=6)[1:6]
                if step==0 or (step+1)%20==0 or step+1==args.updates:
                    entry=dict(step=step+1,loss=float(result['total'].detach()),gradient_norm=float(norm),
                        elapsed_seconds=time.monotonic()-start,
                        source_counts={'live':len(rows[0]),'closing':len(rows[1])},
                        losses={k:float(v.detach()) for k,v in result['losses'].items()},**exposure(seen))
                    report['training'].append(entry);report.update(completed_updates=step+1,**exposure(seen),
                        difficulty_draws={name:values.tolist() for name,values in draws.items()})
                    atomic_json(args.report,report);print(json.dumps(entry),flush=True)
                del result,batch
            if any(digest(path)!=sha for path,sha in sources.items()):raise ValueError('inputs changed during training')
            report['seen_train_seeds']=sorted(seen['live']|seen['closing'])
            report['seen_seeds_by_bank']={name:sorted(values) for name,values in seen.items()}
            target=Path(args.checkpoint);target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+f'.{os.getpid()}.tmp')
            torch.save(dict(format='pebby.structured-transition.v1',config=model.config(),
                weights={k:v.detach().cpu() for k,v in model.state_dict().items()},sources=sources,cache_manifests=manifests,
                feature_scale=scale.cpu(),event_positive_weights=positive.cpu(),updates=args.updates,batch_size=size,
                batch_source_counts=report['batch_source_counts'],seed=args.seed,objective='mixed_autoregressive_H4',
                initialize=args.initialize,exposure=exposure(seen),policy_integrated=False),temporary)
            os.replace(temporary,target)
            report.update(status='evaluation_running',checkpoint=str(target),checkpoint_sha256=digest(target))
            atomic_json(args.report,report)
            for name,data in [('validation',val),('live_train',live),('closing_train',closing)]:
                report[name+'_evaluation']=evaluate(model,data,scale,args.device,args.eval_batch)
                atomic_json(args.report,report)
            report['status']='complete'
        if any(digest(path)!=sha for path,sha in sources.items()):raise ValueError('experiment sources changed')
        report.update(source_unchanged=True,peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None)
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        signal.alarm(0);report['elapsed_seconds']=time.monotonic()-start;atomic_json(args.report,report)


if __name__=='__main__':main()
