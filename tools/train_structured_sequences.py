"""Bounded generated-only H4 training with true recurrent prediction gradients."""
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
from pebby.agent.structured_transition import StructuredTransition
from tools.train_structured_transition import (LABELS, EVENTS, atomic_json, autocast,
    digest, event_counts, sample_rows, training_scale)


FORMATS = {'train': 'pebby.structured-closing-sequence-cache.v1',
           'validation': 'pebby.structured-sequence-cache.v1'}


def load_cache(path, split):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if (manifest.get('format') != FORMATS[split] or manifest.get('status') != 'complete'
            or manifest.get('source') != 'generated_only' or manifest.get('split') != split
            or not manifest.get('history_verified_against_actual_source_rows')
            or not manifest.get('field_encoder')):
        raise ValueError('requires verified generated chronological cache in requested split')
    if split == 'train' and not manifest.get('final_history_verified_independent_append_reset'):
        raise ValueError('closing cache requires verified final target history')
    arrays = {}
    for name, info in manifest['arrays'].items():
        file = path / (name + '.npy')
        if digest(file) != info['sha256']:
            raise ValueError(f'array checksum mismatch: {name}')
        array = np.load(file, mmap_mode='r', allow_pickle=False)
        if list(array.shape) != info['shape'] or str(array.dtype) != info['dtype']:
            raise ValueError(f'array shape/dtype mismatch: {name}')
        if not np.isfinite(array).all():
            raise ValueError(f'nonfinite cache: {name}')
        arrays[name] = array
    n = len(arrays['seeds'])
    low, high = (0, 1_000_000) if split == 'train' else (1_000_000, 2_000_000)
    if (n < 1 or len(np.unique(arrays['seeds'])) != n
            or not ((arrays['seeds'] >= low) & (arrays['seeds'] < high)).all()):
        raise ValueError('requires distinct generated levels in split namespace')
    shapes = {'fields': (n,148,96), 'next_fields': (n,4,148,96),
              'actions': (n,4), 'player_cell': (n,2), 'next_player_cell': (n,4,2),
              'triple': (n,3), 'next_triple': (n,4,3)}
    for name in ('fields', 'next_fields', 'actions', 'difficulties', *LABELS):
        expected = shapes.get(name, (n,4) if name.startswith('next_') or name in EVENTS else (n,))
        if arrays[name].shape != expected:
            raise ValueError(f'incorrect chronological array shape: {name}')
        if name in ('fields', 'next_fields'):
            if not np.issubdtype(arrays[name].dtype, np.floating):
                raise ValueError('fields must be floating')
        elif name in EVENTS:
            if not np.isin(arrays[name], [0,1]).all():
                raise ValueError('events must be binary')
        elif not np.issubdtype(arrays[name].dtype, np.integer):
            raise ValueError(f'integer labels required: {name}')
    if not np.isin(arrays['actions'], np.arange(4)).all():
        raise ValueError('invalid chronological action')
    if not np.isin(arrays['difficulties'], np.arange(1,6)).all():
        raise ValueError('difficulty outside1..5')
    if any(arrays[name][:,:3].any() for name in EVENTS):
        raise ValueError('interior ending/reset violates chronology')
    if np.any(arrays['won'].astype(bool) & ~arrays['terminal'].astype(bool)):
        raise ValueError('win must terminate')
    if split == 'validation' and any(arrays[name].any() for name in EVENTS):
        raise ValueError('current heldout cache protocol is live-only')
    return arrays, manifest


def sequence_batch(data, rows, device):
    rows = np.asarray(rows)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError('rows must be an integer vector')
    def tensor(name):
        return torch.as_tensor(np.array(data[name][rows]), device=device)
    return (tensor('fields').float(), tensor('next_fields').float(),
            tensor('actions').long(), {name: tensor(name) for name in LABELS})


def new_model(args):
    torch.manual_seed(args.seed)
    model = StructuredTransition(loops=args.loops).to(args.device)
    if args.initialize:
        saved = torch.load(args.initialize, map_location='cpu', weights_only=False)
        if saved.get('format') != 'pebby.structured-transition.v1' or saved['config'] != model.config():
            raise ValueError('initial checkpoint format/config mismatch')
        model.load_state_dict(saved['weights'], strict=True)
    return model


def probe_batch(data, scale, positive, args):
    size = 2 ** (min(args.max_batch, len(data['seeds'])).bit_length()-1)
    attempts = []
    while size:
        model = optimizer = result = batch = None
        started = time.monotonic()
        try:
            model = new_model(args).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
            if args.device == 'cuda': torch.cuda.reset_peak_memory_stats()
            batch = sequence_batch(data, np.arange(size), args.device)
            with autocast(args.device):
                result = sequence_objective(model, *batch, scale, pos_weight=positive,
                                            checkpoint_steps=args.checkpoint_steps)
            if not torch.isfinite(result['total']): raise ValueError('nonfinite probe loss')
            result['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            if args.device == 'cuda': torch.cuda.synchronize()
            attempts.append({'batch_size': size, 'status': 'fits',
                'seconds': time.monotonic()-started,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else None})
            return size, attempts
        except torch.cuda.OutOfMemoryError:
            attempts.append({'batch_size': size, 'status': 'out_of_memory'})
            size //= 2
        finally:
            del model, optimizer, result, batch
            gc.collect()
            if args.device == 'cuda': torch.cuda.empty_cache()
    raise RuntimeError('no batch fits')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train-cache', 'validation-cache', 'checkpoint', 'report'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--initialize')
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--updates', type=int, default=400)
    parser.add_argument('--max-batch', type=int, default=1024)
    parser.add_argument('--eval-batch', type=int, default=32)
    parser.add_argument('--loops', type=int, default=2)
    parser.add_argument('--lr', type=float, default=.0003)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seconds', type=int, default=1200)
    parser.add_argument('--checkpoint-steps', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.max_batch <= 1024 or args.max_batch & (args.max_batch-1):
        parser.error('batch cap must be a power of two <=1024')
    if not 1 <= args.updates <= 2000 or not 1 <= args.seconds <= 1800 or args.eval_batch < 1:
        parser.error('bounded run requires1..2000 updates and1..1800 seconds')
    if Path(args.report).exists() or Path(args.checkpoint).exists():
        parser.error('refusing to overwrite artifacts')
    torch.set_num_threads(1)
    start = time.monotonic()
    report = dict(status='running', pid=os.getpid(), args=vars(args), policy_integrated=False,
        official_inputs_used=False, scope='generated chronological H4 prediction training',
        limitations=['Training closing cache has no resets or terminal failures.',
                     'Heldout live-only cache cannot measure ending/reset generalization.',
                     'Frozen field targets contain imperfect learned semantic probabilities.',
                     'No policy/search/control integration or official completion measurement.'])
    atomic_json(args.report, report)
    def expired(*_): raise TimeoutError('bounded H4 experiment deadline')
    signal.signal(signal.SIGALRM, expired); signal.alarm(args.seconds)
    try:
        train, tm = load_cache(args.train_cache, 'train')
        val, vm = load_cache(args.validation_cache, 'validation')
        if tm['field_encoder'] != vm['field_encoder'] or np.intersect1d(train['seeds'],val['seeds']).size:
            raise ValueError('encoder mismatch or train/validation overlap')
        sources = {}
        for directory, manifest in ((args.train_cache,tm),(args.validation_cache,vm)):
            sources[str(Path(directory)/'manifest.json')] = digest(Path(directory)/'manifest.json')
            sources.update({str(Path(directory)/(name+'.npy')): info['sha256']
                            for name,info in manifest['arrays'].items()})
        for file in (__file__, 'tools/train_structured_transition.py',
                     'pebby/agent/structured_transition.py', 'pebby/agent/structured_objective.py',
                     'pebby/agent/structured_sequence_objective.py'):
            sources[str(file)] = digest(file)
        if args.initialize:
            sources[args.initialize] = digest(args.initialize)
            saved = torch.load(args.initialize, map_location='cpu', weights_only=False)
            if any(m['field_encoder'] != tm['field_encoder'] for m in saved['cache_manifests'].values()):
                raise ValueError('initial checkpoint uses different feature encoder')
        scale = torch.as_tensor(training_scale(train), device=args.device)
        prevalence = np.array([np.asarray(train[n],bool).mean() for n in EVENTS])
        positive = torch.as_tensor(np.clip((1-prevalence)/np.maximum(prevalence,1e-8),1,20),
                                   device=args.device, dtype=torch.float32)
        report.update(sources=sources,train_levels=len(train['seeds']),validation_levels=len(val['seeds']),
                      train_events=event_counts(train),validation_events=event_counts(val),
                      feature_scale=scale.tolist(),event_positive_weights=positive.tolist())
        size, attempts = probe_batch(train, scale, positive, args)
        report.update(batch_size=size,batch_probe=attempts)
        atomic_json(args.report,report)
        if args.preflight_only:
            report['status']='preflight_complete'
        else:
            from tools.structured_sequence_metrics import evaluate
            sources['tools/structured_sequence_metrics.py'] = digest('tools/structured_sequence_metrics.py')
            model = new_model(args).train()
            optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
            rng = np.random.default_rng(args.seed)
            draws = np.zeros(5,dtype=np.int64)
            report.update(parameters=model.parameter_count(),training=[])
            for step in range(args.updates):
                rows = sample_rows(train,size,step/max(args.updates-1,1),rng)
                if len(np.unique(train['seeds'][rows])) != size: raise RuntimeError('repeated batch level')
                batch = sequence_batch(train,rows,args.device)
                optimizer.zero_grad(set_to_none=True)
                with autocast(args.device):
                    result = sequence_objective(model,*batch,scale,pos_weight=positive,
                                                checkpoint_steps=args.checkpoint_steps)
                if not torch.isfinite(result['total']): raise ValueError('nonfinite H4 loss')
                result['total'].backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
                optimizer.step()
                draws += np.bincount(train['difficulties'][rows],minlength=6)[1:6]
                if step == 0 or (step+1)%20 == 0 or step+1 == args.updates:
                    entry=dict(step=step+1,loss=float(result['total'].detach()),gradient_norm=float(norm),
                        elapsed_seconds=time.monotonic()-start,
                        losses={k:float(v.detach()) for k,v in result['losses'].items()})
                    report['training'].append(entry)
                    report.update(completed_updates=step+1,difficulty_draws=draws.tolist())
                    atomic_json(args.report,report); print(json.dumps(entry),flush=True)
                del result,batch
            if any(digest(p)!=sha for p,sha in sources.items()): raise ValueError('inputs changed during training')
            target=Path(args.checkpoint); target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+f'.{os.getpid()}.tmp')
            torch.save(dict(format='pebby.structured-transition.v1',config=model.config(),
                weights={k:v.detach().cpu() for k,v in model.state_dict().items()},
                sources=sources,cache_manifests={'train':tm,'validation':vm},
                feature_scale=scale.cpu(),updates=args.updates,batch_size=size,seed=args.seed,
                objective='autoregressive_H4',initialize=args.initialize,policy_integrated=False),temporary)
            os.replace(temporary,target)
            report.update(status='evaluation_running',checkpoint=str(target),checkpoint_sha256=digest(target))
            atomic_json(args.report,report)
            report['validation']=evaluate(model,val,scale,args.device,args.eval_batch)
            report['train_evaluation']=evaluate(model,train,scale,args.device,args.eval_batch)
            report['status']='complete'
        if any(digest(p)!=sha for p,sha in sources.items()): raise ValueError('experiment inputs changed')
        report.update(source_unchanged=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None)
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds']=time.monotonic()-start
        atomic_json(args.report,report)


if __name__ == '__main__': main()
