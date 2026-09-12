"""Disposable fresh seven-tier world optimizer capacity probe; never saves weights.

The report selects a batch size, not a trained model. The subsequent fit must
construct a fresh model with the recorded seed/config/weights and memory options.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from pebby.agent.curriculum_sampling import CurriculumSampler
from pebby.agent.world_model import WorldModelConfig, WorldPolicy, DEFAULT_WEIGHTS, parameter_groups
from pebby.agent.world_training_objectives import world_losses
from pebby.agent.world_runtime import configure_execution
from pebby.agent.world_train import (load_dataset, require_verified_data, require_winning_coverage,
                                    as_tensors, initial_state_sha256)


def fresh_config():
    return WorldModelConfig(max_distance=128, grounding=True, state_recall=True,
                            glyph_recall=True, query_readout=True)


def check_distance_support(data, maximum=128):
    """Exact finite support is 0..maximum-1; -1 is unreachable, never overflow."""
    if type(maximum) is not int or maximum < 2:
        raise ValueError('max_distance must be an integer >=2')
    values = np.asarray(data['distances'])
    if values.ndim != 2 or values.shape[1] != 4 or values.dtype.kind not in 'iu':
        raise ValueError('distances must be integer[N,4]')
    if not values.size or np.any(values < -1):
        raise ValueError('empty or invalid distance labels')
    high = int(values.max())
    if high >= maximum:
        raise ValueError(f'distance overflow: {high} exceeds exact support {maximum-1}; rebuild head config')
    return dict(maximum_finite_distance=high, unreachable_branches=int((values < 0).sum()),
                branches=int(values.size), exact_maximum=maximum-1, overflow_branches=0)


def candidates(maximum):
    if type(maximum) is not int or maximum < 1 or maximum > 1024 or maximum & (maximum-1):
        raise ValueError('max_batch must be a power of two <=1024')
    return [maximum >> i for i in range(maximum.bit_length())]


def choose_capacity(maximum, attempt, record):
    """Only CUDA OOM permits fallback; programming/data/numerical errors propagate."""
    for size in candidates(maximum):
        try:
            result = attempt(size)
        except torch.cuda.OutOfMemoryError:
            result = dict(batch_size=size, status='cuda_oom')
            record(result)
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue
        record(result)
        return size
    raise RuntimeError('no batch size fits')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_report(path, report):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', type=Path, required=True)
    p.add_argument('--validation', type=Path, required=True)
    p.add_argument('--data-cache-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--curriculum-start', type=float, nargs=7, default=[.40,.25,.15,.09,.05,.04,.02])
    p.add_argument('--curriculum-end', type=float, nargs=7, default=[.04,.08,.12,.18,.22,.22,.14])
    p.add_argument('--config', type=Path)
    p.add_argument('--weights', type=Path)
    p.add_argument('--max-batch', type=int, default=1024)
    p.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    p.add_argument('--smoke', action='store_true', help='Software-only tiny cache; not production capacity evidence')
    p.add_argument('--seconds', type=int, default=600)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--encoder-chunk-size', type=int, default=128)
    p.add_argument('--checkpoint-encoder', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--checkpoint-loops', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--compile-core', action='store_true')
    p.add_argument('--temporal-backend', choices=('auto', 'math', 'cudnn', 'flash'), default='auto')
    args = p.parse_args(argv)
    candidates(args.max_batch)
    if args.out.exists(): raise FileExistsError(args.out)
    if not 1 <= args.seconds <= 1800 or args.encoder_chunk_size < 1:
        p.error('positive chunk size and seconds1..1800 required')
    if not args.smoke and (args.device != 'cuda' or args.max_batch != 1024):
        p.error('production starts at CUDA B1024; use --smoke for software tests')
    cfg = WorldModelConfig(**json.loads(args.config.read_text())) if args.config else fresh_config()
    if cfg.cell_recall: raise ValueError('fresh base preflight cannot import a frozen appearance checkpoint')
    weights = {**DEFAULT_WEIGHTS, 'successor_policy': 1.}
    if args.weights:
        overrides = json.loads(args.weights.read_text())
        if set(overrides) - set(weights): raise ValueError('unknown loss weights')
        weights.update(overrides)
    if any(not isinstance(v, (int,float)) or not np.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError('weights must be finite nonnegative')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise ValueError('CUDA native BF16 required')
    started = time.monotonic()
    report = dict(status='running', pid=os.getpid(), config=cfg.__dict__, weights=weights,
                  seed=args.seed, attempts=[], smoke=args.smoke, checkpoint_saved=False,
                  fresh_initialization=True, precision='bf16' if args.device=='cuda' else 'float32',
                  checkpoint_encoder=args.checkpoint_encoder, checkpoint_loops=args.checkpoint_loops,
                  encoder_chunk_size=args.encoder_chunk_size,
                  optimizer=dict(name='AdamW',lr=.0003,weight_decay=.05,clip_norm=1.),
                  limits=['Capacity evidence only; all updated weights discarded.',
                          'One-step counterfactual dynamics; no chronological H4 supervision or learned search.',
                          'Actual-reset distance and imagined unsafe labels share a value head.'])
    print('PID',os.getpid(),flush=True)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('preflight deadline')))
    signal.alarm(args.seconds)
    write_report(args.out,report)
    try:
        paths = [args.train,args.validation,Path(__file__),Path('pebby/ls20/provenance.py')]
        paths += [Path('pebby/agent')/name for name in ('world_model.py','world_training_objectives.py',
                  'world_train.py','world_cache.py','world_grounding.py','world_rollout.py','curriculum_sampling.py',
                  'glyph_model.py','world_readout.py','world_runtime.py')]
        paths += [x for x in (args.config,args.weights) if x]
        sources = {str(x.resolve()):digest(x) for x in paths}
        report['sources']=sources
        train = load_dataset(args.train,cfg.history,args.data_cache_dir)
        validation = load_dataset(args.validation,cfg.history,args.data_cache_dir)
        for name,data in [('train',train),('validation',validation)]:
            require_verified_data(data);require_winning_coverage(data)
            if not data['meta'].get('levels') or any(level.get('coverage') != 'mixed_failure'
                    for level in data['meta']['levels'] if 'excluded' not in level):
                raise ValueError(f'{name}: mixed_failure collection required')
            report[name+'_distance_support']=check_distance_support(data,cfg.max_distance)
            if weights['successor_policy'] and data.get('next_optimal') is None:
                raise ValueError('successor policy requires exact next_optimal labels')
        if np.intersect1d(train['seeds'],validation['seeds']).size:
            raise ValueError('TRAIN/validation seed overlap')
        counts = [len(np.unique(x['seeds'])) for x in (train,validation)]
        if not args.smoke and (counts[0]<10000 or counts[1]<500):
            raise ValueError('production needs >=10000 TRAIN and >=500 validation levels')
        report['level_counts']=dict(train=counts[0],validation=counts[1])
        sampler=CurriculumSampler(train,start=args.curriculum_start,end=args.curriculum_end)
        val_sampler=CurriculumSampler(validation)
        if sampler.difficulty_version!='ls20-reference-v1' or val_sampler.difficulty_version!=sampler.difficulty_version:
            raise ValueError('separate seven-reference contract required on both splits')
        sampler.check_coverage(args.max_batch)
        tensors=as_tensors(train)
        report['curriculum']=dict(start=sampler.ratios(0).tolist(),end=sampler.ratios(1).tolist())
        def attempt(size):
            model=optimizer=batch=output=None
            try:
                torch.manual_seed(args.seed)
                model=WorldPolicy(cfg).to(args.device).train()
                model.checkpoint_encoder=args.checkpoint_encoder;model.checkpoint_loops=args.checkpoint_loops
                model.encoder_chunk_size=args.encoder_chunk_size
                report['execution']=configure_execution(model,compile_core=args.compile_core,
                                                        temporal_backend=args.temporal_backend)
                optimizer=torch.optim.AdamW(parameter_groups(model,.05),lr=.0003)
                if args.device=='cuda': torch.cuda.reset_peak_memory_stats()
                result=dict(batch_size=size,status='complete',parameters=model.parameter_count(),steps=[],
                            initial_weights_sha256=initial_state_sha256(model))
                for progress in (0.,.5,1.):
                    rows=sampler.indices(size,progress,torch.Generator().manual_seed(args.seed+int(progress*2)))
                    if len(set(sampler.last_level_seeds))!=size: raise ValueError('non-distinct batch')
                    batch={k:v[rows].to(args.device) for k,v in tensors.items()}
                    optimizer.zero_grad(set_to_none=True)
                    tick=time.monotonic()
                    with torch.autocast(args.device,dtype=torch.bfloat16,enabled=args.device=='cuda'):
                        output=world_losses(model,batch,weights)
                    if not bool(torch.isfinite(output['total'])): raise ValueError('nonfinite loss')
                    output['total'].backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step()
                    if args.device=='cuda': torch.cuda.synchronize()
                    result['steps'].append(dict(progress=progress,loss=float(output['total'].detach()),
                        gradient_norm=float(norm),seconds=time.monotonic()-tick,
                        seeds=list(map(int,sampler.last_level_seeds)),difficulty_counts=sampler.last_difficulty_counts.copy()))
                    batch=output=None
                if args.device=='cuda':
                    result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
                return result
            finally:
                model=optimizer=batch=output=None
                gc.collect()
                if args.device=='cuda': torch.cuda.empty_cache()
        def record(value):
            report['attempts'].append(value);write_report(args.out,report)
            print(json.dumps({'batch_size':value['batch_size'],'status':value['status']}),flush=True)
        report['selected_batch_size']=choose_capacity(args.max_batch,attempt,record)
        if any(digest(path)!=sha for path,sha in sources.items()): raise ValueError('source drift')
        report.update(status='complete',source_unchanged=True,requires_fresh_fit_initialization=True)
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds']=time.monotonic()-started
        write_report(args.out,report)

if __name__=='__main__': main()
