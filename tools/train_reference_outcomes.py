"""Train a fresh one-step neural outcome planner on frozen fresh-base features.

No teacher enters the forward pass. Actual successor properties supervise four
separate loss families, without SIGReg. Qualification is a disposable B1024
backward/memory check plus a finite tiny-fit control, not a gameplay result.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import torch

from pebby.agent.curriculum_sampling import CurriculumSampler
from pebby.agent.neural_outcome_planner import NeuralOutcomePlanner, neural_outcome_losses
from pebby.agent.neural_outcome_policy import FORMAT, weights_sha256
from pebby.agent.world_train import initial_state_sha256, parameter_groups
from tools.train_reference_repair import PARENT, PARENT_SHA, validate_parent
from tools.run_reference_base_pipeline import START, END

ROOT = Path(__file__).resolve().parents[1]
ARRAYS = ('raw', 'state', 'glyph', 'seeds', 'rows', 'optimal', 'next_optimal',
          'next_player_cell', 'next_triple', 'next_steps', 'next_lives', 'distances',
          'lost_life', 'terminal', 'won', 'player_cell', 'current_triple',
          'current_steps', 'current_lives')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def guard():
    available = next(int(x.split()[1]) * 1024 for x in Path('/proc/meminfo').read_text().splitlines()
                     if x.startswith('MemAvailable:'))
    if available < 6 * 2**30:
        raise MemoryError('required 6 GiB MemAvailable reserve exhausted')
    return available


def gpu_available():
    found = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, check=True, timeout=10)
    foreign = [int(x.strip()) for x in found.stdout.splitlines() if x.strip() and int(x) != os.getpid()]
    if foreign:
        raise RuntimeError(f'other GPU compute processes are active: {foreign}')


def load_data(path):
    from tools.cache_reference_outcome_inputs import validate_published
    manifest = validate_published(path)
    expected_runtime = dict(precision='float32', execution='native_eager', temporal_backend='auto',
                            matmul_tf32=False, cudnn_tf32=True, encoder_chunk_size=0)
    if any(manifest.get('settings', {}).get(k) != v for k, v in expected_runtime.items()):
        raise ValueError('cache encoder runtime differs from the policy inference contract')
    result = {split: {name: np.load(path / split / (name + '.npy'), mmap_mode='r', allow_pickle=False)
                      for name in ARRAYS} for split in ('train', 'validation')}
    return result, manifest['validated_output_hashes'], manifest['validated_output_stats']


def batch(arrays, indices, device):
    return {key: torch.from_numpy(np.array(value[indices], copy=True)).to(device)
            for key, value in arrays.items() if key not in ('rows', 'seeds')}


def forward(model, items, precision):
    with torch.autocast(items['raw'].device.type, dtype=torch.bfloat16, enabled=precision == 'bf16'):
        predictions = model(items['raw'], items['state'], items['glyph'])
        return neural_outcome_losses(predictions, items)


def scalar_metrics(record):
    return {key: None if value is None else float(value.detach().float())
            for key, value in record['diagnostics'].items()}


def evaluate(model, arrays, device, batch_size):
    totals, supports = Counter(), Counter()
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(arrays['seeds']), batch_size):
            indices = np.arange(start, min(start + batch_size, len(arrays['seeds'])))
            record = forward(model, batch(arrays, indices, device), 'float32')
            for name, value in record['diagnostics'].items():
                if value is None:
                    continue
                weight = float(record.get('diagnostic_weights', {}).get(name, len(indices)))
                if weight > 0:
                    # Counts are totals; accuracies/fractions are weighted means.
                    totals[name] += float(value) if name.endswith('_support') or name.endswith('_count') else float(value) * weight
                    supports[name] += weight
    model.train(was_training)
    return {name: totals[name] if name.endswith('_support') or name.endswith('_count') else totals[name] / supports[name]
            for name in totals}, dict(supports)


def optimizer_for(model, learning_rate):
    groups = parameter_groups(model, .05)
    actual = [id(p) for group in groups for p in group['params']]
    expected = [id(p) for p in model.parameters() if p.requires_grad]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError('optimizer must cover every trainable tensor exactly once')
    return torch.optim.AdamW(groups, lr=learning_rate, fused=next(model.parameters()).is_cuda)


TINY_CRITERIA = dict(policy_accuracy=1., minimum_each_physical_accuracy=.90,
                     minimum_value_accuracy=.85, maximum_event_bce=.20,
                     minimum_supported_event_recall=.90, sustained_steps=10)


def tiny_gate(record):
    metrics = scalar_metrics(record)
    return (metrics.get('set_accuracy') == TINY_CRITERIA['policy_accuracy']
            and all(metrics.get(name + '_accuracy', 0.) >= TINY_CRITERIA['minimum_each_physical_accuracy']
                    for name in ('player', 'shape', 'color', 'rotation', 'steps', 'lives'))
            and metrics.get('value_accuracy', 0.) >= TINY_CRITERIA['minimum_value_accuracy']
            and float(record['losses']['events'].detach()) <= TINY_CRITERIA['maximum_event_bce']
            and all(metrics.get(name + '_support', 0.) == 0.
                    or (metrics.get(name + '_recall') is not None
                        and metrics[name + '_recall'] >= TINY_CRITERIA['minimum_supported_event_recall'])
                    for name in ('lost_life', 'terminal', 'won'))
            and bool(torch.isfinite(record['total'])))


def tiny_indices(arrays):
    # Distinct levels and single optimal targets balance this trainability test.
    masks, seeds = np.asarray(arrays['optimal']), np.asarray(arrays['seeds'])
    rng, chosen, used = np.random.default_rng(42), [], set()
    for action in range(4):
        candidates = np.flatnonzero(masks == 1 << action)
        rng.shuffle(candidates)
        picked = 0
        for index in candidates:
            seed = int(seeds[index])
            if seed in used:
                continue
            used.add(seed); chosen.append(int(index)); picked += 1
            if picked == 16:
                break
        if picked != 16:
            raise ValueError('cannot construct balanced, distinct-level tiny fit')
    return np.asarray(chosen)


def qualify(args, arrays, sampler, report):
    # Each failed allocation is a disposable model, never a production update.
    attempts = []
    selected = None
    for size in (1024, 512, 256, 128, 64):
        guard(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(42); model = opt = None
        rows = sampler.indices(size, 0., torch.Generator().manual_seed(42)).numpy()
        entry = {'batch_size': size, 'distinct_levels': len(np.unique(arrays['seeds'][rows]))}
        try:
            model = NeuralOutcomePlanner().cuda().train()
            opt = optimizer_for(model, args.lr)
            items = batch(arrays, rows, 'cuda')
            torch.cuda.synchronize(); started = time.monotonic()
            record = forward(model, items, args.precision)
            if not torch.isfinite(record['total']):
                raise ValueError('nonfinite qualification loss')
            record['total'].backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise ValueError('qualification requires finite gradients on all planner parameters')
            before_step = initial_state_sha256(model)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step(); torch.cuda.synchronize()
            if before_step == initial_state_sha256(model) or any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise ValueError('qualification optimizer did not make a finite parameter update')
            entry.update(status='passed', all_parameters_have_finite_gradients=True, optimizer_changed_weights=True, loss=float(record['total'].detach()), unclipped_gradient_norm=float(norm),
                         elapsed_seconds=time.monotonic()-started,
                         peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                         parameters=model.parameter_count())
            selected = size
        except torch.cuda.OutOfMemoryError as error:
            entry.update(status='out_of_memory', error=str(error))
        finally:
            attempts.append(entry)
            report['batch_qualification'] = attempts
            for name in ('record', 'items'):
                if name == 'record' and 'record' in locals(): del record
                if name == 'items' and 'items' in locals(): del items
            del opt, model
            torch.cuda.empty_cache()
        if selected:
            break
    report['batch_qualification'] = attempts
    if selected is None:
        raise RuntimeError('no tested power-of-two batch fits')
    report['selected_batch_size'] = selected
    # New initialization and optimizer: none of the fit uses the B1024 update.
    torch.manual_seed(42); model = NeuralOutcomePlanner().cuda().train()
    report['initial_planner_weights_sha256'] = initial_state_sha256(model)
    opt = optimizer_for(model, args.lr)
    indices = tiny_indices(arrays)
    items = batch(arrays, indices, 'cuda')
    first = forward(model, items, args.precision)
    report['tiny_fit'] = {'indices': indices.tolist(), 'seeds': arrays['seeds'][indices].tolist(),
                          'initial': scalar_metrics(first), 'batch_size': 64, 'criteria': TINY_CRITERIA,
                          'initial_losses': {k: float(v.detach()) for k,v in first['losses'].items()}}
    del first
    started, consecutive = time.monotonic(), 0
    for step in range(args.tiny_steps):
        guard()
        opt.zero_grad(set_to_none=True)
        record = forward(model, items, args.precision)
        record['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        with torch.no_grad():
            check = forward(model, items, args.precision)
        metrics = scalar_metrics(check)
        consecutive = consecutive + 1 if tiny_gate(check) else 0
        if (step + 1) % 25 == 0:
            print(json.dumps({'event': 'tiny_fit', 'steps': step + 1, 'metrics': metrics}), flush=True)
        if consecutive >= 10 or time.monotonic() - started > 120:
            break
    model.eval()
    with torch.no_grad():
        final = forward(model, items, 'float32')
    report['tiny_fit'].update(steps=step+1, elapsed_seconds=time.monotonic()-started,
        final_fp32=scalar_metrics(final), final_losses={k: float(v.detach()) for k,v in final['losses'].items()},
        sustained_outcome_gate_steps=consecutive, stop_reason='sustained_gate' if consecutive >= 10 else 'finite_budget_exhausted',
        interpretation='Finite trainability check only; not held-level generalization or gameplay.')
    if not tiny_gate(final) or consecutive < 10:
        raise RuntimeError('tiny fit failed declared physical/value/event/policy trainability gate; no training admitted')
    report['qualification_passed'] = True


def train(args, arrays, sampler, report, output):
    qualification = json.loads(args.qualification.read_text())
    if qualification.get('status') != 'complete' or not qualification.get('qualification_passed'):
        raise ValueError('completed finite qualification required')
    if qualification.get('precision') != args.precision or qualification.get('learning_rate') != args.lr:
        raise ValueError('training precision or learning rate differs from qualification')
    if qualification['selected_batch_size'] != args.batch_size:
        raise ValueError('batch size differs from largest qualified power of two')
    for path, digest in qualification['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError(f'qualification source changed: {path}')
    if qualification['cache_manifest_sha256'] != report['cache_manifest_sha256']:
        raise ValueError('qualification used a different cache')
    torch.manual_seed(42); model = NeuralOutcomePlanner().cuda().train()
    if initial_state_sha256(model) != qualification['initial_planner_weights_sha256']:
        raise ValueError('fresh planner initialization differs')
    parent = torch.load(PARENT, map_location='cpu', weights_only=True)
    validate_parent(parent)
    opt = optimizer_for(model, args.lr)
    generator = torch.Generator().manual_seed(42)
    started = time.monotonic(); report['updates'] = []
    for step in range(args.steps):
        guard()
        indices = sampler.indices(args.batch_size, step / max(1,args.steps-1), generator).numpy()
        if len(np.unique(arrays['train']['seeds'][indices])) != args.batch_size:
            raise ValueError('training batch does not contain distinct levels')
        items = batch(arrays['train'], indices, 'cuda')
        warmup = max(1, round(.03 * args.steps))
        fraction = min(1., (step+1)/warmup) if step < warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,args.steps-warmup)))
        for group in opt.param_groups: group['lr'] = args.lr * fraction
        opt.zero_grad(set_to_none=True)
        record = forward(model, items, args.precision)
        record['total'].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        if (step+1) % 25 == 0 or step+1 == args.steps:
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            progress = dict(step=step+1, loss=float(record['total']), metrics=scalar_metrics(record),
                unclipped_gradient_norm=float(norm), lr=opt.param_groups[0]['lr'],
                elapsed_seconds=elapsed, eta_seconds=(args.steps-step-1)*elapsed/(step+1),
                difficulty_counts=sampler.last_difficulty_counts,
                distinct_levels=sampler.last_distinct_levels,
                peak_gpu_bytes=torch.cuda.max_memory_allocated())
            report['updates'].append(progress); write(output/'report.json',report)
            print(json.dumps({'event':'progress', **progress}), flush=True)
    metrics, supports = evaluate(model, arrays['validation'], 'cuda', args.batch_size)
    report['validation'] = {'metrics': metrics, 'supports': supports, 'states':4000,'levels':500,
                            'scope':'Previously inspected generated validation; not untouched test or gameplay.'}
    checkpoint = dict(format=FORMAT, encoder_config=parent['config'], encoder_weights=parent['weights'],
        encoder_weights_sha256=weights_sha256(parent['weights']),
        planner_config=model.config(), planner_weights={k:v.detach().cpu() for k,v in model.state_dict().items()},
        score_weights={'planner':1.,'direct':0.}, encoder_parent_sha256=PARENT_SHA,
        planner_initialization={'kind':'random','seed':42,'weights_sha256':qualification['initial_planner_weights_sha256']},
        optimizer_state='new; no resumed moments', optimizer='AdamW', optimizer_steps=args.steps,
        learning_rate=args.lr, weight_decay=.05, batch_size=args.batch_size,
        train_levels=10000, training_cache=str(args.cache), cache_manifest_sha256=report['cache_manifest_sha256'],
        official_training_inputs=False, planner_horizon=1, planner_refinement_loops=1,
        persistent_game_memory=False, learned_voluntary_reset=False,
        encoder_frozen=True, planner_parameters=model.parameter_count(),
        encoder_runtime=dict(precision='float32', temporal_backend='auto', matmul_tf32=False, cudnn_tf32=True),
        source_sha256=report['source_sha256'])
    # Publish only after cache and source guards pass in main.
    torch.save(checkpoint,output/'model.pending.pt')
    report['checkpoint_sha256']=sha(output/'model.pending.pt')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=ROOT/'data/reference-outcome-inputs-v1')
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--qualify',action='store_true')
    parser.add_argument('--qualification',type=Path)
    parser.add_argument('--steps',type=int,default=780)
    parser.add_argument('--tiny-steps',type=int,default=300)
    parser.add_argument('--batch-size',type=int,default=1024)
    parser.add_argument('--lr',type=float,default=.0003)
    parser.add_argument('--precision',choices=['bf16','float32'],default='bf16')
    parser.add_argument('--max-seconds',type=int,default=1800)
    args=parser.parse_args(argv)
    if min(args.steps,args.tiny_steps,args.max_seconds)>0 and args.batch_size in (64,128,256,512,1024) and math.isfinite(args.lr) and args.lr>0:
        pass
    else: parser.error('invalid positive training parameters or power-of-two batch')
    if not args.qualify and args.qualification is None: parser.error('--qualification required for training')
    if args.out_dir.exists(): raise FileExistsError(args.out_dir)
    gpu_available(); guard(); torch.set_num_threads(4)
    if sha(PARENT)!=PARENT_SHA: raise ValueError('protected parent hash changed')
    args.out_dir.mkdir(parents=True)
    report=dict(status='running',pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
        started_local=datetime.now().astimezone().isoformat(),mode='qualification' if args.qualify else 'training',
        parent_sha256=PARENT_SHA, cache_manifest_sha256=sha(args.cache/'manifest.json'),
        precision=args.precision, learning_rate=args.lr, batch_size=args.batch_size, actual_distinct_level_batch=True,
        source_sha256={str(path):sha(path) for path in [Path(__file__).resolve(),
            ROOT/'pebby/agent/neural_outcome_planner.py',ROOT/'pebby/agent/neural_outcome_policy.py',
            ROOT/'pebby/agent/curriculum_sampling.py',ROOT/'pebby/agent/world_train.py',
            ROOT/'pebby/agent/world_model.py', ROOT/'pebby/agent/world_grounding.py',
            ROOT/'tools/cache_reference_outcome_inputs.py', ROOT/'tools/train_reference_repair.py',
            ROOT/'tools/run_reference_base_pipeline.py', ROOT/'data/ls20-reference-unequal-v1/train.jsonl']},
        official_inputs_used=False, encoder_frozen=True, planner_horizon=1,
        planner_refinement_loops=1, optimizer='AdamW', optimizer_state='new')
    write(args.out_dir/'report.json',report)
    print(json.dumps({'event':'started','pid':report['pid'],'start_ticks':report['start_ticks']}),flush=True)
    def timeout(*_): raise TimeoutError('bounded trainer time limit reached')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(args.max_seconds)
    try:
        arrays, hashes, stats=load_data(args.cache)
        report['verified_cache_hashes']=hashes
        levels=[json.loads(line) for line in (ROOT/'data/ls20-reference-unequal-v1/train.jsonl').read_text().splitlines()]
        sampler=CurriculumSampler({'seeds':np.array(arrays['train']['seeds']), 'meta':{'levels':levels}},start=START,end=END)
        sampler.check_coverage(args.batch_size)
        if args.qualify: qualify(args,arrays['train'],sampler,report)
        else: train(args,arrays,sampler,report,args.out_dir)
        for file,expected in stats.items():
            s=Path(file).stat()
            if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=expected:raise ValueError('cache changed during execution')
        if sha(PARENT)!=PARENT_SHA or any(sha(p)!=h for p,h in report['source_sha256'].items()):
            raise ValueError('source or parent changed during execution')
        if not args.qualify:
            (args.out_dir/'model.pending.pt').replace(args.out_dir/'model.pt')
        report.update(status='complete',finished_local=datetime.now().astimezone().isoformat(),sources_unchanged=True)
    except BaseException as error:
        (args.out_dir/'model.pending.pt').unlink(missing_ok=True)
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0);write(args.out_dir/'report.json',report)


if __name__=='__main__':main()
