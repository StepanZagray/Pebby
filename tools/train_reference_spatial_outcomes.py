"""Fresh spatial outcome training on public frozen features of generated levels."""
import argparse
from collections import Counter
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F

from pebby.agent.curriculum_sampling import CurriculumSampler
from pebby.agent.spatial_outcome_planner import SpatialOutcomePlanner
from pebby.agent.spatial_outcome_objective import spatial_outcome_losses, training_weights
from pebby.agent.spatial_outcome_policy import FORMAT
from pebby.agent.neural_outcome_policy import PARENT_SHA, ENCODER_RUNTIME, weights_sha256
from tools.train_reference_outcomes import load_data, batch, optimizer_for, guard, gpu_available, sha, write, tiny_indices, tiny_gate, scalar_metrics
from tools.train_reference_repair import PARENT, validate_parent
from tools.run_reference_base_pipeline import START, END

ROOT = Path(__file__).resolve().parents[1]


def player_probabilities(items, weights):
    with torch.no_grad(), torch.autocast(items['state'].device.type, enabled=False):
        return F.linear(items['state'][:, :144].float(), weights['player_head.weight'],
                        weights['player_head.bias']).squeeze(-1).softmax(-1)


def forward(model, items, player_weights, objective_weights, precision, *, distance_ordering=0.):
    player = player_probabilities(items, player_weights)
    with torch.autocast(items['raw'].device.type, dtype=torch.bfloat16, enabled=precision == 'bf16'):
        prediction = model(items['raw'], items['state'], items['glyph'], player)
        return spatial_outcome_losses(model, prediction, items, objective_weights, distance_ordering=distance_ordering)


def evaluate(model, arrays, player_weights, objective_weights, size, *, distance_ordering=0.):
    totals, supports = Counter(), Counter()
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(arrays['seeds']), size):
            guard()
            record = forward(model, batch(arrays, np.arange(start, min(start+size, len(arrays['seeds']))), 'cuda'),
                             player_weights, objective_weights, 'float32', distance_ordering=distance_ordering)
            for name, value in record['diagnostics'].items():
                if value is None:
                    continue
                weight = float(record['diagnostic_weights'][name])
                totals[name] += float(value) if name.endswith('_support') or name.endswith('_count') else float(value) * weight
                supports[name] += weight
    return dict(metrics={name: totals[name] if name.endswith('_support') or name.endswith('_count') else totals[name]/supports[name]
                         for name in totals if supports[name] > 0}, supports=dict(supports))


def tiny_rows(arrays):
    """64 distinct TRAIN levels: retain action balance while covering rare targets."""
    rows = tiny_indices(arrays).tolist()
    candidates = [np.asarray(arrays[n]).any(1) for n in ('lost_life', 'terminal', 'won')]
    candidates += [(arrays['next_triple'][..., j] != arrays['current_triple'][:, None, j]).any(1) for j in range(3)]
    # Replace tail rows only when a required cohort is absent. The original
    # 16-per-action balance is a starting point, not a claim after replacements.
    replaced = 0
    for mask in candidates:
        if np.asarray(mask)[rows].any():
            continue
        used = {int(arrays['seeds'][r]) for r in rows}
        replacement = next((int(r) for r in np.flatnonzero(mask) if int(arrays['seeds'][r]) not in used), None)
        if replacement is None:
            raise ValueError('cannot cover rare TRAIN outcomes with distinct levels')
        rows[-1-replaced] = replacement
        replaced += 1
    if not all(np.asarray(mask)[rows].any() for mask in candidates):
        raise ValueError('tiny qualification lacks required positive cohorts')
    return np.asarray(rows)


def fit_step(model, optimizer, items, player_weights, objective_weights, precision, *, distance_ordering=0.):
    optimizer.zero_grad(set_to_none=True)
    record = forward(model, items, player_weights, objective_weights, precision, distance_ordering=distance_ordering)
    if not bool(torch.isfinite(record['total'])):
        raise ValueError('nonfinite training loss')
    record['total'].backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise ValueError('all trainable tensors must have finite gradients')
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return record, norm


def qualify(args, arrays, sampler, player_weights, objective_weights, report):
    report['batch_qualification'] = []
    selected = None
    for size in (1024, 512, 256, 128, 64):
        guard(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model = optimizer = items = record = None
        entry = {'batch_size': size}
        try:
            torch.manual_seed(42); model = SpatialOutcomePlanner().cuda().train()
            optimizer = optimizer_for(model, args.lr)
            rows = sampler.indices(size, 0., torch.Generator().manual_seed(42)).numpy()
            if len(np.unique(arrays['seeds'][rows])) != size:
                raise ValueError('qualification batch contains repeated levels')
            items = batch(arrays, rows, 'cuda')
            before = weights_sha256(model.state_dict())
            torch.cuda.synchronize(); started = time.monotonic()
            record, norm = fit_step(model, optimizer, items, player_weights, objective_weights, args.precision, distance_ordering=args.distance_ordering)
            torch.cuda.synchronize()
            if weights_sha256(model.state_dict()) == before or any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise ValueError('optimizer did not change finite weights')
            entry.update(status='passed', distinct_levels=size, elapsed_seconds=time.monotonic()-started,
                         peak_gpu_bytes=torch.cuda.max_memory_allocated(), parameters=model.parameter_count(),
                         unclipped_gradient_norm=float(norm))
            selected = size
        except torch.cuda.OutOfMemoryError:
            entry['status'] = 'out_of_memory'
        finally:
            report['batch_qualification'].append(entry)
            del model, optimizer, items, record
            torch.cuda.empty_cache()
        if selected is not None:
            break
    if selected is None:
        raise RuntimeError('no qualified power-of-two batch')
    report['selected_batch_size'] = selected
    torch.manual_seed(42); model = SpatialOutcomePlanner().cuda().train()
    report['initial_planner_weights_sha256'] = weights_sha256(model.state_dict())
    optimizer = optimizer_for(model, args.lr)
    rows = tiny_rows(arrays); items = batch(arrays, rows, 'cuda')
    report['tiny_fit'] = dict(rows=rows.tolist(), seeds=arrays['seeds'][rows].tolist(), distinct_levels=64,
                             positive_cohorts_required=['lost_life','terminal','won','shape_change','color_change','rotation_change'])
    started, consecutive = time.monotonic(), 0
    for step in range(args.tiny_steps):
        guard()
        fit_step(model, optimizer, items, player_weights, objective_weights, args.precision, distance_ordering=args.distance_ordering)
        with torch.no_grad():
            record = forward(model, items, player_weights, objective_weights, args.precision, distance_ordering=args.distance_ordering)
        metrics = scalar_metrics(record)
        changed_ok = all(metrics.get(name+'_changed_accuracy', 0.) >= .90 for name in ('player','shape','color','rotation'))
        consecutive = consecutive+1 if tiny_gate(record) and changed_ok else 0
        if (step+1) % 100 == 0:
            print(json.dumps(dict(event='tiny_fit', step=step+1, metrics=metrics)), flush=True)
        if consecutive >= 10 or time.monotonic()-started > 180:
            break
    model.eval()
    with torch.no_grad():
        final = forward(model, items, player_weights, objective_weights, 'float32', distance_ordering=args.distance_ordering)
    final_metrics = scalar_metrics(final)
    passed = tiny_gate(final) and all(final_metrics.get(n+'_changed_accuracy', 0.) >= .90 for n in ('player','shape','color','rotation'))
    report['tiny_fit'].update(steps=step+1, elapsed_seconds=time.monotonic()-started, final_fp32=final_metrics,
                             final_losses={k:float(v) for k,v in final['losses'].items()}, consecutive_gate_steps=consecutive)
    report['qualification_passed'] = bool(passed and consecutive >= 10)
    if not report['qualification_passed']:
        raise RuntimeError('finite tiny-fit gate failed; full training not admitted')


def train(args, arrays, sampler, player_weights, objective_weights, parent, report):
    qualification = json.loads(args.qualification.read_text())
    if (qualification.get('status') != 'complete' or not qualification.get('qualification_passed')
            or qualification['selected_batch_size'] != args.batch_size or qualification['precision'] != args.precision
            or qualification['learning_rate'] != args.lr or qualification['cache_manifest_sha256'] != report['cache_manifest_sha256']
            or qualification['objective_weights'] != objective_weights
            or qualification.get('distance_ordering') != args.distance_ordering):
        raise ValueError('training differs from completed qualification')
    if any(sha(p) != h for p,h in qualification['source_sha256'].items()):
        raise ValueError('qualification sources changed')
    torch.manual_seed(42); model = SpatialOutcomePlanner().cuda().train()
    initial = weights_sha256(model.state_dict())
    if initial != qualification['initial_planner_weights_sha256']:
        raise ValueError('fresh initialization differs from qualification')
    optimizer = optimizer_for(model, args.lr)
    if optimizer.state:
        raise ValueError('fresh optimizer moments required')
    generator = torch.Generator().manual_seed(42)
    started = time.monotonic(); report['updates'] = []; torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        available = guard()
        rows = sampler.indices(args.batch_size, step/max(1,args.steps-1), generator).numpy()
        if len(np.unique(arrays['train']['seeds'][rows])) != args.batch_size:
            raise ValueError('batch has repeated levels')
        warmup = max(1,round(.03*args.steps))
        fraction = min(1.,(step+1)/warmup) if step < warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,args.steps-warmup)))
        for group in optimizer.param_groups:
            group['lr'] = args.lr*fraction
        model.train()
        record,norm = fit_step(model, optimizer, batch(arrays['train'],rows,'cuda'), player_weights,objective_weights,args.precision, distance_ordering=args.distance_ordering)
        if (step+1)%25 == 0 or step+1 == args.steps:
            torch.cuda.synchronize(); elapsed = time.monotonic()-started
            progress = dict(step=step+1, loss=float(record['total']), metrics=scalar_metrics(record),
                            unclipped_gradient_norm=float(norm), elapsed_seconds=elapsed,
                            eta_seconds=(args.steps-step-1)*elapsed/(step+1), available_host_bytes=available,
                            peak_gpu_bytes=torch.cuda.max_memory_allocated(), difficulty_counts=sampler.last_difficulty_counts)
            report['updates'].append(progress); write(args.out_dir/'report.json',report)
            print(json.dumps(dict(event='progress', **progress)),flush=True)
    report['training_elapsed_seconds'] = time.monotonic()-started
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError('nonfinite final model parameters; checkpoint refused')
    report['validation'] = evaluate(model, arrays['validation'], player_weights, objective_weights, args.batch_size, distance_ordering=args.distance_ordering)
    checkpoint = dict(format=FORMAT, encoder_config=parent['config'], encoder_weights=parent['weights'],
        encoder_weights_sha256=weights_sha256(parent['weights']), encoder_parent_sha256=PARENT_SHA,
        encoder_runtime=ENCODER_RUNTIME, encoder_frozen=True, planner_config=model.config(),
        planner_weights={k:v.detach().cpu() for k,v in model.state_dict().items()}, planner_parameters=model.parameter_count(),
        score_weights={'planner':1.,'direct':0.}, planner_initialization={'kind':'random','seed':42,'weights_sha256':initial},
        optimizer_state='new; no resumed moments', optimizer='AdamW fused', optimizer_steps=args.steps,
        batch_size=args.batch_size, learning_rate=args.lr, weight_decay=.05, objective_weights=objective_weights, distance_ordering=args.distance_ordering,
        train_levels=10000, official_training_inputs=False, actual_outcome_comparator_auxiliary_training=True,
        privileged_inference_inputs=False, planner_horizon=1, planner_refinement_loops=1,
        learned_voluntary_reset=False, persistent_game_memory=False, training_cache=str(args.cache),
        cache_manifest_sha256=report['cache_manifest_sha256'], source_sha256=report['source_sha256'],
        qualification_sha256=report['qualification_sha256'])
    torch.save(checkpoint,args.out_dir/'model.pending.pt')
    report['checkpoint_sha256'] = sha(args.out_dir/'model.pending.pt')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=ROOT/'data/reference-outcome-inputs-v1')
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--qualify',action='store_true')
    parser.add_argument('--qualification',type=Path)
    parser.add_argument('--steps',type=int,default=1560)
    parser.add_argument('--tiny-steps',type=int,default=2000)
    parser.add_argument('--batch-size',type=int,default=1024)
    parser.add_argument('--lr',type=float,default=.0003)
    parser.add_argument('--distance-ordering', type=float, default=0., help='nonnegative safe sibling-distance ranking weight; zero preserves the original objective')
    parser.add_argument('--precision',choices=('bf16','float32'),default='bf16')
    parser.add_argument('--max-seconds',type=int,default=1800)
    args = parser.parse_args(argv)
    if not math.isfinite(args.distance_ordering) or args.distance_ordering < 0:
        parser.error('distance-ordering must be finite and nonnegative')
    if min(args.steps,args.tiny_steps,args.max_seconds)<=0 or args.batch_size not in (64,128,256,512,1024) or not math.isfinite(args.lr) or args.lr<=0:
        parser.error('invalid finite budget or power-of-two batch')
    if not args.qualify and args.qualification is None:
        parser.error('completed --qualification required for training')
    gpu_available(); guard(); args.out_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=True
    report = dict(status='running',pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
                  started_local=datetime.now().astimezone().isoformat(),mode='qualification' if args.qualify else 'training',
                  official_inputs_used=False, encoder_frozen=True, learning_rate=args.lr, precision=args.precision, distance_ordering=args.distance_ordering,
                  batch_size=args.batch_size, planned_steps=args.steps, cache_manifest_sha256=sha(args.cache/'manifest.json'))
    sources = [Path(__file__).resolve(), PARENT, args.cache/'manifest.json', ROOT/'tools/train_reference_outcomes.py', ROOT/'tools/train_reference_repair.py',
               ROOT/'tools/run_reference_base_pipeline.py', ROOT/'tools/cache_reference_outcome_inputs.py',
               ROOT/'data/ls20-reference-unequal-v1/train.jsonl']
    sources += [ROOT/'pebby/agent'/name for name in ('spatial_outcome_planner.py','spatial_outcome_policy.py','spatial_outcome_objective.py',
                'outcome_ordering.py','planning_decision_metrics.py','neural_outcome_planner.py','neural_outcome_policy.py','world_model.py','world_train.py','world_grounding.py','curriculum_sampling.py')]
    if args.qualification is not None:
        sources.append(args.qualification.resolve())
        report['qualification_sha256'] = sha(args.qualification)
    report['source_sha256'] = {str(p):sha(p) for p in sources}
    write(args.out_dir/'report.json',report); print(json.dumps(dict(event='started',pid=report['pid'],start_ticks=report['start_ticks'])),flush=True)
    def timeout(*_):
        raise TimeoutError('bounded spatial trainer deadline reached')
    signal.signal(signal.SIGALRM,timeout); signal.alarm(args.max_seconds)
    try:
        if sha(PARENT) != PARENT_SHA:
            raise ValueError('protected parent changed')
        parent = torch.load(PARENT,map_location='cpu',weights_only=True); validate_parent(parent)
        arrays,hashes,stats = load_data(args.cache)
        report['verified_cache_hashes'] = hashes
        weights = training_weights(arrays['train']); report['objective_weights'] = weights
        public_player = {k:v.cuda() for k,v in parent['weights'].items() if k.startswith('player_head.')}
        levels = [json.loads(line) for line in (ROOT/'data/ls20-reference-unequal-v1/train.jsonl').read_text().splitlines()]
        sampler = CurriculumSampler({'seeds':np.array(arrays['train']['seeds']),'meta':{'levels':levels}},start=START,end=END)
        sampler.check_coverage(args.batch_size)
        if args.qualify:
            qualify(args,arrays['train'],sampler,public_player,weights,report)
        else:
            train(args,arrays,sampler,public_player,weights,parent,report)
        for p,expected in stats.items():
            s = Path(p).stat()
            if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns] != expected:
                raise ValueError('input cache changed')
        if any(sha(p) != h for p,h in report['source_sha256'].items()):
            raise ValueError('source or parent changed during execution')
        if not args.qualify:
            (args.out_dir/'model.pending.pt').replace(args.out_dir/'model.pt')
        report.update(status='complete',sources_unchanged=True,finished_local=datetime.now().astimezone().isoformat())
    except BaseException as error:
        (args.out_dir/'model.pending.pt').unlink(missing_ok=True)
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0); write(args.out_dir/'report.json',report)


if __name__ == '__main__':
    main()
