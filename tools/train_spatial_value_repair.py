"""Matched value-head-only replay on all verified original generated TRAIN roots.

Control uses 130-class CE; ordinal adds 10 times normalized finite-conditional
CDF squared error. Both use identical public summaries, initialization, level
draws and optimizer budgets. Unsupported policy roots remain value-supervised.
The encoder, summaries, physical/event heads and comparator remain frozen.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from pebby.agent.neural_outcome_policy import weights_sha256
from pebby.agent.spatial_outcome_planner import SpatialOutcomePlanner
from pebby.agent.spatial_value_objective import spatial_value_loss, value_targets
from tools.cache_reference_outcome_inputs import Bindings, release, stat
from tools.train_reference_outcomes import batch, gpu_available, guard, load_data, optimizer_for, sha, write
from tools.train_reference_spatial_outcomes import player_probabilities
from tools.train_spatial_event_repair import BATCH_SIZE, CHECKPOINT, CHECKPOINT_SHA, LevelSampler, ROOT, cache_summaries
from tools.train_spatial_recovery_comparison import schedule, validate_checkpoint

ARMS = {'control': 0., 'ordinal': 10.}


def freeze_except_value_head(model):
    names = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('value_head.'))
        parameter.grad = None
        if parameter.requires_grad:
            names.append(name)
    if set(names) != {'value_head.weight', 'value_head.bias'} or model.value_head.out_features != 130:
        raise ValueError('value repair requires exactly the existing 130-class linear head')
    return names


def frozen_digest(model):
    return weights_sha256({key: value for key, value in model.state_dict().items()
                           if not key.startswith('value_head.')})


def selected_items(arrays, summaries, rows, device):
    summary = torch.from_numpy(np.array(summaries[rows], copy=True)).to(device)
    distances = torch.from_numpy(np.array(arrays['distances'][rows], copy=True)).to(device)
    return summary, distances


def fit_step(model, optimizer, summaries, distances, ordinal_weight):
    optimizer.zero_grad(set_to_none=True)
    record = spatial_value_loss(model.value_head(summaries), distances, ordinal_weight=ordinal_weight)
    if not bool(torch.isfinite(record['total'])):
        raise ValueError('nonfinite value loss')
    record['total'].backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise ValueError(f'missing or nonfinite value gradient: {name}')
        elif parameter.grad is not None:
            raise ValueError(f'frozen parameter acquired gradient: {name}')
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
    optimizer.step()
    if any(not bool(torch.isfinite(parameter).all()) for parameter in trainable):
        raise ValueError('nonfinite value parameter')
    return record, float(norm)


def evaluate_value(model, arrays, summaries, device='cuda'):
    totals = dict(cross_entropy=0., cdf_mse=0.)
    counts = {key: [0, 0] for key in ('all', 'reachable', 'unreachable', 'zero_optimal')}
    with torch.inference_mode():
        for start in range(0, len(summaries), BATCH_SIZE):
            guard()
            rows = np.arange(start, min(start + BATCH_SIZE, len(summaries)))
            summary, distances = selected_items(arrays, summaries, rows, device)
            logits = model.value_head(summary)
            record = spatial_value_loss(logits, distances)
            reachable = distances >= 0
            totals['cross_entropy'] += float(record['cross_entropy']) * distances.numel()
            totals['cdf_mse'] += float(record['cdf_mse']) * int(reachable.sum())
            zero = torch.from_numpy(np.array(arrays['optimal'][rows] == 0)).to(device)[:, None].expand_as(distances)
            correct = logits.argmax(-1) == value_targets(logits, distances)
            for key, mask in [('all', torch.ones_like(reachable)), ('reachable', reachable),
                              ('unreachable', ~reachable), ('zero_optimal', zero)]:
                counts[key][0] += int((correct & mask).sum())
                counts[key][1] += int(mask.sum())
    return dict(cross_entropy=totals['cross_entropy'] / counts['all'][1],
                cdf_mse=totals['cdf_mse'] / counts['reachable'][1] if counts['reachable'][1] else None,
                accuracy={key: dict(correct=correct, support=count, rate=correct / count if count else None)
                          for key, (correct, count) in counts.items()},
                scope='cached validation value labels only; no model selection or native gameplay')


def verify_frozen_predictions(model, items, encoder_weights, before):
    with torch.inference_mode():
        after = model(items['raw'], items['state'], items['glyph'], player_probabilities(items, encoder_weights))
        unchanged = [*zip(before['field_logits'], after['field_logits']),
                     (before['event_logits'], after['event_logits']),
                     (before['action_logits'], model.score_outcomes(before['field_logits'], before['value_logits'], before['event_logits']))]
        if not all(torch.equal(left, right) for left, right in unchanged):
            raise ValueError('frozen physical/event predictions or fixed-input comparator changed')
        if not bool(torch.isfinite(after['value_logits']).all()) or torch.equal(before['value_logits'], after['value_logits']):
            raise ValueError('native value logits did not make a finite change')
        return dict(physical_and_event_predictions_bitwise_unchanged=True,
                    fixed_input_comparator_bitwise_unchanged=True, native_value_logits_changed=True,
                    native_action_changes_on_probe=int((before['action_logits'].argmax(-1) != after['action_logits'].argmax(-1)).sum()))


def checkpoint_for(parent, model, arm, args, report):
    result = copy.copy(parent)
    stale = ('supplemental_cache', 'supplemental_manifest_sha256', 'quality_manifest', 'quality_manifest_sha256',
             'quality_row_file_sha256', 'quality_filter', 'recovery_fraction', 'policy_row_fraction',
             'comparator_continuation', 'event_repair_continuation', 'value_repair_continuation')
    history = (*stale, 'objective_weights', 'sampling', 'continuation_arm', 'source_checkpoint_sha256',
               'optimizer_steps', 'learning_rate', 'batch_size', 'planner_initialization', 'parent_training_metadata',
               'actual_outcome_comparator_auxiliary_training')
    result['parent_training_metadata'] = {key: copy.deepcopy(parent[key]) for key in history if key in parent}
    for key in stale:
        result.pop(key, None)
    result.update(planner_weights={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                  planner_initialization=dict(kind='warm_start', source_checkpoint_sha256=args.checkpoint_sha256,
                                              weights_sha256=report['initial_planner_weights_sha256']),
                  continuation_arm=arm, source_checkpoint_sha256=args.checkpoint_sha256,
                  optimizer='AdamW fused', optimizer_state='new; no resumed moments', optimizer_steps=args.steps,
                  precision='float32', seed=args.seed,
                  learning_rate=args.lr, weight_decay=.05, batch_size=BATCH_SIZE, train_levels=report['train_levels'],
                  actual_outcome_comparator_auxiliary_training=False,
                  learning_rate_schedule='3% linear warmup then cosine decay',
                  training_cache=str(args.cache.resolve()), cache_manifest_sha256=report['cache_manifest_sha256'],
                  source_sha256=dict(report['source_sha256']), sampling=report['sampling'],
                  objective_weights=dict(value_cross_entropy=1., finite_conditional_cdf_mse=ARMS[arm],
                                         physical=0., events=0., policy=0., pairwise_ordering=0.),
                  value_repair_continuation=dict(kind='original_train_replay_value_head_repair', arm=arm,
                      steps=args.steps, learning_rate=args.lr, seed=args.seed, batch_size=BATCH_SIZE, precision='float32',
                      trainable_names=report['trainable_names'], frozen_weights_sha256=report['frozen_weights_sha256'],
                      all_original_train_roots=True, zero_optimal_roots_retained=True, supplement_used=False,
                      unreachable_category=129, maximum_finite_distance=128, finite_thresholds=list(range(128)),
                      reachable_overflow='clip to finite distance 128; never map to unreachable',
                      comparator_parameters_frozen=True, shared_summary_frozen=True,
                      physical_and_event_heads_frozen=True, gameplay_gain_established=False))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--checkpoint-sha256', default=CHECKPOINT_SHA)
    parser.add_argument('--cache', type=Path, default=ROOT / 'data/reference-outcome-inputs-v1')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-seconds', type=int, default=600)
    parser.add_argument('--qualify', action='store_true', help='one disposable B1024 update per arm; no checkpoint')
    args = parser.parse_args(argv)
    if args.steps < 1 or not math.isfinite(args.lr) or not 0 < args.lr < 1 or not 1 <= args.max_seconds <= 600 or args.seed < 0:
        parser.error('positive steps/lr, nonnegative seed, and wall bound at most 600 seconds required')
    if args.out.exists():
        raise FileExistsError(args.out)
    started = time.monotonic()
    report = dict(status='running', pid=os.getpid(), started_local=datetime.now().astimezone().isoformat(),
                  planned_steps=args.steps, mode='qualification' if args.qualify else 'training', max_seconds=args.max_seconds,
                  batch_size=BATCH_SIZE, learning_rate=args.lr, seed=args.seed, arms={}, source_sha256={},
                  parent_sha256=args.checkpoint_sha256, sampling='uniform distinct TRAIN levels then uniform root among all eight',
                  official_training_inputs=False, privileged_inference_inputs=False, supplement_used=False,
                  ordinal_weights=ARMS, ordinal_normalization='mean over 128 thresholds and reachable branches only',
                  learning_rate_schedule='3% linear warmup then cosine decay',
                  precision='float32', device='cuda', torch_version=str(torch.__version__),
                  limits=['Original TRAIN-only replay; no new current-policy collection or validation selection.',
                          'Only the existing value head trains; frozen public summaries may limit recoverability.',
                          'Physical/event heads and comparator parameters remain frozen; native actions can change.',
                          'Exact-value substitution gains do not establish that this head-only repair improves gameplay.'])
    previous_handler = signal.getsignal(signal.SIGALRM)
    def timeout(*_):
        raise TimeoutError('bounded value repair deadline exceeded')
    arrays, summaries, created = {}, {}, False
    try:
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(args.max_seconds)
        guard(); gpu_available()
        args.out.mkdir(parents=True); created = True
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        bindings = Bindings()
        bindings.add(args.checkpoint, args.checkpoint_sha256)
        for path in [Path(__file__), args.cache / 'manifest.json', *sorted((ROOT / 'pebby/agent').glob('*.py')),
                     *[ROOT / 'tools' / name for name in ('train_reference_outcomes.py', 'cache_reference_outcome_inputs.py',
                         'train_reference_spatial_outcomes.py', 'train_spatial_recovery_comparison.py', 'train_spatial_event_repair.py')]]:
            bindings.add(path)
        parent = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        validate_checkpoint(parent)
        cache_hash = bindings.add(args.cache / 'manifest.json')
        if cache_hash != parent['cache_manifest_sha256']:
            raise ValueError('parent/cache manifest mismatch')
        arrays, cache_hashes, cache_stats = load_data(args.cache)
        sampler = LevelSampler(arrays['train']['seeds'], arrays['validation']['seeds'])
        if len(sampler.seeds) < BATCH_SIZE:
            raise ValueError('at least 1024 distinct TRAIN levels required')
        model = SpatialOutcomePlanner(parent['planner_config']).cuda().eval()
        model.load_state_dict(parent['planner_weights'], strict=True)
        trainable = freeze_except_value_head(model)
        digest = frozen_digest(model)
        report.update(cache_manifest_sha256=cache_hash, trainable_names=trainable,
                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      initial_planner_weights_sha256=weights_sha256(model.state_dict()), frozen_weights_sha256=digest,
                      train_levels=len(sampler.seeds), train_roots=len(arrays['train']['seeds']),
                      zero_optimal_train_roots=int((arrays['train']['optimal'] == 0).sum()), validation_seed_disjoint=True,
                      train_distance_max=int(arrays['train']['distances'].max()),
                      train_finite_overflow_branches=int((arrays['train']['distances'] > 128).sum()))
        encoder_weights = {key: value.cuda() for key, value in parent['encoder_weights'].items() if key.startswith('player_head.')}
        probe_items = batch(arrays['train'], np.arange(128), 'cuda')
        with torch.inference_mode():
            probe = model(probe_items['raw'], probe_items['state'], probe_items['glyph'], player_probabilities(probe_items, encoder_weights))
        for split in ('train', 'validation'):
            path = args.out / (split + '-summary.npy')
            summaries[split] = cache_summaries(model, arrays[split], encoder_weights, path)
            bindings.add(path)
        with torch.inference_mode():
            summary, _ = selected_items(arrays['train'], summaries['train'], np.arange(128), 'cuda')
            if not torch.equal(model.value_head(summary), probe['value_logits']):
                raise ValueError('cached public summary does not reproduce native value logits')
        report['cached_value_probe_bitwise_equal'] = True
        report['source_sha256'] = {**cache_hashes, **bindings.hashes}
        report['parent_validation'] = evaluate_value(model, arrays['validation'], summaries['validation'])
        write(args.out / 'report.json', report)
        for arm, ordinal_weight in ARMS.items():
            guard(); gpu_available()
            torch.cuda.reset_peak_memory_stats()
            model.load_state_dict(parent['planner_weights'], strict=True)
            freeze_except_value_head(model)
            optimizer = optimizer_for(model, args.lr)
            rng = np.random.default_rng(args.seed)
            arm_report = dict(status='running', initial_weights_sha256=weights_sha256(model.state_dict()),
                              new_optimizer_moments=not bool(optimizer.state), ordinal_weight=ordinal_weight, updates=[])
            report['arms'][arm] = arm_report
            sampling_hash = hashlib.sha256()
            zero_roots, reachable_count, unreachable_count = 0, 0, 0
            for step in range(1 if args.qualify else args.steps):
                available = guard()
                rows = sampler.sample(rng)
                if len(np.unique(arrays['train']['seeds'][rows])) != BATCH_SIZE:
                    raise ValueError('sampled batch repeats TRAIN levels')
                sampling_hash.update(rows.tobytes())
                summary, distances = selected_items(arrays['train'], summaries['train'], rows, 'cuda')
                for group in optimizer.param_groups:
                    group['lr'] = args.lr * schedule(step, args.steps)
                record, norm = fit_step(model, optimizer, summary, distances, ordinal_weight)
                zero_roots += int((arrays['train']['optimal'][rows] == 0).sum())
                reachable_count += int((distances >= 0).sum())
                unreachable_count += int((distances < 0).sum())
                if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.steps:
                    row = dict(step=step + 1, losses={key: float(value.detach()) for key, value in record.items()},
                               all_trainable_gradients_finite=True, frozen_gradients_absent=True,
                               unclipped_gradient_norm=norm, available_host_bytes=available,
                               peak_gpu_bytes=torch.cuda.max_memory_allocated())
                    arm_report['updates'].append(row)
                    write(args.out / 'report.json', report)
                    print(f'{arm} update {step + 1}: loss={row["losses"]["total"]:.6f}', flush=True)
            if frozen_digest(model) != digest:
                raise ValueError('frozen model weights changed')
            final_hash = weights_sha256(model.state_dict())
            if final_hash == arm_report['initial_weights_sha256']:
                raise ValueError('value optimizer did not change weights')
            arm_report.update(verify_frozen_predictions(model, probe_items, encoder_weights, probe))
            arm_report.update(status='complete', final_weights_sha256=final_hash, frozen_weights_unchanged=True,
                              sampling_sha256=sampling_hash.hexdigest(), zero_optimal_roots_sampled=zero_roots,
                              reachable_branches_sampled=reachable_count, unreachable_branches_sampled=unreachable_count,
                              peak_gpu_bytes=torch.cuda.max_memory_allocated(), qualification_updates_discarded=args.qualify,
                              validation=evaluate_value(model, arrays['validation'], summaries['validation']))
            if not args.qualify:
                destination = args.out / (arm + '.pending.pt')
                torch.save(checkpoint_for(parent, model, arm, args, report), destination)
                arm_report['checkpoint_sha256'] = sha(destination)
            del optimizer, record, summary, distances
        for key in ('initial_weights_sha256', 'sampling_sha256', 'zero_optimal_roots_sampled',
                    'reachable_branches_sampled', 'unreachable_branches_sampled'):
            if report['arms']['control'][key] != report['arms']['ordinal'][key]:
                raise ValueError(f'matched arms differ: {key}')
        bindings.verify()
        if any(stat(path) != expected for path, expected in cache_stats.items()):
            raise ValueError('verified base cache changed during repair')
        if weights_sha256(parent['encoder_weights']) != parent['encoder_weights_sha256']:
            raise ValueError('embedded encoder changed')
        if not args.qualify:
            for arm in ARMS:
                (args.out / (arm + '.pending.pt')).replace(args.out / (arm + '.pt'))
        report.update(status='complete', sources_unchanged=True, qualification_passed=args.qualify)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        if created:
            for arm in ARMS:
                (args.out / (arm + '.pending.pt')).unlink(missing_ok=True)
                (args.out / (arm + '.pt')).unlink(missing_ok=True)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        for values in arrays.values():
            release(values, close=True)
        release(summaries, close=True)
        report.update(elapsed_seconds=time.monotonic() - started, finished_local=datetime.now().astimezone().isoformat())
        if created:
            write(args.out / 'report.json', report)
    return report


if __name__ == '__main__':
    main()
