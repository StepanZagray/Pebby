"""Matched budget/lives/event-head replay on frozen public spatial summaries.

Both arms restore all original generated TRAIN roots, including optimal=0.
Control uses natural-frequency CE/BCE. Balanced changes only budget/lives CE
weights, computed once from TRAIN. No policy/value loss or new collection is
used. Frozen comparator parameters do not imply unchanged action decisions:
the comparator consumes the repaired predictions.
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
from torch.nn import functional as F

from pebby.agent.neural_outcome_planner import EVENT_NAMES, FIELD_NAMES
from pebby.agent.neural_outcome_policy import weights_sha256
from pebby.agent.spatial_outcome_planner import SpatialOutcomePlanner
from pebby.agent.world_grounding import SIZES
from tools.cache_reference_outcome_inputs import Bindings, release, stat
from tools.train_reference_outcomes import batch, gpu_available, guard, load_data, optimizer_for, sha, write
from tools.train_reference_spatial_outcomes import player_probabilities
from tools.train_spatial_recovery_comparison import schedule, validate_checkpoint

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'artifacts/spatial-recovery-v1/quality-fit/recovery.pt'
CHECKPOINT_SHA = 'ff88327214b6dc2d4278167e0d61edcc37c683292b788be6927b71286331a5f8'
BATCH_SIZE = 1024
REPAIR_PREFIXES = ('field_heads.3.', 'field_heads.4.', 'event_head.')
BUDGET_COHORTS = ('live_decrease', 'live_unchanged', 'live_refill', 'life_loss', 'terminal_without_life_loss')
TARGET_KEYS = ('current_steps', 'current_lives', 'next_steps', 'next_lives', *EVENT_NAMES)


def freeze_except_repair_heads(model):
    # field_logits prepends the separate player head to these five field heads.
    if (FIELD_NAMES[4:] != ('steps', 'lives') or len(model.field_heads) != 5
            or tuple(head.out_features for head in model.field_heads) != tuple(SIZES[1:])):
        raise ValueError('spatial steps/lives head indices changed')
    names = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(REPAIR_PREFIXES))
        parameter.grad = None
        if parameter.requires_grad:
            names.append(name)
    expected = {prefix + suffix for prefix in REPAIR_PREFIXES for suffix in ('weight', 'bias')}
    if set(names) != expected:
        raise ValueError('repair must train exactly three linear heads')
    return names


def frozen_digest(model):
    return weights_sha256({key: value for key, value in model.state_dict().items()
                           if not key.startswith(REPAIR_PREFIXES)})


def repair_targets(items):
    following = items['next_steps'].long()
    if following.ndim != 2 or following.shape[1] != 4 or not len(following):
        raise ValueError('next steps must be nonempty [B,4]')
    size = len(following)
    for key in TARGET_KEYS:
        target = items[key]
        expected = (size,) if key.startswith('current_') else (size, 4)
        if target.shape != expected or target.is_floating_point():
            raise ValueError(f'integer or boolean target shape mismatch: {key}')
    events = torch.stack([items[name] for name in EVENT_NAMES], -1)
    if not bool(((events == 0) | (events == 1)).all()):
        raise ValueError('event targets must be binary')
    steps, lives = following.clamp_min(-1) + 1, items['next_lives'].long()
    if bool(((steps < 0) | (steps >= SIZES[4])).any() or ((lives < 0) | (lives >= SIZES[5])).any()):
        raise ValueError('budget/lives targets outside decoder categories')
    lost, terminal = items['lost_life'].bool(), items['terminal'].bool()
    live, previous = ~lost & ~terminal, items['current_steps'][:, None]
    groups = torch.full_like(following, 4)
    groups[live & (following < previous)] = 0
    groups[live & (following == previous)] = 1
    groups[live & (following > previous)] = 2
    groups[lost] = 3
    return dict(steps=steps, lives=lives, events=events.float(), budget_group=groups, lives_group=lost.long())


def cohort_weights(arrays):
    items = {key: torch.from_numpy(np.array(arrays[key], copy=True)) for key in TARGET_KEYS}
    target = repair_targets(items)
    budget = torch.bincount(target['budget_group'].flatten(), minlength=5).numpy()
    lives = torch.bincount(target['lives_group'].flatten(), minlength=2).numpy()
    if np.any(budget[:4] == 0) or np.any(lives == 0):
        raise ValueError('TRAIN must contain each of four budget and two lives cohorts')
    # Reallocate the original aggregate mass of the four named cohorts equally.
    # Terminal-without-loss branches remain at their natural weight, not dropped.
    budget_weights = np.ones(5, dtype=np.float64)
    budget_weights[:4] = budget[:4].sum() / (4 * budget[:4])
    lives_weights = lives.sum() / (2 * lives)
    return dict(budget_cohort_names=list(BUDGET_COHORTS), budget_counts=budget.tolist(),
                budget_weights=budget_weights.tolist(), lives_cohort_names=['no_life_loss', 'life_loss'],
                lives_counts=lives.tolist(), lives_weights=lives_weights.tolist(),
                event_bce='natural frequency; no positive-class weight in either arm',
                normalization='fixed TRAIN weights; mean over all sampled branches, not mean of present cohorts')


class LevelSampler:
    """Uniform distinct levels, then uniform original root; no action filtering."""

    def __init__(self, train_seeds, validation_seeds):
        seeds = np.asarray(train_seeds)
        if seeds.ndim != 1 or not len(seeds) or not np.issubdtype(seeds.dtype, np.integer):
            raise ValueError('nonempty integer TRAIN seeds required')
        self.seeds, counts = np.unique(seeds, return_counts=True)
        if np.intersect1d(self.seeds, np.asarray(validation_seeds)).size:
            raise ValueError('TRAIN and validation seeds overlap')
        if np.any(counts != 8):
            raise ValueError('original TRAIN replay requires eight roots per level')
        self.rows = np.argsort(seeds, kind='stable').reshape(len(self.seeds), 8)

    def sample(self, rng, size=BATCH_SIZE):
        if size < 1 or size > len(self.seeds):
            raise ValueError('insufficient distinct TRAIN levels for batch')
        levels = rng.choice(len(self.seeds), size=size, replace=False)
        return self.rows[levels, rng.integers(0, 8, size=size)]


def head_predictions(model, summaries):
    if summaries.ndim != 3 or summaries.shape[1:] != (4, model.cfg.summary):
        raise ValueError('cached summaries must have shape [B,4,summary]')
    return dict(steps=model.field_heads[3](summaries), lives=model.field_heads[4](summaries),
                events=model.event_head(summaries))


def repair_loss(model, summaries, items, weights, balanced):
    target, prediction = repair_targets(items), head_predictions(model, summaries)
    steps = F.cross_entropy(prediction['steps'].flatten(0, 1), target['steps'].flatten(), reduction='none').reshape_as(target['steps'])
    lives = F.cross_entropy(prediction['lives'].flatten(0, 1), target['lives'].flatten(), reduction='none').reshape_as(target['lives'])
    if balanced:
        steps = steps * torch.as_tensor(weights['budget_weights'], device=steps.device, dtype=steps.dtype)[target['budget_group']]
        lives = lives * torch.as_tensor(weights['lives_weights'], device=lives.device, dtype=lives.dtype)[target['lives_group']]
    losses = dict(steps=steps.mean(), lives=lives.mean(), events=F.binary_cross_entropy_with_logits(prediction['events'], target['events']))
    return dict(total=sum(losses.values()), losses=losses, target=target, prediction=prediction)


def fit_step(model, optimizer, summaries, items, weights, balanced):
    optimizer.zero_grad(set_to_none=True)
    record = repair_loss(model, summaries, items, weights, balanced)
    if not bool(torch.isfinite(record['total'])):
        raise ValueError('nonfinite repair loss')
    record['total'].backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise ValueError(f'missing or nonfinite repair gradient: {name}')
        elif parameter.grad is not None:
            raise ValueError(f'frozen parameter acquired gradient: {name}')
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
    optimizer.step()
    if any(not bool(torch.isfinite(parameter).all()) for parameter in trainable):
        raise ValueError('nonfinite repaired parameter')
    return record, float(norm)


def cache_summaries(model, arrays, encoder_weights, path, *, device='cuda', size=128):
    """Capture the public-only summary once; never concatenate labels to features."""
    output = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                     shape=(len(arrays['seeds']), 4, model.cfg.summary))
    captured = []
    hook = model.summary_head.register_forward_hook(lambda _module, _inputs, summary: captured.append(summary))
    try:
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(output), size):
                guard()
                rows = np.arange(start, min(start + size, len(output)))
                items = batch(arrays, rows, device)
                captured.clear()
                prediction = model(items['raw'], items['state'], items['glyph'], player_probabilities(items, encoder_weights))
                if len(captured) != 1 or captured[0].shape != (len(rows), 4, model.cfg.summary):
                    raise ValueError('summary hook did not capture exactly one public summary')
                summary = captured[0]
                if not bool(torch.isfinite(summary).all()):
                    raise ValueError('nonfinite frozen summary')
                heads = head_predictions(model, summary)
                for key, original in [('steps', prediction['field_logits'][4]),
                                      ('lives', prediction['field_logits'][5]), ('events', prediction['event_logits'])]:
                    if not torch.equal(heads[key], original):
                        raise ValueError(f'cached summary does not reproduce original {key} logits')
                output[rows] = summary.float().cpu().numpy()
                if start == 0 or (start + len(rows)) % 4096 == 0 or start + len(rows) == len(output):
                    print(f'summary cache {path.name}: {start + len(rows)}/{len(output)} roots', flush=True)
        output.flush()
    finally:
        hook.remove()
        captured.clear()
        release({'summary': output}, close=True)
    path.chmod(0o444)
    return np.load(path, mmap_mode='r', allow_pickle=False)


def selected_items(arrays, summaries, rows, device):
    items = {key: torch.from_numpy(np.array(arrays[key][rows], copy=True)).to(device) for key in TARGET_KEYS}
    return torch.from_numpy(np.array(summaries[rows], copy=True)).to(device), items


def evaluate_heads(model, arrays, summaries, device='cuda'):
    totals = {name: 0. for name in ('steps_ce', 'lives_ce', 'events_bce', 'events_brier')}
    counts = {name: [0, 0] for name in (*BUDGET_COHORTS, 'lives', 'lives_on_life_loss', 'refill_increase', *EVENT_NAMES)}
    contradictions = dict(life_loss_without_lives_decrease=0, won_without_terminal=0)
    branches = 0
    with torch.inference_mode():
        for start in range(0, len(summaries), BATCH_SIZE):
            guard()
            rows = np.arange(start, min(start + BATCH_SIZE, len(summaries)))
            summary, items = selected_items(arrays, summaries, rows, device)
            result = repair_loss(model, summary, items, {}, False)
            target, pred = result['target'], result['prediction']
            steps, lives, events = pred['steps'].argmax(-1), pred['lives'].argmax(-1), pred['events'] >= 0
            count = len(rows) * 4
            branches += count
            for key in ('steps', 'lives', 'events'):
                totals[key + ('_bce' if key == 'events' else '_ce')] += float(result['losses'][key]) * count
            totals['events_brier'] += float((pred['events'].sigmoid() - target['events']).square().mean()) * count
            populations = [(name, steps == target['steps'], target['budget_group'] == index) for index, name in enumerate(BUDGET_COHORTS)]
            populations += [('lives', lives == target['lives'], torch.ones_like(lives, dtype=torch.bool)),
                            ('lives_on_life_loss', lives == target['lives'], items['lost_life'].bool()),
                            ('refill_increase', steps - 1 > items['current_steps'][:, None], target['budget_group'] == 2)]
            populations += [(name, events[..., index], target['events'][..., index].bool()) for index, name in enumerate(EVENT_NAMES)]
            for name, correct, mask in populations:
                counts[name][0] += int((correct & mask).sum())
                counts[name][1] += int(mask.sum())
            contradictions['life_loss_without_lives_decrease'] += int((events[..., 0] & (lives >= items['current_lives'][:, None])).sum())
            contradictions['won_without_terminal'] += int((events[..., 2] & ~events[..., 1]).sum())
    return dict(losses={key: value / branches for key, value in totals.items()},
                cohorts={name: dict(correct=correct, support=support, rate=correct / support if support else None)
                         for name, (correct, support) in counts.items()},
                event_rates_are_recall=True, contradictions_at_zero_logit=contradictions, branches=branches,
                scope='cached validation summaries, not native gameplay or calibrated risk probabilities')


def verify_frozen_predictions(model, items, encoder_weights, before):
    with torch.inference_mode():
        after = model(items['raw'], items['state'], items['glyph'], player_probabilities(items, encoder_weights))
        unchanged = [*zip(before['field_logits'][:4], after['field_logits'][:4]),
                     (before['value_logits'], after['value_logits']),
                     (before['action_logits'], model.score_outcomes(before['field_logits'], before['value_logits'], before['event_logits']))]
        if not all(torch.equal(left, right) for left, right in unchanged):
            raise ValueError('frozen field/value predictions or fixed-input comparator changed')
        return dict(unaffected_fields_and_value_bitwise_unchanged=True, fixed_input_comparator_bitwise_unchanged=True,
                    native_action_changes_on_probe=int((before['action_logits'].argmax(-1) != after['action_logits'].argmax(-1)).sum()))


def checkpoint_for(parent, model, arm, args, report):
    result = copy.copy(parent)
    stale = ('supplemental_cache', 'supplemental_manifest_sha256', 'quality_manifest', 'quality_manifest_sha256',
             'quality_row_file_sha256', 'quality_filter', 'recovery_fraction', 'policy_row_fraction', 'comparator_continuation')
    history = (*stale, 'objective_weights', 'sampling', 'continuation_arm', 'source_checkpoint_sha256',
               'actual_outcome_comparator_auxiliary_training',
               'optimizer_steps', 'learning_rate', 'batch_size', 'planner_initialization')
    result['parent_training_metadata'] = {key: copy.deepcopy(parent[key]) for key in history if key in parent}
    for key in stale:
        result.pop(key, None)
    balanced = arm == 'balanced'
    result.update(planner_weights={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                  planner_initialization=dict(kind='warm_start', source_checkpoint_sha256=args.checkpoint_sha256,
                                              weights_sha256=report['initial_planner_weights_sha256']),
                  continuation_arm=arm, source_checkpoint_sha256=args.checkpoint_sha256,
                  optimizer='AdamW fused', optimizer_state='new; no resumed moments', optimizer_steps=args.steps,
                  weight_decay=.05, actual_outcome_comparator_auxiliary_training=False,
                  learning_rate=args.lr, batch_size=BATCH_SIZE, train_levels=report['train_levels'],
                  training_cache=str(args.cache.resolve()), cache_manifest_sha256=report['cache_manifest_sha256'],
                  source_sha256=dict(report['source_sha256']), sampling=report['sampling'],
                  objective_weights=dict(steps=1., lives=1., events=1., policy=0., value=0.,
                                         cohort_balanced=balanced, event_positive_weight=None),
                  event_repair_continuation=dict(kind='original_train_replay_head_repair', arm=arm,
                      steps=args.steps, learning_rate=args.lr, seed=args.seed, batch_size=BATCH_SIZE, precision='float32',
                      trainable_names=report['trainable_names'], frozen_weights_sha256=report['frozen_weights_sha256'],
                      cohort_weights=report['cohort_weights'] if balanced else None,
                      all_original_train_roots=True, zero_optimal_roots_retained=True, supplement_used=False,
                      policy_or_value_loss=False, comparator_parameters_frozen=True, shared_summary_frozen=True,
                      gameplay_gain_established=False))
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
                  batch_size=BATCH_SIZE, learning_rate=args.lr, seed=args.seed, precision='float32',
                  matmul_tf32=False, cudnn_tf32=True, arms={}, source_sha256={},
                  parent_sha256=args.checkpoint_sha256, sampling='uniform distinct TRAIN levels then uniform root among all eight',
                  official_training_inputs=False, privileged_inference_inputs=False, supplement_used=False,
                  limits=['Original TRAIN-only replay restoration; no new current-policy collection.',
                          'Only three linear heads train; frozen summaries may limit recoverability.',
                          'Comparator parameters, other fields, and distance head remain frozen; native actions can change.',
                          'No persistent memory, voluntary reset, horizon increase, calibration guarantee, or gameplay gain established.'])
    previous_handler = signal.getsignal(signal.SIGALRM)
    def timeout(*_):
        raise TimeoutError('head repair exceeded its wall bound')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.max_seconds)
    arrays, summaries = {}, {}
    created = False
    try:
        guard()
        gpu_available()
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU required for the qualified training path')
        args.out.mkdir(parents=True)
        created = True
        write(args.out / 'report.json', report)
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        bindings = Bindings()
        bindings.add(args.checkpoint, args.checkpoint_sha256)
        for path in [Path(__file__), args.cache / 'manifest.json', *sorted((ROOT / 'pebby/agent').glob('*.py')),
                     *[ROOT / 'tools' / name for name in ('train_reference_outcomes.py', 'cache_reference_outcome_inputs.py',
                         'train_reference_spatial_outcomes.py', 'train_spatial_recovery_comparison.py')]]:
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
        weights = cohort_weights(arrays['train'])
        model = SpatialOutcomePlanner(parent['planner_config']).cuda().eval()
        model.load_state_dict(parent['planner_weights'], strict=True)
        trainable = freeze_except_repair_heads(model)
        digest = frozen_digest(model)
        report.update(cache_manifest_sha256=cache_hash, cohort_weights=weights, trainable_names=trainable,
                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      initial_planner_weights_sha256=weights_sha256(model.state_dict()), frozen_weights_sha256=digest,
                      train_levels=len(sampler.seeds), train_roots=len(arrays['train']['seeds']),
                      zero_optimal_train_roots=int((arrays['train']['optimal'] == 0).sum()), validation_seed_disjoint=True)
        encoder_weights = {key: value.cuda() for key, value in parent['encoder_weights'].items() if key.startswith('player_head.')}
        probe_items = batch(arrays['validation'], np.arange(8), 'cuda')
        with torch.inference_mode():
            probe = model(probe_items['raw'], probe_items['state'], probe_items['glyph'], player_probabilities(probe_items, encoder_weights))
        for split in ('train', 'validation'):
            path = args.out / (split + '-summary.npy')
            summaries[split] = cache_summaries(model, arrays[split], encoder_weights, path)
            bindings.add(path)
        report['source_sha256'] = {**cache_hashes, **bindings.hashes}
        report['parent_validation'] = evaluate_heads(model, arrays['validation'], summaries['validation'])
        write(args.out / 'report.json', report)
        for arm in ('control', 'balanced'):
            guard()
            gpu_available()
            torch.cuda.reset_peak_memory_stats()
            model.load_state_dict(parent['planner_weights'], strict=True)
            freeze_except_repair_heads(model)
            optimizer = optimizer_for(model, args.lr)
            rng = np.random.default_rng(args.seed)
            arm_report = dict(status='running', initial_weights_sha256=weights_sha256(model.state_dict()),
                              new_optimizer_moments=not bool(optimizer.state), updates=[])
            report['arms'][arm] = arm_report
            sampling_hash = hashlib.sha256()
            budget_counts, lives_counts = np.zeros(5, dtype=np.int64), np.zeros(2, dtype=np.int64)
            for step in range(1 if args.qualify else args.steps):
                available = guard()
                rows = sampler.sample(rng)
                if len(np.unique(arrays['train']['seeds'][rows])) != BATCH_SIZE:
                    raise ValueError('sampled batch repeats TRAIN levels')
                sampling_hash.update(rows.tobytes())
                summary, items = selected_items(arrays['train'], summaries['train'], rows, 'cuda')
                for group in optimizer.param_groups:
                    group['lr'] = args.lr * schedule(step, args.steps)
                record, norm = fit_step(model, optimizer, summary, items, weights, arm == 'balanced')
                budget_counts += torch.bincount(record['target']['budget_group'].flatten(), minlength=5).cpu().numpy()
                lives_counts += torch.bincount(record['target']['lives_group'].flatten(), minlength=2).cpu().numpy()
                if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.steps:
                    row = dict(step=step + 1, loss=float(record['total'].detach()),
                               losses={key: float(value.detach()) for key, value in record['losses'].items()},
                               all_trainable_gradients_finite=True, frozen_gradients_absent=True,
                               unclipped_gradient_norm=norm, available_host_bytes=available,
                               peak_gpu_bytes=torch.cuda.max_memory_allocated())
                    arm_report['updates'].append(row)
                    write(args.out / 'report.json', report)
                    print(f'{arm} update {step + 1}: loss={row["loss"]:.6f}', flush=True)
            if frozen_digest(model) != digest:
                raise ValueError('frozen model weights changed')
            final_hash = weights_sha256(model.state_dict())
            if final_hash == arm_report['initial_weights_sha256']:
                raise ValueError('repair optimizer did not change weights')
            arm_report.update(verify_frozen_predictions(model, probe_items, encoder_weights, probe))
            arm_report.update(status='complete', final_weights_sha256=final_hash, frozen_weights_unchanged=True,
                              sampling_sha256=sampling_hash.hexdigest(), realized_budget_counts=budget_counts.tolist(),
                              realized_lives_counts=lives_counts.tolist(), peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                              qualification_updates_discarded=args.qualify,
                              validation=evaluate_heads(model, arrays['validation'], summaries['validation']))
            if not args.qualify:
                destination = args.out / (arm + '.pending.pt')
                torch.save(checkpoint_for(parent, model, arm, args, report), destination)
                arm_report['checkpoint_sha256'] = sha(destination)
            del optimizer, record, summary, items
        for key in ('initial_weights_sha256', 'sampling_sha256', 'realized_budget_counts', 'realized_lives_counts'):
            if report['arms']['control'][key] != report['arms']['balanced'][key]:
                raise ValueError(f'matched arms differ: {key}')
        bindings.verify()
        if any(stat(path) != expected for path, expected in cache_stats.items()):
            raise ValueError('verified base cache changed during repair')
        if weights_sha256(parent['encoder_weights']) != parent['encoder_weights_sha256']:
            raise ValueError('embedded encoder changed')
        if not args.qualify:
            for arm in ('control', 'balanced'):
                (args.out / (arm + '.pending.pt')).replace(args.out / (arm + '.pt'))
        report.update(status='complete', sources_unchanged=True, qualification_passed=args.qualify)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        if created:
            for arm in ('control', 'balanced'):
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
