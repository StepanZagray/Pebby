"""Matched value-head CE versus CE plus native optimal-action cross entropy.

Only the existing value head trains. Policy gradients traverse the frozen
comparator into predicted values; physical/events and public summaries are
frozen. Every original TRAIN root is cached, including optimal=0 roots, which
retain distance CE but have no policy target. No ordinal or teacher-score loss.
"""
import argparse
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

from pebby.agent.outcome_ordering import masked_optimal_set_cross_entropy
from pebby.agent.spatial_outcome_planner import SpatialOutcomePlanner
from pebby.agent.spatial_value_objective import value_targets
from pebby.agent.world_grounding import SIZES
from tools.audit_spatial_decisions import score_probabilities
from tools.cache_reference_outcome_inputs import Bindings, release, stat
from tools.train_reference_outcomes import batch, gpu_available, guard, load_data, optimizer_for, sha, write
from tools.train_reference_spatial_outcomes import player_probabilities
from tools.train_spatial_event_repair import BATCH_SIZE, CHECKPOINT, CHECKPOINT_SHA, LevelSampler, ROOT
from tools.train_spatial_recovery_comparison import schedule, validate_checkpoint
from tools import train_spatial_value_repair as value_repair

ARMS = {'control': 0., 'decision': 1.}
PHYSICAL_WIDTH = sum(SIZES)
NONVALUE_WIDTH = PHYSICAL_WIDTH + 3


def native_scores(model, value_logits, nonvalue):
    """Fixed public probabilities plus differentiable predicted value probabilities."""
    if (nonvalue.shape != (*value_logits.shape[:2], NONVALUE_WIDTH)
            or nonvalue.device != value_logits.device or not nonvalue.is_floating_point()):
        raise ValueError('nonvalue probabilities must match [B,4,209] on the value device')
    fixed = nonvalue.detach()
    probabilities = torch.cat((fixed[..., :PHYSICAL_WIDTH], value_logits.float().softmax(-1),
                               fixed[..., PHYSICAL_WIDTH:]), -1)
    # Freezing comparator parameters must not disable autograd through its input.
    return score_probabilities(model, probabilities)


def decision_loss(model, summaries, nonvalue, distances, optimal, policy_weight):
    if not math.isfinite(policy_weight) or policy_weight < 0:
        raise ValueError('policy weight must be finite and nonnegative')
    value = model.value_head(summaries.detach())
    target = value_targets(value, distances)
    ce = F.cross_entropy(value.float().flatten(0, 1), target.flatten())
    policy = masked_optimal_set_cross_entropy(native_scores(model, value, nonvalue), optimal)
    return dict(total=ce + policy_weight * policy, value_cross_entropy=ce, native_policy_cross_entropy=policy)


def fit_step(model, optimizer, payload, policy_weight, *, probe_policy_gradient=False):
    optimizer.zero_grad(set_to_none=True)
    record = decision_loss(model, *payload, policy_weight)
    if not bool(torch.isfinite(record['total'])):
        raise ValueError('nonfinite native-decision loss')
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    diagnostic = {}
    if probe_policy_gradient:
        if not bool((payload[-1] != 0).any()):
            raise ValueError('policy-gradient qualification needs supported roots')
        policy_gradients = torch.autograd.grad(record['native_policy_cross_entropy'], trainable, retain_graph=True)
        if any(not bool(torch.isfinite(gradient).all()) for gradient in policy_gradients):
            raise ValueError('nonfinite native policy-only gradient')
        norm = float(torch.stack([gradient.square().sum() for gradient in policy_gradients]).sum().sqrt())
        if norm == 0:
            raise ValueError('native policy gradients do not reach the value head')
        diagnostic = dict(policy_only_value_gradient_norm=norm, native_policy_gradient_reaches_value_head=True)
    record['total'].backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise ValueError(f'missing or nonfinite value gradient: {name}')
        elif parameter.grad is not None:
            raise ValueError(f'frozen parameter acquired gradient: {name}')
    diagnostic['unclipped_gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True))
    optimizer.step()
    if any(not bool(torch.isfinite(parameter).all()) for parameter in trainable):
        raise ValueError('nonfinite updated value parameter')
    return record, diagnostic


def cache_public_inputs(model, arrays, encoder_weights, directory, *, device='cuda', size=128):
    """Cache all root IDs in one pass, proving full native score parity per batch."""
    directory.mkdir()
    paths = {key: directory / (key + '.npy') for key in ('summary', 'nonvalue')}
    widths = dict(summary=model.cfg.summary, nonvalue=NONVALUE_WIDTH)
    outputs = {key: np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                           shape=(len(arrays['seeds']), 4, widths[key])) for key, path in paths.items()}
    captured = []
    hook = model.summary_head.register_forward_hook(lambda _module, _inputs, summary: captured.append(summary))
    try:
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(arrays['seeds']), size):
                guard()
                rows = np.arange(start, min(start + size, len(arrays['seeds'])))
                items = batch(arrays, rows, device)
                captured.clear()
                original = model(items['raw'], items['state'], items['glyph'], player_probabilities(items, encoder_weights))
                if len(captured) != 1 or captured[0].shape != (len(rows), 4, model.cfg.summary):
                    raise ValueError('public summary hook did not capture exactly one tensor')
                fixed = torch.cat([*[scores.float().softmax(-1) for scores in original['field_logits']],
                                   original['event_logits'].float().sigmoid()], -1)
                for key, tensor in [('summary', captured[0]), ('nonvalue', fixed)]:
                    if not bool(torch.isfinite(tensor).all()):
                        raise ValueError('nonfinite public cache inputs')
                    outputs[key][rows] = tensor.float().cpu().numpy()
                # Roundtrip both arrays, including roots with no action supervision.
                summary = torch.from_numpy(np.array(outputs['summary'][rows], copy=True)).to(device)
                nonvalue = torch.from_numpy(np.array(outputs['nonvalue'][rows], copy=True)).to(device)
                values = model.value_head(summary)
                if (not torch.equal(values, original['value_logits'])
                        or not torch.equal(native_scores(model, values, nonvalue), original['action_logits'])):
                    raise ValueError('cached public inputs do not reproduce native logits bitwise')
                if start == 0 or (start + len(rows)) % 4096 == 0 or start + len(rows) == len(arrays['seeds']):
                    print(f'{directory.name} public cache: {start + len(rows)}/{len(arrays["seeds"])} roots', flush=True)
        for output in outputs.values():
            output.flush()
    finally:
        hook.remove()
        captured.clear()
        release(outputs, close=True)
    for path in paths.values():
        path.chmod(0o444)
    return {key: np.load(path, mmap_mode='r', allow_pickle=False) for key, path in paths.items()}


def selected_items(arrays, cache, rows, device):
    sources = (cache['summary'], cache['nonvalue'], arrays['distances'], arrays['optimal'])
    return tuple(torch.from_numpy(np.array(source[rows], copy=True)).to(device) for source in sources)


def evaluate(model, arrays, cache, device='cuda'):
    total, correct, support = 0., 0, 0
    with torch.inference_mode():
        for start in range(0, len(cache['summary']), BATCH_SIZE):
            guard()
            rows = np.arange(start, min(start + BATCH_SIZE, len(cache['summary'])))
            summary, fixed, _distance, optimal = selected_items(arrays, cache, rows, device)
            scores = native_scores(model, model.value_head(summary), fixed)
            valid = optimal != 0
            count = int(valid.sum())
            total += float(masked_optimal_set_cross_entropy(scores, optimal)) * count
            chosen = scores.argmax(-1)
            correct += int((((optimal.long() >> chosen) & 1).bool() & valid).sum())
            support += count
    return dict(value=value_repair.evaluate_value(model, arrays, cache['summary'], device),
                native_policy=dict(cross_entropy=total / support if support else None,
                                   correct=correct, support=support, accuracy=correct / support if support else None),
                scope='cached public predictions; no native gameplay, validation selection, or calibrated risk claim')


def checkpoint_for(parent, model, arm, args, report):
    # Reuse the checked inference/provenance envelope, replacing the latest loss metadata.
    result = value_repair.checkpoint_for(parent, model, 'control', args, report)
    continuation = result.pop('value_repair_continuation')
    continuation.pop('finite_thresholds')
    if 'value_decision_continuation' in parent:
        result['parent_training_metadata']['value_decision_continuation'] = parent['value_decision_continuation']
    continuation.update(kind='value_head_native_decision_repair', arm=arm, policy_weight=ARMS[arm], ordinal_weight=0.,
                        teacher_outcome_policy_loss=False, pairwise_ordering_loss=False,
                        native_policy_gradient_route='frozen comparator input -> value probabilities -> value_head',
                        public_nonvalue_cache_covers_all_roots=True)
    result.update(continuation_arm=arm, value_decision_continuation=continuation,
                  objective_weights=dict(value_cross_entropy=1., native_policy_cross_entropy=ARMS[arm],
                                         physical=0., events=0., finite_conditional_cdf_mse=0.,
                                         teacher_policy=0., pairwise_ordering=0.))
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
    parser.add_argument('--qualify', action='store_true', help='one disposable B1024 update per arm; no checkpoints')
    args = parser.parse_args(argv)
    if args.steps < 1 or not math.isfinite(args.lr) or not 0 < args.lr < 1 or not 1 <= args.max_seconds <= 600 or args.seed < 0:
        parser.error('positive steps/lr, nonnegative seed, wall bound at most 600 seconds required')
    if args.out.exists():
        raise FileExistsError(args.out)
    started = time.monotonic()
    report = dict(status='running', pid=os.getpid(), started_local=datetime.now().astimezone().isoformat(),
                  mode='qualification' if args.qualify else 'training', qualification_mode=args.qualify,
                  qualification_passed=None, planned_steps=args.steps, max_seconds=args.max_seconds,
                  batch_size=BATCH_SIZE, learning_rate=args.lr, seed=args.seed, arms={}, source_sha256={},
                  weight_decay=.05, learning_rate_schedule='3% linear warmup then cosine decay',
                  parent_sha256=args.checkpoint_sha256, policy_weights=ARMS, ordinal_weight=0.,
                  sampling='uniform distinct TRAIN levels then uniform root among all eight',
                  policy_normalization='mean over nonzero optimal masks; uniform distribution on each optimal action set',
                  value_normalization='natural mean over all branches, including optimal=0 roots',
                  precision='float32', device='cuda', torch_version=str(torch.__version__),
                  matmul_tf32=False, cudnn_tf32=True, official_training_inputs=False, privileged_inference_inputs=False,
                  supplement_used=False, all_original_train_roots=True,
                  limits=['Native policy gradients may trade value fidelity for action choices; monitor both.',
                          'Frozen summaries may limit value recovery; physical/life-loss errors remain unchanged by design.',
                          'No new on-policy collection, teacher-score loss, or deeper planning.',
                          'No persistent memory or voluntary reset; no gameplay gain established.'])
    previous_handler = signal.getsignal(signal.SIGALRM)
    def timeout(*_):
        raise TimeoutError('bounded native-decision repair deadline exceeded')
    arrays, compact, created = {}, {}, False
    try:
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(args.max_seconds)
        guard(); gpu_available()
        args.out.mkdir(parents=True); created = True
        write(args.out / 'report.json', report)
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        bindings = Bindings()
        bindings.add(args.checkpoint, args.checkpoint_sha256)
        for path in [Path(__file__), args.cache / 'manifest.json', *sorted((ROOT / 'pebby/agent').glob('*.py')),
                     *[ROOT / 'tools' / name for name in ('audit_spatial_decisions.py', 'train_reference_outcomes.py',
                         'cache_reference_outcome_inputs.py', 'train_reference_spatial_outcomes.py',
                         'train_spatial_recovery_comparison.py', 'train_spatial_event_repair.py', 'train_spatial_value_repair.py')]]:
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
        names = value_repair.freeze_except_value_head(model)
        digest = value_repair.frozen_digest(model)
        report.update(cache_manifest_sha256=cache_hash, trainable_names=names,
                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      initial_planner_weights_sha256=value_repair.weights_sha256(model.state_dict()), frozen_weights_sha256=digest,
                      train_levels=len(sampler.seeds), train_roots=len(arrays['train']['seeds']),
                      zero_optimal_train_roots=int((arrays['train']['optimal'] == 0).sum()), validation_seed_disjoint=True)
        player = {key: value.cuda() for key, value in parent['encoder_weights'].items() if key.startswith('player_head.')}
        probe_items = batch(arrays['train'], np.arange(128), 'cuda')
        with torch.inference_mode():
            probe = model(probe_items['raw'], probe_items['state'], probe_items['glyph'], player_probabilities(probe_items, player))
        for split in ('train', 'validation'):
            compact[split] = cache_public_inputs(model, arrays[split], player, args.out / split)
            for key in compact[split]:
                bindings.add(args.out / split / (key + '.npy'))
        report.update(source_sha256={**cache_hashes, **bindings.hashes}, public_cache_all_roots_native_parity=True,
                      parent_validation=evaluate(model, arrays['validation'], compact['validation']))
        write(args.out / 'report.json', report)
        for arm, weight in ARMS.items():
            guard(); gpu_available()
            torch.cuda.reset_peak_memory_stats()
            model.load_state_dict(parent['planner_weights'], strict=True)
            value_repair.freeze_except_value_head(model)
            optimizer = optimizer_for(model, args.lr)
            rng = np.random.default_rng(args.seed)
            entry = dict(status='running', initial_weights_sha256=value_repair.weights_sha256(model.state_dict()),
                         new_optimizer_moments=not bool(optimizer.state), policy_weight=weight, updates=[])
            report['arms'][arm] = entry
            sampling_hash, zero_roots, reachable, unreachable = hashlib.sha256(), 0, 0, 0
            for step in range(1 if args.qualify else args.steps):
                available = guard()
                rows = sampler.sample(rng)
                if len(np.unique(arrays['train']['seeds'][rows])) != BATCH_SIZE:
                    raise ValueError('sampled batch repeats TRAIN levels')
                sampling_hash.update(rows.tobytes())
                payload = selected_items(arrays['train'], compact['train'], rows, 'cuda')
                for group in optimizer.param_groups:
                    group['lr'] = args.lr * schedule(step, args.steps)
                record, diagnostics = fit_step(model, optimizer, payload, weight, probe_policy_gradient=step == 0)
                zero_roots += int((payload[-1] == 0).sum())
                reachable += int((payload[-2] >= 0).sum())
                unreachable += int((payload[-2] < 0).sum())
                if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.steps:
                    row = dict(step=step + 1, losses={key: float(value.detach()) for key, value in record.items()},
                               all_trainable_gradients_finite=True, frozen_gradients_absent=True,
                               available_host_bytes=available, peak_gpu_bytes=torch.cuda.max_memory_allocated(), **diagnostics)
                    entry['updates'].append(row)
                    write(args.out / 'report.json', report)
                    print(f'{arm} update {step + 1}: loss={row["losses"]["total"]:.6f}', flush=True)
            if value_repair.frozen_digest(model) != digest:
                raise ValueError('frozen model weights changed')
            final_hash = value_repair.weights_sha256(model.state_dict())
            if final_hash == entry['initial_weights_sha256']:
                raise ValueError('native-decision optimizer did not change value weights')
            entry.update(value_repair.verify_frozen_predictions(model, probe_items, player, probe))
            entry.update(status='complete', final_weights_sha256=final_hash, frozen_weights_unchanged=True,
                         sampling_sha256=sampling_hash.hexdigest(), zero_optimal_roots_sampled=zero_roots,
                         reachable_branches_sampled=reachable, unreachable_branches_sampled=unreachable,
                         peak_gpu_bytes=torch.cuda.max_memory_allocated(), qualification_updates_discarded=args.qualify,
                         validation=evaluate(model, arrays['validation'], compact['validation']))
            if not args.qualify:
                destination = args.out / (arm + '.pending.pt')
                torch.save(checkpoint_for(parent, model, arm, args, report), destination)
                entry['checkpoint_sha256'] = sha(destination)
            del optimizer, payload, record
        for key in ('initial_weights_sha256', 'sampling_sha256', 'zero_optimal_roots_sampled',
                    'reachable_branches_sampled', 'unreachable_branches_sampled'):
            if report['arms']['control'][key] != report['arms']['decision'][key]:
                raise ValueError(f'matched arms differ: {key}')
        bindings.verify()
        if any(stat(path) != expected for path, expected in cache_stats.items()):
            raise ValueError('verified original cache changed during native-decision repair')
        if value_repair.weights_sha256(parent['encoder_weights']) != parent['encoder_weights_sha256']:
            raise ValueError('embedded frozen encoder changed')
        if not args.qualify:
            for arm in ARMS:
                (args.out / (arm + '.pending.pt')).replace(args.out / (arm + '.pt'))
        report.update(status='complete', sources_unchanged=True, qualification_passed=True if args.qualify else None)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}',
                      qualification_passed=False if args.qualify else None)
        if created:
            for arm in ARMS:
                (args.out / (arm + '.pending.pt')).unlink(missing_ok=True)
                (args.out / (arm + '.pt')).unlink(missing_ok=True)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        for values in (*arrays.values(), *compact.values()):
            release(values, close=True)
        report.update(elapsed_seconds=time.monotonic() - started, finished_local=datetime.now().astimezone().isoformat())
        if created:
            write(args.out / 'report.json', report)
    return report


if __name__ == '__main__':
    main()
