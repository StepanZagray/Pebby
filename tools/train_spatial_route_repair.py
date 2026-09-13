"""Whole-spatial replay repair, paired with a learned all-cell route readout.

The encoder stays frozen. Both arms train every planner tensor on the same
verified examples and loss families. Undefined policy labels only mask policy
supervision. Qualification is disposable and cannot publish a candidate.
"""
import argparse
from collections import Counter
from datetime import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from pebby.agent.route_repair_sampling import RouteReplaySampler, TIER_COUNTS, grouped_rows
from pebby.agent.spatial_outcome_planner import SpatialOutcomePlanner
from pebby.agent.spatial_route_outcome_planner import SpatialRouteOutcomePlanner
from pebby.agent.spatial_route_outcome_policy import checkpoint_from_parent
from pebby.agent.spatial_outcome_objective import training_weights
from pebby.agent.spatial_value_objective import value_targets
from pebby.agent.neural_outcome_policy import weights_sha256
from tools.cache_reference_outcome_inputs import Bindings, release, stat
from tools.train_reference_outcomes import ARRAYS, batch, gpu_available, guard, load_data, optimizer_for, scalar_metrics, sha, tiny_gate, write
from tools.train_reference_spatial_outcomes import fit_step, forward, player_probabilities
from tools.train_spatial_recovery_comparison import schedule, validate_checkpoint

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'artifacts/spatial-recovery-v1/quality-fit/recovery.pt'
PARENT_SHA = 'ff88327214b6dc2d4278167e0d61edcc37c683292b788be6927b71286331a5f8'
BANK = ROOT / 'data/ls20-reference-unequal-v1/train.jsonl'
BANK_SHA = 'f968f9b4a3690be69041eadd1afe129a3ba0527d55829d75687aa5c260532a81'
ARMS = ('control', 'route')


def build_model(parent, arm, device='cuda', seed=42):
    if arm not in ARMS:
        raise ValueError('unknown route repair arm')
    torch.manual_seed(seed)
    model = SpatialOutcomePlanner(parent['planner_config']).to(device).eval()
    model.load_state_dict(parent['planner_weights'], strict=True)
    if arm == 'route':
        model = SpatialRouteOutcomePlanner.from_parent(model)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def matched_items(base, recent, selection, device):
    rows, replacements = selection['base_rows'], selection['recent_rows']
    mask = replacements >= 0
    if not np.array_equal(base['seeds'][rows], selection['seeds']):
        raise ValueError('base rows do not match sampled levels')
    if mask.any() and (recent is None or not np.array_equal(recent['seeds'][replacements[mask]], selection['seeds'][mask])):
        raise ValueError('recent rows do not match sampled levels')
    result = {}
    for key in ARRAYS:
        if key in ('rows', 'seeds'):
            continue
        array = np.array(base[key][rows], copy=True)
        if mask.any():
            array[mask] = recent[key][replacements[mask]]
        result[key] = torch.from_numpy(array).to(device)
    return result


def tiny_selection(arrays, tiers, count=64, seed=42):
    """Distinct TRAIN levels, balanced across tiers and with rare-outcome cover."""
    rng = np.random.default_rng(seed)
    groups = grouped_rows(arrays['seeds'])
    quotas = [count // 7 + (tier < count % 7) for tier in range(7)]
    selected = []
    for tier, quota in enumerate(quotas, 1):
        choices = np.array(sorted(s for s in groups if tiers[s] == tier))
        for level in rng.choice(choices, quota, replace=False):
            rows = groups[int(level)]
            valid = rows[arrays['optimal'][rows] != 0]
            selected.append(int(rng.choice(valid if len(valid) else rows)))
    # Ensure useful positive targets without replacing a level's tier quota.
    cohorts = [np.asarray(arrays[name]).any(1) for name in ('lost_life', 'terminal', 'won')]
    cohorts += [(arrays['next_triple'][..., field] != arrays['current_triple'][:, None, field]).any(1) for field in range(3)]
    cohorts.append(arrays['optimal'] == 0)
    for cohort in cohorts:
        if cohort[selected].any():
            continue
        replaced = False
        for index, old in enumerate(selected):
            level = int(arrays['seeds'][old])
            candidates = groups[level][cohort[groups[level]]]
            if len(candidates):
                proposal = selected.copy()
                proposal[index] = int(candidates[0])
                if all(not mask[selected].any() or mask[proposal].any() for mask in cohorts):
                    selected, replaced = proposal, True
                    break
        if not replaced:
            raise ValueError('balanced tiny panel cannot cover all positive/failure cohorts')
    if len({int(arrays['seeds'][r]) for r in selected}) != count or not all(mask[selected].any() for mask in cohorts):
        raise ValueError('tiny panel diversity/coverage failed')
    return np.asarray(selected, dtype=np.int64)


def evaluate(model, arrays, encoder, weights, size=256):
    totals, supports = Counter(), Counter()
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(arrays['seeds']), size):
            guard()
            record = forward(model, batch(arrays, np.arange(start, min(start + size, len(arrays['seeds']))), 'cuda'),
                             encoder, weights, 'float32')
            for name, value in record['diagnostics'].items():
                if value is None:
                    continue
                support = float(record['diagnostic_weights'][name])
                totals[name] += float(value) if name.endswith(('_count', '_support')) else float(value) * support
                supports[name] += support
    return {name: totals[name] if name.endswith(('_count', '_support')) else totals[name] / supports[name]
            for name in totals if supports[name]}


def route_gradient_norm(model):
    gradients = [p.grad for name, p in model.named_parameters()
                 if name.startswith('route_readout.') and not name.startswith('route_readout.output_projection.')]
    if not gradients:
        return None
    if any(value is None or not bool(torch.isfinite(value).all()) for value in gradients):
        raise ValueError('missing or nonfinite route interior gradient')
    return float(torch.stack([g.float().square().sum() for g in gradients]).sum().sqrt())


def reachable_value_accuracy(model, items, encoder):
    with torch.no_grad():
        result = model(items['raw'], items['state'], items['glyph'], player_probabilities(items, encoder))
        mask = items['distances'] >= 0
        target = value_targets(result['value_logits'], items['distances'])
        return dict(support=int(mask.sum()), accuracy=float((result['value_logits'].argmax(-1)[mask] == target[mask]).float().mean()))


def qualification(parent, arrays, tiers, encoder, weights, args, report):
    sampler = RouteReplaySampler(arrays['train'], None, tiers, tier_counts=args.tier_counts, recent_fraction=0.)
    selected = sampler.sample(np.random.default_rng(args.seed))
    items = matched_items(arrays['train'], None, selected, 'cuda')
    tiny_rows = tiny_selection(arrays['train'], tiers, seed=args.seed)
    tiny = batch(arrays['train'], tiny_rows, 'cuda')
    report['tiny_panel'] = dict(rows=tiny_rows.tolist(), seeds=arrays['train']['seeds'][tiny_rows].tolist(),
                              tiers=dict(Counter(tiers[int(s)] for s in arrays['train']['seeds'][tiny_rows])),
                              zero_policy_roots=int((arrays['train']['optimal'][tiny_rows] == 0).sum()))
    report['tiny_criteria'] = dict(supported_policy_accuracy=1., minimum_each_physical_accuracy=.9,
        minimum_all_value_accuracy=.85, minimum_reachable_value_accuracy=.9, maximum_weighted_event_bce=.2,
        minimum_supported_event_recall=.9, consecutive_steps=10)
    reference = None
    for arm in ARMS:
        gpu_available(); guard(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model = build_model(parent, arm, seed=args.seed)
        with torch.no_grad():
            prediction = model(items['raw'][:64], items['state'][:64], items['glyph'][:64],
                               player_probabilities({k: v[:64] for k, v in items.items()}, encoder))
        if reference is None:
            reference = {key: value if key != 'field_logits' else tuple(t.clone() for t in value)
                         for key, value in prediction.items()}
        else:
            for key in ('action_logits', 'value_logits', 'event_logits'):
                torch.testing.assert_close(prediction[key], reference[key], atol=0, rtol=0)
            for left, right in zip(prediction['field_logits'], reference['field_logits']):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
        entry = dict(initial_weights_sha256=weights_sha256(model.state_dict()), parameters=model.parameter_count(),
                     original_parameter_count=sum(p.numel() for n, p in model.named_parameters() if not n.startswith('route_readout.')),
                     qualification_updates=[])
        report['arms'][arm] = entry
        optimizer = optimizer_for(model, args.lr)
        for step in range(2):
            torch.cuda.synchronize(); started = time.monotonic()
            record, norm = fit_step(model.train(), optimizer, items, encoder, weights, args.precision)
            torch.cuda.synchronize()
            route_norm = route_gradient_norm(model)
            if arm == 'route' and step == 1 and not route_norm > 0:
                raise ValueError('route interior remains inactive after two real updates')
            entry['qualification_updates'].append(dict(step=step + 1, elapsed_seconds=time.monotonic() - started,
                loss=float(record['total']), gradient_norm=float(norm), route_interior_gradient_norm=route_norm,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(), available_host_bytes=guard()))
        del model, optimizer, record
        gc.collect(); torch.cuda.empty_cache()
        model = build_model(parent, arm, seed=args.seed)
        optimizer = optimizer_for(model, args.lr)
        consecutive, started = 0, time.monotonic()
        for step in range(args.tiny_steps):
            guard()
            fit_step(model.train(), optimizer, tiny, encoder, weights, args.precision)
            with torch.no_grad():
                result = forward(model.eval(), tiny, encoder, weights, 'float32')
            finite = reachable_value_accuracy(model, tiny, encoder)
            consecutive = consecutive + 1 if tiny_gate(result) and finite['accuracy'] >= .9 else 0
            if (step + 1) % 100 == 0:
                print(json.dumps(dict(event='tiny', arm=arm, step=step + 1, metrics=scalar_metrics(result))), flush=True)
            if consecutive >= 10:
                break
        entry['tiny_fit'] = dict(steps=step + 1, seconds=time.monotonic() - started, consecutive_gate_steps=consecutive,
                                passed=consecutive >= 10, metrics=scalar_metrics(result), reachable_values=finite)
        write(args.out / 'report.json', report)
        if consecutive < 10:
            raise RuntimeError(f'{arm} balanced tiny-fit gate failed; full training not admitted')
        del model, optimizer, result
        gc.collect(); torch.cuda.empty_cache()
    report.update(qualification_passed=True, initial_route_outputs_equal_parent=True,
                  qualification_updates_discarded=True, selected_batch_size=sampler.batch_size)


def checkpoint(parent, model, arm, args, report):
    result = checkpoint_from_parent(parent, model, parent_checkpoint_sha256=PARENT_SHA) if arm == 'route' else dict(parent)
    stale = ('quality_filter', 'quality_manifest', 'quality_manifest_sha256', 'recovery_fraction', 'policy_row_fraction',
             'continuation_arm', 'continuation_parent_sha256', 'supplemental_cache', 'supplemental_manifest_sha256',
             'planner_initialization', 'optimizer_steps', 'learning_rate', 'objective_weights',
             'sampling', 'precision', 'seed', 'quality_row_file_sha256')
    history = {key: result.pop(key) for key in stale if key in result}
    result.update(planner_config=model.config(), planner_weights={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                  planner_parameters=model.parameter_count(), parent_training_metadata=history,
                  source_checkpoint_sha256=PARENT_SHA, source_sha256=report['source_sha256'],
                  optimizer='AdamW', optimizer_state='fresh moments', optimizer_steps=args.steps, learning_rate=args.lr,
                  weight_decay=.05, batch_size=sum(args.tier_counts), train_levels=10000,
                  sampling=report['sampling'], precision=args.precision, seed=args.seed,
                  learning_rate_schedule='3% linear warmup then cosine decay',
                  quality_row_file_sha256=report['quality_row_file_sha256'],
                  training_cache=str(args.cache), supplemental_cache=str(args.recent),
                  cache_manifest_sha256=report['base_manifest_sha256'], supplemental_manifest_sha256=report['recent_manifest_sha256'],
                  objective_weights=report['objective_weights'], actual_outcome_comparator_auxiliary_training=True,
                  route_repair=dict(arm=arm, seed=args.seed, all_planner_parameters_trained=True,
                      sampling=report['sampling'], full_original_failure_supervision=True,
                      encoder_frozen=True, gameplay_gain_established=False, persistent_game_memory=False,
                      learned_voluntary_reset=False, qualification_report_sha256=report['qualification_sha256']))
    return result


def train(parent, arrays, recent, approved, tiers, encoder, weights, args, report):
    proof = json.loads(args.qualification.read_text())
    if (proof.get('status') != 'complete' or not proof.get('qualification_passed')
            or proof['learning_rate'] != args.lr or proof['precision'] != args.precision
            or proof['tier_counts'] != args.tier_counts or proof['parent_sha256'] != PARENT_SHA
            or proof['base_manifest_sha256'] != report['base_manifest_sha256']
            or proof['objective_weights'] != weights):
        raise ValueError('training differs from the completed qualification')
    for name, digest in proof['source_sha256'].items():
        if sha(name) != digest:
            raise ValueError(f'qualification dependency changed: {name}')
    sampler = RouteReplaySampler(arrays['train'], recent, tiers, tier_counts=args.tier_counts,
                                 recent_fraction=args.recent_fraction, base_rows=approved['base_rows'],
                                 recent_rows=approved['supplement_rows'])
    report.update(sampling=sampler.config(), qualification_sha256=sha(args.qualification))
    for arm in ARMS:
        guard(); gpu_available(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model = build_model(parent, arm, seed=args.seed).train()
        initial = weights_sha256(model.state_dict())
        if initial != proof['arms'][arm]['initial_weights_sha256']:
            raise ValueError('initial weights differ from qualified arm')
        optimizer = optimizer_for(model, args.lr)
        rng, sampling_hash = np.random.default_rng(args.seed), hashlib.sha256()
        entry = dict(status='running', initial_weights_sha256=initial, parameters=model.parameter_count(), updates=[])
        report['arms'][arm] = entry
        zero_rows, replay_kinds, tier_counts = 0, Counter(), Counter()
        started = time.monotonic()
        for step in range(args.steps):
            available = guard()
            selection = sampler.sample(rng)
            for key in sorted(selection):
                sampling_hash.update(selection[key].tobytes())
            items = matched_items(arrays['train'], recent, selection, 'cuda')
            for group in optimizer.param_groups:
                group['lr'] = args.lr * schedule(step, args.steps)
            record, norm = fit_step(model.train(), optimizer, items, encoder, weights, args.precision)
            zero_rows += int((items['optimal'] == 0).sum())
            replay_kinds.update(map(int, selection['row_kinds']))
            tier_counts.update(tiers[int(seed)] for seed in selection['seeds'])
            if step == 0 or (step + 1) % 25 == 0 or step + 1 == args.steps:
                torch.cuda.synchronize()
                update = dict(step=step + 1, elapsed_seconds=time.monotonic() - started,
                    loss=float(record['total']), metrics=scalar_metrics(record), gradient_norm=float(norm),
                    peak_gpu_bytes=torch.cuda.max_memory_allocated(), available_host_bytes=available,
                    route_interior_gradient_norm=route_gradient_norm(model))
                entry['updates'].append(update); write(args.out / 'report.json', report)
                print(json.dumps(dict(event='training', arm=arm, **update)), flush=True)
        entry.update(status='complete', elapsed_seconds=time.monotonic() - started,
            final_weights_sha256=weights_sha256(model.state_dict()), sampling_sha256=sampling_hash.hexdigest(),
            zero_policy_roots_sampled=zero_rows, trajectory_kinds=dict(replay_kinds), tier_counts=dict(tier_counts),
            validation=evaluate(model, arrays['validation'], encoder, weights))
        if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
            raise ValueError('nonfinite final planner tensor')
        destination = args.out / (arm + '.pending.pt')
        torch.save(checkpoint(parent, model, arm, args, report), destination)
        entry['checkpoint_sha256'] = sha(destination)
        write(args.out / 'report.json', report)
        del model, optimizer, items, record
        gc.collect(); torch.cuda.empty_cache()
    if report['arms']['control']['sampling_sha256'] != report['arms']['route']['sampling_sha256']:
        raise ValueError('matched arms used different rows')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / 'data/reference-outcome-inputs-v1')
    parser.add_argument('--recent', type=Path)
    parser.add_argument('--qualification', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--qualify', action='store_true')
    parser.add_argument('--steps', type=int, default=1560)
    parser.add_argument('--tiny-steps', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=.0001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tier-counts', nargs=7, type=int, default=list(TIER_COUNTS))
    parser.add_argument('--recent-fraction', type=float, default=.25)
    parser.add_argument('--precision', choices=('float32', 'bf16'), default='float32')
    parser.add_argument('--max-seconds', type=int, default=2400)
    args = parser.parse_args(argv)
    if (min(args.steps, args.tiny_steps, args.max_seconds) < 1 or args.max_seconds > 3600
            or not math.isfinite(args.lr) or not 0 < args.lr < 1 or args.seed < 0
            or sum(args.tier_counts) not in (64, 128, 256, 512, 1024)):
        parser.error('bounded positive budgets, learning rate and power-of-two batch up to1024 required')
    if not args.qualify and (args.recent is None or args.qualification is None):
        parser.error('full training requires recent replay and a completed qualification')
    if args.out.exists():
        raise FileExistsError(args.out)
    report = dict(status='running', pid=os.getpid(), started_local=datetime.now().astimezone().isoformat(),
        mode='qualification' if args.qualify else 'training', learning_rate=args.lr, precision=args.precision,
        seed=args.seed, parent_sha256=PARENT_SHA, tier_counts=args.tier_counts, planned_steps=args.steps,
        max_seconds=args.max_seconds, arms={}, source_sha256={}, encoder_frozen=True,
        official_training_inputs=False, privileged_inference_inputs=False,
        limits=['All-cell route readout is learned scene aggregation, not multi-step planning.',
                'Encoder, persistent-memory capability and voluntary-reset capability are unchanged.',
                'Tiny fit only demonstrates trainability; native retention and new confirmation remain required.'])
    started, arrays, recent, approved, created = time.monotonic(), {}, None, {}, False
    old_handler = signal.getsignal(signal.SIGALRM)
    def timeout(*_):
        raise TimeoutError('bounded whole-spatial repair deadline exceeded')
    try:
        signal.signal(signal.SIGALRM, timeout); signal.alarm(args.max_seconds)
        guard(); gpu_available()
        args.out.mkdir(parents=True); created = True
        write(args.out / 'report.json', report)
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        bindings = Bindings(); bindings.add(PARENT, PARENT_SHA); bindings.add(BANK, BANK_SHA)
        for path in [Path(__file__), *sorted((ROOT / 'pebby/agent').glob('*.py')),
                     *[ROOT / 'tools' / name for name in ('train_reference_outcomes.py', 'train_reference_spatial_outcomes.py',
                         'train_spatial_recovery_comparison.py', 'cache_reference_outcome_inputs.py')]]:
            bindings.add(path)
        parent = torch.load(PARENT, map_location='cpu', weights_only=True); validate_checkpoint(parent)
        arrays, hashes, stats = load_data(args.cache)
        report.update(base_manifest_sha256=bindings.add(args.cache / 'manifest.json'))
        if report['base_manifest_sha256'] != parent['cache_manifest_sha256']:
            raise ValueError('base cache differs from parent lineage')
        tiers = {int(row['seed']): int(row['difficulty']) for row in map(json.loads, BANK.read_text().splitlines())}
        if len(tiers) != 10000 or np.isin(arrays['validation']['seeds'], list(tiers)).any():
            raise ValueError('original TRAIN mapping and validation must remain disjoint')
        if not args.qualify:
            from tools import cache_spatial_repair_v3 as recent_cache
            bindings.add(Path(recent_cache.__file__))
            for name in ('cache_spatial_recovery.py', 'collect_spatial_repair_v3.py'):
                bindings.add(ROOT / 'tools' / name)
            manifest = recent_cache.validate_published(args.recent)
            if manifest['source_checkpoint_sha256'] != PARENT_SHA:
                raise ValueError('recent collection must use the retained recovery policy')
            recent = {name: np.load(args.recent / (name + '.npy'), mmap_mode='r', allow_pickle=False)
                      for name in (*ARRAYS, 'row_kind', 'policy_valid', 'dynamics_valid')}
            hashes.update(manifest['validated_output_hashes']); stats.update(manifest['validated_output_stats'])
            report['recent_manifest_sha256'] = bindings.add(args.recent / 'manifest.json')
            for name in ('base_rows', 'supplement_rows'):
                info = manifest['quality']['files'][name]
                path = (args.recent / info['path']).resolve()
                if not path.is_relative_to(args.recent.resolve()):
                    raise ValueError('quality row path escapes cache')
                bindings.add(path, info['sha256'])
                approved[name] = np.load(path, allow_pickle=False)
            report['quality_row_file_sha256'] = {name: manifest['quality']['files'][name]['sha256'] for name in approved}
            report['approved_replay'] = dict(base_rows=len(approved['base_rows']),
                recent_rows=len(approved['supplement_rows']),
                base_zero_policy_rows=int((arrays['train']['optimal'][approved['base_rows']] == 0).sum()),
                recent_zero_policy_rows=int((recent['optimal'][approved['supplement_rows']] == 0).sum()))
            bindings.add(args.qualification)
        weights = training_weights(arrays['train'])
        report.update(objective_weights=weights, source_sha256={**hashes, **bindings.hashes},
                      train_roots=len(arrays['train']['seeds']), zero_policy_train_roots=int((arrays['train']['optimal'] == 0).sum()))
        encoder = {k: v.cuda() for k, v in parent['encoder_weights'].items() if k.startswith('player_head.')}
        if args.qualify:
            qualification(parent, arrays, tiers, encoder, weights, args, report)
        else:
            train(parent, arrays, recent, approved, tiers, encoder, weights, args, report)
        bindings.verify()
        if any(stat(path) != value for path, value in stats.items()):
            raise ValueError('bound feature arrays changed during experiment')
        if weights_sha256(parent['encoder_weights']) != parent['encoder_weights_sha256']:
            raise ValueError('frozen encoder weights changed')
        if not args.qualify:
            for arm in ARMS:
                (args.out / (arm + '.pending.pt')).replace(args.out / (arm + '.pt'))
        report.update(status='complete', sources_unchanged=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        if created:
            for arm in ARMS:
                (args.out / (arm + '.pending.pt')).unlink(missing_ok=True)
                (args.out / (arm + '.pt')).unlink(missing_ok=True)
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_handler)
        for group in [*arrays.values(), recent]:
            if group is not None:
                release(group, close=True)
        report.update(elapsed_seconds=time.monotonic() - started, finished_local=datetime.now().astimezone().isoformat())
        if created:
            write(args.out / 'report.json', report)
    return report


if __name__ == '__main__':
    main()
