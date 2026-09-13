"""Fixed four-arm public policy-objective pilot; no qualification or promotion.

Every arm starts from the same semantic actor. Joint arms train the full planner;
actor arms freeze every nonactor tensor. Only generated continuation TRAIN rows
enter optimizers; heldout continuation levels are excluded from both caches.
"""
import argparse
from collections import Counter
import copy
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch

from pebby.agent.neural_outcome_policy import weights_sha256
from pebby.agent.optimal_set_objective import optimal_action_loss
from pebby.agent.route_repair_sampling import RouteReplaySampler
from pebby.agent.spatial_outcome_objective import spatial_outcome_losses, training_weights
from pebby.agent.spatial_semantic_outcome_policy import (checkpoint_from_parent, load_checkpoint,
                                                        load_perceptor, TEACHER_SHA256)
from tools.cache_reference_outcome_inputs import Bindings, process_start_ticks, release, stat
from tools.train_reference_outcomes import ARRAYS, gpu_available, guard, load_data, optimizer_for, sha, write
from tools.train_reference_outcomes import batch
from tools.train_reference_spatial_outcomes import player_probabilities
from tools.train_spatial_recovery_comparison import schedule, validate_checkpoint
from tools.train_spatial_semantic_repair import (add_semantic, build_model, matched_items,
                                                ROOT, PARENT, PARENT_SHA, BANK, BANK_SHA, TEACHER)

ARMS = ('joint-uniform', 'joint-set', 'actor-uniform', 'actor-set')
STEPS = 400
TIER_COUNTS = (44, 44, 44, 44, 40, 20, 20)
LR = .0001
RECENT_FRACTION = .25
PROTOCOL = ROOT / 'artifacts/spatial-repair-v5/decision-protocol.json'


def configure_arm(model, arm):
    """Set the intended trainable scope without changing any initial tensor."""
    if arm not in ARMS or model.actor_readout is None:
        raise ValueError('a declared pilot arm and semantic actor model are required')
    joint = arm.startswith('joint-')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(joint or name.startswith('actor_readout.'))
    return model


def frozen_planner_digest(model):
    """All nonactor state, including buffers, protected in actor-only arms."""
    return weights_sha256({name: value for name, value in model.state_dict().items()
                           if not name.startswith('actor_readout.')})


def loss_record(model, items, encoder, objective_weights, arm):
    if arm not in ARMS:
        raise ValueError('unknown pilot arm')
    predicted = model(items['raw'], items['state'], items['glyph'],
                      player_probabilities(items, encoder), items['semantic'])
    native = optimal_action_loss(predicted['action_logits'], items['optimal'], arm.split('-')[1])
    if arm.startswith('joint-'):
        record = spatial_outcome_losses(model, predicted, items, objective_weights)
        losses = {**record['losses'], 'policy': native}
    else:
        losses = {'policy': native}
    return dict(total=sum(losses.values()), losses=losses)


def gradient_norm(model, prefix, *, exclude=None):
    gradients = [parameter.grad for name, parameter in model.named_parameters()
                 if parameter.requires_grad and name.startswith(prefix) and not (exclude and name.startswith(exclude))]
    if not gradients:
        return None
    if any(value is None or not bool(torch.isfinite(value).all()) for value in gradients):
        raise ValueError('missing or nonfinite gradient in ' + prefix)
    return float(torch.stack([value.detach().float().square().sum() for value in gradients]).sum().sqrt())


def fit_step(model, optimizer, items, encoder, objective_weights, arm):
    actor_before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                    if name.startswith('actor_readout.')}
    optimizer.zero_grad(set_to_none=True)
    record = loss_record(model, items, encoder, objective_weights, arm)
    if not bool(torch.isfinite(record['total'])):
        raise ValueError('nonfinite policy-probe loss')
    record['total'].backward()
    for parameter in model.parameters():
        if parameter.requires_grad:
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise ValueError('every trainable tensor must receive a finite gradient')
        elif parameter.grad is not None:
            raise ValueError('frozen parameter received a gradient')
    gradients = dict(actor_preclip=gradient_norm(model, 'actor_readout.'),
        actor_interior_preclip=gradient_norm(model, 'actor_readout.', exclude='actor_readout.output_projection.'),
        route_interior_preclip=gradient_norm(model, 'route_readout.', exclude='route_readout.output_projection.'),
        semantic_grid_preclip=gradient_norm(model, 'semantic_grid_projection.'))
    parameters = [p for p in model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    gradients.update(global_preclip=float(norm), actor_postclip=gradient_norm(model, 'actor_readout.'),
                     clip_scale=min(1., 1. / (float(norm) + 1e-6)))
    optimizer.step()
    current = dict(model.named_parameters())
    for label, prefix in (('actor', 'actor_readout.'), ('actor_output', 'actor_readout.output_projection.')):
        names = [name for name in actor_before if name.startswith(prefix)]
        previous = torch.stack([actor_before[name].float().square().sum() for name in names]).sum().sqrt()
        delta = torch.stack([(current[name].detach().float() - actor_before[name].float()).square().sum()
                             for name in names]).sum().sqrt()
        gradients[label + '_update_l2'] = float(delta)
        gradients[label + '_relative_update_l2'] = float(delta / previous) if float(previous) else None
    return record, gradients


def continuation_rows(base, recent, base_approved, recent_approved, heldout, level_tiers):
    """Exclude heldout level identities in both sources before constructing replay."""
    heldout = set(map(int, heldout))
    if not heldout or not heldout <= set(level_tiers):
        raise ValueError('heldout continuation levels must be a nonempty subset of original TRAIN')
    result = {}
    for name, arrays, allowed in (('base_rows', base, base_approved), ('recent_rows', recent, recent_approved)):
        allowed = np.asarray(allowed)
        if (allowed.ndim != 1 or not np.issubdtype(allowed.dtype, np.integer)
                or len(allowed) != len(np.unique(allowed)) or np.any((allowed < 0) | (allowed >= len(arrays['seeds'])))):
            raise ValueError('quality rows must be unique in-range integer indices')
        result[name] = allowed[~np.isin(arrays['seeds'][allowed], list(heldout))]
    tiers = {int(seed): tier for seed, tier in level_tiers.items() if seed not in heldout}
    if set(map(int, base['seeds'][result['base_rows']])) != set(tiers):
        raise ValueError('continuation base rows must cover every nonheldout TRAIN level')
    if not set(map(int, recent['seeds'][result['recent_rows']])) <= set(tiers):
        raise ValueError('recent continuation contains non-TRAIN levels')
    return result, tiers


def load_split(path, base, recent, quality, level_tiers, expected_sources, bindings):
    """Admit the frozen continuation split and its exact source identities."""
    split = json.loads(Path(path).read_text())
    if (split.get('format') != 'pebby.policy-probe-split.v1' or split.get('seed') != 20260913
            or split.get('source_sha256') != expected_sources):
        raise ValueError('split format, seed or source bindings differ from this pilot')
    for source, digest in expected_sources.items():
        bindings.add(source, digest)
    for name in ('fit_seeds', 'heldout_seeds', 'heldout_recent_rows', 'heldout_base_rows'):
        values = split.get(name)
        if (not isinstance(values, list) or not values or any(type(value) is not int or value < 0 for value in values)
                or values != sorted(set(values))):
            raise ValueError('split lists must be nonempty sorted distinct nonnegative integers: ' + name)
    fit, heldout = set(split['fit_seeds']), set(split['heldout_seeds'])
    recent_levels = set(map(int, recent['seeds']))
    if (fit & heldout or fit | heldout != recent_levels or not recent_levels <= set(level_tiers)
            or len(recent_levels) != 320):
        raise ValueError('split must partition all current320 TRAIN levels')
    for tier in range(1, 8):
        levels = {seed for seed in recent_levels if level_tiers[seed] == tier}
        if len(fit & levels) != round(.8 * len(levels)):
            raise ValueError('split differs from the declared tier-stratified 80/20 allocation')
    expected_base = np.flatnonzero(np.isin(base['seeds'], list(heldout)))
    if not np.array_equal(expected_base, split['heldout_base_rows']):
        raise ValueError('split omitted or altered original rows of heldout continuation levels')
    rows = np.asarray(split['heldout_recent_rows'], dtype=np.int64)
    approved = np.asarray(quality['supplement_rows'])
    if (np.any(rows >= len(recent['seeds'])) or not np.isin(rows, approved).all()
            or set(map(int, recent['seeds'][rows])) != heldout or len(rows) > 2640):
        raise ValueError('heldout recent panel is not approved, bounded and level-complete')
    for seed in heldout:
        available = int((recent['seeds'][approved] == seed).sum())
        if int((recent['seeds'][rows] == seed).sum()) != min(40, available):
            raise ValueError('heldout recent panel must retain up to40 approved rows per level')
    return {**split, 'heldout_rows': rows}


INFERENCE_KEYS = frozenset(('format', 'encoder_parent_sha256', 'encoder_config', 'encoder_weights',
    'encoder_weights_sha256', 'encoder_runtime', 'encoder_frozen', 'official_training_inputs',
    'privileged_inference_inputs', 'score_weights', 'planner_config', 'planner_weights', 'planner_parameters',
    'source_checkpoint_sha256', 'perceptor_format', 'perceptor_config', 'perceptor_weights',
    'perceptor_weights_sha256', 'perceptor_checkpoint_sha256', 'perceptor_frozen', 'perceptor_parameters',
    'semantic_architecture'))


def checkpoint(parent, model, teacher, arm, args, report):
    """Build a current-stage envelope instead of inheriting stale qualification claims."""
    envelope = checkpoint_from_parent(parent, model, teacher, parent_checkpoint_sha256=PARENT_SHA)
    result = {key: value for key, value in envelope.items() if key in INFERENCE_KEYS}
    history = {key: copy.deepcopy(value) for key, value in parent.items()
               if key not in INFERENCE_KEYS and key not in ('weights', 'parent_training_metadata')}
    if 'parent_training_metadata' in parent:
        history['parent_training_metadata'] = copy.deepcopy(parent['parent_training_metadata'])
    result.update(parent_training_metadata=history, optimizer='AdamW', optimizer_state='fresh moments',
        optimizer_steps=STEPS, learning_rate=LR, weight_decay=.05, batch_size=sum(TIER_COUNTS),
        precision='float32', seed=args.seed, learning_rate_schedule='3% linear warmup then cosine decay',
        train_levels=report['continuation']['train_levels'], sampling=report['sampling'],
        source_sha256=report['source_sha256'], cache_manifest_sha256=report['base_manifest_sha256'],
        supplemental_manifest_sha256=report['recent_manifest_sha256'],
        semantic_manifest_sha256=report['semantic_manifest_sha256'],
        training_cache=str(args.cache), supplemental_cache=str(args.recent), semantic_cache=str(args.semantics),
        actual_outcome_comparator_auxiliary_training=arm.startswith('joint-'),
        objective_weights=report['objective_weights'] if arm.startswith('joint-') else {'native_policy': 1.},
        native_policy_objective=arm.split('-')[1], planner_horizon=1, planner_refinement_loops=1,
        persistent_game_memory=False, learned_voluntary_reset=False,
        policy_probe=dict(version='v5-policy-factorial-pilot', arm=arm, final_step=STEPS,
            actor_only=arm.startswith('actor-'), qualification_required=False,
            old_v4_qualification_admission=False, promotion=False,
            protocol_sha256=report['protocol_sha256'], split_sha256=report['split_sha256'],
            heldout_recent_levels=report['continuation']['heldout_levels'],
            all_approved_fit_failure_rows_eligible=arm.startswith('joint-')))
    return result


def validate_protocol(protocol):
    expected = dict(steps=STEPS, batch_size=sum(TIER_COUNTS), tier_counts=list(TIER_COUNTS),
        recent_fraction=RECENT_FRACTION, learning_rate=LR, schedule='3% warmup then cosine',
        optimizer='AdamW', weight_decay=.05, precision='float32', gradient_clip_global=1.0,
        primary_seed=42, replication_seed=43, encoder_frozen=True, perceptor_frozen=True)
    if (protocol.get('version') != 'v5-policy-factorial-pilot' or protocol.get('arms') != list(ARMS)
            or protocol.get('baseline_checkpoint_sha256') != PARENT_SHA
            or any(protocol.get('training', {}).get(key) != value for key, value in expected.items())
            or protocol.get('comparison', {}).get('validation_every') != 100
            or protocol.get('comparison', {}).get('primary_checkpoint') != 'final400 for eacharm, no best-step rescue'
            or protocol.get('comparison', {}).get('ablation_admission') is not False
            or protocol.get('decision', {}).get('no_promotion_from_pilot_alone') is not True):
        raise ValueError('frozen v5 protocol differs from the implemented pilot')


def evaluate(model, arrays, recent, heldout_rows, tiers, encoder):
    from tools.spatial_policy_probe_metrics import evaluate_panel
    return dict(heldout_recent=evaluate_panel(model, recent, heldout_rows, tiers, encoder),
                existing_validation=evaluate_panel(model, arrays['validation'],
                    np.arange(len(arrays['validation']['seeds'])), tiers, encoder))


def verify_baseline(model, parent, recent, rows, encoder):
    from pebby.agent.spatial_outcome_planner import SpatialOutcomePlanner
    original = SpatialOutcomePlanner(parent['planner_config']).cuda().eval()
    original.load_state_dict(parent['planner_weights'], strict=True)
    items = batch(recent, rows[:32], 'cuda')
    public = (items['raw'], items['state'], items['glyph'], player_probabilities(items, encoder))
    with torch.no_grad():
        expected = original(*public)
        actual = model(*public, items['semantic'])
    for key in ('action_logits', 'value_logits', 'event_logits'):
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    for left, right in zip(actual['field_logits'], expected['field_logits']):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    return dict(rows=len(items['optimal']), bitwise_parent_outputs=True,
                reason='Zero-initialized semantic, route and actor residuals preserve retained parent scores.')


def train(parent, teacher, arrays, recent, allowed, tiers, heldout_rows, encoder, weights, args, report):
    sampler = RouteReplaySampler(arrays['train'], recent, tiers, tier_counts=TIER_COUNTS,
        recent_fraction=RECENT_FRACTION, base_rows=allowed['base_rows'], recent_rows=allowed['recent_rows'])
    report['sampling'] = sampler.config()
    reference_initial, reference_sampling = None, None
    for arm in args.arms:
        guard(); gpu_available(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model = configure_arm(build_model(parent, 'actor', seed=args.seed), arm).train()
        initial = weights_sha256(model.state_dict())
        if reference_initial is not None and initial != reference_initial:
            raise ValueError('pilot arms differ in their initial weights')
        reference_initial = initial
        frozen = frozen_planner_digest(model)
        if 'baseline' not in report:
            report['baseline_parity'] = verify_baseline(model, parent, recent, heldout_rows, encoder)
            report['baseline'] = evaluate(model, arrays, recent, heldout_rows, report['evaluation_tiers'], encoder)
            write(args.out / 'report.json', report)
        entry = dict(status='running', initial_weights_sha256=initial, updates=[], evaluations={},
            parameters=model.parameter_count(), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            initial_nonactor_weights_sha256=frozen, trained_parameter_scope='all planner' if arm.startswith('joint-') else 'actor_readout only')
        report['arms'][arm] = entry
        optimizer = optimizer_for(model, LR)
        rng, sample_hash = np.random.default_rng(args.seed), hashlib.sha256()
        zero_roots, row_kinds, sampled_tiers, clipped_updates = 0, Counter(), Counter(), 0
        started = time.monotonic()
        for step in range(STEPS):
            available = guard()
            selection = sampler.sample(rng)
            if any(int(seed) not in tiers for seed in selection['seeds']):
                raise ValueError('sample contains a heldout or non-TRAIN level')
            for key in sorted(selection):
                sample_hash.update(key.encode()); sample_hash.update(selection[key].tobytes())
            items = matched_items(arrays['train'], recent, selection, 'cuda')
            for group in optimizer.param_groups:
                group['lr'] = LR * schedule(step, STEPS)
            record, gradients = fit_step(model, optimizer, items, encoder, weights, arm)
            clipped_updates += gradients['clip_scale'] < 1.
            if step == 1:
                required = ['actor_interior_preclip']
                if arm.startswith('joint-'):
                    required += ['route_interior_preclip', 'semantic_grid_preclip']
                if any(gradients[name] is None or gradients[name] <= 0 for name in required):
                    raise ValueError('intended trainable path has no gradient after two actual updates')
            zero_roots += int((items['optimal'] == 0).sum())
            row_kinds.update(map(int, selection['row_kinds']))
            sampled_tiers.update(tiers[int(seed)] for seed in selection['seeds'])
            if step < 2 or (step + 1) % 25 == 0:
                torch.cuda.synchronize()
                elapsed = time.monotonic() - started
                update = dict(step=step + 1, elapsed_seconds=elapsed,
                    projected_arm_seconds_from_elapsed_progress=elapsed / (step + 1) * STEPS,
                    losses={name: float(value.detach()) for name, value in record['losses'].items()},
                    total_loss=float(record['total'].detach()), gradients=gradients,
                    available_host_bytes=available, peak_gpu_bytes=torch.cuda.max_memory_allocated())
                entry['updates'].append(update)
                write(args.out / 'report.json', report)
                print(json.dumps(dict(event='training', arm=arm, **update)), flush=True)
            if (step + 1) % 100 == 0:
                entry['evaluations'][str(step + 1)] = evaluate(model, arrays, recent, heldout_rows, report['evaluation_tiers'], encoder)
                if arm.startswith('actor-') and frozen_planner_digest(model) != frozen:
                    raise ValueError('actor-only arm changed a frozen nonactor tensor')
                write(args.out / 'report.json', report)
        final_sampling = sample_hash.hexdigest()
        if reference_sampling is not None and final_sampling != reference_sampling:
            raise ValueError('pilot arms used different sampled row sequences')
        reference_sampling = final_sampling
        if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
            raise ValueError('nonfinite final model tensor')
        if arm.startswith('actor-') and frozen_planner_digest(model) != frozen:
            raise ValueError('actor-only final nonactor weights changed')
        destination = args.out / (arm + '.pending.pt')
        torch.save(checkpoint(parent, model, teacher, arm, args, report), destination)
        loaded, metadata = load_checkpoint(destination, 'cpu')
        if weights_sha256(loaded.planner.state_dict()) != weights_sha256(model.state_dict()):
            raise ValueError('saved planner does not roundtrip through strict semantic loader')
        if metadata.get('optimizer_steps') != STEPS or metadata.get('native_policy_objective') != arm.split('-')[1]:
            raise ValueError('checkpoint latest-stage metadata differs')
        entry.update(status='complete', final_weights_sha256=weights_sha256(model.state_dict()),
            final_nonactor_weights_sha256=frozen_planner_digest(model), frozen_nonactor_unchanged=(frozen_planner_digest(model) == frozen),
            sampling_sha256=final_sampling, zero_policy_roots_sampled=zero_roots,
            trajectory_kinds=dict(row_kinds), sampled_tier_counts=dict(sampled_tiers),
            final_step=STEPS, checkpoint_sha256=sha(destination), elapsed_seconds=time.monotonic() - started,
            strict_checkpoint_roundtrip=True, clipped_updates=int(clipped_updates), clipped_update_fraction=clipped_updates / STEPS)
        write(args.out / 'report.json', report)
        del model, optimizer, items, record, loaded, metadata
        gc.collect(); torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--split', type=Path, default=ROOT / 'artifacts/spatial-repair-v5/pilot-split.json')
    parser.add_argument('--seed', type=int, choices=(42, 43), default=42)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS),
                        help='A proper subset is labelled a diagnostic, not the complete factorial pilot.')
    parser.add_argument('--cache', type=Path, default=ROOT / 'data/reference-outcome-inputs-v1')
    parser.add_argument('--recent', type=Path, default=ROOT / 'data/spatial-repair-v3-current320-inputs')
    parser.add_argument('--semantics', type=Path, default=ROOT / 'data/spatial-repair-v4-semantics')
    parser.add_argument('--max-seconds', type=int, default=1800)
    args = parser.parse_args(argv)
    if not 1 <= args.max_seconds <= 1800 or len(args.arms) != len(set(args.arms)):
        parser.error('deadline must be 1..1800 seconds and arms must be distinct')
    if args.out.exists():
        raise FileExistsError(args.out)
    report = dict(status='running', format='pebby.spatial-policy-probe.v1', pid=os.getpid(),
        process_start_ticks=process_start_ticks(), started_local=datetime.now().astimezone().isoformat(),
        selected_arms=args.arms, diagnostic_subset=set(args.arms) != set(ARMS), seed=args.seed,
        planned_steps=STEPS, batch_size=sum(TIER_COUNTS), tier_counts=list(TIER_COUNTS),
        learning_rate=LR, weight_decay=.05, schedule='3% warmup then cosine', precision='float32',
        max_seconds=args.max_seconds, host_reserve_bytes=6 * 2**30, encoder_frozen=True, perceptor_frozen=True,
        official_training_inputs=False, privileged_inference_inputs=False, promotion=False,
        old_v4_qualification_required=False, checkpoint_publication='pending verification', arms={},
        limits=['This is a fixed 400-update diagnostic, not convergence or a promotion decision.',
                'Heldout continuation levels were already seen in parent pretraining.',
                'Current320 is one retained-policy replay round, not iterative on-policy collection.',
                'Joint-vs-actor changes auxiliary sharing and clipping; gradient norms alone do not identify conflict.',
                'H8 only; no persistent memory, voluntary-reset policy, or multistep rollout.'])
    started, arrays, recent, created = time.monotonic(), {}, None, False
    old_handler = signal.getsignal(signal.SIGALRM)
    def timeout(*_):
        raise TimeoutError('bounded policy-probe deadline exceeded')
    try:
        signal.signal(signal.SIGALRM, timeout); signal.alarm(args.max_seconds)
        guard(); gpu_available()
        args.out.mkdir(parents=True); created = True
        write(args.out / 'report.json', report)
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        bindings = Bindings()
        bindings.add(PARENT, PARENT_SHA); bindings.add(BANK, BANK_SHA); bindings.add(TEACHER, TEACHER_SHA256)
        report.update(protocol_sha256=bindings.add(PROTOCOL), split_sha256=bindings.add(args.split))
        validate_protocol(json.loads(PROTOCOL.read_text()))
        from tools import cache_spatial_repair_v3 as recent_cache, cache_spatial_semantics as semantic_cache
        from tools import spatial_policy_probe_metrics
        sources = [Path(__file__), Path(spatial_policy_probe_metrics.__file__),
            *sorted((ROOT / 'pebby/agent').glob('*.py')),
            *[ROOT / 'tools' / name for name in ('train_reference_outcomes.py', 'train_reference_spatial_outcomes.py',
                'train_spatial_semantic_repair.py', 'train_spatial_recovery_comparison.py',
                'cache_reference_outcome_inputs.py', 'cache_spatial_repair_v3.py', 'cache_spatial_recovery.py',
                'collect_spatial_repair_v3.py', 'cache_spatial_semantics.py', 'audit_spatial_semantic_decisions.py',
                'audit_spatial_route_resources.py')]]
        for path in sources:
            bindings.add(path)
        parent = torch.load(PARENT, map_location='cpu', weights_only=True)
        validate_checkpoint(parent)
        teacher = load_perceptor(TEACHER)
        teacher_digest = weights_sha256(teacher.state_dict())
        arrays, hashes, stats = load_data(args.cache)
        report['base_manifest_sha256'] = bindings.add(args.cache / 'manifest.json')
        if report['base_manifest_sha256'] != parent['cache_manifest_sha256']:
            raise ValueError('original cache differs from retained parent')
        manifest = recent_cache.validate_published(args.recent)
        if manifest['source_checkpoint_sha256'] != PARENT_SHA:
            raise ValueError('recent replay does not belong to retained parent')
        hashes.update(manifest['validated_output_hashes']); stats.update(manifest['validated_output_stats'])
        report['recent_manifest_sha256'] = bindings.add(args.recent / 'manifest.json')
        recent = {name: np.load(args.recent / (name + '.npy'), mmap_mode='r', allow_pickle=False)
                  for name in (*ARRAYS, 'row_kind', 'policy_valid', 'dynamics_valid')}
        quality = {}
        for name in ('base_rows', 'supplement_rows'):
            info = manifest['quality']['files'][name]
            path = (args.recent / info['path']).resolve()
            if not path.is_relative_to(args.recent.resolve()):
                raise ValueError('quality path escapes recent cache')
            bindings.add(path, info['sha256'])
            quality[name] = np.load(path, allow_pickle=False)
        report['quality_row_file_sha256'] = {name: manifest['quality']['files'][name]['sha256'] for name in quality}
        semantic = semantic_cache.validate_published(args.semantics)
        if (semantic['teacher_sha256'] != TEACHER_SHA256
                or semantic['base_manifest_sha256'] != report['base_manifest_sha256']
                or semantic['recent_manifest_sha256'] != report['recent_manifest_sha256']):
            raise ValueError('semantic addon differs from exact outcome caches')
        report['semantic_manifest_sha256'] = bindings.add(args.semantics / 'manifest.json')
        hashes.update(semantic['validated_output_hashes']); stats.update(semantic['validated_output_stats'])
        for split, group in [*arrays.items(), ('recent', recent)]:
            add_semantic(group, args.semantics / split)
        tiers = {int(row['seed']): int(row['difficulty']) for row in map(json.loads, BANK.read_text().splitlines())}
        validation_bank = BANK.parent / 'validation.jsonl'
        report['validation_bank_sha256'] = bindings.add(validation_bank)
        validation_tiers = {int(row['seed']): int(row['difficulty'])
                            for row in map(json.loads, validation_bank.read_text().splitlines())}
        if (len(tiers) != 10000 or len(validation_tiers) != 500 or set(tiers) & set(validation_tiers)
                or set(map(int, arrays['validation']['seeds'])) != set(validation_tiers)):
            raise ValueError('TRAIN and existing validation tier mapping differs')
        expected_sources = {str(Path(path).resolve()): digest for path, digest in (
            (args.recent / 'manifest.json', report['recent_manifest_sha256']),
            (args.recent / 'seeds.npy', hashes[str((args.recent / 'seeds.npy').resolve())]),
            (args.recent / manifest['quality']['files']['supplement_rows']['path'], report['quality_row_file_sha256']['supplement_rows']),
            (args.recent / manifest['quality']['files']['base_rows']['path'], report['quality_row_file_sha256']['base_rows']),
            (args.cache / 'manifest.json', report['base_manifest_sha256']),
            (args.cache / 'train/seeds.npy', hashes[str((args.cache / 'train/seeds.npy').resolve())]),
            (args.semantics / 'manifest.json', report['semantic_manifest_sha256']), (BANK, BANK_SHA))}
        split = load_split(args.split, arrays['train'], recent, quality, tiers, expected_sources, bindings)
        allowed, fit_tiers = continuation_rows(arrays['train'], recent, quality['base_rows'], quality['supplement_rows'],
                                               split['heldout_seeds'], tiers)
        report.update(continuation=dict(train_levels=len(fit_tiers), heldout_levels=len(split['heldout_seeds']),
            fit_recent_levels=len(set(map(int, recent['seeds'][allowed['recent_rows']]))),
            base_rows=len(allowed['base_rows']), recent_rows=len(allowed['recent_rows']),
            heldout_recent_rows=len(split['heldout_rows']), heldout_level_seeds=split['heldout_seeds']),
            evaluation_tiers={**tiers, **validation_tiers})
        # Do not use heldout continuation labels even to estimate class weights.
        balancing = {name: arrays['train'][name][allowed['base_rows']]
                     for name in ('next_triple', 'current_triple', 'lost_life', 'terminal', 'won')}
        weights = training_weights(balancing)
        del balancing
        report['objective_weights'] = weights
        bindings.hashes.update(hashes); bindings.before.update(stats)
        # Bind actually imported local helpers too, including transitive metric,
        # optimizer and cache-admission implementations.
        for module in list(sys.modules.values()):
            filename = getattr(module, '__file__', None)
            if filename:
                path = Path(filename).resolve()
                if path.suffix == '.py' and path.is_relative_to(ROOT) and path.is_file():
                    bindings.add(path)
        report['source_sha256'] = dict(bindings.hashes)
        report['resources'] = dict(cuda_device=torch.cuda.get_device_name(),
            cuda_total_bytes=torch.cuda.get_device_properties(0).total_memory,
            optimizer_updates=STEPS * len(args.arms), sampled_roots=STEPS * sum(TIER_COUNTS) * len(args.arms),
            evaluation_roots_each_checkpoint=4000 + len(split['heldout_rows']))
        report['runtime'] = dict(torch_version=str(torch.__version__), numpy_version=np.__version__,
                                 cuda_version=torch.version.cuda, device='cuda', host_threads=1,
                                 matmul_tf32=False, cudnn_tf32=True)
        encoder = {name: value.cuda() for name, value in parent['encoder_weights'].items() if name.startswith('player_head.')}
        write(args.out / 'report.json', report)
        train(parent, teacher, arrays, recent, allowed, fit_tiers, split['heldout_rows'], encoder, weights, args, report)
        bindings.verify()
        if any(stat(path) != value for path, value in stats.items()):
            raise ValueError('feature array changed during pilot')
        if (sha(PARENT) != PARENT_SHA or sha(TEACHER) != TEACHER_SHA256
                or weights_sha256(parent['encoder_weights']) != parent['encoder_weights_sha256']
                or weights_sha256(teacher.state_dict()) != teacher_digest):
            raise ValueError('protected parent/encoder/perceptor changed')
        for arm in args.arms:
            pending = args.out / (arm + '.pending.pt')
            if sha(pending) != report['arms'][arm]['checkpoint_sha256']:
                raise ValueError('pending checkpoint bytes changed')
        for arm in args.arms:
            (args.out / (arm + '.pending.pt')).replace(args.out / (arm + '.pt'))
        report.update(status='complete', sources_unchanged=True, encoder_weights_unchanged=True,
                      teacher_weights_unchanged=True, checkpoint_publication='published',
                      matched_initialization=True, matched_sampling=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', checkpoint_publication='not published')
        if created:
            for arm in args.arms:
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
