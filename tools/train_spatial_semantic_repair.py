"""Learned public scene features, paired with a direct scene-attention actor.

The encoder stays frozen. Both arms train every planner tensor on the same
verified examples and loss families. Undefined policy labels only mask policy
supervision. Qualification is disposable and cannot publish a candidate.
"""
import argparse
import copy
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
from pebby.agent.spatial_semantic_outcome_planner import SpatialSemanticOutcomePlanner
from pebby.agent.spatial_semantic_outcome_policy import checkpoint_from_parent, load_perceptor, TEACHER_SHA256
from pebby.agent.spatial_outcome_objective import training_weights, spatial_outcome_losses
from pebby.agent.spatial_value_objective import value_targets
from pebby.agent.neural_outcome_policy import weights_sha256
from tools.cache_reference_outcome_inputs import Bindings, release, stat
from tools.train_reference_outcomes import ARRAYS, batch, gpu_available, guard, load_data, optimizer_for, scalar_metrics, sha, tiny_gate, write
from tools.train_reference_spatial_outcomes import player_probabilities
from tools.train_spatial_recovery_comparison import schedule, validate_checkpoint

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'artifacts/spatial-recovery-v1/quality-fit/recovery.pt'
PARENT_SHA = 'ff88327214b6dc2d4278167e0d61edcc37c683292b788be6927b71286331a5f8'
BANK = ROOT / 'data/ls20-reference-unequal-v1/train.jsonl'
BANK_SHA = 'f968f9b4a3690be69041eadd1afe129a3ba0527d55829d75687aa5c260532a81'
ARMS = ('control', 'actor')
VALIDATION_BANK = ROOT / 'data/ls20-reference-unequal-v1/validation.jsonl'
TEACHER = ROOT / 'artifacts/reference-scene-initial-probe-v1/pixel-control/cell-appearance-initial-candidate.pt'



def add_semantic(arrays, path):
    for name in ('rows', 'seeds'):
        if not np.array_equal(arrays[name], np.load(path / (name + '.npy'), allow_pickle=False)):
            raise ValueError('semantic row alignment differs: ' + name)
    values = np.load(path / 'semantic.npy', mmap_mode='r', allow_pickle=False)
    if values.shape != (len(arrays['seeds']), 144, 22) or values.dtype != np.float32:
        raise ValueError('semantic array shape/dtype differs')
    arrays['semantic'] = values


def forward(model, items, player_weights, objective_weights, precision):
    player = player_probabilities(items, player_weights)
    with torch.autocast(items['raw'].device.type, dtype=torch.bfloat16, enabled=precision == 'bf16'):
        prediction = model(items['raw'], items['state'], items['glyph'], player, items['semantic'])
        return spatial_outcome_losses(model, prediction, items, objective_weights)


def fit_step(model, optimizer, items, player_weights, objective_weights, precision):
    optimizer.zero_grad(set_to_none=True)
    record = forward(model, items, player_weights, objective_weights, precision)
    if not bool(torch.isfinite(record['total'])):
        raise ValueError('nonfinite training loss')
    record['total'].backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise ValueError('all trainable tensors must have finite gradients')
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return record, norm


def path_gradients(model):
    result = {}
    groups = {
        'semantic_grid': lambda n: n.startswith('semantic_grid_projection.'),
        'route_semantic': lambda n: n.startswith('route_readout.semantic_projection.'),
        'actor_interior': lambda n: n.startswith('actor_readout.') and not n.startswith('actor_readout.output_projection.'),
    }
    for name, predicate in groups.items():
        tensors = [p.grad for n, p in model.named_parameters() if predicate(n)]
        if not tensors:
            result[name] = None
        elif any(g is None or not bool(torch.isfinite(g).all()) for g in tensors):
            raise ValueError('missing or nonfinite gradient in ' + name)
        else:
            result[name] = float(torch.stack([g.float().square().sum() for g in tensors]).sum().sqrt())
    return result


def input_reliance(model, items, encoder):
    """Descriptive TRAIN diagnostic; corruption is not a natural gameplay counterfactual."""
    def scores(current):
        return model(current['raw'], current['state'], current['glyph'],
                     player_probabilities(current, encoder), current['semantic'])['action_logits'].float()
    bits = (items['optimal'].long()[:, None] & (1 << torch.arange(4, device=items['optimal'].device))) != 0
    valid = bits.any(-1)
    target = bits.float() / bits.sum(-1, keepdim=True).clamp_min(1)
    def metric(logits):
        chosen = logits.argmax(-1)
        return dict(supported_roots=int(valid.sum()), correct=int(bits.gather(1, chosen[:, None]).squeeze(1)[valid].sum()),
                    policy_ce=float(-(target * logits.log_softmax(-1)).sum() / valid.sum().clamp_min(1)))
    with torch.no_grad():
        native = scores(items)
        shuffled = scores({**items, 'semantic': items['semantic'].roll(37, dims=1)})
        result = dict(native=metric(native), shuffled_cells=metric(shuffled),
                      shuffled_logit_max_change=float((native - shuffled).abs().max()),
                      shuffled_action_changes=int((native.argmax(-1) != shuffled.argmax(-1)).sum()),
                      limitations='This corrupts features on TRAIN examples; it is neither held-out accuracy nor evidence that hidden cells are known.')
        if model.actor_readout is not None:
            handle = model.actor_readout.register_forward_hook(lambda module, inputs, output: torch.zeros_like(output))
            try:
                no_actor = scores(items)
            finally:
                handle.remove()
            result.update(no_actor=metric(no_actor), actor_logit_max_change=float((native - no_actor).abs().max()),
                          actor_action_changes=int((native.argmax(-1) != no_actor.argmax(-1)).sum()))
    return result


def build_model(parent, arm, device='cuda', seed=42):
    if arm not in ARMS:
        raise ValueError('unknown semantic repair arm')
    torch.manual_seed(seed)
    model = SpatialOutcomePlanner(parent['planner_config']).to(device).eval()
    model.load_state_dict(parent['planner_weights'], strict=True)
    model = SpatialSemanticOutcomePlanner.from_parent(model, actor=arm == 'actor')
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
    for key in (*ARRAYS, 'semantic'):
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


def validation_steps(steps, every):
    if type(steps) is not int or type(every) is not int or min(steps, every) < 1:
        raise ValueError('positive integer steps and validation interval required')
    # Check arithmetic bounds before creating a schedule, including before the
    # CLI installs its wall-clock deadline and memory guard.
    if steps > 1_000_000:
        raise ValueError('training steps must not exceed 1000000')
    evaluations = steps // every + int(steps % every != 0)
    if evaluations > 10_000:
        raise ValueError('validation schedule must not exceed 10000 evaluations')
    return tuple(sorted({*range(every, steps + 1, every), steps}))


class ValidationStats:
    """Exact support-weighted full-panel losses and policy-set likelihood."""
    def __init__(self):
        self.rows = self.valid = 0
        self.nll_sum = 0.
        self.loss_sums, self.loss_support = Counter(), Counter()
        self.metric_sums, self.metric_support = Counter(), Counter()
        self.metric_names = set()

    def update(self, scores, optimal, record):
        if scores.shape != (len(optimal), 4) or not bool(torch.isfinite(scores).all()):
            raise ValueError('validation needs finite four-action native scores')
        bits = (optimal.long()[:, None] & (1 << torch.arange(4, device=scores.device))) != 0
        valid = bits.any(-1); count = int(valid.sum()); self.rows += len(optimal); self.valid += count
        if count:
            logp = scores.float().log_softmax(-1)
            nll = -logp[valid].masked_fill(~bits[valid], -torch.inf).logsumexp(-1)
            self.nll_sum += float(nll.clamp_min(0).double().sum())
        for name, value in record['losses'].items():
            support = count if name in ('policy', 'teacher_policy') else len(optimal)
            self.loss_sums[name] += float(value) * support; self.loss_support[name] += support
        for name, value in record['diagnostics'].items():
            self.metric_names.add(name)
            if value is None:
                continue
            support = float(record['diagnostic_weights'][name])
            self.metric_sums[name] += float(value) if name.endswith(('_count', '_support')) else float(value) * support
            self.metric_support[name] += support

    def result(self):
        losses = {name: self.loss_sums[name] / support if support else None for name, support in self.loss_support.items()}
        metrics = {name: (self.metric_sums[name] if name.endswith(('_count', '_support')) else
                         self.metric_sums[name] / self.metric_support[name] if self.metric_support[name] else None)
                   for name in sorted(self.metric_names)}
        return dict(rows=self.rows, policy_defined_roots=self.valid, policy_undefined_roots=self.rows - self.valid,
            optimal_set_nll=self.nll_sum / self.valid if self.valid else None, losses=losses,
            total_loss=sum(value for value in losses.values() if value is not None) if self.rows else None, metrics=metrics,
            loss_supports=dict(self.loss_support), metric_supports=dict(self.metric_support))


def validation_summary(overall, tiers):
    groups = {str(tier): value.result() for tier, value in tiers.items()}
    eligible = [name for name, result in groups.items() if result['optimal_set_nll'] is not None]
    criterion = sum(groups[name]['optimal_set_nll'] for name in eligible) / len(eligible) if eligible else None
    return dict(**overall.result(), tiers=groups, tier_balanced_policy_set_nll=criterion,
                included_tiers=eligible, excluded_tiers=[name for name in groups if name not in eligible])


def evaluate(model, arrays, encoder, weights, tiers, size=256, device='cuda'):
    """Each selection row once; no TRAIN rows or RNG consumed by validation."""
    seeds = np.asarray(arrays['seeds'])
    if set(map(int, seeds)) != set(tiers) or any(value not in range(1, 8) for value in tiers.values()):
        raise ValueError('validation tier mapping must cover every validation level exactly')
    root_tiers = np.array([tiers[int(seed)] for seed in seeds])
    overall, grouped = ValidationStats(), {tier: ValidationStats() for tier in range(1, 8)}
    was_training = model.training; model.eval()
    try:
        with torch.inference_mode():
            for tier, accumulator in grouped.items():
                chosen = np.flatnonzero(root_tiers == tier)
                for start in range(0, len(chosen), size):
                    guard()
                    items = batch(arrays, chosen[start:start + size], device)
                    prediction = model(items['raw'], items['state'], items['glyph'],
                                       player_probabilities(items, encoder), items['semantic'])
                    record = spatial_outcome_losses(model, prediction, items, weights)
                    for target in (overall, accumulator):
                        target.update(prediction['action_logits'], items['optimal'], record)
    finally:
        model.train(was_training)
    return validation_summary(overall, grouped)


class CheckpointSelection:
    """Independent CPU snapshots; exact ties retain the first observed minimum."""
    def __init__(self, mode):
        if mode not in ('final', 'validation-policy'):
            raise ValueError('unknown checkpoint selection mode')
        self.mode, self.selected, self.final = mode, None, None

    def observe(self, model, step, planned_steps, validation, provenance):
        if not 1 <= step <= planned_steps:
            raise ValueError('snapshot step must be an actual completed update')
        score = validation['tier_balanced_policy_set_nll']
        if score is not None and not math.isfinite(score):
            raise ValueError('nonfinite validation selection score')
        candidate = (self.mode == 'final' and step == planned_steps) or (
            self.mode == 'validation-policy' and score is not None and
            (self.selected is None or score < self.selected['criterion_value']))
        if candidate or step == planned_steps:
            snapshot = dict(step=step, criterion_value=score, validation=copy.deepcopy(validation),
                provenance=copy.deepcopy(provenance),
                weights={name: value.detach().cpu().clone() for name, value in model.state_dict().items()})
            if candidate:
                self.selected = snapshot
            if step == planned_steps:
                self.final = copy.deepcopy(snapshot)


def selection_metadata(args, snapshot, role):
    if (role not in ('selected', 'final') or not snapshot or not 1 <= snapshot['step'] <= args.steps
            or ((role == 'final' or args.selection == 'final') and snapshot['step'] != args.steps)):
        raise ValueError('valid selected/final completed snapshot required')
    return dict(role=role, mode=args.selection, selected_step=snapshot['step'],
        planned_steps=args.steps, completed_run_steps=args.steps,
        criterion=('fixed final optimizer step' if role == 'final' or args.selection == 'final' else 'equal mean of supported tier native optimal-set NLL'),
        validation_criterion_value=snapshot['criterion_value'],
        criterion_value=snapshot['step'] if role == 'final' or args.selection == 'final' else snapshot['criterion_value'], validation_every=args.validation_every,
        validation_steps=list(validation_steps(args.steps, args.validation_every)),
        ties='first strict minimum; exact equal scores keep earliest step',
        exposed_validation_is_fresh=False, validation_split='existing validation4000/500levels',
        policy_undefined_roots_excluded=True, included_tiers=snapshot['validation']['included_tiers'],
        excluded_tiers=snapshot['validation']['excluded_tiers'],
        sampling_prefix=copy.deepcopy(snapshot['provenance']))


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
        result = model(items['raw'], items['state'], items['glyph'], player_probabilities(items, encoder), items['semantic'])
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
        minimum_supported_event_recall=.9, consecutive_steps=10,
        semantic_shuffle_minimum_ce_increase=.01, actor_removal_minimum_ce_increase=.01)
    reference = None
    for arm in args.arms:
        gpu_available(); guard(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model = build_model(parent, arm, seed=args.seed)
        with torch.no_grad():
            prediction = model(items['raw'][:64], items['state'][:64], items['glyph'][:64],
                               player_probabilities({k: v[:64] for k, v in items.items()}, encoder), items['semantic'][:64])
        if reference is None:
            reference = {key: value if key != 'field_logits' else tuple(t.clone() for t in value)
                         for key, value in prediction.items()}
        else:
            for key in ('action_logits', 'value_logits', 'event_logits'):
                torch.testing.assert_close(prediction[key], reference[key], atol=0, rtol=0)
            for left, right in zip(prediction['field_logits'], reference['field_logits']):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
        entry = dict(initial_weights_sha256=weights_sha256(model.state_dict()), parameters=model.parameter_count(),
                     original_parameter_count=sum(p.numel() for n, p in model.named_parameters() if not n.startswith(('route_readout.', 'semantic_grid_projection.', 'actor_readout.'))),
                     qualification_updates=[])
        report['arms'][arm] = entry
        optimizer = optimizer_for(model, args.lr)
        for step in range(2):
            torch.cuda.synchronize(); started = time.monotonic()
            record, norm = fit_step(model.train(), optimizer, items, encoder, weights, args.precision)
            torch.cuda.synchronize()
            route_norm = route_gradient_norm(model)
            paths = path_gradients(model)
            if step == 1 and any(v is not None and v <= 0 for v in paths.values()):
                raise ValueError('semantic/actor path remains inactive after two real updates')
            if step == 1 and not route_norm > 0:
                raise ValueError('route interior remains inactive after two real updates')
            entry['qualification_updates'].append(dict(step=step + 1, elapsed_seconds=time.monotonic() - started,
                loss=float(record['total']), gradient_norm=float(norm), route_interior_gradient_norm=route_norm, path_gradients=paths,
                preclip_global_gradient_norm=float(norm), postclip_route_interior_gradient_norm=route_norm,
                postclip_path_gradients=paths, gradient_was_clipped=float(norm) > 1.,
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
        entry['tiny_input_reliance'] = input_reliance(model.eval(), tiny, encoder)
        write(args.out / 'report.json', report)
        reliance = entry['tiny_input_reliance']
        entry['input_reliance_passed'] = reliance['shuffled_cells']['policy_ce'] > reliance['native']['policy_ce'] + .01
        if arm == 'actor':
            entry['input_reliance_passed'] &= reliance['no_actor']['policy_ce'] > reliance['native']['policy_ce'] + .01
        if not entry['input_reliance_passed']:
            raise RuntimeError(f'{arm} new inputs lack measurable TRAIN policy benefit; investigate before full training')
        if consecutive < 10:
            raise RuntimeError(f'{arm} balanced tiny-fit gate failed; full training not admitted')
        del model, optimizer, result
        gc.collect(); torch.cuda.empty_cache()
    report.update(qualification_passed=True, initial_semantic_arm_outputs_equal=True,
                  qualification_updates_discarded=True, selected_batch_size=sampler.batch_size, qualified_arms=list(args.arms))


def checkpoint(parent, model, arm, args, report, snapshot, *, role):
    if weights_sha256(model.state_dict()) != weights_sha256(snapshot['weights']):
        raise ValueError('checkpoint model weights differ from selected snapshot metadata')
    result = checkpoint_from_parent(parent, model, load_perceptor(TEACHER), parent_checkpoint_sha256=PARENT_SHA)
    stale = ('quality_filter', 'quality_manifest', 'quality_manifest_sha256', 'recovery_fraction', 'policy_row_fraction',
             'continuation_arm', 'continuation_parent_sha256', 'supplemental_cache', 'supplemental_manifest_sha256',
             'planner_initialization', 'optimizer_steps', 'learning_rate', 'objective_weights',
             'sampling', 'precision', 'seed', 'quality_row_file_sha256')
    history = {key: result.pop(key) for key in stale if key in result}
    result.update(planner_config=model.config(), planner_weights={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                  planner_parameters=model.parameter_count(), parent_training_metadata=history,
                  source_checkpoint_sha256=PARENT_SHA, source_sha256=report['source_sha256'],
                  optimizer='AdamW', optimizer_state='not stored; fresh moments at run initialization', optimizer_steps=snapshot['step'], learning_rate=args.lr,
                  weight_decay=.05, batch_size=sum(args.tier_counts), train_levels=10000,
                  sampling=report['sampling'], precision=args.precision, seed=args.seed,
                  learning_rate_schedule='3% linear warmup then cosine decay',
                  quality_row_file_sha256=report['quality_row_file_sha256'],
                  training_cache=str(args.cache), supplemental_cache=str(args.recent),
                  cache_manifest_sha256=report['base_manifest_sha256'], supplemental_manifest_sha256=report['recent_manifest_sha256'],
                  objective_weights=report['objective_weights'], actual_outcome_comparator_auxiliary_training=True,
                  semantic_repair=dict(arm=arm, seed=args.seed, all_planner_parameters_trained=True,
                      sampling=report['sampling'], full_original_failure_supervision=True,
                      encoder_frozen=True, gameplay_gain_established=False, persistent_game_memory=False,
                      learned_voluntary_reset=False, qualification_report_sha256=report['qualification_sha256']),
                  semantic_cache=str(args.semantics), semantic_manifest_sha256=report['semantic_manifest_sha256'],
                  checkpoint_selection=selection_metadata(args, snapshot, role),
                  planner_sampling_sha256=snapshot['provenance']['sampling_sha256'])
    return result


def train(parent, arrays, recent, approved, tiers, validation_tiers, encoder, weights, args, report):
    proof = json.loads(args.qualification.read_text())
    if (proof.get('status') != 'complete' or not proof.get('qualification_passed')
            or proof['learning_rate'] != args.lr or proof['precision'] != args.precision
            or proof['tier_counts'] != args.tier_counts or proof['parent_sha256'] != PARENT_SHA
            or proof['base_manifest_sha256'] != report['base_manifest_sha256']
            or proof['objective_weights'] != weights
            or proof['semantic_manifest_sha256'] != report['semantic_manifest_sha256']
            or proof.get('qualified_arms') != list(ARMS)):
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
        entry = dict(status='running', initial_weights_sha256=initial, parameters=model.parameter_count(), updates=[], validation_curve=[])
        selector = CheckpointSelection(args.selection)
        evaluation_steps = set(validation_steps(args.steps, args.validation_every))
        clipped_updates = 0
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
            clipped_updates += int(float(norm) > 1.)
            zero_rows += int((items['optimal'] == 0).sum())
            replay_kinds.update(map(int, selection['row_kinds']))
            tier_counts.update(tiers[int(seed)] for seed in selection['seeds'])
            if step == 0 or (step + 1) % 25 == 0 or step + 1 == args.steps:
                torch.cuda.synchronize()
                update = dict(step=step + 1, elapsed_seconds=time.monotonic() - started,
                    loss=float(record['total']), metrics=scalar_metrics(record), gradient_norm=float(norm),
                    peak_gpu_bytes=torch.cuda.max_memory_allocated(), available_host_bytes=available,
                    route_interior_gradient_norm=route_gradient_norm(model), path_gradients=path_gradients(model),
                    preclip_global_gradient_norm=float(norm), postclip_route_interior_gradient_norm=route_gradient_norm(model),
                    postclip_path_gradients=path_gradients(model), clipped_update_fraction=clipped_updates / (step + 1))
                entry['updates'].append(update); write(args.out / 'report.json', report)
                print(json.dumps(dict(event='training', arm=arm, **update)), flush=True)
            if step + 1 in evaluation_steps:
                validation = evaluate(model, arrays['validation'], encoder, weights, validation_tiers)
                provenance = dict(sampling_sha256=sampling_hash.hexdigest(), optimizer_steps=step + 1,
                    zero_policy_roots_sampled=zero_rows, trajectory_kinds=dict(replay_kinds), tier_counts=dict(tier_counts))
                selector.observe(model, step + 1, args.steps, validation, provenance)
                entry['validation_curve'].append(dict(step=step + 1, elapsed_seconds=time.monotonic() - started,
                                                     selected_so_far=selector.selected['step'] if selector.selected else None,
                                                     **validation))
                write(args.out / 'report.json', report)
                print(json.dumps(dict(event='validation', arm=arm, step=step + 1,
                    tier_balanced_policy_set_nll=validation['tier_balanced_policy_set_nll'])), flush=True)
        if selector.selected is None or selector.final is None:
            raise ValueError('no eligible checkpoint could be selected from complete fixed-interval validation')
        entry.update(status='complete', elapsed_seconds=time.monotonic() - started,
            final_weights_sha256=weights_sha256(selector.final['weights']), sampling_sha256=sampling_hash.hexdigest(),
            selected_weights_sha256=weights_sha256(selector.selected['weights']),
            selected_step=selector.selected['step'], completed_optimizer_steps=args.steps,
            clipped_updates=clipped_updates, clipped_update_fraction=clipped_updates / args.steps,
            zero_policy_roots_sampled=zero_rows, trajectory_kinds=dict(replay_kinds), tier_counts=dict(tier_counts),
            validation=selector.selected['validation']['metrics'], selected_validation=selector.selected['validation'],
            final_validation=selector.final['validation'], checkpoint_selection=selection_metadata(args, selector.selected, 'selected'))
        for role, snapshot, filename in [('final', selector.final, arm + '.final.pending.pt'),
                                          ('selected', selector.selected, arm + '.pending.pt')]:
            model.load_state_dict(snapshot['weights'], strict=True)
            if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
                raise ValueError('nonfinite checkpoint planner tensor')
            destination = args.out / filename
            torch.save(checkpoint(parent, model, arm, args, report, snapshot, role=role), destination)
            entry['final_checkpoint_sha256' if role == 'final' else 'checkpoint_sha256'] = sha(destination)
        entry['validation_input_reliance'] = input_reliance(model.eval(), batch(arrays['validation'], np.arange(256), 'cuda'), encoder)
        write(args.out / 'report.json', report)
        del model, optimizer, items, record
        gc.collect(); torch.cuda.empty_cache()
    if report['arms']['control']['sampling_sha256'] != report['arms']['actor']['sampling_sha256']:
        raise ValueError('matched arms used different rows')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / 'data/reference-outcome-inputs-v1')
    parser.add_argument('--recent', type=Path)
    parser.add_argument('--semantics', type=Path, default=ROOT / 'data/spatial-repair-v4-semantics')
    parser.add_argument('--qualification', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--qualify', action='store_true')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS), help='Subset is permitted only for disposable qualification diagnostics.')
    parser.add_argument('--steps', type=int, default=1560)
    parser.add_argument('--selection', choices=('final', 'validation-policy'), default='final')
    parser.add_argument('--validation-every', type=int, default=100)
    parser.add_argument('--tiny-steps', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=.0001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tier-counts', nargs=7, type=int, default=list(TIER_COUNTS))
    parser.add_argument('--recent-fraction', type=float, default=.25)
    parser.add_argument('--precision', choices=('float32', 'bf16'), default='float32')
    parser.add_argument('--max-seconds', type=int, default=2400)
    args = parser.parse_args(argv)
    if (min(args.steps, args.tiny_steps, args.max_seconds, args.validation_every) < 1 or args.max_seconds > 3600
            or not math.isfinite(args.lr) or not 0 < args.lr < 1 or args.seed < 0
            or sum(args.tier_counts) not in (64, 128, 256, 512, 1024)):
        parser.error('bounded positive budgets, learning rate and power-of-two batch up to1024 required')
    try:
        scheduled_validation_steps = validation_steps(args.steps, args.validation_every)
    except ValueError as error:
        parser.error(str(error))
    if not args.qualify and (args.recent is None or args.qualification is None):
        parser.error('full training requires recent replay and a completed qualification')
    if len(set(args.arms)) != len(args.arms) or (not args.qualify and args.arms != list(ARMS)):
        parser.error('arm subsets must be distinct and are qualification-only')
    if args.out.exists():
        raise FileExistsError(args.out)
    report = dict(status='running', pid=os.getpid(), started_local=datetime.now().astimezone().isoformat(),
        mode='qualification' if args.qualify else 'training', learning_rate=args.lr, precision=args.precision,
        seed=args.seed, parent_sha256=PARENT_SHA, tier_counts=args.tier_counts, planned_steps=args.steps,
        selection_protocol=dict(mode=args.selection, validation_every=args.validation_every,
            validation_steps=list(scheduled_validation_steps),
            criterion=('fixed final optimizer step' if args.selection == 'final' else 'equal mean of supported tier native optimal-set NLL; policy-defined roots only'),
            validation_metric='equal mean of supported tier native optimal-set NLL; policy-defined roots only',
            ties='first strict minimum', unsupported_tiers='excluded and reported; all unsupported gives None',
            existing_validation_is_fresh=False, selected_and_final_saved_separately=True),
        gradient_telemetry=dict(clip_threshold=1., gradient_norm='preclip global L2 norm (legacy key)',
            route_interior_gradient_norm='postclip L2 norm; route output projection excluded (legacy key)',
            path_gradients='postclip per-path L2 norms (legacy key)',
            limits='Parameter-gradient norms are not loss-task shares or AdamW update shares.'),
        max_seconds=args.max_seconds, arms={}, source_sha256={}, encoder_frozen=True,
        official_training_inputs=False, privileged_inference_inputs=False,
        limits=['Scene attention refines current-scene queries; it is not multi-step planning.',
                'Encoder, persistent-memory capability and voluntary-reset capability are unchanged.',
                'Tiny fit only demonstrates trainability; native retention and new confirmation remain required.',
                'Fixed update/deadline budgets; no convergence test or adaptive early stopping.',
                'Existing validation is exposed development/selection data, not fresh confirmation.'])
    started, arrays, recent, approved, created = time.monotonic(), {}, None, {}, False
    old_handler = signal.getsignal(signal.SIGALRM)
    def timeout(*_):
        raise TimeoutError('bounded semantic repair deadline exceeded')
    try:
        signal.signal(signal.SIGALRM, timeout); signal.alarm(args.max_seconds)
        guard(); gpu_available()
        args.out.mkdir(parents=True); created = True
        write(args.out / 'report.json', report)
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        bindings = Bindings(); bindings.add(PARENT, PARENT_SHA); bindings.add(BANK, BANK_SHA); bindings.add(TEACHER, TEACHER_SHA256)
        for path in [Path(__file__), *sorted((ROOT / 'pebby/agent').glob('*.py')),
                     *[ROOT / 'tools' / name for name in ('train_reference_outcomes.py', 'train_reference_spatial_outcomes.py',
                         'train_spatial_recovery_comparison.py', 'cache_reference_outcome_inputs.py')]]:
            bindings.add(path)
        parent = torch.load(PARENT, map_location='cpu', weights_only=True); validate_checkpoint(parent)
        arrays, hashes, stats = load_data(args.cache)
        report.update(base_manifest_sha256=bindings.add(args.cache / 'manifest.json'))
        if report['base_manifest_sha256'] != parent['cache_manifest_sha256']:
            raise ValueError('base cache differs from parent lineage')
        bindings.add(VALIDATION_BANK)
        validation_rows = [json.loads(line) for line in VALIDATION_BANK.read_text().splitlines() if line.strip()]
        validation_tiers = {int(row['seed']): int(row['difficulty']) for row in validation_rows}
        if (len(validation_rows) != 500 or len(validation_tiers) != 500
                or set(validation_tiers) != set(map(int, arrays['validation']['seeds']))
                or any(tier not in range(1, 8) for tier in validation_tiers.values())):
            raise ValueError('complete disjoint validation500 tier mapping required')
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
        from tools import cache_spatial_semantics as semantic_cache
        bindings.add(Path(semantic_cache.__file__))
        semantic_manifest = semantic_cache.validate_published(args.semantics)
        if (semantic_manifest['teacher_sha256'] != TEACHER_SHA256
                or semantic_manifest['base_manifest_sha256'] != report['base_manifest_sha256']
                or (recent is not None and semantic_manifest['recent_manifest_sha256'] != report['recent_manifest_sha256'])):
            raise ValueError('semantic cache is not aligned to these exact outcome caches')
        report['semantic_manifest_sha256'] = bindings.add(args.semantics / 'manifest.json')
        hashes.update(semantic_manifest['validated_output_hashes'])
        stats.update(semantic_manifest['validated_output_stats'])
        for split, group in [*arrays.items(), *([('recent', recent)] if recent is not None else [])]:
            add_semantic(group, args.semantics / split)
        weights = training_weights(arrays['train'])
        report.update(objective_weights=weights, source_sha256={**hashes, **bindings.hashes},
                      train_roots=len(arrays['train']['seeds']), zero_policy_train_roots=int((arrays['train']['optimal'] == 0).sum()))
        encoder = {k: v.cuda() for k, v in parent['encoder_weights'].items() if k.startswith('player_head.')}
        if args.qualify:
            qualification(parent, arrays, tiers, encoder, weights, args, report)
        else:
            train(parent, arrays, recent, approved, tiers, validation_tiers, encoder, weights, args, report)
        bindings.verify()
        if any(stat(path) != value for path, value in stats.items()):
            raise ValueError('bound feature arrays changed during experiment')
        if weights_sha256(parent['encoder_weights']) != parent['encoder_weights_sha256']:
            raise ValueError('frozen encoder weights changed')
        if not args.qualify:
            for arm in ARMS:
                for suffix in ('', '.final'):
                    (args.out / (arm + suffix + '.pending.pt')).replace(args.out / (arm + suffix + '.pt'))
        report.update(status='complete', sources_unchanged=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        if created:
            for arm in ARMS:
                for suffix in ('', '.final'):
                    (args.out / (arm + suffix + '.pending.pt')).unlink(missing_ok=True)
                    (args.out / (arm + suffix + '.pt')).unlink(missing_ok=True)
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
