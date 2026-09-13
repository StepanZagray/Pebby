"""Read-only CPU VAL decisions and destructive-input diagnostics; no fitting."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from pebby.agent.neural_outcome_policy import weights_sha256
from pebby.agent.spatial_semantic_outcome_policy import load_checkpoint, TEACHER_SHA256
from pebby.agent.spatial_outcome_policy import load_checkpoint as load_baseline
from tools import cache_spatial_semantics
from tools.audit_spatial_route_resources import ResourceMetrics, resource_targets
from tools.cache_reference_outcome_inputs import Bindings, process_start_ticks, release, stat
from tools.train_reference_outcomes import batch, guard, load_data, write
from tools.train_reference_spatial_outcomes import player_probabilities

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'artifacts/spatial-recovery-v1/quality-fit/recovery.pt'
BASELINE_SHA = 'ff88327214b6dc2d4278167e0d61edcc37c683292b788be6927b71286331a5f8'
VAL_BANK = ROOT / 'data/ls20-reference-unequal-v1/validation.jsonl'


def decision_values(scores, items):
    """Per-root quantities; undefined optimal masks never enter policy means."""
    if scores.ndim != 2 or scores.shape[1] != 4 or not bool(torch.isfinite(scores).all()):
        raise ValueError('finite action scores[B,4] required')
    masks = items['optimal'].long()
    if masks.shape != (len(scores),) or bool(((masks < 0) | (masks > 15)).any()):
        raise ValueError('optimal masks must be four action bits per root')
    bits = (masks[:, None] & (1 << torch.arange(4, device=scores.device))) != 0
    valid = bits.any(-1); chosen = scores.argmax(-1); logp = scores.float().log_softmax(-1)
    chosen_optimal = bits.gather(1, chosen[:, None]).squeeze(1)
    uniform = bits.float() / bits.sum(-1, keepdim=True).clamp_min(1)
    ce = -(uniform * logp).sum(-1)
    # Avoid an all-negative-infinity reduction on undefined policy targets.
    allowed = torch.where(valid[:, None], bits, torch.ones_like(bits))
    set_nll = -logp.masked_fill(~allowed, -torch.inf).logsumexp(-1)
    _, _, cohorts = resource_targets(items)
    refill = cohorts['live_refill']; optimal_refill = refill & bits
    available_refill = optimal_refill.any(-1)
    required_refill = available_refill & ~(bits & ~refill).any(-1)
    picked_optimal_refill = optimal_refill.gather(1, chosen[:, None]).squeeze(1)
    distances = items['distances']
    if distances.shape != scores.shape or distances.is_floating_point():
        raise ValueError('true distances must be integer[B,4]')
    reachable = distances >= 0; has_reachable = reachable.any(-1)
    chosen_unreachable = ~reachable.gather(1, chosen[:, None]).squeeze(1)
    return dict(chosen=chosen, valid=valid, correct=chosen_optimal, uniform_ce=ce, set_nll=set_nll,
        optimal_refill_available=available_refill, optimal_refill_required=required_refill,
        picked_optimal_refill=picked_optimal_refill, has_reachable=has_reachable,
        picked_unreachable=chosen_unreachable,
        picked_life_loss=items['lost_life'].bool().gather(1, chosen[:, None]).squeeze(1),
        picked_live_refill=refill.gather(1, chosen[:, None]).squeeze(1))


class DecisionMetrics:
    def __init__(self):
        self.groups = {name: dict(roots=0, policy_defined=0, correct=0, uniform_ce_sum=0., set_nll_sum=0.,
            optimal_live_refill_available=0, missed_available_optimal_live_refill=0,
            optimal_live_refill_required=0, missed_required_optimal_live_refill=0,
            reachable_roots=0, reachable_to_unreachable=0, selected_life_loss=0, selected_live_refill=0)
            for name in ('all', *map(str, range(1, 8)))}

    def update(self, scores, items, tiers):
        values = decision_values(scores, items)
        tiers = torch.as_tensor(tiers, device=scores.device)
        if tiers.shape != (len(scores),) or bool(((tiers < 1) | (tiers > 7)).any()):
            raise ValueError('each validation root needs tier1..7')
        for name, count in self.groups.items():
            selected = torch.ones_like(values['valid']) if name == 'all' else tiers == int(name)
            policy = selected & values['valid']
            count['roots'] += int(selected.sum()); count['policy_defined'] += int(policy.sum())
            count['correct'] += int((policy & values['correct']).sum())
            count['uniform_ce_sum'] += float(values['uniform_ce'][policy].double().sum())
            count['set_nll_sum'] += float(values['set_nll'][policy].double().sum())
            for cohort, support, missed in [
                ('optimal_refill_available', 'optimal_live_refill_available', 'missed_available_optimal_live_refill'),
                ('optimal_refill_required', 'optimal_live_refill_required', 'missed_required_optimal_live_refill')]:
                eligible = selected & values[cohort]
                count[support] += int(eligible.sum())
                count[missed] += int((eligible & ~values['picked_optimal_refill']).sum())
            reachable = selected & values['has_reachable']
            count['reachable_roots'] += int(reachable.sum())
            count['reachable_to_unreachable'] += int((reachable & values['picked_unreachable']).sum())
            count['selected_life_loss'] += int((selected & values['picked_life_loss']).sum())
            count['selected_live_refill'] += int((selected & values['picked_live_refill']).sum())
        return values

    def result(self):
        result = {}
        for name, counts in self.groups.items():
            n = counts['policy_defined']
            result[name] = {**counts, 'policy_undefined': counts['roots'] - n,
                'optimal_set_accuracy': counts['correct'] / n if n else None,
                'uniform_optimal_ce': counts['uniform_ce_sum'] / n if n else None,
                'optimal_set_nll': counts['set_nll_sum'] / n if n else None,
                'reachable_to_unreachable_rate': counts['reachable_to_unreachable'] / counts['reachable_roots'] if counts['reachable_roots'] else None}
        return result


def compare_decisions(native, other):
    if native['chosen'].shape != other['chosen'].shape or not torch.equal(native['valid'], other['valid']):
        raise ValueError('decision comparisons must use identical root masks')
    valid = native['valid']; changed = native['chosen'] != other['chosen']
    return dict(roots=len(valid), policy_defined=int(valid.sum()), action_changes=int(changed.sum()),
        policy_defined_action_changes=int((changed & valid).sum()),
        helped=int((valid & ~native['correct'] & other['correct']).sum()),
        hurt=int((valid & native['correct'] & ~other['correct']).sum()))


def validate_alignment(arrays, semantic_path):
    for key in ('rows', 'seeds'):
        expected = np.load(semantic_path / f'{key}.npy', allow_pickle=False)
        if not np.array_equal(arrays[key], expected):
            raise ValueError('semantic/encoder row alignment differs: ' + key)
    value = np.load(semantic_path / 'semantic.npy', mmap_mode='r', allow_pickle=False)
    if value.shape != (len(arrays['seeds']), 144, 22) or value.dtype != np.float32:
        release({'semantic': value}, close=True)
        raise ValueError('semantic validation array schema differs')
    return value


def native_score_weight(checkpoint):
    weights = checkpoint['score_weights']
    if weights.get('direct') != 0 or not isinstance(weights.get('planner'), (int, float)) or not 0 < weights['planner'] < float('inf'):
        raise ValueError('cached diagnostic requires pure planner native score weights')
    return weights['planner']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--semantics', type=Path, default=ROOT / 'data/spatial-repair-v4-semantics')
    parser.add_argument('--cache', type=Path, default=ROOT / 'data/reference-outcome-inputs-v1')
    parser.add_argument('--without-baseline', action='store_true')
    parser.add_argument('--batch-size', type=int, choices=(32, 64, 128), default=64)
    parser.add_argument('--max-seconds', type=int, default=120)
    args = parser.parse_args(argv)
    if not 1 <= args.max_seconds <= 120 or len(args.checkpoint_sha256) != 64:
        parser.error('deadline1..120s and full checkpoint SHA256 required')
    if args.out.exists():
        raise FileExistsError(args.out)
    started = time.monotonic(); previous_threads = torch.get_num_threads()
    all_arrays, semantic = {}, None; bindings = Bindings(); guards = {}; policy = baseline = None
    report = dict(status='running', pid=os.getpid(), start_ticks=process_start_ticks(),
        started_local=datetime.now().astimezone().isoformat(), device='cpu', max_seconds=args.max_seconds,
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=args.checkpoint_sha256, training=False,
        split='validation', batch_size=args.batch_size, privileged_inputs=False,
        definitions=dict(policy_defined='optimal bitmask !=0; undefined roots excluded only from policy CE/NLL/accuracy',
            uniform_optimal_ce='Mean cross entropy against uniform probability across true optimal actions; same target as native training CE.',
            optimal_set_nll='Mean negative log total predicted probability of the true optimal action set.',
            live_refill='No life loss, terminal or win, and successor budget exceeds current budget.',
            refill_available='At least one optimal action refills; a missed refill may be another equally optimal action.',
            refill_required='Every optimal action is a live refill; missing this cohort is a policy error.',
            reachable_to_unreachable='A true finite-distance successor exists, but the chosen successor distance is negative.',
            actor_removed='Frozen outcome comparator scores only; typed predictions unchanged.',
            semantic_cell_shuffle='Roll22-channel cell vectors37places, misaligning semantic positions with original raw/state features.'),
        limits=['Cached one-step VAL diagnostics; no native episode/gameplay claim.',
            'Semantic cell shuffling is artificial feature corruption/OOD, not a natural causal gameplay intervention.',
            'Actor removal measures sensitivity of this fixed model, not the outcome of training without an actor.',
            'Existing validation has been inspected during development; not fresh confirmation.',
            'Predicted public semantic probabilities remain imperfect; source parity is not a scene-accuracy guarantee.'])
    def expired(*_):
        raise TimeoutError('120-second bounded CPU semantic decision audit expired')
    previous_alarm = signal.signal(signal.SIGALRM, expired); signal.alarm(args.max_seconds)
    print(json.dumps(dict(pid=os.getpid(), started_local=report['started_local'], status='validating')), flush=True)
    try:
        torch.set_num_threads(1); guard(); bindings.add(args.checkpoint, args.checkpoint_sha256)
        policy, checkpoint = load_checkpoint(args.checkpoint, 'cpu'); policy.eval().requires_grad_(False)
        scale = native_score_weight(checkpoint)
        before = weights_sha256(policy.state_dict())
        all_arrays, hashes, guards = load_data(args.cache)
        bindings.hashes.update(hashes); bindings.before.update(guards)
        original_manifest_sha = bindings.add(args.cache / 'manifest.json')
        if checkpoint['cache_manifest_sha256'] != original_manifest_sha:
            raise ValueError('encoder cache differs from checkpoint lineage')
        semantic_manifest = cache_spatial_semantics.validate_published(args.semantics)
        bindings.hashes.update(semantic_manifest['validated_output_hashes']); bindings.before.update(semantic_manifest['validated_output_stats'])
        semantic_sha = bindings.add(args.semantics / 'manifest.json')
        if (semantic_manifest['base_manifest_sha256'] != original_manifest_sha
                or semantic_manifest['teacher_sha256'] != TEACHER_SHA256
                or checkpoint.get('semantic_manifest_sha256') != semantic_sha):
            raise ValueError('semantic addon differs from checkpoint training lineage')
        arrays = all_arrays['validation']; semantic = validate_alignment(arrays, args.semantics / 'validation')
        if len(arrays['seeds']) != 4000 or len(np.unique(arrays['seeds'])) != 500:
            raise ValueError('complete4000-root/500-level validation panel required')
        bindings.add(VAL_BANK)
        levels = [json.loads(line) for line in VAL_BANK.read_text().splitlines() if line.strip()]
        tier_by_seed = {int(level['seed']): int(level['difficulty']) for level in levels}
        if len(tier_by_seed) != 500 or set(map(int, arrays['seeds'])) != set(tier_by_seed):
            raise ValueError('validation feature seeds differ from generated tier bank')
        source_files = [Path(__file__), *sorted((ROOT / 'pebby/agent').glob('*.py')),
            *[ROOT / 'tools' / name for name in ('cache_spatial_semantics.py', 'audit_spatial_route_resources.py',
                'cache_reference_outcome_inputs.py', 'train_reference_outcomes.py', 'train_reference_spatial_outcomes.py')]]
        for source in source_files:
            bindings.add(source)
        for source in [ROOT / 'pebby/agent' / name for name in ('spatial_semantic_outcome_planner.py',
                       'spatial_semantic_outcome_policy.py', 'cell_appearance.py', 'spatial_outcome_planner.py')]:
            if checkpoint.get('source_sha256', {}).get(str(source)) != bindings.hashes[str(source)]:
                raise ValueError('checkpoint inference source differs from audited implementation')
        if not args.without_baseline:
            bindings.add(BASELINE, BASELINE_SHA); baseline, baseline_cp = load_baseline(BASELINE, 'cpu')
            baseline.eval().requires_grad_(False); baseline_before = weights_sha256(baseline.state_dict())
            baseline_scale = native_score_weight(baseline_cp)
            if (baseline_cp['cache_manifest_sha256'] != original_manifest_sha
                    or baseline_cp['encoder_weights_sha256'] != checkpoint['encoder_weights_sha256']):
                raise ValueError('matched baseline public encoder/cache differs')
        modes = ['native', 'actor_removed', 'semantic_cell_shuffle'] + ([] if baseline is None else ['baseline'])
        metrics = {name: DecisionMetrics() for name in modes}
        resources = {name: ResourceMetrics() for name in ['native', 'semantic_cell_shuffle'] + ([] if baseline is None else ['baseline'])}
        comparisons = {name: {} for name in modes if name != 'native'}
        saved = {name: [] for name in modes}; component_square = dict(actor=0., outcome=0., values=0)
        encoder = {key: value for key, value in checkpoint['encoder_weights'].items() if key.startswith('player_head.')}
        with torch.inference_mode():
            for start in range(0, len(arrays['seeds']), args.batch_size):
                guard(); rows = np.arange(start, min(start + args.batch_size, len(arrays['seeds'])))
                items = batch(arrays, rows, 'cpu'); scene = torch.from_numpy(np.array(semantic[rows], copy=True))
                player = player_probabilities(items, encoder); inputs = (items['raw'], items['state'], items['glyph'], player)
                native = policy.planner(*inputs, scene, return_components=True)
                torch.testing.assert_close(native['action_logits'], native['outcome_action_logits'] + native['actor_action_logits'], atol=0, rtol=0)
                shuffled = policy.planner(*inputs, scene.roll(37, dims=1))
                predictions = dict(native=native, semantic_cell_shuffle=shuffled)
                scores = dict(native=native['action_logits'] * scale, actor_removed=native['outcome_action_logits'] * scale,
                              semantic_cell_shuffle=shuffled['action_logits'] * scale)
                if baseline is not None:
                    predictions['baseline'] = baseline.planner(*inputs)
                    scores['baseline'] = predictions['baseline']['action_logits'] * baseline_scale
                tiers = [tier_by_seed[int(s)] for s in arrays['seeds'][rows]]
                quantities = {name: metrics[name].update(value, items, tiers) for name, value in scores.items()}
                for name, prediction in predictions.items(): resources[name].update(prediction, items)
                for name in comparisons:
                    for key, value in compare_decisions(quantities['native'], quantities[name]).items():
                        comparisons[name][key] = comparisons[name].get(key, 0) + value
                for name, value in quantities.items(): saved[name].extend(value['chosen'].tolist())
                for name in ('actor', 'outcome'):
                    value = native[f'{name}_action_logits'].float(); centered = value - value.mean(-1, keepdim=True)
                    component_square[name] += float(centered.double().square().sum())
                component_square['values'] += native['action_logits'].numel()
        if before != weights_sha256(policy.state_dict()) or any(p.grad is not None or p.requires_grad for p in policy.parameters()):
            raise ValueError('frozen candidate changed or received gradients')
        if baseline is not None and baseline_before != weights_sha256(baseline.state_dict()):
            raise ValueError('frozen baseline changed')
        bindings.verify()
        if torch.cuda.is_initialized():
            raise ValueError('CPU diagnostic initialized CUDA')
        report.update(status='complete', rows=4000, levels=500, actor_present=policy.planner.actor_readout is not None,
            baseline_sha256=None if baseline is None else BASELINE_SHA,
            cache_manifest_sha256=original_manifest_sha, semantic_manifest_sha256=semantic_sha,
            source_sha256=bindings.hashes, source_stats=bindings.before, sources_unchanged=True,
            weights_unchanged=True, frozen_policy_weights_sha256=before,
            decision_metrics={key: value.result() for key, value in metrics.items()},
            resource_metrics={key: value.result() for key, value in resources.items()},
            relative_to_native=comparisons,
            centered_component_rms={name: (component_square[name] / component_square['values'])**.5 for name in ('actor', 'outcome')},
            decisions=dict(seeds=arrays['seeds'].tolist(), source_rows=arrays['rows'].tolist(), **saved),
            cuda_initialized=False)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous_alarm)
        for group in all_arrays.values(): release(group, close=True)
        if semantic is not None: release({'semantic': semantic}, close=True)
        torch.set_num_threads(previous_threads)
        report.update(elapsed_seconds=time.monotonic() - started, finished_local=datetime.now().astimezone().isoformat())
        args.out.parent.mkdir(parents=True, exist_ok=True); write(args.out, report)
    return report


if __name__ == '__main__':
    main()
