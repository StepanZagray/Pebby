"""Bounded K1/K4 neural planning comparison with identical frozen learned D.

Generated cached fields are public-encoder outputs. Actual successors enter
only auxiliary continuation-policy supervision, never imagined root scoring.
This experiment does not qualify reset dynamics or persistent game memory.
"""
import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from pebby.agent.planning_decision_metrics import decision_metrics
from pebby.agent.neural_imagination_policy import NeuralImaginationPolicy, save_checkpoint, load_checkpoint
from pebby.agent.structured_factored_policy import load_factored_policy_checkpoint, state_digest
from tools.train_navigation_probe import Budget, digest, write_json
from tools.train_structured_policy import load_policy_cache, check_policy_encoder, policy_terms


def batch(data, rows, device):
    return {name: torch.as_tensor(np.array(data[name][rows]), device=device)
            for name in ('fields', 'optimal', 'next_fields', 'next_optimal', 'distances', 'lost_life')}


def continuation_loss(planner, items, actions):
    current = policy_terms(planner.continuation_logits(items['fields'].float()), items['optimal'])['ce'].mean()
    ids = torch.arange(len(actions), device=actions.device)
    actual = items['next_fields'][ids, actions].float()
    masks = items['next_optimal'][ids, actions]
    terms = policy_terms(planner.continuation_logits(actual), masks, allow_unreachable=True)
    following = terms['ce'].sum() / terms['valid'].sum().clamp_min(1)
    return (current + following) / 2


def stratified_rows(difficulties, count, seed):
    rng = np.random.default_rng(seed)
    pools = [list(rng.permutation(np.flatnonzero(difficulties == tier))) for tier in np.unique(difficulties)]
    rows = []
    while len(rows) < min(count, len(difficulties)):
        for pool in pools:
            if pool and len(rows) < count:
                rows.append(pool.pop())
    return np.asarray(rows, dtype=np.int64)


@torch.no_grad()
def evaluate(planner, data, rows, device, size, guard):
    planner.eval()
    result = {}
    for ablation in (False, True):
        correct, ce, count = 0, 0., 0
        traces = []
        decision_counts = {}
        for first in range(0, len(rows), size):
            guard()
            selected = rows[first:first + size]
            items = batch(data, selected, device)
            output = planner.imagine(items['fields'].float(), tail_ablation=ablation)
            terms = policy_terms(output['action_logits'], items['optimal'])
            metrics = decision_metrics(output['action_logits'], items['distances'], items['lost_life'], items['optimal'])
            for key, value in metrics.items():
                decision_counts[key] = decision_counts.get(key, 0) + value
            correct += int(terms['correct'].sum())
            ce += float(terms['ce'].sum())
            count += len(selected)
            if not ablation:
                traces.extend(dict(row=int(row), seed=int(data['seeds'][row]), actions=trace)
                              for row, trace in zip(selected, output['imagined_actions'].cpu().tolist()))
        result['repeat_h1_tail' if ablation else 'native'] = dict(
            correct=correct, count=count, accuracy=correct / count, ce=ce / count,
            imagined_traces=traces, decision_counts=decision_counts)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=Path('checkpoints/ls20-structured-policy-paired-local-h4-600.pt'))
    parser.add_argument('--parent-sha256', required=True)
    parser.add_argument('--dynamics-parent', type=Path, help='Pinned repaired learned dynamics; encoder must match parent')
    parser.add_argument('--dynamics-sha256')
    parser.add_argument('--continuation-parent', type=Path)
    parser.add_argument('--continuation-sha256')
    parser.add_argument('--cache', type=Path, default=Path('data/structured-field-16384'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--offline-diagnostics', action='store_true',
                        help='Optional cached-state diagnostics AFTER actual gameplay; never a selection gate')
    parser.add_argument('--gameplay-baseline', type=Path, default=Path('artifacts/spatial-recovery-v1/quality-fit/recovery.pt'))
    parser.add_argument('--gameplay-baseline-sha256', default='ff88327214b6dc2d4278167e0d61edcc37c683292b788be6927b71286331a5f8')
    parser.add_argument('--updates', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--validation-count', type=int, default=128)
    parser.add_argument('--horizons', nargs='+', type=int, choices=(1, 4), default=[1, 4])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=.0003)
    parser.add_argument('--max-seconds', type=float, default=600)
    args = parser.parse_args(argv)
    if (not 1 <= args.updates <= 2000 or not 1 <= args.batch_size <= 128
            or args.validation_count < 1 or len(set(args.horizons)) != len(args.horizons)
            or not np.isfinite(args.lr) or args.lr <= 0
            or not np.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 1800):
        parser.error('invalid experiment size, learning rate or budget')
    if args.device == 'cpu':
        if torch.cuda.is_initialized():
            raise RuntimeError('fresh process required for CPU isolation')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if digest(args.gameplay_baseline) != args.gameplay_baseline_sha256:
        raise ValueError('gameplay baseline SHA256 differs')
    if digest(args.parent) != args.parent_sha256:
        raise ValueError('parent SHA256 differs')
    if bool(args.dynamics_parent) != bool(args.dynamics_sha256):
        parser.error('repaired dynamics checkpoint and SHA256 must be supplied together')
    if args.dynamics_parent and digest(args.dynamics_parent) != args.dynamics_sha256:
        raise ValueError('dynamics SHA256 differs')
    if bool(args.continuation_parent) != bool(args.continuation_sha256):
        parser.error('continuation checkpoint and SHA256 must be supplied together')
    if args.continuation_parent and digest(args.continuation_parent) != args.continuation_sha256:
        raise ValueError('continuation SHA256 differs')
    args.out.mkdir(parents=True, exist_ok=False)
    guard = Budget(args.max_seconds, reserve_gib=7)
    torch.set_num_threads(1)
    report = dict(format='pebby.neural-imagination-experiment.v1', status='running', pid=os.getpid(),
                  started_local=datetime.now().astimezone().isoformat(), args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  arms={}, official_training_inputs=False, selector='neural', promoted=False,
                  evaluation_priority='actual_sequential_gameplay', offline_metrics_select_checkpoint=False,
                  checkpoint_selection='Fixed final update per arm; actual gameplay comparison; no automatic promotion',
                  limits=['Frozen learned dynamics initialized for selector comparison; not a new dynamics fit.',
                          'Legacy generated field bank; this is not confirmation or a full-game score.',
                          'No persistent game memory or learned voluntary RESET.',
                          'Dynamics sequence training lacks resets/terminal failures; imagined terminal tails are unqualified.',
                          'Hard continuation actions trained by separate real-state supervision, not selector gradients.',
                          'Tail intervention measures reliance; a score change alone is not evidence of useful planning.'])
    write_json(args.out / 'report.json', report)
    print(json.dumps(dict(event='started', pid=os.getpid())), flush=True)
    data = {}
    try:
        parent, _ = load_factored_policy_checkpoint(args.parent, 'cpu')
        sources = {str(args.parent.resolve()):args.parent_sha256, **parent.sources['code_hashes']}
        save_policy, reload_policy, checkpoint_parent = save_checkpoint, load_checkpoint, args.parent
        if args.dynamics_parent:
            from pebby.agent import repaired_imagination_policy as repaired
            adapted, dynamics_metadata = repaired.load_dynamics_parent(
                args.dynamics_parent, 'cpu', expected_sha256=args.dynamics_sha256)
            if state_digest(adapted.encoder.state_dict()) != state_digest(parent.encoder.state_dict()):
                raise ValueError('repaired dynamics must use the same public encoder')
            parent = adapted
            sources.update(parent.sources['code_hashes'])
            sources[str(args.dynamics_parent.resolve())] = args.dynamics_sha256
            report['dynamics_repair'] = dynamics_metadata
            report['limits'][0] = 'Fresh neural selectors over separately repaired, frozen learned dynamics.'
            report['limits'][3] = 'Reset and ending coverage comes from the dynamics repair report; post-terminal tails remain unqualified.'
            save_policy, reload_policy, checkpoint_parent = repaired.save_checkpoint, repaired.load_checkpoint, args.dynamics_parent
        for name in ('neural_imagination.py', 'neural_imagination_policy.py', 'planning_decision_metrics.py', 'outcome_ordering.py'):
            path = Path('pebby/agent') / name
            sources[str(path.resolve())] = digest(path)
        for path in [Path(__file__), Path('tools/train_structured_policy.py'),
                     Path('tools/train_structured_transition.py'), Path('tools/train_navigation_probe.py'),
                     Path('pebby/agent/gameplay_gate.py'), Path('tools/evaluate_navigation_probe.py'),
                     Path('pebby/agent/spatial_outcome_policy.py'), Path('pebby/agent/world_features.py')]:
            sources[str(path.resolve())] = digest(path)
        for split in ('train', 'validation'):
            guard()
            data[split], manifest = load_policy_cache(args.cache / split, split)
            check_policy_encoder(manifest['field_encoder'], parent, True)
            sources[str((args.cache / split / 'manifest.json').resolve())] = digest(args.cache / split / 'manifest.json')
            sources.update({str((args.cache / split / (name + '.npy')).resolve()):info['sha256']
                            for name, info in manifest['arrays'].items()})
        if np.intersect1d(data['train']['seeds'], data['validation']['seeds']).size:
            raise ValueError('training/validation level overlap')
        sources[str(args.gameplay_baseline.resolve())] = args.gameplay_baseline_sha256
        report['sources'] = sources
        from pebby.agent.gameplay_gate import evaluate_sequential, assess_gameplay
        from tools.evaluate_navigation_probe import load_policy
        baseline_policy, _ = load_policy(args.gameplay_baseline, args.device)
        report['gameplay_baseline'] = evaluate_sequential(baseline_policy, args.device, guard, per_level_cap=300)
        del baseline_policy
        write_json(args.out / 'report.json', report)
        rows = stratified_rows(data['validation']['difficulties'], args.validation_count, args.seed)
        report['validation_rows'] = rows.tolist()
        report['validation_tiers'] = {str(tier): int((data['validation']['difficulties'][rows] == tier).sum())
                                      for tier in np.unique(data['validation']['difficulties'])}
        frozen_digest = state_digest(parent.dynamics.state_dict())
        # One shared supervised continuation fit avoids horizon-dependent GPU
        # numerical drift and keeps the comparison's imagined action policy fixed.
        torch.manual_seed(args.seed)
        warm = NeuralImaginationPolicy(copy.deepcopy(parent), dict(horizon=1)).to(args.device)
        warm.train()
        if args.continuation_parent:
            from pebby.agent.structured_policy import load_structured_policy_checkpoint
            continuation_parent, _ = load_structured_policy_checkpoint(args.continuation_parent, 'cpu')
            if (continuation_parent.cfg.mode != 'direct'
                    or state_digest(continuation_parent.encoder.state_dict()) != state_digest(parent.encoder.state_dict())):
                raise ValueError('continuation must be a direct policy over the identical public encoder')
            warm.planner.continuation.load_state_dict(continuation_parent.readout.state_dict(), strict=True)
            sources[str(args.continuation_parent.resolve())] = args.continuation_sha256
            report['continuation_initialization'] = dict(checkpoint=str(args.continuation_parent), sha256=args.continuation_sha256,
                                                        frozen_for_selector_comparison=True)
            del continuation_parent
        warm_optimizer = torch.optim.AdamW(warm.planner.continuation.parameters(), lr=args.lr)
        warm_rng = np.random.default_rng(args.seed + 100)
        report['continuation_pretraining'] = []
        for step in range(0 if args.continuation_parent else args.updates):
            guard()
            selected = warm_rng.choice(len(data['train']['seeds']), size=args.batch_size,
                                       replace=args.batch_size > len(data['train']['seeds']))
            actions = torch.as_tensor(warm_rng.integers(0, 4, args.batch_size), device=args.device)
            items = batch(data['train'], selected, args.device)
            warm_optimizer.zero_grad(set_to_none=True)
            loss = continuation_loss(warm.planner, items, actions)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite continuation loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(warm.planner.continuation.parameters(), 1., error_if_nonfinite=True)
            warm_optimizer.step()
            report['continuation_pretraining'].append(dict(step=step + 1, loss=float(loss.detach()), rows=selected.tolist(), actions=actions.tolist()))
            if (step + 1) % 25 == 0:
                print(json.dumps(dict(event='continuation_update', step=step + 1, loss=float(loss.detach()))), flush=True)
        continuation_state = {k:v.detach().cpu().clone() for k,v in warm.planner.continuation.state_dict().items()}
        del warm, warm_optimizer
        for horizon in args.horizons:
            guard()
            torch.manual_seed(args.seed)
            policy = NeuralImaginationPolicy(copy.deepcopy(parent), dict(horizon=horizon)).to(args.device)
            policy.planner.continuation.load_state_dict(continuation_state)
            policy.planner.continuation.requires_grad_(False)
            selector_parameters = [p for name, p in policy.planner.named_parameters()
                                   if p.requires_grad and not name.startswith('continuation.')]
            optimizer = torch.optim.AdamW(selector_parameters, lr=args.lr)
            rng = np.random.default_rng(args.seed)
            record = dict(status='running', steps=[], sampled_rows=[], horizon=horizon,
                          initial_state_sha256=state_digest(policy.planner.state_dict()))
            report['arms'][str(horizon)] = record
            policy.train()
            for step in range(args.updates):
                guard()
                selected = rng.choice(len(data['train']['seeds']), size=args.batch_size,
                                      replace=args.batch_size > len(data['train']['seeds']))
                items = batch(data['train'], selected, args.device)
                optimizer.zero_grad(set_to_none=True)
                loss = policy_terms(policy.planner(items['fields'].float()), items['optimal'])['ce'].mean()
                parts = dict(selector=float(loss.detach()))
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(selector_parameters, 1., error_if_nonfinite=True)
                optimizer.step()
                record['steps'].append(dict(step=step + 1, **parts, gradient_norm=float(norm)))
                record['sampled_rows'].append(selected.tolist())
                if (step + 1) % 25 == 0:
                    write_json(args.out / 'report.json', report)
                    print(json.dumps(dict(event='update', horizon=horizon, step=step + 1, **parts)), flush=True)
            record['gameplay'] = evaluate_sequential(policy, args.device, guard, per_level_cap=300)
            record['gameplay_gate'] = assess_gameplay(report['gameplay_baseline'], record['gameplay'])
            print(json.dumps(dict(event='gameplay_result', horizon=horizon,
                                  levels_completed=record['gameplay']['levels_completed'],
                                  gate=record['gameplay_gate']['status'])), flush=True)
            write_json(args.out / 'report.json', report)
            if args.offline_diagnostics:
                record['offline_diagnostics'] = evaluate(policy.planner, data['validation'], rows, args.device, args.batch_size, guard)
            if state_digest(policy.planner.dynamics.state_dict()) != frozen_digest:
                raise RuntimeError('frozen dynamics changed')
            checkpoint = args.out / f'k{horizon}.pt'
            metadata = dict(official_training_inputs=False, source='generated_only',
                            horizon=horizon, updates=args.updates, seed=args.seed, sources=sources,
                            confirmation_used=False, experiment_limits=report['limits'], promoted=False,
                            selection_metric='actual_sequential_gameplay',
                            gameplay=record['gameplay'], gameplay_gate=record['gameplay_gate'],
                            offline_metrics_select_checkpoint=False)
            save_policy(checkpoint, policy, checkpoint_parent, metadata)
            reloaded, _ = reload_policy(checkpoint, args.device)
            with torch.no_grad():
                first = batch(data['validation'], rows[:1], args.device)['fields'].float()
                torch.testing.assert_close(policy.planner(first), reloaded.planner(first), rtol=0, atol=0)
            record.update(status='complete', checkpoint=str(checkpoint.resolve()), checkpoint_sha256=digest(checkpoint),
                          dynamics_unchanged=True, strict_reload_equal=True,
                          continuation_sha256=state_digest(policy.planner.continuation.state_dict()))
            write_json(args.out / 'report.json', report)
            del policy, optimizer, reloaded
        if len({arm['initial_state_sha256'] for arm in report['arms'].values()}) != 1:
            raise RuntimeError('horizon arms did not share initialization')
        if len({arm['continuation_sha256'] for arm in report['arms'].values()}) != 1:
            raise RuntimeError('horizon arms did not learn identical continuation policies')
        for path, expected in sources.items():
            guard()
            if digest(path) != expected:
                raise ValueError('source changed during training: ' + path)
        report.update(status='complete', sources_unchanged=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        for arrays in data.values():
            for array in arrays.values():
                if getattr(array, '_mmap', None) is not None:
                    array._mmap.close()
        report.update(elapsed_seconds=time.monotonic() - guard.started,
                      finished_local=datetime.now().astimezone().isoformat())
        write_json(args.out / 'report.json', report)
    return report


if __name__ == '__main__':
    main()
