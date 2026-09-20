"""Joint raw-pixel K4 training, selected only by actual engine gameplay.

Generated TRAIN shards supply gradients. Disjoint generated validation games
and the exposed shipped sequential session select snapshots, with shipped
completion first. These development games are not an untouched test set.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import tempfile
import time

import numpy as np
import torch

from pebby.agent import gameplay_gate
from pebby.agent.joint_goal_planning import JointGoalPlanning, load_checkpoint, save_checkpoint
from pebby.agent.joint_goal_sequence_objective import LOSS_WEIGHT_PRESETS
from pebby.agent.neural_outcome_policy import weights_sha256
from tools.train_joint_goal_planning import gameplay
from tools.train_navigation_probe import Budget, digest, write_json

ROOT = Path(__file__).resolve().parents[1]


def gameplay_key(sequential, validation):
    """No loss, accuracy, action count or other cached metric can break ties."""
    if (sequential.get('evaluation_kind') != 'actual_sequential_gameplay'
            or sequential.get('source_unchanged') is not True):
        raise ValueError('verified actual sequential gameplay required')
    completed, levels = validation['completed'], validation['levels']
    runs = validation.get('runs', [])
    if (not 0 <= completed <= levels or levels < 1 or len(runs) != levels
            or sum(bool(run['completed']) for run in runs) != completed):
        raise ValueError('complete generated gameplay panel required')
    return int(sequential['levels_completed']), int(completed)


def verify_specs(train, validation):
    for name, specs in (('train', train), ('validation', validation)):
        if {spec['difficulty'] for spec in specs} != set(range(1, 8)):
            raise ValueError(name + ' replay must cover all seven difficulty tiers')
        if any(spec.get('split') != name for spec in specs):
            raise ValueError('explicit generated split required')
        if any(spec.get('source') != 'generated_only' or spec.get('geometry_split') != name for spec in specs):
            raise ValueError('generated provenance and geometry split required')
        if len({spec['seed'] for spec in specs}) != len(specs):
            raise ValueError('duplicate seeds in ' + name)
    if {s['seed'] for s in train} & {s['seed'] for s in validation}:
        raise ValueError('TRAIN and validation seeds overlap')
    # Ignore provenance when checking exact layout duplication across splits.
    def layout_key(spec):
        keys = ('walls', 'start', 'start_triple', 'goals', 'cyclers', 'refills',
                'launchers', 'rails', 'fog', 'step_counter', 'step_cost')
        return json.dumps({k: spec.get(k) for k in keys}, sort_keys=True)
    if {layout_key(s) for s in train} & {layout_key(s) for s in validation}:
        raise ValueError('TRAIN and validation layouts overlap')


def panel_gameplay(model, specs, device, guard, split):
    result = gameplay(model, specs, device, guard, cap=300)
    tiers = {spec['seed']: spec['difficulty'] for spec in specs}
    for run in result['runs']:
        run['difficulty'] = tiers[run['seed']]
    result.update(training_games=split == 'train', generalization_evidence=False,
                  split=split, exposed_development_selection=split == 'validation',
                  per_tier={str(tier): dict(
                      completed=sum(run['completed'] for run in result['runs'] if run['difficulty'] == tier),
                      levels=sum(run['difficulty'] == tier for run in result['runs'])) for tier in range(1, 8)})
    return result


def save_training_state(path, state):
    """Keep optimizer/RNG progress separately from deployable neural weights."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.training-state-',
                                         suffix='.pt', delete=False) as stream:
            temporary = Path(stream.name)
        torch.save(state, temporary)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=1500)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--initialize-from', type=Path,
                        help='explicit warm start; optimizer is always new, recorded as not from scratch')
    parser.add_argument('--initialize-sha256')
    parser.add_argument('--without-terminal-gating', action='store_true',
                        help='fresh-run ablation; warm starts inherit the checkpoint setting')
    parser.add_argument('--mechanics-only', action='store_true')
    parser.add_argument('--loss-weights', choices=tuple(LOSS_WEIGHT_PRESETS), default='default',
                        help="'control' raises the policy/continuation terms and damps the "
                             'reconstruction-style terms that otherwise own the shared encoder gradient')
    parser.add_argument('--grad-clip', type=float, default=1.)
    parser.add_argument('--max-seconds', type=int, default=1200)
    args = parser.parse_args(argv)
    if (not 1 <= args.updates <= 5000 or not 1 <= args.batch_size <= 32
            or not 1 <= args.eval_every <= args.updates or not 1 <= args.max_seconds <= 1800
            or not np.isfinite(args.lr) or not 0 < args.lr < .1
            or not np.isfinite(args.grad_clip) or not 0 < args.grad_clip <= 100.):
        parser.error('invalid bounded training settings')
    if bool(args.initialize_from) != bool(args.initialize_sha256):
        parser.error('warm start requires both checkpoint path and SHA256')
    if args.initialize_from and args.without_terminal_gating:
        parser.error('warm starts inherit terminal gating from their checkpoint')
    if args.device == 'cpu':
        if torch.cuda.is_initialized():
            raise RuntimeError('fresh CPU process required')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        from tools.train_reference_outcomes import gpu_available
        gpu_available()
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise RuntimeError('training requires ownership of the real-time alarm')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    guard = Budget(args.max_seconds, reserve_gib=7)
    started = time.monotonic()
    report = dict(format='pebby.joint-goal-replay-training.v1', status='running', pid=os.getpid(),
                  started_local=datetime.now().astimezone().isoformat(),
                  args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  from_random_initialization=args.initialize_from is None,
                  optimizer_resumed=False, frozen_parameters=0, promoted=False,
                  official_training_inputs=False, seven_level_run_completed=False,
                  actual_transition_horizon=4, imagined_horizon=4,
                  offline_metrics_used_for_selection=False,
                  heldout_losses_are_diagnostic_only=True,
                  selection='shipped sequential levels, then disjoint generated validation wins; earliest ties',
                  limits=['Shipped and generated validation gameplay are exposed development selection data.',
                          'No persistent cross-life or RESET memory; no learned voluntary RESET.',
                          'All components train; inherited discarded output channels are not useful task parameters.',
                          'Fixed collected behavior traces train multi-step dynamics; new on-policy collection may be needed.',
                          'A small seven-tier pilot does not establish adequate training diversity.'],
                  sampler_event_fraction=.25, loss_weights=dict(LOSS_WEIGHT_PRESETS[args.loss_weights]),
                  gradient_clip=args.grad_clip,
                  training=[], gameplay=[], training_snapshots=[])
    def persist():
        write_json(args.out / 'report.json', report)
    def timeout(*_):
        raise TimeoutError('joint replay training hard deadline')
    old_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.max_seconds)
    train = validation = None
    print(json.dumps(dict(event='started', pid=os.getpid())), flush=True)
    persist()
    try:
        from pebby.agent.joint_goal_replay import JointGoalReplay
        from pebby.agent.joint_goal_sequence_objective import joint_goal_sequence_loss
        guard()
        train = JointGoalReplay(args.train, split='train', guard=guard)
        validation = JointGoalReplay(args.validation, split='validation', guard=guard)
        verify_specs(train.specs, validation.specs)
        report.update(train_data=train.metadata, validation_data=validation.metadata)
        from arcengine import base_game
        sources = [Path(__file__), Path(base_game.__file__), ROOT / 'third_party/ls20/ls20.py',
                   ROOT / 'tools/train_joint_goal_planning.py', ROOT / 'tools/train_navigation_probe.py',
                   ROOT / 'tools/train_reference_outcomes.py',
                   *sorted((ROOT / 'pebby/agent').glob('*.py')),
                   *sorted((ROOT / 'pebby/ls20').glob('*.py'))]
        bindings = {str(path.resolve()): digest(path) for path in sources}
        for replay in (train, validation):
            for path, expected in replay.source_bindings.items():
                if digest(path) != expected:
                    raise ValueError('replay binding changed: ' + path)
                bindings[path] = expected
        archives = args.out / 'sources'
        archives.mkdir()
        report['runtime_source_archives'] = {}
        for path, expected in list(bindings.items()):
            if Path(path).suffix != '.py':
                continue
            content = Path(path).read_bytes()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError('runtime source changed before archival: ' + path)
            archived = archives / expected
            if not archived.exists():
                archived.write_bytes(content)
            bindings[str(archived.resolve())] = expected
            report['runtime_source_archives'][path] = str(archived.resolve())
        torch.manual_seed(args.seed)
        if args.initialize_from:
            if digest(args.initialize_from) != args.initialize_sha256:
                raise ValueError('warm-start checkpoint SHA256 mismatch')
            model, parent = load_checkpoint(args.initialize_from, args.device)
            bindings[str(args.initialize_from.resolve())] = args.initialize_sha256
            report['parent_checkpoint'] = dict(path=str(args.initialize_from.resolve()),
                                             sha256=args.initialize_sha256, format=parent['format'])
        else:
            model = JointGoalPlanning(dict(terminal_gating=not args.without_terminal_gating)).to(args.device)
        report['source_sha256'] = bindings
        report['learned_terminal_gating'] = model.cfg.terminal_gating
        if model.cfg.horizon != 4 or any(not p.requires_grad for p in model.parameters()):
            raise ValueError('all deployed components must train with K4 planning')
        report.update(parameter_counts=model.parameter_counts(), initial_weights_sha256=weights_sha256(model.state_dict()))
        components = dict(encoder=model.encoder, dynamics=model.dynamics, continuation=model.continuation)
        report['initial_component_sha256'] = {k: weights_sha256(m.state_dict()) for k, m in components.items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
        rng, sampling = np.random.default_rng(args.seed), hashlib.sha256()
        # Diagnostic only. Training loss falling while held-out loss rises is the
        # recorded failure mode of this recipe, so both are logged. Gameplay
        # alone still selects; see gameplay_key.
        def heldout_batches(replay, seed):
            source = np.random.default_rng(seed)
            return [replay.sample(args.batch_size, source, event_fraction=.25) for _ in range(6)]
        heldout = dict(train=(train, heldout_batches(train, 10 ** 6 + args.seed)),
                       validation=(validation, heldout_batches(validation, 2 * 10 ** 6 + args.seed)))
        def heldout_losses():
            out = {}
            with torch.no_grad():
                for split, (replay, batches) in heldout.items():
                    totals = {}
                    for rows in batches:
                        guard()
                        public, targets = replay.batch(rows, device=args.device)
                        measured = joint_goal_sequence_loss(model, {**public, **targets},
                                                            weights=args.loss_weights)
                        for name, value in measured['losses'].items():
                            totals.setdefault(name, []).append(float(value))
                    out[split] = {name: sum(v) / len(v) for name, v in totals.items()}
            return out
        best = None
        updates = 1 if args.mechanics_only else args.updates
        for step in range(1, updates + 1):
            guard()
            model.train()
            rows = train.sample(args.batch_size, rng, event_fraction=.25)
            sampling.update(np.asarray(rows, dtype=np.int64).tobytes())
            public, targets = train.batch(rows, device=args.device)
            if set(public) != {'frames', 'history_valid', 'previous_actions'} or set(public) & set(targets):
                raise ValueError('replay public/target boundary is invalid')
            optimizer.zero_grad(set_to_none=True)
            result = joint_goal_sequence_loss(model, {**public, **targets}, weights=args.loss_weights)
            result['total'].backward()
            missing = [name for name, p in model.named_parameters() if p.grad is None]
            if missing:
                raise ValueError('missing gradients: ' + str(missing))
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            if step % 25 == 0 or step == updates:
                entry = dict(step=step, total=float(result['total'].detach()),
                             losses={k: float(v.detach()) for k, v in result['losses'].items()},
                             gradient_norm=float(norm), elapsed_seconds=time.monotonic() - started)
                report['training'].append(entry)
                print(json.dumps(dict(event='training', **entry)), flush=True)
                persist()
            del public, targets, result
            if not args.mechanics_only and (step % args.eval_every == 0 or step == updates):
                model.eval()
                sequential = gameplay_gate.evaluate_sequential(model, args.device, guard, per_level_cap=300)
                generated = panel_gameplay(model, validation.specs, args.device, guard, 'validation')
                key = gameplay_key(sequential, generated)
                measured = heldout_losses()
                entry = dict(step=step, selection_key=list(key), sequential=sequential,
                             validation=generated, heldout_losses=measured,
                             weights_sha256=weights_sha256(model.state_dict()))
                report['gameplay'].append(entry)
                if best is None or key > best['key']:
                    best = dict(step=step, key=key, weights={n: p.detach().cpu().clone()
                                                          for n, p in model.state_dict().items()})
                print(json.dumps(dict(event='actual_gameplay', step=step, shipped=key[0],
                                      validation_wins=key[1], validation_levels=len(validation.specs),
                                      heldout_policy=dict(train=round(measured['train']['policy'], 4),
                                                          validation=round(measured['validation']['policy'], 4)))),
                      flush=True)
                # Keep the latest trained weights even when an earlier snapshot
                # remains the gameplay winner. Continuation need not repeat work.
                if any(digest(path) != expected for path, expected in bindings.items()):
                    raise ValueError('bound source/data changed before intermediate snapshot')
                snapshot = args.out / f'step-{step:06d}.pt'
                save_checkpoint(snapshot, model, dict(source_sha256=bindings, step=step, promoted=False,
                    official_training_inputs=False, actual_transition_horizon=4,
                    from_random_initialization=args.initialize_from is None, frozen_parameters=0,
                    sequential_gameplay=sequential, validation_gameplay=generated))
                snapshot_sha = digest(snapshot)
                state_path = args.out / f'step-{step:06d}.training.pt'
                save_training_state(state_path, dict(format='pebby.joint-goal-training-state.v1',
                    step=step, checkpoint=str(snapshot.resolve()), checkpoint_sha256=snapshot_sha,
                    optimizer=optimizer.state_dict(), numpy_rng_json=json.dumps(rng.bit_generator.state),
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state() if args.device == 'cuda' else None,
                    sampler_prefix_sha256=sampling.hexdigest(),
                    best_step=best['step'], best_key=list(best['key']),
                    best_checkpoint=str((args.out / f"step-{best['step']:06d}.pt").resolve()),
                    best_checkpoint_sha256=digest(args.out / f"step-{best['step']:06d}.pt"),
                    train_manifest_sha256=digest(args.train / 'manifest.json'),
                    validation_manifest_sha256=digest(args.validation / 'manifest.json'),
                    training_args=report['args'], source_sha256=bindings))
                report['training_snapshots'].append(dict(step=step, checkpoint=str(snapshot.resolve()),
                    checkpoint_sha256=snapshot_sha, training_state=str(state_path.resolve()),
                    training_state_sha256=digest(state_path), selected_so_far=best['step']))
                persist()
        report.update(sampling_sha256=sampling.hexdigest(), final_weights_sha256=weights_sha256(model.state_dict()),
                      final_component_sha256={k: weights_sha256(m.state_dict()) for k, m in components.items()},
                      peak_gpu_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else 0)
        if any(report['initial_component_sha256'][k] == v for k, v in report['final_component_sha256'].items()):
            raise ValueError('a deployed learned component did not change')
        if args.mechanics_only:
            report['mechanics_passed'] = True
        else:
            # Learning the TRAIN games and transfer are separate questions.
            # An early gameplay tie must not hide what the final fit learned.
            model.eval()
            report['final_train_gameplay'] = panel_gameplay(model, train.specs, args.device, guard, 'train')
            model.load_state_dict(best.pop('weights'))
            model.eval()
            report['selected'] = best
            report['selected_train_gameplay'] = (report['final_train_gameplay'] if best['step'] == updates else
                                                panel_gameplay(model, train.specs, args.device, guard, 'train'))
            report['sequential'] = gameplay_gate.evaluate_sequential(model, args.device, guard, per_level_cap=300)
            if report['sequential']['levels_completed'] != best['key'][0]:
                raise ValueError('selected checkpoint sequential replay did not reproduce')
            report['seven_level_run_completed'] = report['sequential']['levels_completed'] == 7
            if any(digest(path) != expected for path, expected in bindings.items()):
                raise ValueError('bound source/data changed before publication')
            checkpoint = args.out / 'candidate.pt'
            save_checkpoint(checkpoint, model, dict(source_sha256=bindings, training_report=str((args.out / 'report.json').resolve()),
                official_training_inputs=False, from_random_initialization=args.initialize_from is None,
                frozen_parameters=0, actual_transition_horizon=4, selected_step=best['step'],
                selected_on=report['selection'], gameplay=report['sequential'], promoted=False))
            reloaded, _ = load_checkpoint(checkpoint, args.device)
            if weights_sha256(model.state_dict()) != weights_sha256(reloaded.state_dict()):
                raise ValueError('checkpoint reload mismatch')
            del reloaded
            report['checkpoint_sha256'] = digest(checkpoint)
        if any(digest(path) != expected for path, expected in bindings.items()):
            raise ValueError('bound source/data changed during run')
        report.update(status='complete', sources_unchanged=True)
    except BaseException as error:
        (args.out / 'candidate.pt').unlink(missing_ok=True)
        report.update(status='failed', seven_level_run_completed=False, sources_unchanged=False,
                      error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        for replay in (train, validation):
            if replay is not None:
                replay.close()
        report.update(finished_local=datetime.now().astimezone().isoformat(), elapsed_seconds=time.monotonic() - started)
        persist()
    return report


if __name__ == '__main__':
    main()
