"""Bounded fresh joint-model learnability run, judged by actual gameplay.

Eight generated TRAIN games qualify wiring/learning, not generalization. H1
transition labels train dynamics while the K4 selector uses its own predicted
futures. No frozen parent, stored route, or offline checkpoint selection.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from pebby.agent import evaluate, gameplay_gate
from pebby.agent.joint_goal_planning import JointGoalPlanning, save_checkpoint, load_checkpoint
from pebby.agent.joint_goal_objective import joint_goal_loss
from pebby.agent.neural_outcome_policy import weights_sha256
from pebby.ls20.env import Ls20Scenario
from pebby.ls20.generate import build_level
from pebby.ls20.provenance import generated_context
from tools.train_navigation_probe import Budget, digest, write_json

ROOT = Path(__file__).resolve().parents[1]


def load_data(directory):
    from pebby.agent.joint_goal_data import load_dataset
    return load_dataset(directory)


def gameplay(model, specs, device, guard, cap=100):
    """Real engine episodes; exact targets and teacher routes are not inputs."""
    model.eval()
    runs = []
    # evaluate.rollout owns public causal history and strict four-logit argmax.
    class Checked:
        def __getattr__(self, key):
            return getattr(model, key)

        def __call__(self, *args, **kwargs):
            guard()
            result = model(*args, **kwargs)
            if result.shape != (len(args[0]), 4) or not bool(torch.isfinite(result).all()):
                raise ValueError('gameplay requires finite logits[B,4]')
            guard()
            return result

    class CheckedEnvironment:
        def __init__(self, env):
            self.env = env

        def __getattr__(self, key):
            return getattr(self.env, key)

        def reset(self):
            guard()
            return self.env.reset()

        def perform(self, action):
            guard()
            result = self.env.perform(action)
            if result.frame is None and not result.finished:
                raise ValueError('live qualification transition has no public frame')
            guard()
            return result

    for spec in specs:
        guard()
        env = CheckedEnvironment(Ls20Scenario(build_level(spec), generated_context(spec)))
        result = evaluate.rollout(Checked(), env, cap, torch.device(device),
                                  oracle_length=None, on_stall='repeat', temperature=0.)
        guard()
        runs.append(dict(seed=spec['seed'], **result))
    return dict(completed=sum(run['completed'] for run in runs), levels=len(runs),
                actions=sum(run['actions'] for run in runs), runs=runs,
                training_games=True, generalization_evidence=False, cap=cap)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=.0003)
    parser.add_argument('--without-relation-feedback', action='store_true')
    parser.add_argument('--mechanics-only', action='store_true', help='disposable one-update backward check; no learning claim or checkpoint')
    parser.add_argument('--max-seconds', type=int, default=1200)
    args = parser.parse_args(argv)
    if (not 1 <= args.updates <= 2000 or not 1 <= args.batch_size <= 32 or args.eval_every < 1
            or not 1 <= args.max_seconds <= 1800 or not np.isfinite(args.lr) or not 0 < args.lr < .1):
        parser.error('invalid bounded qualification settings')
    if args.device == 'cpu':
        if torch.cuda.is_initialized():
            raise RuntimeError('fresh CPU process required')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        from tools.train_reference_outcomes import gpu_available
        gpu_available()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    guard = Budget(args.max_seconds, reserve_gib=7)
    started = time.monotonic()
    report = dict(status='running', pid=os.getpid(), started_local=datetime.now().astimezone().isoformat(),
        format='pebby.joint-goal-learning-qualification.v1', args={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        from_random_initialization=True, frozen_parameters=0, pretrained_checkpoints_loaded=[],
        official_training_inputs=False, promoted=False, qualification_passed=False,
        offline_metrics_used_for_selection=False, actual_transition_horizon=1, imagined_horizon=4,
        limits=['Eight TRAIN-game completion is learnability qualification, not seven-level or generalization success.',
                'Dynamics has H1 targets; H2+ receive policy gradients but no exact transition supervision in this first bank.',
                'No persistent cross-life/RESET memory; no learned voluntary RESET.',
                'No old cached field targets; all deployed components are freshly initialized and trainable.',
                'Trainability is component-level: inherited dynamics contains discarded output channels and cancelling scalar biases.',
                'Shipped gameplay is exposed development evidence; no automatic checkpoint promotion.'],
        training=[], gameplay=[])
    def persist():
        write_json(args.out / 'report.json', report)
    def timeout(*_):
        raise TimeoutError('joint planning qualification hard deadline')
    old_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.max_seconds)
    print(json.dumps(dict(event='started', pid=os.getpid())), flush=True)
    persist()
    try:
        guard()
        arrays, specs, manifest = load_data(args.data)
        from arcengine import base_game
        sources = [Path(__file__), Path(base_game.__file__), ROOT / 'third_party/ls20/ls20.py',
                   ROOT / 'tools/train_navigation_probe.py', ROOT / 'tools/train_reference_outcomes.py',
                   *sorted((ROOT / 'pebby/agent').glob('*.py')), *sorted((ROOT / 'pebby/ls20').glob('*.py')),
                   args.data / 'manifest.json', args.data / 'data.npz', args.data / 'bank.json']
        bindings = {str(path.resolve()): digest(path) for path in sources}
        for path,expected in manifest['sources'].items():
            if digest(path) != expected:
                raise ValueError('dataset producer source changed before training: ' + path)
            bindings[path] = expected
        report.update(source_sha256=bindings, dataset_rows=len(arrays['seeds']), dataset_manifest=manifest)
        torch.manual_seed(args.seed)
        model = JointGoalPlanning(dict(relation_feedback=not args.without_relation_feedback)).to(args.device)
        report.update(parameter_counts=model.parameter_counts(), initial_weights_sha256=weights_sha256(model.state_dict()))
        report['initial_component_sha256'] = {name:weights_sha256(module.state_dict()) for name,module in
            (('encoder', model.encoder), ('dynamics', model.dynamics), ('continuation', model.continuation))}
        if any(not parameter.requires_grad for parameter in model.parameters()):
            raise ValueError('fresh model unexpectedly contains frozen parameters')
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
        rng = np.random.default_rng(args.seed)
        pools = {spec['seed']: np.flatnonzero(arrays['seeds'] == spec['seed']) for spec in specs}
        if any(len(pool) == 0 for pool in pools.values()):
            raise ValueError('every qualification level needs training rows')
        seeds = list(pools)
        selection_hash = hashlib.sha256()
        best = None
        total_updates = 1 if args.mechanics_only else args.updates
        for step in range(1, total_updates + 1):
            guard()
            model.train()
            chosen = rng.choice(seeds, size=args.batch_size, replace=True)
            rows = np.array([rng.choice(pools[int(seed)]) for seed in chosen], dtype=np.int64)
            selection_hash.update(rows.tobytes())
            batch = {name: torch.as_tensor(value[rows], device=args.device) for name,value in arrays.items()}
            optimizer.zero_grad(set_to_none=True)
            result = joint_goal_loss(model, batch)
            result['total'].backward()
            missing = [name for name,p in model.named_parameters() if p.grad is None]
            if missing or any(not bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None):
                raise ValueError('missing/nonfinite gradients: ' + str(missing))
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if step % 25 == 0 or step == total_updates:
                entry = dict(step=step, total=float(result['total'].detach()),
                    losses={key:float(value.detach()) for key,value in result['losses'].items()},
                    gradient_norm=float(norm), elapsed_seconds=time.monotonic()-started)
                report['training'].append(entry)
                print(json.dumps(dict(event='training', **entry)), flush=True)
                persist()
            del batch, result
            if not args.mechanics_only and (step % args.eval_every == 0 or step == total_updates):
                result = gameplay(model, specs, args.device, guard)
                entry = dict(step=step, weights_sha256=weights_sha256(model.state_dict()),
                             sampling_sha256=selection_hash.hexdigest(), **result)
                report['gameplay'].append(entry)
                if best is None or result['completed'] > best['completed']:
                    best = dict(step=step, completed=result['completed'], gameplay=result,
                                weights={name:p.detach().cpu().clone() for name,p in model.state_dict().items()})
                print(json.dumps(dict(event='actual_training_gameplay', step=step, completed=result['completed'], levels=8)), flush=True)
                persist()
        report['sampling_sha256'] = selection_hash.hexdigest()
        report['final_weights_sha256'] = weights_sha256(model.state_dict())
        report['final_component_sha256'] = {name:weights_sha256(module.state_dict()) for name,module in
            (('encoder', model.encoder), ('dynamics', model.dynamics), ('continuation', model.continuation))}
        report['peak_gpu_bytes'] = torch.cuda.max_memory_allocated() if args.device == 'cuda' else 0
        if any(report['initial_component_sha256'][name] == value
               for name,value in report['final_component_sha256'].items()):
            raise ValueError('a deployed learned component did not update')
        if report['final_weights_sha256'] == report['initial_weights_sha256']:
            raise ValueError('no learned parameter updates')
        if args.mechanics_only:
            report['mechanics_passed'] = True
        else:
            model.load_state_dict(best.pop('weights'))
            model.eval()
            report['selected'] = best
            report['qualification_passed'] = best['completed'] == 8
            report['sequential'] = gameplay_gate.evaluate_sequential(model, args.device, guard, per_level_cap=300)
            report['objective_complete'] = report['sequential']['levels_completed'] == 7
            metadata = dict(source_sha256=bindings, training_report=str((args.out / 'report.json').resolve()),
                official_training_inputs=False, from_random_initialization=True, frozen_parameters=0,
                actual_transition_horizon=1, selected_on='actual_generated_TRAIN_gameplay',
                selected_step=best['step'], qualification_passed=report['qualification_passed'],
                gameplay=report['sequential'], promoted=False)
            path = args.out / 'candidate.pt'
            if any(digest(path) != expected for path,expected in bindings.items()):
                raise ValueError('source/data changed before checkpoint publication')
            save_checkpoint(path, model, metadata)
            report['checkpoint_sha256'] = digest(path)
            reloaded, _ = load_checkpoint(path, args.device)
            if weights_sha256(model.state_dict()) != weights_sha256(reloaded.state_dict()):
                raise ValueError('checkpoint reload weights differ')
            del reloaded
        if any(digest(path) != expected for path,expected in bindings.items()):
            raise ValueError('source/data changed during fit')
        report.update(status='complete', sources_unchanged=True)
    except BaseException as error:
        (args.out / 'candidate.pt').unlink(missing_ok=True)
        report.update(status='failed', qualification_passed=False, objective_complete=False,
                      sources_unchanged=False, error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        report.update(finished_local=datetime.now().astimezone().isoformat(), elapsed_seconds=time.monotonic()-started)
        persist()
    return report


if __name__ == '__main__':
    main()
