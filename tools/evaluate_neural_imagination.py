"""CPU-only generated-monitor gameplay and initial imagined-trace fidelity.

All imagined actions are fixed before real branches run. Engine state is read
only for diagnostic labels. No Oracle, teacher route or training is invoked.
"""
import argparse
from collections import Counter
from datetime import datetime
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from pebby.agent.evaluate import rollout
from tools.evaluate_navigation_probe import load_policy as load_checkpoint
from pebby.agent.structured_transition import steps_targets
from pebby.agent.world_data import clone_env, history_arrays
from pebby.ls20 import names
from pebby.ls20.env import Ls20Scenario
from tools.evaluate_structured_workspace_gameplay import BANK, BANK_SHA, checked_bank
from tools.train_navigation_probe import Budget, digest, write_json
from tools.train_neural_imagination import stratified_rows


EVENTS = ('lost_life', 'terminal', 'won')
STATE_KEYS = ('player', 'glyph', 'budget_class', 'lives')


def replay_fixed_actions(env, actions, guard=lambda: None):
    """Replay an already selected trace, including resets until true terminal.

    A life loss is a valid transition and does not end a trace while lives
    remain. Only entries after actual WIN/GAME_OVER are missing targets.
    """
    supplied = tuple(actions)
    if not supplied or any(isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer))
                           or action not in range(4) for action in supplied):
        raise ValueError('a nonempty trace of action indices 0..3 is required')
    trace = tuple(map(int, supplied))
    branch = clone_env(env)
    records = []
    for action in trace:
        guard()
        old_lives = branch.lives()
        observation = branch.perform(names.ACTION_IDS[action])
        cell = branch.player_cell()
        steps = int(branch.steps_left())
        records.append(dict(action=action, player=int(cell[1]) * 12 + int(cell[0]),
                            glyph=list(map(int, branch.triple())), steps=steps,
                            budget_class=int(steps_targets(torch.tensor(steps))),
                            lives=int(branch.lives()), lost_life=branch.lives() < old_lives,
                            terminal=bool(observation.finished), won=bool(observation.won)))
        if observation.finished:
            break
    return records + [None] * (len(trace) - len(records))


@torch.inference_mode()
def initial_fidelity(policy, env, guard=lambda: None):
    frame = env.reset()
    public = history_arrays([frame], [-1], 8)
    field = policy.encoder(*(torch.as_tensor(value)[None] for value in public))
    guard()
    imagination = policy.planner.imagine(field)
    actions = imagination['imagined_actions'][0].detach().clone()
    roots = imagination['root_actions'][0].tolist()
    # Same four fixed hard-action traces; no actual future input enters D.
    predicted = policy.planner.dynamics.rollout(field.expand(4, -1, -1), actions)
    readout, glyph = predicted['readout'], predicted['glyph_logits']
    states = dict(player=readout['player_logits'].argmax(-1),
                  budget_class=readout['steps_logits'].argmax(-1),
                  lives=readout['lives_logits'].argmax(-1),
                  glyph=torch.stack([glyph[key].argmax(-1) for key in ('shape', 'color', 'rotation')], -1))
    probabilities = {key: predicted['events'][key + '_logits'].sigmoid() for key in EVENTS}
    branches = []
    for index, trace in enumerate(actions.tolist()):
        actual = replay_fixed_actions(env, trace, guard)
        steps = []
        for horizon, truth in enumerate(actual):
            estimate = {key: value[index, horizon].tolist() for key, value in states.items()}
            event = {key: float(value[index, horizon]) for key, value in probabilities.items()}
            steps.append(dict(valid=truth is not None, actual=truth, predicted=estimate, event_probability=event))
        branches.append(dict(root_action=roots[index], actions=trace, steps=steps))
    return dict(selected_action=int(imagination['action_logits'][0].argmax()), branches=branches)


def fidelity_metrics(cases):
    horizons = max((len(branch['steps']) for case in cases for branch in case['branches']), default=0)
    metrics = []
    for horizon in range(horizons):
        rows = [branch['steps'][horizon] for case in cases for branch in case['branches']
                if horizon < len(branch['steps']) and branch['steps'][horizon]['valid']]
        states = {key: dict(correct=sum(row['predicted'][key] == row['actual'][key] for row in rows),
                            count=len(rows)) for key in STATE_KEYS}
        for value in states.values():
            value['accuracy'] = value['correct'] / len(rows) if rows else None
        events = {}
        for key in EVENTS:
            truth = [row['actual'][key] for row in rows]
            probabilities = [row['event_probability'][key] for row in rows]
            events[key] = dict(count=len(rows), positives=sum(truth),
                               tp=sum(t and p >= .5 for t, p in zip(truth, probabilities)),
                               fp=sum(not t and p >= .5 for t, p in zip(truth, probabilities)),
                               fn=sum(t and p < .5 for t, p in zip(truth, probabilities)),
                               brier=sum((p - int(t)) ** 2 for t, p in zip(truth, probabilities)) / len(rows) if rows else None)
        metrics.append(dict(horizon=horizon + 1, state=states, events=events))
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--levels', type=int, default=10)
    parser.add_argument('--max-actions', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-seconds', type=int, default=180)
    args = parser.parse_args(argv)
    if not 5 <= args.levels <= 100 or args.levels % 5 or not 1 <= args.max_actions <= 100 or not 1 <= args.max_seconds <= 180:
        parser.error('levels must be 5..100 in multiples of 5; actions 1..100; seconds 1..180')
    if torch.cuda.is_initialized():
        raise RuntimeError('fresh CPU process required')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    if digest(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError('checkpoint SHA256 differs')
    args.out.mkdir(parents=True, exist_ok=False)
    guard = Budget(args.max_seconds, reserve_gib=7)
    report = dict(format='pebby.neural-imagination-evaluation.v1', status='running', pid=os.getpid(),
                  started_local=datetime.now().astimezone().isoformat(), device='cpu', cpu_threads=1,
                  checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=args.checkpoint_sha256,
                  bank=str(BANK), bank_sha256=BANK_SHA, source='generated_only', split='validation',
                  official_inputs_used=False, training_performed=False, cases=[],
                  max_actions=args.max_actions, selection_seed=args.seed, on_stall='repeat', temperature=0.,
                  limits=['Reused generator-v2 development monitor: legacy difficulties 1..5, not seven-tier confirmation.',
                          'Fidelity roots are initial observations only; later on-policy histories are not audited.',
                          'No route or exact search is used; stored optima are reporting metadata only.',
                          'Glyph predictions use the global carried-glyph head; other state predictions use learned readouts.',
                          'Budget class 43 groups negative budgets; exact negative-budget magnitude is not predicted.',
                          'Post-terminal actual targets are masked; post-life-loss transitions remain and are unqualified by training.',
                          'Independent event probabilities are uncalibrated; zero-positive strata cannot establish event recall.',
                          'Partial interrupted gameplay is excluded from completion totals.'])
    source_paths = [Path(__file__), Path('tools/train_neural_imagination.py'), Path('tools/train_navigation_probe.py'),
                    Path('tools/evaluate_structured_workspace_gameplay.py'), Path('pebby/agent/evaluate.py'),
                    Path('pebby/agent/history.py'), Path('pebby/agent/world_data.py'), Path('pebby/agent/structured_transition.py'),
                    Path('pebby/agent/neural_imagination.py'), Path('pebby/agent/neural_imagination_policy.py'),
                    Path('pebby/agent/repaired_imagination_policy.py'), Path('tools/evaluate_navigation_probe.py'),
                    Path('pebby/ls20/env.py'), Path('pebby/ls20/names.py'), Path('pebby/ls20/generate.py'),
                    Path('pebby/ls20/bank.py'), Path('third_party/ls20/ls20.py')]
    sources = {str(path.resolve()): digest(path) for path in source_paths}
    report['sources'] = sources
    def persist():
        report['elapsed_seconds'] = time.monotonic() - guard.started
        write_json(args.out / 'report.json', report)
    def timeout(*_):
        raise TimeoutError('generated evaluation wall-clock budget exhausted')
    old_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.max_seconds)
    hook = None
    print(f'PID {os.getpid()}', flush=True)
    persist()
    try:
        policy, _ = load_checkpoint(args.checkpoint, 'cpu')
        levels, optima, specs = checked_bank()
        selected = stratified_rows(np.asarray([spec['difficulty'] for spec in specs]), args.levels, args.seed)
        report['selected_rows'] = selected.tolist()
        report['difficulty_counts'] = dict(Counter(str(specs[int(row)]['difficulty']) for row in selected))
        # Budget check on every closed-loop policy call; no action modification.
        hook = policy.register_forward_pre_hook(lambda *_: guard())
        for index in selected:
            guard()
            spec = specs[int(index)]
            env = Ls20Scenario(levels[int(index)], spec['training_context_index'])
            case = dict(seed=spec['seed'], difficulty=spec['difficulty'], fog=bool(spec.get('fog')),
                        context_index=spec['training_context_index'], status='fidelity_running')
            report['cases'].append(case)
            case['fidelity'] = initial_fidelity(policy, env, guard)
            case['status'] = 'gameplay_running'
            persist()
            case['gameplay'] = rollout(policy, env, args.max_actions, 'cpu', optima[int(index)], on_stall='repeat', temperature=0.)
            case['status'] = 'complete'
            report['fidelity_by_horizon'] = fidelity_metrics([item['fidelity'] for item in report['cases'] if 'fidelity' in item])
            games = [item['gameplay'] for item in report['cases'] if item['status'] == 'complete']
            report['gameplay_summary'] = dict(levels=len(games), completed=sum(game['completed'] for game in games),
                                             goals_cleared=sum(game['goals_cleared'] for game in games),
                                             goals_total=sum(game['goals_total'] for game in games),
                                             actions=sum(game['actions'] for game in games))
            persist()
        for path, expected in {**sources, str(args.checkpoint): args.checkpoint_sha256, str(BANK): BANK_SHA}.items():
            guard()
            if digest(path) != expected:
                raise ValueError('evaluation source changed: ' + path)
        report.update(status='complete', sources_unchanged=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        if hook is not None:
            hook.remove()
        report['finished_local'] = datetime.now().astimezone().isoformat()
        persist()
    return report


if __name__ == '__main__':
    main()
