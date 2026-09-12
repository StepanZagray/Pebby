"""Bounded independent shortest-path and real-engine checks for generated LS20 rows.

The transition model and BFS use only tools.independent_ls20 (stdlib-only),
transcribed separately from the vendored game. The engine adapter imports the
normal sprite builder for replay; it never invokes the production planner.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.independent_ls20 import ACTIONS, DELTAS, Level, ORIGINAL_TRANSCRIPTION_SHA256, bfs, step


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _model_snapshot(level, state, lives=3):
    return (state[0], state[1:4], state[6],
            tuple(bool(state[4] & (1 << i)) for i in range(level.n_goals)),
            tuple(cell for i, cell in enumerate(level.refills) if not state[5] & (1 << i)),
            tuple(sorted((p['kind'], p['cells'][tick]) for p, tick in zip(level.patrollers, state[7]))), lives)


def _engine_snapshot(env):
    from pebby.ls20 import names
    refills = tuple(sorted(names.pixel_to_cell(sprite.x, sprite.y) for sprite in
                           env.game.current_level.get_sprites_by_tag(names.TAG_STEP_REFILL)))
    patrollers = []
    for patroller in getattr(env.game, names.ATTR_PATROLLERS):
        sprite = patroller._sprite
        kind = next(kind for tag, kind in names.CYCLER_TAGS.items() if tag in sprite.tags)
        patrollers.append((kind, names.pixel_to_cell(sprite.x, sprite.y)))
    return (env.player_cell(), env.triple(), env.steps_left(), tuple(env.goals_solved()),
            refills, tuple(sorted(patrollers)), env.lives())


def check_engine(row, offroute=64):
    """Check each stored-route state and bounded random one-action deviations."""
    from pebby.ls20.env import Ls20Scenario
    from pebby.ls20.generate import build_level
    level = Level(row)
    def fresh():
        return Ls20Scenario(build_level(row), row['training_context_index'])
    env = fresh()
    if env.goal_triples() != [tuple(goal['triple']) for goal in row['goals']]:
        raise ValueError('real engine target order/pairing differs from spec')
    state = level.start_state()
    if _engine_snapshot(env) != _model_snapshot(level, state):
        raise ValueError('real engine initial state differs from independent spec')
    prefixes, outcomes = [state], Counter()
    for index, action in enumerate(row['solution']):
        state, outcome = step(level, state, action)
        observation = env.perform(action)
        outcomes[outcome] += 1
        prefixes.append(state)
        if _engine_snapshot(env) != _model_snapshot(level, state):
            raise ValueError(f'route state mismatch at action {index+1}: {outcome}')
        if observation.won != (outcome == 'won'):
            raise ValueError(f'route win mismatch at action {index+1}')
        if observation.finished and index+1 != len(row['solution']):
            raise ValueError('stored route terminates early')
    if not observation.won or env.lives() != 3 or env.levels_completed != 1:
        raise ValueError('stored route does not win with three lives')
    candidates = [(prefix, action) for prefix in range(len(row['solution'])) for action in ACTIONS]
    rng = random.Random(f'independent-offroute:{row["seed"]}:v1')
    probes = rng.sample(candidates, min(offroute, len(candidates)))
    probe_outcomes = Counter()
    for prefix, action in probes:
        env = fresh()
        for prior in row['solution'][:prefix]:
            env.perform(prior)
        state, outcome = step(level, prefixes[prefix], action)
        observation = env.perform(action)
        label = 'wall_bump' if outcome == 'moved' and state[0] == prefixes[prefix][0] else outcome
        probe_outcomes[label] += 1
        expected = _model_snapshot(level, level.start_state(), lives=2) if outcome == 'died' else _model_snapshot(level, state)
        if _engine_snapshot(env) != expected or observation.won != (outcome == 'won'):
            raise ValueError(f'off-route engine mismatch prefix={prefix} action={action} outcome={outcome}')
    # Random one-action deviations may contain no deaths. Force a separately
    # reported budget exhaustion by bumping a wall from a safe route prefix.
    candidates = []
    for prefix, state in enumerate(prefixes[:-1]):
        if any(state[0] in triggers for triggers, _, _ in level.launchers):
            continue
        for action, (dx, dy) in DELTAS.items():
            if not level.free((state[0][0]+dx, state[0][1]+dy)):
                candidates.append((state[6], prefix, action))
    death_actions = 0
    if candidates:
        _, prefix, action = min(candidates)
        env = fresh()
        for prior in row['solution'][:prefix]:
            env.perform(prior)
        state = prefixes[prefix]
        for _ in range(level.max_steps // level.cost + 2):
            state, outcome = step(level, state, action)
            observation = env.perform(action)
            death_actions += 1
            expected = _model_snapshot(level, level.start_state(), lives=2) if outcome == 'died' else _model_snapshot(level, state)
            if _engine_snapshot(env) != expected or observation.won:
                raise ValueError('controlled budget-exhaustion transition mismatch')
            if outcome == 'died':
                break
        else:
            raise ValueError('controlled budget-exhaustion probe never lost a life')
    return dict(route_actions_checked=len(row['solution']), route_outcomes=dict(outcomes),
                offroute_transitions_checked=len(probes), offroute_outcomes=dict(probe_outcomes),
                controlled_death_transitions=death_actions, controlled_life_losses=int(bool(candidates)),
                engine_win_three_lives=True, mismatches=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path, help='generated single-row JSON fixtures or JSONL banks')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--states', type=int, default=2_000_000)
    parser.add_argument('--seconds', type=float, default=120, help='independent BFS bound per row')
    parser.add_argument('--memory-mib', type=int, default=2048)
    parser.add_argument('--offroute', type=int, default=64)
    parser.add_argument('--plain-bfs', action='store_true', help='disable budget dominance for a separate search comparison')
    args = parser.parse_args(argv)
    if min(args.states, args.seconds, args.memory_mib) <= 0 or args.offroute < 0:
        parser.error('positive state/time/memory bounds and nonnegative offroute count required')
    if args.report.exists():
        raise FileExistsError(args.report)
    memory = args.memory_mib * 1024**2
    hard = resource.getrlimit(resource.RLIMIT_AS)[1]
    resource.setrlimit(resource.RLIMIT_AS, (min(memory, hard) if hard != resource.RLIM_INFINITY else memory, hard))
    report = dict(format='pebby.independent-ls20-audit.v1', status='running', pid=os.getpid(),
                  independence='Model and BFS contain no production planner/layout imports or calls; replay uses upstream engine via existing sprite builder.',
                  original_transcription_sha256=ORIGINAL_TRANSCRIPTION_SHA256,
                  source_hashes={str(p): _hash(p) for p in (Path(__file__).resolve(),
                                 REPO_ROOT/'tools/independent_ls20.py', REPO_ROOT/'third_party/ls20/ls20.py')},
                  memory_limit_bytes=memory, rows=[])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    def persist():
        temporary = args.report.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(args.report)
    failed = False
    for path in args.inputs:
        raw = path.read_bytes()
        values = [json.loads(line) for line in raw.splitlines() if line.strip()] if path.suffix == '.jsonl' else [json.loads(raw)]
        for row in values:
            result = dict(seed=row['seed'], difficulty=row['difficulty'], input_path=str(path.resolve()),
                          input_sha256=hashlib.sha256(raw).hexdigest(), stored_actions=row['optimal_actions'])
            try:
                result['engine'] = check_engine(row, args.offroute)
                result['search'] = bfs(Level(row), limit=args.states, seconds=args.seconds,
                                       budget_dominance=not args.plain_bfs)
                result['minimum_proved'] = (result['search']['complete']
                                            and result['search']['minimum_actions'] == row['optimal_actions'])
                failed |= not result['minimum_proved']
            except Exception as error:
                failed = True
                result.update(minimum_proved=False, error=f'{type(error).__name__}: {error}')
            report['rows'].append(result)
            persist()
            print(json.dumps(result), flush=True)
    report['source_hashes_unchanged'] = all(_hash(path) == sha for path, sha in report['source_hashes'].items())
    report['inputs_unchanged'] = all(_hash(row['input_path']) == row['input_sha256'] for row in report['rows'])
    failed |= not report['source_hashes_unchanged'] or not report['inputs_unchanged']
    report['status'] = 'failed_closed' if failed else 'complete'
    persist()
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
