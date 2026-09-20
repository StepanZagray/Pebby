"""Label EVERY state along oracle routes, learner rollouts and uncapped recoveries.

Earlier controller data was eight random snapshots per level plus recovery
demonstrations capped at sixteen actions, with every unwinnable state filtered
out. The model therefore never saw a whole route, the end of a route, or a
state it could not win from. This module keeps all of them:

* row_kind 0: every state on the exact verified oracle route, from reset to
  the winning action;
* row_kind 1: every state the learner visits under strict argmax from reset
  until win, game over, an exact public-history repeat or the action cap;
* row_kind 2: for each learner life, an UNCAPPED oracle recovery from the
  first state whose chosen action was not in the optimal set, run to the win.

Unwinnable states are retained exactly as ``world_data._expand`` labels them:
``optimal == 0`` and ``distances == -1`` where no route remains. Rows share
``world_data._row``'s contract so existing objectives consume them unchanged;
``row_kind``, ``trajectory_id``, ``step``, ``chosen_action`` and ``solvable``
are appended. Exact (state, public history) duplicates within a level are
emitted once, credited to whichever trajectory reached them first.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import world_data as wd
from ..ls20 import names
from ..ls20.plan import Oracle
from ..ls20.reference_profiles import SEARCH_LIMITS

FORMAT = 'pebby.ls20-full-route-rows.v1'
ROW_KINDS = {'route': 0, 'learner': 1, 'recovery': 2}
KIND_NAMES = {value: key for key, value in ROW_KINDS.items()}
EXTRA_KEYS = ('row_kind', 'trajectory_id', 'step', 'chosen_action', 'solvable')
ROUTE_TRAJECTORY, LEARNER_TRAJECTORY, FIRST_RECOVERY_TRAJECTORY = 0, 1, 2


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def spec_sha(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def search_limit_for_spec(spec, default=600_000):
    difficulty = spec.get('difficulty')
    if type(difficulty) is int and 1 <= difficulty <= len(SEARCH_LIMITS):
        return SEARCH_LIMITS[difficulty - 1]
    return default


def native_context(spec, search_limit=None):
    """Complete fast-kernel contextual oracle plus a real-engine verified start.

    Mirrors ``tools.collect_spatial_recovery.native_context``: the reference
    Python search is never allowed to substitute for the C kernel, so memory
    stays bounded on the large tier 6-7 state spaces.
    """
    limit = search_limit_for_spec(spec) if search_limit is None else search_limit

    def native(*args, **kwargs):
        return Oracle(*args, **kwargs, engine='fast')
    with patch.object(wd, 'Oracle', native):
        initial, oracle, proof = wd.verified_context(spec, search_limit=limit)
    if (initial is None or oracle is None or oracle.truncated or oracle.engine != 'fast'
            or proof.get('oracle_backend') != 'fast' or proof.get('search_truncated') is not False):
        raise ValueError(f'complete native teacher required: {proof}')
    return initial, oracle, proof


def argmax_choice(policy, observed, valid, previous):
    """Strict argmax over the public H8 input; mirrors diagnose_reference_policy.public_choice."""
    with torch.inference_mode():
        logits = policy(torch.from_numpy(observed[None]).long(),
                        history_valid=torch.from_numpy(valid[None]),
                        previous_actions=torch.from_numpy(previous[None]))
        if tuple(logits.shape) != (1, 4) or not bool(torch.isfinite(logits).all()):
            raise ValueError('invalid public policy logits')
        return int(logits[0].argmax()), logits[0].float().softmax(-1).tolist()


class _Level:
    """Row sink for one level: labels states, dedupes exact keys, steps engines."""

    def __init__(self, oracle, seed, context_index, history):
        self.oracle, self.seed, self.context, self.history = oracle, seed, context_index, history
        self.rows, self.seen, self.duplicates = [], {}, Counter()
        self.emitted_by_trajectory, self.skipped_by_trajectory = Counter(), Counter()

    def public(self, frames, actions):
        return wd.history_arrays(frames, actions, self.history)

    def key(self, env, observed, valid, previous):
        return wd.state_history_key(env, self.oracle, observed, valid, previous)

    def distance(self, env):
        return self.oracle.distance_for(self.oracle.state_of(env))

    def label(self, env, frames, actions, kind, trajectory, step, choice):
        """Emit the row for ``env`` unless its exact key was already emitted.

        Returns ``(mask, expansion)``; ``expansion`` is ``(branches, results)``
        from the four real-engine branches, or None when the row was a
        duplicate and the caller must step the engine itself.
        """
        observed, valid, previous = self.public(frames, actions)
        key = self.key(env, observed, valid, previous)
        before = self.distance(env)
        if key in self.seen:
            self.duplicates[KIND_NAMES[kind]] += 1
            self.skipped_by_trajectory[trajectory] += 1
            return self.seen[key], None
        targets, branches, results, mask = wd._expand(env, self.oracle, before, self.seed, step)
        row = wd._row(targets, observed, valid, previous, self.seed, self.context)
        row.update(row_kind=np.uint8(kind), trajectory_id=np.int64(trajectory), step=np.int64(step),
                   chosen_action=np.int8(choice), solvable=np.bool_(before is not None))
        self.rows.append(row)
        self.seen[key] = mask
        self.emitted_by_trajectory[trajectory] += 1
        return mask, (branches, results)

    def account(self, record):
        """Attach emitted/duplicate row counts to a trajectory proof record."""
        trajectory = record['trajectory_id']
        return {**record, 'rows': int(self.emitted_by_trajectory.get(trajectory, 0)),
                'duplicate_states_skipped': int(self.skipped_by_trajectory.get(trajectory, 0))}

    @staticmethod
    def step(env, choice, expansion):
        """Apply ``choice``; adopt the already-executed branch when available."""
        old_lives = env.lives()
        if expansion is None:
            result = env.perform(names.ACTION_IDS[choice])
        else:
            branches, results = expansion
            env, result = branches[choice], results[choice]
        if result.frame is None:
            raise RuntimeError('a live-state action returned no public successor observation')
        return env, result, env.lives() < old_lives

    def advance_history(self, frames, actions, result, choice, lost_life):
        """Histories never cross a life loss; they restart on the reset frame."""
        if lost_life:
            return [result.frame], [-1]
        return (frames + [result.frame])[-self.history:], (actions + [choice])[-self.history:]


def _walk_route(level, initial, solution):
    """Every state on the verified route, checked against the complete teacher."""
    env, frames, actions = wd.clone_env(initial), [initial.render()], [-1]
    seed, length, start_lives = level.seed, len(solution), initial.lives()
    for index, action in enumerate(solution):
        remaining = length - index
        if level.distance(env) != remaining:
            raise ValueError(f'route state disagrees with the complete teacher at seed {seed}, step {index}')
        choice = names.ACTION_IDS.index(action)
        mask, expansion = level.label(env, frames, actions, ROW_KINDS['route'], ROUTE_TRAJECTORY, index, choice)
        if not mask & (1 << choice):
            raise ValueError(f'route action is not optimal for the complete teacher at seed {seed}, step {index}')
        env, result, lost_life = level.step(env, choice, expansion)
        if lost_life:
            raise ValueError(f'oracle route lost a life at seed {seed}, step {index}')
        if index + 1 < length:
            if result.finished:
                raise ValueError(f'oracle route ended early at seed {seed}, step {index}')
        elif not (result.won and env.lives() == start_lives):
            raise ValueError(f'oracle route did not win in the real engine at seed {seed}')
        frames, actions = level.advance_history(frames, actions, result, choice, lost_life)
    return {'trajectory_id': ROUTE_TRAJECTORY, 'kind': 'route', 'actions': [names.ACTION_IDS.index(a) for a in solution],
            'length': length, 'stop': 'won'}


def _roll_learner(level, initial, policy, learner_lives, learner_cap):
    """Strict-argmax rollout from reset; returns its proof and per-life candidates.

    A candidate is the state from which a recovery should start: the first
    state in that life whose chosen action was outside the optimal set, or,
    if an unwinnable state were somehow met first, the last winnable state
    before it. From a solvable state every action leading to an unsolvable
    one is already outside the mask, so the second trigger is a guard rather
    than the common path.
    """
    env, frames, actions = wd.clone_env(initial), [initial.render()], [-1]
    stats, visited, steps = Counter(), set(), []
    lives, candidates, life = [], [], {'start_step': 0, 'first_mistake': None, 'first_unwinnable': None}
    candidate, last_winnable, stop = None, None, 'cap'
    for step in range(learner_cap):
        observed, valid, previous = level.public(frames, actions)
        key = level.key(env, observed, valid, previous)
        if key in visited:
            # Strict argmax over identical public input and identical logical
            # state repeats forever; nothing new can follow.
            stop = 'exact_attractor_repeat'
            break
        visited.add(key)
        choice, probabilities = argmax_choice(policy, observed, valid, previous)
        before = level.distance(env)
        solvable = before is not None
        if candidate is None and solvable:
            last_winnable = (wd.clone_env(env), list(frames), list(actions), step)
        mask, expansion = level.label(env, frames, actions, ROW_KINDS['learner'], LEARNER_TRAJECTORY, step, choice)
        optimal = bool(mask & (1 << choice)) if solvable else None
        stats['actions'] += 1
        if solvable:
            stats['solvable_decisions'] += 1
            stats['optimal_choices'] += bool(optimal)
        else:
            stats['unwinnable_decisions'] += 1
        if candidate is None:
            if solvable and not optimal:
                life['first_mistake'] = step
                candidate = (*last_winnable, 'suboptimal_choice')
            elif not solvable:
                life['first_unwinnable'] = step
                if last_winnable is not None:
                    candidate = (*last_winnable, 'unwinnable_state')
        elif not solvable and life['first_unwinnable'] is None:
            life['first_unwinnable'] = step
        env, result, lost_life = level.step(env, choice, expansion)
        steps.append({'step': step, 'action': choice, 'optimal': optimal, 'solvable': solvable,
                      'distance': None if before is None else int(before), 'lost_life': bool(lost_life),
                      'probabilities': probabilities})
        stats['life_losses'] += lost_life
        if lost_life or result.finished:
            life.update(end_step=step, lost_life=bool(lost_life), finished=bool(result.finished))
            lives.append(life)
            candidates.append(candidate)
            candidate, last_winnable = None, None
            life = {'start_step': step + 1, 'first_mistake': None, 'first_unwinnable': None}
        if result.finished:
            stop = 'won' if result.won else 'game_over'
            break
        if lost_life and len(lives) >= learner_lives:
            stop = 'lives_exhausted'
            break
        frames, actions = level.advance_history(frames, actions, result, choice, lost_life)
    if life['start_step'] < stats['actions']:
        # The open life made decisions but ended at the cap or on a repeat.
        life.update(end_step=stats['actions'] - 1, lost_life=False, finished=False)
        lives.append(life)
        candidates.append(candidate)
    proof = {'trajectory_id': LEARNER_TRAJECTORY, 'kind': 'learner', 'stop': stop, 'won': stop == 'won',
             'actions': stats['actions'], 'solvable_decisions': stats['solvable_decisions'],
             'optimal_choices': stats['optimal_choices'],
             'optimal_choice_rate': (stats['optimal_choices'] / stats['solvable_decisions']
                                     if stats['solvable_decisions'] else None),
             'unwinnable_decisions': stats['unwinnable_decisions'], 'life_losses': stats['life_losses'],
             'lives': lives, 'steps': steps}
    return proof, candidates


def _recover(level, candidate, trajectory, rng_seed):
    """Uncapped oracle play from ``candidate`` to the real-engine win."""
    env, frames, actions, learner_step, trigger = candidate
    start_distance = level.distance(env)
    if start_distance is None:
        raise ValueError('recovery must start from a winnable state')
    taken, life_losses, step, stop = [], 0, 0, None
    tie_seed = repr((level.seed, rng_seed, trajectory))
    while True:
        before = level.distance(env)
        action = level.oracle.action_at(env, seed=tie_seed)
        if before is None or action is None:
            raise ValueError('recovery teacher became unreachable')
        choice = names.ACTION_IDS.index(action)
        mask, expansion = level.label(env, frames, actions, ROW_KINDS['recovery'], trajectory, step, choice)
        if not mask & (1 << choice):
            raise ValueError('recovery action is not in the complete teacher optimal set')
        env, result, lost_life = level.step(env, choice, expansion)
        taken.append(choice)
        step += 1
        if result.finished:
            if not result.won:
                raise ValueError('teacher recovery finished without winning')
            stop = 'won'
            break
        if lost_life:
            life_losses += 1
        elif level.distance(env) != before - 1:
            raise ValueError('recovery did not reduce exact distance by one')
        if step > (life_losses + 1) * start_distance:
            raise RuntimeError('recovery exceeded the teacher distance without winning')
        frames, actions = level.advance_history(frames, actions, result, choice, lost_life)
    return {'trajectory_id': trajectory, 'kind': 'recovery', 'trigger': trigger, 'learner_step': learner_step,
            'start_distance': int(start_distance), 'actions': taken, 'life_losses': life_losses, 'stop': stop}


def collect_level(spec, policy=None, *, history=8, learner_lives=3, learner_cap=200, recovery=True, rng_seed=0):
    """Rows and proof for one level: full route, learner rollout, uncapped recoveries."""
    if not 1 <= history <= 64 or not 1 <= learner_lives <= 3 or not 1 <= learner_cap <= 10_000:
        raise ValueError('history 1..64, learner_lives 1..3 and learner_cap 1..10000 required')
    if policy is not None and hasattr(policy, 'config') and policy.config().get('history') not in (None, history):
        raise ValueError('policy history length must equal the collected history')
    initial, oracle, verification = native_context(spec)
    seed, context = spec['seed'], verification['context_index']
    level = _Level(oracle, seed, context, history)
    solution = oracle.solution(seed=seed)
    trajectories = [level.account(_walk_route(level, initial, solution))]
    route_rows = len(level.rows)
    learner, candidates = None, []
    if policy is not None:
        if hasattr(policy, 'eval'):
            policy.eval()
        learner, candidates = _roll_learner(level, initial, policy, learner_lives, learner_cap)
        trajectories.append(level.account({key: value for key, value in learner.items() if key != 'steps'}))
    learner_rows = len(level.rows) - route_rows
    if recovery:
        next_trajectory = FIRST_RECOVERY_TRAJECTORY
        for candidate in candidates:
            if candidate is None:
                continue
            trajectories.append(level.account(_recover(level, candidate, next_trajectory, rng_seed)))
            next_trajectory += 1
    rows = level.rows
    counts = Counter(KIND_NAMES[int(row['row_kind'])] for row in rows)
    recoveries = [t for t in trajectories if t['kind'] == 'recovery']
    if not any(bool(row['won'].any()) for row in rows):
        raise ValueError(f'no winning successor recorded at seed {seed}')
    proof = {**verification, 'format': FORMAT, 'seed': seed, 'difficulty': spec.get('difficulty'),
             'fog': bool(spec.get('fog')), 'history': history, 'learner_cap': learner_cap,
             'learner_lives': learner_lives, 'rng_seed': rng_seed, 'policy_used': policy is not None,
             'rows': len(rows), 'route_length': len(solution),
             'counts_by_kind': {name: int(counts.get(name, 0)) for name in ROW_KINDS},
             'route_rows': route_rows, 'learner_rows': learner_rows,
             'recovery_rows': len(rows) - route_rows - learner_rows,
             'duplicate_states_skipped': {name: int(level.duplicates.get(name, 0)) for name in ROW_KINDS},
             'unwinnable_rows': int(sum(not bool(row['solvable']) for row in rows)),
             'unlabelled_rows': int(sum(int(row['optimal']) == 0 for row in rows)),
             'win_rows': int(sum(bool(row['won'].any()) for row in rows)),
             'learner_stop': None if learner is None else learner['stop'],
             'learner_won': None if learner is None else learner['won'],
             'learner_actions': 0 if learner is None else learner['actions'],
             'optimal_choice_rate': None if learner is None else learner['optimal_choice_rate'],
             'recoveries': len(recoveries), 'recoveries_won': sum(t['stop'] == 'won' for t in recoveries),
             'oracle_backend': oracle.engine, 'search_truncated': bool(oracle.truncated),
             'reachable_states': oracle._reachable, 'trajectories': trajectories,
             'learner_steps': None if learner is None else learner['steps']}
    if proof['search_truncated'] or proof['oracle_backend'] != 'fast':
        raise ValueError('complete fast oracle proof required')
    return rows, proof


def stack_rows(rows):
    if not rows:
        raise ValueError('no rows to stack')
    return {key: np.stack([row[key] for row in rows]) for key in rows[0]}


def level_paths(out_dir, seed):
    array = Path(out_dir) / 'levels' / f'{int(seed)}.npz'
    return array, array.with_suffix('.json')


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def write_level(out_dir, spec, rows, proof):
    """levels/<seed>.npz (stacked per key) and levels/<seed>.json with its sha256."""
    array, proof_path = level_paths(out_dir, spec['seed'])
    array.parent.mkdir(parents=True, exist_ok=True)
    temporary = array.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **stack_rows(rows))
    temporary.replace(array)
    record = {**proof, 'status': 'complete', 'array_path': str(array), 'array_sha256': digest(array),
              'spec_sha256': spec_sha(spec)}
    write_json(proof_path, record)
    return array, proof_path


def level_complete(out_dir, spec, *, verify=True):
    """True when a complete proof for ``spec`` exists and its array digest matches."""
    array, proof_path = level_paths(out_dir, spec['seed'])
    if not array.exists() or not proof_path.exists():
        return False
    try:
        proof = json.loads(proof_path.read_text())
    except ValueError:
        return False
    if proof.get('status') != 'complete' or proof.get('spec_sha256') != spec_sha(spec):
        return False
    return not verify or proof.get('array_sha256') == digest(array)


def load_rows(out_dir, *, verify=True):
    """Concatenate every complete level; adds a per-row int64 ``seed`` array."""
    out_dir = Path(out_dir)
    parts, seeds = [], []
    for proof_path in sorted((out_dir / 'levels').glob('*.json')):
        proof = json.loads(proof_path.read_text())
        array = proof_path.with_suffix('.npz')
        if proof.get('status') != 'complete' or not array.exists():
            continue
        if verify and proof.get('array_sha256') != digest(array):
            raise ValueError(f'array digest differs from proof: {array}')
        with np.load(array, allow_pickle=False) as data:
            part = {key: data[key] for key in data.files}
        parts.append(part)
        seeds.append(np.full(len(part['optimal']), int(proof['seed']), dtype=np.int64))
    if not parts:
        raise ValueError(f'no complete levels under {out_dir}')
    arrays = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    arrays['seed'] = np.concatenate(seeds)
    return arrays
