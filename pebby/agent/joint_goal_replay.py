"""Generated raw H8/K4 replay; privileged labels never enter behavior forward.

Each immutable level shard holds actual engine successors of four action
sequences chosen beforehand in the behavior model's own imagination. No cached
encoder fields exist. Static goal anchors and live semantic surfaces use the
qualification label contract, including fog, moving cyclers and occlusion.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .joint_goal_data import PUBLIC_KEYS, array_contract as h1_contract, observe, state_targets, spec_digest
from .world_data import clone_env, history_arrays, spread_indices, successor_optimal_mask
from ..ls20 import names
from ..ls20.env import Ls20Scenario
from ..ls20.generate import build_level
from ..ls20.layout import extract
from ..ls20.plan import Oracle, simulate
from ..ls20.provenance import generated_context
from ..ls20.reference_profiles import profile_errors, SEARCH_LIMITS

FORMAT = 'pebby.joint-goal-replay.v1'
HORIZON = 4
COLLECTIONS = {'teacher': 0, 'deviation': 1, 'recovery': 2, 'exhaustion': 3, 'on_policy': 4, 'post_loss': 5}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def check_spec(spec, split):
    low, high = {'train': (0, 1_000_000), 'validation': (1_000_000, 2_000_000)}[split]
    if (spec.get('source') != 'generated_only' or spec.get('split') != split
            or type(spec.get('seed')) is not int or not low <= spec['seed'] < high
            or spec.get('generator_version') != 3 or spec.get('geometry_split') != split
            or spec.get('context_engine_verified') is not True or spec.get('search_truncated') is not False
            or spec.get('context_index') != generated_context(spec)):
        raise ValueError('verified generated-v3 spec in the requested disjoint split required')
    errors = profile_errors(spec)
    if errors:
        raise ValueError('unsupported reference profile: ' + '; '.join(errors))
    if any(goal.get('vanishing_ring') or goal.get('moving') for goal in spec['goals']):
        raise ValueError('only static goal anchors supported')


def array_contract():
    result = h1_contract()
    for key, (shape, dtype) in list(result.items()):
        if key.startswith('next_') or key in ('lost_life', 'terminal', 'won'):
            result[key] = ((4, 4) + shape[1:], dtype)
    result.update(actions=((4, 4), 'int64'), transition_valid=((4, 4), 'bool'),
                  next_distance=((4, 4), 'int16'), next_distance_valid=((4, 4), 'bool'),
                  difficulties=((), 'int8'))
    return result


def schema():
    return dict(history=8, horizon=4, branches=4, public_inputs=list(PUBLIC_KEYS), collections=COLLECTIONS,
                arrays={key: dict(shape=list(shape), dtype=dtype) for key, (shape, dtype) in array_contract().items()},
                axes='Every shape above excludes N; branch axis is canonical first action 0:UP,1:DOWN,2:LEFT,3:RIGHT.',
                chronology='Private branch H8; append actual frame/action; life loss clears to successor/-1; missing terminal frame and padded tails have no history.',
                distance='Valid -1 is proven unreachable before next life loss; invalid is unknown. Finite 0..128. Death reset distance is not a reward for dying.',
                semantics='Persistent static goal anchors; active unsolved goal role; current live cycler/refill surfaces; visible5x5/support7x7 exclude fog/HUD; semantic_valid excludes underlays/transients.',
                no_teacher_forcing=True, cached_features=False,
                label_boundary='Only root public tuple enters behavior encode; actual future observations and every label are targets only.')


def successor_history(row, branch, step):
    if not 0 <= branch < 4 or not 0 <= step < 4:
        raise ValueError('branch and step must be in 0..3')
    if not row['next_frame_valid'][branch, step]:
        return None
    valid = row['history_valid']
    history = (list(row['frames'][valid]), list(row['previous_actions'][valid]))
    for index in range(step + 1):
        if not row['next_frame_valid'][branch, index]:
            raise ValueError('valid successor follows missing frame')
        history = observe(history, row['next_frames'][branch, index], int(row['actions'][branch, index]),
                          bool(row['next_history_reset'][branch, index]))
    return dict(zip(PUBLIC_KEYS, history_arrays(*history, 8)))


def _distance(oracle, env, observation=None):
    if observation is not None and (observation.frame is None or (observation.finished and not observation.won)):
        return -1, False
    value = 0 if observation is not None and observation.won else oracle.distance_for(oracle.state_of(env))
    if value is not None and not 0 <= value <= 128:
        raise ValueError('distance exceeds fresh value-head finite range 0..128')
    return -1 if value is None else int(value), True


def checked_step(env, oracle, action):
    before = oracle.state_of(env)
    expected, outcome = simulate(oracle.layout, before, action, oracle.refills)
    lives = env.lives()
    observation = env.perform(names.ACTION_IDS[action])
    lost = env.lives() < lives
    if (lost != (outcome == 'died') or bool(observation.won) != (outcome == 'won')
            or (not lost and oracle.state_of(env) != expected)):
        raise ValueError('actual generated engine differs from supported complete oracle')
    if observation.frame is None and not observation.finished:
        raise ValueError('missing actual frame on a live transition')
    return observation, lost


@torch.inference_mode()
def collect_root(policy, spec, env, oracle, history, guard=lambda: None):
    if oracle.truncated or oracle.engine != 'fast':
        raise ValueError('complete native oracle required')
    if env.state.value in ('WIN', 'GAME_OVER'):
        raise ValueError('root must be live')
    guard()
    public = dict(zip(PUBLIC_KEYS, history_arrays(*history, 8)))
    # The policy sees exactly three public tensors, once, before any branches.
    field = policy.encode(*(torch.as_tensor(public[key])[None] for key in PUBLIC_KEYS))
    imagined = policy.imagine(field, horizon=4)
    actions = imagined['imagined_actions'][0].detach().cpu().numpy().copy()
    if (actions.shape != (4, 4) or not np.issubdtype(actions.dtype, np.integer)
            or ((actions < 0) | (actions > 3)).any() or not np.array_equal(np.sort(actions[:, 0]), np.arange(4))):
        raise ValueError('model must imagine exactly four integer K4 sequences with distinct first actions')
    actions = actions[np.argsort(actions[:, 0])].astype(np.int64)
    guard()
    row = {key: np.zeros(shape, dtype) for key, (shape, dtype) in array_contract().items()}
    row['next_goal_triple'].fill(-1)
    row['next_distance'].fill(-1)
    row.update(public)
    row.update(state_targets(spec, env, history[0][-1]))
    row.update(actions=actions, current_distance=np.int16(_distance(oracle, env)[0]),
               current_distance_valid=np.bool_(True),
               optimal=np.uint8(successor_optimal_mask(oracle, oracle.state_of(env))))
    row['optimal_valid'] = np.bool_(row['optimal'] != 0)
    for branch_index, trace in enumerate(actions):
        branch = clone_env(env)
        for step, action in enumerate(trace):
            guard()
            observation, lost = checked_step(branch, oracle, int(action))
            for key, value in state_targets(spec, branch, observation.frame).items():
                row['next_' + key][branch_index, step] = value
            row['transition_valid'][branch_index, step] = True
            for key, value in (('lost_life', lost), ('next_history_reset', lost),
                               ('terminal', observation.finished), ('won', observation.won)):
                row[key][branch_index, step] = value
            if observation.frame is not None:
                row['next_frames'][branch_index, step] = np.asarray(observation.frame, np.uint8)
                row['next_frame_valid'][branch_index, step] = True
            distance, distance_valid = _distance(oracle, branch, observation)
            row['next_distance'][branch_index, step] = distance
            row['next_distance_valid'][branch_index, step] = distance_valid
            mask = successor_optimal_mask(oracle, oracle.state_of(branch), terminal=observation.finished)
            row['next_optimal'][branch_index, step] = mask
            row['next_optimal_valid'][branch_index, step] = mask != 0
            if observation.finished:
                break
    row['distances'] = row['next_distance'][:, 0].copy()
    row['distance_valid'] = row['next_distance_valid'][:, 0].copy()
    for action in range(4):
        if row['optimal'] & (1 << action):
            if row['lost_life'][action, 0] or row['distances'][action] != row['current_distance'] - 1:
                raise ValueError('actual H1 branch fails optimal-action proof')
    return row


@torch.inference_mode()
def collect_level(policy, spec, split, *, max_roots=16, guard=lambda: None):
    """Representative teacher roots, bounded deviations/recovery and exhaustion."""
    check_spec(spec, split)
    if not 8 <= max_roots <= 128:
        raise ValueError('max_roots must be 8..128')
    guard()
    env = Ls20Scenario(build_level(spec), generated_context(spec))
    oracle = Oracle(extract(env), limit=SEARCH_LIMITS[spec['difficulty'] - 1], engine='fast')
    try:
        guard()
        if oracle.truncated or not oracle.solvable:
            raise ValueError('native graph incomplete or generated level unsolvable')
        route = [names.ACTION_IDS.index(action) for action in oracle.solution(seed=spec['seed'])]
        if not 1 <= len(route) <= 128:
            raise ValueError('teacher route outside bounded finite range')
        # Inspect the full actual route for representative mechanics, then
        # replay only bounded selected roots. Route selection is training-only.
        probe = clone_env(env)
        reasons = {0: ['initial'], len(route) - 1: ['pre_win']}
        matching_before = False
        for index, action in enumerate(route):
            guard()
            triple = probe.triple()
            matching = any(tuple(goal['triple']) == triple and not solved
                           for goal, solved in zip(spec['goals'], probe.goals_solved()))
            if matching and not matching_before:
                reasons.setdefault(index, []).append('matching_entry_or_exit')
                if index + 1 < len(route):
                    reasons.setdefault(index + 1, []).append('matching_navigation')
            matching_before = matching
            steps = probe.steps_left()
            observation, lost = checked_step(probe, oracle, action)
            if lost:
                raise ValueError('verified teacher route lost a life')
            if probe.steps_left() > steps:
                reasons.setdefault(index, []).append('pre_refill')
                if index + 1 < len(route):
                    reasons.setdefault(index + 1, []).append('post_refill')
        if not observation.won or probe.lives() != 3 or probe.levels_completed != 1:
            raise ValueError('complete teacher did not win actual generated engine')
        on_policy_quota = min(24, max(2, max_roots // 3))
        # Authentic post-loss recovery roots only exist after a real life loss,
        # so reserve part of the learner budget for them; otherwise ordinary
        # visited/mistake roots exhaust the quota long before the first death.
        post_loss_reserve = min(3, max(1, on_policy_quota // 4))
        visited_quota = on_policy_quota - post_loss_reserve
        # Preserve every real teacher-route state when the bound admits it;
        # reserve three deviation/recovery, one pre-death and two reset roots.
        teacher_limit = max(1, max_roots - on_policy_quota - 6)
        priority = [len(route) - 1] + sorted(reasons)
        chosen = list(dict.fromkeys(priority))[:teacher_limit]
        for index in spread_indices(list(range(len(route))), teacher_limit):
            if len(chosen) < teacher_limit and index not in chosen:
                chosen.append(index)
        chosen = set(chosen)
        rows, origins = [], []

        def add(current, history, path, collection, why):
            row = collect_root(policy, spec, current, oracle, history, guard)
            row.update(seeds=np.int64(spec['seed']), difficulties=np.int8(spec['difficulty']),
                       context_index=np.int8(generated_context(spec)), collection=np.int8(COLLECTIONS[collection]),
                       trajectory_step=np.int16(len(path)))
            rows.append(row)
            origins.append(dict(collection=collection, reasons=why, actions=list(path),
                                public_sha256=hashlib.sha256(b''.join(np.asarray(row[key]).tobytes() for key in PUBLIC_KEYS)).hexdigest()))

        history = ([env.render()], [-1])
        path = []
        anchor = None
        for index, action in enumerate(route):
            guard()
            if index in chosen:
                add(env, history, path, 'teacher', reasons.get(index, ['spread']))
                if anchor is None and index >= len(route) // 3:
                    anchor = (clone_env(env), (list(history[0]), list(history[1])), list(path))
            observation, lost = checked_step(env, oracle, action)
            path.append(action)
            if observation.finished:
                break
            history = observe(history, observation.frame, action, lost)
        # DAgger-style collection only: the actual closed-loop policy sees the
        # public tuple and receives no teacher action steering. The complete
        # oracle labels visited roots after the policy has decided its action.
        on_policy = Ls20Scenario(build_level(spec), generated_context(spec))
        policy_history, policy_path = ([on_policy.render()], [-1]), []
        seen, policy_count, policy_decisions = set(), 0, []
        repeated_positions, previous_position = 0, None
        policy_visited, policy_life_losses, policy_post_loss_roots = 0, 0, 0
        # The later deviation/recovery, pre-death and reset stages reserve six
        # rows; learner roots must never consume that reserve.
        on_policy_reserve = max_roots - 6
        for step in range(100):
            guard()
            public = history_arrays(*policy_history, 8)
            logits = policy(*(torch.as_tensor(value)[None] for value in public))
            if logits.shape != (1, 4) or not bool(torch.isfinite(logits).all()):
                raise ValueError('behavior policy must return finite four-action logits')
            action = int(logits[0].argmax())
            state = oracle.state_of(on_policy)
            mask = successor_optimal_mask(oracle, state)
            mistake = not bool(mask & (1 << action))
            identity = (on_policy.lives(), state)
            if identity not in seen and policy_visited < visited_quota and (step == 0 or mistake or step % 8 == 0):
                add(on_policy, policy_history, policy_path, 'on_policy',
                    ['actual_policy_mistake' if mistake else 'actual_policy_visited'])
                seen.add(identity)
                policy_visited += 1
                policy_count += 1
            observation, lost = checked_step(on_policy, oracle, action)
            policy_path.append(action)
            policy_decisions.append(dict(action=action, lost_life=lost, terminal=bool(observation.finished),
                                         won=bool(observation.won)))
            position = (on_policy.player_cell(), on_policy.triple(), on_policy.lives())
            repeated_positions = repeated_positions + 1 if position == previous_position else 0
            previous_position = position
            policy_life_losses += bool(lost)
            if observation.finished:
                break
            policy_history = observe(policy_history, observation.frame, action, lost)
            if lost:
                # Authentic learner recovery: the actual reset state the learner
                # must now act from, labelled by the complete oracle only after
                # the learner already chose the action that cost the life.
                if on_policy.lives() not in (1, 2) or len(policy_history[0]) != 1 or policy_history[1] != [-1]:
                    raise ValueError('actual learner life loss must leave a live reset one-frame history')
                identity = (on_policy.lives(), oracle.state_of(on_policy))
                if identity not in seen and policy_count < on_policy_quota and len(rows) < on_policy_reserve:
                    add(on_policy, policy_history, policy_path, 'on_policy', ['actual_policy_post_loss'])
                    seen.add(identity)
                    policy_count += 1
                    policy_post_loss_roots += 1
            # Keep the stall guard, but never cut the rollout off, by quota or
            # by stall, before the learner's own death loop has produced at
            # least one labelled reset root. The step cap still bounds this.
            if (policy_count >= on_policy_quota or (policy_visited >= visited_quota and policy_post_loss_roots)
                    or (repeated_positions >= 3 and policy_life_losses)):
                break
        # One deterministic off-route action then up to two exact recoveries.
        if anchor is not None:
            deviated, history, path = anchor
            optimal = successor_optimal_mask(oracle, oracle.state_of(deviated))
            alternatives = [action for action in range(4) if not optimal & (1 << action)]
            action = alternatives[spec['seed'] % len(alternatives)] if alternatives else 0
            observation, lost = checked_step(deviated, oracle, action)
            path.append(action)
            if not observation.finished:
                history = observe(history, observation.frame, action, lost)
                for recovery in range(3):
                    if len(rows) >= max_roots - 3:
                        break
                    add(deviated, history, path, 'deviation' if recovery == 0 else 'recovery', ['bounded_off_route'])
                    action = oracle.action_for(oracle.state_of(deviated), seed=spec['seed'])
                    if action is None:
                        break
                    observation, lost = checked_step(deviated, oracle, action)
                    path.append(action)
                    if observation.finished:
                        break
                    history = observe(history, observation.frame, action, lost)
        # Real exhaustion continues through two native life losses. These are
        # direct encoder/root-policy examples, not synthetic HUD edits or only
        # successor-D labels. No voluntary RESET or state teleport is used.
        exhausted = Ls20Scenario(build_level(spec), generated_context(spec))
        history, path = ([exhausted.render()], [-1]), []
        post_loss_lives = []
        for _ in range(300):
            guard()
            state = oracle.state_of(exhausted)
            candidates = [simulate(oracle.layout, state, action, oracle.refills) for action in range(4)]
            dying = [action for action, (_, outcome) in enumerate(candidates) if outcome == 'died']
            if dying:
                if not post_loss_lives and len(rows) < max_roots - 2:
                    add(exhausted, history, path, 'exhaustion', ['actual_pre_death'])
                action = dying[0]
            else:
                eligible = [(following[6], action) for action, (following, outcome) in enumerate(candidates) if outcome != 'won']
                if not eligible:
                    break
                _, action = min(eligible)
            observation, lost = checked_step(exhausted, oracle, action)
            path.append(action)
            if observation.finished:
                break
            history = observe(history, observation.frame, action, lost)
            if lost:
                if exhausted.lives() not in (1, 2) or len(rows) >= max_roots:
                    raise ValueError('actual post-loss root quota or native-life contract violated')
                add(exhausted, history, path, 'post_loss', ['actual_life_loss_history_reset'])
                post_loss_lives.append(exhausted.lives())
                if exhausted.lives() == 1:
                    break
        if post_loss_lives != [2, 1]:
            raise ValueError('bounded actual exhaustion did not produce both lives2 and lives1 roots')
        arrays = {key: np.stack([row[key] for row in rows]) for key in array_contract()}
        # Learner roots are authentic closed-loop states; the post-loss ones must
        # be real reset states and the manifest counts must match the arrays.
        learner_resets = [index for index, origin in enumerate(origins) if 'actual_policy_post_loss' in origin['reasons']]
        if (int((arrays['collection'] == COLLECTIONS['on_policy']).sum()) != policy_count
                or policy_visited + policy_post_loss_roots != policy_count or policy_count > on_policy_quota
                or len(learner_resets) != policy_post_loss_roots or policy_post_loss_roots > policy_life_losses):
            raise ValueError('actual learner root counts differ from the recorded on-policy manifest')
        if learner_resets and (not np.isin(arrays['lives'][learner_resets], (1, 2)).all()
                               or not (arrays['history_valid'][learner_resets].sum(1) == 1).all()
                               or not (arrays['previous_actions'][learner_resets] == -1).all()):
            raise ValueError('invalid actual learner post-life-loss root or reset history')
        validate_arrays(arrays, spec)
        return arrays, dict(seed=spec['seed'], difficulty=spec['difficulty'], spec_sha256=spec_digest(spec),
                            roots=len(rows), teacher_replay_won=True, teacher_actions=route,
                            oracle_backend='fast', search_truncated=False, reachable_states=oracle._reachable,
                            collection_counts={name: int((arrays['collection'] == code).sum()) for name, code in COLLECTIONS.items()},
                            actual_on_policy_actions=policy_decisions, on_policy_action_cap=100,
                            on_policy_teacher_steering=False, on_policy_root_quota=on_policy_quota,
                            on_policy_roots=policy_count, on_policy_visited_root_quota=visited_quota,
                            on_policy_post_loss_root_reserve=post_loss_reserve,
                            actual_on_policy_life_losses=policy_life_losses,
                            actual_on_policy_post_loss_roots=policy_post_loss_roots,
                            teacher_root_steps=sorted(chosen), full_teacher_route_roots=len(chosen) == len(route),
                            post_loss_lives=post_loss_lives, exhaustion_action_cap=300,
                            origins=origins, events={key: arrays[key].sum((0, 1)).tolist() for key in ('lost_life', 'terminal', 'won')},
                            known_unreachable_roots=int((arrays['current_distance'] == -1).sum()),
                            known_unreachable_successors=int((arrays['next_distance_valid'] & (arrays['next_distance'] == -1)).sum()))
    finally:
        distance = getattr(oracle, '_distance', None)
        if hasattr(distance, 'close'):
            distance.close()


def validate_arrays(arrays, spec):
    contract = array_contract()
    if set(arrays) != set(contract):
        raise ValueError('missing or unexpected replay arrays')
    count = len(arrays['seeds'])
    if not 1 <= count <= 128:
        raise ValueError('level shard must contain 1..128 roots')
    for key, (shape, dtype) in contract.items():
        if arrays[key].shape != (count,) + shape or arrays[key].dtype != np.dtype(dtype):
            raise ValueError('invalid shape/dtype: ' + key)
    valid, terminal, frames = arrays['transition_valid'], arrays['terminal'], arrays['next_frame_valid']
    if (not valid[:, :, 0].all() or not np.array_equal(valid[:, :, 1:], valid[:, :, :-1] & ~terminal[:, :, :-1])
            or (frames & ~valid).any() or (valid & ~frames & ~terminal).any()
            or (terminal & ~valid).any() or (arrays['won'] & ~terminal).any()
            or (arrays['lost_life'] & ~valid).any()
            or not np.array_equal(arrays['next_history_reset'], arrays['lost_life'])
            or arrays['next_frames'][~frames].any()):
        raise ValueError('invalid causal prefix/terminal/history masks')
    if (((arrays['actions'] < 0) | (arrays['actions'] > 3)).any()
            or not np.array_equal(arrays['actions'][:, :, 0], np.broadcast_to(np.arange(4), (count, 4)))):
        raise ValueError('invalid canonical K4 actions')
    history = arrays['history_valid']
    previous = arrays['previous_actions']
    if (not history[:, -1].all() or (history[:, :-1] & ~history[:, 1:]).any()
            or ((previous < -1) | (previous > 3)).any() or (previous[~history] != -1).any()
            or (arrays['frames'] > 15).any() or (arrays['next_frames'] > 15).any()):
        raise ValueError('invalid public history or palette')
    for prefix in ('', 'next_'):
        get = lambda key: arrays[prefix + key]
        if ((get('support') & ~get('visible')).any() or (get('semantic_valid') & ~get('support')).any()
                or (get('goal_solved') & ~get('goal_presence')).any()
                or (get('goal_attribute_valid') & ~(get('semantic_valid') & get('goal_presence') & ~get('goal_solved'))).any()
                or (get('goal_triple')[~get('goal_presence')] != -1).any()):
            raise ValueError('inconsistent semantic masks')
        if ((get('player_cell') < 0) | (get('player_cell') >= 12)).any() or ((get('lives') < 0) | (get('lives') > 3)).any():
            raise ValueError('invalid physical labels')
        for column, size in enumerate((6, 4, 4)):
            attrs = get('goal_triple')[..., column][get('goal_presence')]
            if ((attrs < 0) | (attrs >= size)).any() or ((get('triple')[..., column] < 0) | (get('triple')[..., column] >= size)).any():
                raise ValueError('invalid appearance labels')
    for key in ('visible', 'support', 'semantic_valid', 'goal_attribute_valid'):
        if arrays['next_' + key][~frames].any():
            raise ValueError('missing public frame has semantic targets')
    for key in ('current_distance', 'distances', 'next_distance'):
        if ((arrays[key] < -1) | (arrays[key] > 128)).any():
            raise ValueError('distance outside finite/unreachable range')
    if (not arrays['current_distance_valid'].all()
            or not np.array_equal(arrays['next_distance_valid'], frames & (~terminal | arrays['won']))
            or not np.array_equal(arrays['distances'], arrays['next_distance'][:, :, 0])
            or not np.array_equal(arrays['distance_valid'], arrays['next_distance_valid'][:, :, 0])):
        raise ValueError('invalid complete-proof distance validity')
    for key, mask in (('optimal', 'optimal_valid'), ('next_optimal', 'next_optimal_valid')):
        if (arrays[key] > 15).any() or not np.array_equal(arrays[mask], arrays[key] != 0):
            raise ValueError('invalid optimal bitmask')
    if (arrays['next_optimal'][~valid | terminal] != 0).any():
        raise ValueError('padded or terminal state has an action target')
    if (not (arrays['seeds'] == spec['seed']).all() or not (arrays['difficulties'] == spec['difficulty']).all()
            or not (arrays['context_index'] == generated_context(spec)).all()
            or not np.isin(arrays['collection'], list(COLLECTIONS.values())).all()):
        raise ValueError('row provenance differs from exact spec')
    resets = arrays['collection'] == COLLECTIONS['post_loss']
    if (not np.isin(arrays['lives'][resets], (1, 2)).all()
            or not (arrays['history_valid'][resets].sum(1) == 1).all()
            or not (arrays['previous_actions'][resets] == -1).all()
            or not arrays['optimal_valid'][resets].all()):
        raise ValueError('invalid actual post-life-loss root/history/policy target')


class JointGoalReplay:
    """Verified immutable level shards; only requested rows become tensor copies.

    At most 128 level shards can be mapped in one reader. This deliberately
    bounds open mappings; full 10k-level replay needs a later lazy shard cache.
    """
    def __init__(self, directory, split=None, verify_hashes=True, guard=lambda: None, *, allow_incomplete=False):
        self.directory = Path(directory).resolve()
        self._shards = []
        self._closed = False
        self.guard = guard
        try:
            self.metadata = json.loads((self.directory / 'manifest.json').read_text())
            meta = self.metadata
            if meta.get('format') != FORMAT or (meta.get('status') != 'complete' and not allow_incomplete):
                raise ValueError('complete supported replay manifest required')
            if not 1 <= len(meta.get('shards', [])) <= 128:
                raise ValueError('bounded reader supports 1..128 complete level shards')
            if meta.get('split') not in ('train', 'validation') or (split is not None and split != meta['split']):
                raise ValueError('replay split mismatch')
            if meta.get('source_archive_contract') and meta['source_archive_contract'].get('version') != 1:
                raise ValueError('unsupported producer source archive contract')
            # Producer provenance is durable: bind verified content-addressed
            # archives, not the current checkout's potentially evolved code.
            # Legacy data without an archive must still match its live source.
            self.source_bindings = {}
            for path, expected in meta['sources'].items():
                guard()
                if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
                    raise ValueError('invalid producer SHA256')
                archive = self.directory / 'sources' / expected
                selected = archive if archive.exists() else Path(path)
                if meta.get('source_archive_contract') and not archive.is_file():
                    raise ValueError('required producer source archive missing: ' + path)
                if digest(selected) != expected:
                    raise ValueError('producer source/input checksum mismatch: ' + str(selected))
                self.source_bindings[str(selected.resolve())] = expected
            self.source_bindings[str(self.directory / 'manifest.json')] = digest(self.directory / 'manifest.json')
            bank_path = self.directory / 'bank.json'
            if digest(bank_path) != meta['bank_sha256']:
                raise ValueError('exact bank checksum mismatch')
            self.source_bindings[str(bank_path)] = meta['bank_sha256']
            bank = json.loads(bank_path.read_text())
            if len({spec['seed'] for spec in bank}) != len(bank):
                raise ValueError('duplicate bank seeds')
            by_seed = {spec['seed']: spec for spec in bank}
            self.specs, offsets, seen = [], [0], set()
            for entry in meta['shards']:
                guard()
                seed = entry['seed']
                if seed in seen or seed not in by_seed:
                    raise ValueError('duplicate or unbound shard level')
                seen.add(seed)
                spec = by_seed[seed]
                check_spec(spec, meta['split'])
                if entry['spec_sha256'] != spec_digest(spec):
                    raise ValueError('shard spec checksum mismatch')
                folder = (self.directory / entry['directory']).resolve()
                if not folder.is_relative_to(self.directory):
                    raise ValueError('shard directory escapes replay')
                shard_manifest = folder / 'manifest.json'
                if digest(shard_manifest) != entry['manifest_sha256']:
                    raise ValueError('shard manifest checksum mismatch')
                self.source_bindings[str(shard_manifest)] = entry['manifest_sha256']
                info = json.loads(shard_manifest.read_text())
                if (info['spec_sha256'] != entry['spec_sha256'] or info['status'] != 'complete'
                        or info.get('format') != FORMAT
                        or info.get('producer_bindings_sha256') != spec_digest(meta['sources'])):
                    raise ValueError('incomplete or mismatched level shard')
                arrays = {}
                self._shards.append(arrays)  # close partial opens if any check fails
                if set(info['arrays']) != set(array_contract()):
                    raise ValueError('shard array schema mismatch')
                for key, binding in info['arrays'].items():
                    guard()
                    path = folder / (key + '.npy')
                    if path.stat().st_size != binding['bytes'] or (verify_hashes and digest(path) != binding['sha256']):
                        raise ValueError('shard array checksum/size mismatch: ' + key)
                    arrays[key] = np.load(path, mmap_mode='r', allow_pickle=False)
                    self.source_bindings[str(path)] = binding['sha256']
                validate_arrays(arrays, spec)
                if len(arrays['seeds']) != entry['roots']:
                    raise ValueError('manifest root count mismatch')
                self.specs.append(spec)
                offsets.append(offsets[-1] + len(arrays['seeds']))
            if not self.specs or (meta['status'] == 'complete' and seen != set(by_seed)):
                raise ValueError('complete bank/shard coverage required')
            self._offsets = np.asarray(offsets, np.int64)
            self.rows = int(offsets[-1])
            self.seeds = np.concatenate([arrays['seeds'] for arrays in self._shards])
            self.tiers = np.concatenate([arrays['difficulties'] for arrays in self._shards])
            self._tier_shards = {tier: [index for index, spec in enumerate(self.specs) if spec['difficulty'] == tier]
                                 for tier in sorted({spec['difficulty'] for spec in self.specs})}
            self._events = [np.flatnonzero(arrays['lost_life'].any((1, 2)) | arrays['won'].any((1, 2))
                                           | (arrays['next_steps'][:, :, 0] > arrays['steps'][:, None]).any(1)
                                           | (arrays['next_triple'][:, :, 0] != arrays['triple'][:, None]).any((1, 2)))
                            for arrays in self._shards]
            self.metadata = dict(meta, loaded_roots=self.rows, loaded_levels=len(self.specs),
                                 maximum_supported_shards=128, default_event_fraction=.25,
                                 arrays_hash_verified=bool(verify_hashes), sampler='uniform tier, then level, then root; with replacement; optional within-level event mixture')
        except BaseException:
            self.close()
            raise

    def __len__(self):
        return self.rows

    def _check_open(self):
        if self._closed:
            raise RuntimeError('replay is closed')

    def sample(self, batch_size, rng, event_fraction=.25):
        self._check_open()
        if type(batch_size) is not int or batch_size < 1 or not 0 <= event_fraction <= 1:
            raise ValueError('positive batch size and event fraction in 0..1 required')
        result = []
        for _ in range(batch_size):
            tier = int(rng.choice(list(self._tier_shards)))
            shard = int(rng.choice(self._tier_shards[tier]))
            events = self._events[shard]
            local = int(rng.choice(events)) if len(events) and rng.random() < event_fraction else int(rng.integers(len(self._shards[shard]['seeds'])))
            result.append(int(self._offsets[shard]) + local)
        return np.asarray(result, np.int64)

    def batch(self, indices, device='cpu'):
        self._check_open()
        indices = np.asarray(indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or not len(indices) or ((indices < 0) | (indices >= self.rows)).any():
            raise ValueError('nonempty one-dimensional valid integer row indices required')
        self.guard()
        shard_ids = np.searchsorted(self._offsets[1:], indices, side='right')
        result = {}
        for key, (shape, dtype) in array_contract().items():
            values = np.empty((len(indices),) + shape, dtype)
            for shard in np.unique(shard_ids):
                positions = np.flatnonzero(shard_ids == shard)
                values[positions] = self._shards[shard][key][indices[positions] - self._offsets[shard]]
            result[key] = torch.from_numpy(values).to(device)
        self.guard()
        return ({key: result.pop(key) for key in PUBLIC_KEYS}, result)

    def close(self):
        for arrays in self._shards:
            for value in arrays.values():
                mmap = getattr(value, '_mmap', None)
                if mmap is not None and not mmap.closed:
                    mmap.close()
        self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()
