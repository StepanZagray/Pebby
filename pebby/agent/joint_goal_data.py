"""Generated-only raw public H8 and exact H1 supervision for fresh training.

This module collects targets; it never constructs or calls a neural policy.
Only PUBLIC_KEYS may enter a policy. Engine labels, complete solver distances,
collection reasons and generated teacher routes belong exclusively to training.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .world_data import clone_env, history_arrays, spread_indices, successor_optimal_mask, verified_context
from ..ls20 import names
from ..ls20.plan import simulate
from tools.audit_cell_appearance_dynamics import current_masks, current_surface_labels
from tools.prepare_visible_cell_labels import public_masks

FORMAT = 'pebby.joint-goal-qualification.v1'
SPEC_FORMAT = 'pebby.joint-goal-generated-spec.v1'
PUBLIC_KEYS = ('frames', 'history_valid', 'previous_actions')
HISTORY = 8
ROLE_NAMES = ('wall', 'goal', 'cycler_shape', 'cycler_color', 'cycler_rotation', 'launcher', 'refill', 'player')
COLLECTIONS = {'teacher': 0, 'deviation': 1, 'recovery': 2, 'exhaustion': 3}


def qualification_specs():
    """Eight new compact layouts, unrelated to any shipped layout or route.

    The seed is an identity in the reserved TRAIN namespace. Geometry is
    explicitly constructed here, not attributed to the random v3 generator.
    """
    cases = [
        ('matching_corridor', [(x, 3) for x in range(2, 7)], (2, 3), (0, 0, 0),
         [((6, 3), (0, 0, 0))], [], [], 10, 1, 0, True),
        ('matching_detour', [(2, 2), (2, 3), (2, 4), (3, 4), (4, 4), (4, 3), (4, 2), (1, 4)],
         (2, 2), (2, 1, 2), [((4, 2), (2, 1, 2))], [], [], 14, 1, 0, False),
        ('matching_two_goals', [(x, 6) for x in range(2, 8)], (2, 6), (4, 2, 1),
         [((4, 6), (4, 2, 1)), ((7, 6), (4, 2, 1))], [], [], 12, 1, 1, False),
        ('rotation_exit', [(2, 4), (3, 4), (3, 3), (4, 3), (5, 3), (6, 3), (3, 5), (2, 5)],
         (2, 4), (5, 1, 3), [((6, 3), (5, 1, 0))], [(3, 4)], [], 16, 1, 0, False),
        ('rotation_three_clicks', [(7, 5), (6, 5), (6, 4), (5, 4), (4, 4), (6, 6), (7, 6)],
         (7, 5), (3, 0, 1), [((4, 4), (3, 0, 0))], [(6, 5)], [], 18, 1, 1, False),
        ('rotation_two_goals', [(3, 3), (4, 3), (5, 3), (6, 3), (3, 4), (4, 4), (5, 4), (4, 5), (5, 5)],
         (4, 5), (1, 2, 0), [((3, 3), (1, 2, 1)), ((6, 3), (1, 2, 2))], [(4, 4)], [], 24, 1, 1, False),
        ('refill_required', [(2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (6, 6)],
         (2, 7), (1, 3, 2), [((6, 6), (1, 3, 2))], [], [(4, 7)], 3, 1, 1, False),
        ('refill_and_rotation', [(7, 7), (6, 7), (6, 6), (5, 6), (4, 6), (4, 5), (4, 4), (7, 6), (5, 7)],
         (7, 7), (2, 3, 2), [((4, 4), (2, 3, 3))], [(6, 7)], [(5, 6)], 6, 2, 1, False),
    ]
    result = []
    for index, (case, free, start, carried, goals, cyclers, refills, budget, cost, context, fog) in enumerate(cases):
        free = set(free)
        result.append(dict(format=SPEC_FORMAT, source='generated_only', split='train',
                           generator='joint_goal_qualification_templates', generator_version=1,
                           seed=910_100 + index, case=case, difficulty=context + 1,
                           difficulty_version='ls20-reference-v1', training_context_index=context,
                           qualification_only=True, full_seven_tier_coverage=False,
                           walls=[[x, y] for y in range(12) for x in range(12) if (x, y) not in free],
                           start=list(start), start_triple=list(carried),
                           goals=[dict(cell=list(cell), triple=list(triple)) for cell, triple in goals],
                           cyclers=[dict(cell=list(cell), kind='rotation') for cell in cyclers],
                           refills=[list(cell) for cell in refills], launchers=[], rails=[],
                           step_counter=budget, step_cost=cost, fog=fog))
    return result


def spec_digest(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_paths():
    import inspect
    from arcengine import ARCBaseGame
    root = Path(__file__).resolve().parents[2]
    paths = [root / path for path in (
        'tools/collect_joint_goal_qualification.py', 'pebby/agent/joint_goal_data.py',
        'pebby/agent/world_data.py', 'tools/audit_cell_appearance_dynamics.py',
        'tools/prepare_visible_cell_labels.py', 'pebby/ls20/env.py', 'pebby/ls20/names.py',
        'pebby/ls20/generate.py', 'pebby/ls20/provenance.py', 'pebby/ls20/layout.py',
        'pebby/ls20/plan.py', 'pebby/ls20/fastplan.py', 'pebby/ls20/_fastplan.c',
        'pebby/ls20/rails.py', 'third_party/ls20/ls20.py')]
    return paths + [Path(inspect.getfile(ARCBaseGame)).resolve()]


def check_spec(spec):
    if (spec.get('format') != SPEC_FORMAT or spec.get('source') != 'generated_only'
            or spec.get('split') != 'train' or type(spec.get('seed')) is not int
            or not 0 <= spec['seed'] < 1_000_000 or spec.get('qualification_only') is not True):
        raise ValueError('explicit generated TRAIN qualification spec required')
    if spec.get('rails') or spec.get('launchers'):
        raise ValueError('qualification data supports static goals/cyclers and no launchers')
    if not 1 <= spec.get('step_counter', 0) <= 42 or spec.get('step_cost') not in (1, 2, 3):
        raise ValueError('bounded supported fuel configuration required')
    if spec.get('training_context_index') != spec.get('difficulty', 0) - 1:
        raise ValueError('explicit difficulty/context binding required')
    if spec.get('difficulty_version') != 'ls20-reference-v1' or spec.get('difficulty') not in (1, 2):
        raise ValueError('qualification supports explicit contexts zero and one only')
    if spec.get('search_truncated') or spec.get('search_limit', 50_000) > 50_000:
        raise ValueError('qualification refuses incomplete proofs or expanded search budgets')


def observe(history, frame, action, reset=False):
    if frame is None:
        raise ValueError('cannot append a missing public frame')
    if reset:
        return [frame], [-1]
    return (history[0] + [frame])[-HISTORY:], (history[1] + [action])[-HISTORY:]


def public_inputs(row):
    """Explicit policy boundary: return no labels, masks or teacher information."""
    return {key: row[key] for key in PUBLIC_KEYS}


def successor_history(row, action):
    """Reconstruct one raw H1 successor's causal public H8, or None."""
    if type(action) is not int or not 0 <= action < 4:
        raise ValueError('action must be an integer in 0..3')
    if not row['next_frame_valid'][action]:
        return None
    valid = row['history_valid']
    history = (list(row['frames'][valid]), list(row['previous_actions'][valid]))
    history = observe(history, row['next_frames'][action], action, bool(row['next_history_reset'][action]))
    return dict(zip(PUBLIC_KEYS, history_arrays(*history, HISTORY)))


def state_targets(spec, env, frame):
    """Exact state plus separately masked visible-surface supervision.

    Goal presence denotes a persistent static anchor, including solved goals;
    the goal role denotes an active unsolved surface. Goal attributes are
    supervised only for active, supported, uncovered, non-transient goals.
    """
    labels = dict(player_cell=np.asarray(env.player_cell(), np.int16), triple=np.asarray(env.triple(), np.int8),
                  steps=np.int16(env.steps_left()), lives=np.int8(env.lives()))
    roles = np.zeros((144, 8), bool)
    goal_triple = np.full((144, 3), -1, np.int8)
    goal_presence = np.zeros(144, bool)
    goal_solved = np.zeros(144, bool)
    for index, goal in enumerate(spec['goals']):
        cell = goal['cell'][1] * 12 + goal['cell'][0]
        goal_presence[cell] = True
        goal_triple[cell] = env.goal_triples()[index]
        goal_solved[cell] = env.goals_solved()[index]
    support = np.zeros(144, bool)
    visible = np.zeros(144, bool)
    semantic_valid = np.zeros(144, bool)
    goal_attribute_valid = np.zeros(144, bool)
    if frame is not None:
        frame = np.asarray(frame)
        if frame.shape != (64, 64) or not np.issubdtype(frame.dtype, np.integer) or ((frame < 0) | (frame > 15)).any():
            raise ValueError('actual public frame must be a 64x64 integer palette grid')
        support7, _, _ = current_masks(env, frame)
        _, fully_visible, hud_overlap = public_masks(env, frame)
        support = support7.reshape(-1)
        visible = (fully_visible & ~hud_overlap).reshape(-1)
        transient = any(bool(getattr(env.game, name)) for name in
                        (names.ATTR_ACTIVE_ANIMATIONS, names.ATTR_DEATH_FLASH, names.ATTR_REJECT_FLASH))
        bits, _, active_goals, _, excluded, player_overlap = current_surface_labels(spec, env, support7, transient)
        roles = (bits.reshape(-1, 1) & (1 << np.arange(8))[None]) != 0
        semantic_valid = support & ~excluded
        if player_overlap:
            x, y = env.player_cell()
            semantic_valid[y * 12 + x] = False
        if transient:
            semantic_valid[:] = False
        goal_attribute_valid = active_goals.reshape(-1) & semantic_valid
    return dict(**labels, roles=roles, goal_triple=goal_triple, goal_presence=goal_presence,
                goal_solved=goal_solved, visible=visible, support=support,
                semantic_valid=semantic_valid, goal_attribute_valid=goal_attribute_valid)


def collect_root(spec, env, oracle, history, guard=lambda: None):
    """Four actual actions, independently verified against the complete solver."""
    check_spec(spec)
    if oracle.truncated:
        raise ValueError('complete oracle required; never guess action/value labels')
    if env.state.value in ('WIN', 'GAME_OVER'):
        raise ValueError('joint roots must be actual live generated states')
    guard()
    before_state = oracle.state_of(env)
    before_distance = oracle.distance_for(before_state)
    row = {**dict(zip(PUBLIC_KEYS, history_arrays(*history, HISTORY))),
           **state_targets(spec, env, history[0][-1]),
           'current_distance': np.int16(-1 if before_distance is None else before_distance),
           # Complete forward search and actual supported behavior establish
           # that None here is known unreachable, not an unsearched state.
           'current_distance_valid': np.bool_(True)}
    next_labels, frames, frame_valid, distances, distance_valid, masks, lost, terminal, won = [], [], [], [], [], [], [], [], []
    for action in range(4):
        guard()
        branch = clone_env(env)
        expected, outcome = simulate(oracle.layout, before_state, action, oracle.refills)
        observation = branch.perform(names.ACTION_IDS[action])
        life_loss = branch.lives() < env.lives()
        if (life_loss != (outcome == 'died') or bool(observation.won) != (outcome == 'won')
                or (not life_loss and oracle.state_of(branch) != expected)):
            raise ValueError('actual engine successor differs from the complete logical teacher')
        distance = (0 if observation.won else None if observation.finished else oracle.distance_for(oracle.state_of(branch)))
        distances.append(-1 if distance is None else distance)
        distance_valid.append(observation.frame is not None and (not observation.finished or observation.won))
        masks.append(successor_optimal_mask(oracle, oracle.state_of(branch), terminal=observation.finished))
        lost.append(life_loss)
        terminal.append(observation.finished)
        won.append(observation.won)
        frames.append(np.zeros((64, 64), np.uint8) if observation.frame is None else np.asarray(observation.frame, np.uint8))
        frame_valid.append(observation.frame is not None)
        if observation.frame is None and not observation.finished:
            raise ValueError('missing observation on a live branch')
        next_labels.append(state_targets(spec, branch, observation.frame))
    optimal = sum(1 << action for action, distance in enumerate(distances)
                  if before_distance is not None and distance == before_distance - 1 and not lost[action])
    if before_distance is not None and not optimal:
        raise ValueError('complete teacher lacks an actual distance-decreasing action')
    row.update({'next_' + key: np.stack([label[key] for label in next_labels]) for key in next_labels[0]})
    row.update(next_frames=np.stack(frames), next_frame_valid=np.asarray(frame_valid, bool),
               next_history_reset=np.asarray(lost, bool), lost_life=np.asarray(lost, bool),
               terminal=np.asarray(terminal, bool), won=np.asarray(won, bool), optimal=np.uint8(optimal),
               optimal_valid=np.bool_(bool(optimal)), distances=np.asarray(distances, np.int16),
               distance_valid=np.asarray(distance_valid, bool), next_optimal=np.asarray(masks, np.uint8),
               next_optimal_valid=np.asarray(masks, np.uint8) != 0)
    return row


def collect_level(spec, guard=lambda: None, *, max_roots=64, recovery_steps=3):
    """Complete teacher route, four short deviations/recoveries, and exhaustion."""
    check_spec(spec)
    if not 1 <= max_roots <= 96 or not 0 <= recovery_steps <= 4:
        raise ValueError('roots must be 1..96 and recovery_steps 0..4')
    guard()
    env, oracle, proof = verified_context(spec, search_limit=50_000)
    guard()
    if env is None:
        raise ValueError('qualification spec lacks complete engine-verified proof: ' + str(proof))
    if spec['refills']:
        without_refills = {**spec, 'refills': []}
        no_refill_env, _, no_refill_proof = verified_context(without_refills, search_limit=50_000)
        guard()
        if no_refill_env is not None or no_refill_proof.get('search_truncated') is not False:
            raise ValueError('refill qualification case must be proven unsolvable without its refill')
        proof['refill_necessity_verified'] = True
    route = oracle.solution(seed=spec['seed'])
    if not route or len(route) > 32:
        raise ValueError('qualification teacher route must contain 1..32 actions')
    rows, origins, anchors = [], [], []
    history = ([env.render()], [-1])
    scheduled = set(spread_indices(list(range(len(route))), 3))

    def record(current, buffers, kind, step):
        if len(rows) >= max_roots:
            raise ValueError('root bound reached; refusing a silently truncated qualification dataset')
        row = collect_root(spec, current, oracle, buffers, guard)
        row.update(seeds=np.int64(spec['seed']), context_index=np.int8(spec['training_context_index']),
                   collection=np.int8(COLLECTIONS[kind]), trajectory_step=np.int16(step))
        rows.append(row)
        origins.append(dict(collection=kind, step=step))
        return row

    def advance(current, buffers, action):
        old_lives = current.lives()
        observation = current.perform(names.ACTION_IDS[action])
        if observation.frame is not None:
            buffers = observe(buffers, observation.frame, action, current.lives() < old_lives)
        elif not observation.finished:
            raise ValueError('missing live observation during behavior')
        return buffers, observation

    for step, engine_action in enumerate(route):
        row = record(env, history, 'teacher', step)
        matched_cycler = (tuple(env.player_cell()) in oracle.layout.cyclers
                          and tuple(env.triple()) in [tuple(t) for t in env.goal_triples()])
        if len(anchors) < 4 and (step in scheduled or matched_cycler):
            anchors.append((clone_env(env), copy.deepcopy(history), step, int(row['optimal'])))
        history, observation = advance(env, history, names.ACTION_IDS.index(engine_action))
        if observation.finished and (step != len(route) - 1 or not observation.won):
            raise ValueError('verified generated teacher route diverged on actual replay')
    if not observation.won or env.lives() != 3:
        raise ValueError('generated teacher replay must win without life loss')
    rng = np.random.default_rng(spec['seed'])
    for branch, buffers, root_step, optimal in anchors:
        candidates = [a for a in range(4) if not optimal & (1 << a)]
        action = int(rng.choice(candidates if candidates else range(4)))
        for offset, selected in enumerate((action, action ^ 1)):
            buffers, observation = advance(branch, buffers, selected)
            if observation.finished:
                break
            record(branch, buffers, 'deviation', root_step + offset + 1)
        if observation.finished:
            continue
        for offset in range(recovery_steps):
            selected = oracle.action_at(branch, seed=spec['seed'] + offset)
            if selected is None:
                break
            buffers, observation = advance(branch, buffers, names.ACTION_IDS.index(selected))
            if observation.finished:
                break
            record(branch, buffers, 'recovery', root_step + 3 + offset)
    # A fixed wall attempt gives real life-loss/GAME_OVER targets, with no
    # privileged action choice ever entering a policy (there is no policy).
    initial = clone_env(env)
    buffers = ([initial.reset()], [-1])
    blocked = next((a for a, (dx, dy) in enumerate(names.ACTION_DELTAS)
                    if not oracle.layout.free((initial.player_cell()[0] + dx, initial.player_cell()[1] + dy))), None)
    if blocked is None:
        raise ValueError('qualification start requires one blocked exhaustion direction')
    for step in range(132):
        guard()
        if initial.steps_left() <= initial.step_cost():
            record(initial, buffers, 'exhaustion', step)
        buffers, observation = advance(initial, buffers, blocked)
        if observation.finished:
            break
    if not observation.finished or observation.won:
        raise ValueError('bounded exhaustion trajectory must reach real GAME_OVER')
    return rows, dict(**proof, spec_sha256=spec_digest(spec), case=spec['case'],
                     teacher_route=route, teacher_replay_won=True, teacher_lives_left=3,
                     roots=len(rows), origins=origins, qualification_only=True)


def schema():
    return dict(format=FORMAT, history=8, horizon=1, branches=4, actions='0:UP 1:DOWN 2:LEFT 3:RIGHT',
                public_inputs=list(PUBLIC_KEYS), role_names=list(ROLE_NAMES), collections=COLLECTIONS,
                physical='player_cell[2], triple[3], steps, lives; next_ variants [4,...]',
                semantic='roles[144,8], goal_triple[144,3], goal_presence/goal_solved/visible/support[144]; next_ variants [4,...]',
                masks='visible: full5x5 nonHUD public; support: full7x7 nonHUD public; semantic_valid excludes covered underlay and transient overlays; goal_attribute_valid also requires an active unsolved public goal',
                goals='goal_presence is a persistent static anchor, including solved goals; roles.goal is an active unsolved surface; absent goal_triple=-1',
                successors='next_frames[4,64,64] actual public uint8; next_frame_valid; next_history_reset=lost_life; reconstruct with successor_history',
                distance='Exact remaining actions to win without another life loss. Valid=True with -1 is proven unreachable and trains the unreachable class; valid=False is unknown/unsupported (GAME_OVER or missing successor frame). Live roots use complete reachable-graph proof. Never regress -1 as a numeric distance.',
                optimal='optimal/next_optimal[4] uint8 bitsets; zero and optimal_valid/next_optimal_valid=False excludes policy supervision',
                label_boundary='All nonpublic arrays are training-only labels/metadata; no engine labels, support masks or routes enter model forward',
                qualification_only=True, full_seven_tier_coverage=False,
                fresh_targets=True, cached_features=False)


def array_contract():
    """Unbatched shapes and exact storage dtypes; prepend N for the NPZ."""
    physical = dict(player_cell=((2,), 'int16'), triple=((3,), 'int8'), steps=((), 'int16'), lives=((), 'int8'))
    semantic = dict(roles=((144, 8), 'bool'), goal_triple=((144, 3), 'int8'),
                    **{key: ((144,), 'bool') for key in ('goal_presence', 'goal_solved', 'visible', 'support',
                                                       'semantic_valid', 'goal_attribute_valid')})
    contract = dict(frames=((8, 64, 64), 'uint8'), history_valid=((8,), 'bool'), previous_actions=((8,), 'int64'),
                    **physical, **semantic)
    contract.update({'next_' + key: ((4,) + shape, dtype) for key, (shape, dtype) in (physical | semantic).items()})
    contract.update(next_frames=((4, 64, 64), 'uint8'),
                    **{key: ((4,), 'bool') for key in ('next_frame_valid', 'next_history_reset', 'lost_life', 'terminal',
                                                      'won', 'distance_valid', 'next_optimal_valid')},
                    optimal=((), 'uint8'), optimal_valid=((), 'bool'), next_optimal=((4,), 'uint8'),
                    distances=((4,), 'int16'), current_distance=((), 'int16'), current_distance_valid=((), 'bool'),
                    seeds=((), 'int64'), context_index=((), 'int8'), collection=((), 'int8'), trajectory_step=((), 'int16'))
    return contract


def validate_arrays(arrays, specs):
    contract = array_contract()
    if set(arrays) != set(contract):
        raise ValueError('unexpected or missing raw joint dataset arrays')
    count = len(arrays['seeds'])
    if not 1 <= count <= 8 * 96:
        raise ValueError('qualification row count outside bounded schema')
    for key, (shape, dtype) in contract.items():
        if arrays[key].shape != (count,) + shape or arrays[key].dtype != np.dtype(dtype):
            raise ValueError('invalid shape/dtype: ' + key)
    if any((arrays[key] > 15).any() for key in ('frames', 'next_frames')):
        raise ValueError('public palette outside 0..15')
    valid = arrays['history_valid']
    previous = arrays['previous_actions']
    if (not valid[:, -1].all() or (valid[:, :-1] & ~valid[:, 1:]).any()
            or ((previous < -1) | (previous > 3)).any() or (previous[~valid] != -1).any()):
        raise ValueError('invalid causal public history')
    for prefix in ('', 'next_'):
        get = lambda name: arrays[prefix + name]
        if ((get('support') & ~get('visible')).any() or (get('semantic_valid') & ~get('support')).any()
                or (get('goal_attribute_valid') & ~(get('semantic_valid') & get('goal_presence') & ~get('goal_solved'))).any()
                or (get('goal_solved') & ~get('goal_presence')).any()):
            raise ValueError('inconsistent semantic supervision masks')
        if (get('goal_triple')[~get('goal_presence')] != -1).any():
            raise ValueError('absent goals require unknown attribute sentinel')
        for column, size in enumerate((6, 4, 4)):
            attrs = get('goal_triple')[..., column][get('goal_presence')]
            if ((attrs < 0) | (attrs >= size)).any() or ((get('triple')[..., column] < 0) | (get('triple')[..., column] >= size)).any():
                raise ValueError('invalid appearance category')
        if ((get('player_cell') < 0) | (get('player_cell') >= 12)).any() or ((get('lives') < 0) | (get('lives') > 3)).any():
            raise ValueError('invalid physical state')
    for values in ('current_distance', 'distances'):
        if ((arrays[values] < -1) | (arrays[values] > 128)).any():
            raise ValueError('distance outside bounded finite/unreachable classes')
    if (not arrays['current_distance_valid'].all()
            or not np.array_equal(arrays['distance_valid'], arrays['next_frame_valid'] & (~arrays['terminal'] | arrays['won']))):
        raise ValueError('invalid complete-proof distance validity')
    for values, mask in (('optimal', 'optimal_valid'), ('next_optimal', 'next_optimal_valid')):
        if (arrays[values] > 15).any() or not np.array_equal(arrays[mask], arrays[values] != 0):
            raise ValueError('invalid optimal action bitmask')
    if (not np.array_equal(arrays['next_history_reset'], arrays['lost_life'])
            or (arrays['won'] & ~arrays['terminal']).any()
            or (~arrays['next_frame_valid'] & ~arrays['terminal']).any()
            or arrays['next_frames'][~arrays['next_frame_valid']].any()):
        raise ValueError('invalid successor observation/event masks')
    by_seed = {}
    for spec in specs:
        check_spec(spec)
        if spec['seed'] in by_seed:
            raise ValueError('duplicate generated level seed')
        by_seed[spec['seed']] = spec
    if set(arrays['seeds'].tolist()) != set(by_seed):
        raise ValueError('array level identities differ from exact bank')
    for seed, context, collection in zip(arrays['seeds'], arrays['context_index'], arrays['collection']):
        if context != by_seed[int(seed)]['training_context_index'] or int(collection) not in COLLECTIONS.values():
            raise ValueError('row provenance differs from bank/schema')


def load_dataset(path):
    """Load bounded qualification arrays with bank/data/source hashes verified."""
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if (manifest.get('format') != FORMAT or manifest.get('status') != 'complete'
            or manifest.get('source') != 'generated_only' or manifest.get('split') != 'train'
            or manifest.get('qualification_only') is not True or manifest.get('sources_unchanged') is not True):
        raise ValueError('completed generated TRAIN qualification manifest required')
    def digest(file):
        with Path(file).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest(path / 'data.npz') != manifest.get('data_sha256') or digest(path / 'bank.json') != manifest.get('bank_sha256'):
        raise ValueError('qualification data/bank checksum differs')
    sources = manifest.get('sources', {})
    if not isinstance(sources, dict) or not {str(p.resolve()) for p in source_paths()}.issubset(sources):
        raise ValueError('dataset source bindings incomplete')
    for source, expected in sources.items():
        if digest(source) != expected:
            raise ValueError('qualification source changed: ' + source)
    specs = json.loads((path / 'bank.json').read_text())
    if not isinstance(specs, list) or len(specs) != 8 or manifest.get('requested_levels') != 8:
        raise ValueError('exact eight-level qualification bank required')
    with np.load(path / 'data.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    validate_arrays(arrays, specs)
    levels = manifest.get('levels', [])
    if len(levels) != 8 or {level['seed']: level['spec_sha256'] for level in levels} != {spec['seed']: spec_digest(spec) for spec in specs}:
        raise ValueError('per-level proofs differ from exact bank')
    if any(level.get('search_truncated') is not False or level.get('context_engine_verified') is not True
           or level.get('teacher_replay_won') is not True for level in levels):
        raise ValueError('incomplete generated teacher proof')
    actual_arrays = {key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in arrays.items()}
    if manifest.get('arrays') != actual_arrays or manifest.get('schema') != schema():
        raise ValueError('manifest tensor/schema declaration differs')
    return arrays, specs, manifest
