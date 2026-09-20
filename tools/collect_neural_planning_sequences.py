"""Bounded generated trajectories selected in the learned world model.

Actual branch observations become targets only. They never select imagined
continuation actions. Optional teacher endings use verified generated routes
only to reach collection roots; imagined actions remain model-selected. No
Oracle or shipped route is used.
"""
import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from pebby.agent.neural_imagination_policy import load_checkpoint
from pebby.agent.structured_factored_policy import state_digest
from pebby.agent.world_data import clone_env, history_arrays
from pebby.ls20 import names
from pebby.ls20.bank import load as load_bank
from pebby.ls20.env import Ls20Scenario
from pebby.ls20.generate import build_level
from pebby.ls20.provenance import DIFFICULTY_VERSION, generated_context, validate_difficulty
from tools.train_navigation_probe import Budget, digest, write_json
from tools.train_neural_imagination import stratified_rows


HORIZON = 4
HISTORY = 8


def raw_successor_schema():
    """Versioned public observations for re-encoding with a fresh encoder."""
    return dict(version=1, history=HISTORY, action_encoding='zero_based_up_down_left_right',
                next_frames='uint8[N,4,4,64,64]: actual successor public frame; zero when invalid',
                next_frame_valid='bool[N,4,4]: actual frame exists, including an emitted terminal frame',
                next_history_reset='bool[N,4,4]: life-loss history boundary; false on transition padding',
                reconstruction='Start each branch from valid root frames/previous_actions; append each valid successor with actions[branch,step], retaining H8. On next_history_reset clear to successor only with action -1. Missing frames have no successor history; never append zero padding.',
                target_semantics='Re-encode reconstructed successor public histories with the fresh training encoder/target encoder. Cached fields/next_fields remain frozen-encoder outputs, not fresh targets.',
                input_boundary='Only frames, history_valid, previous_actions enter an encoder. Successor observations are training targets, never current policy inputs; state/event labels never become feature channels.')


def reconstruct_successor_history(row, branch, step):
    """Return one causal successor H8 as (frames, valid, previous), or None.

    Operates on a single unbatched v2 row and only its public observations,
    recorded actions and history-boundary masks. Each branch starts privately
    from the root. A terminal observation may exist; missing terminal frames
    and padded tails produce no encoder input. Frozen fields are not read.
    """
    if not 0 <= branch < 4 or not 0 <= step < HORIZON:
        raise ValueError('branch and step must be within the stored four K4 traces')
    if not row['next_frame_valid'][branch, step]:
        return None
    valid = row['history_valid']
    history = (list(row['frames'][valid]), list(row['previous_actions'][valid]))
    for index in range(step + 1):
        if not row['next_frame_valid'][branch, index]:
            raise ValueError('a valid successor cannot follow a missing branch frame')
        history = observe(history, row['next_frames'][branch, index],
                          int(row['actions'][branch, index]),
                          reset=bool(row['next_history_reset'][branch, index]))
    return history_arrays(*history, HISTORY)


def checked_specs(path, split='train'):
    if split not in ('train', 'validation'):
        raise ValueError('split must be train or validation')
    low, high = (0, 1_000_000) if split == 'train' else (1_000_000, 2_000_000)
    specs = load_bank(path)
    seeds = set()
    for spec in specs:
        seed = spec.get('seed')
        if (type(seed) is not int or not low <= seed < high or seed in seeds
                or spec.get('source') != 'generated_only' or spec.get('split') != split
                or spec.get('generator_version') != 3 or spec.get('geometry_split') != split
                or spec.get('difficulty_version') != DIFFICULTY_VERSION
                or spec.get('context_engine_verified') is not True or spec.get('search_truncated') is not False):
            raise ValueError('distinct verified generator-v3 seven-tier specs in the requested split required')
        if (type(spec.get('training_context_index')) is not int
                or spec.get('training_context_index') != generated_context(spec)
                or spec.get('context_index') != generated_context(spec)):
            raise ValueError('declared context differs from calibrated generated context')
        validate_difficulty(spec)
        seeds.add(seed)
    if {spec['difficulty'] for spec in specs} != set(range(1, 8)):
        raise ValueError('bank must cover all seven tiers')
    return specs


def observe(history, frame, action, *, reset=False):
    """Return private causal buffers; a life reset starts a new H8 history."""
    if frame is None:
        raise ValueError('cannot observe a missing public frame')
    if reset:
        return [frame], [-1]
    frames, actions = history
    return (frames + [frame])[-HISTORY:], (actions + [action])[-HISTORY:]


@torch.inference_mode()
def encode_history(encoder, history):
    public = history_arrays(*history, HISTORY)
    field = encoder(*(torch.as_tensor(value)[None] for value in public))
    if tuple(field.shape) != (1, 148, 96) or not torch.isfinite(field).all():
        raise ValueError('public encoder must return finite [1,148,96] fields')
    return field


def state_labels(env):
    return dict(player_cell=np.asarray(env.player_cell(), dtype=np.int16),
                triple=np.asarray(env.triple(), dtype=np.int16),
                steps=np.asarray(env.steps_left(), dtype=np.int16),
                lives=np.asarray(env.lives(), dtype=np.int16))


@torch.inference_mode()
def collect_root(policy, env, history, guard=lambda: None, *, raw_successor_public=False):
    """Record four fixed K4 action traces and chronologically encoded targets."""
    current = encode_history(policy.encoder, history)
    guard()
    imagined = policy.planner.imagine(current, horizon=HORIZON)
    root_actions = imagined['root_actions'][0].detach().cpu().numpy()
    actions = imagined['imagined_actions'][0].detach().cpu().numpy().copy()
    if (root_actions.shape != (4,) or not np.array_equal(np.sort(root_actions), np.arange(4))
            or actions.shape != (4, HORIZON) or not np.issubdtype(actions.dtype, np.integer)
            or ((actions < 0) | (actions > 3)).any() or not np.array_equal(actions[:, 0], root_actions)):
        raise ValueError('imagination must supply four permuted roots and integer K4 traces')
    # Save all rows in canonical first-action order, independently of planner batching.
    actions = actions[np.argsort(root_actions)].astype(np.int64, copy=True)
    result = dict(fields=current[0].float().cpu().numpy().copy(), actions=actions,
                  next_fields=np.zeros((4, HORIZON, 148, 96), np.float32),
                  transition_valid=np.zeros((4, HORIZON), bool),
                  next_field_valid=np.zeros((4, HORIZON), bool),
                  next_player_cell=np.zeros((4, HORIZON, 2), np.int16),
                  next_triple=np.zeros((4, HORIZON, 3), np.int16),
                  next_steps=np.zeros((4, HORIZON), np.int16),
                  next_lives=np.zeros((4, HORIZON), np.int16),
                  **{key: np.zeros((4, HORIZON), bool) for key in ('lost_life', 'terminal', 'won')},
                  **state_labels(env))
    public = history_arrays(*history, HISTORY)
    result.update(zip(('frames', 'history_valid', 'previous_actions'), public))
    if raw_successor_public:
        result.update(next_frames=np.zeros((4, HORIZON, 64, 64), np.uint8),
                      next_frame_valid=np.zeros((4, HORIZON), bool),
                      next_history_reset=np.zeros((4, HORIZON), bool))
    for root, trace in enumerate(actions):
        branch = clone_env(env)
        branch_history = (list(history[0]), list(history[1]))
        for step, action in enumerate(trace):
            guard()
            old_lives = branch.lives()
            observation = branch.perform(names.ACTION_IDS[int(action)])
            lost_life = branch.lives() < old_lives
            result['transition_valid'][root, step] = True
            if raw_successor_public:
                result['next_history_reset'][root, step] = lost_life
            for key, value in state_labels(branch).items():
                result['next_' + key][root, step] = value
            for key, value in (('lost_life', lost_life), ('terminal', observation.finished), ('won', observation.won)):
                result[key][root, step] = bool(value)
            if observation.frame is not None:
                if raw_successor_public:
                    frame = np.asarray(observation.frame)
                    if (frame.shape != (64, 64) or not np.issubdtype(frame.dtype, np.integer)
                            or ((frame < 0) | (frame > 15)).any()):
                        raise ValueError('successor public frame must be a 64x64 color-index grid')
                    result['next_frames'][root, step] = frame
                    result['next_frame_valid'][root, step] = True
                branch_history = observe(branch_history, observation.frame, int(action), reset=lost_life)
                result['next_fields'][root, step] = encode_history(policy.encoder, branch_history)[0].float().cpu().numpy()
                result['next_field_valid'][root, step] = True
            elif not observation.finished:
                raise ValueError('missing public frame on a live branch')
            if observation.finished:
                break
    return result


@torch.inference_mode()
def collect_level(policy, spec, root_steps, random_fraction, rng, guard=lambda: None,
                  *, behavior_actions=96, max_roots=8, raw_successor_public=False):
    env = Ls20Scenario(build_level(spec), generated_context(spec))
    history = ([env.reset()], [-1])
    rows, behavior, root_reasons = [], [], []
    budget_lives = set()
    ending = 'action_bound'
    for step in range(behavior_actions + 1):
        guard()
        # Engine budget is a collection selector only, never a model input or
        # an action override. Capture each life's first near-exhaustion root.
        near_budget = env.steps_left() <= 4 and env.lives() not in budget_lives
        reasons = (['scheduled'] if step in root_steps else []) + (['near_budget'] if near_budget else [])
        if reasons and len(rows) < max_roots:
            row = collect_root(policy, env, history, guard, **({'raw_successor_public': True} if raw_successor_public else {}))
            row.update(seeds=np.asarray(spec['seed'], np.int64), difficulties=np.asarray(spec['difficulty'], np.int8),
                       context_index=np.asarray(generated_context(spec), np.int8), root_step=np.asarray(step, np.int16))
            rows.append(row)
            root_reasons.append(dict(step=step, reasons=reasons, lives=int(env.lives()), budget=int(env.steps_left())))
            if near_budget:
                budget_lives.add(env.lives())
        if step == behavior_actions:
            break
        field = encode_history(policy.encoder, history)
        # State labels and stored routes are unavailable to this behavior decision.
        logits = policy.planner.continuation_logits(field)
        if tuple(logits.shape) != (1, 4) or not torch.isfinite(logits).all():
            raise ValueError('continuation policy must return finite [1,4] logits')
        random_action = bool(rng.random() < random_fraction)
        action = int(rng.integers(4)) if random_action else int(logits[0].argmax())
        old_lives = env.lives()
        observation = env.perform(names.ACTION_IDS[action])
        behavior.append(dict(action=action, random=random_action, lost_life=env.lives() < old_lives,
                             terminal=bool(observation.finished), won=bool(observation.won)))
        if observation.finished:
            ending = 'won' if observation.won else 'game_over'
            break
        if observation.frame is None:
            raise ValueError('missing public frame on live behavior trajectory')
        history = observe(history, observation.frame, action, reset=env.lives() < old_lives)
    return rows, dict(seed=spec['seed'], difficulty=spec['difficulty'], context_index=generated_context(spec),
                      root_steps=[int(row['root_step']) for row in rows], root_reasons=root_reasons,
                      root_limit_reached=len(rows) == max_roots, behavior=behavior, ending=ending)


@torch.inference_mode()
def collect_teacher_endings(policy, spec, guard=lambda: None, *, raw_successor_public=False):
    """Reach last-four/last-one roots via generated teacher behavior only.

    ``context_solution`` contains engine action IDs 1..4. At each root the
    unchanged collect_root API gets public history, never the remaining route.
    The complete supplied route must win in the actual contextual engine.
    """
    supplied = spec.get('context_solution')
    length = spec.get('context_optimal_actions')
    if (not isinstance(supplied, list) or type(length) is not int or length < 1
            or len(supplied) != length
            or any(type(action) is not int or action not in names.ACTION_IDS for action in supplied)):
        raise ValueError('context_solution must contain exactly context_optimal_actions valid engine action IDs')
    route = tuple(supplied)
    roots = sorted({max(0, length - HORIZON), length - 1})
    env = Ls20Scenario(build_level(spec), generated_context(spec))
    history = ([env.reset()], [-1])
    rows, behavior = [], []
    for step, engine_action in enumerate(route):
        guard()
        if step in roots:
            # Teacher suffix is deliberately absent from this call.
            row = collect_root(policy, env, history, guard, **({'raw_successor_public': True} if raw_successor_public else {}))
            row.update(seeds=np.asarray(spec['seed'], np.int64), difficulties=np.asarray(spec['difficulty'], np.int8),
                       context_index=np.asarray(generated_context(spec), np.int8), root_step=np.asarray(step, np.int16))
            if step == length - 1 and not bool((row['won'][:, 0] & row['transition_valid'][:, 0]).any()):
                raise ValueError('last-one teacher root lacks the required real winning first-action branch')
            rows.append(row)
        old_lives = env.lives()
        observation = env.perform(engine_action)
        action = names.ACTION_IDS.index(engine_action)
        lost_life = env.lives() < old_lives
        behavior.append(dict(action=action, teacher=True, lost_life=lost_life,
                             terminal=bool(observation.finished), won=bool(observation.won)))
        if observation.finished and (step != length - 1 or not observation.won):
            raise ValueError('generated teacher route terminated before its declared successful ending')
        if step == length - 1 and not (observation.finished and observation.won):
            raise ValueError('generated teacher route failed to win at its declared final action')
        if observation.frame is not None:
            history = observe(history, observation.frame, action, reset=lost_life)
        elif not observation.finished:
            raise ValueError('missing public frame during live teacher-prefix replay')
    return rows, dict(seed=spec['seed'], difficulty=spec['difficulty'], context_index=generated_context(spec),
                      root_steps=roots, teacher_prefix=True, stored_solutions_used=True,
                      teacher_route_actions=length, teacher_route_verified_won=True,
                      root_reasons=[dict(step=step, reasons=[reason for reason, index in
                                    [('teacher_last_four', max(0, length - HORIZON)), ('teacher_last_one', length - 1)]
                                    if step == index]) for step in roots],
                      behavior=behavior, ending='won')


def summarize(arrays):
    valid = arrays['transition_valid']
    return dict(roots=len(valid), distinct_levels=len(np.unique(arrays['seeds'])),
                transitions_by_horizon=valid.sum((0, 1)).tolist(),
                next_fields_by_horizon=arrays['next_field_valid'].sum((0, 1)).tolist(),
                events={key: (arrays[key] & valid).sum((0, 1)).tolist() for key in ('lost_life', 'terminal', 'won')},
                root_steps=dict(Counter(map(str, arrays['root_step'].tolist()))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--split', choices=('train', 'validation'), default='train')
    parser.add_argument('--bank', type=Path, help='defaults to data/ls20-reference-unequal-v1/<split>.jsonl')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--levels', type=int, default=14)
    parser.add_argument('--root-steps', nargs='+', type=int, default=[0, 12, 24, 36, 48])
    parser.add_argument('--behavior-actions', type=int, default=96)
    parser.add_argument('--max-roots', type=int, default=8)
    parser.add_argument('--random-fraction', type=float, default=.25)
    parser.add_argument('--teacher-endings-only', action='store_true',
                        help='use verified generated context_solution prefixes to reach last-four/last-one roots; ignore behavior/root options')
    parser.add_argument('--raw-successor-public', action='store_true',
                        help='write v2 raw successor frames and causal history masks for fresh joint encoder training')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-seconds', type=int, default=600)
    args = parser.parse_args(argv)
    if args.bank is None:
        args.bank = Path('data/ls20-reference-unequal-v1') / (args.split + '.jsonl')
    if not 7 <= args.levels <= 64 or not 1 <= args.max_seconds <= 600:
        parser.error('levels must be 7..64 and seconds 1..600')
    if not args.teacher_endings_only and (not args.root_steps or args.root_steps != sorted(set(args.root_steps))
            or args.root_steps[0] != 0 or args.root_steps[-1] > args.behavior_actions or len(args.root_steps) > 5
            or not 1 <= args.behavior_actions <= 96 or not len(args.root_steps) <= args.max_roots <= 8
            or not np.isfinite(args.random_fraction) or not 0 <= args.random_fraction <= 1):
        parser.error('levels 7..64; <=5 sorted roots starting 0 within behavior bound 1..96; roots<=8; fraction 0..1; seconds 1..600')
    if torch.cuda.is_initialized():
        raise RuntimeError('fresh CPU process required')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    if digest(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError('checkpoint SHA256 differs')
    args.out.mkdir(parents=True, exist_ok=False)
    guard = Budget(args.max_seconds, reserve_gib=7)
    encoder_paths = [Path('pebby/agent') / name for name in
                     ('structured_field.py', 'world_model.py', 'world_readout.py', 'world_grounding.py', 'world_rollout.py',
                      'cell_appearance.py', 'cell_appearance_dense.py', 'glyph_model.py', 'cell_visibility.py')]
    source_paths = [Path(__file__), Path('tools/train_neural_imagination.py'), Path('tools/train_navigation_probe.py'),
                    *[Path('pebby/agent') / name for name in ('neural_imagination.py', 'neural_imagination_policy.py', 'world_data.py')],
                    *encoder_paths,
                    *[Path('pebby/ls20') / name for name in ('env.py', 'names.py', 'generate.py', 'bank.py', 'provenance.py')],
                    Path('third_party/ls20/ls20.py'), args.bank, args.checkpoint]
    sources = {str(path.resolve()): digest(path) for path in source_paths}
    manifest = dict(format='pebby.neural-planning-sequences.v2' if args.raw_successor_public else 'pebby.neural-planning-sequences.v1', status='running', pid=os.getpid(),
                    source='generated_only', split=args.split, generator_version=3, difficulty_version=DIFFICULTY_VERSION,
                    started_local=datetime.now().astimezone().isoformat(), cpu_threads=1, device='cpu',
                    checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=args.checkpoint_sha256,
                    sources=sources, levels=[], horizon=HORIZON, requested_levels=args.levels,
                    root_steps=None if args.teacher_endings_only else args.root_steps,
                    random_fraction=None if args.teacher_endings_only else args.random_fraction, selection_seed=args.seed,
                    behavior_actions=None if args.teacher_endings_only else args.behavior_actions,
                    max_roots_per_level=2 if args.teacher_endings_only else args.max_roots,
                    collection_kind='teacher_endings' if args.teacher_endings_only else 'onpolicy',
                    teacher_prefix=args.teacher_endings_only,
                    ignored_options=['root_steps', 'behavior_actions', 'max_roots', 'random_fraction'] if args.teacher_endings_only else [],
                    opportunistic_roots=None if args.teacher_endings_only else 'First remaining-budget<=4 observation per life, subject to root bound; collection selection only.',
                    official_inputs_used=False, oracle_invoked=False, stored_solutions_used=args.teacher_endings_only,
                    limits=['Collector only: no training or gameplay benchmark; validation outputs must never enter training.',
                            'All four root traces are chosen from imagined fields before their actual replay.',
                            'Exact labels never enter any neural policy or encoder input.',
                            'Teacher-ending mode uses generated routes only for real prefix behavior; no teacher suffix enters imagination.',
                            'Each real branch has private causal H8 history, reset after actual life loss.',
                            'Use transition_valid for exact labels; use next_field_valid for field targets. Zero padding is never a target.',
                            'Life-loss transitions remain; actual terminal tails are masked, including missing terminal frames.',
                            'Near-exhaustion collection does not guarantee life-loss positives: inspect realized event counts.',
                            'Multiple roots per level are correlated; minibatches must account for repeated level identities.',
                            'Persistent memory remains unimplemented; collecting reset labels does not establish observability from H8.'])
    if args.raw_successor_public:
        manifest['raw_successor_public'] = raw_successor_schema()
    temporary = args.out / '.sequences.npz.tmp'
    def persist():
        manifest['elapsed_seconds'] = time.monotonic() - guard.started
        write_json(args.out / 'manifest.json', manifest)
    def timeout(*_):
        raise TimeoutError('planning collection wall-clock budget exhausted')
    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.max_seconds)
    print(f'PID {os.getpid()}', flush=True)
    persist()
    try:
        specs = checked_specs(args.bank, args.split)
        selected = stratified_rows(np.asarray([spec['difficulty'] for spec in specs]), args.levels, args.seed)
        manifest['selected_rows'] = selected.tolist()
        manifest['difficulty_counts'] = dict(Counter(str(specs[int(index)]['difficulty']) for index in selected))
        policy, _ = load_checkpoint(args.checkpoint, 'cpu')
        if policy.planner.cfg.horizon != HORIZON:
            raise ValueError('a K4 imagination checkpoint is required')
        manifest['field_encoder'] = policy.encoder.metadata()
        manifest['field_encoder']['code_hashes'] = {str(path.resolve()): sources[str(path.resolve())] for path in encoder_paths}
        checkpoint_paths = [policy.encoder.world_checkpoint_path, policy.encoder.visibility_checkpoint_path]
        manifest['field_encoder']['checkpoint_hashes'] = {str(path.resolve()): digest(path) for path in checkpoint_paths}
        sources.update(manifest['field_encoder']['checkpoint_hashes'])
        manifest['encoder_state_sha256'] = state_digest(policy.encoder.state_dict())
        manifest['dynamics_state_sha256'] = state_digest(policy.planner.dynamics.state_dict())
        manifest['dynamics_config'] = policy.planner.dynamics.config()
        manifest['continuation_state_sha256'] = state_digest(policy.planner.continuation.state_dict())
        rng = np.random.default_rng(args.seed)
        records = []
        for index in selected:
            guard()
            if args.teacher_endings_only:
                rows, record = collect_teacher_endings(policy, specs[int(index)], guard,
                                                       raw_successor_public=args.raw_successor_public)
            else:
                rows, record = collect_level(policy, specs[int(index)], args.root_steps, args.random_fraction, rng, guard,
                                             behavior_actions=args.behavior_actions, max_roots=args.max_roots,
                                             raw_successor_public=args.raw_successor_public)
            records.extend(rows)
            manifest['levels'].append(record)
            persist()
        guard()
        arrays = {key: np.stack([row[key] for row in records]) for key in records[0]}
        manifest['coverage'] = summarize(arrays)
        manifest['arrays'] = {key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in arrays.items()}
        with temporary.open('xb') as stream:
            np.savez(stream, **arrays)
        for path, expected in sources.items():
            guard()
            if digest(path) != expected:
                raise ValueError('collection source changed: ' + path)
        target = args.out / 'sequences.npz'
        temporary.rename(target)
        manifest.update(status='complete', data_file=target.name, data_sha256=digest(target), sources_unchanged=True)
    except BaseException as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        temporary.unlink(missing_ok=True)
        manifest['finished_local'] = datetime.now().astimezone().isoformat()
        persist()
    return manifest


if __name__ == '__main__':
    main()
