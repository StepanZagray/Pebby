"""Collect exploration trajectories from LS20 action-permutation variant games.

Each game fixes one of the 24 action permutations and plays one generated level
per tier (default tiers 1..7) sequentially on the real engine with competition
reset semantics (see ``pebby.variants``). The exploration policy takes a
uniformly random agent action with probability ``epsilon`` and otherwise the
oracle-optimal agent action (the exact planner's engine action pulled back
through the inverse permutation). Every step is recorded as factored
before/after states; one NPZ + JSON per game under ``OUT/games``, plus
``OUT/manifest.json``. Re-running skips games whose JSON says ``complete``.

Levels are never force-advanced: a level that is not won within
``--max-actions-per-level`` actions truncates the game.

Example::

    PYTHONPATH=. uv run python tools/collect_variant_games.py \
        --bank data/ls20-reference-unequal-v1/train.jsonl --out-dir /tmp/variant-pilot \
        --games-per-variant 2 --variants 0 5 --levels-per-tier 4 --workers 4 --seed 1
"""

import argparse
from collections import Counter, OrderedDict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pebby import variants as V  # noqa: E402
from pebby.ls20.bank import load as load_bank  # noqa: E402
from pebby.ls20.plan import Oracle  # noqa: E402
from pebby.ls20.reference_profiles import SEARCH_LIMITS  # noqa: E402

FORMAT = 'pebby.variant-games.v1'

# Per-step arrays, in the order they are written. Anyone reading the NPZ can code against this.
STEP_INT16_FIELDS = ('player_x', 'player_y', 'shape', 'color', 'rotation', 'steps_left', 'lives', 'goals_mask')
STEP_KEYS = (
    ('level_index', np.int8), ('step_in_level', np.int16),
    *[(f'before_{f}', np.int16) for f in STEP_INT16_FIELDS],
    ('neighbours', np.int8), ('agent_action', np.int8), ('engine_action', np.int8),
    *[(f'after_{f}', np.int16) for f in STEP_INT16_FIELDS],
    ('life_lost', np.bool_), ('level_changed', np.bool_), ('reset', np.bool_), ('won', np.bool_),
    ('finished', np.bool_),
    # Extras beyond the minimum contract: which rule chose the action and what the oracle wanted.
    ('action_source', np.int8),        # 0 random (epsilon), 1 oracle, 2 random because oracle had no route
    ('oracle_agent_action', np.int8),  # optimal AGENT action at the before-state, -1 if none
)
SCALAR_KEYS = ('game_id', 'variant_id', 'action_map', 'tier_seeds', 'tiers', 'truncated',
               'levels_completed', 'game_won')   # 'game_won' avoids clashing with the per-step 'won'
ACTION_RANDOM, ACTION_ORACLE, ACTION_FALLBACK = 0, 1, 2


def sha256_of(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


# -- oracle cache (one per worker process) ---------------------------------------

class OracleCache:
    """Bounded LRU of oracles keyed by (tier, seed, level_index); tiers 6-7 cost seconds and ~0.5 GB."""

    def __init__(self, capacity=16, engine='fast'):
        self.capacity = capacity
        self.engine = engine
        self._items = OrderedDict()
        self.build_seconds = []   # (tier, seed, seconds, reachable, truncated)

    def get(self, game):
        spec = game.current_spec()
        tier = int(spec['difficulty'])
        key = (tier, int(spec['seed']), game.level_index)
        oracle = self._items.get(key)
        if oracle is not None:
            self._items.move_to_end(key)
            return oracle
        started = time.perf_counter()
        oracle = Oracle(game.layout(), limit=SEARCH_LIMITS[tier - 1], engine=self.engine)
        elapsed = time.perf_counter() - started
        self.build_seconds.append(dict(tier=tier, seed=key[1], seconds=round(elapsed, 3),
                                       reachable=int(oracle._reachable), truncated=bool(oracle.truncated),
                                       solvable=bool(oracle.solvable)))
        self._items[key] = oracle
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)
        return oracle


_CACHE = None


def _cache(engine='fast', capacity=16):
    global _CACHE
    if _CACHE is None:
        _CACHE = OracleCache(capacity=capacity, engine=engine)
    return _CACHE


# -- one game --------------------------------------------------------------------

def play_game(specs, variant_id, *, rng, epsilon, max_actions_per_level, cache, tiers):
    """Run one game to WIN or truncation. Returns (arrays dict, summary dict)."""
    game = V.VariantGame(specs, variant_id)
    columns = {key: [] for key, _ in STEP_KEYS}
    per_level_actions = [0] * game.level_count
    truncated = False
    started = time.perf_counter()
    while not game.finished:
        level = game.level_index
        if per_level_actions[level] >= max_actions_per_level:
            truncated = True
            break
        oracle = cache.get(game)
        before = game.state()
        neighbours = game.neighbours()
        optimal = V.oracle_agent_action(oracle, oracle.state_of(game.env), variant_id)
        if rng.random() < epsilon:
            action, source = rng.randrange(4), ACTION_RANDOM
        elif optimal is None:
            action, source = rng.randrange(4), ACTION_FALLBACK
        else:
            action, source = optimal, ACTION_ORACLE
        after, info = game.step(action)
        per_level_actions[level] += 1

        columns['level_index'].append(level)
        columns['step_in_level'].append(per_level_actions[level])
        for field in STEP_INT16_FIELDS:
            columns[f'before_{field}'].append(before[field])
            columns[f'after_{field}'].append(after[field])
        columns['neighbours'].append(neighbours)
        columns['agent_action'].append(action)
        columns['engine_action'].append(info['engine_action'])
        for flag in ('life_lost', 'level_changed', 'reset', 'won', 'finished'):
            columns[flag].append(info[flag])
        columns['action_source'].append(source)
        columns['oracle_agent_action'].append(-1 if optimal is None else optimal)
    seconds = time.perf_counter() - started

    arrays = {}
    for key, dtype in STEP_KEYS:
        values = columns[key]
        if key == 'neighbours':
            arrays[key] = np.asarray(values, dtype=dtype).reshape(len(values), 4)
        else:
            arrays[key] = np.asarray(values, dtype=dtype)
    summary = dict(variant_id=int(variant_id), action_map=[int(a) for a in game.action_map],
                   tier_seeds=[int(spec['seed']) for spec in specs], tiers=[int(t) for t in tiers],
                   truncated=bool(truncated), levels_completed=int(game.levels_completed),
                   won=bool(game.won), steps=int(arrays['level_index'].shape[0]),
                   per_level_actions=per_level_actions, seconds=round(seconds, 3),
                   action_sources=dict(random=int((arrays['action_source'] == ACTION_RANDOM).sum()),
                                       oracle=int((arrays['action_source'] == ACTION_ORACLE).sum()),
                                       fallback=int((arrays['action_source'] == ACTION_FALLBACK).sum())),
                   lives_lost=int(arrays['life_lost'].sum()), resets=int(arrays['reset'].sum()))
    return arrays, summary


def _atomic_write_bytes(path, payload):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as handle:
        handle.write(payload)
    os.replace(tmp, path)


def save_game(games_dir, game_id, arrays, summary):
    games_dir = Path(games_dir)
    games_dir.mkdir(parents=True, exist_ok=True)
    scalars = dict(game_id=np.int64(game_id), variant_id=np.int8(summary['variant_id']),
                   action_map=np.asarray(summary['action_map'], dtype=np.int8),
                   tier_seeds=np.asarray(summary['tier_seeds'], dtype=np.int64),
                   tiers=np.asarray(summary['tiers'], dtype=np.int8),
                   truncated=np.bool_(summary['truncated']),
                   levels_completed=np.int8(summary['levels_completed']), game_won=np.bool_(summary['won']))
    import io
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays, **scalars)
    npz_path = games_dir / f'{game_id:06d}.npz'
    _atomic_write_bytes(npz_path, buffer.getvalue())
    record = dict(format=FORMAT, game_id=int(game_id), complete=True, npz=npz_path.name, **summary)
    json_path = games_dir / f'{game_id:06d}.json'
    _atomic_write_bytes(json_path, json.dumps(record, indent=1).encode())
    return record


def load_complete(games_dir, game_id):
    json_path = Path(games_dir) / f'{game_id:06d}.json'
    npz_path = Path(games_dir) / f'{game_id:06d}.npz'
    if not (json_path.exists() and npz_path.exists()):
        return None
    try:
        record = json.loads(json_path.read_text())
    except json.JSONDecodeError:
        return None
    return record if record.get('complete') else None


# -- worker ----------------------------------------------------------------------

def run_task(task):
    """Play and save one game. ``task`` is a plain dict so it pickles for spawn."""
    cache = _cache(task['oracle_engine'], task['oracle_cache'])
    rng = random.Random(f"variant-game:{task['seed']}:{task['game_id']}")  # str seeds hash deterministically
    marker = len(cache.build_seconds)
    arrays, summary = play_game(task['specs'], task['variant_id'], rng=rng, epsilon=task['epsilon'],
                                max_actions_per_level=task['max_actions_per_level'], cache=cache,
                                tiers=task['tiers'])
    summary['oracle_builds'] = cache.build_seconds[marker:]
    summary['worker_pid'] = os.getpid()
    return save_game(task['games_dir'], task['game_id'], arrays, summary)


# -- manifest --------------------------------------------------------------------

def build_manifest(args, bank_sha, pool, games, records, tiers):
    by_variant = Counter(r['variant_id'] for r in records)
    completed = Counter(r['levels_completed'] for r in records)
    builds = [b for r in records for b in r.get('oracle_builds', ())]
    tier_build = {}
    for build in builds:
        tier_build.setdefault(build['tier'], []).append(build['seconds'])
    return dict(
        format=FORMAT, bank=str(args.bank), bank_sha256=bank_sha, seed=args.seed, epsilon=args.epsilon,
        max_actions_per_level=args.max_actions_per_level, levels_per_tier=args.levels_per_tier,
        tiers=list(tiers), variants=list(args.variants), games_per_variant=args.games_per_variant,
        oracle_engine=args.oracle_engine,
        pool_seeds={str(tier): sorted(int(s['seed']) for s in pool if s['difficulty'] == tier) for tier in tiers},
        games_requested=len(games), games_complete=len(records),
        total_steps=int(sum(r['steps'] for r in records)),
        per_variant_games={str(v): by_variant.get(v, 0) for v in args.variants},
        levels_completed_histogram={str(k): completed[k] for k in sorted(completed)},
        truncated_games=int(sum(r['truncated'] for r in records)),
        won_games=int(sum(r['won'] for r in records)),
        seconds_per_game=[r['seconds'] for r in records],
        oracle_build_seconds_by_tier={str(t): dict(count=len(v), min=min(v), max=max(v),
                                                   mean=round(sum(v) / len(v), 3))
                                      for t, v in sorted(tier_build.items())},
        games=[dict(game_id=r['game_id'], variant_id=r['variant_id'], tier_seeds=r['tier_seeds'],
                    steps=r['steps'], levels_completed=r['levels_completed'], truncated=r['truncated'],
                    won=r['won'], seconds=r['seconds'], per_level_actions=r['per_level_actions'])
               for r in sorted(records, key=lambda r: r['game_id'])],
        step_keys=[key for key, _ in STEP_KEYS], scalar_keys=list(SCALAR_KEYS),
        tile_classes=list(V.TILE_CLASSES))


# -- CLI -------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bank', type=Path, default=Path('data/ls20-reference-unequal-v1/train.jsonl'))
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--games-per-variant', type=int, required=True)
    parser.add_argument('--variants', type=int, nargs='+', required=True)
    parser.add_argument('--levels-per-tier', type=int, default=30, help='per-tier pool size')
    parser.add_argument('--tiers', type=int, nargs='+', default=list(V.DEFAULT_TIERS),
                        help='tiers played in order (default 1..7; fewer only for tests)')
    parser.add_argument('--max-actions-per-level', type=int, default=150)
    parser.add_argument('--epsilon', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--oracle-engine', choices=('fast', 'auto', 'reference'), default='fast')
    parser.add_argument('--oracle-cache', type=int, default=16, help='oracles kept per worker (LRU)')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)
    if args.games_per_variant < 1 or args.levels_per_tier < 1 or args.max_actions_per_level < 1 \
            or args.workers < 1 or args.oracle_cache < 1:
        parser.error('counts must be positive')
    if not 0.0 <= args.epsilon <= 1.0:
        parser.error('--epsilon must be in [0, 1]')
    for variant in args.variants:
        if not 0 <= variant < V.VARIANT_COUNT:
            parser.error(f'variants must be in 0..{V.VARIANT_COUNT - 1}')
    if len(set(args.variants)) != len(args.variants):
        parser.error('variants must be distinct')
    if any(t not in V.DEFAULT_TIERS for t in args.tiers) or len(set(args.tiers)) != len(args.tiers):
        parser.error('tiers must be distinct values in 1..7')
    return args


def main(argv=None):
    args = parse_args(argv)
    tiers = tuple(args.tiers)
    out_dir = Path(args.out_dir)
    games_dir = out_dir / 'games'
    games_dir.mkdir(parents=True, exist_ok=True)
    log = (lambda *a, **k: None) if args.quiet else (lambda *a, **k: print(*a, **k, flush=True))

    started = time.perf_counter()
    bank_sha = sha256_of(args.bank)
    bank = load_bank(args.bank)
    rng = random.Random(args.seed)
    pool = V.pool_specs(bank, args.levels_per_tier, rng, tiers)
    games = V.sample_games(pool, args.games_per_variant, args.variants, rng, tiers)
    log(f'bank {args.bank} ({len(bank)} levels, sha256 {bank_sha[:12]}); pool '
        f'{ {t: sum(s["difficulty"] == t for s in pool) for t in tiers} }; {len(games)} games')

    records, tasks = [], []
    for game_id, (variant_id, seeds) in enumerate(games):
        existing = load_complete(games_dir, game_id)
        if existing is not None and existing['variant_id'] == variant_id and existing['tier_seeds'] == seeds:
            records.append(existing)
            continue
        tasks.append(dict(game_id=game_id, variant_id=variant_id, specs=V.game_specs(pool, seeds, tiers),
                          seed=args.seed, epsilon=args.epsilon, max_actions_per_level=args.max_actions_per_level,
                          tiers=list(tiers), games_dir=str(games_dir), oracle_engine=args.oracle_engine,
                          oracle_cache=args.oracle_cache))
    log(f'{len(records)} games already complete, {len(tasks)} to play with {args.workers} worker(s)')
    # Group games sharing the heaviest tiers' seeds so one worker's oracle cache serves them.
    tasks.sort(key=lambda t: ([int(s['seed']) for s in reversed(t['specs'])], t['game_id']))

    def report(record):
        records.append(record)
        builds = ', '.join(f"t{b['tier']}:{b['seconds']:.1f}s" for b in record.get('oracle_builds', ()))
        log(f"game {record['game_id']:06d} variant {record['variant_id']:2d} steps {record['steps']:5d} "
            f"levels {record['levels_completed']}/{len(tiers)} {'TRUNC' if record['truncated'] else 'win  '} "
            f"{record['seconds']:6.1f}s  oracle builds [{builds}]")

    if tasks:
        if args.workers == 1:
            for task in tasks:
                report(run_task(task))
        else:
            context = multiprocessing.get_context('spawn')
            chunksize = max(1, -(-len(tasks) // args.workers))
            with context.Pool(processes=min(args.workers, len(tasks))) as pool_:
                for record in pool_.imap_unordered(run_task, tasks, chunksize=chunksize):
                    report(record)

    manifest = build_manifest(args, bank_sha, pool, games, records, tiers)
    manifest['wall_seconds'] = round(time.perf_counter() - started, 3)
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    log(f"wrote {out_dir / 'manifest.json'}: {manifest['games_complete']} games, "
        f"{manifest['total_steps']} steps, levels completed {manifest['levels_completed_histogram']}, "
        f"{manifest['wall_seconds']:.1f}s")
    return manifest


if __name__ == '__main__':
    main()
