"""Read-only integrity and coverage audit of regenerated v2 training banks.

Structural checks inspect stored proofs. Optional bounded spotchecks independently
re-search optimal length and replay the stored route in its real engine context.
No controller is trained or evaluated by this tool.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from pebby.ls20 import names
from pebby.ls20.extended_curriculum import gameplay_hash
from pebby.ls20.generate import FORMAT, GENERATOR_VERSION, RESERVED
from pebby.ls20.generation_quality import budget_floor, geometry_partition, route_budget_slack


BOARD = {(x, y) for x in range(12) for y in range(12)}
KINDS = ('shape', 'color', 'rotation')


def _key(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def _distribution(values):
    values = list(values)
    return dict(count=len(values), min=min(values), median=statistics.median(values),
                max=max(values), mean=statistics.mean(values)) if values else None


def _version(row):
    # Training and pilot strings intentionally differ. Their quality and sprite
    # generator versions, rather than those labels, define compatibility.
    if 'extended_curriculum_version' in row:
        return ('extended', row['generator_version'], row['extended_curriculum_version'])
    if 'quality_version' in row:
        return ('mechanism', row['generator_version'], row['quality_version'])
    raise ValueError('missing supported curriculum quality version')


def _features(row):
    free = BOARD - {tuple(cell) for cell in row['walls']}
    left, top = min(x for x, _ in free), min(y for _, y in free)
    def normalized(cells):
        return sorted((x-left, y-top) for x, y in cells)
    rail_cells = {tuple(cell) for rail in row.get('rails', []) for cell in rail['cells']}
    moving = [cycler for cycler in row['cyclers'] if tuple(cycler['cell']) in rail_cells]
    mechanics = row['solution_mechanics']
    neighbors = [sum((x+dx, y+dy) in free for dx, dy in names.ACTION_DELTAS) for x, y in free]
    return dict(
        attributes=sum(any(goal['triple'][i] != row['start_triple'][i] for goal in row['goals'])
                       for i in range(3)),
        free_cells=len(free), corridor_fraction=sum(n <= 2 for n in neighbors)/len(free),
        moving_installed=len(moving), moving_kinds=len({cycler['kind'] for cycler in moving}),
        moving_used=mechanics.get('used_patroller_count', mechanics.get('distinct_moving_cyclers')),
        launchers_installed=len(row.get('launchers', [])),
        launchers_used=mechanics.get('used_launcher_count', mechanics.get('distinct_launchers')),
        start_normalized=(row['start'][0]-left, row['start'][1]-top),
        goals_normalized=normalized([goal['cell'] for goal in row['goals']]),
        refills_normalized=normalized(row['refills']),
        rails_normalized=sorted(normalized(rail['cells']) for rail in row.get('rails', [])),
    )


def _actions(rows):
    marginal, first, last, transitions = Counter(), Counter(), Counter(), defaultdict(Counter)
    for row in rows:
        route = row['solution']
        marginal.update(route)
        if route:
            first[route[0]] += 1
            last[route[-1]] += 1
        # Never create a transition from one episode's last action to the next.
        for before, after in zip(route, route[1:]):
            transitions[before][after] += 1
    count, pairs = sum(marginal.values()), sum(sum(v.values()) for v in transitions.values())
    successor_marginal = sum(transitions.values(), Counter())
    return dict(marginal=dict(marginal), first=dict(first), last=dict(last),
                action_count=count, within_episode_transition_count=pairs,
                transition_counts={str(k): dict(v) for k, v in transitions.items()},
                best_constant_accuracy=max(marginal.values())/count if count else None,
                successor_best_constant_accuracy=max(successor_marginal.values())/pairs if pairs else None,
                previous_action_markov_accuracy=sum(max(v.values()) for v in transitions.values())/pairs if pairs else None,
                interpretation='In-sample descriptive autocorrelation; not a controller score or proof of poor learning.')


def _summary(rows):
    features = [_features(row) for row in rows]
    result = dict(levels=len(rows), route_actions=_distribution(row['optimal_actions'] for row in rows),
                  final_slack_moves=_distribution(row['slack_moves'] for row in rows),
                  minimum_slack_moves=_distribution(row['minimum_slack_moves'] for row in rows),
                  changing_attributes=dict(Counter(f['attributes'] for f in features)),
                  contexts=dict(Counter(row['training_context_index'] for row in rows)),
                  quality_profiles={}, difficulty={}, actions=_actions(rows))
    for profile in sorted({row['quality_profile'] for row in rows}):
        selected = [row for row in rows if row['quality_profile'] == profile]
        result['quality_profiles'][profile] = dict(levels=len(selected), floor=budget_floor(selected[0]),
            minimum_slack_moves=_distribution(row['minimum_slack_moves'] for row in selected))
    for difficulty in sorted({row['difficulty'] for row in rows}):
        selected = [row for row in rows if row['difficulty'] == difficulty]
        result['difficulty'][str(difficulty)] = dict(levels=len(selected),
            route_actions=_distribution(row['optimal_actions'] for row in selected),
            minimum_slack_moves=_distribution(row['minimum_slack_moves'] for row in selected),
            reachable_states=_distribution(row['reachable_states'] for row in selected))
    result['installed_vs_used'] = {
        'moving_cyclers': dict(installed_total=sum(f['moving_installed'] for f in features),
                              used_total=sum(f['moving_used'] for f in features),
                              installed_counts=dict(Counter(f['moving_installed'] for f in features)),
                              used_counts=dict(Counter(f['moving_used'] for f in features))),
        'launchers': dict(installed_total=sum(f['launchers_installed'] for f in features),
                          used_total=sum(f['launchers_used'] for f in features),
                          installed_counts=dict(Counter(f['launchers_installed'] for f in features)),
                          used_counts=dict(Counter(f['launchers_used'] for f in features))),
    }
    result['thresholds'] = dict(
        refills_ge3=sum(len(row['refills']) >= 3 for row in rows),
        refills_ge6=sum(len(row['refills']) >= 6 for row in rows),
        installed_moving_kinds_eq3=sum(f['moving_kinds'] == 3 for f in features),
        three_moving_kinds_all_used=sum(f['moving_kinds'] == 3 and f['moving_used'] == f['moving_installed']
                                      for f in features),
        launchers_ge3=sum(f['launchers_installed'] >= 3 for f in features),
        launchers_ge8=sum(f['launchers_installed'] >= 8 for f in features),
        cost2_route_ge39=sum(row['step_cost'] == 2 and row['optimal_actions'] >= 39 for row in rows),
        route_ge72=sum(row['optimal_actions'] >= 72 for row in rows),
        large_corridor_route=sum(f['free_cells'] >= 57 and f['corridor_fraction'] >= .35
                                and row['optimal_actions'] >= 39 for row, f in zip(rows, features)),
    )
    result['per_mode'] = {}
    for mode in sorted({row.get('pilot_mode', 'extended') for row in rows}):
        pairs = [(row, f) for row, f in zip(rows, features) if row.get('pilot_mode', 'extended') == mode]
        selected = [row for row, _ in pairs]
        result['per_mode'][mode] = dict(levels=len(pairs),
            normalized_geometries=len({geometry_partition(row)[0] for row in selected}),
            distinct_starts=len({_key(row['start']) for row in selected}),
            distinct_normalized_starts=len({_key(f['start_normalized']) for _, f in pairs}),
            distinct_goal_placements=len({_key(sorted(goal['cell'] for goal in row['goals'])) for row in selected}),
            distinct_normalized_goal_placements=len({_key(f['goals_normalized']) for _, f in pairs}),
            distinct_refill_placements=len({_key(sorted(row['refills'])) for row in selected}),
            distinct_normalized_refill_placements=len({_key(f['refills_normalized']) for _, f in pairs}),
            distinct_normalized_rail_patterns=len({_key(f['rails_normalized']) for row, f in pairs if row.get('rails')}),
            route_actions=_distribution(row['optimal_actions'] for row in selected))
    return result


def audit(trainrows, valrows, min_validation=500):
    """Audit in-memory rows without mutating them or invoking a search."""
    if min_validation < 0:
        raise ValueError('min_validation must be nonnegative')
    errors, summaries, identities, versions = [], {}, {}, {}
    for split, rows in (('train', trainrows), ('validation', valrows)):
        identities[split] = {key: set() for key in ('seeds', 'gameplay', 'geometry')}
        versions[split] = set()
        valid = []
        for number, row in enumerate(rows):
            if not isinstance(row, dict):
                errors.append(f'{split} row {number}: expected a level object')
                continue
            label = f'{split} row {number} seed {row.get("seed", "missing")}'
            try:
                geometry, partition = geometry_partition(row)
                fingerprint = gameplay_hash(row)
                for key, value in (('seeds', row['seed']), ('gameplay', fingerprint), ('geometry', geometry)):
                    if key != 'geometry' and value in identities[split][key]:
                        errors.append(f'{label}: duplicate {key} within split')
                    identities[split][key].add(value)
                version = _version(row)
                versions[split].add(version)
                if version[1:] != (GENERATOR_VERSION, 2):
                    raise ValueError(f'requires sprite generator v{GENERATOR_VERSION} and curriculum quality v2')
                if row.get('format') != FORMAT or row.get('size') != 64:
                    raise ValueError('unexpected level format/size')
                if row.get('split', split) != split:
                    raise ValueError('declared split mismatch')
                if (partition != split or row.get('geometry_split') != partition
                        or row.get('geometry_sha256') != geometry):
                    raise ValueError('normalized geometry partition proof mismatch')
                if 'gameplay_sha256' in row and row['gameplay_sha256'] != fingerprint:
                    raise ValueError('gameplay hash mismatch')
                if (row.get('search_truncated') is not False or row.get('engine_verified') is not True
                        or row.get('context_engine_verified') is not True or row.get('verification_lives') != 3
                        or row.get('oracle_backend') not in ('fast', 'reference')
                        or not 0 < row['reachable_states'] <= row['search_limit'] <= 600000):
                    raise ValueError('incomplete contextual search/engine proof')
                if (row['training_context_index'] != row['seed'] % 7
                        or row['verification_level_index'] != row['seed'] % 7):
                    raise ValueError('stored context disagrees with seed context')
                if row['seed'] % 7 == 0 and row.get('launchers'):
                    raise ValueError('unsupported context-zero launcher combination')
                if (len(row['solution']) != row['optimal_actions'] or row['optimal_actions'] != row['context_optimal_actions']
                        or any(action not in names.ACTION_IDS for action in row['solution'])):
                    raise ValueError('stored route/context optimal length mismatch or invalid action')
                route_floor = 72 if row.get('pilot_mode') == 'long_route' else 39 if row.get('pilot_mode') == 'refill_chain' else 10
                if row['optimal_actions'] < route_floor:
                    raise ValueError('route below mode quality floor')
                if row['step_counter'] != 42 or row['step_cost'] not in (1, 2):
                    raise ValueError('unsupported transfer budget/cost')
                floor = budget_floor(row)
                if (row.get('quality_profile') not in ('learning', 'challenge') or row.get('budget_floor') != floor
                        or not floor <= row['minimum_slack_moves'] <= row['slack_moves'] <= 42//row['step_cost']):
                    raise ValueError('headroom/profile proof mismatch')
                if not row['goals'] or any(tuple(goal['triple']) == tuple(row['start_triple'])
                                           or tuple(goal['cell']) == tuple(row['start']) for goal in row['goals']):
                    raise ValueError('missing goal or goal satisfied at spawn')
                free = BOARD - {tuple(cell) for cell in row['walls']}
                entities = [row['start'], *row['refills'], *(g['cell'] for g in row['goals']),
                            *(c['cell'] for c in row['cyclers']), *(p['cell'] for p in row.get('launchers', []))]
                if not RESERVED <= set(map(tuple, row['walls'])) or any(tuple(cell) not in free for cell in entities):
                    raise ValueError('reserved or blocked entity cell')
                features = _features(row)
                if row.get('free_cells') != len(free) or row.get('changing_attributes') != features['attributes']:
                    raise ValueError('free-cell/attribute count mismatch')
                for kind in ('moving', 'launchers'):
                    used = features[f'{kind}_used']
                    if not isinstance(used, int) or not 0 <= used <= features[f'{kind}_installed']:
                        raise ValueError(f'missing or invalid distinct {kind} contact count')
                if row.get('pilot_mode') == 'three_rails' and (features['moving_kinds'] != 3 or features['moving_used'] != 3):
                    raise ValueError('three-rail mode lacks three distinct used moving kinds')
                if row.get('distractor_count', 0) < 1:
                    raise ValueError('missing unused distractor proof')
                valid.append(row)
            except (KeyError, ValueError, TypeError, IndexError) as error:
                errors.append(f'{label}: {error}')
        summaries[split] = _summary(valid)
        summaries[split]['input_levels'] = len(rows)
        summaries[split]['distinct'] = {key: len(values) for key, values in identities[split].items()}
    overlap = {key: len(identities['train'][key] & identities['validation'][key]) for key in identities['train']}
    for key, count in overlap.items():
        if count:
            errors.append(f'train/validation {key} overlap: {count}')
    if versions['train'] != versions['validation']:
        errors.append('train/validation generator quality versions do not match')
    if not trainrows:
        errors.append('training bank is empty')
    if len(valrows) < min_validation:
        errors.append(f'validation count {len(valrows)} below required {min_validation}')
    return dict(format='pebby.generated-bank-audit.v1', status='complete' if not errors else 'failed_closed',
                errors=errors, required_validation_levels=min_validation, overlap=overlap,
                versions={split: sorted(values) for split, values in versions.items()}, splits=summaries,
                definitions={'corridor_fraction': 'Free cells with at most two cardinal free neighbors / all free cells.',
                             'rail_patterns': 'Joint rail cell sets translated by the playable geometry origin.',
                             'version_matching': 'Family, generator_version and quality version; pilot/training label strings may differ.'},
                limitations=['Structural checks inspect stored proof metadata; only requested spotchecks repeat search and engine replay.',
                             'Coverage and action autocorrelation are descriptive, not controller performance.',
                             'Geometry holdout concerns new v2 banks, not historical banks or whole topology families.',
                             'Accepted coverage cannot reveal excluded candidate frequencies without generation reports.'])


def spotcheck(rows, count):
    """Deterministically sample across input order and verify lengths, not tie paths."""
    from pebby.ls20.env import Ls20Scenario
    from pebby.ls20.generate import build_level
    from pebby.ls20.layout import extract
    from pebby.ls20.plan import Oracle, simulate
    selected = sorted({i*len(rows)//min(count, len(rows)) for i in range(min(count, len(rows)))}) if rows and count else []
    proofs = []
    for index in selected:
        row = rows[index]
        env = Ls20Scenario(build_level(row), row['training_context_index'])
        layout = extract(env)
        oracle = Oracle(layout, engine='fast', limit=min(row['search_limit'], 600000))
        if oracle.truncated or not oracle.solvable or len(oracle.solution()) != row['context_optimal_actions']:
            raise ValueError(f'seed {row["seed"]}: independent optimal-length search failed')
        state, minimum, result = oracle.start, oracle.start[6]//layout.step_cost, None
        for number, action in enumerate(row['solution']):
            before = state
            action_index = names.ACTION_IDS.index(action)
            state, outcome = simulate(layout, state, action_index, oracle.refills)
            minimum = min(minimum, route_budget_slack(layout, before, state, action=action_index, outcome=outcome))
            result = env.perform(action)
            if env.lives() != 3 or (result.finished and number != len(row['solution'])-1):
                raise ValueError(f'seed {row["seed"]}: stored route failed contextual replay')
            if not result.finished and oracle.state_of(env) != state:
                raise ValueError(f'seed {row["seed"]}: transition disagrees with engine')
        if (result is None or not result.won or env.levels_completed != 1
                or minimum != row['minimum_slack_moves'] or state[6]//layout.step_cost != row['slack_moves']):
            raise ValueError(f'seed {row["seed"]}: final win/headroom proof mismatch')
        proofs.append(dict(seed=row['seed'], context=row['training_context_index'],
                           optimal_actions=len(row['solution']), minimum_slack_moves=minimum, engine_win_three_lives=True))
    return proofs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, nargs='+', required=True)
    parser.add_argument('--validation', type=Path, nargs='+', required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--min-validation', type=int, default=500)
    parser.add_argument('--spotcheck', type=int, default=0, help='Levels per split, maximum20; CPU-only complete search.')
    args = parser.parse_args(argv)
    if args.min_validation < 0 or not 0 <= args.spotcheck <= 20:
        parser.error('nonnegative minimum validation count and spotcheck0..20 required')
    if args.report.exists():
        raise FileExistsError('refusing to overwrite audit report')
    paths = [*args.train, *args.validation]
    if args.report.resolve() in {path.resolve() for path in paths}:
        raise ValueError('report must not alias an input bank')
    contents = {str(path): path.read_bytes() for path in paths}
    inputs = {path: hashlib.sha256(data).hexdigest() for path, data in contents.items()}
    rows = {split: [json.loads(line) for path in selected for line in contents[str(path)].splitlines() if line.strip()]
            for split, selected in (('train', args.train), ('validation', args.validation))}
    result = audit(rows['train'], rows['validation'], args.min_validation)
    result['input_sha256'] = inputs
    result['audit_code_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in (Path(__file__), Path('pebby/ls20/generation_quality.py'))}
    result['spotchecks'] = {}
    if result['status'] == 'complete' and args.spotcheck:
        try:
            result['spotchecks'] = {split: spotcheck(values, args.spotcheck) for split, values in rows.items()}
        except (ValueError, RuntimeError) as error:
            result['status'] = 'failed_closed'
            result['errors'].append(f'spotcheck: {error}')
    result['input_hashes_unchanged'] = all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in inputs.items())
    if not result['input_hashes_unchanged']:
        result['status'] = 'failed_closed'
        result['errors'].append('input banks changed during audit')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], errors=len(result['errors']), report=str(args.report))))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
