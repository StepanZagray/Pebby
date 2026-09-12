"""Read-only integrity and coverage audit of regenerated v2 training banks.

Structural checks inspect stored proofs. Optional bounded spotchecks repeat the
generator's planner search and replay routes in the real engine. Planner agreement
is not an independent optimality proof.
No controller is trained or evaluated by this tool.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import random
from pathlib import Path
import statistics

from pebby.ls20 import names
from pebby.ls20.extended_curriculum import gameplay_hash
from pebby.ls20.generate import FORMAT, GENERATOR_VERSION, RESERVED
from pebby.ls20.generation_quality import (budget_floor, geometry_partition, geometry_d4_hash,
                                         geometry_d4_partition, route_budget_slack)


BOARD = {(x, y) for x in range(12) for y in range(12)}
KINDS = ('shape', 'color', 'rotation')
REPO_ROOT = Path(__file__).resolve().parents[1]


_d4_hash = geometry_d4_hash

def _proof_errors(row):
    proof = row.get('proof')
    if proof is None:
        return ['missing nested contextual proof'] if row.get('difficulty_version') else []
    if not isinstance(proof, dict):
        return ['nested contextual proof must be an object']
    fields = {'seed': 'seed', 'context_index': 'training_context_index',
              'context_engine_verified': 'context_engine_verified',
              'search_truncated': 'search_truncated', 'context_optimal_actions': 'context_optimal_actions',
              'oracle_backend': 'oracle_backend'}
    fields.update({key: key for key in proof.keys() & row.keys() if key != 'proof' and key not in fields})
    return [f'nested proof {key} disagrees with top-level {top}' for key, top in fields.items()
            if key not in proof or type(proof[key]) is not type(row.get(top)) or proof[key] != row.get(top)]


def _coverage_errors(rows, summary, policy):
    """Full banks must cover the contract; samples retain measurable diversity gates."""
    errors = []
    if not rows:
        return errors
    if policy == 'full':
        reference = any(row.get('difficulty_version') for row in rows)
        required = set(range(1, 8 if reference else 6))
        axes = [('difficulty', required, {r['difficulty'] for r in rows}),
                ('context', set(range(7)), {r['training_context_index'] for r in rows}),
                ('changing attributes', {1, 2, 3}, {r['changing_attributes'] for r in rows})]
        if reference:
            axes.append(('quality profile', {'learning', 'challenge'}, {r['quality_profile'] for r in rows}))
        elif any('pilot_mode' in r for r in rows):
            from tools.generate_mechanism_pilot import MODES
            axes += [('mode', set(MODES), {r.get('pilot_mode') for r in rows}),
                     ('quality profile', {'learning', 'challenge'}, {r['quality_profile'] for r in rows})]
        for axis, required, actual in axes:
            if missing := required - actual:
                errors.append(f'missing {axis} coverage: {sorted(missing)}')
    # Even an explicitly small sample cannot certify a repeatedly copied room.
    required_geometry = min(8, len(rows)) if policy == 'full' else min(2, len(rows))
    if len({_d4_hash(r) for r in rows}) < required_geometry:
        errors.append(f'D4 geometry diversity below required {required_geometry}')
    if len(rows) >= 7 and len({r['changing_attributes'] for r in rows}) < 2:
        errors.append('changing-attribute diversity collapsed to one value')
    for mode, values in summary['per_mode'].items():
        count = values['levels']
        if count >= 4:
            selected = [r for r in rows if r.get('pilot_mode', 'extended') == mode]
            if len({_d4_hash(r) for r in selected}) < 2:
                errors.append(f'{mode}: D4 geometry diversity below required 2')
            if mode in ('long_route', 'refill_chain'):
                for key in ('distinct_normalized_starts', 'distinct_normalized_goal_placements',
                            'distinct_normalized_refill_placements'):
                    if values[key] < 2:
                        errors.append(f'{mode}: {key} diversity below required 2')
    return errors


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


def audit(trainrows, valrows, min_validation=500, *, coverage_policy='full'):
    """Audit in-memory rows without mutating them or invoking a search."""
    if min_validation < 0:
        raise ValueError('min_validation must be nonnegative')
    if coverage_policy not in ('full', 'sample'):
        raise ValueError('coverage_policy must be full or sample')
    errors, summaries, identities, versions = [], {}, {}, {}
    for split, rows in (('train', trainrows), ('validation', valrows)):
        identities[split] = {key: set() for key in ('seeds', 'gameplay', 'geometry', 'geometry_d4')}
        versions[split] = set()
        valid = []
        for number, row in enumerate(rows):
            if not isinstance(row, dict):
                errors.append(f'{split} row {number}: expected a level object')
                continue
            label = f'{split} row {number} seed {row.get("seed", "missing")}'
            try:
                reference = row.get('difficulty_version') == 'ls20-reference-v1'
                geometry, partition = (geometry_d4_partition if reference else geometry_partition)(row)
                fingerprint = gameplay_hash(row)
                for key, value in (('seeds', row['seed']), ('gameplay', fingerprint),
                                   ('geometry', geometry), ('geometry_d4', _d4_hash(row))):
                    if key in ('seeds', 'gameplay') and value in identities[split][key]:
                        errors.append(f'{label}: duplicate {key} within split')
                    identities[split][key].add(value)
                version = _version(row)
                versions[split].add(version)
                expected_quality = 3 if reference else 2
                if version[1:] != (GENERATOR_VERSION, expected_quality):
                    raise ValueError(f'requires sprite generator v{GENERATOR_VERSION} and curriculum quality v{expected_quality}')
                if row.get('format') != FORMAT or row.get('size') != 64:
                    raise ValueError('unexpected level format/size')
                if row.get('split', split) != split:
                    raise ValueError('declared split mismatch')
                if (partition != split or row.get('geometry_split') != partition
                        or row.get('geometry_sha256') != geometry):
                    raise ValueError('normalized geometry partition proof mismatch')
                if reference and (row.get('geometry_version') != 'dihedral-v1'
                                  or row.get('geometry_d4_sha256') != geometry):
                    raise ValueError('missing or inconsistent D4 geometry proof')
                if 'gameplay_sha256' in row and row['gameplay_sha256'] != fingerprint:
                    raise ValueError('gameplay hash mismatch')
                max_search = 600000
                if reference:
                    from pebby.ls20.reference_profiles import PROFILES
                    max_search = PROFILES.get(row.get('difficulty'), {}).get('search_limit', 0)
                if (row.get('search_truncated') is not False or row.get('engine_verified') is not True
                        or row.get('context_engine_verified') is not True or row.get('verification_lives') != 3
                        or row.get('oracle_backend') not in ('fast', 'reference')
                        or not 0 < row['reachable_states'] <= row['search_limit'] <= max_search):
                    raise ValueError('incomplete contextual search/engine proof')
                if row.get('difficulty_version') == 'ls20-reference-v1':
                    from pebby.ls20.reference_profiles import profile_errors
                    if problems := profile_errors(row):
                        raise ValueError('; '.join(problems))
                    context = row['difficulty'] - 1
                    expected_profile = 'learning' if row['difficulty'] == 1 else 'challenge'
                    if row.get('quality_profile') != expected_profile:
                        raise ValueError('quality profile disagrees with reference tier')
                elif row.get('difficulty_version') is not None:
                    raise ValueError('unsupported difficulty version')
                else:
                    if type(row.get('difficulty')) is not int or row['difficulty'] not in range(1, 6):
                        raise ValueError('legacy difficulty must be an integer in 1..5')
                    context = row['seed'] % 7
                if (type(row['training_context_index']) is not int or type(row['verification_level_index']) is not int
                        or row['training_context_index'] != context or row['verification_level_index'] != context):
                    raise ValueError('stored context disagrees with declared difficulty/seed context')
                if 'context_index' in row and (type(row['context_index']) is not int or row['context_index'] != context):
                    raise ValueError('top-level context_index disagrees with training context')
                if problems := _proof_errors(row):
                    raise ValueError('; '.join(problems))
                if context == 0 and row.get('launchers'):
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
                if 'won' in row['solution_mechanics'] and row['solution_mechanics']['won'] is not True:
                    raise ValueError('solution mechanics does not record a win')
                if row.get('free_cells') != len(free) or row.get('changing_attributes') != features['attributes']:
                    raise ValueError('free-cell/attribute count mismatch')
                for kind in ('moving', 'launchers'):
                    used = features[f'{kind}_used']
                    if not isinstance(used, int) or not 0 <= used <= features[f'{kind}_installed']:
                        raise ValueError(f'missing or invalid distinct {kind} contact count')
                if row.get('pilot_mode') == 'three_rails' and (features['moving_kinds'] != 3 or features['moving_used'] != 3):
                    raise ValueError('three-rail mode lacks three distinct used moving kinds')
                if not reference and row.get('distractor_count', 0) < 1:
                    raise ValueError('missing unused distractor proof')
                valid.append(row)
            except (KeyError, ValueError, TypeError, IndexError, ZeroDivisionError) as error:
                errors.append(f'{label}: {error}')
        summaries[split] = _summary(valid)
        errors.extend(f'{split}: {error}' for error in _coverage_errors(valid, summaries[split], coverage_policy))
        summaries[split]['input_levels'] = len(rows)
        summaries[split]['rows_without_nested_proof'] = sum('proof' not in row for row in valid)
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
                coverage_policy=coverage_policy,
                versions={split: sorted(values) for split, values in versions.items()}, splits=summaries,
                definitions={'corridor_fraction': 'Free cells with at most two cardinal free neighbors / all free cells.',
                             'rail_patterns': 'Joint rail cell sets translated by the playable geometry origin.',
                             'version_matching': 'Family, generator_version and quality version; pilot/training label strings may differ.'},
                limitations=['Structural checks inspect stored proof metadata; only requested spotchecks repeat search and engine replay.',
                             'Coverage gates check bank diversity, not controller performance; action autocorrelation is descriptive.',
                             'Sample policy does not certify complete tier/mode/context coverage.',
                             'Geometry holdout includes translation, rotations and reflections, not whole topology families.',
                             'Accepted coverage cannot reveal excluded candidate frequencies without generation reports.'])


def _spotcheck_indices(rows, count, *, seed=0):
    """Cover tier/mode/profile strata before repeating, with seeded shuffled ties."""
    if count < 0:
        raise ValueError('spotcheck count must be nonnegative')
    candidates = list(range(len(rows)))
    random.Random(seed).shuffle(candidates)
    covered, selected = set(), []
    def strata(index):
        row = rows[index]
        return {(key, row.get(key)) for key in ('difficulty', 'pilot_mode', 'quality_profile')}
    while candidates and len(selected) < count:
        index = max(candidates, key=lambda i: len(strata(i) - covered))
        selected.append(index)
        covered.update(strata(index))
        candidates.remove(index)
    return selected


def _spotcheck_coverage(rows, count):
    selected = [rows[i] for i in _spotcheck_indices(rows, count)]
    axes = {}
    for key in ('difficulty', 'pilot_mode', 'quality_profile'):
        available = {row.get(key, 'extended' if key == 'pilot_mode' else None) for row in rows}
        covered = {row.get(key, 'extended' if key == 'pilot_mode' else None) for row in selected}
        axes[key] = dict(available=sorted(available, key=str), covered=sorted(covered, key=str),
                         uncovered=sorted(available-covered, key=str))
    return dict(requested=count, selected=len(selected), axes=axes)


def spotcheck(rows, count):
    """Stratified same-planner re-search plus independent real-engine winning replay."""
    from pebby.ls20.env import Ls20Scenario
    from pebby.ls20.generate import build_level
    from pebby.ls20.layout import extract
    from pebby.ls20.plan import Oracle, simulate
    selected = _spotcheck_indices(rows, count)
    proofs = []
    for index in selected:
        row = rows[index]
        env = Ls20Scenario(build_level(row), row['training_context_index'])
        layout = extract(env)
        oracle = Oracle(layout, engine='fast', limit=row['search_limit'])
        if oracle.truncated or not oracle.solvable or len(oracle.solution()) != row['context_optimal_actions']:
            raise ValueError(f'seed {row["seed"]}: same-planner optimal-length re-search failed')
        if oracle._reachable != row['reachable_states']:
            raise ValueError(f'seed {row["seed"]}: reachable-state re-search mismatch')
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
                           difficulty=row['difficulty'], mode=row.get('pilot_mode', 'extended'),
                           quality_profile=row['quality_profile'],
                           optimal_actions=len(row['solution']), minimum_slack_moves=minimum,
                           reachable_states=oracle._reachable, same_planner_search_agrees=True,
                           independent_optimality_verified=False, engine_win_three_lives=True))
    return proofs


def _generation_evidence(report, split_inputs):
    """Bind rejection metadata to the exact bank bytes before using it as evidence."""
    if report.get('status') != 'complete':
        raise ValueError('generation report is not complete')
    for split, inputs in split_inputs.items():
        banks = report.get('banks', {}).get(split)
        banks = banks if isinstance(banks, list) else [banks]
        if any(not isinstance(bank, dict) for bank in banks):
            raise ValueError(f'generation report missing {split} bank binding')
        expected = sorted((bank.get('sha256'), bank.get('levels')) for bank in banks)
        if expected != sorted(inputs):
            raise ValueError(f'generation report {split} bank hash/count mismatch')
    counts = report.get('rejection_counts')
    if not isinstance(counts, dict) or any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError('invalid generation rejection counts')
    if 'rejections' in report:
        observed = Counter(item['reason'] for item in report['rejections'])
        if dict(observed) != {key: n for key, n in counts.items() if n}:
            raise ValueError('generation rejection detail/count mismatch')
    def capped(reason):
        return any(word in reason for word in ('search limit', 'search_limit', 'search_truncated'))
    cap = sum(n for reason, n in counts.items() if capped(reason))
    unsolved = sum(n for reason, n in counts.items()
                   if ('unsolved' in reason or 'unsolvable' in reason) and not capped(reason))
    total = sum(counts.values())
    return dict(binding_verified=True, rejected_candidates=total, rejection_counts=counts,
                search_cap_rejections=cap, unsolved_rejections=unsolved,
                search_cap_fraction_of_rejections=cap/total if total else 0,
                unsolved_fraction_of_rejections=unsolved/total if total else 0,
                cap_or_unsolved_fraction_of_rejections=(cap+unsolved)/total if total else 0,
                limitation='Rejected candidates are excluded from accepted-bank statistics; cap rejection is not unsolvability.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, nargs='+', required=True)
    parser.add_argument('--validation', type=Path, nargs='+', required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--min-validation', type=int, default=500)
    parser.add_argument('--coverage-policy', choices=('full', 'sample'), default='full',
                        help='Sample skips complete coverage quotas; use only for explicit small fixtures.')
    parser.add_argument('--generation-report', type=Path,
                        help='Optional generation rejection evidence, bound to exact input bank hashes/counts.')
    parser.add_argument('--spotcheck', type=int, default=7,
                        help='Levels per split, maximum20; stratified same-planner re-search and real-engine replay.')
    args = parser.parse_args(argv)
    if args.min_validation < 0 or not 0 <= args.spotcheck <= 20:
        parser.error('nonnegative minimum validation count and spotcheck0..20 required')
    if args.report.exists():
        raise FileExistsError('refusing to overwrite audit report')
    bank_paths = [*args.train, *args.validation]
    paths = [*bank_paths, *([args.generation_report] if args.generation_report else [])]
    if args.report.resolve() in {path.resolve() for path in paths}:
        raise ValueError('report must not alias an input bank')
    contents = {str(path): path.read_bytes() for path in paths}
    inputs = {path: hashlib.sha256(data).hexdigest() for path, data in contents.items()}
    rows = {split: [json.loads(line) for path in selected for line in contents[str(path)].splitlines() if line.strip()]
            for split, selected in (('train', args.train), ('validation', args.validation))}
    result = audit(rows['train'], rows['validation'], args.min_validation, coverage_policy=args.coverage_policy)
    result['input_sha256'] = inputs
    result['audit_code_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in (Path(__file__).resolve(), REPO_ROOT/'pebby/ls20/generation_quality.py',
                                                 REPO_ROOT/'pebby/ls20/reference_profiles.py')}
    result['generation_evidence'] = None
    if args.generation_report:
        try:
            split_inputs = {split: [(inputs[str(path)], sum(bool(line.strip()) for line in contents[str(path)].splitlines()))
                                    for path in selected]
                            for split, selected in (('train', args.train), ('validation', args.validation))}
            result['generation_evidence'] = _generation_evidence(
                json.loads(contents[str(args.generation_report)]), split_inputs)
        except (ValueError, KeyError, TypeError) as error:
            result['status'] = 'failed_closed'
            result['errors'].append(f'generation report: {error}')
    result['spotchecks'] = {}
    result['spotcheck_selection'] = {split: _spotcheck_coverage(values, args.spotcheck)
                                     for split, values in rows.items()} if result['status'] == 'complete' else {}
    result['spotcheck_limitations'] = [
        'Zero requested spotchecks means no engine replay or planner re-search in this audit.',
        'A budget smaller than the number of strata cannot cover every mode/tier/profile.',
        'Planner re-search shares the generator implementation and does not independently prove optimality.']
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
