"""Opt-in generator v4: goal relations, vanishing rings, three-way holdout.

Aggregate difficulty/context contracts remain reference-v1. Version 3's RNG,
drafts and seed splitter are deliberately unchanged in reference_generator.py.
No official layouts or routes are used to construct these procedural drafts.
"""
from collections import Counter
from numbers import Integral
import random

from .generation_quality import geometry_d4_hash
from .reference_generator import draft as legacy_draft, verify as verify_reference
from .reference_profiles import DIFFICULTIES, KINDS, PROFILES, profile_errors, structural_metrics

GENERATOR_VERSION = 4
MECHANICS_VERSION = 'ls20-reference-relations-v2'
GEOMETRY_VERSION = 'dihedral-three-way-v2'
SPLITS = ('train', 'validation', 'test')


def seed_split(seed):
    """Declared bank ranges, with historical high validation seeds preserved."""
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if 1_000_000 <= seed < 2_000_000 or seed >= 8_000_000:
        return 'validation'
    if 2_000_000 <= seed < 3_000_000:
        return 'test'
    return 'train'


def geometry_partition(spec):
    """Disjoint geometry buckets for this version, invariant under all D4 maps."""
    fingerprint = geometry_d4_hash(spec)
    return fingerprint, SPLITS[int(fingerprint, 16) % len(SPLITS)]


def draft(rng, difficulty):
    """Keep aggregate tier geometry; sample relations in later goal targets.

    The first goal changes every required attribute. Later goals may restore
    any proper subset to its initial value, including the empty subset. They
    remain distinct and never match the entire initial glyph. This does not
    prescribe goal visitation order. Multi-goal rings independently may vanish.
    """
    spec = legacy_draft(rng, difficulty)
    if spec is None:
        return None
    kinds = PROFILES[difficulty]['changed_kinds']
    for goal in spec['goals'][1:]:
        mask = rng.randrange((1 << len(kinds)) - 1)
        for bit, kind in enumerate(kinds):
            if mask & (1 << bit):
                index = KINDS.index(kind)
                goal['triple'][index] = spec['start_triple'][index]
    if len(spec['goals']) > 1:
        for goal in spec['goals']:
            if rng.randrange(2):
                goal['vanishing_ring'] = True
    spec.update(generator_version=GENERATOR_VERSION, mechanics_version=MECHANICS_VERSION)
    spec.update(structural_metrics(spec))
    spec['changed_kinds'] = list(spec['changed_kinds'])
    return None if profile_errors(spec, require_proof=False) else spec


def generate_level(seed, difficulty=1, attempts=400, min_slack=None, search_limit=None,
                   *, split=None, record_rejection=None):
    """Generate and engine-verify a v4 row; no fallback to legacy splitting."""
    inferred = seed_split(seed)
    if type(difficulty) is not int or difficulty not in DIFFICULTIES:
        raise ValueError('difficulty must be 1..7')
    if (type(attempts) is not int or attempts < 1
            or (search_limit is not None and (type(search_limit) is not int or not 0 < search_limit <= 32_000_000))
            or (min_slack is not None and (type(min_slack) is not int or min_slack < 0))):
        raise ValueError('positive attempts/search limit <=32000000 and nonnegative slack required')
    split = inferred if split is None else split
    if split not in SPLITS:
        raise ValueError('split must be train, validation or test')
    if split != inferred:
        raise ValueError('explicit split differs from the versioned seed range')
    rng = random.Random(f'{MECHANICS_VERSION}:{int(seed)}:{difficulty}')
    exclusions = Counter()
    for attempt in range(1, attempts + 1):
        spec = None
        for _ in range(64):
            spec = draft(rng, difficulty)
            if spec is None:
                exclusions['invalid_geometry'] += 1
                continue
            if geometry_partition(spec)[1] != split:
                exclusions['geometry_split'] += 1
                spec = None
                continue
            break
        if spec is None:
            accepted, reason = None, 'geometry_attempts_exhausted'
        else:
            spec.update(seed=int(seed), generation_attempt=attempt, split=split, source='generated_only')
            accepted, reason = verify_reference(spec, search_limit, min_slack)
        if accepted:
            fingerprint, partition = geometry_partition(accepted)
            accepted.update(geometry_sha256=fingerprint, geometry_d4_sha256=fingerprint,
                            geometry_split=partition, geometry_version=GEOMETRY_VERSION,
                            generation_exclusions=dict(exclusions))
            accepted['proof'].update(generator_version=GENERATOR_VERSION, mechanics_version=MECHANICS_VERSION,
                                     split=split, geometry_version=GEOMETRY_VERSION, geometry_split=partition)
            return accepted
        exclusions[reason] += 1
        if record_rejection:
            record_rejection(dict(seed=int(seed), difficulty=difficulty, attempt=attempt, reason=reason,
                                  generator_version=GENERATOR_VERSION, mechanics_version=MECHANICS_VERSION))
    raise RuntimeError(f'no v4 reference level seed={seed} difficulty={difficulty} after {attempts} attempts: {dict(exclusions)}')
