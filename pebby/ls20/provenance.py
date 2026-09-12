"""Versioned difficulty/context semantics, independent of generation and the engine.

Unversioned artifacts retain the historical five tiers and seed-modulo context.
Only explicitly calibrated artifacts use seven reference tiers. Never infer the
version from a tier number or a claimed verification context.
"""
from numbers import Integral

DIFFICULTY_VERSION = 'ls20-reference-v1'
LEGACY_DIFFICULTIES = (1, 2, 3, 4, 5)
CALIBRATED_DIFFICULTIES = (1, 2, 3, 4, 5, 6, 7)


def difficulty_version(record):
    version = record.get('difficulty_version')
    if version not in (None, DIFFICULTY_VERSION):
        raise ValueError(f'unsupported difficulty_version: {version!r}')
    return version


def difficulty_stages(record):
    return CALIBRATED_DIFFICULTIES if difficulty_version(record) else LEGACY_DIFFICULTIES


def validate_difficulty(record):
    value = record.get('difficulty')
    stages = difficulty_stages(record)
    if isinstance(value, bool) or not isinstance(value, Integral) or value not in stages:
        raise ValueError(f'difficulty must be an integer from 1 to {len(stages)}')
    return int(value)


def generated_context(record):
    """Expected game context from a level spec or its preserved row proof."""
    if difficulty_version(record):
        return validate_difficulty(record) - 1
    seed = record.get('seed')
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError('level seed must be an integer')
    return int(seed) % 7


def difficulty_provenance(record):
    """Fields needed when a full generated spec becomes a compact proof."""
    result = {'difficulty': record.get('difficulty')}
    if difficulty_version(record):
        validate_difficulty(record)
        result['difficulty_version'] = DIFFICULTY_VERSION
    return result


def search_limit_for(record, requested):
    """Honor calibrated complete-proof budgets, bounded to the approved 32M cap."""
    if not difficulty_version(record):
        return requested
    if (isinstance(requested, bool) or not isinstance(requested, Integral)
            or not 1 <= requested <= 32_000_000):
        raise ValueError('calibrated requested search_limit must be an integer in 1..32000000')
    declared = record.get('search_limit', requested)
    if (isinstance(declared, bool) or not isinstance(declared, Integral)
            or not 1 <= declared <= 32_000_000):
        raise ValueError('calibrated search_limit must be an integer in 1..32000000')
    return max(requested, int(declared))


def row_contexts(seeds, levels):
    """Expected contexts in row order, retaining each level's original version."""
    by_seed = {int(level['seed']): generated_context(level) for level in levels}
    return [by_seed[int(seed)] for seed in seeds]


def metadata_difficulty_version(metadata):
    """Resolve one profile from source metadata or a structured cache manifest."""
    source = metadata.get('source_metadata', metadata)
    levels = metadata.get('difficulty_levels', source.get('levels'))
    if not isinstance(levels, (list, tuple)):
        if difficulty_version(metadata):
            raise ValueError('calibrated cache needs per-level difficulty_levels provenance')
        return None
    versions = {difficulty_version(level) for level in levels if 'excluded' not in level}
    if len(versions) > 1:
        raise ValueError('mixed legacy and calibrated difficulty versions')
    version = next(iter(versions), None)
    declared = difficulty_version(metadata)
    if declared is not None and declared != version:
        raise ValueError('cache difficulty_version disagrees with per-level provenance')
    return version


def metadata_difficulty_stages(metadata):
    version = metadata_difficulty_version(metadata)
    source = metadata.get('source_metadata', metadata)
    levels = metadata.get('difficulty_levels', source.get('levels'))
    if isinstance(levels, (list, tuple)):
        for level in levels:
            if 'excluded' not in level and ('difficulty' in level or version):
                validate_difficulty(level)
    return difficulty_stages({'difficulty_version': version})


def validate_row_difficulties(seeds, difficulties, metadata):
    """Bind compact cache/paired-sampler labels to original per-level tiers."""
    version = metadata_difficulty_version(metadata)
    stages = difficulty_stages({'difficulty_version': version})
    if len(seeds) != len(difficulties):
        raise ValueError('row seed and difficulty lengths differ')
    source = metadata.get('source_metadata', metadata)
    levels = metadata.get('difficulty_levels', source.get('levels'))
    lookup = {}
    if isinstance(levels, (list, tuple)):
        for level in levels:
            if 'excluded' in level:
                continue
            seed, difficulty = int(level['seed']), validate_difficulty(level)
            if seed in lookup and lookup[seed] != difficulty:
                raise ValueError('inconsistent per-level difficulty provenance')
            lookup[seed] = difficulty
    for seed, difficulty in zip(seeds, difficulties):
        if (isinstance(difficulty, bool) or not isinstance(difficulty, Integral)
                or difficulty not in stages):
            raise ValueError(f'difficulty must be an integer in 1..{len(stages)} for this version')
        if isinstance(levels, (list, tuple)) and lookup.get(int(seed)) != difficulty:
            raise ValueError('row difficulty differs from per-level provenance')
    return stages


def cache_difficulty_metadata(metadata, seeds):
    """Carry selected levels' tiers across a cache boundary without relabelling."""
    version = metadata_difficulty_version(metadata)
    source = metadata.get('source_metadata', metadata)
    levels = metadata.get('difficulty_levels', source.get('levels'))
    wanted = set(map(int, seeds))
    selected = [{'seed': int(level['seed']), **difficulty_provenance(level)}
                for level in levels if int(level['seed']) in wanted and 'excluded' not in level]
    if set(level['seed'] for level in selected) != wanted:
        raise ValueError('selected cache levels lack difficulty provenance')
    return {'difficulty_version': version, 'difficulty_levels': selected}


def require_legacy_experiment(metadata):
    """Frozen paired experiments cannot be repurposed by swapping one input."""
    if metadata_difficulty_version(metadata):
        raise ValueError('this frozen paired experiment requires legacy five-tier caches; '
                         'use tools/train_structured_sequences.py --train-cache PATH '
                         '--validation-cache PATH for calibrated seven-tier caches, '
                         'or pebby.agent.world_train for generated transition archives')
