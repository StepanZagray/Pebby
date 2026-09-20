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
