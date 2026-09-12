"""Verify behavior provenance for raw or aggregated generated policy rollouts.

This checks source binding and marked-row alignment. The trainer separately
checks transition schemas, engine proofs, winning coverage and split separation.
Aggregates contain raw collector outputs only; provenance is never recursive.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _public_policy(meta):
    if not isinstance(meta, dict):
        raise ValueError('on-policy metadata must be a JSON object')
    provenance = meta.get('on_policy_provenance', {})
    if (meta.get('source') != 'generated_only' or
            meta.get('collection_policy') != 'model_greedy' or
            not isinstance(provenance, dict) or
            provenance.get('official_inputs_used') is not False or
            type(provenance.get('oracle_actions_in_policy_rollout')) is not int or
            provenance['oracle_actions_in_policy_rollout'] != 0):
        raise ValueError('on-policy data lacks generated-only public-policy provenance')


def _bound_file(record, kind):
    if not isinstance(record, dict) or not isinstance(record.get('path'), str) or not record['path']:
        raise ValueError(f'on-policy {kind} needs a file path')
    try:
        matches = file_digest(record['path']) == record.get('sha256')
    except OSError as error:
        raise ValueError(f'on-policy {kind} file unavailable') from error
    if not matches:
        raise ValueError(f'on-policy {kind} hash mismatch')


def _indices(meta, size):
    indices = meta.get('on_policy_rows')
    if (not isinstance(indices, list) or not indices or
            any(type(i) is not int or not 0 <= i < size for i in indices) or
            len(set(indices)) != len(indices)):
        raise ValueError('on-policy provenance has invalid marked row indices')
    return indices


def _auxiliary_indices(meta, size, marked):
    indices = meta.get('auxiliary_rows', [])
    if (not isinstance(indices, list) or any(type(i) is not int or not 0 <= i < size for i in indices)
            or len(set(indices)) != len(indices) or set(indices) & set(marked)):
        raise ValueError('on-policy provenance has invalid auxiliary row indices')
    return indices


def validate_on_policy_provenance(data):
    """Return verified behavior metadata suitable for the training checkpoint."""
    meta, seeds = data['meta'], np.asarray(data['seeds'])
    _public_policy(meta)
    if seeds.ndim != 1 or not np.issubdtype(seeds.dtype, np.integer) or np.any((seeds < 0) | (seeds >= 1_000_000)):
        raise ValueError('on-policy provenance requires generated training seeds')
    marked = _indices(meta, len(seeds))
    auxiliary = _auxiliary_indices(meta, len(seeds), marked)
    if 'behavior_checkpoint' in meta:
        if 'behavior_checkpoints' in meta or 'on_policy_sources' in meta:
            raise ValueError('on-policy provenance mixes raw and aggregate formats')
        _bound_file(meta['behavior_checkpoint'], 'behavior checkpoint')
        return {'behavior_checkpoint': meta['behavior_checkpoint']}

    behaviors, sources = meta.get('behavior_checkpoints'), meta.get('on_policy_sources')
    if not isinstance(behaviors, list) or not behaviors or not isinstance(sources, list) or not sources:
        raise ValueError('on-policy aggregate needs behavior checkpoints and source ranges')
    for behavior in behaviors:
        _bound_file(behavior, 'behavior checkpoint')
    identities = [(str(Path(b['path']).resolve()), b['sha256']) for b in behaviors]
    if len(set(identities)) != len(identities):
        raise ValueError('on-policy aggregate has duplicate behavior records')

    offset, all_marked, seen_seeds, seen_paths, used_behaviors = 0, [], set(), set(), set()
    all_auxiliary = []
    for source in sources:
        _bound_file(source, 'source')
        path = Path(source['path']).resolve()
        if path in seen_paths:
            raise ValueError('on-policy aggregate repeats a source file')
        seen_paths.add(path)
        start, stop, behavior_index = (source.get(k) for k in ('row_start', 'row_stop', 'behavior_index'))
        if (type(start) is not int or type(stop) is not int or start != offset or
                not start < stop <= len(seeds) or type(behavior_index) is not int or
                not 0 <= behavior_index < len(behaviors)):
            raise ValueError('on-policy aggregate has invalid source ranges or behavior index')
        with np.load(path, allow_pickle=False) as archive:
            if len(set(archive.files)) != len(archive.files):
                raise ValueError('on-policy source has duplicate archive members')
            source_meta = json.loads(str(archive['meta'].item()))
            source_seeds = archive['seeds']
        _public_policy(source_meta)
        if 'behavior_checkpoints' in source_meta or 'on_policy_sources' in source_meta:
            raise ValueError('nested on-policy aggregates are not supported')
        if source_meta.get('behavior_checkpoint') != behaviors[behavior_index]:
            raise ValueError('on-policy source behavior does not match its assigned checkpoint')
        if not np.array_equal(source_seeds, seeds[start:stop]):
            raise ValueError('on-policy source row seeds do not align with aggregate')
        seed_set = set(map(int, source_seeds))
        if seed_set & seen_seeds:
            raise ValueError('on-policy sources must have disjoint level seeds')
        seen_seeds.update(seed_set)
        expected = [start + i for i in _indices(source_meta, stop - start)]
        if source.get('on_policy_rows') != expected:
            raise ValueError('on-policy source marked rows do not align with aggregate')
        all_marked.extend(expected)
        expected_auxiliary = [start + i for i in _auxiliary_indices(source_meta, stop - start,
                                                                   _indices(source_meta, stop - start))]
        if source.get('auxiliary_rows', []) != expected_auxiliary:
            raise ValueError('on-policy source auxiliary rows do not align with aggregate')
        all_auxiliary.extend(expected_auxiliary)
        used_behaviors.add(behavior_index)
        offset = stop
        # Detect mutation while the small source arrays were being read.
        _bound_file(source, 'source')
    if offset != len(seeds) or all_marked != marked or all_auxiliary != auxiliary or used_behaviors != set(range(len(behaviors))):
        raise ValueError('on-policy aggregate source coverage is incomplete')
    return {'behavior_checkpoints': behaviors, 'on_policy_sources': sources}
