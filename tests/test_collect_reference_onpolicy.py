from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from tools import collect_reference_onpolicy as c


def specimens():
    return [dict(seed=100 + tier * 10 + i, difficulty=tier, split='train',
                 difficulty_version=c.DIFFICULTY_VERSION, launchers=[1])
            for tier in range(1, 8) for i in range(3)]


def test_selection_exact_deterministic_all_eligible():
    specs = specimens()
    with patch.object(c, 'profile_errors', return_value=[]):
        selected = c.select_levels(specs, count=21, quotas=[3] * 7)
        assert {s['seed'] for s in selected} == {s['seed'] for s in specs}
        assert selected == c.select_levels(list(reversed(specs)), count=21, quotas=[3] * 7)
        assert c.select_levels(specs, count=1, quotas=[1, 0, 0, 0, 0, 0, 0])[0]['difficulty'] == 1
        with pytest.raises(ValueError, match='duplicate'):
            c.select_levels(specs + specs[:1], count=21, quotas=[3] * 7)
        specs[0]['split'] = 'validation'
        with pytest.raises(ValueError, match='TRAIN'):
            c.select_levels(specs, count=21, quotas=[3] * 7)


@pytest.mark.parametrize('count,quotas', [(1000, [1] * 7), (0, [0] * 7), (1, [-1, 2, 0, 0, 0, 0, 0])])
def test_bad_quotas(count, quotas):
    with pytest.raises(ValueError, match='quotas'):
        c.select_levels([], count=count, quotas=quotas)


def test_source_hash_fail_closed(tmp_path):
    source = tmp_path / 'source'; source.write_text('old bank')
    with pytest.raises(ValueError, match='exact fresh parent'):
        c.validate_sources(source, source, source)


@pytest.mark.parametrize('engine,truncated,limit', [('reference', False, 600000), ('fast', True, 600000), ('fast', False, 1)])
def test_native_guard_rejects_fallback_or_incomplete(engine, truncated, limit):
    def context(spec, **kwargs):
        assert kwargs['search_limit'] == c.SEARCH_LIMITS[0]
        return object(), SimpleNamespace(engine=engine, truncated=truncated), dict(oracle_backend=engine, search_limit=limit)
    def collect(spec, policy, **kwargs):
        return c.wd.verified_context(spec)
    with patch.object(c.wd, 'verified_context', context), patch.object(c, 'collect_level', collect):
        with pytest.raises(ValueError, match='complete native'):
            c.collect_native(dict(difficulty=1), object(), 8)


def test_native_forced_before_context_search():
    seen = []
    def oracle(*args, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(engine='fast', truncated=False)
    def context(spec, **kwargs):
        result = c.wd.Oracle(object(), limit=kwargs['search_limit'])
        return object(), result, dict(oracle_backend='fast', search_limit=kwargs['search_limit'])
    def collect(spec, policy, **kwargs):
        return c.wd.verified_context(spec)
    with patch.object(c, 'Oracle', oracle), patch.object(c.wd, 'verified_context', context), patch.object(c, 'collect_level', collect):
        c.collect_native(dict(difficulty=7), object(), 8)
    assert seen == [dict(limit=32000000, engine='fast')]


def test_output_requires_exact_seeds_partition_and_version():
    data = dict(seeds=np.array([12, 12]), meta=dict(difficulty_version=c.DIFFICULTY_VERSION,
                on_policy_rows=[0], auxiliary_rows=[1]))
    with patch.object(c, 'require_verified_data'), patch.object(c, 'require_winning_coverage'), patch.object(c, 'validate_successor_labels'), patch.object(c, 'validate_on_policy_provenance'):
        c.verify_output(data, [dict(seed=12)], [1000000])
        with pytest.raises(ValueError, match='disjointness'):
            c.verify_output(data, [dict(seed=12)], [12])
        data['meta']['auxiliary_rows'] = []
        with pytest.raises(ValueError, match='partition'):
            c.verify_output(data, [dict(seed=12)], [])
        data['meta']['difficulty_version'] = 'legacy'
        with pytest.raises(ValueError, match='difficulty_version'):
            c.verify_output(data, [dict(seed=12)], [])


def test_published_directory_never_overwritten(tmp_path):
    with pytest.raises(FileExistsError, match='overwrite'):
        c.main(['--out-dir', str(tmp_path)])


def test_admission_reserves_only_unallocated_worker_headroom():
    active = [(SimpleNamespace(pid=1), None, None, None), (SimpleNamespace(pid=2), None, None, None)]
    with patch.object(c, 'resident_bytes', side_effect=[2 * c.GIB, 4 * c.GIB]):
        assert c.admission_bytes(active) == 10 * c.GIB
    assert c.admission_bytes([]) == 9 * c.GIB
