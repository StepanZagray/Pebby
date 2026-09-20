"""Reject supplemental lineage and cache substitutions before GPU training."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import train_reference_onpolicy as runner


@pytest.fixture
def publication(tmp_path, monkeypatch):
    selected = tmp_path / 'selected'
    selected.mkdir()
    specs = []
    for tier, count in enumerate(runner.QUOTAS, 1):
        specs += [dict(seed=100000 + len(specs) + i, difficulty=tier) for i in range(count)]
    body = ''.join(json.dumps(s) + '\n' for s in specs)
    bank = tmp_path / 'source.jsonl'
    bank.write_text(body)
    (selected / 'train.jsonl').write_text(body)
    monkeypatch.setattr(runner, 'BANK', bank)
    monkeypatch.setattr(runner, 'BANK_SHA', runner.repair.digest(bank))
    meta = dict(source='generated_only', collection_policy='model_greedy', history=8,
                alternatives_per_state=4, on_policy_rows=list(range(1000)),
                on_policy_provenance=dict(official_inputs_used=False, oracle_actions_in_policy_rollout=0),
                behavior_checkpoint=dict(path=str(runner.repair.PARENT), sha256=runner.repair.PARENT_SHA))
    np.savez(selected / 'train.npz', seeds=[s['seed'] for s in specs], distances=np.zeros((1000, 4)),
             meta=json.dumps(meta))
    report = dict(status='complete', output_sha256=runner.repair.digest(selected / 'train.npz'),
                  selected_bank=dict(path=str(selected / 'train.jsonl'), count=1000,
                                     sha256=runner.repair.digest(selected / 'train.jsonl')),
                  selected_seeds=[s['seed'] for s in specs], count=1000,
                  max_policy_actions_per_level=150, verified_reload=True, sources_unchanged=True,
                  workers_exited=True, validation_disjoint=True,
                  sourcebindings={str(bank): runner.BANK_SHA, str(runner.repair.PARENT): runner.repair.PARENT_SHA})
    (selected / 'report.json').write_text(json.dumps(report))
    # The real parent is separately hash/lineage checked by the delegated runner;
    # keep this synthetic publication test independent of local checkpoints.
    original = runner.repair.digest
    monkeypatch.setattr(runner.repair, 'digest', lambda p: runner.repair.PARENT_SHA
                        if Path(p) == runner.repair.PARENT else original(p))
    return selected, report


def test_accepts_matching_publication(publication):
    directory, report = publication
    assert runner.validate_supplement(directory) == (directory / 'train.npz', report['output_sha256'])


@pytest.mark.parametrize('field', ['verified_reload', 'sources_unchanged', 'workers_exited', 'validation_disjoint'])
def test_rejects_incomplete_receipt(publication, field):
    directory, report = publication
    report[field] = False
    (directory / 'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='completion receipt'):
        runner.validate_supplement(directory)


def test_rejects_changed_selected_bank_receipt(publication):
    directory, report = publication
    report['selected_bank']['sha256'] = '0' * 64
    (directory / 'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='selected-bank'):
        runner.validate_supplement(directory)


def test_rejects_other_behavior_checkpoint_even_with_valid_archive_hash(publication):
    directory, report = publication
    with np.load(directory / 'train.npz') as archive:
        data = dict(archive)
    meta = json.loads(str(data['meta'].item()))
    meta['behavior_checkpoint']['sha256'] = '0' * 64
    data['meta'] = json.dumps(meta)
    np.savez(directory / 'train.npz', **data)
    report['output_sha256'] = runner.repair.digest(directory / 'train.npz')
    (directory / 'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='fresh-parent provenance'):
        runner.validate_supplement(directory)


def test_rejects_wrong_schema_cache(tmp_path, monkeypatch):
    source = tmp_path / 'train.npz'
    source.write_bytes(b'archive')
    sha = runner.repair.digest(source)
    cache = tmp_path / 'array-cache' / (sha + '-wrong-schema')
    cache.mkdir(parents=True)
    (cache / 'manifest.json').write_text(json.dumps(dict(source_sha256=sha, arrays={})))
    monkeypatch.setattr(runner.repair, 'DATA', tmp_path)
    with pytest.raises(ValueError, match='exact trainer-schema'):
        runner.extended_guard(tmp_path, source, sha, lambda: SimpleNamespace(hashes={}, stats={}))
