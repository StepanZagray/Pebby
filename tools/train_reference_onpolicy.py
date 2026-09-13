"""One fixed-budget continuation with fresh-reference model-visited data.

Reuses the matched repair runner's initialization, optimizer-step, source,
batch-identity and memory guards. Only the 25% supplemental data mixture changes;
the supplemental expert/failure auxiliaries are excluded from this experiment.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from tools import train_reference_repair as repair

QUOTAS = (300, 200, 150, 120, 100, 80, 50)
BANK = repair.ROOT / 'data/ls20-reference-unequal-v1/train.jsonl'
BANK_SHA = 'f968f9b4a3690be69041eadd1afe129a3ba0527d55829d75687aa5c260532a81'


def validate_supplement(directory):
    """Check published lineage and small columns before allocating a GPU model."""
    import numpy as np

    directory = Path(directory).resolve()
    report_path, source = directory / 'report.json', directory / 'train.npz'
    report = json.loads(report_path.read_text())
    if report.get('status') != 'complete':
        raise ValueError('completed on-policy publication required')
    actual_sha = repair.digest(source)
    if report.get('output_sha256') != actual_sha:
        raise ValueError('supplemental archive differs from publication receipt')
    if repair.digest(BANK) != BANK_SHA:
        raise ValueError('reference TRAIN bank differs from bound fresh bank')
    published = [json.loads(line) for line in (directory / 'train.jsonl').read_text().splitlines()]
    all_specs = {spec['seed']: spec for spec in map(json.loads, BANK.read_text().splitlines())}
    expected = {spec['seed'] for spec in published}
    if (len(published) != 1000 or len(expected) != 1000 or
            any(all_specs.get(spec['seed']) != spec for spec in published)):
        raise ValueError('supplement must contain 1000 exact current TRAIN records')
    if Counter(spec['difficulty'] for spec in published) != Counter(dict(enumerate(QUOTAS, 1))):
        raise ValueError('supplemental difficulty quotas differ from the experiment')
    bank_receipt = report.get('selected_bank', {})
    if (bank_receipt.get('sha256') != repair.digest(directory / 'train.jsonl') or
            bank_receipt.get('count') != 1000 or
            Path(bank_receipt.get('path', '')).resolve() != directory / 'train.jsonl' or
            report.get('count') != 1000 or report.get('selected_seeds') != [s['seed'] for s in published] or
            report.get('max_policy_actions_per_level') != 150 or
            any(report.get(key) is not True for key in
                ('verified_reload', 'sources_unchanged', 'workers_exited', 'validation_disjoint'))):
        raise ValueError('supplemental selected-bank/completion receipt differs')
    sourcebindings = report.get('sourcebindings', {})
    if sourcebindings.get(str(BANK)) != BANK_SHA or sourcebindings.get(str(repair.PARENT)) != repair.PARENT_SHA:
        raise ValueError('publication does not bind the fresh bank and parent')
    if any(repair.digest(path) != expected_sha for path, expected_sha in sourcebindings.items()):
        raise ValueError('collection source binding changed before adoption')
    with np.load(source, allow_pickle=False) as archive:
        meta = json.loads(str(archive['meta'].item()))
        seeds = archive['seeds']
        if set(map(int, seeds)) != expected:
            raise ValueError('supplemental row seeds differ from selected bank')
        distances = archive['distances']
        if np.any(distances >= 128):
            raise ValueError('supplement exceeds the existing exact distance-head capacity')
        marked = meta.get('on_policy_rows')
        if (not isinstance(marked, list) or not marked or
                any(type(i) is not int or not 0 <= i < len(seeds) for i in marked) or
                len(set(marked)) != len(marked) or
                set(map(int, seeds[marked])) != expected):
            raise ValueError('all selected levels need valid model-visited row indices')
    provenance = meta.get('on_policy_provenance', {})
    behavior = meta.get('behavior_checkpoint', {})
    if (meta.get('source') != 'generated_only' or meta.get('collection_policy') != 'model_greedy' or
            provenance.get('official_inputs_used') is not False or
            type(provenance.get('oracle_actions_in_policy_rollout')) is not int or
            provenance['oracle_actions_in_policy_rollout'] != 0 or
            behavior.get('sha256') != repair.PARENT_SHA or
            Path(behavior.get('path', '')).resolve() != repair.PARENT):
        raise ValueError('supplement needs public-policy fresh-parent provenance')
    if meta.get('history') != 8 or meta.get('alternatives_per_state') != 4:
        raise ValueError('supplement requires H8 and all four branches')
    return source, actual_sha


def extended_guard(directory, source, source_sha, base_guard):
    from pebby.agent.world_model import REQUIRED_ARRAYS, OPTIONAL_ARRAYS

    guard = base_guard()
    paths = [Path(__file__).resolve(), directory / 'report.json', directory / 'train.jsonl', BANK,
             repair.ROOT / 'tools/collect_reference_onpolicy.py',
             repair.ROOT / 'tools/collect_onpolicy_world.py']
    names = sorted(set((*REQUIRED_ARRAYS, *OPTIONAL_ARRAYS, 'context_index', 'meta')))
    schema_sha = hashlib.sha256(json.dumps(names).encode()).hexdigest()[:16]
    manifest_path = repair.DATA / 'array-cache' / f'{source_sha}-{schema_sha}' / 'manifest.json'
    if not manifest_path.is_file():
        raise ValueError('the exact trainer-schema supplemental mmap cache is required')
    manifest = json.loads(manifest_path.read_text())
    if manifest['source_sha256'] != source_sha:
        raise ValueError('supplemental mmap source hash differs')
    paths.append(manifest_path)
    guard.hashes.update({str(path): repair.digest(path) for path in paths})
    large = [source, *[manifest_path.parent / (name + '.npy') for name in manifest['arrays']]]
    guard.stats.update({str(path): guard.stat(path) for path in large})
    if repair.digest(source) != source_sha:
        raise ValueError('supplement changed before fitting')
    return guard


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    directory = args.data_dir.resolve()
    repair.memory_check()
    source, source_sha = validate_supplement(directory)
    base_arguments, base_guard = repair.training_arguments, repair.source_guard

    def arguments(options, parent):
        result = base_arguments(options, parent)
        result += ['--on-policy-data', str(source), '--on-policy-fraction', '.25',
                   '--on-policy-auxiliary-fraction', '0']
        return result

    def guard():
        return extended_guard(directory, source, source_sha, base_guard)

    with patch.object(repair, 'training_arguments', arguments), patch.object(repair, 'source_guard', guard):
        repair.main(['--out-dir', str(args.out_dir), '--grounding-weight', '1',
                     '--sigreg-weight', '.1', '--lr', '.0001', '--seed', '42'])


if __name__ == '__main__':
    main()
