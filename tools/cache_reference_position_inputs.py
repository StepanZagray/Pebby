"""Read-only, bounded frozen public feature extraction for position recall.

Run with PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m tools.cache_reference_position_inputs.
No training or official inputs. Publication is an atomic directory rename.
"""
import gc
import hashlib
import json
import mmap
import os
from pathlib import Path
import signal
import time
from datetime import datetime
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from pebby.agent.world_model import CELLS, load_world_checkpoint, REQUIRED_ARRAYS, OPTIONAL_ARRAYS
from pebby.agent.world_grounding import labels, SIZES
from pebby.agent.world_runtime import configure_execution
from pebby.agent.world_training_objectives import world_losses
from tools.train_reference_onpolicy import validate_supplement

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/reference-world-base-v1'
OUTPUT = ROOT / 'data/reference-position-inputs-v1'
EVIDENCE = ROOT / 'artifacts/reference-position-repair-v1/cache'
PARENT = ROOT / 'checkpoints/ls20-reference-base-v1.pt'
PARENT_SHA = '6db7f40d4008ff809e9b3a8b05d6a9a18801b343f02e52a9f585eece5e31cff9'
HASHES = {
    'base': 'e5be3fcd888ffe56c7be74d5117bcc51ec58ce03e43aa3a3dc929a4db355a6a7',
    'validation': '0f6569c2a88014723bf4982882d3d1b201c6ae733b861ac6e24644aa94cafe52',
    'supplement': 'accfed3f5c239b3c730cac02960a0e6712965ca2934146aa7cac11f89f9a8932',
}
PUBLIC = ('frames', 'history_valid', 'previous_actions')
TEACHERS = ('player_cell', 'current_triple', 'current_steps', 'current_lives',
            'next_player_cell', 'next_triple', 'next_steps', 'next_lives')
NAMES = ('player', 'shape', 'color', 'rotation', 'steps', 'lives')
SCHEMA = {'inputs': ('float32', (1766,)), 'labels': ('int64', (6,)),
          'seeds': ('int64', ()), 'source': ('uint8', ()), 'rows': ('int64', ()),
          'branch': ('int8', ()), 'terminal': ('bool', ()),
          'lost_life': ('bool', ()), 'won': ('bool', ())}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stat(path):
    value = Path(path).stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def write(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def guard(report):
    available = next(int(line.split()[1]) * 1024 for line in
                     Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    report['minimum_memavailable_bytes'] = min(available, report.get('minimum_memavailable_bytes', available))
    if available < 7 * 2**30:
        raise MemoryError('MemAvailable below 7 GiB abort threshold; 6 GiB reserve')
    if torch.cuda.is_initialized() and torch.cuda.max_memory_allocated() >= 7 * 2**30:
        raise MemoryError('GPU allocation reached 7 GiB ceiling')


def release(arrays):
    for value in arrays.values():
        value._mmap.madvise(mmap.MADV_DONTNEED)


def observations(arrays, rows, branches):
    """Reconstruct only public history; events choose reset semantics, never model inputs."""
    frames, valid, actions = [np.array(arrays[key][rows], copy=True) for key in PUBLIC]
    actual = branches >= 0
    if actual.any():
        next_frames = np.array(arrays['next_frames'][rows[actual], branches[actual]], copy=True)
        frames[actual, :-1] = frames[actual, 1:]
        frames[actual, -1] = next_frames
        valid[actual, :-1] = valid[actual, 1:]
        valid[actual, -1] = True
        actions[actual, :-1] = actions[actual, 1:]
        actions[actual, -1] = branches[actual]
        reset = np.zeros(len(rows), dtype=bool)
        reset[actual] = arrays['lost_life'][rows[actual], branches[actual]]
        frames[reset] = frames[reset, -1, None]
        valid[reset] = False
        valid[reset, -1] = True
        actions[reset] = -1
    return frames, valid, actions


def features(model, encoding):
    original = model.projector_inputs(encoding['state'], encoding['raw'][:, CELLS:].flatten(1), encoding['glyph'])
    assert original.shape[-1] == 1742
    probabilities = model.player_weights(encoding['cells'])[1].detach().reshape(-1, 12, 12)
    marginals = torch.cat((probabilities.sum(2), probabilities.sum(1)), -1).detach()
    torch.testing.assert_close(marginals[:, :12].sum(-1), torch.ones(len(marginals), device='cuda'))
    torch.testing.assert_close(marginals[:, 12:].sum(-1), torch.ones(len(marginals), device='cuda'))
    result = torch.cat((original, marginals), -1).detach()
    assert result.dtype == torch.float32 and bool(torch.isfinite(result).all())
    return result


def parity(model, arrays):
    lost = np.asarray(arrays['lost_life'])
    reset_rows = np.flatnonzero(lost.any(1))[:4]
    normal_rows = np.flatnonzero(~lost.any(1))[:4]
    rows = np.concatenate((normal_rows, reset_rows))
    assert len(reset_rows) == len(normal_rows) == 4
    batch = {key: torch.from_numpy(np.array(value[rows], copy=True)).cuda()
             for key, value in arrays.items() if key != 'meta'}
    captures = []
    original = model.assemble

    def capture(tokens, valid, actions, *args, **kwargs):
        result = original(tokens, valid, actions, *args, **kwargs)
        captures.append((tokens.detach().clone(), valid.detach().clone(), actions.detach().clone(),
                         {key: value.detach().clone() if value is not None else None for key, value in result.items()}))
        return result

    model.encoder_chunk_size = 0
    try:
        with patch.object(model, 'assemble', capture):
            world_losses(model, batch)
        assert len(captures) == 2
        inputs = observations(arrays, np.repeat(rows, 4), np.tile(np.arange(4), len(rows)))
        inputs = [torch.from_numpy(value).cuda() for value in inputs]
        frames, valid, actions = model._prepare(*inputs)
        actual_tokens = model.frame_tokens(frames.flatten(0, 1)).view(len(frames), 8, model.tokens, -1)
        torch.testing.assert_close(actual_tokens, captures[1][0], rtol=1e-5, atol=1e-6)
        assert torch.equal(valid, captures[1][1]) and torch.equal(actions, captures[1][2])
        encoding = model.encode(*inputs)
        differences = {}
        for key in ('state', 'raw', 'glyph', 'latent'):
            torch.testing.assert_close(encoding[key], captures[1][3][key], rtol=1e-4, atol=3e-5)
            differences[key] = float((encoding[key] - captures[1][3][key]).abs().max())
        torch.testing.assert_close(features(model, encoding), features(model, captures[1][3]), rtol=1e-4, atol=3e-5)
        return dict(rows=rows.tolist(), actual_observations=len(rows) * 4,
                    reset_observations=int(lost[rows].sum()), normal_observations=int((~lost[rows]).sum()),
                    tokens_match=True, validity_exact=True, actions_exact=True,
                    encoding_max_absolute_difference=differences, features_match=True,
                    reference='captured actual encoding from active world_training_objectives.world_losses')
    finally:
        model.encoder_chunk_size = 128
        release(arrays)


def select(arrays, candidates, rng):
    seeds = np.asarray(arrays['seeds'][candidates])
    order = np.argsort(seeds, kind='stable')
    grouped = candidates[order]
    _, starts, counts = np.unique(seeds[order], return_index=True, return_counts=True)
    result = np.empty(int(np.minimum(counts, 8).sum()), dtype=np.int64)
    offset = 0
    for start, count in zip(starts, counts):
        n = min(8, int(count))
        result[offset:offset+n] = rng.choice(grouped[start:start+count], n, replace=False)
        offset += n
    assert len(np.unique(result)) == len(result)
    return result


def add_metrics(bucket, scores, control, target, masks):
    ground_correct = torch.stack([(s.argmax(-1) == y) for s, y in zip(scores, target)], 1).float()
    ground_ce = torch.stack([F.cross_entropy(s, y, reduction='none') for s, y in zip(scores, target)], 1)
    control_correct = torch.stack([(s.argmax(-1) == y) for s, y in zip(control, target)], 1).float()
    control_ce = torch.stack([F.cross_entropy(s, y, reduction='none') for s, y in zip(control, target)], 1)
    for population, mask in masks.items():
        mask = torch.as_tensor(mask, device='cuda')
        entry = bucket.setdefault(population, {'count': 0, 'ground_correct': np.zeros(6),
            'ground_ce': np.zeros(6), 'control_correct': np.zeros(4), 'control_ce': np.zeros(4)})
        entry['count'] += int(mask.sum())
        for key, values in [('ground_correct', ground_correct), ('ground_ce', ground_ce),
                            ('control_correct', control_correct), ('control_ce', control_ce)]:
            entry[key] += values[mask].sum(0).cpu().numpy()


def formatted(metrics):
    result = {}
    for source, populations in metrics.items():
        result[source] = {}
        for name, values in populations.items():
            n = values['count']
            entry = {'count': n}
            for prefix, title, names in [('ground', 'existing_grounding', NAMES), ('control', 'dedicated_controls', NAMES[:4])]:
                entry[title] = {field: {'accuracy': float(values[prefix+'_correct'][i] / n) if n else None,
                    'cross_entropy': float(values[prefix+'_ce'][i] / n) if n else None} for i, field in enumerate(names)}
            result[source][name] = entry
    return result


def main():
    assert not OUTPUT.exists(), 'refusing overwrite of published cache'
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    assert not (EVIDENCE / 'report.json').exists(), 'refusing overwrite of previous evidence'
    staging = OUTPUT.with_name('.' + OUTPUT.name + '.staging-' + str(os.getpid()))
    staging.mkdir()
    report = dict(status='validating', pid=os.getpid(), started_local=datetime.now().astimezone().isoformat(),
                  parent_sha256=PARENT_SHA, parent_path=str(PARENT), official_inputs_used=False, output=str(OUTPUT), staging=str(staging),
                  public_input_contract=list(PUBLIC), label_sizes=list(SIZES), feature_width=1766,
                  feature_layout=[['original_projector_inputs', 1742], ['player_softmax_row_marginals', 12],
                                  ['player_softmax_column_marginals', 12]],
                  settings=dict(device='cuda', precision='float32', tf32=False, temporal_backend='math',
                                encoder_chunk_size=128, feature_batch_size=1024, sampling_seed=20260916,
                                gpu_deadline_seconds=600, reserve_gib=6, abort_memavailable_gib=7, gpu_ceiling_gib=7),
                  policy_optimizer_used=False, progress={})
    write(EVIDENCE / 'report.json', report)
    print('PID', os.getpid(), 'schema ready:', SCHEMA, flush=True)
    started = time.monotonic()
    arrays, outputs, metrics = {}, {}, {}
    try:
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        guard(report)
        supplement, supplement_sha = validate_supplement(ROOT / 'data/reference-onpolicy-v1')
        assert supplement_sha == HASHES['supplement']
        source_manifest = json.loads((DATA / 'manifest.json').read_text())
        bindings = dict(source_manifest['sources'])
        bindings.update({str(path): sha(path) for path in [Path(__file__), PARENT, DATA/'manifest.json',
            DATA/'build-report.json', ROOT/'data/reference-onpolicy-v1/report.json',
            ROOT/'data/reference-onpolicy-v1/train.jsonl', ROOT/'tools/train_reference_onpolicy.py',
            ROOT/'tools/train_reference_repair.py',
            *[ROOT/'pebby/agent'/name for name in ('world_model.py', 'world_runtime.py', 'world_grounding.py',
                'world_training_objectives.py', 'world_rollout.py', 'world_cache.py', 'glyph_model.py', 'world_readout.py')]]})
        assert bindings[str(PARENT)] == PARENT_SHA
        for path, expected in bindings.items():
            assert sha(path) == expected, path
        build_report = json.loads((DATA/'build-report.json').read_text())
        assert build_report['status'] == 'complete' and build_report['sources_unchanged'] is True
        bank_report = json.loads((ROOT/'data/ls20-reference-unequal-v1/generation-report.json').read_text())
        assert build_report['official_frames_or_routes_used'] is False
        report['upstream_bank_calibration'] = {
            'official_inputs_used': bank_report['official_inputs_used'],
            'scope': bank_report['official_input_scope'],
            'official_frames_or_routes_used_in_feature_source': False,
        }
        banks = {split: {json.loads(line)['seed'] for line in
                 (ROOT/'data/ls20-reference-unequal-v1'/f'{split}.jsonl').read_text().splitlines()}
                 for split in ('train', 'validation')}
        assert len(banks['train']) == 10000 and len(banks['validation']) == 500
        assert banks['train'].isdisjoint(banks['validation'])
        large_stats, verified_hashes, marked = {}, {}, None
        schema = sorted(set((*REQUIRED_ARRAYS, *OPTIONAL_ARRAYS, 'context_index', 'meta')))
        assert hashlib.sha256(json.dumps(schema).encode()).hexdigest()[:16] == '58c1c61b602f42df'
        for name, expected in HASHES.items():
            source = supplement if name == 'supplement' else DATA / ('train.npz' if name == 'base' else 'validation.npz')
            large_stats[str(source)] = stat(source)
            assert sha(source) == expected
            cache = DATA/'array-cache'/(expected+'-58c1c61b602f42df')
            manifest_path = cache/'manifest.json'
            bindings[str(manifest_path)] = sha(manifest_path)
            manifest = json.loads(manifest_path.read_text())
            assert manifest['source_sha256'] == expected
            assert set(manifest['arrays']) == set(schema)
            arrays[name] = {}
            for key, info in manifest['arrays'].items():
                guard(report)
                path = cache/(key+'.npy')
                large_stats[str(path)] = stat(path)
                actual = sha(path)
                assert actual == info['sha256'], str(path)
                verified_hashes[str(path)] = actual
                value = np.load(path, mmap_mode='r', allow_pickle=False)
                assert list(value.shape) == info['shape'] and value.dtype.str == info['dtype']
                arrays[name][key] = value
            meta = json.loads(str(arrays[name]['meta'].item()))
            assert meta['source'] == 'generated_only'
            assert arrays[name]['frames'].shape[1:] == (8, 64, 64)
            if name == 'supplement':
                marked = np.asarray(meta['on_policy_rows'], dtype=np.int64)
                assert meta['on_policy_provenance']['official_inputs_used'] is False
                assert set(map(int, arrays[name]['seeds'])) <= banks['train']
            else:
                split = 'train' if name == 'base' else 'validation'
                assert set(map(int, arrays[name]['seeds'])) == banks[split]
                assert build_report['splits'][split]['sha256'] == expected
            del meta
            release(arrays[name])
            print('verified', name, flush=True)
        report.update(source_bindings=bindings, large_array_stats_before=large_stats,
                      verified_source_array_hashes=verified_hashes, published_npz_sha256=HASHES,
                      array_verification='read-only equivalent of existing cached_arrays: NPZ hash, schema hash, every array hash/shape/dtype',
                      validation_disjoint=True, exact_supplement_bank_validated=True)
        rng = np.random.default_rng(20260916)
        indices = {'base': select(arrays['base'], np.arange(len(arrays['base']['seeds'])), rng),
                   'supplement': select(arrays['supplement'], marked, rng)}
        fixed = ROOT/'artifacts/reference-state-decoding-probe-v1/validation-indices.npy'
        bindings[str(fixed)] = sha(fixed)
        indices['validation'] = np.load(fixed, allow_pickle=False)
        assert len(indices['base']) == 80000 and len(indices['validation']) == 4000
        selection = {}
        for name, chosen in indices.items():
            assert len(np.unique(chosen)) == len(chosen)
            seeds, counts = np.unique(arrays[name]['seeds'][chosen], return_counts=True)
            assert counts.max() <= 8
            assert len(seeds) == {'base': 10000, 'supplement': 1000, 'validation': 500}[name]
            if name == 'validation':
                assert np.all(counts == 8)
            path = staging/(name+'-selected-indices.npy')
            np.save(path, chosen)
            selection[name] = dict(rows=len(chosen), seeds=len(seeds), min_rows_per_seed=int(counts.min()),
                                   max_rows_per_seed=int(counts.max()), sha256=sha(path), path=path.name)
        report['selection'] = selection
        report['source_bindings'] = bindings
        write(EVIDENCE/'report.json', report)
        guard(report)
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('600-second GPU extraction deadline')))
        signal.alarm(600)
        gpu_started = time.monotonic()
        print('GPU_START', datetime.now().astimezone().isoformat(), flush=True)
        model, _ = load_world_checkpoint(PARENT, 'cuda')
        model.eval().requires_grad_(False)
        model.encoder_chunk_size = 128
        configure_execution(model, temporal_backend='math')
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            report['history_parity'] = {name: parity(model, arrays[name]) for name in arrays}
            for split, sources in [('train', ('base', 'supplement')), ('validation', ('validation',))]:
                count = sum(len(indices[name]) * 5 for name in sources)
                directory = staging/split
                directory.mkdir()
                outputs[split] = {key: np.lib.format.open_memmap(directory/(key+'.npy'), mode='w+', dtype=dtype,
                                  shape=(count, *tail)) for key, (dtype, tail) in SCHEMA.items()}
                offset = 0
                for name in sources:
                    chosen, source_arrays = indices[name], arrays[name]
                    bucket = metrics.setdefault(split+'/'+name, {})
                    for begin in range(0, len(chosen) * 5, 1024):
                        guard(report)
                        flat = np.arange(begin, min(begin+1024, len(chosen)*5))
                        rows, branch = chosen[flat//5], (flat%5-1).astype(np.int8)
                        current, actual = branch == -1, branch >= 0
                        public = observations(source_arrays, rows, branch)
                        x = [torch.from_numpy(value).cuda() for value in public]
                        encoding = model.encode(*x)
                        feature = features(model, encoding)
                        target = torch.empty((len(rows), 6), dtype=torch.int64, device='cuda')
                        for mask, next_state in ((current, False), (actual, True)):
                            label_batch = {key: np.array(source_arrays[key][rows[mask]], copy=True) for key in TEACHERS}
                            if next_state:
                                label_batch = {key: (value[np.arange(mask.sum()), branch[mask]] if key.startswith('next_') else value)
                                               for key, value in label_batch.items()}
                            target[torch.as_tensor(mask, device='cuda')] = torch.stack(labels(label_batch, next_state=next_state, device='cuda'), 1)
                        events = {}
                        for key in ('terminal', 'lost_life', 'won'):
                            events[key] = np.zeros(len(rows), dtype=bool)
                            events[key][actual] = source_arrays[key][rows[actual], branch[actual]]
                        masks = {'current': current, 'actual': actual, 'actual_nonterminal': actual & ~events['terminal'],
                                 'terminal': events['terminal'], 'lost_life': events['lost_life'], 'won': events['won']}
                        control = [model.player_weights(encoding['cells'])[0],
                                   *encoding['glyph'].clamp_min(1e-30).log().split((6, 4, 4), -1)]
                        add_metrics(bucket, model.grounding_head(encoding['latent']), control, target.unbind(1), masks)
                        payload = dict(inputs=feature.cpu().numpy(), labels=target.cpu().numpy(),
                            seeds=np.array(source_arrays['seeds'][rows]), source=np.full(len(rows), int(name == 'supplement'), dtype=np.uint8),
                            rows=rows, branch=branch, **events)
                        for key, value in payload.items():
                            outputs[split][key][offset:offset+len(rows)] = value
                        offset += len(rows)
                        release(source_arrays)
                        if begin % (1024 * 16) == 0 or begin + len(rows) == len(chosen)*5:
                            for value in outputs[split].values():
                                value.flush()
                                value._mmap.madvise(mmap.MADV_DONTNEED)
                            report['progress'][split+'/'+name] = begin + len(rows)
                            print('extract', split, name, begin+len(rows), '/', len(chosen)*5, flush=True)
                            write(EVIDENCE/'report.json', report)
                    del x, encoding, feature, control, target, payload, public, label_batch
                assert offset == count
                for value in outputs[split].values():
                    value.flush()
                    value._mmap.madvise(mmap.MADV_DONTNEED)
            assert all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters())
            report['policy_gradients_absent'] = True
        report['peak_gpu_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
        report['gpu_elapsed_seconds'] = time.monotonic()-gpu_started
        assert report['gpu_elapsed_seconds'] <= 600
        del model
        gc.collect()
        torch.cuda.empty_cache()
        signal.alarm(0)
        report['gpu_extraction_finished'] = True
        report['baselines'] = formatted(metrics)
        report['outputs'] = {}
        for split, values in outputs.items():
            report['outputs'][split] = {}
            for key, value in values.items():
                guard(report)
                path = staging/split/(key+'.npy')
                report['outputs'][split][key] = dict(shape=list(value.shape), dtype=value.dtype.str,
                                                     sha256=sha(path), size_bytes=path.stat().st_size)
                value._mmap.close()
        report['source_bindings_after'] = {path: sha(path) for path in bindings}
        assert report['source_bindings_after'] == bindings
        report['large_array_stats_after'] = {path: stat(path) for path in large_stats}
        assert report['large_array_stats_after'] == large_stats
        report.update(status='complete', sources_unchanged=True, finite_features=True,
                      feature_storage_bytes=sum(info['inputs']['size_bytes'] for info in report['outputs'].values()),
                      public_input_only=True, no_engine_or_teacher_encoder_input=True,
                      learned_player_marginals_detached=True, policy_frozen=True,
                      child_processes_started=0, process_cleanup='no children; GPU objects released; launcher verifies extraction PID exit')
        report['elapsed_seconds'] = time.monotonic()-started
        report['finished_local'] = datetime.now().astimezone().isoformat()
        write(staging/'manifest.json', report)
        staging.rename(OUTPUT)
        print('PUBLISHED', OUTPUT, 'gpu_seconds', report['gpu_elapsed_seconds'], flush=True)
    except BaseException as error:
        report.update(status='failed', error=repr(error), partial_unpublished=True)
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds'] = time.monotonic()-started
        report['finished_local'] = datetime.now().astimezone().isoformat()
        write(EVIDENCE/'report.json', report)


if __name__ == '__main__':
    main()
