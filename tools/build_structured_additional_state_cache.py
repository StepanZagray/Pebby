"""A source-bound second H1 training view for the same ordered generated levels.

This is a separate cache, never a concatenation of duplicate level IDs. Encoder
chunks are independent of the eventual policy's 1024-distinct-level batch.
"""
from pebby.ls20.provenance import metadata_difficulty_stages, cache_difficulty_metadata

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time

import numpy as np
import torch

from tools.build_structured_field_cache import (FORMAT, LABELS, digest,
    encode_fields, event_counts, exact_labels)


def select_additional_rows(data, original_rows, ordered_seeds, difficulties, seed=43):
    meta = data['meta']
    if (meta.get('source'), meta.get('oracle_search'), meta.get('split')) != (
            'generated_only', 'complete_only', 'train'):
        raise ValueError('requires complete generated TRAIN source')
    source_seeds = np.asarray(data['seeds'])
    old = np.asarray(original_rows)
    levels = np.asarray(ordered_seeds)
    tiers = np.asarray(difficulties)
    if old.ndim != 1 or old.dtype.kind not in 'iu' or levels.shape != old.shape or tiers.shape != old.shape:
        raise ValueError('invalid paired row/level shapes')
    if len(np.unique(levels)) != len(levels) or len(levels) == 0:
        raise ValueError('paired levels must be nonempty and distinct')
    if not ((source_seeds >= 0) & (source_seeds < 1_000_000)).all():
        raise ValueError('TRAIN namespace mismatch')
    if not ((old >= 0) & (old < len(source_seeds))).all() or not np.array_equal(source_seeds[old], levels):
        raise ValueError('paired source row/seed order mismatch')
    mapping = {int(x['seed']): int(x['difficulty']) for x in meta['levels']}
    if not np.array_equal(tiers, [mapping[int(s)] for s in levels]):
        raise ValueError('paired difficulty mismatch')
    for name in LABELS.values():
        if data.get(name) is None:
            raise ValueError(f'exact source label missing: {name}')
    grouped = {int(s): [] for s in levels}
    for row, s in enumerate(source_seeds):
        if int(s) in grouped:
            grouped[int(s)].append(row)
    rng = np.random.default_rng(seed)
    chosen = []
    singleton = []
    for row, s in zip(old, levels):
        candidates = [r for r in grouped[int(s)] if r != int(row)]
        singleton.append(not candidates)
        chosen.append(int(rng.choice(candidates)) if candidates else int(row))
    chosen = np.asarray(chosen, dtype=np.int64)
    singleton = np.asarray(singleton, dtype=bool)
    if not np.array_equal(chosen == old, singleton):
        raise AssertionError('additional selection repeated a nonsingleton row')
    return chosen, singleton


def load_parent(path):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if any(manifest.get(k) != v for k, v in {
            'format': FORMAT, 'status': 'complete', 'source': 'generated_only', 'split': 'train'}.items()):
        raise ValueError('paired cache must be complete generated TRAIN H1')
    arrays = {}
    required = {'fields', 'next_fields', 'seeds', 'source_rows', 'difficulties', *LABELS}
    if not required <= manifest['arrays'].keys():
        raise ValueError('paired array inventory incomplete')
    for name, info in manifest['arrays'].items():
        if not name.replace('_', '').isalnum():
            raise ValueError('invalid array name')
        item = path / f'{name}.npy'
        if digest(item) != info['sha256']:
            raise ValueError(f'paired array hash mismatch: {name}')
        array = np.load(item, mmap_mode='r', allow_pickle=False)
        if list(array.shape) != info['shape'] or str(array.dtype) != info['dtype']:
            raise ValueError(f'paired array schema mismatch: {name}')
        arrays[name] = array
    n = len(arrays['seeds'])
    if arrays['fields'].shape != (n, 148, 96) or arrays['next_fields'].shape != (n, 4, 148, 96):
        raise ValueError('paired fields must have four alternative successors')
    return arrays, manifest


def checked_hashes(hashes):
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError(f'bound source changed: {path}')


def build(data, source, parent, output, assembler, *, seed=43, batch_size=32,
          max_encoder_batch=128, device='cpu'):
    start = time.monotonic()
    parent, output, source = Path(parent), Path(output), Path(source)
    if output.exists():
        raise FileExistsError(output)
    if batch_size < 1 or not 1 <= max_encoder_batch <= 128 or device not in ('cpu', 'cuda'):
        raise ValueError('invalid batch/device; encoder chunks must be1..128')
    arrays, previous = load_parent(parent)
    if source.resolve() != Path(previous['source_path']).resolve() or digest(source) != previous['source_sha256']:
        raise ValueError('paired source path/hash mismatch')
    from pebby.agent.structured_factored_policy import canonical_metadata
    actual = canonical_metadata(assembler.metadata())
    expected = canonical_metadata(previous['field_encoder'])
    for key in ('config', 'sources', 'parameter_counts'):
        if actual.get(key) != expected.get(key):
            raise ValueError(f'paired encoder mismatch: {key}')
    hashes = {str(parent / 'manifest.json'): digest(parent / 'manifest.json'),
              str(source): previous['source_sha256'], __file__: digest(__file__),
              'tools/build_structured_field_cache.py': digest('tools/build_structured_field_cache.py')}
    hashes.update(previous['field_encoder']['code_hashes'])
    hashes.update(previous['field_encoder'].get('checkpoint_hashes', {}))
    for key in ('world_checkpoint', 'visibility'):
        binding = previous['field_encoder']['sources'][key]
        hashes[binding.get('path', binding.get('checkpoint'))] = binding['sha256']
    checked_hashes(hashes)
    old = arrays['source_rows']
    rows, singleton = select_additional_rows(data, old, arrays['seeds'], arrays['difficulties'], seed)
    for target, original in LABELS.items():
        if not np.array_equal(arrays[target], data[original][old]):
            raise ValueError(f'paired exact source label mismatch: {target}')
    labels = exact_labels(data, rows)
    n = len(rows)
    if not np.array_equal(data['seeds'][rows], arrays['seeds']):
        raise AssertionError('changed ordered levels')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assembler.eval()
    try:
        fields = np.lib.format.open_memmap(temporary / 'fields.npy', mode='w+', dtype=np.float16, shape=(n,148,96))
        future = np.lib.format.open_memmap(temporary / 'next_fields.npy', mode='w+', dtype=np.float16, shape=(n,4,148,96))
        encoding_seconds = 0.; quantization = 0.
        for begin in range(0, n, batch_size):
            selected = rows[begin:begin+batch_size]
            batch = {key: np.array(data[key][selected], copy=True) for key in
                     ('frames','history_valid','previous_actions','next_frames','lost_life')}
            tick = time.monotonic()
            f, nxt, error = encode_fields(assembler, batch, max_encoder_batch)
            encoding_seconds += time.monotonic() - tick
            fields[begin:begin+len(selected)] = f
            future[begin:begin+len(selected)] = nxt
            quantization = max(quantization, error)
            if begin + len(selected) == n or (begin+len(selected)) % 512 == 0:
                print(json.dumps({'encoded':begin+len(selected),'levels':n,'encoding_seconds':encoding_seconds}), flush=True)
        fields.flush(); future.flush(); del fields, future
        saved = {'seeds':np.array(arrays['seeds']), 'difficulties':np.array(arrays['difficulties']),
                 'source_rows':rows, **labels}
        if data.get('context_index') is not None:
            saved['context_index'] = np.array(data['context_index'][rows])
        for name, array in saved.items():
            np.save(temporary / f'{name}.npy', array, allow_pickle=False)
        inventory = {}
        for path in sorted(temporary.glob('*.npy')):
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            for begin in range(0,len(array),128):
                if not np.isfinite(array[begin:begin+128]).all():
                    raise ValueError(f'nonfinite output: {path.name}')
            inventory[path.stem] = {'shape':list(array.shape),'dtype':str(array.dtype),'sha256':digest(path)}
        checked_hashes(hashes)
        result = {**cache_difficulty_metadata(data['meta'], arrays['seeds']), 'format':FORMAT,'status':'complete','source':'generated_only','split':'train',
                  'source_path':str(source),'source_sha256':previous['source_sha256'],
                  'builder_sha256':digest(__file__),'source_hashes':hashes,
                  'field_encoder':previous['field_encoder'],'arrays':inventory,'selected_levels':n,
                  'paired_source':{'manifest_path':str(parent/'manifest.json'),'manifest_sha256':hashes[str(parent/'manifest.json')],
                       'seeds_sha256':previous['arrays']['seeds']['sha256'],
                       'source_rows_sha256':previous['arrays']['source_rows']['sha256'],
                       'identical_ordered_seeds':True,'different_source_rows':int((~singleton).sum()),
                       'singleton_seeds':list(map(int,arrays['seeds'][singleton]))},
                  'selection':{'seed':seed,'method':'same ordered TRAIN levels; uniform different source row per level; retain original row only for explicit singleton',
                       'difficulty_counts':{str(d):int((arrays['difficulties']==d).sum()) for d in metadata_difficulty_stages(data['meta'])},
                       'event_enrichment':False,'validation_selection':'none; existing validation unchanged'},
                  'encoding_precision':{'device':device,'compute_dtype':'float32','cache_dtype':'float16',
                       'autocast':False,'tf32':False,'max_encoder_batch':max_encoder_batch},
                  'event_coverage':event_counts(labels),'history_contract':previous.get('history_contract'),
                  'field_contract':'learned public-input semantic distillation; exact labels are supervision only',
                  'supervision_contract':{'exact_label_source_mapping':LABELS,'labels_not_assembler_inputs':True,'negative_steps_preserved':True},
                  'encoding_seconds':encoding_seconds,'elapsed_seconds':time.monotonic()-start,
                  'float16_max_absolute_quantization_error':quantization,'pid':os.getpid()}
        (temporary/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
        temporary.rename(output)
    finally:
        if temporary.exists():shutil.rmtree(temporary)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired-train', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-encoder-batch', type=int, default=128)
    parser.add_argument('--seconds', type=int, default=180)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 600:raise ValueError('seconds must be1..600')
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('additional-state cache time budget')))
    signal.alarm(args.seconds);torch.set_num_threads(1)
    print('PID',os.getpid(),flush=True)
    previous = json.loads((args.paired_train/'manifest.json').read_text())
    source = Path(previous['source_path'])
    if digest(source) != previous['source_sha256']:raise ValueError('paired source hash mismatch')
    from pebby.agent.world_train import load_dataset
    from pebby.agent.structured_field import load_structured_field_encoder
    provenance = previous['field_encoder']['sources']
    encoder = load_structured_field_encoder(provenance['world_checkpoint']['path'],
                                           provenance['visibility']['checkpoint'],device=args.device)
    data = load_dataset(source, history=8, cache_dir=Path('data/world-array-cache'))
    result = build(data,source,args.paired_train,args.output,encoder,seed=args.seed,batch_size=args.batch_size,
                   max_encoder_batch=args.max_encoder_batch,device=args.device)
    signal.alarm(0)
    print(json.dumps({k:result[k] for k in ('status','pid','selected_levels','paired_source','event_coverage','encoding_seconds','elapsed_seconds')}),flush=True)


if __name__ == '__main__':main()
