"""Separate generated closing-K4 TRAIN field cache; final targets use real branches."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import tempfile
import time

import numpy as np
import torch

from pebby.agent import world_closing_sequences as closing
from pebby.agent.world_data import history_arrays
from pebby.agent.world_rollout import replace_sequence_histories
from pebby.agent.world_train import as_tensors, load_dataset
from tools.build_structured_field_cache import digest, event_counts
from tools.build_structured_sequence_cache import CURRENT

FORMAT = 'pebby.structured-closing-sequence-cache.v1'
DEFAULT_ATTESTATION_SHA = 'eeb73a76c9db26e3b23cf4961be822c2cee4c346067c73c37ee3f89e051290d0'


def select_rows(data, index, count, seed=42):
    if index.meta.get('mode') != 'closing_only_train' or index.meta.get('split') != 'train':
        raise ValueError('closing TRAIN index required')
    levels = np.asarray(data['seeds'])[index.anchor_row]
    if len(np.unique(levels)) != len(levels) or not ((levels >= 0) & (levels < 1_000_000)).all():
        raise ValueError('distinct generated TRAIN levels required')
    if not 1 <= count <= len(levels):
        raise ValueError('requested count exceeds distinct closing levels')
    difficulty = {int(x['seed']): int(x['difficulty']) for x in data['meta']['levels']}
    rng = np.random.default_rng(seed)
    groups = [list(rng.permutation(index.anchor_row[[difficulty[int(s)] == d for s in levels]])) for d in range(1, 6)]
    selected = []
    while len(selected) < count:
        for group in groups:
            if group and len(selected) < count:
                selected.append(group.pop())
    return np.asarray(selected, np.int64)


def sequence_batch(data, tensors, index, anchors):
    rows, actions = index.lookup(anchors)
    if not np.array_equal(rows, np.asarray(anchors)[:, None] + np.arange(4)):
        raise ValueError('closing branch rows must be contiguous')
    for name in (*closing.TARGETS, *CURRENT.values()):
        if data.get(name) is None:
            raise ValueError(f'exact source label missing: {name}')
    targets = closing.sequence_targets(tensors, index, torch.from_numpy(np.array(anchors)))
    if any(bool(targets[key][:, :3].any()) for key in ('terminal', 'won', 'lost_life')):
        raise ValueError('first three transitions must remain live without resets')
    current = tuple(tensors[key][anchors] for key in ('frames', 'history_valid', 'previous_actions'))
    b, h = current[0].shape[:2]
    # All rows are chronological. The production helper applies a reset only at H4.
    dummy = (current[0].new_zeros((b * 4, h, 64, 64)),
             current[1].new_zeros((b * 4, h)), current[2].new_zeros((b * 4, h)))
    histories = replace_sequence_histories(*dummy, current[0], targets['next_frames'][:, :, None],
        current[1], current[2], torch.ones(b, dtype=torch.bool), targets['rollout_actions'], targets['lost_life'])
    histories = tuple(value.reshape(b, 4, *value.shape[1:]) for value in histories)
    for actual, key in zip(histories, ('frames', 'history_valid', 'previous_actions')):
        if not torch.equal(actual[:, :3], tensors[key][rows[:, 1:]]):
            raise ValueError(f'interior chronological {key} mismatch')
    # Independent history_arrays uses the actual last public source row + chosen frame.
    for i, (last, action) in enumerate(zip(rows[:, -1], actions[:, -1])):
        if data['lost_life'][last, action]:
            expected = history_arrays([data['next_frames'][last, action]], [-1], 8)
        else:
            mask = np.asarray(data['history_valid'][last])
            observed = list(data['frames'][last][mask]) + [data['next_frames'][last, action]]
            producing = list(data['previous_actions'][last][mask]) + [int(action)]
            expected = history_arrays(observed, producing, 8)
        for actual, value in zip(histories, expected):
            if not np.array_equal(actual[i, -1].numpy(), value):
                raise ValueError('closing target history differs from independent append/reset')
    labels = {out: np.array(data[source][anchors], copy=True) for out, source in CURRENT.items()}
    for key in closing.TARGETS:
        if key != 'next_frames':
            labels[key] = np.array(data[key][rows, actions], copy=True)
            if not np.array_equal(labels[key], targets[key].numpy()):
                raise ValueError('actual closing branch label mismatch')
    labels['actions'] = np.array(actions, copy=True)
    labels['branch_rows'] = np.array(rows, copy=True)
    return current, histories, labels


def encode_sequences(assembler, data, tensors, index, rows, encoder_batch_size=None):
    current, histories, labels = sequence_batch(data, tensors, index, rows)
    def encode(inputs):
        size = encoder_batch_size or len(inputs[0])
        return torch.cat([assembler(*(value[first:first+size] for value in inputs)).cpu()
                          for first in range(0, len(inputs[0]), size)])
    with torch.inference_mode():
        fields = encode(current)
        following = encode(tuple(value.flatten(0, 1) for value in histories))
    if fields.shape != (len(rows), 148, 96) or following.shape != (len(rows) * 4, 148, 96):
        raise ValueError('field shape must be148x96')
    result = []; error = 0.
    for value in (fields, following.reshape(len(rows), 4, 148, 96)):
        raw = value.float().numpy(); quantized = raw.astype(np.float16)
        if not np.isfinite(raw).all() or not np.isfinite(quantized).all():
            raise ValueError('nonfinite field or quantization overflow')
        error = max(error, float(np.abs(raw - quantized.astype(np.float32)).max()))
        result.append(quantized)
    return *result, labels, error


def write_cache(data, index, out, assembler, provenance, source_hashes, count=8, seed=42, batch_size=8,
                device='cpu', initial_data=None, initial_provenance=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    if batch_size < 1:
        raise ValueError('positive batch size required')
    rows = select_rows(data, index, count, seed); tensors = as_tensors(data)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.' + out.name + '-', dir=out.parent))
    start = time.monotonic(); encoding = 0.; error = 0.
    try:
        fields = np.lib.format.open_memmap(temporary/'fields.npy', mode='w+', dtype=np.float16, shape=(count,148,96))
        future = np.lib.format.open_memmap(temporary/'next_fields.npy', mode='w+', dtype=np.float16, shape=(count,4,148,96))
        batches = []
        for begin in range(0, count, batch_size):
            chosen = rows[begin:begin+batch_size]; tick = time.monotonic()
            current, targets, labels, discrepancy = encode_sequences(assembler, data, tensors, index, chosen, batch_size)
            encoding += time.monotonic() - tick; error = max(error, discrepancy)
            fields[begin:begin+len(chosen)] = current; future[begin:begin+len(chosen)] = targets
            batches.append(labels)
            print(json.dumps({'encoded':begin+len(chosen),'total':count}),flush=True)
        fields.flush(); future.flush(); del fields, future
        arrays = {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
        difficulty = {int(x['seed']): int(x['difficulty']) for x in data['meta']['levels']}
        arrays.update(source_rows=rows, seeds=np.array(data['seeds'][rows], copy=True),
                      difficulties=np.asarray([difficulty[int(s)] for s in data['seeds'][rows]], np.int8))
        if initial_data is not None:
            from tools.build_structured_initial_cache import select_initial_rows
            arrays['source_initial_rows'] = select_initial_rows(initial_data,arrays['seeds'],'train')
        for name, value in arrays.items():
            np.save(temporary/f'{name}.npy', value, allow_pickle=False)
        inventory = {}
        for path in sorted(temporary.glob('*.npy')):
            value = np.load(path, mmap_mode='r', allow_pickle=False)
            inventory[path.stem] = dict(shape=list(value.shape), dtype=str(value.dtype), sha256=digest(path))
        all_labels = {key: np.asarray(data[key])[index.branch_rows, index.branch_actions]
                      for key in ('lost_life','terminal','won','distances')}
        manifest = dict(format=FORMAT, status='complete', source='generated_only', split='train',
            mode='closing_only_chronological_K4', levels=count, field_encoder=provenance,
            source_hashes=source_hashes, source_index_metadata=index.meta, arrays=inventory,
            encoding_precision={'device':device,'compute_dtype':'float32','cache_dtype':'float16',
                                'autocast':False,'tf32':False,'max_encoder_batch':batch_size},
            initial_source_mapping=initial_provenance,
            selection_seed=seed, selection='Difficulty round-robin from independently shuffled per-difficulty pools; no event enrichment. Redistributes exhausted difficulty quotas.',
            difficulty_counts={str(d):int((arrays['difficulties']==d).sum()) for d in range(1,6)},
            event_coverage=event_counts(arrays), available_index_event_coverage=event_counts(all_labels),
            current_zero_optimal=int((arrays['optimal']==0).sum()),
            next_zero_optimal_by_horizon=(arrays['next_optimal']==0).sum(0).tolist(),
            source_current_zero_optimal=int((np.asarray(data['optimal'])[index.anchor_row]==0).sum()),
            history_verified_against_actual_source_rows=True,
            final_history_verified_independent_append_reset=True,
            encoding_seconds=encoding, elapsed_seconds=time.monotonic()-start, float16_max_absolute_error=error,
            target_kind='Frozen public field semantic distillation; exact branch labels are separate supervision.',
            current_inputs='Only public H8 frames/history_valid/previous_actions. Future flags affect target reset history only.',
            chronology='actions[N,4] records first3 producing actions plus externally attested last behavior action; branch_rows selects exact actual targets.',
            limitations=['No invented ending future current row; all four targets are recorded actual source branches.',
                         'First three transitions remain live. Last may win, become unreachable, or reset; source currently has zero closing resets.',
                         'Zero optimal masks for terminal/unreachable targets are valid; field regression does not require a policy target.',
                         'Generated TRAIN cache only; not a control result. Initial memory fields are not included.'])
        if any(digest(path) != sha for path, sha in source_hashes.items()):
            raise ValueError('source/index/attestation/encoder code changed')
        (temporary/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        temporary.rename(out)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('data/ls20-world-onpolicy-aggregate2-train.npz'))
    p.add_argument('--index', type=Path, default=Path('data/ls20-world-onpolicy-aggregate2-closing-k4.npz'))
    p.add_argument('--attestation', type=Path, default=Path('artifacts/world-closing-actions-batch1.json'))
    p.add_argument('--attestation-sha256', default=DEFAULT_ATTESTATION_SHA)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--levels', type=int, default=8)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    p.add_argument('--seconds', type=int, default=60)
    p.add_argument('--initial-source', type=Path)
    args = p.parse_args(); torch.set_num_threads(1); start = time.monotonic(); print('PID', os.getpid(), flush=True)
    if not 1 <= args.seconds <= 180:
        raise ValueError('seconds must be1..180')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    def timeout(*_):
        raise TimeoutError(f'closing TRAIN cache{args.seconds}second deadline')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(args.seconds)
    from pebby.agent.structured_field import load_structured_field_encoder
    world = Path('checkpoints/ls20-world-cell-recall-b1024.pt')
    visibility = Path('checkpoints/ls20-cell-visibility-initial-200.pt')
    code = [Path('pebby/agent')/name for name in ('structured_field.py','world_model.py','world_readout.py','world_grounding.py','world_rollout.py','cell_appearance.py','cell_appearance_dense.py','glyph_model.py','cell_visibility.py')]
    encoder_hashes = {str(path):digest(path) for path in code}
    checkpoints = {str(path):digest(path) for path in (world,visibility)}
    guards = {**encoder_hashes, **checkpoints, **{str(path):digest(path) for path in
        (args.source,args.index,args.attestation,Path(__file__),Path('pebby/agent/world_closing_sequences.py'),Path('pebby/agent/world_sequences.py'))}}
    initial_data = initial_provenance = None
    if args.initial_source is not None:
        guards[str(args.initial_source)] = digest(args.initial_source)
        guards['tools/build_structured_initial_cache.py'] = digest('tools/build_structured_initial_cache.py')
        with np.load(args.initial_source,allow_pickle=False) as archive:
            meta = json.loads(str(archive['meta'].item()))
            if meta.get('source')!='generated_only' or meta.get('split')!='train' or meta.get('oracle_search')!='complete_only':
                raise ValueError('initial candidate mapping requires complete generated TRAIN source')
            initial_data = {key:archive[key] for key in ('seeds','history_valid','previous_actions','current_lives')}
        initial_provenance = {'source_path':str(args.initial_source.resolve()),
            'source_sha256':guards[str(args.initial_source)],
            'array':'source_initial_rows',
            'meaning':'Unique public initial-history candidate row in combined TRAIN, not aggregate branch source rows.',
            'checks':'Same seed, lives3, only last H8 slot valid, all producing actions-1.',
            'limitations':'No initial pixels or fields cached; separate initial-memory builder must validate public frames.'}
    assembler = load_structured_field_encoder(world,visibility,device=args.device)
    if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
    provenance = {**assembler.metadata(), 'code_hashes':encoder_hashes, 'checkpoint_hashes':checkpoints}
    data = load_dataset(args.source, history=8, cache_dir=Path('data/world-array-cache'))
    index = closing.load_sidecar(args.index,args.source,data,attestation=args.attestation,
                                 attestation_sha256=args.attestation_sha256)
    manifest = write_cache(data,index,args.out,assembler,provenance,guards,args.levels,args.seed,args.batch_size,
                           args.device,initial_data,initial_provenance)
    report = dict(status='complete',pid=os.getpid(),elapsed_seconds=time.monotonic()-start,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        device=args.device,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else 0,
        manifest_sha256=digest(args.out/'manifest.json'),event_coverage=manifest['event_coverage'],
        available_index_event_coverage=manifest['available_index_event_coverage'],
        current_zero_optimal=manifest['current_zero_optimal'],source_current_zero_optimal=manifest['source_current_zero_optimal'])
    (args.out/'build-report.json').write_text(json.dumps(report,indent=2)+'\n')
    signal.alarm(0); print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
