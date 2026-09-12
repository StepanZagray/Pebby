"""Verified live exploratory TRAIN clips, separate from policy and validation indices."""
from collections import Counter
from ..ls20.provenance import generated_context, difficulty_provenance

import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .on_policy_provenance import file_digest
from .world_sequences import FourStepIndex, LABELS, _link, exact_current_distances

FORMAT = 'pebby.ls20-exploratory-train-four-step-index.v1'
MODE = 'explore_train'


def prefixes(data):
    meta = data['meta']
    seeds = np.asarray(data['seeds'])
    if (meta.get('source') != 'generated_only' or meta.get('oracle_search') != 'complete_only'
            or meta.get('split') not in (None, 'train')
            or meta.get('on_policy_rows') is not None):
        raise ValueError('requires ordinary generated complete-teacher TRAIN source')
    if seeds.ndim != 1 or not np.issubdtype(seeds.dtype, np.integer) or not len(seeds):
        raise ValueError('invalid seeds')
    if np.any((seeds < 0) | (seeds >= 1_000_000)):
        raise ValueError('TRAIN seed namespace required')
    boundaries = np.r_[0, np.flatnonzero(seeds[1:] != seeds[:-1])+1, len(seeds)]
    ranges = {}
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        seed = int(seeds[start])
        if seed in ranges: raise ValueError('noncontiguous level rows')
        ranges[seed] = (int(start), int(stop))
    seen = set()
    for level in meta['levels']:
        seed = int(level['seed'])
        if seed in seen or seed not in ranges: raise ValueError('duplicate/missing level proof')
        seen.add(seed)
        if (level.get('context_engine_verified') is not True
                or level.get('search_truncated', False)
                or level.get('context_index') != generated_context(level)):
            raise ValueError('complete contextual proof required')
        start, stop = ranges[seed]
        count = level.get('explore_samples')
        if type(count) is not int or not 0 <= count <= stop-start:
            raise ValueError('invalid exploratory prefix count')
        yield np.arange(start, start+count, dtype=np.int64)
    if seen != set(ranges): raise ValueError('row/proof seed sets differ')


def build_index_arrays(data):
    if data['frames'].shape[1:] != (8,64,64): raise ValueError('requires public H8')
    for key in ('next_frames','history_valid','previous_actions','lost_life','terminal','won',
                'optimal','distances',*LABELS,*LABELS.values()):
        if data.get(key) is None: raise ValueError(f'missing {key}')
    anchors = []; counts = Counter()
    for prefix in prefixes(data):
        valid = []
        for i,j in zip(prefix[:-1],prefix[1:]):
            good, reason = _link(data,int(i),int(j))
            if good:
                action = int(data['previous_actions'][j,-1])
                if not int(data['optimal'][j]):
                    good, reason = False, 'unreachable'
                else:
                    distance = int(exact_current_distances(data,np.array([j]))[0])
                    if int(data['distances'][i,action]) != distance:
                        raise ValueError('exact branch/current distance mismatch')
                    if data.get('next_optimal') is not None and int(data['next_optimal'][i,action]) != int(data['optimal'][j]):
                        raise ValueError('branch/current optimal mask mismatch')
            valid.append(good); counts[reason] += 1
        for offset in range(max(0,len(prefix)-4)):
            if all(valid[offset:offset+4]): anchors.append(int(prefix[offset]))
    anchors = np.asarray(sorted(anchors),dtype=np.int64)
    future = anchors[:,None]+np.arange(1,5,dtype=np.int64)
    exact_current_distances(data,future)
    return anchors,future,dict(counts)


def build_sidecar(data, source, out):
    source,out = Path(source),Path(out)
    if out.exists(): raise ValueError('refusing overwrite')
    before = file_digest(source)
    anchors,future,counts = build_index_arrays(data)
    if not len(anchors): raise ValueError('no eligible exploratory TRAIN clips')
    meta = dict(format=FORMAT,mode=MODE,split='train',source='generated_only',K=4,history=8,
        source_path=str(source.resolve()),source_sha256=before,source_rows=len(data['seeds']),
        anchors=len(anchors),eligible_levels=len(np.unique(data['seeds'][anchors])),link_checks=counts,
        target_slots='chronological t+1..t+4; no action-alternative interpretation',
        provenance='ordinary exploratory prefixes only; no policy/expert rows',
        terminal_reset_padding='none; four live complete reachable transitions')
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out.parent,suffix='.npz',delete=False) as f:
        temp = Path(f.name)
        np.savez_compressed(f,anchor_row=anchors,future_rows=future,meta=np.array(json.dumps(meta)))
        f.flush(); os.fsync(f.fileno())
    try:
        if file_digest(source) != before: raise ValueError('source hash changed')
        os.link(temp,out)
    finally: temp.unlink(missing_ok=True)
    return FourStepIndex(anchors,future,meta)


def load_sidecar(path,source,data):
    with np.load(path,allow_pickle=False) as z:
        if len(z.files)!=3 or set(z.files)!={'anchor_row','future_rows','meta'}:
            raise ValueError('invalid sidecar members')
        meta=json.loads(str(z['meta'].item())); a=z['anchor_row']; f=z['future_rows']
    if (meta.get('format')!=FORMAT or meta.get('mode')!=MODE or meta.get('split')!='train'
            or meta.get('source')!='generated_only' or meta.get('K')!=4 or meta.get('history')!=8):
        raise ValueError('exploratory TRAIN format/mode/split mismatch')
    if (Path(meta['source_path']).resolve()!=Path(source).resolve()
            or meta['source_sha256']!=file_digest(source) or meta['source_rows']!=len(data['seeds'])):
        raise ValueError('source binding mismatch')
    aa,ff,_=build_index_arrays(data)
    if a.dtype!=np.int64 or f.dtype!=np.int64 or not np.array_equal(a,aa) or not np.array_equal(f,ff):
        raise ValueError('indices differ from fully verified clips')
    if meta.get('anchors')!=len(a) or meta.get('eligible_levels')!=len(np.unique(data['seeds'][a])):
        raise ValueError('index counts mismatch')
    return FourStepIndex(a,f,meta)
