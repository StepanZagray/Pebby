"""Closing-only generated K4 clips with externally pinned batch-one action proofs.

Each target indexes an existing actual branch. Never reconstruct a future frame
from expert rows, rerender the engine, or use teacher labels to select actions.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .on_policy_provenance import file_digest, validate_on_policy_provenance
from .on_policy_sampling import OnPolicySampler, mixed_batch
from .world_sequences import _prefix_rows, _link

FORMAT = 'pebby.ls20-closing-four-step-index.v1'
ATTESTATION = 'pebby.ls20-closing-behavior-attestation.v1'
TARGETS = ('next_frames', 'distances', 'terminal', 'won', 'lost_life', 'next_optimal',
           'next_player_cell', 'next_triple', 'next_steps', 'next_lives')


def public_history_digest(data, row):
    return hashlib.sha256(b''.join(np.asarray(data[key][row]).tobytes()
        for key in ('frames', 'history_valid', 'previous_actions'))).hexdigest()


def _attested_actions(data, source, attestation, expected_sha256):
    # The expected checksum is a caller-owned trust anchor, not supplied by the
    # sidecar being validated. A changed sidecar+changed proof cannot approve itself.
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError('externally pinned action attestation checksum required')
    if file_digest(attestation) != expected_sha256:
        raise ValueError('action attestation checksum mismatch')
    proof = json.loads(Path(attestation).read_text())
    if (proof.get('format') != ATTESTATION or proof.get('source') != 'generated_only'
            or proof.get('split') != 'train' or proof.get('inference') != 'greedy_argmax_batch1'
            or proof.get('batch_size') != 1 or proof.get('official_inputs') is not False
            or proof.get('oracle_calls') != 0 or proof.get('engine_calls') != 0):
        raise ValueError('invalid original public batch-one action attestation')
    if (Path(proof['source_path']).resolve() != Path(source).resolve()
            or proof['source_sha256'] != file_digest(source)):
        raise ValueError('action attestation source binding mismatch')
    provenance = validate_on_policy_provenance(data)
    behaviors = provenance.get('behavior_checkpoints', [provenance.get('behavior_checkpoint')])
    if proof.get('behavior_checkpoints') != behaviors:
        raise ValueError('attestation behavior checkpoints do not match source')
    source_ranges = [{k:s[k] for k in ('path','sha256','row_start','row_stop','behavior_index')}
                     for s in data['meta'].get('on_policy_sources', [])]
    if proof.get('source_ranges', []) != source_ranges:
        raise ValueError('attestation behavior source ranges do not match')
    prefixes = _prefix_rows(data, 'on_policy_train')
    last = {int(rows[-1]):int(data['seeds'][rows[-1]]) for rows in prefixes if len(rows)}
    records = proof.get('records', [])
    if len(records) != len(last):raise ValueError('attestation must cover each final policy row exactly')
    actions = {}
    for record in records:
        row, action = record.get('row'), record.get('action')
        if (type(row) is not int or row not in last or row in actions
                or type(action) is not int or not 0 <= action < 4 or record.get('seed') != last[row]):
            raise ValueError('invalid attested final row/action/seed')
        bi = next((s['behavior_index'] for s in source_ranges if s['row_start'] <= row < s['row_stop']), 0)
        if record.get('behavior_index') != bi:
            raise ValueError('attested row uses wrong behavior checkpoint')
        if record.get('public_history_sha256') != public_history_digest(data, row):
            raise ValueError('attested public history bytes changed')
        actions[row] = action
    return actions, prefixes


@dataclass(frozen=True)
class ClosingIndex:
    anchor_row: np.ndarray
    branch_rows: np.ndarray
    branch_actions: np.ndarray
    meta: dict

    def lookup(self, rows):
        positions = np.searchsorted(self.anchor_row, np.asarray(rows, dtype=np.int64))
        if np.any(positions >= len(self.anchor_row)) or not np.array_equal(self.anchor_row[positions], rows):
            raise ValueError('requested row is not a closing-only anchor')
        return self.branch_rows[positions], self.branch_actions[positions]


def build_index_arrays(data, source, attestation, expected_sha256):
    if data['frames'].shape[1:] != (8,64,64):raise ValueError('closing sequences require public H8')
    for name in TARGETS:
        if name not in data:raise ValueError(f'closing source lacks {name}')
    actions, prefixes = _attested_actions(data, source, attestation, expected_sha256)
    levels = {int(level['seed']):level for level in data['meta']['levels']}
    anchors, branches, choices = [], [], []
    for prefix in prefixes:
        if len(prefix) < 4:continue
        rows = prefix[-4:]
        if not all(_link(data, int(i), int(j))[0] for i,j in zip(rows[:-1], rows[1:])):continue
        picked = [int(data['previous_actions'][j,-1]) for j in rows[1:]] + [actions[int(rows[-1])]]
        last, action = int(rows[-1]), picked[-1]
        stop = levels[int(data['seeds'][last])].get('stop')
        if stop == 'won' and not data['won'][last,action]:
            raise ValueError('attested closing action conflicts with recorded win stop')
        if stop == 'unreachable_deadend' and (data['terminal'][last,action] or data['distances'][last,action] >= 0):
            raise ValueError('attested closing action conflicts with recorded deadend stop')
        anchors.append(int(rows[0]));branches.append(rows);choices.append(picked)
    order = np.argsort(anchors)
    ar = np.asarray(anchors, dtype=np.int64)[order]
    br = np.asarray(branches, dtype=np.int64).reshape(-1,4)[order]
    ba = np.asarray(choices, dtype=np.int64).reshape(-1,4)[order]
    if len(np.unique(np.asarray(data['seeds'])[ar])) != len(ar):raise ValueError('closing index repeats a level')
    # First three transition labels match their actual next current rows through _link.
    # The closing label is copied verbatim, including terminal/deadend/reset outcomes.
    terminal = np.asarray(data['terminal'])[br,ba]
    won = np.asarray(data['won'])[br,ba]
    lost = np.asarray(data['lost_life'])[br,ba]
    distances = np.asarray(data['distances'])[br,ba]
    masks = np.asarray(data['next_optimal'])[br,ba]
    if np.any((terminal|won|lost)[:,:3]):raise ValueError('interior closing transition resets or terminates')
    if (np.any(won & ~terminal) or np.any(won & (distances != 0))
            or np.any((terminal & ~won) & (distances != -1))
            or np.any((terminal | (distances < 0)) & (masks != 0))):
        raise ValueError('closing branch terminal/distance/optimal labels conflict')
    return ar, br, ba


def build_sidecar(data, source, out, *, attestation, attestation_sha256):
    source, out, attestation = Path(source), Path(out), Path(attestation)
    if out.exists():raise ValueError('refusing closing sidecar overwrite')
    before = file_digest(source)
    ar, br, ba = build_index_arrays(data, source, attestation, attestation_sha256)
    if not len(ar):raise ValueError('no eligible closing-only anchors')
    meta = {'format':FORMAT,'source':'generated_only','split':'train','mode':'closing_only_train',
            'K':4,'history':8,'source_path':str(source.resolve()),'source_sha256':before,
            'source_rows':len(data['seeds']),'anchors':len(ar),'eligible_levels':len(ar),
            'attestation_path':str(attestation.resolve()),'attestation_sha256':attestation_sha256,
            'target_slots':'actual branches along chronology; only final slot may terminate/reset',
            'selection':'one closing anchor per eligible level; excludes ordinary interior K4 anchors'}
    if file_digest(source) != before:raise ValueError('closing source changed during build')
    import os,tempfile
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out.parent,prefix='.closing-',suffix='.npz',delete=False) as f:
        temporary=Path(f.name)
        np.savez_compressed(f,anchor_row=ar,branch_rows=br,branch_actions=ba,meta=np.array(json.dumps(meta)))
    try:os.link(temporary,out)
    finally:temporary.unlink(missing_ok=True)
    return ClosingIndex(ar,br,ba,meta)


def load_sidecar(path, source, data, *, attestation, attestation_sha256):
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files) != {'anchor_row','branch_rows','branch_actions','meta'} or len(archive.files)!=4:
            raise ValueError('invalid closing sidecar members')
        meta=json.loads(str(archive['meta'].item()))
        actual=tuple(archive[k] for k in ('anchor_row','branch_rows','branch_actions'))
    if (meta.get('format')!=FORMAT or meta.get('source')!='generated_only' or meta.get('split')!='train'
            or meta.get('mode')!='closing_only_train' or meta.get('K')!=4 or meta.get('history')!=8):
        raise ValueError('closing sidecar format/split/mode mismatch')
    if (Path(meta['source_path']).resolve()!=Path(source).resolve() or meta['source_sha256']!=file_digest(source)
            or meta['source_rows']!=len(data['seeds']) or meta.get('attestation_sha256')!=attestation_sha256
            or Path(meta['attestation_path']).resolve()!=Path(attestation).resolve()):
        raise ValueError('closing sidecar source/attestation binding mismatch')
    expected=build_index_arrays(data,source,attestation,attestation_sha256)
    if any(a.dtype!=np.int64 or not np.array_equal(a,b) for a,b in zip(actual,expected)):
        raise ValueError('closing indices/actions differ from fully checked source and attestation')
    if meta.get('anchors')!=len(actual[0]) or meta.get('eligible_levels')!=len(actual[0]):
        raise ValueError('closing sidecar counts mismatch')
    return ClosingIndex(*actual,meta)


class ClosingSampler(OnPolicySampler):
    def __init__(self,base,supplemental,index,**kwargs):
        if index.meta.get('mode')!='closing_only_train' or index.meta.get('split')!='train':
            raise ValueError('only closing training index may enter ClosingSampler')
        if index.meta.get('source_rows')!=len(supplemental['seeds']):raise ValueError('closing source row count mismatch')
        if not np.array_equal(index.branch_rows,index.anchor_row[:,None]+np.arange(4)):
            raise ValueError('closing branch rows must be contiguous')
        marked=set(supplemental['meta']['on_policy_rows'])
        if not set(map(int,index.branch_rows.flat))<=marked:raise ValueError('closing branch crosses expert boundary')
        self.sequence_index=index
        view={**supplemental,'meta':{**supplemental['meta'],'on_policy_rows':index.anchor_row.tolist()}}
        super().__init__(base,view,fraction=kwargs.pop('fraction',.5),**kwargs)


def sequence_targets(supplemental,index,anchor_rows):
    rows,actions=index.lookup(torch.as_tensor(anchor_rows).cpu().numpy())
    rows=torch.from_numpy(np.array(rows,copy=True));actions=torch.from_numpy(np.array(actions,copy=True))
    return {**{key:supplemental[key][rows,actions] for key in TARGETS},'rollout_actions':actions}


def closing_mixed_batch(base,supplemental,indices,index,*,auxiliary_rows=()):
    if index.meta.get('mode')!='closing_only_train':raise ValueError('only closing training indices may form batches')
    bi,pi,order=indices;output=mixed_batch(base,supplemental,indices)
    selected = torch.isin(pi, torch.as_tensor(index.anchor_row.copy()))
    if not set(pi[~selected].tolist()) <= set(auxiliary_rows):
        raise ValueError('supplemental row is neither a sequence anchor nor an authorized auxiliary row')
    mask=torch.cat((torch.zeros(len(bi),dtype=torch.bool),selected))[order]
    targets=sequence_targets(supplemental,index,pi[selected])
    locations=torch.empty_like(order);locations[order]=torch.arange(len(order));positions=locations[len(bi):][selected]
    for key,value in targets.items():
        if key!='rollout_actions':output[key][positions]=value.to(output[key].dtype)
    output['rollout_mask']=mask
    output['rollout_actions']=torch.full((len(order),4),-1,dtype=torch.long)
    output['rollout_actions'][positions]=targets['rollout_actions']
    return output
