"""Source-bound four-step recorded histories; explicit chronology, never action slots.

Training uses only true policy prefixes. Validation uses separate exploratory
prefixes and cannot enter FourStepSampler. No engine, rendering or teacher search.
"""
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from .on_policy_provenance import file_digest,validate_on_policy_provenance
from .on_policy_sampling import OnPolicySampler,mixed_batch

FORMAT='pebby.ls20-four-step-index.v1'
MODES={'on_policy_train':'train','explore_validation':'validation'}
K=4
LABELS={'next_player_cell':'player_cell','next_triple':'current_triple','next_steps':'current_steps','next_lives':'current_lives'}


@dataclass(frozen=True)
class FourStepIndex:
    anchor_row: np.ndarray
    future_rows: np.ndarray
    meta: dict

    def lookup(self,rows):
        rows=np.asarray(rows,dtype=np.int64)
        positions=np.searchsorted(self.anchor_row,rows)
        if np.any(positions>=len(self.anchor_row)) or not np.array_equal(self.anchor_row[positions],rows):
            raise ValueError('requested row is not an eligible four-step anchor')
        return self.future_rows[positions]


def exact_current_distances(data,rows):
    rows=np.asarray(rows,dtype=np.int64)
    masks=np.asarray(data['optimal'])[rows].astype(np.int64)
    distances=np.asarray(data['distances'])[rows]
    bits=(masks[...,None]&(1<<np.arange(4)))!=0
    if np.any((masks<1)|(masks>15)) or not np.issubdtype(distances.dtype,np.integer):
        raise ValueError('invalid exact-distance teacher labels')
    distances=distances.astype(np.int64,copy=False)
    selected=np.where(bits,distances,np.iinfo(np.int64).max).min(-1)
    bad=bits&((distances!=selected[...,None])|np.asarray(data['lost_life'])[rows]|
              (np.asarray(data['terminal'])[rows]&~np.asarray(data['won'])[rows]))
    if np.any(selected<0) or bad.any():
        raise ValueError('optimal branch distances do not define one live-state exact distance')
    return selected+1


def _prefix_rows(data,mode):
    if mode not in MODES:raise ValueError('unknown sequence index mode')
    seeds=np.asarray(data['seeds']);meta=data['meta']
    if meta.get('source')!='generated_only' or meta.get('oracle_search')!='complete_only':
        raise ValueError('sequences require generated-only complete-teacher provenance')
    if meta.get('split') not in (None,MODES[mode]):raise ValueError('sequence source split mismatch')
    if seeds.ndim!=1 or not np.issubdtype(seeds.dtype,np.integer):raise ValueError('invalid sequence seeds')
    lo,hi=(0,1_000_000) if mode=='on_policy_train' else (1_000_000,2_000_000)
    if np.any((seeds<lo)|(seeds>=hi)):raise ValueError('sequence mode and seed split disagree')
    if mode=='on_policy_train':
        marked=meta.get('on_policy_rows')
        if not isinstance(marked,list) or any(type(i) is not int for i in marked) or marked!=sorted(set(marked)) or not marked or marked[0]<0 or marked[-1]>=len(seeds):
            raise ValueError('invalid true-policy marked indices')
        marked=np.asarray(marked,dtype=np.int64)
    levels=meta['levels'];seen=set();prefixes=[]
    for level in levels:
        seed=int(level['seed'])
        if seed in seen:raise ValueError('duplicate sequence level proof')
        seen.add(seed)
        if level.get('context_engine_verified') is not True or level.get('search_truncated',False) or level.get('context_index')!=seed%7:
            raise ValueError('sequence level lacks complete contextual proof')
        rows=np.flatnonzero(seeds==seed)
        if not len(rows) or np.any(np.diff(rows)!=1):raise ValueError('level rows are not contiguous')
        count=level.get('on_policy_samples' if mode=='on_policy_train' else 'explore_samples')
        if type(count) is not int or not 0<=count<=len(rows):raise ValueError('missing or invalid chronological prefix length')
        prefix=rows[:count]
        if mode=='on_policy_train' and not np.array_equal(prefix,marked[seeds[marked]==seed]):
            raise ValueError('marked rows are not the exact policy prefix')
        prefixes.append(prefix)
    if seen!=set(map(int,seeds)):raise ValueError('sequence row/proof seed sets differ')
    # Aggregated source ranges must not bisect a level/prefix.
    for source in meta.get('on_policy_sources',[]):
        for prefix in prefixes:
            if len(prefix) and source['row_start']<=prefix[0]<source['row_stop'] and prefix[-1]>=source['row_stop']:
                raise ValueError('sequence crosses a source boundary')
    return prefixes


def _link(data,i,j):
    action=int(data['previous_actions'][j,-1])
    if action<0:return False,'reset_or_missing_action'
    if action>3:raise ValueError('invalid recorded producing action')
    if data['lost_life'][i,action] or data['terminal'][i,action] or data['won'][i,action]:
        return False,'reset_or_terminal'
    a,b=data['frames'][i],data['frames'][j]
    if not np.array_equal(b[-1],data['next_frames'][i,action]) or not np.array_equal(b[:-1],a[1:]):
        raise ValueError('sequence full history/image continuity mismatch')
    if not np.array_equal(data['history_valid'][j],np.r_[data['history_valid'][i,1:],True]):
        raise ValueError('sequence history validity mismatch')
    if not np.array_equal(data['previous_actions'][j],np.r_[data['previous_actions'][i,1:],action]):
        raise ValueError('sequence history action mismatch')
    for target,current in LABELS.items():
        if not np.array_equal(data[target][i,action],data[current][j]):raise ValueError('sequence teacher label continuity mismatch')
    if int(data['current_lives'][i])!=int(data['current_lives'][j]):raise ValueError('unmarked life reset')
    return True,'valid'


def build_index_arrays(data,mode='on_policy_train'):
    if data['frames'].shape[1:]!=(8,64,64):raise ValueError('four-step path currently requires public H8')
    for name in ('frames','next_frames','history_valid','previous_actions','lost_life','terminal','won','optimal','distances',*LABELS,*LABELS.values()):
        if data.get(name) is None:raise ValueError(f'sequence input lacks {name}')
    anchors=[];counts=Counter()
    for prefix in _prefix_rows(data,mode):
        valid=[]
        for i,j in zip(prefix[:-1],prefix[1:]):
            good,reason=_link(data,int(i),int(j));valid.append(good);counts[reason]+=1
        for offset in range(max(0,len(prefix)-K)):
            if all(valid[offset:offset+K]):
                # Current-distance reconstruction needs a surviving optimal
                # action. Dead-end rows remain available as ordinary dynamics
                # examples but cannot supply this chronological value target.
                if np.any(np.asarray(data['optimal'])[prefix[offset + 1:offset + K + 1]] == 0):
                    counts['unreachable_future'] += 1
                    continue
                anchors.append(int(prefix[offset]))
    anchors=np.asarray(anchors,dtype=np.int64);anchors.sort()
    future=anchors[:,None]+np.arange(1,K+1,dtype=np.int64)
    exact_current_distances(data,future)
    return anchors,future,dict(counts)


def build_sidecar(data,source,out,mode='on_policy_train'):
    source=Path(source);out=Path(out)
    if out.exists():raise ValueError('refusing sequence sidecar overwrite')
    before=file_digest(source)
    if mode=='on_policy_train':validate_on_policy_provenance(data)
    anchors,future,checks=build_index_arrays(data,mode)
    if not len(anchors):raise ValueError('no eligible four-step anchors')
    meta={'format':FORMAT,'source':'generated_only','split':MODES[mode],'mode':mode,'K':K,'history':8,
          'source_path':str(source.resolve()),'source_sha256':before,'source_rows':len(data['seeds']),
          'anchors':len(anchors),'eligible_levels':len(np.unique(data['seeds'][anchors])),
          'link_checks':checks,'target_slots':'chronological t+1,t+2,t+3,t+4; not action alternatives',
          'terminal_reset_padding':'none; all four transitions must be live and contiguous'}
    if file_digest(source)!=before:raise ValueError('sequence source changed')
    out.parent.mkdir(parents=True,exist_ok=True)
    import tempfile,os
    with tempfile.NamedTemporaryFile(dir=out.parent,prefix='.sequence-',suffix='.npz',delete=False) as stream:
        temporary=Path(stream.name)
        np.savez_compressed(stream,anchor_row=anchors,future_rows=future,meta=np.array(json.dumps(meta)))
    try:
        if file_digest(source)!=before:raise ValueError('sequence source changed before publication')
        os.link(temporary,out)
    finally:temporary.unlink(missing_ok=True)
    return FourStepIndex(anchors,future,meta)


def load_sidecar(path,source,data,*,mode='on_policy_train'):
    with np.load(path,allow_pickle=False) as archive:
        if len(set(archive.files))!=len(archive.files) or set(archive.files)!={'anchor_row','future_rows','meta'}:
            raise ValueError('invalid sequence sidecar members')
        meta=json.loads(str(archive['meta'].item()));anchors=archive['anchor_row'];future=archive['future_rows']
    if meta.get('format')!=FORMAT or meta.get('mode')!=mode or meta.get('split')!=MODES.get(mode) or meta.get('source')!='generated_only' or meta.get('K')!=4 or meta.get('history')!=8:
        raise ValueError('sequence sidecar format/split/mode mismatch')
    if Path(meta['source_path']).resolve()!=Path(source).resolve() or meta['source_sha256']!=file_digest(source) or meta['source_rows']!=len(data['seeds']):
        raise ValueError('sequence sidecar source hash/path/row binding mismatch')
    if mode=='on_policy_train':validate_on_policy_provenance(data)
    expected_anchors,expected_future,_=build_index_arrays(data,mode)
    if anchors.dtype!=np.int64 or future.dtype!=np.int64 or not np.array_equal(anchors,expected_anchors) or not np.array_equal(future,expected_future):
        raise ValueError('sequence sidecar indices do not match fully checked eligible paths')
    if meta.get('anchors')!=len(anchors) or meta.get('eligible_levels')!=len(np.unique(data['seeds'][anchors])):
        raise ValueError('sequence sidecar counts mismatch')
    return FourStepIndex(anchors,future,meta)


class FourStepSampler(OnPolicySampler):
    def __init__(self,base,supplemental,index,**kwargs):
        if index.meta.get('mode')!='on_policy_train' or index.meta.get('split')!='train':
            raise ValueError('validation sequence index cannot enter training sampler')
        if index.meta.get('source_rows')!=len(supplemental['seeds']):raise ValueError('sequence row-count mismatch')
        if index.anchor_row.ndim!=1 or index.future_rows.shape!=(len(index.anchor_row),4):raise ValueError('invalid sequence index shape')
        if not np.array_equal(index.future_rows,index.anchor_row[:,None]+np.arange(1,5)):
            raise ValueError('noncontiguous sequence index')
        original=set(supplemental['meta']['on_policy_rows'])
        if not set(map(int,index.anchor_row))<=original or not set(map(int,index.future_rows.flat))<=original:
            raise ValueError('sequence anchor/future includes expert rows')
        self.sequence_index=index
        # Private shallow view: original aggregate/provenance is never mutated.
        view={**supplemental,'meta':{**supplemental['meta'],'on_policy_rows':index.anchor_row.tolist()}}
        super().__init__(base,view,fraction=kwargs.pop('fraction',.5),**kwargs)


def sequence_targets(supplemental,index,anchor_rows):
    """Chronological target-only tensor mapping, usable for held-out evaluation too."""
    future=torch.from_numpy(np.array(index.lookup(torch.as_tensor(anchor_rows).cpu().numpy()),copy=True))
    actions=supplemental['previous_actions'][future,-1].long()
    if bool(((actions<0)|(actions>3)).any()):raise ValueError('sequence includes reset/missing producing action')
    masks=supplemental['optimal'][future].long();distances=supplemental['distances'][future].long()
    bits=(masks[...,None]&(1<<torch.arange(4)))!=0
    exact=distances.masked_fill(~bits,torch.iinfo(torch.long).max).min(-1).values
    bad=bits&((distances!=exact[...,None])|supplemental['lost_life'][future]|
              (supplemental['terminal'][future]&~supplemental['won'][future]))
    if bool(((masks<1)|(masks>15)).any()) or bool((exact<0).any()) or bool(bad.any()):
        raise ValueError('future exact distance labels inconsistent')
    result={'next_frames':supplemental['frames'][future,-1],'next_optimal':supplemental['optimal'][future],
            'distances':exact+1,'rollout_actions':actions}
    for target,current in LABELS.items():result[target]=supplemental[current][future]
    for key in ('terminal','won','lost_life'):result[key]=torch.zeros_like(supplemental[key][anchor_rows],dtype=torch.bool)
    return result


def four_step_mixed_batch(base,supplemental,indices,index,*,auxiliary_rows=()):
    if index.meta.get('mode')!='on_policy_train':raise ValueError('only training sequence indices may form training batches')
    bi,pi,order=indices
    output=mixed_batch(base,supplemental,indices)
    selected = torch.isin(pi, torch.as_tensor(index.anchor_row.copy()))
    if not set(pi[~selected].tolist()) <= set(auxiliary_rows):
        raise ValueError('supplemental row is neither a sequence anchor nor an authorized auxiliary row')
    mask=torch.cat((torch.zeros(len(bi),dtype=torch.bool),selected))[order]
    targets=sequence_targets(supplemental,index,pi[selected])
    # sequence_targets follows pi order; order maps those rows to final positions.
    locations=torch.empty_like(order);locations[order]=torch.arange(len(order));positions=locations[len(bi):][selected]
    for key,value in targets.items():
        if key=='rollout_actions':continue
        if key not in output:raise ValueError(f'mixed sequence batch lacks {key}')
        output[key][positions]=value.to(output[key].dtype)
    output['rollout_mask']=mask
    output['rollout_actions']=torch.full((len(order),4),-1,dtype=torch.long)
    output['rollout_actions'][positions]=targets['rollout_actions']
    return output
