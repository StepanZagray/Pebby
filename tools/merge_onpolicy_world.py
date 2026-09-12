"""Stream raw single-behavior generated training banks into one supplemental NPZ.

Each source remains tied to its own behavior checkpoint and exact global policy
row indices. Nested aggregates are rejected; pass all original round banks.
"""
from pebby.ls20.provenance import generated_context, row_contexts

import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from tools import merge_world_data as stream
from pebby.agent.on_policy_provenance import validate_on_policy_provenance

MergeError = stream.MergeError
REQUIRED = {'frames','history_valid','previous_actions','next_frames','terminal','won',
            'optimal','distances','seeds','next_optimal'}


def inspect_source(path):
    path=Path(path).resolve();sha=stream._sha256(path)
    with np.load(path,allow_pickle=False) as archive:
        meta=stream._json_meta(archive,path);stream._validate_provenance(meta,path,'train')
        if any(key in meta for key in ('behavior_checkpoints','on_policy_sources')) or meta.get('on_policy_aggregation'):
            raise MergeError('nested aggregated inputs are not supported; pass original raw round banks')
        behavior=meta.get('behavior_checkpoint')
        if not isinstance(behavior,dict) or not isinstance(behavior.get('config'),dict) or type(behavior.get('parameters')) is not int or behavior['parameters']<=0:
            raise MergeError('source lacks behavior checkpoint configuration/parameter provenance')
        names,schema,seeds=stream._schema(archive,path)
        try:validate_on_policy_provenance({'meta':meta,'seeds':seeds})
        except ValueError as error:raise MergeError(str(error)) from error
        if not REQUIRED<=set(names):raise MergeError('missing required transition/successor arrays')
        if not len(seeds) or seeds.ndim!=1 or np.any((seeds<0)|(seeds>=1_000_000)):
            raise MergeError('source requires nonempty generated training seeds, never validation')
        history=meta.get('history')
        if type(history) is not int or history<1 or behavior['config'].get('history')!=history:
            raise MergeError('behavior/data history mismatch')
        expected={'frames':(history,64,64),'next_frames':(4,64,64),'history_valid':(history,),
                  'previous_actions':(history,),'terminal':(4,),'won':(4,),'optimal':(),
                  'distances':(4,),'seeds':(),'next_optimal':(4,)}
        for name,shape in expected.items():
            if schema[name][1]!=shape:raise MergeError(f'{name} has invalid shape')
        for name in ('frames','next_frames'):
            if np.dtype(schema[name][0])!=np.uint8:raise MergeError(f'{name} must be uint8')
        small={name:archive[name] for name in ('history_valid','previous_actions','terminal','won','optimal','next_optimal')}
        for name in ('history_valid','terminal','won'):
            if small[name].dtype!=np.bool_:raise MergeError(f'{name} must be boolean')
        if not small['history_valid'][:,-1].all():raise MergeError('current history slot must be valid')
        actions=small['previous_actions']
        if not np.issubdtype(actions.dtype,np.integer) or np.any((actions < -1)|(actions>3)) or np.any(actions[~small['history_valid']]!=-1):
            raise MergeError('invalid public history actions')
        for name,minimum in (('optimal',0),('next_optimal',0)):
            values=small[name]
            if not np.issubdtype(values.dtype,np.integer) or np.any((values<minimum)|(values>15)):
                raise MergeError(f'invalid {name} masks')
        if np.any(small['optimal'] == 0):
            unreachable = archive['distances'] < 0
            if 'lost_life' in archive.files:
                lost = archive['lost_life']
                if lost.shape != unreachable.shape or lost.dtype != np.bool_:
                    raise MergeError('invalid lost_life labels')
                unreachable |= lost
            if np.any((small['optimal'] == 0) & ~unreachable.all(axis=1)):
                raise MergeError('empty optimal masks need unreachable or lost-life successors')
        if np.any(small['won']&~small['terminal']) or np.any(small['terminal']&(small['next_optimal']!=0)):
            raise MergeError('terminal/winning successor contradiction')
        unique=set(map(int,seeds))
        if set(map(int,seeds[small['won'].any(1)]))!=unique:raise MergeError('source lacks winning coverage for every seed')
        levels=stream._source_levels(meta,path)
        if set(levels)!=unique:raise MergeError('source proof level set differs from row seeds')
        for seed,level in levels.items():
            if level.get('context_engine_verified') is not True or level.get('search_truncated',False) or level.get('context_index')!=generated_context(level) or 'excluded' in level:
                raise MergeError('source lacks complete contextual engine proof')
        if 'context_index' in names and not np.array_equal(archive['context_index'],row_contexts(seeds, levels.values())):
            raise MergeError('row context disagrees with seed')
        marked=meta.get('on_policy_rows')
        if not isinstance(marked,list) or not marked or any(type(index) is not int for index in marked) or marked!=sorted(set(marked)) or marked[0]<0 or marked[-1]>=len(seeds):
            raise MergeError('invalid on_policy_rows indices')
        for seed,level in levels.items():
            if level.get('samples')!=int(np.sum(seeds==seed)) or level.get('on_policy_samples')!=int(np.sum(seeds[marked]==seed)) or level.get('on_policy_samples',0)<1:
                raise MergeError('per-level policy row provenance mismatch')
    return {'path':path,'sha256':sha,'meta':meta,'names':names,'schema':schema,'seeds':seeds,
            'rows':len(seeds),'win_rows':int(small['won'].any(1).sum())}


def plan_sources(inputs):
    paths=[Path(path).resolve() for path in inputs]
    if len(paths)<2 or len(set(paths))!=len(paths):raise MergeError('need at least two distinct raw source paths')
    infos=[];seen=set();offset=0;marked=[];sources=[];behaviors=[];auxiliary=[]
    for path in paths:
        info=inspect_source(path)
        if infos and (info['names']!=infos[0]['names'] or info['schema']!=infos[0]['schema'] or info['meta']['history']!=infos[0]['meta']['history']):
            raise MergeError('source array schema/history mismatch')
        unique=set(map(int,info['seeds']))
        if unique&seen:raise MergeError('duplicate seeds across input sources')
        seen.update(unique)
        policy_rows=[offset+index for index in info['meta']['on_policy_rows']]
        auxiliary_rows=[offset+index for index in info['meta'].get('auxiliary_rows', [])]
        behavior=copy.deepcopy(info['meta']['behavior_checkpoint'])
        identity=(Path(behavior['path']).resolve(),behavior['sha256'])
        matches=[i for i,b in enumerate(behaviors) if (Path(b['path']).resolve(),b['sha256'])==identity]
        if matches:
            behavior_index=matches[0]
            if behaviors[behavior_index]!=behavior:raise MergeError('inconsistent metadata for the same behavior checkpoint')
        else:
            behavior_index=len(behaviors);behaviors.append(behavior)
        sources.append({'path':str(path),'sha256':info['sha256'],'row_start':offset,'row_stop':offset+info['rows'],
                        'behavior_index':behavior_index,'on_policy_rows':policy_rows,
                        'auxiliary_rows':auxiliary_rows,
                        'source_on_policy_provenance':copy.deepcopy(info['meta']['on_policy_provenance'])})
        marked.extend(policy_rows);auxiliary.extend(auxiliary_rows);offset+=info['rows'];infos.append(info)
    meta={'format':stream.FORMAT,'source':'generated_only','oracle_search':'complete_only','split':'train',
          'history':infos[0]['meta']['history'],'samples':offset,'seeds':sorted(seen),
          'alternatives_per_state':4,'accepted_levels':len(seen),'win_covered_levels':len(seen),
          'win_rows':sum(info['win_rows'] for info in infos),'coverage':'on_policy_with_expert_anchors',
          'collection_policy':'model_greedy','on_policy_aggregation':True,'behavior_checkpoints':behaviors,
          'on_policy_sources':sources,'on_policy_rows':marked,'auxiliary_rows':auxiliary,
          'on_policy_provenance':{'official_inputs_used':False,'oracle_actions_in_policy_rollout':0},
          'levels':[copy.deepcopy(level) for info in infos for level in info['meta']['levels']]}
    for name in ('state_supervision_version','successor_policy_supervision_version'):
        values=[info['meta'].get(name) for info in infos]
        if any(value!=values[0] for value in values):raise MergeError(f'{name} mismatch')
        if values[0] is not None:meta[name]=values[0]
    return infos,meta


def merge(inputs,out):
    out=Path(out)
    if out.exists():raise MergeError('refusing output overwrite')
    out.parent.mkdir(parents=True,exist_ok=True)
    infos,meta=plan_sources(inputs)
    with tempfile.TemporaryDirectory(prefix='.merge-onpolicy-',dir=out.parent) as temporary:
        temporary=Path(temporary);staged=temporary/'result.npz'
        with zipfile.ZipFile(staged,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as archive:
            for name in infos[0]['names']:
                dtype,shape=infos[0]['schema'][name];column_path=temporary/'column.npy'
                column=np.lib.format.open_memmap(column_path,mode='w+',dtype=np.dtype(dtype),shape=(meta['samples'],*shape))
                offset=0
                for info in infos:
                    stream._copy_selected_rows(info['path'],name,column,offset,np.arange(info['rows']),info['schema'][name],info['rows'])
                    offset+=info['rows']
                column.flush()
                with archive.open(name+'.npy','w',force_zip64=True) as handle:
                    np.lib.format.write_array(handle,column,allow_pickle=False)
                del column;column_path.unlink()
            with archive.open('meta.npy','w',force_zip64=True) as handle:
                np.lib.format.write_array(handle,np.array(json.dumps(meta)),allow_pickle=False)
        # Re-open only small arrays: exact order and marked-row source membership.
        with np.load(staged,allow_pickle=False) as archive:
            seeds=archive['seeds']
            if not np.array_equal(seeds,np.concatenate([info['seeds'] for info in infos])):
                raise MergeError('published row order mismatch')
            if stream._json_meta(archive,staged)!=meta:raise MergeError('published metadata mismatch')
            if set(map(int,seeds[archive['won'].any(1)]))!=set(meta['seeds']):raise MergeError('published winning coverage mismatch')
        for info in infos:
            if stream._sha256(info['path'])!=info['sha256']:raise MergeError('input changed while merging')
        try:validate_on_policy_provenance({'meta':meta,'seeds':seeds})
        except ValueError as error:raise MergeError(str(error)) from error
        # Atomic no-clobber publish on the same filesystem (unlike replace).
        try:os.link(staged,out)
        except FileExistsError as error:raise MergeError('refusing output overwrite') from error
    return meta


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,action='append',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv);print('PID',os.getpid(),flush=True)
    meta=merge(args.input,args.out)
    print(json.dumps({'output':str(args.out),'sha256':stream._sha256(args.out),'rows':meta['samples'],
                      'levels':len(meta['seeds']),'on_policy_rows':len(meta['on_policy_rows']),
                      'behavior_checkpoints':len(meta['behavior_checkpoints'])}),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
