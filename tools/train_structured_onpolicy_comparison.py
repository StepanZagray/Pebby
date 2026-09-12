"""Matched evolving-head continuation: replay versus same-level visited states.

No architecture/objective change. Four actual/imagined action alternatives remain
supervised training inputs; no playable inference interface is changed here.
"""
from pebby.ls20.provenance import require_legacy_experiment

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from pebby.agent.structured_distance import current_distance_labels
from pebby.agent.structured_transition import steps_targets
from pebby.agent.structured_factored_policy import canonical_metadata, state_digest
from pebby.agent.structured_workspace_policy import StructuredWorkspaceReadout
from pebby.agent.world_train import require_verified_data
from tools.preflight_structured_workspace import load_inputs, backward
from tools.structured_onpolicy_sampling import PairedStateSampler
from tools.train_structured_policy import check_policy_encoder
from tools.train_structured_workspace_comparison import load_validation, evaluate, save_head, load_head
from tools.train_structured_transition import digest, atomic_json

CACHE_FORMAT = 'pebby.structured-onpolicy-field-cache.v2'
ACTOR = 'checkpoints/ls20-structured-policy-paired-local-h4-600.pt'
WARMSTART = 'checkpoints/ls20-structured-workspace-comparison-600-evolving.pt'
DEPTHS = (1, 2, 4)


def merge_sources(target, extra):
    canonical = {str(Path(k).resolve()): v for k,v in target.items()}
    for path, sha in extra.items():
        resolved = str(Path(path).resolve())
        if resolved in canonical and canonical[resolved] != sha: raise ValueError(f'conflicting source binding: {path}')
        target[path] = sha; canonical[resolved] = sha


def verify_sources(sources):
    if not isinstance(sources, dict) or not sources: raise ValueError('nonempty source guards required')
    for path, sha in sources.items():
        if digest(path) != sha: raise ValueError(f'source changed: {path}')


def validate_rows(data, manifest, base):
    """Validate the independent grouping/labels/proofs before sampling any row."""
    require_legacy_experiment(manifest)
    n = manifest['rows']; levels = manifest['levels']
    if type(n) is not int or n < 1 or type(levels) is not int or levels < 1: raise ValueError('positive row/level counts required')
    shapes = {name: (n,) for name in ('seeds','difficulties','source_rows','on_policy','current_distance','steps','lives','optimal')}
    shapes.update(fields=(n,148,96),next_fields=(n,4,148,96),imagined_fields=(n,4,148,96),
        player_cell=(n,2),next_player_cell=(n,4,2),triple=(n,3),next_triple=(n,4,3),
        level_seeds=(levels,),level_offsets=(levels+1,),level_rows=(n,))
    for key in ('next_steps','next_lives','next_optimal','distances','terminal','won','lost_life'): shapes[key]=(n,4)
    if 'context_index' in data: shapes['context_index']=(n,)
    if set(data) != set(shapes): raise ValueError('unexpected/missing cache array inventory')
    booleans = {'on_policy','terminal','won','lost_life'}
    for name, shape in shapes.items():
        array=data[name]
        if array.shape != shape: raise ValueError(f'bad cache shape: {name}')
        expected = np.float32 if name=='imagined_fields' else np.float16 if name in ('fields','next_fields') else np.bool_ if name in booleans else None
        if expected is not None:
            if array.dtype != expected: raise ValueError(f'bad cache dtype: {name}')
        elif not np.issubdtype(array.dtype,np.integer): raise ValueError(f'integer cache array required: {name}')
    seeds=data['seeds']; flags=data['on_policy']
    if np.any((seeds<0)|(seeds>=1_000_000)): raise ValueError('TRAIN seed namespace required')
    unique, counts=np.unique(seeds,return_counts=True)
    if (not np.array_equal(data['level_seeds'],unique) or len(unique)!=levels
            or not np.array_equal(data['level_offsets'],np.r_[0,np.cumsum(counts)])
            or not np.array_equal(data['level_rows'],np.argsort(seeds,kind='stable'))
            or not np.array_equal(data['source_rows'],np.arange(n))):
        raise ValueError('invalid stable per-level row grouping/source order')
    if not flags.any() or manifest['on_policy_rows'] != int(flags.sum()) or manifest['expert_rows'] + manifest.get('failure_rows',0) != int((~flags).sum()):
        raise ValueError('on-policy/expert census mismatch')
    lookup={int(s):int(d) for s,d in zip(base['seeds'],base['difficulties'])}
    if any(int(s) not in lookup for s in unique): raise ValueError('trajectory levels absent from legacy TRAIN bank')
    expected=np.array([lookup[int(s)] for s in seeds])
    if not np.array_equal(data['difficulties'],expected): raise ValueError('difficulty mismatch with base level')
    distance=current_distance_labels(*(data[k] for k in ('optimal','distances','terminal','won','lost_life')), allow_unreachable=True).numpy()
    if not np.array_equal(distance,data['current_distance']): raise ValueError('current distance mismatch')
    for key in ('player_cell','next_player_cell'):
        if np.any((data[key]<0)|(data[key]>=12)): raise ValueError('invalid player cell')
    for key in ('triple','next_triple'):
        if np.any((data[key]<0)|(data[key]>=np.array([6,4,4]))): raise ValueError('invalid glyph triple')
    for key in ('steps','next_steps'): steps_targets(torch.as_tensor(np.array(data[key])))
    lives,nl,loss,t,w=(data[k] for k in ('lives','next_lives','lost_life','terminal','won'))
    if (np.any((lives<1)|(lives>3)) or np.any((nl<0)|(nl>3)) or np.any(nl!=lives[:,None]-loss.astype(np.int16))
            or np.any((t&~w)!=(loss&(nl==0))) or np.any(w&loss)):
        raise ValueError('inconsistent life/terminal events')
    masks=data['next_optimal']
    if np.any((masks<0)|(masks>15)) or np.any(((data['distances']>0)&~t)!=(masks!=0)):
        raise ValueError('next optimal/reachability mismatch')
    meta=manifest['source_metadata']
    if (meta.get('source')!='generated_only' or meta.get('history')!=8 or meta.get('alternatives_per_state')!=4
            or meta.get('collection_policy')!='model_greedy' or meta.get('official_inputs_used',False)
            or meta.get('split','train')!='train'):
        raise ValueError('public generated TRAIN collection proof required')
    require_verified_data({'meta':meta,'seeds':seeds})
    marked=meta.get('on_policy_rows')
    if (not isinstance(marked,list) or any(type(x) is not int for x in marked)
            or not np.array_equal(np.sort(marked),np.flatnonzero(flags))):
        raise ValueError('marked source on-policy rows mismatch')
    proofs=meta.get('levels',[])
    if len(proofs)!=levels or len({p['seed'] for p in proofs})!=levels: raise ValueError('level proof count mismatch')
    offset=0
    for proof in proofs:
        start=proof.get('row_start');count=proof.get('row_count');policy_count=proof.get('on_policy_row_count')
        if (type(start) is not int or type(count) is not int or type(policy_count) is not int
                or start!=offset or not 1<=policy_count<=count or start+count>n
                or not np.all(seeds[start:start+count]==proof['seed'])
                or not flags[start:start+policy_count].all() or flags[start+policy_count:start+count].any()
                or proof.get('on_policy_samples')!=policy_count or proof.get('samples')!=count
                or proof.get('branch_checks',{}).get('branches')!=4*count
                or proof.get('branch_checks',{}).get('expansions')!=count
                or proof.get('win_covered') is not True or not w[start:start+count].any()):
            raise ValueError('source prefix/expert/branch/winning proof mismatch')
        if 'context_index' in data and not np.all(data['context_index'][start:start+count]==proof['context_index']):
            raise ValueError('context row/proof mismatch')
        offset+=count
    if offset!=n: raise ValueError('source proof ranges do not partition cache')


def load_trajectory(path, policy, base, warmstart, sources):
    path=Path(path);raw=(path/'manifest.json').read_bytes();manifest=json.loads(raw)
    require_legacy_experiment(manifest)
    if (manifest.get('format')!=CACHE_FORMAT or manifest.get('status')!='complete'
            or manifest.get('source')!='generated_only' or manifest.get('split')!='train'
            or manifest.get('no_future_inputs_to_current_or_imagined_fields') is not True):
        raise ValueError('verified v2 TRAIN on-policy field cache required')
    precision=manifest.get('precision',{})
    if (precision.get('compute_dtype')!='float32' or precision.get('autocast') is not False
            or precision.get('matmul_tf32') is not False or precision.get('cudnn_tf32') is not False
            or precision.get('imagined_input')!='stored current FP16 promoted to FP32'
            or precision.get('current_storage_dtype')!='float16' or precision.get('actual_storage_dtype')!='float16'
            or precision.get('imagined_storage_dtype')!='float32' or precision.get('imagined_quantization') is not False):
        raise ValueError('FP32 frozen prediction cache required')
    check_policy_encoder(manifest['field_encoder'],policy,True)
    if canonical_metadata(manifest['actor_sources'])!=canonical_metadata(policy.sources):
        raise ValueError('cache frozen encoder/dynamics binding mismatch')
    behavior=manifest['source_metadata'].get('behavior_checkpoint',{})
    if Path(behavior.get('path','')).resolve()!=Path(warmstart).resolve() or behavior.get('sha256')!=digest(warmstart):
        raise ValueError('trajectory behavior must be the evolving warmstart')
    verify_sources(manifest['source_hashes']);merge_sources(sources,manifest['source_hashes'])
    data={}
    for name,info in manifest['arrays'].items():
        if not isinstance(name,str) or not name.replace('_','').isalnum(): raise ValueError('invalid array member name')
        file=path/(name+'.npy')
        if digest(file)!=info['sha256']: raise ValueError(f'cache array changed: {name}')
        array=np.load(file,mmap_mode='r',allow_pickle=False)
        if list(array.shape)!=info['shape'] or str(array.dtype)!=info['dtype']: raise ValueError('array metadata mismatch')
        for first in range(0,len(array),32):
            if not np.isfinite(array[first:first+32]).all(): raise ValueError(f'nonfinite cache: {name}')
        data[name]=array;merge_sources(sources,{str(file):info['sha256']})
    validate_rows(data,manifest,base)
    auxiliary = manifest['source_metadata'].get('auxiliary_rows', [])
    if (not isinstance(auxiliary, list) or len(set(auxiliary)) != len(auxiliary)
            or any(type(i) is not int or not 0 <= i < len(data['seeds']) for i in auxiliary)):
        raise ValueError('invalid bound auxiliary row indices')
    if auxiliary:
        flags = np.zeros(len(data['seeds']), dtype=bool)
        flags[auxiliary] = True
        if np.any(flags & data['on_policy']):
            raise ValueError('auxiliary rows overlap policy visits')
        data['auxiliary'] = flags
    sha=hashlib.sha256(raw).hexdigest()
    if digest(path/'manifest.json')!=sha: raise ValueError('cache manifest changed while loading')
    merge_sources(sources,{str(path/'manifest.json'):sha})
    return data,manifest


def prepare_batch(views, trajectory, selection, treatment):
    """Same selected levels; only the marked states differ across arms."""
    rows=selection.base_rows;which=selection.base_views;n=len(rows)
    if len(np.unique(views[0]['seeds'][rows]))!=n: raise ValueError('duplicate level in batch')
    positions=np.flatnonzero(selection.trajectory_rows>=0);selected=selection.trajectory_rows[positions]
    if treatment:
        permitted = trajectory['on_policy'] | trajectory.get('auxiliary', np.zeros_like(trajectory['on_policy']))
        if not permitted[selected].all(): raise ValueError('unmarked expert anchor replacement forbidden')
        if not np.array_equal(trajectory['seeds'][selected],views[0]['seeds'][rows[positions]]):
            raise ValueError('replacement crossed levels')
    batch={}
    for key in ('optimal','next_fields','imagined_fields','seeds','difficulties','source_rows'):
        first=views[0][key];batch[key]=np.empty((n,*first.shape[1:]),dtype=first.dtype)
        for view in (0,1):
            positions=np.flatnonzero(which==view)
            batch[key][positions]=views[view][key][rows[positions]]
        if treatment:
            positions=np.flatnonzero(selection.trajectory_rows>=0)
            batch[key][positions]=trajectory[key][selected]
    if batch['imagined_fields'].dtype!=np.float32: raise ValueError('imagined fields must stay FP32')
    return batch


def selection_digest(hasher, selection, depth):
    for array in (selection.base_rows,selection.base_views,selection.trajectory_rows):
        hasher.update(np.asarray(array,dtype='<i8').tobytes())
    hasher.update(bytes([depth]))


def fresh_head(initial, device):
    head=StructuredWorkspaceReadout(initial.config())
    head.load_state_dict(initial.state_dict(),strict=True)
    if state_digest(head.state_dict())!=state_digest(initial.state_dict()): raise ValueError('initialization mismatch')
    return head.to(device).train()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory-cache',type=Path,required=True)
    parser.add_argument('--warmstart',type=Path,default=Path(WARMSTART))
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--batch-size',type=int,default=1024)
    parser.add_argument('--updates',type=int,default=200)
    parser.add_argument('--auxiliary-fraction',type=float,default=.25,
                        help='Probability of a bound expert/failure row in a same-level treatment replacement')
    parser.add_argument('--seconds',type=int,default=1800)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--checkpoint-prefix',type=Path,required=True)
    parser.add_argument('--capacity-report',type=Path,default=Path('artifacts/structured-workspace-preflight-gpu.json'))
    args=parser.parse_args()
    if args.report.exists() or any(Path(str(args.checkpoint_prefix)+'-'+arm+'.pt').exists() for arm in ('replay','onpolicy')):
        parser.error('new outputs required')
    if args.batch_size<2 or args.batch_size>1024 or args.batch_size&(args.batch_size-1) or not 1<=args.seconds<=1800:
        parser.error('power-of-two B2..1024 and deadline1..1800 required')
    if args.smoke:
        if args.device!='cpu' or args.batch_size>8 or args.updates!=2:parser.error('software smoke requires CPU B<=8 updates2')
    elif args.batch_size!=1024 or args.updates!=200:
        parser.error('production fixed B1024/200 updates; fallback requires reviewed plan')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();print('PID',os.getpid(),flush=True)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('comparison deadline')))
    signal.alarm(args.seconds)
    report={'status':'loading','pid':os.getpid(),'source':'generated_only','official_inputs_used':False,'smoke':args.smoke,
        'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'arms':{},'primary_depth':2,
        'trained_depths':list(DEPTHS),'precision':'training CUDA BF16 or CPU FP32; evaluation CPU FP32; imagined caches FP32, actual caches FP16',
        'limits':['Isolates state distribution at matched levels; both arms use evolving201315-parameter readout.',
                  'Trajectory source excludes within-life unreachable states and expert anchors from replacements.',
                  'No added objective, public factory changes, or gameplay claim.',
                  'Prior memory probe is architecture-only evidence, not a probe of this warmstart or dataset.']}
    def persist():
        report['elapsed_seconds']=time.monotonic()-started;atomic_json(args.report,report)
    persist();head=optimizer=None
    try:
        policy,views,sources=load_inputs()
        warm_sha=digest(args.warmstart);initial,warm=load_head(args.warmstart)
        if digest(args.warmstart)!=warm_sha:raise ValueError('warmstart changed during load')
        wp=warm.get('training_provenance',{})
        if (initial.cfg.memory_mode!='evolving' or initial.cfg.mode!='successors' or not initial.cfg.checkpoint_workspace
                or initial.parameter_count()!=201315 or initial.trainable_parameter_count()!=201315
                or warm.get('actor_checkpoint')!=ACTOR or warm.get('actor_sha256')!=sources[ACTOR]
                or wp.get('updates')!=600 or wp.get('batch_size')!=1024 or wp.get('smoke') is not False
                or wp.get('depths')!=list(DEPTHS) or wp.get('source')!='generated_only'
                or wp.get('official_inputs_used') is not False):
            raise ValueError('expected production evolving600/B1024 warmstart on same frozen actor')
        compatible=initial.config();compatible.pop('memory_mode');compatible.pop('checkpoint_workspace')
        if compatible!=policy.readout.config():raise ValueError('warmstart/base actor readout configuration mismatch')
        inherited_draws=wp.get('depth_draws',{})
        if set(inherited_draws)!={'1','2','4'} or any(type(v) is not int or v<=0 for v in inherited_draws.values()):
            raise ValueError('warmstart must have trained every inherited depth')
        merge_sources(sources,warm['sources']);merge_sources(sources,{str(args.warmstart):warm_sha})
        report['warmstart']={'path':str(args.warmstart),'sha256':warm_sha,'state_sha256':state_digest(initial.state_dict())}
        trajectory,manifest=load_trajectory(args.trajectory_cache,policy,views[0],args.warmstart,sources)
        sampler=PairedStateSampler(views[0]['seeds'],views[0]['difficulties'],trajectory['seeds'],trajectory['on_policy'],
                                  auxiliary=trajectory.get('auxiliary'), auxiliary_fraction=args.auxiliary_fraction)
        if not args.smoke and len(sampler.eligible)<1024:raise ValueError('production requires at least1024 on-policy eligible levels')
        replacements=min(args.batch_size//2,len(sampler.eligible))
        report['replacements']=replacements;report['trajectory_levels']=len(sampler.eligible);report['trajectory_rows']=int(trajectory['on_policy'].sum())
        report['auxiliary_rows']=int(trajectory.get('auxiliary', np.zeros_like(trajectory['on_policy'])).sum())
        report['auxiliary_replacement_probability']=sampler.auxiliary_fraction
        validations=load_validation(policy,sources)
        val_rows=np.arange(min(8,len(validations[0]['seeds'])) if args.smoke else len(validations[0]['seeds']))
        if not args.smoke and len(val_rows)!=512:raise ValueError('fixed512 validation levels required')
        for val in validations:
            if set(map(int,val['seeds']))&set(map(int,trajectory['seeds'])):raise ValueError('trajectory/validation seed overlap')
        report['validation_views_identical']=len(validations)==1;report['validation_rows']=val_rows.tolist()
        for path in (__file__,'tools/structured_onpolicy_sampling.py','tools/train_structured_workspace_comparison.py',
                     'tools/build_structured_onpolicy_fp32_cache.py','pebby/agent/structured_workspace_controller.py'):
            merge_sources(sources,{str(path):digest(path)})
        if not args.smoke:
            capacity=json.loads(args.capacity_report.read_text())
            if (capacity.get('status')!='complete' or capacity.get('batch_size')!=1024
                    or capacity.get('args',{}).get('device')!='cuda' or capacity.get('source_unchanged') is not True):
                raise ValueError('completed B1024 CUDA architecture capacity report required')
            module=Path('pebby/agent/structured_workspace_policy.py').resolve()
            matches=[sha for path,sha in capacity.get('sources',{}).items() if Path(path).resolve()==module]
            if not matches or any(sha!=digest(module) for sha in matches):
                raise ValueError('capacity report describes different workspace architecture source')
            merge_sources(sources,{str(args.capacity_report):digest(args.capacity_report)})
            report['capacity_evidence']={'path':str(args.capacity_report),'sha256':digest(args.capacity_report),
                'scope':'Earlier identical architecture/depth4/optimizer, different initializer and data; no fresh capacity run in this tool.'}
        verify_sources(sources);report['sources']=sources
        reference=None
        for arm in ('replay','onpolicy'):
            torch.manual_seed(42);head=fresh_head(initial,args.device)
            if state_digest(head.state_dict())!=report['warmstart']['state_sha256']:raise ValueError('arm initializer differs')
            optimizer=torch.optim.AdamW(head.parameters(),lr=.0003,weight_decay=.01)
            rng=np.random.default_rng(42);depth_rng=np.random.default_rng(44)
            stream=hashlib.sha256();used=hashlib.sha256();seen=set();candidate_seen=set();levels_seen=set()
            counts=np.zeros(5,np.int64);depth_counts={str(d):0 for d in DEPTHS}
            entry={'status':'training','parameters':head.parameter_count(),'active_parameters':head.trainable_parameter_count(),
                'initial_state_sha256':state_digest(head.state_dict()),'training':[],'clipped_steps':0}
            report['arms'][arm]=entry
            for step in range(args.updates):
                tick=time.monotonic()
                selection=sampler.draw(args.batch_size,replacements,step/max(args.updates-1,1),rng)
                depth=int(depth_rng.choice(DEPTHS));selection_digest(stream,selection,depth)
                batch=prepare_batch(views,trajectory,selection,arm=='onpolicy')
                used.update(np.asarray(batch['source_rows'],dtype='<i8').tobytes())
                used.update(np.asarray(selection.trajectory_rows>=0 if arm=='onpolicy' else np.zeros(args.batch_size,bool),dtype='u1').tobytes())
                optimizer.zero_grad(set_to_none=True)
                losses=backward(head,batch,args.device,depth)
                norm=torch.nn.utils.clip_grad_norm_(head.parameters(),10.,error_if_nonfinite=True)
                optimizer.step()
                value=float(norm);entry['clipped_steps']+=int(value>10)
                selected=selection.trajectory_rows[selection.trajectory_rows>=0]
                candidate_seen.update(map(int,selected))
                if arm=='onpolicy':seen.update(map(int,selected))
                levels_seen.update(map(int,batch['seeds']));counts+=np.bincount(batch['difficulties'],minlength=6)[1:6]
                depth_counts[str(depth)]+=1
                gates={name:float(p.detach()) for name,p in head.workspace.named_parameters() if name.endswith('_gate')}
                if not all(np.isfinite(v) for v in gates.values()):raise ValueError('nonfinite workspace gates')
                entry['training'].append({'step':step+1,'loops':depth,'losses':losses,'gradient_norm':value,
                    'on_policy_replacements':int(trajectory['on_policy'][selected].sum()) if arm=='onpolicy' else 0,
                    'auxiliary_replacements':int((~trajectory['on_policy'][selected]).sum()) if arm=='onpolicy' else 0,
                    'gates':gates,'seconds':time.monotonic()-tick})
                entry.update(completed_updates=step+1,selection_sha256=stream.hexdigest(),used_state_rows_sha256=used.hexdigest(),
                    distinct_levels_seen=len(levels_seen),trajectory_states_seen=len(seen),candidate_trajectory_states_seen=len(candidate_seen),
                    difficulty_draws=counts.tolist(),depth_draws=dict(depth_counts))
                if step==0 or (step+1)%25==0 or step+1==args.updates:
                    report['status']=arm+'_training';persist();print(json.dumps({'arm':arm,**entry['training'][-1]}),flush=True)
                del batch
            if reference is not None and reference!=stream.hexdigest():raise ValueError('paired row/view/trajectory/depth selections diverged')
            reference=stream.hexdigest()
            if not args.smoke and any(v==0 for v in depth_counts.values()):raise ValueError('declared training depth not sampled')
            del optimizer;optimizer=None;head.to('cpu').eval();gc.collect()
            if args.device=='cuda':torch.cuda.empty_cache()
            report['status']=arm+'_evaluation';entry['status']='evaluating';persist()
            entry['validation']=[evaluate(head,val,val_rows) for val in validations]
            verify_sources(sources)
            provenance={'arm':arm,'experiment':'matched_onpolicy_state_distribution','updates':args.updates,'final_update':args.updates,
                'batch_size':args.batch_size,'replacements':replacements,'seed':42,'depth_seed':44,'depths':list(DEPTHS),
                'trained_depths':list(DEPTHS),'depth_draws':dict(depth_counts),'continuation_depth_draws':dict(depth_counts),
                'inherited_depth_draws':inherited_draws,'trained_depths_scope':'Cumulative depth support inherited from evolving600 plus this continuation',
                'primary_depth':2,'fixed_final':True,'smoke':args.smoke,
                'source':'generated_only','official_inputs_used':False,'encoder_and_dynamics_frozen':True,
                'selection_sha256':stream.hexdigest(),'used_state_rows_sha256':used.hexdigest(),
                'evolving_warmstart':report['warmstart'],'trajectory_cache':str(args.trajectory_cache),
                'trajectory_manifest_sha256':digest(args.trajectory_cache/'manifest.json')}
            entry['checkpoint']=save_head(str(args.checkpoint_prefix)+'-'+arm+'.pt',head,sources,provenance)
            entry['status']='complete';persist();del head;head=None;gc.collect()
        verify_sources(sources);report.update(status='complete',sources_unchanged=True,paired_selections_exact=True,initializations_exact=True)
    except BaseException as error:
        report.update(status='failed_partial',error=str(error));raise
    finally:
        head=optimizer=None;gc.collect()
        if args.device=='cuda' and torch.cuda.is_available():torch.cuda.empty_cache()
        signal.alarm(0);persist()


if __name__=='__main__':main()
