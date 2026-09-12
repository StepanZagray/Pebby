"""Generated-only exact-pixel visibility aliases; no fitted models or engine calls."""
import collections
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time
import numpy as np
import torch
from pebby.agent.cell_appearance import cell_patches


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started=time.monotonic();print('PID',os.getpid(),flush=True)
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('120s budget')));signal.alarm(120)
    torch.set_num_threads(1)
    source=Path('data/ls20-visible-cell-labels.npz');out=Path('artifacts/world-patch-visibility-identifiability.json')
    if out.exists():raise FileExistsError(out)
    paths=[source,Path('pebby/agent/cell_appearance.py'),Path(__file__)]
    hashes={str(p):digest(p) for p in paths}
    expected=json.loads(Path('artifacts/cell-appearance-initial-400.json').read_text())['source_hashes'][str(source)]
    assert hashes[str(source)]==expected
    with np.load(source,allow_pickle=False) as archive:data={k:archive[k] for k in archive.files}
    assert len(data['seeds'])==350 and len(set(data['seeds']))==350
    assert all((0<=int(seed)<1_000_000 if split=='train' else 1_000_000<=int(seed)<2_000_000) for seed,split in zip(data['seeds'],data['split']))
    patches=cell_patches(torch.from_numpy(data['frames'])).numpy().astype(np.uint8)
    groups=collections.defaultdict(list);reasons=collections.Counter();interior_groups=collections.defaultdict(list)
    inferred_fog=~data['fully_visible'].all(1)
    for r in range(350):
        px,py=map(int,data['player_cell'][r]);cx,cy=4+5*px+1.5,5*py+1.5
        for cell in range(144):
            row,col=divmod(cell,12);left,top=3+5*col,5*row-1
            boundary=top<0 or left<0 or top+7>64 or left+7>64
            hud=top+7>52
            fog=bool(inferred_fog[r]) and any((x-cx)**2+(y-cy)**2>400 for y in range(top,top+7) for x in range(left,left+7))
            reason='boundary' if boundary else 'hud' if hud else 'fog' if fog else 'visible'
            visible=bool(data['support7_label_mask'][r,cell]);assert visible==(reason=='visible'),(r,cell,reason)
            # Independent slicing checks padding and pixel anchor against the model helper.
            padded=np.pad(data['frames'][r],1);independent=padded[top+1:top+8,left+1:left+8]
            assert np.array_equal(patches[r,cell],independent)
            key=patches[r,cell].tobytes();entry=(r,cell,visible,reason)
            groups[key].append(entry);reasons[reason]+=1
            if not boundary and not hud:interior_groups[key].append(entry)
    def census(pool,details=False):
        conflicts=[];affected=collections.Counter();seedset=set();minimum_errors=0
        for key,items in pool.items():
            counts=collections.Counter(v for _,_,v,_ in items)
            if len(counts)!=2:continue
            minimum_errors+=min(counts.values())
            for r,cell,visible,reason in items:affected[reason]+=1;seedset.add(int(data['seeds'][r]))
            record={'patch_sha256':hashlib.sha256(key).hexdigest(),'visible':counts[True],'excluded':counts[False], 'reason_counts':dict(collections.Counter(x[3] for x in items))}
            if details:record.update(pixels=np.frombuffer(key,dtype=np.uint8).reshape(7,7).tolist(),occurrences=[{'record':r,'cell':cell,'seed':int(data['seeds'][r]),'split':str(data['split'][r]),'context':int(data['context_index'][r]),'player_cell':data['player_cell'][r].tolist(),'visible':v,'reason':reason} for r,cell,v,reason in items])
            conflicts.append(record)
        return {'patches':sum(map(len,pool.values())),'unique_patterns':len(pool),'conflicting_patterns':len(conflicts),'affected_occurrences':sum(affected.values()),'affected_by_reason':dict(affected),'affected_distinct_seeds':len(seedset),'minimum_unavoidable_patch_only_binary_errors_in_sample':minimum_errors,'conflicts':conflicts}
    all_stats=census(groups,True);interior_stats=census(interior_groups,False)
    assert all(digest(p)==h for p,h in hashes.items())
    result={'status':'complete','source_hashes':hashes,'records':350,'patches':50400,'source':'generated_initial_states_only','cpu_threads':1,'pid':os.getpid(),'fog_records_inferred_from_stored_full_visibility':int(inferred_fog.sum()),'mask_reconstruction_mismatches':0,'patch_extraction_mismatches':0,'reason_counts':dict(reasons),'all144_including_padding':all_stats,'interior_non_hud_only':interior_stats,'method':'Exact49-byte grouping, no mask-based filtering. Separate first rejection boundary, HUD, then fog. Fog record status inferred from stored fully_visible; full7x7 circular support recomputed from stored diagnostic player cell. All persisted support masks matched. Source NPZ SHA matches the initial decoder run.','limitations':['No visibility classifier trained; masks/player/context are diagnostic labels only, never proposed inference inputs.','Any interior conflicting pixel pattern rules out certainty of visibility from that local patch alone; frequency is sample-specific, not a population guarantee.','No conflict would establish only finite-sample consistency, not general identifiability.','Fixed HUD and frame boundaries can be handled by known global coordinates; this does not resolve genuine interior fog ambiguity.','No official inputs, engine execution, Oracle calls, or controller evaluation.'],'elapsed_seconds':time.monotonic()-started,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    out.write_text(json.dumps(result,indent=2)+'\n');signal.alarm(0)
    print(json.dumps({k:v for k,v in result.items() if k not in ['all144_including_padding','interior_non_hud_only']}),flush=True)
    for name in ['all144_including_padding','interior_non_hud_only']:print(name,json.dumps({k:v for k,v in result[name].items() if k!='conflicts'}),flush=True)

if __name__=='__main__':main()
