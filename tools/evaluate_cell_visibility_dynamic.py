"""Frozen visibility head on audited generated gameplay frames, no fitting."""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time
import numpy as np
import torch
from torch.nn import functional as F
from pebby.agent.cell_visibility import CellVisibility, FORMAT
from tools.train_cell_visibility import binary_metrics, digest

DATA = Path('data/cell-appearance-dynamic-public-frames.npz')
PROOF = Path('artifacts/cell-appearance-dynamic-audit.json')
CHECKPOINT = Path('checkpoints/ls20-cell-visibility-initial-200.pt')
OUTPUT = Path('artifacts/cell-visibility-dynamic-evaluation.json')
EXPECTED = {str(DATA): '7ea620fd7dae5ab3111dadef1263ae70b3910fb1b8739fd96e277072ea3a341b',
            str(PROOF): 'a151c13635b31818dacacb952fc513a2a0bab3b9c03827aca28aac9094f8c234',
            str(CHECKPOINT): '6c0145297f5b1e88532f03aab19fe048c215be61cf06e44d071d8306943d4c7f'}


def main():
    start = time.monotonic(); print('PID', os.getpid(), flush=True)
    def timeout(*_):
        raise TimeoutError('frozen visibility evaluation120second deadline')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(120)
    torch.set_num_threads(1)
    if OUTPUT.exists():raise FileExistsError(OUTPUT)
    paths = [DATA, PROOF, CHECKPOINT, Path('pebby/agent/cell_visibility.py'),
             Path('tools/train_cell_visibility.py'), Path(__file__)]
    hashes = {str(p): digest(p) for p in paths}
    if any(hashes[k] != v for k,v in EXPECTED.items()):raise ValueError('pinned source changed')
    proof = json.loads(PROOF.read_text())
    if proof['status'] != 'complete' or proof['public_export']['sha256'] != hashes[str(DATA)]:
        raise ValueError('export proof mismatch')
    with np.load(DATA, allow_pickle=False) as archive:data = {k:archive[k] for k in archive.files}
    if not np.all((data['seed'] >= 1_000_000) & (data['seed'] < 2_000_000)):
        raise ValueError('only generated validation levels accepted')
    n = len(data['frames']); labels = data['support7']
    assert n == 3148 and data['frames'].shape == (n,64,64) and labels.shape == (n,144)
    assert data['frames'].dtype == np.uint8 and labels.dtype == bool
    model = CellVisibility(); checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    assert checkpoint['format'] == FORMAT
    assert checkpoint['source_hashes']['pebby/agent/cell_visibility.py'] == hashes['pebby/agent/cell_visibility.py']
    model.load_state_dict(checkpoint['weights']); model.eval();model.requires_grad_(False)
    training_data = Path('data/ls20-visible-cell-labels-2k.npz')
    assert digest(training_data) == checkpoint['source_hashes'][str(training_data)]
    with np.load(training_data,allow_pickle=False) as archive:
        train_seeds=set(map(int,archive['seeds'][archive['split']=='train']))
    assert not train_seeds.intersection(map(int,data['seed']))
    with torch.inference_mode():
        # Support masks, category flags, coordinates and labels are never used
        # here: the complete actor input is one public frame.
        logits = torch.cat([model(frames) for frames in torch.from_numpy(data['frames']).split(128)])
    predicted = logits.ge(0).numpy()
    row,col=np.divmod(np.arange(144),12)
    boundary=(row==0)|(col==11);hud=(row>=10)&~boundary
    assert not labels[:,boundary|hud].any()
    interior_excluded=~labels & ~(boundary|hud)[None]
    reasons={'boundary':np.broadcast_to(boundary,labels.shape),'hud':np.broadcast_to(hud,labels.shape),
             'fog':interior_excluded & data['fog'][:,None],
             'nonfog_interior':interior_excluded & ~data['fog'][:,None]}
    categories={'all':np.ones(n,bool),'fog':data['fog'],'nonfog':~data['fog']}
    for name in np.unique(data['collection']):categories[str(name)]=data['collection']==name
    for name in ['player_overlap','goal_covered','goal_solved_removed','goal_active_public',
                 'terminal','won','animation','death_flash','life_lost','player_reset_to_spawn','moving_goal']:
        categories[name]=data[name]
    results={}
    for name,mask in categories.items():
        count=int(mask.sum())
        if count == 0:
            results[name]={'frames':0,'metrics':None,'note':'No exported frames in this category; not tested.'};continue
        reason_subset={key:value[mask] for key,value in reasons.items()}
        metrics=binary_metrics(predicted[mask],labels[mask],reason_subset)
        metrics['bce']=float(F.binary_cross_entropy_with_logits(logits[mask],torch.from_numpy(labels[mask]).float()))
        metrics['always_visible']=binary_metrics(np.ones_like(labels[mask]),labels[mask],reason_subset)
        results[name]={'frames':count,'levels':len(set(map(int,data['seed'][mask]))),'metrics':metrics}
    # Cell-level overlap errors distinguish support visibility from whether the
    # underlay's semantic role is recoverable under the player sprite.
    overlap=data['overlap_cells'];error=predicted!=labels
    overlap_result={'cells':int(overlap.sum()),'false_positive':int((predicted & ~labels & overlap).sum()),
                    'false_negative':int((~predicted & labels & overlap).sum())}
    errors=[]
    for r in np.flatnonzero(error.any(1)):
        errors.append({'row':int(r),'seed':int(data['seed'][r]),'collection':str(data['collection'][r]),
                       'route_index':int(data['route_index'][r]),'action':int(data['action'][r]),
                       'fog':bool(data['fog'][r]),'player_overlap':bool(data['player_overlap'][r]),
                       'false_positive_cells':np.flatnonzero(predicted[r]&~labels[r]).tolist(),
                       'false_negative_cells':np.flatnonzero(~predicted[r]&labels[r]).tolist()})
    assert all(digest(p)==value for p,value in hashes.items())
    report={'status':'complete','training_performed':False,'parameters':model.parameter_count(),
            'checkpoint':str(CHECKPOINT),'source_hashes':hashes,'frame_batch_size':128,
            'cpu_threads':1,'device':'cpu','pid':os.getpid(),'train_level_overlap':0,
            'levels':len(set(map(int,data['seed']))),'metrics_by_category':results,
            'overlap_cell_visibility':overlap_result,'error_frames':errors,
            'public_input':'full64x64frame only; no provided support mask/flags/geometry',
            'threshold':'fixed logit0 from original experiment; no threshold adjustment or geometry postmask',
            'elapsed_seconds':time.monotonic()-start,
            'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'limits':['Generated validation gameplay trajectories and sampled action branches only; repeated/correlated frames.',
                      'Zero exported life-loss/reset/animation/death-flash/moving-goal frames; separate random-reset outcome counts are not input frames or evaluated coverage.',
                      'Support labels and category flags are used only for scoring; public support visibility does not prove semantic underlay identifiability.',
                      'No visibility gating added to appearance, dynamics or policy; this is not certification or controller completion.']}
    OUTPUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');signal.alarm(0)
    print(json.dumps({'all':results['all'],'fog':results['fog'],'player_overlap':results['player_overlap'],
                      'elapsed_seconds':report['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
