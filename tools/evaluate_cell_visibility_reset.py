"""Frozen public-frame visibility on generated before/after first-life-loss pairs."""
import json
import os
from pathlib import Path
import resource
import signal
import time
import numpy as np
import torch
from pebby.agent.cell_visibility import CellVisibility, FORMAT
from tools.train_cell_visibility import binary_metrics, digest

DATA=Path('data/cell-appearance-reset-public-frames.npz')
PROOF=Path('artifacts/cell-appearance-reset-audit.json')
CHECKPOINT=Path('checkpoints/ls20-cell-visibility-initial-200.pt')
OUTPUT=Path('artifacts/cell-visibility-reset-evaluation.json')
EXPECTED={str(DATA):'37b7cc5d786eb5a0b98c198e97b9f5e4615d015b36656ca40e576459aef89c16',
          str(PROOF):'1b37740b3356780117ced0abc81f7574a4c89bf5c4dc5cf816ccafeef27a9ead',
          str(CHECKPOINT):'6c0145297f5b1e88532f03aab19fe048c215be61cf06e44d071d8306943d4c7f'}


def main():
    start=time.monotonic();print('PID',os.getpid(),flush=True);torch.set_num_threads(1)
    def timeout(*_):raise TimeoutError('reset visibility evaluation60second deadline')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(60)
    if OUTPUT.exists():raise FileExistsError(OUTPUT)
    paths=[DATA,PROOF,CHECKPOINT,Path(__file__),Path('pebby/agent/cell_visibility.py'),Path('tools/train_cell_visibility.py')]
    hashes={str(p):digest(p) for p in paths}
    if any(hashes[k]!=v for k,v in EXPECTED.items()):raise ValueError('pinned source changed')
    proof=json.loads(PROOF.read_text())
    assert proof['status']=='complete' and proof['public_export']['sha256']==hashes[str(DATA)]
    with np.load(DATA,allow_pickle=False) as a:data={k:a[k] for k in a.files}
    assert data['frames'].shape==(20,64,64) and data['support7'].shape==(20,144)
    assert data['frames'].dtype==np.uint8 and data['support7'].dtype==bool
    assert ((data['seed']>=1_000_000)&(data['seed']<2_000_000)).all()
    cp=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    assert cp['format']==FORMAT and cp['source_hashes']['pebby/agent/cell_visibility.py']==hashes['pebby/agent/cell_visibility.py']
    original=Path('data/ls20-visible-cell-labels-2k.npz')
    assert digest(original)==cp['source_hashes'][str(original)]
    with np.load(original,allow_pickle=False) as a:training=set(map(int,a['seeds'][a['split']=='train']))
    assert not training.intersection(map(int,data['seed']))
    model=CellVisibility();model.load_state_dict(cp['weights']);model.eval();model.requires_grad_(False)
    with torch.inference_mode():
        # Public single frames are the complete model inputs, including after
        # life loss; no reset flag or support mask enters this computation.
        logits=torch.cat([model(x) for x in torch.from_numpy(data['frames']).split(1)])
    predicted=logits.ge(0).numpy();labels=data['support7']
    row,col=np.divmod(np.arange(144),12);boundary=(row==0)|(col==11);hud=(row>=10)&~boundary
    assert not labels[:,boundary|hud].any()
    interior=~labels&~(boundary|hud)[None]
    reasons={'boundary':np.broadcast_to(boundary,labels.shape),'hud':np.broadcast_to(hud,labels.shape),
             'fog':interior&data['fog'][:,None],'nonfog_interior':interior&~data['fog'][:,None]}
    categories={'all':np.ones(20,bool),'pre_reset':data['collection']=='reset_pre_action',
                'post_reset':data['collection']=='reset_post_action','fog':data['fog'],'nonfog':~data['fog'],
                'player_overlap':data['player_overlap'],'animation':data['animation'],'death_flash':data['death_flash']}
    results={}
    for category,mask in categories.items():
        n=int(mask.sum())
        if n==0:results[category]={'frames':0,'metrics':None};continue
        subset={key:val[mask] for key,val in reasons.items()}
        results[category]={'frames':n,'metrics':binary_metrics(predicted[mask],labels[mask],subset),
                           'always_visible':binary_metrics(np.ones_like(labels[mask]),labels[mask],subset)}
    pairs=[]
    for seed in sorted(set(map(int,data['seed']))):
        pre=np.flatnonzero((data['seed']==seed)&categories['pre_reset']);post=np.flatnonzero((data['seed']==seed)&categories['post_reset'])
        assert len(pre)==len(post)==1
        i,j=int(pre[0]),int(post[0]);assert data['lives'][i]==3 and data['lives'][j]==2
        assert data['life_lost'][j] and data['player_reset_to_spawn'][j]
        true_change=labels[i]!=labels[j];pred_change=predicted[i]!=predicted[j]
        pairs.append({'seed':seed,'pre_row':i,'post_row':j,'fog':bool(data['fog'][i]),
                      'pre_visible_count':int(labels[i].sum()),'post_visible_count':int(labels[j].sum()),
                      'true_mask_changed_bits':int(true_change.sum()),'predicted_mask_changed_bits':int(pred_change.sum()),
                      'true_became_visible_cells':np.flatnonzero(~labels[i]&labels[j]).tolist(),
                      'true_became_hidden_cells':np.flatnonzero(labels[i]&~labels[j]).tolist(),
                      'changed_bit_detection_false_positive':int((pred_change&~true_change).sum()),
                      'changed_bit_detection_false_negative':int((~pred_change&true_change).sum()),
                      'true_changed_bits_both_endpoints_correct':int((true_change&(predicted[i]==labels[i])&(predicted[j]==labels[j])).sum()),
                      'pre_fp':np.flatnonzero(predicted[i]&~labels[i]).tolist(),'pre_fn':np.flatnonzero(~predicted[i]&labels[i]).tolist(),
                      'post_fp':np.flatnonzero(predicted[j]&~labels[j]).tolist(),'post_fn':np.flatnonzero(~predicted[j]&labels[j]).tolist()})
    assert all(digest(p)==v for p,v in hashes.items())
    report={'status':'complete','source_hashes':hashes,'checkpoint':str(CHECKPOINT),'parameters':model.parameter_count(),
            'pid':os.getpid(),'device':'cpu','cpu_threads':1,'frame_batch_size':1,'training_performed':False,
            'training_level_overlap':0,'metrics_by_category':results,'pairs':pairs,
            'mask_count_vs_identity':'Compare full144-bit masks per pair. Equal count is not evidence of identical support.',
            'input_contract':'public64x64frame only; labels/collection/reset/player/fog metadata enter scoring only',
            'threshold':'fixed logit0; no geometry postmask or threshold selection',
            'elapsed_seconds':time.monotonic()-start,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'limits':['Ten generated validation level pairs at first life loss; only one fog pair, no animation/death-flash frames.',
                      'Stateful controller history reset is not evaluated: this model accepts a single current frame.',
                      'Support visibility predictions are not certified and do not establish hidden-underlay semantic identifiability.',
                      'No official inputs, engine replay, Oracle calls, optimization, model integration or changes to prior dynamic artifacts.']}
    OUTPUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');signal.alarm(0)
    print(json.dumps({'all':results['all'],'fog':results['fog'],'changed_pairs':[p for p in pairs if p['true_mask_changed_bits']]}),flush=True)


if __name__=='__main__':main()
