"""Attest original batch-one public-history choices; no engine or oracle."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SNAPSHOT=ROOT/'artifacts/world-onpolicy-round2-code'
sys.path.insert(0,str(SNAPSHOT))
import numpy as np
import torch
from pebby.agent.model import load_checkpoint

def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def input_digest(data,row):
    return hashlib.sha256(b''.join(np.asarray(data[key][row]).tobytes() for key in ('frames','history_valid','previous_actions'))).hexdigest()
def main():
    torch.set_num_threads(1);started=time.monotonic();print('PID',os.getpid(),flush=True)
    def timeout(*_):raise TimeoutError('batch-one attestation 120-second budget')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    prior=ROOT/'artifacts/world-closing-sequence-feasibility.json';old=json.loads(prior.read_text())
    source=ROOT/old['source'];assert digest(source)==old['source_sha256']
    cache=ROOT/'data/world-array-cache'/f"{old['source_sha256']}-07d1d1f252efefe5"
    arrays={k:np.load(cache/f'{k}.npy',mmap_mode='r') for k in ('frames','history_valid','previous_actions','terminal','won','lost_life','distances','next_optimal')}
    hashes=json.loads((SNAPSHOT/'source-hashes.json').read_text());assert all(digest(SNAPSHOT/p)==h for p,h in hashes.items())
    records=[]
    for bi,behavior in enumerate(old['behavior_checkpoints']):
        path=ROOT/behavior['path'];assert digest(path)==behavior['sha256'];model,_=load_checkpoint(path,'cpu');model.eval()
        with torch.inference_mode():
            for item in old['per_level']:
                if item['behavior_index']!=bi:continue
                row=item['last_row'];logits=model(torch.from_numpy(np.array(arrays['frames'][row:row+1])).long(),history_valid=torch.from_numpy(np.array(arrays['history_valid'][row:row+1])),previous_actions=torch.from_numpy(np.array(arrays['previous_actions'][row:row+1])))
                action=int(logits[0].argmax());assert action==item['choice'],(row,action,item['choice'])
                terminal=bool(arrays['terminal'][row,action]);won=bool(arrays['won'][row,action]);distance=int(arrays['distances'][row,action])
                if item['stop']=='won':assert won
                if item['stop']=='unreachable_deadend':assert distance<0 and not terminal
                records.append({'seed':item['seed'],'row':row,'action':action,'behavior_index':bi,'public_history_sha256':input_digest(arrays,row)})
        del model
        print('BEHAVIOR',bi,'count',len(records),flush=True)
    records.sort(key=lambda r:r['row']);assert len(records)==2048
    report={'format':'pebby.ls20-closing-behavior-attestation.v1','source':'generated_only','split':'train','source_path':str(source),'source_sha256':old['source_sha256'],'behavior_checkpoints':old['behavior_checkpoints'],'source_ranges':old['source_ranges'],'inference':'greedy_argmax_batch1','batch_size':1,'cpu_threads':1,'device':'cpu','oracle_calls':0,'engine_calls':0,'official_inputs':False,'frozen_code':str(SNAPSHOT),'code_hashes':hashes,'script_path':str(Path(__file__).resolve()),'script_sha256':digest(__file__),'prior_report_sha256':digest(prior),'all_batch32_choices_match_batch1':True,'all_stop_flags_agree':True,'records':records,'pid':os.getpid(),'runtime_seconds':time.monotonic()-started}
    out=ROOT/'artifacts/world-closing-actions-batch1.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n')
    signal.alarm(0);print(json.dumps({'status':'complete','rows':len(records),'seconds':report['runtime_seconds'],'sha256':digest(out)}),flush=True)
if __name__=='__main__':main()
