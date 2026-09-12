"""Source-bound public-H8 search adapter; no engine/teacher inference inputs.

This format packages references to frozen learned components. It does not
attest that distance/outcome calibration is adequate for deployed planning.
"""
from dataclasses import dataclass,asdict
import copy,hashlib,io,os,tempfile
from pathlib import Path
import torch
from torch import nn
from .structured_factored_policy import StructuredFactoredPolicy,canonical_metadata,state_digest,_digest,_verify_sources
from .structured_distance import StructuredDistanceReadout
from .event_calibration import PositiveSlopePlatt,FORMAT as CALIBRATION_FORMAT,EVENT_NAMES
from .structured_search import search

FORMAT='pebby.structured-search-policy.v1'
DISTANCE_FORMAT='pebby.structured-distance-readout.v1'

@dataclass(frozen=True)
class SearchPolicyConfig:
    depth:int=4
    beam:int=4
    gamma:float=.99
    chunk_size:int=128
    history:int=8
    architecture:str='structured'
    def __post_init__(self):
        if type(self.depth)!=int or not 1<=self.depth<=4 or type(self.beam)!=int or self.beam!=4 or self.gamma!=.99 or isinstance(self.gamma,bool) or type(self.chunk_size)!=int or not 1<=self.chunk_size<=128 or self.history!=8 or self.architecture!='structured':raise ValueError('invalid bounded search policy config')

def _read(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True),{'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}

def _hashes(hashes):
    if not isinstance(hashes,dict) or not hashes:raise ValueError('complete source hashes required')
    if any(not isinstance(p,str) or not isinstance(h,str) or _digest(p)!=h for p,h in hashes.items()):raise ValueError('source hash drift')

def _code():
    return {str(Path(__file__).with_name(f).resolve()):_digest(Path(__file__).with_name(f)) for f in ('structured_search_policy.py','structured_search.py','structured_distance.py','event_calibration.py')}

def _distance(saved,policy):
    if saved.get('format')!=DISTANCE_FORMAT or saved.get('official_inputs_used') is not False or saved.get('policy_integrated') is not False:raise ValueError('invalid generated distance checkpoint')
    if canonical_metadata(saved.get('frozen_binding'))!=canonical_metadata(policy.sources):raise ValueError('distance frozen binding mismatch')
    _hashes(saved.get('sources'))
    manifests=saved.get('cache_manifests')
    if not isinstance(manifests,dict) or set(manifests)!={'train','validation'}:raise ValueError('distance split provenance required')
    for split,m in manifests.items():
        if m.get('split')!=split or m.get('source')!='generated_only' or m.get('status')!='complete':raise ValueError('invalid distance cache provenance')
        if canonical_metadata(m.get('field_encoder'))!=canonical_metadata(policy.sources['dynamics_metadata']['field_encoder']):raise ValueError('distance encoder cache mismatch')
    head=StructuredDistanceReadout(saved['config']);head.load_state_dict(saved['weights'],strict=True)
    if saved.get('parameters')!=head.parameter_count():raise ValueError('distance parameter count mismatch')
    return head

def _calibration(saved,policy,distance):
    if saved.get('format')!=CALIBRATION_FORMAT or saved.get('parameters')!=6 or saved.get('event_names')!=list(EVENT_NAMES):raise ValueError('conditional v2 calibration required')
    config=saved.get('config',{})
    if config!={'initial_slope':1.0,'positive_slope':True,'conditional_won':True,'terminal_semantics':'diagnostic_all_branches','coherent_outcomes':['loss','win','continue']}:raise ValueError('conditional won calibration config required')
    binding=saved.get('source_binding',{})
    if binding.get('format')!=CALIBRATION_FORMAT:raise ValueError('calibration binding format mismatch')
    dyn=policy.sources['artifacts']['dynamics']
    if Path(binding.get('checkpoint','')).resolve()!=Path(dyn['path']).resolve() or binding.get('checkpoint_sha256')!=dyn['sha256']:raise ValueError('calibration dynamics mismatch')
    hashes=binding.get('source_hashes');_hashes(hashes)
    normalized={str(Path(p).resolve()):h for p,h in hashes.items()}
    expected=dict(policy.sources['code_hashes'])
    expected.update({r['path']:r['sha256'] for r in policy.sources['artifacts'].values()})
    for path in (Path(__file__).with_name('event_calibration.py'),Path('tools/calibrate_structured_events.py'),Path('tools/evaluate_structured_event_head.py')):expected[str(path.resolve())]=_digest(path)
    if any(normalized.get(str(Path(p).resolve()))!=h for p,h in expected.items()):raise ValueError('incomplete calibration implementation binding')
    splits=binding.get('splits')
    if not isinstance(splits,dict) or set(splits)!={'train','validation'}:raise ValueError('calibration split hashes required')
    distance_sources={str(Path(p).resolve()):h for p,h in distance['sources'].items()}
    for split,record in splits.items():
        root=Path(record['source_root']);manifest=root/'manifest.json'
        if record.get('source_manifest_sha256')!=_digest(manifest) or distance_sources.get(str(manifest.resolve()))!=record['source_manifest_sha256']:raise ValueError('calibration/distance source cache mismatch')
        _hashes(record.get('cache_fingerprints'))
        _hashes(record.get('source_cache_fingerprints'))
        for group in ('cache_fingerprints','source_cache_fingerprints'):
            for p,h in record[group].items():
                if distance_sources.get(str(Path(p).resolve()))!=h:raise ValueError('calibration/distance array mismatch')
    head=PositiveSlopePlatt(config['initial_slope']);head.load_state_dict(saved['state_dict'],strict=True)
    return head

class StructuredSearchPolicy(nn.Module):
    def __init__(self,encoder,dynamics,distance,calibrator,config=None,*,sources=None):
        super().__init__();self.cfg=SearchPolicyConfig(**(config or {}));self.encoder=encoder;self.dynamics=dynamics;self.distance=distance;self.calibrator=calibrator;self.sources=copy.deepcopy(sources);self.train(False)
    def train(self,mode=True):
        super().train(False);self.requires_grad_(False);return self
    def config(self):return asdict(self.cfg)
    def parameter_counts(self):
        counts={k:sum(p.numel() for p in getattr(self,k).parameters()) for k in ('encoder','dynamics','distance','calibrator')}
        return {**counts,'trainable':0,'total':sum(counts.values())}
    def parameter_count(self):return self.parameter_counts()['total']
    def _transition(self,fields,actions):
        output=self.dynamics(fields,actions);events=output['events']
        return output['field'],torch.stack([events[k+'_logits'] for k in EVENT_NAMES],-1)
    def _outcomes(self,raw):
        p=self.calibrator.coherent_probabilities(raw);return torch.stack([p[k] for k in ('loss','win','continue')],-1)
    @torch.no_grad()
    def search_fields(self,fields):
        return search(fields,self._transition,self.distance,self._outcomes,depth=self.cfg.depth,beam=self.cfg.beam,gamma=self.cfg.gamma,chunk_size=self.cfg.chunk_size)
    @torch.no_grad()
    def forward(self,frames,history_valid=None,previous_actions=None):
        fields=self.encoder(frames,history_valid,previous_actions);result=self.search_fields(fields)
        # The generic history evaluator casts to float32. Ordinal scores preserve
        # exact core ordering even when expected returns differ below float32
        # resolution. These are greedy ordering scores, not sampling logits.
        scores=torch.empty(len(fields),4,device=fields.device)
        for row,item in enumerate(result['results']):
            ordered=sorted(item['roots'],key=lambda r:(-r['score'],r['root_action']))
            for rank,root in enumerate(ordered):scores[row,root['root_action']]=4-rank
        return scores
    @classmethod
    def from_checkpoints(cls,world,visibility,dynamics,distance,calibration,config=None,device='cpu'):
        factory=StructuredFactoredPolicy.from_checkpoints(world,visibility,dynamics,device='cpu')
        d,dp=_read(distance);c,cp=_read(calibration);head=_distance(d,factory);cal=_calibration(c,factory,d)
        policy=cls(factory.encoder,factory.dynamics,head,cal,config).to(device)
        sources={'factored':factory.sources,'distance':dp,'calibration':cp,'code_hashes':_code(),'distance_config':head.config(),'calibration_config':c['config'],'distance_sources':d['sources'],'calibration_binding':c['source_binding']}
        policy.sources=sources;policy.sources['weights_sha256']=state_digest(policy.state_dict())
        for record in (dp,cp):
            if _digest(record['path'])!=record['sha256']:raise ValueError('checkpoint changed while loading')
        _verify_sources(factory.sources)
        if any(not torch.isfinite(p).all() for p in policy.parameters()):raise ValueError('nonfinite frozen weights')
        return policy.eval()

def save_search_policy_checkpoint(path,policy):
    if type(policy)!=StructuredSearchPolicy or not policy.sources:raise ValueError('source-bound search policy required')
    # Reconstruct and compare every frozen component before publishing references.
    checked=_reconstruct(policy.sources,policy.config(),'cpu')
    if state_digest(policy.state_dict())!=state_digest(checked.state_dict()):raise ValueError('frozen search weights changed')
    payload={'format':FORMAT,'config':policy.config(),'sources':copy.deepcopy(policy.sources),'parameters':policy.parameter_count(),'parameter_counts':policy.parameter_counts(),'official_inputs_used':False,'scope':'Learned expected completion before next life loss; calibration adequacy not asserted.'}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        os.link(temp,path)
    finally:Path(temp).unlink(missing_ok=True)
    return payload

def _reconstruct(sources,config,device):
    if not isinstance(sources,dict) or set(sources)!={'factored','distance','calibration','code_hashes','distance_config','calibration_config','distance_sources','calibration_binding','weights_sha256'}:raise ValueError('incomplete search provenance')
    if sources['code_hashes']!=_code():raise ValueError('search implementation drift')
    for record in (sources['distance'],sources['calibration']):
        if set(record)!={'path','sha256'} or _digest(record['path'])!=record['sha256']:raise ValueError('search component hash mismatch')
    art=sources['factored']['artifacts']
    policy=StructuredSearchPolicy.from_checkpoints(*(art[k]['path'] for k in ('world','visibility','dynamics')),sources['distance']['path'],sources['calibration']['path'],config,device)
    if policy.sources!=sources:raise ValueError('search reconstructed provenance mismatch')
    return policy

def load_search_policy_checkpoint(path,device='cpu'):
    saved,_=_read(path)
    if saved.get('format')!=FORMAT or saved.get('official_inputs_used') is not False:raise ValueError('invalid search policy format')
    policy=_reconstruct(saved['sources'],saved['config'],device)
    if saved.get('parameters')!=policy.parameter_count() or saved.get('parameter_counts')!=policy.parameter_counts():raise ValueError('search parameter count mismatch')
    return policy,saved
