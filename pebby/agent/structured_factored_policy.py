"""Playable public-H8 policy over frozen global/local factored successors.

This separate checkpoint format does not modify the earlier base-field policy.
Only the existing shared action readout is trained; no actual futures or labels
are accepted by the public inference interface.
"""
import copy
import hashlib
import io
from pathlib import Path
import torch
from .structured_policy import StructuredFieldPolicy,StructuredPolicyConfig,_config,_digest
from .structured_field import load_structured_field_encoder
from .structured_global_glyph import GlobalGlyphTransition,GLOBAL_GLYPH_FORMAT
from .structured_local_glyph import LocalGlobalGlyphTransition,LOCAL_GLYPH_FORMAT

FACTORED_POLICY_FORMAT='pebby.structured-factored-field-policy.v1'
ENCODER_FILES=('structured_field.py','world_model.py','world_readout.py','world_grounding.py',
               'world_rollout.py','cell_appearance.py','cell_appearance_dense.py','glyph_model.py','cell_visibility.py')


def state_digest(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        value=value.detach().cpu().contiguous()
        h.update(repr((key,str(value.dtype),tuple(value.shape))).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def canonical_metadata(value, key=None):
    if isinstance(value,dict):
        return {(str(Path(k).resolve()) if key in ('code_hashes','checkpoint_hashes','source_hashes') else k):canonical_metadata(v,k) for k,v in value.items()}
    if isinstance(value,list):return [canonical_metadata(v) for v in value]
    if isinstance(value,str) and key in ('path','checkpoint','bank','proof','bound_world_checkpoint'):
        return str(Path(value).resolve())
    return value


def _code(local):
    names=(*ENCODER_FILES,'structured_policy.py','structured_factored_policy.py','structured_transition.py','structured_global_glyph.py')
    if local:names+=('structured_local_glyph.py',)
    return {str(Path(__file__).with_name(name).resolve()):_digest(Path(__file__).with_name(name)) for name in names}


def _verify_sources(sources):
    if not isinstance(sources,dict) or set(sources)!={'artifacts','encoder_metadata','dynamics_metadata','code_hashes','encoder_state_sha256','dynamics_state_sha256'}:
        raise ValueError('incomplete factored policy provenance')
    artifacts=sources['artifacts']
    if not isinstance(artifacts,dict) or set(artifacts)!={'world','visibility','dynamics'}:raise ValueError('three source checkpoints required')
    for record in artifacts.values():
        if set(record)!={'path','sha256'} or _digest(record['path'])!=record['sha256']:raise ValueError('bound checkpoint hash mismatch')
    fmt=sources['dynamics_metadata'].get('format')
    if fmt not in (GLOBAL_GLYPH_FORMAT,LOCAL_GLYPH_FORMAT):raise ValueError('unsupported dynamics format')
    if sources['code_hashes']!=_code(fmt==LOCAL_GLYPH_FORMAT):raise ValueError('complete bound code hashes mismatch')


class StructuredFactoredPolicy(StructuredFieldPolicy):
    def __init__(self,encoder,dynamics,config=None,*,sources=None):
        cfg=_config({'mode':'successors'} if config is None else config)
        if cfg.mode!='successors' or type(dynamics) not in (GlobalGlyphTransition,LocalGlobalGlyphTransition):
            raise ValueError('factored policy requires global/local successors mode')
        super().__init__(encoder,dynamics,cfg,sources=sources)

    @classmethod
    def from_checkpoints(cls,world_checkpoint,visibility_checkpoint,dynamics_checkpoint,config=None,device='cpu'):
        cfg=_config({'mode':'successors'} if config is None else config)
        if cfg.mode!='successors':raise ValueError('factored policy supports successors only')
        paths={k:Path(p).resolve() for k,p in [('world',world_checkpoint),('visibility',visibility_checkpoint),('dynamics',dynamics_checkpoint)]}
        artifacts={k:{'path':str(p),'sha256':_digest(p)} for k,p in paths.items()}
        raw=paths['dynamics'].read_bytes()
        if hashlib.sha256(raw).hexdigest()!=artifacts['dynamics']['sha256']:raise ValueError('dynamics checkpoint changed')
        saved=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        classes={GLOBAL_GLYPH_FORMAT:GlobalGlyphTransition,LOCAL_GLYPH_FORMAT:LocalGlobalGlyphTransition}
        if saved.get('format') not in classes:raise ValueError('unsupported factored dynamics format')
        if saved.get('official_inputs_used') is not False:raise ValueError('generated-only dynamics attestation required')
        code=_code(saved['format']==LOCAL_GLYPH_FORMAT)
        encoder=load_structured_field_encoder(paths['world'],paths['visibility'],device=device)
        manifests=saved.get('cache_manifests')
        if not isinstance(manifests,dict) or not manifests:raise ValueError('dynamics lacks encoder cache provenance')
        bound=None
        for manifest in manifests.values():
            if manifest.get('source')!='generated_only' or manifest.get('status')!='complete' or manifest.get('split') not in ('train','validation'):
                raise ValueError('complete generated cache provenance required')
            current=manifest.get('field_encoder')
            if not isinstance(current,dict) or any(canonical_metadata(current).get(k)!=v for k,v in canonical_metadata(encoder.metadata()).items()):raise ValueError('dynamics encoder metadata mismatch')
            if bound is not None and current!=bound:raise ValueError('inconsistent dynamics encoder provenance')
            bound=current
            encoded_code=current.get('code_hashes',{})
            expected={str(Path(__file__).with_name(name).resolve()) for name in ENCODER_FILES}
            if {str(Path(p).resolve()) for p in encoded_code}!=expected:raise ValueError('complete nine encoder code hashes required')
            for p,h in encoded_code.items():
                if _digest(p)!=h:raise ValueError('encoder code drift')
            cp=current.get('checkpoint_hashes',{})
            if {str(Path(p).resolve()):h for p,h in cp.items()}!={str(paths[k]):artifacts[k]['sha256'] for k in ('world','visibility')}:raise ValueError('encoder checkpoint hashes mismatch')
        dynamics=classes[saved['format']](saved['config']).to(device)
        dynamics.load_state_dict(saved['weights'],strict=True)
        if saved.get('parameters')!=dynamics.parameter_count():raise ValueError('dynamics parameter count mismatch')
        for name in ('structured_transition.py','structured_global_glyph.py')+ (('structured_local_glyph.py',) if saved['format']==LOCAL_GLYPH_FORMAT else ()):
            wanted=Path(__file__).with_name(name).resolve()
            matches=[h for p,h in saved.get('sources',{}).items() if Path(p).resolve()==wanted]
            if matches!=[_digest(wanted)]:raise ValueError('dynamics implementation source binding mismatch')
        sources=dict(artifacts=artifacts,encoder_metadata=canonical_metadata(encoder.metadata()),dynamics_metadata={'format':saved['format'],'config':dynamics.config(),'parameters':dynamics.parameter_count(),'field_encoder':canonical_metadata(bound)},code_hashes=code,encoder_state_sha256=state_digest(encoder.state_dict()),dynamics_state_sha256=state_digest(dynamics.state_dict()))
        _verify_sources(sources)
        return cls(encoder,dynamics,cfg,sources=sources).to(device).eval()


def save_factored_policy_checkpoint(path,policy,training_provenance=None):
    if type(policy)!=StructuredFactoredPolicy:raise ValueError('factored policy required')
    _verify_sources(policy.sources)
    if state_digest(policy.encoder.state_dict())!=policy.sources['encoder_state_sha256'] or state_digest(policy.dynamics.state_dict())!=policy.sources['dynamics_state_sha256']:raise ValueError('frozen source weights changed')
    weights={k:v.detach().cpu().clone() for k,v in policy.readout.state_dict().items()}
    torch.save(dict(format=FACTORED_POLICY_FORMAT,config=policy.config(),parameters=policy.parameter_count(),parameter_counts=policy.parameter_counts(),sources=copy.deepcopy(policy.sources),readout_weights=weights,readout_sha256=state_digest(weights),training_provenance=copy.deepcopy(training_provenance)),Path(path))


def load_factored_policy_checkpoint(path,device='cpu'):
    raw=Path(path).read_bytes();saved=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    if saved.get('format')!=FACTORED_POLICY_FORMAT:raise ValueError('wrong factored policy format')
    sources=saved['sources'];_verify_sources(sources)
    if state_digest(saved['readout_weights'])!=saved.get('readout_sha256'):raise ValueError('readout weight hash mismatch')
    artifacts=sources['artifacts']
    policy=StructuredFactoredPolicy.from_checkpoints(*(artifacts[k]['path'] for k in ('world','visibility','dynamics')),config=saved['config'],device=device)
    if policy.sources!=sources:raise ValueError('reconstructed factored source mismatch')
    policy.readout.load_state_dict(saved['readout_weights'],strict=True)
    if saved.get('parameter_counts')!=policy.parameter_counts() or saved.get('parameters')!=policy.parameter_count():raise ValueError('policy parameter counts mismatch')
    return policy.eval(),saved
