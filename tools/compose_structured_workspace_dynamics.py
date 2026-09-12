"""Dedicated public controller composition; generic model.py/UI do not load it.

The immutable evolving workspace head and encoder are reconstructed normally.
Only its local/global-glyph dynamics is replaced. No actual future or exact
teacher label enters forward. Re-loading means rebuilding from both checkpoints.
"""
import copy
import hashlib
import io
from pathlib import Path

import torch
from torch import nn

from pebby.agent.structured_factored_policy import ENCODER_FILES, canonical_metadata, state_digest
from pebby.agent.structured_local_glyph import LOCAL_GLYPH_FORMAT, LocalGlobalGlyphTransition
from pebby.agent.structured_workspace_controller import load_workspace_policy_checkpoint

FORMAT='pebby.workspace-dynamics-composition.v1'
WARM=Path('checkpoints/ls20-structured-workspace-comparison-600-evolving.pt')
WARM_SHA='6a5d7716fca4403048434a8100e26770dc26fb9d19ce66600fb3eccca7d22e0c'
INITIAL_SHA='dc676f5ac67cc7b5b3cfaf4b133cc6a9c9307d3ca09597f361f1447c9f72385c'


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def normalized_sources(sources):
    if not isinstance(sources,dict) or not sources:raise ValueError('source fingerprint map required')
    result={}
    for path,sha in sources.items():
        path=str(Path(path).resolve())
        if not isinstance(sha,str) or len(sha)!=64 or (path in result and result[path]!=sha):
            raise ValueError('conflicting or invalid source fingerprint')
        result[path]=sha
    return result


def validate_replacement(saved,warm,warm_path,*,allow_smoke=False):
    """Validate architecture and parent/cache lineage before accepting weights."""
    if (saved.get('format')!=LOCAL_GLYPH_FORMAT or saved.get('parameters')!=294664
            or saved.get('official_inputs_used') is not False or saved.get('frozen_encoder') is not True
            or saved.get('objective')!='paired_onpolicy_h1_dynamics_repair'
            or saved.get('arm') not in ('control','onpolicy')
            or type(saved.get('updates')) is not int or saved['updates']<1):
        raise ValueError('verified paired H1 local dynamics repair required')
    if saved.get('smoke') is True:
        if not allow_smoke or saved.get('batch_size') not in (2,4,8) or saved['updates']!=2:
            raise ValueError('disposable CPU smoke requires explicit allow_smoke')
    elif saved.get('smoke') is not False or saved.get('batch_size')!=1024:
        raise ValueError('production B1024 and explicit smoke marker required')
    for name in ('selection_sha256','used_state_rows_sha256'):
        value=saved.get(name)
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('invalid selection/state provenance SHA')
    if (saved.get('initialize_sha256')!=INITIAL_SHA or saved.get('warmstart_workspace_sha256')!=WARM_SHA
            or Path(saved.get('frozen_workspace_readout','')).resolve()!=Path(warm_path).resolve()):
        raise ValueError('repair warm-parent binding mismatch')
    sources=normalized_sources(saved.get('sources'))
    initial=Path(saved.get('initialize','')).resolve()
    if sources.get(str(initial))!=INITIAL_SHA or digest(initial)!=INITIAL_SHA:
        raise ValueError('initial dynamics source mismatch')
    if sources.get(str(Path(warm_path).resolve()))!=WARM_SHA:
        raise ValueError('warm workspace absent from repair sources')
    if warm.sources['artifacts']['dynamics']['sha256']!=INITIAL_SHA:
        raise ValueError('warm actor uses different parent dynamics')
    if (saved.get('frozen_encoder_state_sha256')!=state_digest(warm.encoder.state_dict())
            or saved.get('frozen_workspace_state_sha256')!=state_digest(warm.readout.state_dict())):
        raise ValueError('repair frozen head/encoder state binding mismatch')
    if saved.get('config')!=warm.dynamics.config():raise ValueError('replacement dynamics config mismatch')
    if set(saved.get('weights',{}))!=set(warm.dynamics.state_dict()):raise ValueError('replacement state keys mismatch')
    for key,value in saved['weights'].items():
        reference=warm.dynamics.state_dict()[key]
        if value.shape!=reference.shape or value.dtype!=reference.dtype or not torch.isfinite(value).all():
            raise ValueError('replacement state shape/dtype/finiteness mismatch')
    for name in (*ENCODER_FILES,'structured_transition.py','structured_global_glyph.py','structured_local_glyph.py','structured_workspace_controller.py'):
        path=Path('pebby/agent')/name
        if sources.get(str(path.resolve()))!=digest(path):raise ValueError(f'repair implementation source mismatch: {name}')
    manifests=saved.get('cache_manifests')
    if not isinstance(manifests,dict) or not {'onpolicy_train','legacy_train','legacy_additional_train','validation'}<=set(manifests):
        raise ValueError('repair cache lineage incomplete')
    cache_paths={}
    import json
    actual_encoder=canonical_metadata(warm.encoder.metadata())
    candidates=[Path(p) for p in sources if Path(p).name=='manifest.json']
    for name,manifest in manifests.items():
        if manifest.get('status')!='complete' or manifest.get('source')!='generated_only':
            raise ValueError('unverified cache lineage')
        expected_split='validation' if name=='validation' else 'train'
        if manifest.get('split')!=expected_split:raise ValueError('cache lineage split mismatch')
        encoded=canonical_metadata(manifest.get('field_encoder',{}))
        if any(encoded.get(k)!=v for k,v in actual_encoder.items()):raise ValueError('replacement/public encoder mismatch')
        matches=[]
        for path in candidates:
            if path.is_file() and json.loads(path.read_text())==manifest:matches.append(path)
        if not matches:raise ValueError(f'cache manifest absent from source lineage: {name}')
        path=matches[0]
        if digest(path)!=sources[str(path)]:raise ValueError('cache manifest source drift')
        for array,entry in manifest.get('arrays',{}).items():
            if sources.get(str(path.with_name(array+'.npy')))!=entry['sha256']:
                raise ValueError(f'cache array not bound by source lineage: {name}/{array}')
        cache_paths[name]={'path':str(path),'sha256':sources[str(path)]}
    # Revalidate runtime sources and manifest/checkpoint files. Training array
    # bytes are not scanned at inference; their exact declared hashes are bound
    # above to source manifests and the unchanged-source training checkpoint.
    runtime={p:h for p,h in sources.items() if Path(p).suffix in ('.py','.c','.pt') or Path(p).name=='manifest.json'}
    for path,sha in runtime.items():
        if digest(path)!=sha:raise ValueError(f'repair source changed: {path}')
    return runtime,cache_paths


class ComposedWorkspacePolicy(nn.Module):
    def __init__(self,warm,dynamics,provenance):
        super().__init__();self.encoder=warm.encoder;self.readout=warm.readout;self.dynamics=dynamics
        self._config=copy.deepcopy(warm.config());self.provenance=copy.deepcopy(provenance)
        self.train(False)
    def config(self):return copy.deepcopy(self._config)
    def parameter_count(self):return sum(p.numel() for p in self.parameters())
    def train(self,mode=True):
        super().train(False);self.requires_grad_(False);return self
    @torch.inference_mode()
    def successor_fields(self,current):
        if current.ndim!=3 or current.shape[1:]!=(148,96):raise ValueError('current public fields[B,148,96] required')
        output=[]
        for first in range(0,len(current)*4,128):
            ids=torch.arange(first,min(first+128,len(current)*4),device=current.device)
            output.append(self.dynamics.predict(current[ids//4],ids%4))
        return torch.cat(output).reshape(len(current),4,148,96)
    @torch.inference_mode()
    def forward(self,frames,history_valid=None,previous_actions=None):
        current=self.encoder(frames,history_valid,previous_actions)
        return self.readout(self.successor_fields(current.float()))


def load_composition(replacement,expected_sha,device='cpu',*,warm_path=WARM,allow_smoke=False):
    replacement=Path(replacement);warm_path=Path(warm_path)
    if digest(warm_path)!=WARM_SHA:raise ValueError('exact immutable evolving600 workspace required')
    raw=replacement.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha:raise ValueError('explicit replacement SHA mismatch')
    saved=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    warm,info=load_workspace_policy_checkpoint(warm_path,'cpu',loops=2)
    if warm.config().get('mode')!='successors' or info.get('readout_config',{}).get('memory_mode')!='evolving':
        raise ValueError('evolving successor workspace required')
    frozen={'encoder':state_digest(warm.encoder.state_dict()),'head':state_digest(warm.readout.state_dict())}
    runtime,caches=validate_replacement(saved,warm,warm_path,allow_smoke=allow_smoke)
    model=LocalGlobalGlyphTransition(saved['config']);model.load_state_dict(saved['weights'],strict=True)
    if model.parameter_count()!=294664:raise ValueError('replacement parameter count mismatch')
    runtime.update({str(Path(__file__).resolve()):digest(__file__),str(warm_path.resolve()):WARM_SHA,
                    str(replacement.resolve()):expected_sha})
    provenance={'format':FORMAT,'workspace':{'path':str(warm_path.resolve()),'sha256':WARM_SHA},
        'replacement':{'path':str(replacement.resolve()),'sha256':expected_sha,'format':saved['format'],
            'arm':saved['arm'],'updates':saved['updates'],'batch_size':saved['batch_size'],
            'parameters':294664,'selection_sha256':saved['selection_sha256'],
            'used_state_rows_sha256':saved['used_state_rows_sha256'],'smoke':saved['smoke']},
        'parent_dynamics_sha256':INITIAL_SHA,'frozen_state_sha256':frozen,
        'replacement_state_sha256':state_digest(model.state_dict()),'cache_lineage':caches,
        'runtime_source_hashes':runtime,'dedicated_loader_only':True,'actual_future_inputs':False}
    policy=ComposedWorkspacePolicy(warm,model,provenance).to(device).eval()
    if state_digest(policy.encoder.state_dict())!=frozen['encoder'] or state_digest(policy.readout.state_dict())!=frozen['head']:
        raise ValueError('frozen component changed during composition')
    for path,sha in runtime.items():
        if digest(path)!=sha:raise ValueError('source changed during composition')
    provenance['parameters']=policy.parameter_count()
    policy.provenance=copy.deepcopy(provenance)
    return policy,provenance


@torch.inference_mode()
def public_wiring_check(policy,frames,valid,actions):
    current=policy.encoder(frames,valid,actions)
    predicted=torch.stack([policy.dynamics.predict(current,torch.full((len(current),),action,
        dtype=torch.long,device=current.device)) for action in range(4)],dim=1)
    batched=policy.successor_fields(current)
    # Different GEMM batch sizes can change FP32 roundoff, so compare fields
    # tightly while demanding identical action order and argmax decisions.
    torch.testing.assert_close(batched,predicted,rtol=2e-5,atol=2e-5)
    expected=policy.readout(predicted)
    actual=policy(frames,history_valid=valid,previous_actions=actions)
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-5)
    if not torch.equal(actual.argmax(-1),expected.argmax(-1)):raise ValueError('manual/public action mismatch')
    return {'public_manual_logits_equal':torch.equal(actual,expected),
        'public_manual_max_abs':float((actual-expected).abs().max()),
        'independent_four_actions_checked':True,'argmax_equal':True,'batch':len(frames),'actions':4}
