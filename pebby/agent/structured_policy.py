"""Learned spatial action readouts of frozen public fields, independent of planners.

The neutral zero scorer initially emits four equal logits. It is not a trained
controller: the scorer learns first, then gradients reach the shared attention.
"""
from dataclasses import asdict, dataclass
import copy
import hashlib
from pathlib import Path

import torch
from torch import nn

from .structured_field import load_structured_field_encoder
from .structured_transition import StructuredTransition

STRUCTURED_POLICY_FORMAT = 'pebby.structured-field-policy.v1'


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


@dataclass(frozen=True)
class StructuredPolicyConfig:
    mode: str = 'direct'
    loops: int = 2
    heads: int = 4
    expansion: int = 2
    successor_batch_size: int = 128
    history: int = 8
    architecture: str = 'structured'

    def __post_init__(self):
        if self.mode not in ('direct','successors'):
            raise ValueError('mode must explicitly be direct or successors')
        if self.architecture!='structured' or self.history!=8:
            raise ValueError('structured policy requires architecture structured and H8')
        for name in ('loops','heads','expansion','successor_batch_size'):
            value=getattr(self,name)
            if type(value)!=int or value<1:raise ValueError(f'{name} must be a positive integer')
        if 96%self.heads or self.successor_batch_size>1024:
            raise ValueError('heads must divide96 and successor batch must be<=1024')


def _config(config):
    if config is None:return StructuredPolicyConfig()
    if isinstance(config,dict):return StructuredPolicyConfig(**config)
    if not isinstance(config,StructuredPolicyConfig):raise TypeError('invalid policy config')
    return config


def _fields(fields, successor=False):
    shape=(4,148,96) if successor else (148,96)
    if (not isinstance(fields,torch.Tensor) or tuple(fields.shape[1:])!=shape
            or len(fields)==0 or not fields.is_floating_point()):
        raise ValueError(f'fields must be nonempty floating [B,{",".join(map(str,shape))}]')
    if not bool(torch.isfinite(fields).all()):raise ValueError('nonfinite policy fields')


class StructuredPolicyReadout(nn.Module):
    """Field-level trainable API: direct[B,148,96] or successors[B,4,148,96].

    The four outputs use the same recurrent block and scalar scorer. Successor
    mode scores each future independently with its associated learned action
    query; it receives no current-state features or other action's future.
    """
    def __init__(self,config=None):
        super().__init__();self.cfg=_config(config)
        self.position=nn.Parameter(torch.empty(148,96))
        self.action_queries=nn.Parameter(torch.empty(4,96))
        nn.init.normal_(self.position,std=.02);nn.init.normal_(self.action_queries,std=.02)
        self.source_norm=nn.LayerNorm(96)
        self.query_recall=nn.Linear(192,96,bias=False)
        self.query_norm=nn.LayerNorm(96)
        self.attention=nn.MultiheadAttention(96,self.cfg.heads,dropout=0.,batch_first=True)
        self.mlp_norm=nn.LayerNorm(96)
        self.mlp=nn.Sequential(nn.Linear(96,96*self.cfg.expansion),nn.GELU(),nn.Linear(96*self.cfg.expansion,96))
        self.scorer=nn.Linear(96,1)
        nn.init.zeros_(self.scorer.weight);nn.init.zeros_(self.scorer.bias)

    def config(self):return asdict(self.cfg)
    def parameter_count(self):return sum(p.numel() for p in self.parameters())

    def _score(self,fields,queries):
        memory=self.source_norm(fields+self.position.to(fields.dtype)[None])
        source=queries;state=queries
        for _ in range(self.cfg.loops):
            query=self.query_norm(self.query_recall(torch.cat((state,source),-1)))
            state=state+self.attention(query,memory,memory,need_weights=False)[0]
            state=state+self.mlp(self.mlp_norm(state))
        return self.scorer(state).squeeze(-1)

    def forward(self,fields,action_ids=None):
        successors=self.cfg.mode=='successors';_fields(fields,successors)
        if fields.device!=self.position.device:raise ValueError('fields and readout must share device')
        if not successors:
            if action_ids is not None:raise ValueError('direct mode uses its fixed four action queries')
            return self._score(fields,self.action_queries[None].expand(len(fields),-1,-1))
        if action_ids is None:
            action_ids=torch.arange(4,device=fields.device)[None].expand(len(fields),-1)
        if (not isinstance(action_ids,torch.Tensor) or action_ids.shape!=(len(fields),4)
                or action_ids.dtype not in (torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64)
                or action_ids.device!=fields.device
                or not torch.equal(action_ids.long().sort(-1).values,torch.arange(4,device=fields.device)[None].expand(len(fields),-1))):
            raise ValueError('action_ids must be an integer permutation0..3 for each successor row')
        queries=self.action_queries[action_ids.long()].reshape(-1,1,96)
        return self._score(fields.flatten(0,1),queries).reshape(len(fields),4)


class StructuredFieldPolicy(nn.Module):
    """Public H8 wrapper; only readout parameters are trainable.

    ``forward_fields`` trains from cached CURRENT fields without images. To
    train from precomputed successors, call the successor-mode ``readout`` on
    those fields directly. Both interfaces take no engine labels.
    """
    def __init__(self,encoder,dynamics=None,config=None,*,sources=None):
        super().__init__();self.cfg=_config(config)
        if self.cfg.mode=='successors' and not isinstance(dynamics,StructuredTransition):
            raise ValueError('successors mode requires frozen base StructuredTransition')
        if self.cfg.mode=='direct' and dynamics is not None:
            raise ValueError('direct mode must omit unused dynamics')
        self.encoder=encoder;self.dynamics=dynamics;self.readout=StructuredPolicyReadout(self.cfg)
        self.sources=copy.deepcopy(sources)
        self.train(False)

    def config(self):return asdict(self.cfg)
    def parameter_counts(self):
        encoder=sum(p.numel() for p in self.encoder.parameters())
        dynamics=0 if self.dynamics is None else sum(p.numel() for p in self.dynamics.parameters())
        readout=self.readout.parameter_count()
        return {'encoder':encoder,'dynamics':dynamics,'readout':readout,
                'trainable':sum(p.numel() for p in self.parameters() if p.requires_grad),
                'total':encoder+dynamics+readout}
    def parameter_count(self):return self.parameter_counts()['total']

    def train(self,mode=True):
        super().train(mode)
        self.encoder.eval().requires_grad_(False)
        if self.dynamics is not None:self.dynamics.eval().requires_grad_(False)
        return self

    @torch.no_grad()
    def successor_fields(self,fields):
        if self.cfg.mode!='successors':raise ValueError('direct policy has no successor dynamics')
        _fields(fields)
        flattened=[];size=self.cfg.successor_batch_size
        # Generate B*4 successors in bounded chunks; never materialize repeated
        # current fields for every action before selecting a chunk.
        for begin in range(0,len(fields)*4,size):
            ids=torch.arange(begin,min(begin+size,len(fields)*4),device=fields.device)
            predicted=self.dynamics.predict(fields[ids//4],ids%4)
            flattened.append(predicted.detach())
        return torch.cat(flattened).reshape(len(fields),4,148,96)

    def forward_fields(self,fields):
        _fields(fields)
        fields=fields.detach().to(device=self.readout.position.device,dtype=torch.float32)
        if self.cfg.mode=='direct':return self.readout(fields)
        return self.readout(self.successor_fields(fields))

    def forward(self,frames,history_valid=None,previous_actions=None):
        with torch.no_grad():fields=self.encoder(frames,history_valid,previous_actions)
        return self.forward_fields(fields)

    @classmethod
    def from_checkpoints(cls,world_checkpoint,visibility_checkpoint,dynamics_checkpoint=None,config=None,device='cpu'):
        cfg=_config(config)
        if (cfg.mode=='successors')!=(dynamics_checkpoint is not None):
            raise ValueError('only successors mode requires a dynamics checkpoint')
        paths={'world':Path(world_checkpoint),'visibility':Path(visibility_checkpoint)}
        if dynamics_checkpoint is not None:paths['dynamics']=Path(dynamics_checkpoint)
        artifacts={name:{'path':str(path),'sha256':_digest(path)} for name,path in paths.items()}
        encoder=load_structured_field_encoder(world_checkpoint,visibility_checkpoint,device=device)
        dynamics=None;dynamics_provenance=None
        if dynamics_checkpoint is not None:
            saved=torch.load(dynamics_checkpoint,map_location='cpu',weights_only=False)
            if saved.get('format')!='pebby.structured-transition.v1':
                raise ValueError('unsupported dynamics format; base StructuredTransition required')
            manifests=saved.get('cache_manifests')
            if not isinstance(manifests,dict) or not manifests:
                raise ValueError('dynamics checkpoint lacks field encoder provenance')
            for manifest in manifests.values():
                bound=manifest.get('field_encoder',{})
                actual=encoder.metadata()
                if any(bound.get(key)!=value for key,value in actual.items()):
                    raise ValueError('dynamics and public field encoder mismatch')
                for group in ('code_hashes','checkpoint_hashes'):
                    if not bound.get(group):raise ValueError('dynamics field provenance missing hashes')
                    if any(_digest(path)!=sha for path,sha in bound[group].items()):
                        raise ValueError('dynamics bound encoder source changed')
            dynamics=StructuredTransition(saved['config']).to(device)
            dynamics.load_state_dict(saved['weights'],strict=True)
            dynamics_provenance={'config':dynamics.config(),'parameters':dynamics.parameter_count(),
                                 'field_encoder':next(iter(manifests.values()))['field_encoder']}
        if any(_digest(record['path'])!=record['sha256'] for record in artifacts.values()):
            raise ValueError('source checkpoint changed while loading')
        code={str(Path(__file__)):_digest(__file__)}
        # Direct mode must bind the same complete perception implementation as
        # successor mode; it has no dynamics manifest to provide this guard.
        for file in ('structured_field.py','world_model.py','world_readout.py',
                     'world_grounding.py','world_rollout.py','cell_appearance.py',
                     'cell_appearance_dense.py','glyph_model.py','cell_visibility.py',
                     'structured_transition.py'):
            path=Path(__file__).with_name(file);code[str(path)]=_digest(path)
        sources={'artifacts':artifacts,'encoder_metadata':encoder.metadata(),
                 'dynamics_metadata':dynamics_provenance,'code_hashes':code}
        return cls(encoder,dynamics,cfg,sources=sources).to(device).eval()


def save_structured_policy_checkpoint(path,policy,training_provenance=None):
    if not isinstance(policy,StructuredFieldPolicy) or not policy.sources:
        raise ValueError('checkpoint requires a source-bound StructuredFieldPolicy')
    for record in policy.sources['artifacts'].values():
        if _digest(record['path'])!=record['sha256']:raise ValueError('bound checkpoint source changed')
    if any(_digest(p)!=sha for p,sha in policy.sources['code_hashes'].items()):
        raise ValueError('bound policy code changed')
    checkpoint={'format':STRUCTURED_POLICY_FORMAT,'config':policy.config(),
                'parameters':policy.parameter_count(),
                'parameter_counts':policy.parameter_counts(),'sources':copy.deepcopy(policy.sources),
                'readout_weights':{k:v.detach().cpu() for k,v in policy.readout.state_dict().items()},
                'training_provenance':copy.deepcopy(training_provenance),
                'scope':'Learned spatial readout of frozen public fields; no planner or engine labels at inference.'}
    torch.save(checkpoint,Path(path))


def load_structured_policy_checkpoint(path,device='cpu'):
    checkpoint=torch.load(Path(path),map_location='cpu',weights_only=True)
    if checkpoint.get('format')!=STRUCTURED_POLICY_FORMAT:
        raise ValueError('unsupported structured policy checkpoint format')
    sources=checkpoint['sources'];artifacts=sources['artifacts']
    for record in artifacts.values():
        if _digest(record['path'])!=record['sha256']:raise ValueError('policy bound checkpoint hash mismatch')
    if any(_digest(p)!=sha for p,sha in sources['code_hashes'].items()):
        raise ValueError('policy bound code hash mismatch')
    policy=StructuredFieldPolicy.from_checkpoints(artifacts['world']['path'],artifacts['visibility']['path'],
        artifacts.get('dynamics',{}).get('path'),checkpoint['config'],device)
    if policy.sources!=sources:raise ValueError('reconstructed policy source provenance mismatch')
    policy.readout.load_state_dict(checkpoint['readout_weights'],strict=True)
    if (policy.parameter_counts()!=checkpoint['parameter_counts']
            or checkpoint.get('parameters')!=policy.parameter_count()):
        raise ValueError('policy parameter counts mismatch')
    return policy.eval(),checkpoint


__all__=['STRUCTURED_POLICY_FORMAT','StructuredPolicyConfig','StructuredPolicyReadout',
         'StructuredFieldPolicy','save_structured_policy_checkpoint','load_structured_policy_checkpoint']
