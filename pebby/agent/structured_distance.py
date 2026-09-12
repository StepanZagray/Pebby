"""Learned field-only exact-distance distribution; no planner or event penalty.

Live life-reset successors can have finite distance. A terminal failure is
unreachable; a win has distance zero. A zero action mask is never used alone
as a failure label. Distance support comes explicitly from TRAIN data.
"""
from dataclasses import asdict,dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .structured_policy import _fields

INTEGER=(torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64)


@dataclass(frozen=True)
class DistanceConfig:
    max_distance:int
    loops:int=2
    heads:int=4
    expansion:int=2
    def __post_init__(self):
        if type(self.max_distance)!=int or self.max_distance<1:raise ValueError('positive explicit TRAIN max_distance required')
        if any(type(x)!=int or x<1 for x in (self.loops,self.heads,self.expansion)) or 96%self.heads:raise ValueError('invalid loop/head/expansion config')


class StructuredDistanceReadout(nn.Module):
    """One learned query, shared across refinement loops and all states."""
    def __init__(self,config=None,**kwargs):
        super().__init__()
        if config is None:config=DistanceConfig(**kwargs)
        elif isinstance(config,dict):config=DistanceConfig(**(config|kwargs))
        elif not isinstance(config,DistanceConfig) or kwargs:raise ValueError('invalid distance config')
        self.cfg=config
        self.position=nn.Parameter(torch.randn(148,96)*.02)
        self.query=nn.Parameter(torch.randn(1,96)*.02)
        self.source_norm=nn.LayerNorm(96);self.query_recall=nn.Linear(192,96,bias=False);self.query_norm=nn.LayerNorm(96)
        self.attention=nn.MultiheadAttention(96,config.heads,dropout=0.,batch_first=True)
        self.mlp_norm=nn.LayerNorm(96);self.mlp=nn.Sequential(nn.Linear(96,96*config.expansion),nn.GELU(),nn.Linear(96*config.expansion,96))
        self.output=nn.Linear(96,config.max_distance+2)
    def config(self):return asdict(self.cfg)
    def parameter_count(self):return sum(p.numel() for p in self.parameters())
    def forward(self,fields):
        _fields(fields)
        if fields.device!=self.position.device:raise ValueError('fields/readout device mismatch')
        memory=self.source_norm(fields+self.position[None]);source=self.query[None].expand(len(fields),-1,-1);state=source
        for _ in range(self.cfg.loops):
            query=self.query_norm(self.query_recall(torch.cat((state,source),-1)))
            state=state+self.attention(query,memory,memory,need_weights=False)[0]
            state=state+self.mlp(self.mlp_norm(state))
        return self.output(state[:,0])


def _integers(value,name):
    value=torch.as_tensor(value)
    if value.dtype not in INTEGER:raise ValueError(f'{name} must be integer')
    return value.long()


def _flags(value,shape,name):
    value=torch.as_tensor(value)
    if value.shape!=shape or (value.dtype!=torch.bool and value.dtype not in INTEGER) or not ((value==0)|(value==1)).all():raise ValueError(f'invalid {name} flags')
    return value.bool()


def next_distance_labels(distances,terminal,won):
    """Exact branch labels; loss of life alone does not change finite distance."""
    d=_integers(distances,'distances');t=_flags(terminal,d.shape,'terminal');w=_flags(won,d.shape,'won')
    if t.device!=d.device or w.device!=d.device:raise ValueError('label devices differ')
    if (d < -1).any() or (w&~t).any() or (w&(d!=0)).any() or ((t&~w)&(d!=-1)).any() or ((d==0)&~w).any():raise ValueError('inconsistent win/terminal/distance labels')
    return d


def current_distance_labels(optimal,distances,terminal,won,lost_life,*,allow_unreachable=False):
    """Infer a reachable current-state distance from its complete optimal mask.

    By default reject zero masks. Complete-oracle transition sources can opt in
    to unreachable current labels only when ALL actions are unreachable or lose
    a life; a zero mask alone never establishes reachability.
    """
    d=next_distance_labels(distances,terminal,won);mask=_integers(optimal,'optimal')
    if d.ndim!=2 or d.shape[1]!=4 or mask.shape!=d.shape[:1] or mask.device!=d.device or ((mask<(0 if allow_unreachable else 1))|(mask>15)).any():raise ValueError('reachable current masks[B] and branches[B,4] required')
    lost=_flags(lost_life,d.shape,'lost_life')
    if lost.device!=d.device:raise ValueError('label devices differ')
    bits=(mask[:,None]&(1<<torch.arange(4,device=d.device)))!=0
    best=torch.where(bits,d,torch.iinfo(torch.long).max).min(-1).values
    if (best<0).any() or (bits&((d!=best[:,None])|lost)).any():raise ValueError('optimal branches disagree or lose life')
    # Complete masks must include every tied best no-life-loss alternative.
    eligible=(d>=0)&~lost
    true_best=d.masked_fill(~eligible,torch.iinfo(torch.long).max).min(-1).values
    if not torch.equal(best,true_best) or not torch.equal(bits,eligible&(d==best[:,None])):raise ValueError('incomplete/inconsistent optimal set')
    return torch.where(mask != 0, best, torch.full_like(best, -2)) + 1


def distance_targets(distances,max_distance):
    DistanceConfig(max_distance)
    d=_integers(distances,'distances')
    if ((d < -1)|(d>max_distance)).any():raise ValueError('distance outside explicit TRAIN support; no silent clipping')
    return torch.where(d<0,max_distance+1,d)


def discounted_value(logits,gamma=.99):
    if not isinstance(gamma,(float,int)) or isinstance(gamma,bool) or not math.isfinite(gamma) or not 0<gamma<=1:raise ValueError('gamma must be in (0,1]')
    if logits.ndim!=2 or len(logits)==0 or logits.shape[1]<3 or not logits.is_floating_point() or not torch.isfinite(logits).all():raise ValueError('finite distance logits[B,D+2] required')
    support=gamma**torch.arange(logits.shape[1]-1,device=logits.device,dtype=torch.float32)
    return (logits.float().softmax(-1)[:,:-1]*support).sum(-1)


def distance_loss(logits,distances,max_distance):
    targets=distance_targets(distances,max_distance).to(logits.device)
    if logits.ndim!=2 or len(logits)==0 or logits.shape!=(targets.numel(),max_distance+2) or targets.ndim!=1 or not logits.is_floating_point() or not torch.isfinite(logits).all():raise ValueError('distance logits/labels shape mismatch')
    loss=F.cross_entropy(logits.float(),targets);truth=targets==max_distance+1;guess=logits.argmax(-1);pred_bad=guess==max_distance+1;finite=~truth
    # Finite-distance MAE uses conditional finite expectation; reachability is
    # scored independently, so unreachable probability cannot hide in MAE.
    conditional=logits.float()[:,:-1].softmax(-1);expected=(conditional*torch.arange(max_distance+1,device=logits.device)).sum(-1)
    errors=(expected-targets).abs();metrics={'count':len(targets),'finite_count':int(finite.sum()),'finite_mae_sum':float(errors[finite].detach().sum()),'finite_within1_count':int((errors[finite]<=1).sum()),'unreachable_tp':int((truth&pred_bad).sum()),'unreachable_fp':int((~truth&pred_bad).sum()),'unreachable_fn':int((truth&~pred_bad).sum()),'unreachable_tn':int((~truth&~pred_bad).sum())}
    return {'loss':loss,'metrics':metrics}
