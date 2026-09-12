"""Optional learned recall of the public initial observation, not a world oracle.

Initial memory can be fog-limited. It stays fixed through imagined transitions;
no reset flag, coordinate, engine target, or hard-coded reset rule is consumed.
Base construction precedes every new parameter so same-seed base weights match.
"""
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from .structured_transition import (StructuredTransition, StructuredTransitionConfig,
                                    FIELD_TOKENS, FIELD_WIDTH, _field, _actions)

FORMAT = 'pebby.structured-recall-transition.v1'


@dataclass(frozen=True)
class StructuredRecallConfig(StructuredTransitionConfig):
    memory_heads: int = 4

    def __post_init__(self):
        super().__post_init__()
        if FIELD_WIDTH % self.memory_heads:
            raise ValueError('memory_heads must divide field width96')


class StructuredRecallTransition(StructuredTransition):
    """Same public current field/action API, plus required public initial_field."""
    def __init__(self, config=None, **overrides):
        if config is None:
            config=StructuredRecallConfig(**overrides)
        elif isinstance(config,dict):
            config=StructuredRecallConfig(**(config|overrides))
        elif overrides or not isinstance(config,StructuredRecallConfig):
            raise ValueError('pass a recall config or keyword overrides')
        base_keys=StructuredTransitionConfig.__dataclass_fields__
        super().__init__(StructuredTransitionConfig(**{k:v for k,v in asdict(config).items() if k in base_keys}))
        self.cfg=config
        self.memory_position=nn.Parameter(torch.empty(FIELD_TOKENS,FIELD_WIDTH))
        self.memory_type=nn.Parameter(torch.empty(FIELD_WIDTH))
        nn.init.normal_(self.memory_position,std=.02)
        nn.init.normal_(self.memory_type,std=.02)
        self.memory_norm=nn.LayerNorm(FIELD_WIDTH)
        self.memory_query_norm=nn.LayerNorm(FIELD_WIDTH)
        self.memory_attention=nn.MultiheadAttention(FIELD_WIDTH,config.memory_heads,dropout=0.,batch_first=True)
        self.memory_output=nn.Linear(FIELD_WIDTH,FIELD_WIDTH,bias=False)
        nn.init.zeros_(self.memory_output.weight)

    def predict(self,field,actions,initial_field,*,loops=None):
        _field(field);_field(initial_field)
        if initial_field.shape!=field.shape or initial_field.device!=field.device:
            raise ValueError('initial_field must match current field shape and device')
        actions=_actions(actions,len(field),field.device)
        depth=self.cfg.loops if loops is None else loops
        if type(depth) is not int or depth<1:raise ValueError('loops must be a positive integer')
        source=field+self.position.to(field.dtype)[None]+self.action_embedding(actions).to(field.dtype)[:,None]
        memory=self.memory_norm(initial_field.to(field.dtype)+self.memory_position.to(field.dtype)[None]
                                +self.memory_type.to(field.dtype)[None,None])
        state=source
        for _ in range(depth):
            state=self.block(state,source)
            recalled=self.memory_attention(self.memory_query_norm(state),memory,memory,need_weights=False)[0]
            state=state+self.memory_output(recalled)
        return field+self.output(state)

    def forward(self,field,actions,initial_field,*,loops=None):
        predicted=self.predict(field,actions,initial_field,loops=loops)
        events=self.event_head(torch.cat((self.readout.summary(field),self.readout.summary(predicted),
                                          self.action_embedding(actions.long())),-1))
        return dict(field=predicted,readout=self.readout(predicted),
                    events=dict(lost_life_logits=events[:,0],terminal_logits=events[:,1],won_logits=events[:,2]))

    def rollout(self,field,actions,initial_field,*,loops=None):
        _field(field);_field(initial_field)
        actions=_actions(actions,len(field),field.device,ndim=2)
        if actions.shape[1]<1:raise ValueError('rollout requires at least one action')
        outputs=[]
        for step in range(actions.shape[1]):
            output=self(field,actions[:,step],initial_field,loops=loops)
            outputs.append(output);field=output['field']
        return dict(fields=torch.stack([x['field'] for x in outputs],1),
                    readout={k:torch.stack([x['readout'][k] for x in outputs],1) for k in outputs[0]['readout']},
                    events={k:torch.stack([x['events'][k] for x in outputs],1) for k in outputs[0]['events']})


def save_recall_checkpoint(path,model):
    """Small explicit format; provenance/training reports remain caller-owned."""
    if not isinstance(model,StructuredRecallTransition):raise TypeError('need StructuredRecallTransition')
    torch.save(dict(format=FORMAT,config=model.config(),parameters=model.parameter_count(),
                    weights={k:v.detach().cpu() for k,v in model.state_dict().items()}),Path(path))


def load_recall_checkpoint(path,device='cpu'):
    checkpoint=torch.load(Path(path),map_location='cpu',weights_only=True)
    if checkpoint.get('format')!=FORMAT:raise ValueError('unsupported structured recall checkpoint format')
    model=StructuredRecallTransition(checkpoint['config'])
    if checkpoint.get('parameters')!=model.parameter_count():raise ValueError('recall parameter count mismatch')
    model.load_state_dict(checkpoint['weights'],strict=True)
    return model.to(device).eval(),checkpoint
