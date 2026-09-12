"""Experimental writable spatial memory for the existing learned action readout.

Only public frozen fields enter forward. Reasoning loops are not environment
transitions. Static and evolving arms have identical parameter layouts; static
bypasses the workspace, so its workspace parameters intentionally get no gradient.
Zero residual gates preserve a warmstarted readout exactly. At initialization
only these gates receive workspace gradients; interior weights learn once gates
move (and a freshly zero scorer must itself learn first).
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .structured_policy import StructuredPolicyConfig, StructuredPolicyReadout, _fields


@dataclass(frozen=True)
class WorkspacePolicyConfig(StructuredPolicyConfig):
    memory_mode: str = 'evolving'
    checkpoint_workspace: bool = False

    def __post_init__(self):
        super().__post_init__()
        if type(self.checkpoint_workspace) is not bool:
            raise ValueError('checkpoint_workspace must be bool')
        if self.memory_mode not in ('static', 'evolving'):
            raise ValueError('memory_mode must be static or evolving')


class RecalledWorkspaceBlock(nn.Module):
    """One shared pre-norm attention/MLP block with original-memory recall."""
    def __init__(self, heads, expansion):
        super().__init__()
        self.recall = nn.Linear(192, 96, bias=False)
        self.attention_norm = nn.LayerNorm(96)
        self.attention = nn.MultiheadAttention(96, heads, dropout=0., batch_first=True)
        self.mlp_norm = nn.LayerNorm(96)
        self.mlp = nn.Sequential(nn.Linear(96, 96 * expansion), nn.GELU(),
                                 nn.Linear(96 * expansion, 96))
        self.attention_gate = nn.Parameter(torch.zeros(()))
        self.mlp_gate = nn.Parameter(torch.zeros(()))

    def forward(self, state, original):
        recalled = self.attention_norm(self.recall(torch.cat((state, original), -1)))
        state = state + self.attention_gate * self.attention(
            recalled, recalled, recalled, need_weights=False)[0]
        return state + self.mlp_gate * self.mlp(self.mlp_norm(state))


class StructuredWorkspaceReadout(StructuredPolicyReadout):
    """Same action-query scorer, with optionally writable [148,96] memory.

    ``loops=`` explicitly selects reasoning depth without modifying configuration.
    Use identical sampled depths (proposed 1/2/4) for both experimental arms.
    This module supplies neither a training schedule nor a playable factory.
    """
    def __init__(self, config=None):
        if config is None:
            config = WorkspacePolicyConfig()
        elif isinstance(config, dict):
            config = WorkspacePolicyConfig(**config)
        if not isinstance(config, WorkspacePolicyConfig):
            raise TypeError('expected WorkspacePolicyConfig or dict')
        base = asdict(config)
        base.pop('memory_mode')
        base.pop('checkpoint_workspace')
        # Initialize every inherited weight before drawing any new randomness.
        super().__init__(StructuredPolicyConfig(**base))
        self.cfg = config
        self.workspace = RecalledWorkspaceBlock(config.heads, config.expansion)
        if config.memory_mode == 'static':
            self.workspace.requires_grad_(False)

    @classmethod
    def from_readout(cls, readout, *, evolving=True, checkpoint_workspace=False):
        if type(evolving) is not bool:
            raise ValueError('evolving must be bool')
        model = cls({**readout.config(), 'memory_mode': 'evolving' if evolving else 'static',
                     'checkpoint_workspace': checkpoint_workspace})
        model.to(device=readout.position.device, dtype=readout.position.dtype)
        return model.load_from(readout)

    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def load_from(self, readout):
        """Strict legacy warmstart; requires an untouched neutral workspace.

        Copy all existing weights. No silent mode/shape/depth migration. To choose
        a different depth, use the explicit forward override after warmstarting.
        """
        if type(readout) is not StructuredPolicyReadout:
            raise TypeError('warmstart requires an original StructuredPolicyReadout')
        expected = asdict(self.cfg)
        expected.pop('memory_mode')
        expected.pop('checkpoint_workspace')
        if readout.config() != expected:
            raise ValueError('legacy readout configuration mismatch')
        if self.workspace.attention_gate.detach().item() != 0 or self.workspace.mlp_gate.detach().item() != 0:
            raise ValueError('warmstart requires neutral workspace gates')
        result = self.load_state_dict(readout.state_dict(), strict=False)
        if result.unexpected_keys or set(result.missing_keys) != {
                'workspace.' + key for key in self.workspace.state_dict()}:
            raise ValueError('incomplete legacy warmstart')
        return self

    def _score_depth(self, fields, queries, loops):
        original = self.source_norm(fields + self.position.to(fields.dtype)[None])
        memory = original
        source = queries
        state = queries
        for _ in range(loops):
            if self.cfg.memory_mode == 'evolving':
                memory = (checkpoint(self.workspace, memory, original, use_reentrant=False)
                          if self.cfg.checkpoint_workspace and self.training and torch.is_grad_enabled()
                          else self.workspace(memory, original))
            query = self.query_norm(self.query_recall(torch.cat((state, source), -1)))
            state = state + self.attention(query, memory, memory, need_weights=False)[0]
            state = state + self.mlp(self.mlp_norm(state))
        return self.scorer(state).squeeze(-1)

    def forward(self, fields, action_ids=None, *, loops=None):
        depth = self.cfg.loops if loops is None else loops
        if type(depth) is not int or depth < 1:
            raise ValueError('loops must be a positive integer')
        successors = self.cfg.mode == 'successors'
        _fields(fields, successors)
        if fields.device != self.position.device:
            raise ValueError('fields and readout must share device')
        if not successors:
            if action_ids is not None:
                raise ValueError('direct mode uses fixed action queries')
            return self._score_depth(fields, self.action_queries[None].expand(len(fields), -1, -1), depth)
        if action_ids is None:
            action_ids = torch.arange(4, device=fields.device)[None].expand(len(fields), -1)
        if (not isinstance(action_ids, torch.Tensor) or action_ids.shape != (len(fields), 4)
                or action_ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
                or action_ids.device != fields.device
                or not torch.equal(action_ids.long().sort(-1).values,
                                   torch.arange(4, device=fields.device)[None].expand(len(fields), -1))):
            raise ValueError('action_ids must be integer permutations of 0..3')
        query = self.action_queries[action_ids.long()].reshape(-1, 1, 96)
        return self._score_depth(fields.flatten(0, 1), query, depth).reshape(len(fields), 4)
