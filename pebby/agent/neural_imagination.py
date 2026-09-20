"""Neural interpretation of autoregressive learned futures.

Four hard first actions each start one trajectory. A learned continuation
policy chooses subsequent actions; a reverse GRU and shared neural comparator
choose the real action. There is no tree search or hand-coded return backup.
Hard continuation choices are intentionally not differentiable: supervise the
continuation policy separately on observed TRAIN states. Dynamics gradients
can propagate through the chosen trajectory unless the caller freezes them.

This module supplies architecture, not trained or calibrated LS20 competence.
It accepts learned public fields only, never engine states or teacher futures.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .structured_policy import StructuredPolicyReadout, _fields


@dataclass(frozen=True)
class ImaginationConfig:
    horizon: int = 4
    hidden: int = 96
    heads: int = 4

    def __post_init__(self):
        if type(self.horizon) is not int or not 1 <= self.horizon <= 16:
            raise ValueError('horizon must be an integer in 1..16')
        if type(self.hidden) is not int or self.hidden < 1:
            raise ValueError('hidden must be positive')
        if type(self.heads) is not int or self.heads < 1 or 96 % self.heads:
            raise ValueError('heads must divide field width 96')


class NeuralImagination(nn.Module):
    """Reusable D(field, action) plus entirely neural trajectory selection.

    Dynamics must return StructuredTransition's field/events dictionary. It is
    registered unchanged: construction does not freeze or reinitialize it.
    Caller owns optimization, dynamics supervision, and runtime train/eval mode.
    No state persists between real game decisions in this first experiment.
    """

    def __init__(self, dynamics, config=None, *, continuation=None):
        super().__init__()
        self.cfg = ImaginationConfig(**config) if isinstance(config, dict) else (config or ImaginationConfig())
        if not isinstance(self.cfg, ImaginationConfig) or not isinstance(dynamics, nn.Module):
            raise ValueError('ImaginationConfig and a learned dynamics module required')
        self.dynamics = dynamics
        self.continuation = continuation if continuation is not None else StructuredPolicyReadout()
        self.position = nn.Parameter(torch.randn(148, 96) * .02)
        self.query = nn.Parameter(torch.randn(1, 1, 96) * .02)
        self.norm = nn.LayerNorm(96)
        self.attention = nn.MultiheadAttention(96, self.cfg.heads, batch_first=True, dropout=0.)
        self.trajectory = nn.GRUCell(99, self.cfg.hidden)
        # Root action IDs are deliberately absent; roots are mapped back to
        # canonical action order only after scoring their imagined consequences.
        self.scorer = nn.Sequential(nn.Linear(2 * self.cfg.hidden + 96, self.cfg.hidden),
                                    nn.GELU(), nn.Linear(self.cfg.hidden, 1, bias=False))

    def config(self):
        return asdict(self.cfg)

    def _summary(self, fields):
        memory = self.norm(fields + self.position[None].to(fields.dtype))
        return self.attention(self.query.expand(len(fields), -1, -1).to(fields.dtype),
                              memory, memory, need_weights=False)[0][:, 0]

    def imagine(self, fields, *, root_actions=None, horizon=None, tail_ablation=False):
        _fields(fields)
        depth = self.cfg.horizon if horizon is None else horizon
        if type(depth) is not int or not 1 <= depth <= 16:
            raise ValueError('horizon must be an integer in 1..16')
        batch = len(fields)
        if root_actions is None:
            root_actions = torch.arange(4, device=fields.device)[None].expand(batch, -1)
        expected = torch.arange(4, device=fields.device)[None].expand(batch, -1)
        if (not isinstance(root_actions, torch.Tensor) or root_actions.shape != (batch, 4)
                or root_actions.dtype != torch.long or root_actions.device != fields.device
                or not torch.equal(root_actions.sort(-1).values, expected)):
            raise ValueError('root_actions must be long [B,4] permutations of 0..3')
        current = fields[:, None].expand(-1, 4, -1, -1).reshape(batch * 4, 148, 96)
        actions = root_actions.reshape(-1)
        summaries, events, traces = [], [], []
        for step in range(depth):
            if step:
                # Discrete neural action selection, not a search/argmax over
                # predicted returns. CE on real states trains this proposal.
                with torch.no_grad():
                    proposal = self.continuation(current.detach())
                    if proposal.shape != (batch * 4, 4) or not torch.isfinite(proposal).all():
                        raise ValueError('continuation must return finite [B,4] logits')
                    actions = proposal.argmax(-1)
            prediction = self.dynamics(current, actions)
            following = prediction['field']
            _fields(following)
            if following.shape != current.shape or following.device != current.device:
                raise ValueError('dynamics must preserve field shape/device')
            event = torch.stack([prediction['events'][name + '_logits']
                                 for name in ('lost_life', 'terminal', 'won')], -1)
            if event.shape != (batch * 4, 3) or not torch.isfinite(event).all():
                raise ValueError('dynamics must return three finite event logits')
            summaries.append(self._summary(following))
            events.append(event.sigmoid())
            traces.append(actions.reshape(batch, 4))
            current = following
        # Learned interpretation includes possible terminal events. There is
        # no assertion that independent event probabilities are calibrated, or
        # that a predicted post-terminal field is a real reachable game state.
        if type(tail_ablation) is not bool:
            raise ValueError('tail_ablation must be boolean')
        if tail_ablation:
            # Diagnostic only: same dynamics calls, actions, horizon and GRU
            # computation; replace H2+ evidence with H1 without changing H1.
            summaries = [summaries[0]] * depth
            events = [events[0]] * depth
        hidden = fields.new_zeros(batch * 4, self.cfg.hidden)
        for summary, event in zip(reversed(summaries), reversed(events)):
            hidden = self.trajectory(torch.cat((summary, event), -1), hidden)
        roots = hidden.reshape(batch, 4, -1)
        pooled = roots.mean(1, keepdim=True).expand_as(roots)
        observed = self._summary(fields)[:, None].expand(-1, 4, -1)
        scores = self.scorer(torch.cat((roots, pooled, observed), -1)).squeeze(-1)
        logits = torch.zeros_like(scores).scatter(1, root_actions, scores)
        return dict(action_logits=logits, root_scores=scores,
                    root_actions=root_actions, imagined_actions=torch.stack(traces, -1),
                    transition_count_per_root=depth,
                    transition_count_per_decision=4 * depth,
                    horizon=depth)

    def forward(self, fields):
        return self.imagine(fields)['action_logits']

    def continuation_logits(self, fields):
        """Separate supervised TRAIN path for hard internal action proposals."""
        _fields(fields)
        return self.continuation(fields)
