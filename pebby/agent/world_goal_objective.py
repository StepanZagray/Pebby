"""Training-only goal preservation on current and actual refined cell states.

Soft-target cross entropy, averaged over three attributes and then all selected
cells across both populations. Empty support gives differentiable finite zero.
Teacher labels and visibility annotations never enter the policy forward path.
"""
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from . import world_training_objectives as base


class GoalHead(nn.Sequential):
    def __init__(self, channels=64):
        # CPU-only RNG fork: neither policy initialization nor SIGReg RNG changes.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            super().__init__(nn.Linear(channels, 128), nn.GELU(), nn.Linear(128, 14))


def public_support(player_cell, fog):
    """Exact 7x7 support: outside HUD/boundary, entirely inside fog disk."""
    if player_cell.ndim != 2 or player_cell.shape[1] != 2:
        raise ValueError('player_cell must be [N,2]')
    if player_cell.dtype.is_floating_point or bool(((player_cell < 0) | (player_cell >= 12)).any()):
        raise ValueError('player_cell must contain integer grid coordinates')
    if fog.dtype != torch.bool or fog.shape != (len(player_cell),):
        raise ValueError('fog must be bool [N]')
    cells = torch.arange(144, device=player_cell.device)
    x, y = 4 + 5 * (cells % 12), 5 * (cells // 12)
    inside = (x - 1 >= 0) & (x + 6 <= 64) & (y - 1 >= 0) & (y + 6 <= 52)
    # A convex disk contains the rectangle iff it contains all four corners.
    px = 4 + 5 * player_cell[:, 0] + 1.5
    py = 5 * player_cell[:, 1] + 1.5
    dx = torch.maximum((x[None] - 1 - px[:, None]).abs(), (x[None] + 5 - px[:, None]).abs())
    dy = torch.maximum((y[None] - 1 - py[:, None]).abs(), (y[None] + 5 - py[:, None]).abs())
    return inside[None] & (~fog[:, None] | (dx.square() + dy.square() <= 400))


def annotations(batch, fog_by_seed):
    seeds = batch.get('goal_seeds')
    if seeds is None or seeds.ndim != 1 or len(seeds) != len(batch['frames']):
        raise ValueError('bound per-row goal_seeds required')
    try:
        flags = [fog_by_seed[int(seed)] for seed in seeds.tolist()]
    except KeyError as error:
        raise ValueError(f'missing seed in bound fog bank: {error}') from error
    if any(type(flag) is not bool for flag in flags):
        raise ValueError('bound fog flags must be boolean')
    device = batch['frames'].device
    b = len(seeds)
    current, actual = batch.get('player_cell'), batch.get('next_player_cell')
    if current is None or actual is None or current.shape != (b, 2) or actual.shape != (b, 4, 2):
        raise ValueError('current and actual successor player coordinates required')
    fog = torch.tensor(flags, device=device, dtype=torch.bool)
    players = torch.cat((current, actual.flatten(0, 1)))
    return public_support(players, torch.cat((fog, fog.repeat_interleave(4))))


class GoalObjective:
    def __init__(self, teacher, fog_by_seed, channels=64, teacher_chunk_size=128):
        self.teacher = teacher.eval().requires_grad_(False)
        self.head = GoalHead(channels)
        self.fog_by_seed = dict(fog_by_seed)
        self.teacher_chunk_size = teacher_chunk_size

    def goal_loss(self, states, frames, support):
        self.teacher.eval()
        with torch.no_grad(), torch.autocast(device_type=frames.device.type, enabled=False):
            outputs = [self.teacher(chunk) for chunk in frames.split(self.teacher_chunk_size)]
            roles, *attrs = [torch.cat(values) for values in zip(*outputs)]
            probabilities = [value.float().softmax(-1) for value in attrs]
            selected = support & (roles[..., 1].float().sigmoid() >= .99)
            for probabilities_ in probabilities:
                selected &= probabilities_.amax(-1) >= .99
        # Decode selected cells only; no 5B*144 hidden activations are retained.
        logits = self.head(states[selected]).float().split((6, 4, 4), -1)
        count = selected.sum()
        total = sum(-(target[selected] * F.log_softmax(prediction, -1)).sum()
                    for prediction, target in zip(logits, probabilities)) / (3 * count.clamp_min(1))
        return total, selected

    def __call__(self, model, batch, weights=None, *, loops=None, sigreg_generator=None):
        support = annotations(batch, self.fog_by_seed)  # fail before policy computation
        captures, depth = [], 0
        original = model.assemble

        def assemble(*args, **kwargs):
            nonlocal depth
            depth += 1
            try:
                result = original(*args, **kwargs)
                if depth == 1:
                    captures.append(result['state'][:, :144])
                return result
            finally:
                depth -= 1

        with patch.object(model, 'assemble', assemble):
            out = base.world_losses(model, batch, weights, loops=loops, sigreg_generator=sigreg_generator)
        b = len(batch['frames'])
        if len(captures) != 2 or [len(value) for value in captures] != [b, 4 * b]:
            raise RuntimeError('base objective current/actual assembly contract changed')
        frames = torch.cat((batch['frames'][:, -1], batch['next_frames'].flatten(0, 1)))
        loss, selected = self.goal_loss(torch.cat(captures), frames, support)
        out['losses']['goal_preservation'] = loss
        out['total'] = out['total'] + loss  # fixed lambda_goal = 1
        out['diagnostic_weights']['goal_preservation'] = selected.sum()
        out['diagnostics'].update(goal_selected_current=selected[:b].sum(),
                                  goal_selected_actual=selected[b:].sum())
        return out
