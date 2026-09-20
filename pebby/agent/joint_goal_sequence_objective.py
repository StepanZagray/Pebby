"""Exact causal K4 supervision for the fresh public-pixel neural planner.

Only the root public history enters the encoder. Stored actions drive D's own
preceding predictions; actual successor pixels/state are loss targets only.
Life-loss resets remain ordinary supervised transitions. Terminal observations
are supervised, while every later target in that branch is excluded.
"""
import torch
from torch.nn import functional as F

from .cell_appearance import cell_patches
from .joint_goal_objective import (
    STATE_KEYS, physical_loss, policy_loss, reconstruction_loss,
    relation_loss, semantic_loss,
)
from .structured_transition import steps_targets

STATE_SHAPES = dict(player_cell=(2,), triple=(3,), steps=(), lives=(),
                    roles=(144, 8), goal_triple=(144, 3),
                    **{name: (144,) for name in ('goal_presence', 'goal_solved', 'visible',
                                                'support', 'semantic_valid', 'goal_attribute_valid')})

# Historical balance. Kept as the default so existing runs stay reproducible.
DEFAULT_LOSS_WEIGHTS = dict(physical=1., semantic=1., relation=1., reconstruction=.25,
                            policy=1., continuation=1., value=.5, perception=1.,
                            feedback=1., events=1.)

# Control-weighted balance. Measured at initialization, `policy` delivers a
# gradient norm of 2.8e-4 into the encoder while `feedback`, `semantic`,
# `events` and `perception` each deliver 0.6-0.8, so the shared representation
# is shaped almost entirely by reconstruction-style targets and the action
# terms get no say. These weights raise the two terms that supervise action
# choice and damp the well-fit reconstruction targets. They do not change any
# target, only how much each already-defined term counts.
CONTROL_LOSS_WEIGHTS = dict(DEFAULT_LOSS_WEIGHTS, policy=8., continuation=8.,
                            reconstruction=.05, perception=.5, feedback=.5)

LOSS_WEIGHT_PRESETS = dict(default=DEFAULT_LOSS_WEIGHTS, control=CONTROL_LOSS_WEIGHTS)


def resolve_loss_weights(weights=None):
    """Accept a preset name, an explicit mapping, or None for the default."""
    if weights is None:
        return dict(DEFAULT_LOSS_WEIGHTS)
    if isinstance(weights, str):
        if weights not in LOSS_WEIGHT_PRESETS:
            raise ValueError('unknown loss weight preset: ' + weights)
        return dict(LOSS_WEIGHT_PRESETS[weights])
    if set(weights) != set(DEFAULT_LOSS_WEIGHTS):
        raise ValueError('loss weights must name exactly the ten loss terms')
    resolved = {name: float(value) for name, value in weights.items()}
    if any(not 0. <= value <= 1e3 for value in resolved.values()):
        raise ValueError('loss weights must be finite and within 0..1000')
    if not any(resolved[name] > 0. for name in ('policy', 'continuation')):
        raise ValueError('at least one action-selection term must carry weight')
    return resolved


def _integer(values):
    return values.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _shape(batch, name, shape, *, boolean=False):
    value = batch[name]
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f'{name} must have shape {shape}')
    if boolean and value.dtype != torch.bool:
        raise ValueError(f'{name} must be boolean')
    return value


def validate_sequence_batch(batch):
    """Reject broken chronology instead of silently training through padding."""
    frames = batch['frames']
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or not len(frames):
        raise ValueError('nonempty public H8 batch required')
    b = len(frames)
    _shape(batch, 'frames', (b, 8, 64, 64))
    _shape(batch, 'history_valid', (b, 8), boolean=True)
    _shape(batch, 'previous_actions', (b, 8))
    actions = _shape(batch, 'actions', (b, 4, 4))
    valid = _shape(batch, 'transition_valid', (b, 4, 4), boolean=True)
    for name in ('next_frame_valid', 'next_history_reset', 'lost_life', 'terminal', 'won',
                 'next_distance_valid', 'next_optimal_valid'):
        _shape(batch, name, (b, 4, 4), boolean=True)
    for name in ('next_distance', 'next_optimal'):
        _shape(batch, name, (b, 4, 4))
    for name in ('optimal', 'current_distance'):
        _shape(batch, name, (b,))
    for name in ('optimal_valid', 'current_distance_valid'):
        _shape(batch, name, (b,), boolean=True)
    _shape(batch, 'next_frames', (b, 4, 4, 64, 64))
    for name, shape in STATE_SHAPES.items():
        _shape(batch, name, (b,) + shape)
        _shape(batch, 'next_' + name, (b, 4, 4) + shape)
    if (not _integer(actions) or bool(((actions[valid] < 0) | (actions[valid] > 3)).any())
            or not bool(valid[..., 0].all())
            or not torch.equal(actions[..., 0].long(), torch.arange(4, device=actions.device).expand(b, -1))):
        raise ValueError('four valid canonical first actions and valid action indices required')
    # Stop exactly after a terminal transition. Life loss itself is not terminal.
    expected = valid[..., :-1] & ~batch['terminal'][..., :-1]
    if not torch.equal(valid[..., 1:], expected):
        raise ValueError('transition validity must stop exactly after terminal, without resumption')
    if (bool((batch['next_frame_valid'] & ~valid).any())
            or bool((batch['won'][valid] & ~batch['terminal'][valid]).any())
            or not torch.equal(batch['next_history_reset'][valid], batch['lost_life'][valid])):
        raise ValueError('invalid frame/event/reset chronology')
    return valid


def distance_loss(logits, distance, valid):
    """Class129 means *proved* unreachable; unknown targets have valid=False.

    Finite distances above128 are unsupported and rejected, never clipped into
    a false exact target. Arbitrary values in masked slots are not interpreted.
    """
    if (logits.ndim != 2 or logits.shape[1] != 130 or distance.shape != logits.shape[:1]
            or valid.shape != distance.shape or valid.dtype != torch.bool or not _integer(distance)):
        raise ValueError('value needs logits[B,130], integer distance[B], boolean validity[B]')
    target = distance[valid].long()
    if bool(((target < -1) | (target > 128)).any()):
        raise ValueError('supported value targets must be -1 or exact distances0..128')
    if not len(target):
        return logits.sum() * 0
    return F.cross_entropy(logits[valid].float(), torch.where(target == -1, 129, target))


def _policy(logits, optimal, valid):
    if not _integer(optimal):
        raise ValueError('optimal action bitsets must be integers')
    return policy_loss(logits, torch.where(valid, optimal, 0), valid)


def sequence_reconstruction_loss(model, fields, frames, observed, previous_frames, previous_valid):
    """Pixel targets only; changed cells require two actual consecutive frames."""
    if observed.dtype != torch.bool or previous_valid.dtype != torch.bool:
        raise ValueError('pixel validity must be boolean')
    if observed.shape != fields.shape[:1] or previous_valid.shape != observed.shape:
        raise ValueError('one pixel validity per predicted field required')
    if not bool(observed.any()):
        return fields.sum() * 0
    # Select before palette lookup/CE, making even invalid sentinel pixels inert.
    active_frames = frames[observed]
    logits = model.pixel_logits(fields[observed])
    board = cell_patches(active_frames).long()
    hud = active_frames[:, 52:64].reshape(-1, 12, 4, 16).permute(0, 2, 1, 3).long()
    n = len(board)
    board_ce = F.cross_entropy(logits['board'].float().reshape(-1, 16), board.reshape(-1),
                              reduction='none').reshape(n, 144, 7, 7).mean((-1, -2))
    hud_ce = F.cross_entropy(logits['hud'].float().reshape(-1, 16), hud.reshape(-1))
    loss = (board_ce.mean() + hud_ce) / 2
    pair = previous_valid[observed]
    if bool(pair.any()):
        old = cell_patches(previous_frames[observed][pair]).long()
        changed = (board[pair] != old).any((-1, -2))
        changing = changed.any(-1)
        if bool(changing.any()):
            weighted = torch.where(changed, board_ce[pair], 0.).sum(-1) / changed.sum(-1).clamp_min(1)
            loss = loss + weighted[changing].mean()
    return loss


def joint_goal_sequence_loss(model, batch, weights=None):
    """Train root selection and exact K4 dynamics with no teacher forcing.

    Root labels match joint_goal_data. Causal labels use next_*[B,4,4,...],
    including next_distance/next_distance_valid. The legacy root-action
    distances[B,4] may be retained in the cache but are not used here.
    """
    valid = validate_sequence_batch(batch)
    b = len(valid)
    root_valid = torch.ones(b, dtype=torch.bool, device=valid.device)
    labels = {key: batch[key] for key in STATE_KEYS}
    details = model.encode_details(batch['frames'], batch['history_valid'], batch['previous_actions'])
    root = details['field']
    current = model.readout(root)
    losses = dict(physical=physical_loss(current, labels),
        semantic=semantic_loss(current, model.visibility_logits(root), labels, root_valid),
        relation=relation_loss(model.relation(root), labels, root_valid),
        reconstruction=reconstruction_loss(model, root, batch['frames'][:, -1], root_valid),
        policy=_policy(model.imagine(root)['action_logits'], batch['optimal'], batch['optimal_valid']),
        continuation=_policy(model.continuation_logits(root), batch['optimal'], batch['optimal_valid']),
        value=distance_loss(model.value_logits(root), batch['current_distance'], batch['current_distance_valid']))
    losses['perception'] = semantic_loss(details, details['visibility_logits'], labels, root_valid)
    losses['perception'] += torch.stack([F.cross_entropy(details['carried_' + name + '_logits'].float(),
        labels['triple'][:, i].long()) for i, name in enumerate(('shape', 'color', 'rotation'))]).mean()

    field = root[:, None].expand(-1, 4, -1, -1).reshape(b * 4, 148, 96)
    outputs = []
    for h in range(4):
        # Padding actions have no real interpretation or supervised descendants.
        actions = torch.where(valid[:, :, h], batch['actions'][:, :, h], 0).reshape(-1).long()
        prediction = model.dynamics(field, actions)
        outputs.append(prediction)
        field = prediction['field']  # Deliberately no detach, reset, or E(actual successor).
    # B,branch,horizon order is identical for model outputs and exact labels.
    def stack(values):
        return torch.stack(values, 1).flatten(0, 1)
    fields = stack([output['field'] for output in outputs])
    active = valid.reshape(-1)
    observed = batch['next_frame_valid'].reshape(-1) & active
    following = {key: batch['next_' + key].flatten(0, 2) for key in STATE_KEYS}
    readout = {key: stack([output['readout'][key] for output in outputs]) for key in outputs[0]['readout']}
    selected = {key: value[active] for key, value in following.items()}
    losses['physical'] = (losses['physical'] + physical_loss(
        {key: value[active] for key, value in readout.items()}, selected)) / 2
    visible_readout = {key: value[observed] for key, value in readout.items()}
    visible_labels = {key: value[observed] for key, value in following.items()}
    seen = torch.ones(int(observed.sum()), dtype=torch.bool, device=root.device)
    if len(seen):
        losses['semantic'] = (losses['semantic'] + semantic_loss(visible_readout,
            model.visibility_logits(fields[observed]), visible_labels, seen)) / 2
        losses['relation'] = (losses['relation'] + relation_loss(
            model.relation(fields[observed]), visible_labels, seen)) / 2
    frames = batch['next_frames']
    previous = torch.cat((batch['frames'][:, -1, None, None].expand(-1, 4, 1, -1, -1), frames[:, :, :-1]), 2)
    previous_valid = torch.cat((torch.ones_like(valid[:, :, :1]), batch['next_frame_valid'][:, :, :-1]), 2)
    losses['reconstruction'] = (losses['reconstruction'] + sequence_reconstruction_loss(model, fields,
        frames.flatten(0, 2), observed, previous.flatten(0, 2), previous_valid.flatten())) / 2
    glyph = {key: stack([output['glyph_logits'][key] for output in outputs])[active]
             for key in ('shape', 'color', 'rotation')}
    feedback = {key: stack([output['feedback_logits'][key] for output in outputs])[active]
                for key in ('player', 'steps', 'lives')}
    losses['feedback'] = torch.stack([F.cross_entropy(glyph[name].float(), selected['triple'][:, i].long())
        for i, name in enumerate(('shape', 'color', 'rotation'))]).mean()
    losses['feedback'] += (F.cross_entropy(feedback['player'].float(),
        selected['player_cell'][:, 1].long() * 12 + selected['player_cell'][:, 0].long())
        + F.cross_entropy(feedback['steps'].float(), steps_targets(selected['steps']))
        + F.cross_entropy(feedback['lives'].float(), selected['lives'].long())) / 3
    losses['events'] = torch.stack([F.binary_cross_entropy_with_logits(
        stack([output['events'][name + '_logits'] for output in outputs])[active].float(),
        batch[name].flatten()[active].float()) for name in ('lost_life', 'terminal', 'won')]).mean()
    losses['value'] = (losses['value'] + distance_loss(model.value_logits(fields),
        batch['next_distance'].flatten(), batch['next_distance_valid'].flatten() & active)) / 2
    losses['continuation'] = (losses['continuation'] + _policy(model.continuation_logits(fields),
        batch['next_optimal'].flatten(), batch['next_optimal_valid'].flatten() & active)) / 2
    weights = resolve_loss_weights(weights)
    if set(weights) != set(losses):
        raise ValueError('loss weights must name exactly the computed loss terms')
    total = sum(loss * weights[name] for name, loss in losses.items())
    if not bool(torch.isfinite(total)):
        raise ValueError('nonfinite joint sequence loss')
    return dict(total=total, losses=losses, weights=weights,
                actual_horizon=4, imagined_horizon=model.cfg.horizon,
                valid_transitions_per_horizon=valid.sum((0, 1)).tolist(),
                supervised_roots=int((batch['optimal_valid'] & (batch['optimal'] != 0)).sum()))
