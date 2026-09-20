"""Fresh public-pixel planning supervision; labels never enter model forward.

This first dataset supplies real H1 branches. The policy independently imagines
K4 with its own dynamics; H2+ have policy gradients but no exact transition loss
in this objective. No old encoder features or moving latent targets are used.
"""
import torch
from torch.nn import functional as F

from .cell_appearance import cell_patches
from .structured_transition import steps_targets

STATE_KEYS = ('player_cell', 'triple', 'steps', 'lives', 'roles', 'goal_triple',
              'goal_presence', 'goal_solved', 'visible', 'support',
              'semantic_valid', 'goal_attribute_valid')


def mean_masked(values, mask):
    if values.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError('loss values and boolean mask must have the same shape')
    return torch.where(mask, values, 0.).sum() / mask.sum().clamp_min(1)


def policy_loss(logits, optimal, valid=None):
    if logits.ndim != 2 or logits.shape[1] != 4 or optimal.shape != logits.shape[:1]:
        raise ValueError('policy needs logits[B,4] and optimal bitmask[B]')
    if bool(((optimal < 0) | (optimal > 15)).any()):
        raise ValueError('optimal action masks must be in 0..15')
    mask = (optimal.long()[:, None] & (1 << torch.arange(4, device=logits.device))) != 0
    valid = optimal != 0 if valid is None else valid & (optimal != 0)
    # Uniform probability across equally optimal actions, not a fabricated label
    # for unreachable/unknown roots. Clamp only the normalization of empty rows.
    target = mask.float() / mask.sum(-1, keepdim=True).clamp_min(1)
    ce = -(target * logits.float().log_softmax(-1)).sum(-1)
    return mean_masked(ce, valid)


def physical_loss(readout, labels):
    triple = labels['triple'].long()
    values = [F.cross_entropy(readout['player_logits'].float(),
                              labels['player_cell'][:, 1].long() * 12 + labels['player_cell'][:, 0].long())]
    values += [F.cross_entropy(readout['carried_' + name + '_logits'].float(), triple[:, i])
               for i, name in enumerate(('shape', 'color', 'rotation'))]
    values += [F.cross_entropy(readout['steps_logits'].float(), steps_targets(labels['steps'])),
               F.cross_entropy(readout['lives_logits'].float(), labels['lives'].long())]
    return torch.stack(values).mean()


def semantic_loss(readout, visibility, labels, valid):
    supported = labels['support'].bool() & valid[:, None]
    semantic = labels['semantic_valid'].bool() & supported
    roles = F.binary_cross_entropy_with_logits(readout['role_logits'].float(),
                                                labels['roles'].float(), reduction='none').mean(-1)
    total = mean_masked(roles, semantic)
    goal_mask = labels['goal_attribute_valid'].bool() & supported
    for i, name in enumerate(('shape', 'color', 'rotation')):
        logits = readout['goal_' + name + '_logits'].float()
        # Non-goal cells may use sentinel targets; they are not CE classes.
        target = torch.where(goal_mask, labels['goal_triple'][..., i], 0).long()
        ce = F.cross_entropy(logits.transpose(1, 2), target, reduction='none')
        total = total + mean_masked(ce, goal_mask) / 3
    # Visibility needs negative fog/HUD examples too. Public semantic support
    # already excludes hidden cells and would silently remove those negatives.
    total = total + mean_masked(F.binary_cross_entropy_with_logits(
        visibility.float(), labels['visible'].float(), reduction='none'), valid[:, None].expand_as(supported))
    return total / 3


def relation_loss(relation, labels, valid):
    supported = labels['goal_presence'].bool() & labels['support'].bool() & valid[:, None]
    matching = (labels['goal_triple'] == labels['triple'][:, None]).all(-1)
    # Compatibility is taught where the goal glyph is publicly supported;
    # solved status uses the persistent anchor and separate support validity.
    compatible = supported & labels['goal_attribute_valid'].bool()
    solved_support = supported & labels['semantic_valid'].bool()
    return (mean_masked(F.binary_cross_entropy_with_logits(relation['compatibility_logits'].float(),
                         matching.float(), reduction='none'), compatible)
            + mean_masked(F.binary_cross_entropy_with_logits(relation['solved_logits'].float(),
                         labels['goal_solved'].float(), reduction='none'), solved_support)) / 2


def reconstruction_loss(model, field, frames, valid, previous=None):
    logits = model.pixel_logits(field)
    board = cell_patches(frames).long()
    hud = frames[:, 52:64].reshape(len(frames), 12, 4, 16).permute(0, 2, 1, 3).long()
    board_ce = F.cross_entropy(logits['board'].float().reshape(-1, 16), board.reshape(-1),
                              reduction='none').reshape(len(frames), 144, 7, 7).mean((-1, -2))
    hud_ce = F.cross_entropy(logits['hud'].float().reshape(-1, 16), hud.reshape(-1),
                            reduction='none').reshape(len(frames), 4, 12, 16).mean((1, 2, 3))
    total = (mean_masked(board_ce.mean(-1), valid) + mean_masked(hud_ce, valid)) / 2
    if previous is not None:
        changed = (board != cell_patches(previous).long()).any((-1, -2)) & valid[:, None]
        per_transition = torch.where(changed, board_ce, 0.).sum(-1) / changed.sum(-1).clamp_min(1)
        total = total + mean_masked(per_transition, changed.any(-1))
    return total


def joint_goal_loss(model, batch):
    """Policy-only forward inputs, exact label-side H1 and pixel supervision."""
    frames = batch['frames']
    b = len(frames)
    if frames.shape != (b, 8, 64, 64) or b == 0:
        raise ValueError('public H8 frames required')
    valid = torch.ones(b, dtype=torch.bool, device=frames.device)
    labels = {key: batch[key] for key in STATE_KEYS}
    details = model.encode_details(frames, batch['history_valid'], batch['previous_actions'])
    field = details['field']
    current_readout = model.readout(field)
    losses = dict(physical=physical_loss(current_readout, labels),
                  semantic=semantic_loss(current_readout, model.visibility_logits(field), labels, valid),
                  relation=relation_loss(model.relation(field), labels, valid),
                  reconstruction=reconstruction_loss(model, field, frames[:, -1], valid),
                  policy=policy_loss(model.imagine(field)['action_logits'], batch['optimal']),
                  continuation=policy_loss(model.continuation_logits(field), batch['optimal']))
    # Direct perception supervision reaches the field's actual probability
    # producers, in addition to D's generic state readout.
    losses['perception'] = semantic_loss(details, details['visibility_logits'], labels, valid)
    losses['perception'] += torch.stack([F.cross_entropy(details['carried_' + name + '_logits'].float(),
                                       labels['triple'][:, i].long())
                                       for i, name in enumerate(('shape', 'color', 'rotation'))]).mean()
    repeated = field[:, None].expand(-1, 4, -1, -1).reshape(b * 4, 148, 96)
    actions = torch.arange(4, device=field.device).repeat(b)
    predicted = model.dynamics(repeated, actions)
    next_field = predicted['field']
    next_labels = {key: batch['next_' + key].flatten(0, 1) for key in STATE_KEYS}
    observed = batch['next_frame_valid'].flatten().bool()
    losses['physical'] = (losses['physical'] + physical_loss(predicted['readout'], next_labels)) / 2
    losses['semantic'] = (losses['semantic'] + semantic_loss(predicted['readout'],
                          model.visibility_logits(next_field), next_labels, observed)) / 2
    losses['relation'] = (losses['relation'] + relation_loss(model.relation(next_field), next_labels, observed)) / 2
    losses['reconstruction'] = (losses['reconstruction'] + reconstruction_loss(model, next_field,
        batch['next_frames'].flatten(0, 1), observed,
        frames[:, -1, None].expand(-1, 4, -1, -1).flatten(0, 1))) / 2
    direct = predicted['glyph_logits']
    losses['feedback'] = torch.stack([F.cross_entropy(direct[name].float(),
                                    next_labels['triple'][:, i].long())
                                    for i, name in enumerate(('shape', 'color', 'rotation'))]).mean()
    feedback = predicted['feedback_logits']
    losses['feedback'] += (F.cross_entropy(feedback['player'].float(), next_labels['player_cell'][:, 1].long() * 12
                                          + next_labels['player_cell'][:, 0].long())
        + F.cross_entropy(feedback['steps'].float(), steps_targets(next_labels['steps']))
        + F.cross_entropy(feedback['lives'].float(), next_labels['lives'].long())) / 3
    losses['events'] = torch.stack([F.binary_cross_entropy_with_logits(predicted['events'][name + '_logits'].float(),
                                    batch[name].flatten().float())
                                    for name in ('lost_life', 'terminal', 'won')]).mean()
    value_logits = model.value_logits(next_field)
    distance = batch['distances'].flatten()
    targets = torch.where(distance < 0, 129, distance.long().clamp(0, 128))
    losses['value'] = mean_masked(F.cross_entropy(value_logits.float(), targets, reduction='none'),
                                 batch['distance_valid'].flatten().bool())
    current_distance = batch['current_distance']
    current_targets = torch.where(current_distance < 0, 129, current_distance.long().clamp(0, 128))
    losses['value'] = (losses['value'] + mean_masked(F.cross_entropy(model.value_logits(field).float(),
        current_targets, reduction='none'), batch['current_distance_valid'].bool())) / 2
    # Teach the proposal on predicted successor fields as well as real roots.
    # Exact next-action masks are loss targets, never inputs or imposed actions
    # in the independent K4 policy rollout. This supplies the gradient that hard
    # argmax in imagination cannot carry into the continuation policy.
    losses['continuation'] = (losses['continuation'] + policy_loss(model.continuation_logits(next_field),
        batch['next_optimal'].flatten(), batch['next_optimal_valid'].flatten().bool())) / 2
    total = sum(loss * (.25 if name == 'reconstruction' else .5 if name == 'value' else 1.)
                for name, loss in losses.items())
    if not bool(torch.isfinite(total)):
        raise ValueError('nonfinite joint planning loss')
    return dict(total=total, losses=losses, supervised_roots=int((batch['optimal'] != 0).sum()),
                actual_horizon=1, imagined_horizon=model.cfg.horizon)
