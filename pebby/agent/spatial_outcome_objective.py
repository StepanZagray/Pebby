"""Generated transition supervision; privileged targets never enter policy forward."""
import numpy as np
import torch
from torch.nn import functional as F

from .neural_outcome_planner import EVENT_NAMES, FIELD_NAMES, neural_outcome_losses
from .world_grounding import SIZES


def targets(items, *, current=False):
    player = items['player_cell' if current else 'next_player_cell'].long()
    triple = items['current_triple' if current else 'next_triple'].long()
    steps = items['current_steps' if current else 'next_steps'].long()
    lives = items['current_lives' if current else 'next_lives'].long()
    return (player[..., 1] * 12 + player[..., 0], *triple.unbind(-1), steps.clamp_min(-1) + 1, lives)


def training_weights(arrays):
    """Fixed weights from TRAIN only; glyph changes get equal aggregate mass."""
    triple, old = np.asarray(arrays['next_triple']), np.asarray(arrays['current_triple'])
    branches = len(triple) * 4
    changed = (triple != old[:, None]).sum(axis=(0, 1))
    if np.any(changed == 0) or np.any(changed == branches):
        raise ValueError('glyph balancing requires both changed and unchanged TRAIN examples')
    positives = np.array([np.asarray(arrays[name], dtype=bool).sum() for name in EVENT_NAMES])
    if np.any(positives == 0):
        raise ValueError('event supervision requires positive TRAIN examples')
    return dict(train_branches=branches, glyph_changed_counts=changed.tolist(),
                glyph_change_weights=np.stack((branches / (2 * (branches - changed)),
                                               branches / (2 * changed)), axis=1).tolist(),
                event_positive_counts=positives.tolist(),
                event_positive_weights=np.minimum((branches - positives) / positives, 50.).tolist(),
                event_positive_weight_cap=50.,
                physical_weights=[1., 1/3, 1/3, 1/3, .5, .5])


def actual_outcomes(items):
    """Finite teacher logits for comparator supervision, never inference input."""
    fields = targets(items)
    distance = items['distances'].long()
    bins = torch.where(distance < 0, 129, distance.clamp(0, 128))
    logits = [F.one_hot(t, n).float() * 40 - 20 for t, n in zip((*fields, bins), (*SIZES, 130))]
    events = torch.stack([items[name].float() for name in EVENT_NAMES], -1) * 40 - 20
    return tuple(logits[:6]), logits[6], events


def spatial_outcome_losses(model, predicted, items, weights):
    # This also checks all target/output contracts and reports natural metrics.
    record = neural_outcome_losses(predicted, items)
    following, current = targets(items), targets(items, current=True)
    physical = []
    for index, (name, scores, target, previous) in enumerate(zip(FIELD_NAMES, predicted['field_logits'], following, current)):
        ce = F.cross_entropy(scores.float().flatten(0, 1), target.flatten(), reduction='none').reshape_as(target)
        changed = target != previous[:, None]
        if 1 <= index <= 3:
            pair = torch.as_tensor(weights['glyph_change_weights'][index - 1], device=ce.device)
            ce = ce * pair[changed.long()]
        physical.append(ce.mean() * weights['physical_weights'][index])
        with torch.no_grad():
            correct = scores.argmax(-1) == target
            for suffix, mask in [('changed', changed), ('unchanged', ~changed),
                                 ('changed_no_life_loss', changed & ~items['lost_life'].bool())]:
                key = name + '_' + suffix + '_accuracy'
                count = mask.sum()
                record['diagnostics'][key] = correct[mask].float().mean() if bool(count) else None
                record['diagnostic_weights'][key] = count
    event_targets = torch.stack([items[name].float() for name in EVENT_NAMES], -1)
    pos_weight = torch.as_tensor(weights['event_positive_weights'], device=event_targets.device)
    event_loss = F.binary_cross_entropy_with_logits(predicted['event_logits'].float(), event_targets, pos_weight=pos_weight)
    teacher_scores = model.score_outcomes(*actual_outcomes(items))
    bits = (items['optimal'].long()[:, None] & (1 << torch.arange(4, device=teacher_scores.device))) != 0
    valid = bits.any(-1)
    distribution = bits.float() / bits.sum(-1, keepdim=True).clamp_min(1)
    teacher_loss = -(distribution * teacher_scores.float().log_softmax(-1)).sum() / valid.sum().clamp_min(1)
    with torch.no_grad():
        correct = bits.gather(1, teacher_scores.argmax(-1)[:, None]).squeeze(1)
        record['diagnostics']['teacher_set_accuracy'] = correct[valid].float().mean() if bool(valid.any()) else None
        record['diagnostic_weights']['teacher_set_accuracy'] = valid.sum()
    record['losses'].update(physical=torch.stack(physical).sum(), events=event_loss, teacher_policy=teacher_loss)
    record['total'] = sum(record['losses'].values())
    return record
