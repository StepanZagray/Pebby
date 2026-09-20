"""Masked autoregressive dynamics teaching on the agent's own planned moves.

Only the first observed field enters the dynamics. Subsequent fields are model
predictions, including after a life-loss transition. Actual futures are targets
only. Terminal tails carry no loss; life-loss/reset transitions remain trainable.
Appearance/core targets distill the public encoder, while physical state and
events are exact generated-engine labels. No action-selection loss is included.
Changed appearance cells receive an additional transition-balanced loss. Change
masks compare consecutive actual public fields only when both are available;
they are fallible encoder differences, not exact goal/refill-change labels.
"""
import torch
from torch.nn import functional as F

from .structured_policy import _fields
from .structured_objective import CHANGED_THRESHOLD
from .structured_transition import steps_targets

EVENTS = ('lost_life', 'terminal', 'won')


def planning_sequence_loss(model, fields, actions, next_fields, labels,
                           transition_valid, next_field_valid, *, core_scale=None,
                           event_positive_weights=None):
    _fields(fields)
    batch = len(fields)
    if (actions.ndim != 2 or actions.shape[0] != batch or not 1 <= actions.shape[1] <= 16
            or actions.dtype != torch.long or actions.device != fields.device
            or ((actions < 0) | (actions > 3)).any()):
        raise ValueError('actions must be long [B,K] in 0..3, K1..16')
    shape = actions.shape
    if (next_fields.shape != (*shape, 148, 96) or not next_fields.is_floating_point()
            or next_fields.device != fields.device or not torch.isfinite(next_fields).all()):
        raise ValueError('finite next_fields[B,K,148,96] required')
    for mask in (transition_valid, next_field_valid):
        if mask.shape != shape or mask.dtype != torch.bool or mask.device != fields.device:
            raise ValueError('validity masks must be bool[B,K] on field device')
    if (not transition_valid[:, 0].all() or (transition_valid[:, 1:] & ~transition_valid[:, :-1]).any()
            or (next_field_valid & ~transition_valid).any()):
        raise ValueError('transitions must be nonempty contiguous prefixes; field validity must be a subset')
    schema = {'next_player_cell': (2,), 'next_triple': (3,), 'next_steps': (), 'next_lives': ()}
    for name, tail in schema.items():
        value = labels[name]
        if (value.shape != (*shape, *tail) or value.dtype not in (torch.int16, torch.int32, torch.int64)
                or value.device != fields.device):
            raise ValueError('invalid physical sequence label: ' + name)
    for name in EVENTS:
        value = labels[name]
        if (value.shape != shape or value.device != fields.device
                or value.dtype not in (torch.bool, torch.uint8, torch.int16, torch.int32, torch.int64)
                or not ((value == 0) | (value == 1)).all()):
            raise ValueError('invalid event sequence label: ' + name)
    terminal, won = labels['terminal'].bool(), labels['won'].bool()
    if (won & ~terminal & transition_valid).any() or (terminal[:, :-1] & transition_valid[:, 1:]).any():
        raise ValueError('win must terminate; no valid transitions after terminal')
    scale = torch.ones(48, device=fields.device) if core_scale is None else torch.as_tensor(core_scale, device=fields.device).detach()
    if scale.shape != (48,) or not torch.isfinite(scale).all() or (scale < .1).any():
        raise ValueError('core scale must be finite [48] with minimum .1')
    positive = torch.ones(3, device=fields.device) if event_positive_weights is None else torch.as_tensor(event_positive_weights, device=fields.device).detach()
    if positive.shape != (3,) or not torch.isfinite(positive).all() or (positive <= 0).any() or (positive > 50).any():
        raise ValueError('event positive weights must be finite [3] in (0,50]')
    current = fields.detach()
    sums = {name: fields.new_zeros(()) for name in ('field', 'field_changed', 'player', 'glyph', 'budget', 'lives', 'events')}
    physical_count = int(transition_valid.sum())
    field_count = int(next_field_valid.sum())
    change_comparison_count = changed_transition_count = changed_cell_count = 0
    for step in range(shape[1]):
        prediction = model(current, actions[:, step])
        current = prediction['field']  # Deliberately neither teacher-forced nor detached.
        _fields(current)
        # Preserve a graph-connected zero even if no valid change comparison
        # exists. Invalid future targets never enter this expression.
        sums['field_changed'] = sums['field_changed'] + current.reshape(-1)[:1].sum() * 0
        valid = transition_valid[:, step]
        count = int(valid.sum())
        if not count:
            continue
        readout = prediction['readout']
        cell = labels['next_player_cell'][valid, step].long()
        triple = labels['next_triple'][valid, step].long()
        if (cell < 0).any() or (cell >= 12).any() or (triple < 0).any() or (triple[:, 0] >= 6).any() or (triple[:, 1:] >= 4).any():
            raise ValueError('physical labels outside grid/glyph domain')
        sums['player'] = sums['player'] + count * F.cross_entropy(readout['player_logits'][valid], cell[:, 1] * 12 + cell[:, 0])
        glyph_loss = 0
        for index, name in enumerate(('shape', 'color', 'rotation')):
            # Global logits are the carried probabilities actually inserted
            # into the next recurrent field. The generic readout is taught too.
            glyph_loss = glyph_loss + (F.cross_entropy(prediction['glyph_logits'][name][valid], triple[:, index])
                         + F.cross_entropy(readout[f'carried_{name}_logits'][valid], triple[:, index])) / 6
        sums['glyph'] = sums['glyph'] + count * glyph_loss
        sums['budget'] = sums['budget'] + count * F.cross_entropy(readout['steps_logits'][valid], steps_targets(labels['next_steps'][valid, step]))
        sums['lives'] = sums['lives'] + count * F.cross_entropy(readout['lives_logits'][valid], labels['next_lives'][valid, step].long())
        logits = torch.stack([prediction['events'][name + '_logits'][valid] for name in EVENTS], -1)
        targets = torch.stack([labels[name][valid, step].float() for name in EVENTS], -1)
        sums['events'] = sums['events'] + count * F.binary_cross_entropy_with_logits(logits, targets, pos_weight=positive)
        observed = next_field_valid[:, step]
        if observed.any():
            actual = next_fields[observed, step].detach().float()
            predicted = current[observed].float()
            # Carried channels70:84 learn exact glyph targets above, avoiding
            # conflicting distillation from an imperfect carried-state perceptor.
            losses = (((predicted[..., :48] - actual[..., :48]) / scale).square().mean(),
                      (predicted[..., 48:70] - actual[..., 48:70]).square().mean(),
                      (predicted[..., 84:] - actual[..., 84:]).square().mean())
            sums['field'] = sums['field'] + int(observed.sum()) * sum(losses)
        comparable = observed if step == 0 else observed & next_field_valid[:, step - 1]
        change_comparison_count += int(comparable.sum())
        if comparable.any():
            previous = fields[comparable] if step == 0 else next_fields[comparable, step - 1]
            previous = previous.detach().float()[:, :144, 48:70]
            actual = next_fields[comparable, step].detach().float()[:, :144, 48:70]
            changed = (actual - previous).abs().amax(-1) > CHANGED_THRESHOLD
            counts = changed.sum(-1)
            errors = (current[comparable, :144, 48:70].float() - actual).square().mean(-1)
            # Normalize within each transition first: a one-cell goal update
            # gets the same transition weight as a larger public scene change.
            per_transition = (errors * changed).sum(-1) / counts.clamp_min(1)
            sums['field_changed'] = sums['field_changed'] + per_transition.sum()
            changed_transition_count += int((counts > 0).sum())
            changed_cell_count += int(counts.sum())
    denominators = dict(field=max(field_count, 1), field_changed=max(changed_transition_count, 1))
    losses = {name:value / denominators.get(name, physical_count) for name,value in sums.items()}
    total = losses['field'] + losses['field_changed'] + losses['player'] + losses['glyph'] + .5 * (losses['budget'] + losses['lives']) + losses['events']
    return dict(total=total, losses=losses, transition_count=physical_count, field_count=field_count,
                change_comparison_count=change_comparison_count, changed_transition_count=changed_transition_count,
                changed_cell_count=changed_cell_count,
                event_positive_counts={name:int((labels[name].bool() & transition_valid).sum()) for name in EVENTS})
