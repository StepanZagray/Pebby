"""Autoregressive four-step objective for structured field transitions.

The first field is the only observed input.  Each predicted field is fed to the
next transition, while all four chronological target fields are used only by
the loss and readout diagnostics.  This module deliberately does not alter the
one-step objective or the transition models.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .structured_objective import (CHANGED_THRESHOLD, EVENT_NAMES, GROUPS,
                                   _integer, _readout_losses, _weighted_mean)
from .structured_transition import steps_targets


HORIZONS = 4
READOUT_KEYS = (
    "player_logits", "role_logits", "goal_shape_logits", "goal_color_logits",
    "goal_rotation_logits", "carried_shape_logits", "carried_color_logits",
    "carried_rotation_logits", "steps_logits", "lives_logits",
)
EVENT_LOGIT_KEYS = tuple(f"{name}_logits" for name in EVENT_NAMES)
INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _validate_field(value, shape, name, *, check_probabilities=False):
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
            or not value.is_floating_point()):
        raise ValueError(f"{name} must be floating {tuple(shape)}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    if check_probabilities:
        probabilities = value[..., :144, 48:85]
        if bool(((probabilities < 0) | (probabilities > 1)).any()):
            raise ValueError(f"{name} observed probability channels must be in 0..1")


def _validate_actions(actions, batch, device):
    if (not isinstance(actions, torch.Tensor) or tuple(actions.shape) != (batch, HORIZONS)
            or actions.dtype not in INTEGER_DTYPES or actions.dtype == torch.bool
            or actions.device != device):
        raise ValueError(f"actions must be integer [{batch},4] on the field device")
    if bool(((actions < 0) | (actions > 3)).any()):
        raise ValueError("actions must be in 0..3")
    return actions.long()


def _validate_scale(scale, device):
    if (not isinstance(scale, torch.Tensor) or tuple(scale.shape) != (48,)
            or not scale.is_floating_point() or scale.device != device
            or not bool(torch.isfinite(scale).all()) or bool((scale < .1).any())):
        raise ValueError("scale must be finite floating [48] on the field device with values >= .1")
    return scale.detach().to(dtype=torch.float32)


def _validate_labels(labels, batch, device):
    if not isinstance(labels, Mapping):
        raise ValueError("labels must be a mapping")
    checked = {}
    for name, shape, low, high in (
            ("player_cell", (batch, 2), 0, 11),
            ("triple", (batch, 3), 0, 5),
            ("steps", (batch,), -3, 42),
            ("lives", (batch,), 0, 3)):
        if name not in labels:
            raise ValueError(f"missing label {name}")
        value = _integer(labels[name], shape, name, low, high)
        if value.device != device:
            raise ValueError(f"{name} and fields must share a device")
        if name == "triple" and bool((value[:, 1:] > 3).any()):
            raise ValueError("triple color and rotation labels must be in 0..3")
        checked[name] = value

    for name, shape, low, high in (
            ("next_player_cell", (batch, HORIZONS, 2), 0, 11),
            ("next_triple", (batch, HORIZONS, 3), 0, 5),
            ("next_steps", (batch, HORIZONS), -3, 42),
            ("next_lives", (batch, HORIZONS), 0, 3)):
        if name not in labels:
            raise ValueError(f"missing label {name}")
        value = labels[name]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.dtype not in INTEGER_DTYPES or value.dtype == torch.bool):
            raise ValueError(f"{name} must be integer {shape}")
        value = value.long()
        if bool(((value < low) | (value > high)).any()):
            raise ValueError(f"{name} outside {low}..{high}")
        if value.device != device:
            raise ValueError(f"{name} and fields must share a device")
        if name == "next_triple" and bool((value[:, :, 1:] > 3).any()):
            raise ValueError("next_triple color and rotation labels must be in 0..3")
        checked[name] = value

    for name in EVENT_NAMES:
        if name not in labels:
            raise ValueError(f"missing label {name}")
        value = labels[name]
        if isinstance(value, torch.Tensor) and value.dtype == torch.bool:
            value = value.long()
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != (batch, HORIZONS)
                or value.dtype not in INTEGER_DTYPES or value.dtype == torch.bool):
            raise ValueError(f"{name} must be integer [{batch},4]")
        value = value.long()
        if bool(((value < 0) | (value > 1)).any()):
            raise ValueError(f"{name} must be binary")
        if value.device != device:
            raise ValueError(f"{name} and fields must share a device")
        checked[name] = value

    if bool((checked["won"].bool() & ~checked["terminal"].bool()).any()):
        raise ValueError("won must imply terminal")
    # The first three labels are chronological interior transitions.  A reset,
    # terminal, or win there would make later targets undefined for this loss.
    for name in EVENT_NAMES:
        if bool(checked[name][:, :HORIZONS - 1].any()):
            raise ValueError(f"{name} is only allowed on the final horizon")
    return checked


def _flatten_horizon_labels(labels):
    return {
        "next_player_cell": labels["next_player_cell"].reshape(-1, 2),
        "next_triple": labels["next_triple"].reshape(-1, 3),
        "next_steps": labels["next_steps"].reshape(-1),
        "next_lives": labels["next_lives"].reshape(-1),
    }


def _validate_step_output(output, batch, device):
    if not isinstance(output, Mapping) or set(output) != {"field", "readout", "events"}:
        raise ValueError("transition step must return field/readout/events")
    field = output["field"]
    _validate_field(field, (batch, 148, 96), "predicted field")
    readout = output["readout"]
    events = output["events"]
    if not isinstance(readout, Mapping) or set(readout) != set(READOUT_KEYS):
        raise ValueError("transition readout keys do not match StructuredFieldReadout")
    expected = {
        "player_logits": (batch, 144), "role_logits": (batch, 144, 8),
        "goal_shape_logits": (batch, 144, 6), "goal_color_logits": (batch, 144, 4),
        "goal_rotation_logits": (batch, 144, 4), "carried_shape_logits": (batch, 6),
        "carried_color_logits": (batch, 4), "carried_rotation_logits": (batch, 4),
        "steps_logits": (batch, 44), "lives_logits": (batch, 4),
    }
    for name, shape in expected.items():
        value = readout[name]
        if tuple(value.shape) != shape or value.device != device or not value.is_floating_point():
            raise ValueError(f"readout {name} has an invalid shape/device")
    if not isinstance(events, Mapping) or set(events) != set(EVENT_LOGIT_KEYS):
        raise ValueError("transition event keys do not match the event contract")
    for name in EVENT_LOGIT_KEYS:
        value = events[name]
        if tuple(value.shape) != (batch,) or value.device != device or not value.is_floating_point():
            raise ValueError(f"event {name} has an invalid shape/device")
    return output


def _pack_step(output):
    return (output["field"], *(output["readout"][key] for key in READOUT_KEYS),
            *(output["events"][key] for key in EVENT_LOGIT_KEYS))


def _unpack_step(values):
    readout_start = 1
    readout_end = readout_start + len(READOUT_KEYS)
    return {
        "field": values[0],
        "readout": dict(zip(READOUT_KEYS, values[readout_start:readout_end])),
        "events": dict(zip(EVENT_LOGIT_KEYS, values[readout_end:])),
    }


def _autoregressive_rollout(model, fields, actions, initial_field, *, checkpoint_steps):
    current = fields
    fields_out = []
    readouts = {name: [] for name in READOUT_KEYS}
    events = {name: [] for name in EVENT_LOGIT_KEYS}
    batch = len(fields)

    for horizon in range(HORIZONS):
        action = actions[:, horizon]

        def step(current_field, step_action, memory=None):
            if memory is None:
                return _pack_step(model(current_field, step_action))
            return _pack_step(model(current_field, step_action, memory))

        if checkpoint_steps:
            if initial_field is None:
                packed = checkpoint(step, current, action, use_reentrant=False)
            else:
                packed = checkpoint(step, current, action, initial_field, use_reentrant=False)
            output = _unpack_step(packed)
        elif initial_field is None:
            output = model(current, action)
        else:
            output = model(current, action, initial_field)
        _validate_step_output(output, batch, fields.device)
        fields_out.append(output["field"])
        for name in READOUT_KEYS:
            readouts[name].append(output["readout"][name])
        for name in EVENT_LOGIT_KEYS:
            events[name].append(output["events"][name])
        current = output["field"]

    return (torch.stack(fields_out, 1),
            {name: torch.stack(values, 1) for name, values in readouts.items()},
            {name: torch.stack(values, 1) for name, values in events.items()})


def sequence_objective(model, fields, next_fields, actions, labels, scale, *,
                       initial_field=None, pos_weight=None, checkpoint_steps=False):
    """Compute a causal H4 autoregressive objective.

    ``next_fields[:, h]`` is a target only.  The input at horizon ``h + 1`` is
    the model's own prediction at horizon ``h``.  If ``initial_field`` is
    supplied, it is passed unchanged to every model step (for models such as
    ``StructuredRecallTransition``); it is detached like the observed source.
    """
    if not isinstance(checkpoint_steps, bool):
        raise ValueError("checkpoint_steps must be bool")
    if (not isinstance(fields, torch.Tensor) or fields.ndim != 3
            or tuple(fields.shape[1:]) != (148, 96) or len(fields) < 1
            or not fields.is_floating_point()):
        raise ValueError("fields must be nonempty floating [B,148,96]")
    batch = len(fields)
    device = fields.device
    _validate_field(fields, fields.shape, "fields", check_probabilities=True)
    _validate_field(next_fields, (batch, HORIZONS, 148, 96), "next_fields",
                    check_probabilities=True)
    if next_fields.device != device:
        raise ValueError("next_fields and fields must share a device")
    actions = _validate_actions(actions, batch, device)
    scale = _validate_scale(scale, device)
    labels = _validate_labels(labels, batch, device)
    if initial_field is not None:
        _validate_field(initial_field, (batch, 148, 96), "initial_field",
                        check_probabilities=True)
        if initial_field.device != device:
            raise ValueError("initial_field and fields must share a device")
        initial_field = initial_field.detach()

    if pos_weight is None:
        positive = torch.ones(3, device=device)
    else:
        positive = torch.as_tensor(pos_weight, device=device, dtype=torch.float32).detach()
        if (tuple(positive.shape) != (3,) or not bool(torch.isfinite(positive).all())
                or bool((positive <= 0).any())):
            raise ValueError("pos_weight must be finite positive [3]")
        positive = positive.clamp(max=20)

    # Targets and the initial observed source are detached once at the boundary;
    # recurrent predictions remain connected across all four model steps.
    source = fields.detach()
    target = next_fields.detach()
    predicted_fields, predicted_readout, predicted_events = _autoregressive_rollout(
        model, source, actions, initial_field, checkpoint_steps=checkpoint_steps)

    losses = {}
    for name, (start, stop) in GROUPS.items():
        error = predicted_fields[..., start:stop].float() - target[..., start:stop].float()
        if name == "core":
            error = error / scale
        losses[f"field_{name}"] = error.square().mean()

    previous_actual = torch.cat((source[:, None], target[:, :-1]), dim=1)
    changed = (target[:, :, :144, 48:70] - previous_actual[:, :, :144, 48:70]).abs().amax(-1)
    changed = changed > CHANGED_THRESHOLD
    changed_error = (predicted_fields[:, :, :144, 48:70].float()
                     - target[:, :, :144, 48:70].float()).square().mean(-1)
    losses["field_changed"] = _weighted_mean(changed_error, changed.float())

    actual_flat = target.reshape(batch * HORIZONS, 148, 96)
    actual_readout_flat = model.readout(actual_flat.detach())
    actual_readout = {name: value.reshape(batch, HORIZONS, *value.shape[1:])
                      for name, value in actual_readout_flat.items()}
    current_readout = model.readout(source.detach())
    next_labels = _flatten_horizon_labels(labels)
    predicted_flat = {name: value.reshape(batch * HORIZONS, *value.shape[2:])
                      for name, value in predicted_readout.items()}
    current_losses = _readout_losses(current_readout, source, labels, "")
    actual_losses = _readout_losses(actual_readout_flat, actual_flat, next_labels, "next_")
    predicted_losses = _readout_losses(predicted_flat, actual_flat, next_labels, "next_")
    for name in current_losses:
        losses[f"readout_{name}"] = (current_losses[name] + actual_losses[name]
                                      + predicted_losses[name]) / 3

    event_logits = torch.stack([predicted_events[name] for name in EVENT_LOGIT_KEYS], -1)
    event_targets = torch.stack([labels[name].float() for name in EVENT_NAMES], -1)
    losses["events"] = F.binary_cross_entropy_with_logits(
        event_logits.float(), event_targets, pos_weight=positive)
    total = sum(losses.values())
    outputs = {
        "predicted": {"fields": predicted_fields, "readout": predicted_readout,
                      "events": predicted_events},
        "actual": {"fields": target, "readout": actual_readout},
        "current": {"fields": source, "readout": current_readout},
        "changed_mask": changed,
    }
    return {"total": total, "losses": losses, "outputs": outputs,
            "effective_pos_weight": positive.detach(),
            "checkpoint_steps": checkpoint_steps}


__all__ = ["sequence_objective", "HORIZONS"]
