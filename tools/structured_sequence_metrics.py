"""Streaming metrics for frozen or trained autoregressive structured models.

This scorer keeps only additive per-horizon sums and counts.  It evaluates the
model's predicted rollout separately from readouts of the actual target fields
and the unchanged initial field, making encoder errors visible independently of
transition errors.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Mapping

import torch
from torch.nn import functional as F

from pebby.agent.structured_objective import CHANGED_THRESHOLD, GROUPS
from pebby.agent.structured_transition import steps_targets


HORIZONS = 4
EVENT_NAMES = ("lost_life", "terminal", "won")
READOUTS = (
    ("player", "player_logits"),
    ("carried_shape", "carried_shape_logits"),
    ("carried_color", "carried_color_logits"),
    ("carried_rotation", "carried_rotation_logits"),
    ("steps", "steps_logits"),
    ("lives", "lives_logits"),
)
INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _as_device(value, device, *, dtype=None):
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        # Copy only the current batch.  This avoids the non-writable warning
        # from torch.as_tensor on read-only numpy/memmap slices without loading
        # the full cache.
        try:
            value = value.copy()
        except AttributeError:
            pass
        tensor = torch.as_tensor(value)
    return tensor.to(device=device, dtype=dtype) if dtype is not None else tensor.to(device=device)


def _batch_value(data, name, begin, end, device, *, dtype=None):
    if name not in data:
        raise ValueError(f"missing sequence metric array {name}")
    return _as_device(data[name][begin:end], device, dtype=dtype)


def _validate_data(data):
    if not isinstance(data, Mapping):
        raise ValueError("data must be a mapping of sequence arrays")
    for name in ("fields", "next_fields", "actions", "next_player_cell", "next_triple",
                 "next_steps", "next_lives", *EVENT_NAMES):
        if name not in data:
            raise ValueError(f"missing sequence metric array {name}")
    fields = data["fields"]
    n = len(fields)
    if n < 1:
        raise ValueError("data must contain at least one sequence")
    expected = {
        "fields": (n, 148, 96), "next_fields": (n, HORIZONS, 148, 96),
        "actions": (n, HORIZONS), "next_player_cell": (n, HORIZONS, 2),
        "next_triple": (n, HORIZONS, 3), "next_steps": (n, HORIZONS),
        "next_lives": (n, HORIZONS),
    }
    for name in EVENT_NAMES:
        expected[name] = (n, HORIZONS)
    for name, shape in expected.items():
        value = data[name]
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    for name in ("fields", "next_fields"):
        value = data[name]
        if getattr(value, "dtype", None) is None:
            raise ValueError(f"{name} must expose a dtype")
        if isinstance(value, torch.Tensor):
            floating = value.is_floating_point()
        else:
            # Numpy arrays/memmaps expose dtype.kind; avoid materializing them.
            floating = getattr(value.dtype, "kind", None) == "f"
        if not floating:
            raise ValueError(f"{name} must be floating point")
    return n


def _validate_scale(scale, device):
    scale = _as_device(scale, device, dtype=torch.float32)
    if (tuple(scale.shape) != (48,) or not bool(torch.isfinite(scale).all())
            or bool((scale < .1).any())):
        raise ValueError("scale must be finite [48] with values >= .1")
    return scale


def _validate_actions(actions):
    if actions.dtype not in INTEGER_DTYPES or actions.dtype == torch.bool:
        raise ValueError("actions must be integer")
    if bool(((actions < 0) | (actions > 3)).any()):
        raise ValueError("actions must be in 0..3")
    return actions.long()


def _autocast(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _readout_targets(data, begin, end, device):
    player = _batch_value(data, "next_player_cell", begin, end, device, dtype=torch.long)
    triple = _batch_value(data, "next_triple", begin, end, device, dtype=torch.long)
    steps = _batch_value(data, "next_steps", begin, end, device, dtype=torch.long)
    lives = _batch_value(data, "next_lives", begin, end, device, dtype=torch.long)
    return {
        "player": player[..., 1] * 12 + player[..., 0],
        "carried_shape": triple[..., 0],
        "carried_color": triple[..., 1],
        "carried_rotation": triple[..., 2],
        "steps": steps_targets(steps),
        "lives": lives,
    }


def _empty_readout_metrics():
    return {name: {"correct_sum": [0] * HORIZONS, "count": [0] * HORIZONS,
                   "cross_entropy_sum": [0.] * HORIZONS}
            for name, _ in READOUTS} | {
                "carried_joint": {"correct_sum": [0] * HORIZONS, "count": [0] * HORIZONS,
                                   "cross_entropy_sum": [0.] * HORIZONS}}


def _accumulate_readouts(store, readout, targets):
    glyph_hits = []
    glyph_cross_entropy = None
    for name, logit_key in READOUTS:
        logits = readout[logit_key]
        if logits.ndim != 3 or logits.shape[:2] != targets[name].shape:
            raise ValueError(f"readout {logit_key} must be [B,4,...]")
        target = targets[name]
        hit = logits.argmax(-1) == target
        ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1),
                             reduction="none").reshape_as(hit)
        if name.startswith("carried_"):
            glyph_hits.append(hit)
            glyph_cross_entropy = ce if glyph_cross_entropy is None else glyph_cross_entropy + ce
        for horizon in range(HORIZONS):
            store[name]["correct_sum"][horizon] += int(hit[:, horizon].sum().item())
            store[name]["count"][horizon] += int(hit[:, horizon].numel())
            store[name]["cross_entropy_sum"][horizon] += float(ce[:, horizon].sum().item())
    joint = torch.stack(glyph_hits, -1).all(-1)
    for horizon in range(HORIZONS):
        store["carried_joint"]["correct_sum"][horizon] += int(joint[:, horizon].sum().item())
        store["carried_joint"]["count"][horizon] += int(joint[:, horizon].numel())
        store["carried_joint"]["cross_entropy_sum"][horizon] += float(
            glyph_cross_entropy[:, horizon].sum().item())


def _finalize_readouts(store):
    for values in store.values():
        values["accuracy"] = [
            float(correct / count) if count else None
            for correct, count in zip(values["correct_sum"], values["count"])]
        values["cross_entropy"] = [
            float(total / count) if count else None
            for total, count in zip(values["cross_entropy_sum"], values["count"])]
    return store


def _empty_events():
    return {event: {name: [0] * HORIZONS for name in ("tp", "fp", "fn", "tn", "positive", "count")}
            for event in EVENT_NAMES}


def _accumulate_events(store, event_logits, data, begin, end, device):
    for event in EVENT_NAMES:
        logits = event_logits[f"{event}_logits"]
        if tuple(logits.shape) != (end - begin, HORIZONS):
            raise ValueError(f"event {event} logits must be [B,4]")
        truth = _batch_value(data, event, begin, end, device, dtype=torch.long).bool()
        prediction = logits >= 0
        for horizon in range(HORIZONS):
            actual = truth[:, horizon]
            predicted = prediction[:, horizon]
            counts = store[event]
            counts["tp"][horizon] += int((actual & predicted).sum().item())
            counts["fp"][horizon] += int((~actual & predicted).sum().item())
            counts["fn"][horizon] += int((actual & ~predicted).sum().item())
            counts["tn"][horizon] += int((~actual & ~predicted).sum().item())
            counts["positive"][horizon] += int(actual.sum().item())
            counts["count"][horizon] += int(actual.numel())


def _empty_groups():
    return {source: {name: {"sum": [0.] * HORIZONS, "count": [0] * HORIZONS}
                     for name in GROUPS}
            for source in ("predicted", "initial_copy")}


def _accumulate_groups(store, source, observed, targets, scale):
    for name, (start, stop) in GROUPS.items():
        error = observed[..., start:stop].float() - targets[..., start:stop].float()
        if name == "core":
            error = error / scale
        squared = error.square()
        for horizon in range(HORIZONS):
            store[source][name]["sum"][horizon] += float(squared[:, horizon].sum().item())
            store[source][name]["count"][horizon] += int(squared[:, horizon].numel())


def _empty_changed():
    return {source: {"sum": [0.] * HORIZONS, "count": [0] * HORIZONS}
            for source in ("predicted", "initial_copy")}


def _accumulate_changed(store, source, observed, targets, previous_actual):
    changed = (targets[:, :, :144, 48:70] - previous_actual[:, :, :144, 48:70]).abs().amax(-1)
    changed = changed > CHANGED_THRESHOLD
    squared = (observed[:, :, :144, 48:70].float()
               - targets[:, :, :144, 48:70].float()).square()
    for horizon in range(HORIZONS):
        masked = squared[:, horizon][changed[:, horizon]]
        store[source]["sum"][horizon] += float(masked.sum().item())
        store[source]["count"][horizon] += int(masked.numel())


def _finalize_groups(store):
    for sources in store.values():
        for values in sources.values():
            values["mse"] = [float(total / count) if count else None
                              for total, count in zip(values["sum"], values["count"])]
    return store


def _finalize_changed(store):
    for values in store.values():
        values["mse"] = [float(total / count) if count else None
                          for total, count in zip(values["sum"], values["count"])]
    return store


@torch.inference_mode()
def evaluate(model, data, scale, device="cpu", batch_size=32, initial_fields=None):
    """Evaluate a chronological H4 dataset without retaining full predictions.

    ``data`` must provide ``fields``, ``next_fields``, ``actions``, the four
    next-state label arrays, and ``lost_life``, ``terminal``, and ``won`` event
    arrays.  Only ``fields``, ``actions``, and optional ``initial_fields`` are
    passed to the model rollout.  Actual target fields are used for the separate
    encoder/readout reference and metrics only.
    """
    n = _validate_data(data)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    device = torch.device(device)
    if initial_fields is not None and tuple(initial_fields.shape) != (n, 148, 96):
        raise ValueError("initial_fields must have shape [N,148,96]")
    if initial_fields is not None and len(initial_fields) != n:
        raise ValueError("initial_fields length must match data")
    scale = _validate_scale(scale, device)
    model = model.to(device).eval()
    readout_metrics = {source: _empty_readout_metrics()
                       for source in ("predicted", "actual", "initial_copy")}
    event_metrics = _empty_events()
    group_metrics = _empty_groups()
    changed_metrics = _empty_changed()
    with _autocast(device):
        for begin in range(0, n, batch_size):
            end = min(begin + batch_size, n)
            current = _batch_value(data, "fields", begin, end, device, dtype=torch.float32)
            target = _batch_value(data, "next_fields", begin, end, device, dtype=torch.float32)
            actions = _validate_actions(_batch_value(data, "actions", begin, end, device))
            memory = None if initial_fields is None else _as_device(initial_fields[begin:end], device,
                                                                     dtype=torch.float32)
            if memory is None:
                rollout = model.rollout(current, actions)
            else:
                rollout = model.rollout(current, actions, memory)
            predicted = rollout["fields"]
            if tuple(predicted.shape) != tuple(target.shape) or not bool(torch.isfinite(predicted).all()):
                raise ValueError("model rollout fields must be finite [B,4,148,96]")
            predicted_readout = rollout["readout"]
            actual_flat = target.reshape(end - begin, HORIZONS, 148, 96).reshape(-1, 148, 96)
            actual_flat_readout = model.readout(actual_flat)
            actual_readout = {name: value.reshape(end - begin, HORIZONS, *value.shape[1:])
                              for name, value in actual_flat_readout.items()}
            copy_flat_readout = model.readout(current)
            copy_readout = {name: value[:, None].expand(-1, HORIZONS, *value.shape[1:])
                            for name, value in copy_flat_readout.items()}
            targets = _readout_targets(data, begin, end, device)
            _accumulate_readouts(readout_metrics["predicted"], predicted_readout, targets)
            _accumulate_readouts(readout_metrics["actual"], actual_readout, targets)
            _accumulate_readouts(readout_metrics["initial_copy"], copy_readout, targets)
            _accumulate_events(event_metrics, rollout["events"], data, begin, end, device)
            _accumulate_groups(group_metrics, "predicted", predicted, target, scale)
            _accumulate_groups(group_metrics, "initial_copy", current[:, None], target, scale)
            previous_actual = torch.cat((current[:, None], target[:, :-1]), dim=1)
            _accumulate_changed(changed_metrics, "predicted", predicted, target, previous_actual)
            _accumulate_changed(changed_metrics, "initial_copy", current[:, None], target, previous_actual)

    return {
        "levels": n,
        "transitions": n * HORIZONS,
        "horizons": [1, 2, 3, 4],
        "batch_size": batch_size,
        "device": str(device),
        "autocast_dtype": "bfloat16" if device.type == "cuda" else None,
        "readout": {source: _finalize_readouts(values)
                     for source, values in readout_metrics.items()},
        "events": event_metrics,
        "field_mse": _finalize_groups(group_metrics),
        "changed_appearance": _finalize_changed(changed_metrics),
        "notes": [
            "Predicted readouts measure transition plus readout performance; actual-target readouts measure decoding separately and do not establish an encoder accuracy ceiling.",
            "Carried-joint cross entropy is the sum of the three factorized glyph-head NLLs per transition.",
            "Initial-copy readouts and field errors are a persistence baseline, copied from the current public field at every horizon.",
            "All metrics are additive sums/counts aggregated batchwise; no full-dataset predictions are retained.",
        ],
    }


__all__ = ["evaluate"]
