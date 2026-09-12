"""Balanced carried-glyph losses and generated-data diagnostics.

The transition model's ordinary glyph objective is dominated by persistence:
most chronological transitions keep all three carried attributes unchanged.
This module provides a small auxiliary loss that gives changed and unchanged
examples equal weight per attribute, plus an additive diagnostic accumulator.

The role partition is deliberately only a public-pixel diagnostic.  It uses
the learned role probabilities already present in the current field, the
current player cell, and the action index.  It is not an engine-derived
cycler/launcher label and must not be used as a training target.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.nn import functional as F

from pebby.ls20.names import ACTION_DELTAS


ATTRIBUTES = ("shape", "color", "rotation")
LOGIT_KEYS = tuple(f"carried_{name}_logits" for name in ATTRIBUTES)
CLASS_COUNTS = (6, 4, 4)
INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
ROLE_CHANNELS = slice(48, 56)
DIRECT_CYCLER_ROLE_INDICES = (2, 3, 4)
LAUNCHER_ROLE_INDEX = 5


def _target(value, name, shape, device):
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if tuple(value.shape) != tuple(shape) or value.dtype not in INTEGER_DTYPES or value.dtype == torch.bool:
        raise ValueError(f"{name} must be integer {tuple(shape)}")
    value = value.detach().to(device=device, dtype=torch.long)
    for column, limit in enumerate(CLASS_COUNTS):
        if bool(((value[..., column] < 0) | (value[..., column] >= limit)).any()):
            raise ValueError(f"{name} {ATTRIBUTES[column]} values must be in 0..{limit - 1}")
    return value


def _readout(readout, batch):
    if not isinstance(readout, Mapping):
        raise ValueError("readout must be a mapping with carried glyph logits")
    values = []
    for key, classes in zip(LOGIT_KEYS, CLASS_COUNTS):
        value = readout.get(key)
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != (batch, classes)
                or not value.is_floating_point()):
            raise ValueError(f"{key} must be floating [{batch},{classes}]")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{key} must be finite")
        values.append(value)
    return values


def balanced_glyph_loss(readout, current_triple, next_triple):
    """Return a changed-vs-unchanged balanced CE for the three glyph fields.

    ``current_triple`` and ``next_triple`` are detached before use.  For each
    attribute, the mean CE over changed rows and the mean CE over unchanged
    rows receive equal weight when both groups exist.  If one group is absent,
    its present mean is used.  The three per-attribute losses are then averaged.
    Only the three ``carried_*_logits`` entries are read from ``readout``.
    """
    if not isinstance(current_triple, torch.Tensor):
        current_triple = torch.as_tensor(current_triple)
    if not isinstance(next_triple, torch.Tensor):
        next_triple = torch.as_tensor(next_triple)
    if current_triple.ndim != 2 or tuple(current_triple.shape[1:]) != (3,):
        raise ValueError("current_triple must have shape [B,3]")
    if next_triple.ndim != 2 or next_triple.shape != current_triple.shape:
        raise ValueError("next_triple must have shape [B,3] matching current_triple")
    if len(current_triple) < 1:
        raise ValueError("triple batch must be nonempty")
    # Validate integer ranges before converting, and place detached labels next
    # to the logits so this helper also works with a CPU numpy-backed batch.
    first = next((value for value in readout.values() if isinstance(value, torch.Tensor)), None) \
        if isinstance(readout, Mapping) else None
    if first is None:
        raise ValueError("readout must contain tensor logits")
    current = _target(current_triple, "current_triple", (len(current_triple), 3), first.device)
    target = _target(next_triple, "next_triple", (len(next_triple), 3), first.device)
    logits = _readout(readout, len(current))

    losses = []
    for column, scores in enumerate(logits):
        per_row = F.cross_entropy(scores, target[:, column], reduction="none")
        changed = target[:, column] != current[:, column]
        group_means = []
        if bool(changed.any()):
            group_means.append(per_row[changed].mean())
        if bool((~changed).any()):
            group_means.append(per_row[~changed].mean())
        # A nonempty batch guarantees at least one of the two groups.
        losses.append(torch.stack(group_means).mean())
    return torch.stack(losses).mean()


def _metric():
    return {"correct_sum": 0, "count": 0}


def _add_metric(bucket, correct, count):
    bucket["correct_sum"] += int(correct)
    bucket["count"] += int(count)


def _finish_metric(bucket):
    return {**bucket, "accuracy": (bucket["correct_sum"] / bucket["count"]
                                    if bucket["count"] else None)}


class GlyphDiagnosticAccumulator:
    """Additive changed/unchanged glyph accuracy and public-role diagnostics."""

    def __init__(self):
        self.rows = 0
        self._attributes = {
            attribute: {state: {"predicted": _metric(), "persistence": _metric()}
                        for state in ("changed", "unchanged")}
            for attribute in ATTRIBUTES
        }
        self._partitions = {
            partition: {
                attribute: {state: {"predicted": _metric(), "persistence": _metric()}
                            for state in ("changed", "unchanged")}
                for attribute in ATTRIBUTES
            }
            for partition in ("direct_cycler", "launcher", "other")
        }
        self._role_partition_seen = False

    @staticmethod
    def _public_role_partition(current_fields, player_cell, actions):
        if (not isinstance(current_fields, torch.Tensor) or current_fields.ndim != 3
                or tuple(current_fields.shape[1:]) != (148, 96)
                or not current_fields.is_floating_point()):
            raise ValueError("current_fields must be floating [B,148,96]")
        if not bool(torch.isfinite(current_fields).all()):
            raise ValueError("current_fields must be finite")
        batch = len(current_fields)
        if not isinstance(player_cell, torch.Tensor):
            player_cell = torch.as_tensor(player_cell)
        if (tuple(player_cell.shape) != (batch, 2) or player_cell.dtype not in INTEGER_DTYPES
                or player_cell.dtype == torch.bool):
            raise ValueError(f"player_cell must be integer [{batch},2]")
        player = player_cell.detach().to(device=current_fields.device, dtype=torch.long)
        if bool(((player < 0) | (player > 11)).any()):
            raise ValueError("player_cell coordinates must be in 0..11")
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        if (tuple(actions.shape) != (batch,) or actions.dtype not in INTEGER_DTYPES
                or actions.dtype == torch.bool):
            raise ValueError(f"actions must be integer [{batch}]")
        actions = actions.detach().to(device=current_fields.device, dtype=torch.long)
        if bool(((actions < 0) | (actions >= len(ACTION_DELTAS))).any()):
            raise ValueError("actions must be in 0..3")
        deltas = torch.as_tensor(ACTION_DELTAS, device=current_fields.device, dtype=torch.long)
        target_cell = player + deltas[actions]
        valid = ((target_cell >= 0) & (target_cell < 12)).all(-1)
        flat = (target_cell[:, 1] * 12 + target_cell[:, 0]).clamp(0, 143)
        role_index = current_fields[torch.arange(batch, device=current_fields.device), flat,
                                    ROLE_CHANNELS].argmax(-1)
        partition = torch.full((batch,), 2, dtype=torch.long, device=current_fields.device)
        direct_roles = torch.as_tensor(DIRECT_CYCLER_ROLE_INDICES, device=current_fields.device)
        partition[valid & torch.isin(role_index, direct_roles)] = 0
        partition[valid & (role_index == LAUNCHER_ROLE_INDEX)] = 1
        return partition

    def update(self, readout, current_triple, next_triple, *, current_fields=None,
               player_cell=None, actions=None):
        """Accumulate one H1 batch; labels are used only for scoring.

        The three role-partition inputs must either all be supplied or all be
        omitted.  When supplied, the resulting partition is based on the
        current field's public role argmax at the action target.  It is labeled
        ``public_role_argmax`` in the output to prevent treating it as engine
        ground truth.
        """
        if not isinstance(current_triple, torch.Tensor):
            current_triple = torch.as_tensor(current_triple)
        if not isinstance(next_triple, torch.Tensor):
            next_triple = torch.as_tensor(next_triple)
        if (current_triple.ndim != 2 or tuple(current_triple.shape[1:]) != (3,)
                or next_triple.ndim != 2 or tuple(next_triple.shape) != tuple(current_triple.shape)):
            raise ValueError("current_triple and next_triple must both have shape [B,3]")
        batch = len(current_triple)
        if batch < 1:
            raise ValueError("triple batch must be nonempty")
        first = next((value for value in readout.values() if isinstance(value, torch.Tensor)), None) \
            if isinstance(readout, Mapping) else None
        if first is None:
            raise ValueError("readout must contain tensor logits")
        current = _target(current_triple, "current_triple", (batch, 3), first.device)
        target = _target(next_triple, "next_triple", (batch, 3), first.device)
        logits = _readout(readout, batch)
        predicted = torch.stack([scores.detach().argmax(-1) for scores in logits], -1)
        current = current.detach()
        target = target.detach()
        changed = target != current
        persistence = current
        for column, attribute in enumerate(ATTRIBUTES):
            for state, mask in (("changed", changed[:, column]), ("unchanged", ~changed[:, column])):
                count = int(mask.sum())
                if not count:
                    continue
                _add_metric(self._attributes[attribute][state]["predicted"],
                            (predicted[mask, column] == target[mask, column]).sum(), count)
                _add_metric(self._attributes[attribute][state]["persistence"],
                            (persistence[mask, column] == target[mask, column]).sum(), count)

        supplied = (current_fields is not None, player_cell is not None, actions is not None)
        if any(supplied) and not all(supplied):
            raise ValueError("current_fields, player_cell, and actions must be supplied together")
        if all(supplied):
            self._role_partition_seen = True
            partition = self._public_role_partition(current_fields, player_cell, actions)
            for partition_name, partition_index in (("direct_cycler", 0), ("launcher", 1), ("other", 2)):
                rows = partition == partition_index
                for column, attribute in enumerate(ATTRIBUTES):
                    for state, mask in (("changed", changed[:, column]), ("unchanged", ~changed[:, column])):
                        mask = rows & mask
                        count = int(mask.sum())
                        if not count:
                            continue
                        _add_metric(self._partitions[partition_name][attribute][state]["predicted"],
                                    (predicted[mask, column] == target[mask, column]).sum(), count)
                        _add_metric(self._partitions[partition_name][attribute][state]["persistence"],
                                    (persistence[mask, column] == target[mask, column]).sum(), count)
        self.rows += batch
        return self

    @staticmethod
    def _finalize(groups):
        def visit(value):
            if set(value) == {"correct_sum", "count"}:
                return _finish_metric(value)
            return {name: visit(child) for name, child in value.items()}
        return visit(groups)

    def summary(self):
        """Return JSON-ready additive counts, sums, and accuracies."""
        output = {"rows": self.rows, "attributes": self._finalize(self._attributes)}
        output["role_partition"] = (self._finalize(self._partitions)
                                     if self._role_partition_seen else None)
        output["role_partition_definition"] = (
            "public_role_argmax: direct_cycler when the current field's target-cell role argmax is "
            "cycler_shape/color/rotation, launcher when it is launcher, and other otherwise; "
            "this is a learned public-role prediction, not engine ground truth."
        )
        return output


__all__ = ["ATTRIBUTES", "GlyphDiagnosticAccumulator", "balanced_glyph_loss"]
