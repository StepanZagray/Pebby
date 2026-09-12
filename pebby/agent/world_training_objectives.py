"""Current world-training objectives, separate from the frozen encoder source.

Structured checkpoints bind world_model.py byte-for-byte. Its historical loss
API remains available for reproducibility; active trainers and scorers import
this module to learn from failure events and zero-optimal-action states.
The model architecture, forward pass and checkpoint weights are unchanged.
"""

import math

import torch
from torch.nn import functional as F

from .world_model import (ACTION_COUNT, CELLS, DEFAULT_WEIGHTS, FRAME_SIZE,
                          GLYPH_FIELDS, GRID_COLS, GlyphEncoder, _rank_of_true,
                          glyph_labels, optimal_bits, successor_policy_masks)


def _value_loss(model, latent, distances, terminal, won, lost_life=None):
    distance_logits, terminal_logit, won_logit = model.value(latent)
    unreachable = distances < 0
    if lost_life is not None:
        unreachable = unreachable | lost_life.bool()
    bins = torch.where(unreachable, torch.full_like(distances, model.bins - 1),
                       distances.clamp(0, model.bins - 2))
    loss = (F.cross_entropy(distance_logits, bins)
            + F.binary_cross_entropy_with_logits(terminal_logit, terminal.float())
            + F.binary_cross_entropy_with_logits(won_logit, won.float()))
    return loss, (distance_logits.argmax(-1) == bins).float().mean()


def _risk_diagnostics(model, latent, unsafe, prefix=""):
    """Report target support alongside recall: no positives is not success."""
    probabilities = model.value(latent)[0].softmax(-1)[..., -1]
    predicted = probabilities >= .5
    safe = ~unsafe
    return {prefix + "unsafe_fraction": unsafe.float().mean(),
            prefix + "unsafe_recall": (predicted & unsafe).sum() / unsafe.sum().clamp_min(1),
            prefix + "safe_specificity": (~predicted & safe).sum() / safe.sum().clamp_min(1)}


def world_losses(model, batch, weights=None, *, loops=None, sigreg_generator=None):
    """Every training term from one batch of the NPZ contract.

    ``batch`` holds tensors ``frames [B, H, 64, 64]``, ``history_valid [B, H]``,
    ``previous_actions [B, H]``, ``next_frames [B, 4, 64, 64]``, ``terminal``
    and ``won`` ``[B, 4]``, ``optimal [B]`` bitmask, ``distances [B, 4]`` and
    optionally ``player_cell [B, 2]`` (col, row) or None. A positive
    ``successor_policy`` weight requires integer ``next_optimal [B, 4]``
    masks (0..15); zero masks are excluded and terminal masks must be zero.
    Optional rollout_mask [B] and rollout_actions [B,4] turn selected rows into
    four chronological steps; only step four may terminate or reset a life. Current policy
    still ranks all four first-action successors. Sequence labels are chronological.
    This auxiliary policy reuses actual successor encodings and imagines
    their own next latents; chronological rows separately supervise multistep predictions.
    Labels never touch logits: each policy sees only its own public history.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    frames, history_valid, previous_actions = model._prepare(
        batch["frames"], batch["history_valid"], batch["previous_actions"])
    device = frames.device
    batch_size, history = frames.shape[:2]
    from .world_rollout import (rollout_contract, replace_sequence_histories,
                                mixed_predictions, prediction_diagnostics, diagnostic_weights)
    rollout_mask, rollout_actions = rollout_contract(batch, batch_size, device)
    next_frames = torch.as_tensor(batch["next_frames"], device=device).long()
    if next_frames.shape != (batch_size, ACTION_COUNT, FRAME_SIZE, FRAME_SIZE):
        raise ValueError(f"next_frames must be [B, 4, 64, 64], got {tuple(next_frames.shape)}")
    terminal = torch.as_tensor(batch["terminal"], device=device).bool()
    won = torch.as_tensor(batch["won"], device=device).bool()
    distances = torch.as_tensor(batch["distances"], device=device).long()
    optimal = torch.as_tensor(batch["optimal"], device=device)

    if not math.isfinite(weights["successor_policy"]) or weights["successor_policy"] < 0:
        raise ValueError("successor_policy weight must be finite and nonnegative")
    next_optimal = None
    if weights["successor_policy"] > 0:
        next_optimal = successor_policy_masks(batch, batch_size, device, terminal)

    # Glyph perception: the current crop feeds the policy branch, the actual
    # successor crops feed only the target branch (one shared encoder, gradients on).
    current_glyph = next_glyph = None
    if model.cfg.glyph_recall:
        triples = glyph_labels(batch, batch_size, device)  # fail closed before any forward
        current_glyph = model.glyph_logits(frames[:, -1])
        next_glyph = model.glyph_logits(next_frames.flatten(0, 1))  # [B*4, 14]

    # Current observation: one encoder pass.
    tokens = model.frame_tokens(frames.flatten(0, 1)).view(batch_size, history, model.tokens, -1)
    current = model.assemble(tokens, history_valid, previous_actions, loops, glyph_logits=current_glyph)

    # Target observations: the history shifted by one, ending in each counterfactual
    # successor, through the SAME encoder (shared frame tokens, no stop-gradient).
    successor_tokens = model.frame_tokens(next_frames.flatten(0, 1)).view(
        batch_size, ACTION_COUNT, 1, model.tokens, -1)
    tail = tokens[:, 1:][:, None].expand(-1, ACTION_COUNT, -1, -1, -1)
    target_tokens = torch.cat((tail, successor_tokens), dim=2).flatten(0, 1)
    all_actions = torch.arange(ACTION_COUNT, device=device)[None].expand(batch_size, -1)
    target_valid = torch.cat((history_valid[:, 1:][:, None].expand(-1, ACTION_COUNT, -1),
                              torch.ones(batch_size, ACTION_COUNT, 1, dtype=torch.bool, device=device)),
                             dim=2).flatten(0, 1)
    target_actions = torch.cat((previous_actions[:, 1:][:, None].expand(-1, ACTION_COUNT, -1),
                                all_actions[..., None]), dim=2).flatten(0, 1)
    if batch.get("lost_life") is not None:
        reset = torch.as_tensor(batch["lost_life"], device=device).bool().flatten()
        if reset.shape != (batch_size * ACTION_COUNT,):
            raise ValueError("lost_life must be [B, 4]")
        # Match collector and inference: repeated reset frame, only the final
        # slot valid, no producing action retained across a lost-life boundary.
        reset_tokens = successor_tokens.flatten(0, 1).expand(-1, history, -1, -1)
        target_tokens = torch.where(reset[:, None, None, None], reset_tokens, target_tokens)
        reset_valid = torch.zeros_like(target_valid)
        reset_valid[:, -1] = True
        target_valid = torch.where(reset[:, None], reset_valid, target_valid)
        target_actions = torch.where(reset[:, None], -torch.ones_like(target_actions), target_actions)
    target_tokens, target_valid, target_actions = replace_sequence_histories(
        target_tokens, target_valid, target_actions, tokens, successor_tokens,
        history_valid, previous_actions, rollout_mask, rollout_actions, batch.get("lost_life"))
    actual_encoding = model.assemble(target_tokens, target_valid, target_actions, loops,
                                     glyph_logits=next_glyph)
    targets = actual_encoding["latent"].view(batch_size, ACTION_COUNT, -1)

    # LeWM terms.
    first_successors = model.predict_successors(current["latent"])  # four first-action alternatives
    predicted = mixed_predictions(model, current["latent"], first_successors,
                                  rollout_mask, rollout_actions)
    prediction = F.mse_loss(predicted, targets)
    slots = torch.cat((current["latent"][None], targets.transpose(0, 1)), dim=0)  # [5, B, D]
    sigreg = model.sigreg(slots, generator=sigreg_generator)

    # Task-specific terms.
    logits, extra = model.logits_from(current, successors=first_successors)
    bits = optimal_bits(optimal)
    valid_policy = bits.sum(1) > 0
    target_policy = bits / bits.sum(1, keepdim=True).clamp_min(1.)
    log_probabilities = F.log_softmax(logits, dim=-1)
    policy = -(target_policy * log_probabilities).sum() / valid_policy.sum().clamp_min(1)
    lost_life = (torch.as_tensor(batch["lost_life"], device=device).bool().flatten()
                 if batch.get("lost_life") is not None else None)
    value, distance_accuracy = _value_loss(model, targets.flatten(0, 1), distances.flatten(),
                                           terminal.flatten(), won.flatten())
    # Immediate loss is an action event, not a property of the reset state.
    # Only imagined successors retain the originating state/action context.
    # The existing final bin supplies this event signal without a new head.
    imagined_value, imagined_accuracy = _value_loss(model, predicted.flatten(0, 1), distances.flatten(),
                                                    terminal.flatten(), won.flatten(), lost_life)
    losses = {"prediction": prediction, "sigreg": sigreg, "policy": policy, "value": value,
              "imagined_value": imagined_value}
    grounding_diagnostics = {}
    if next_optimal is not None:
        # Reuse the actual encodings. Their policy imagines its own successors;
        # teacher labels and actual future images never enter the current logits.
        next_logits, _ = model.logits_from(actual_encoding)
        valid_next = next_optimal != 0
        next_bits = optimal_bits(next_optimal[valid_next])
        next_targets = next_bits / next_bits.sum(-1, keepdim=True).clamp_min(1.)
        selected_logits = next_logits[valid_next]
        losses["successor_policy"] = (-(next_targets * F.log_softmax(selected_logits, dim=-1))
                                      .sum() / valid_next.sum().clamp_min(1))
        with torch.no_grad():
            correct = next_bits.gather(1, selected_logits.argmax(-1)[:, None]).sum()
            grounding_diagnostics["successor_policy_set_accuracy"] = correct / valid_next.sum().clamp_min(1)
            grounding_diagnostics["successor_policy_valid_fraction"] = valid_next.float().mean()
    if model.cfg.grounding:
        from .world_grounding import world_grounding_losses
        losses["grounding"], latent_diagnostics = world_grounding_losses(
            model, batch, current["latent"], targets, predicted)
        grounding_diagnostics.update(latent_diagnostics)
    if model.cfg.glyph_recall:
        # Visual classification of the current and the four ACTUAL next crops;
        # imagined successors have no pixels and get no glyph term.
        glyph_logits_all = torch.cat((current_glyph, next_glyph), dim=0)
        losses["glyph"] = GlyphEncoder.loss(glyph_logits_all, triples)
        with torch.no_grad():
            for prefix, scores, labels in (("current", current_glyph, triples[:batch_size]),
                                           ("actual", next_glyph, triples[batch_size:])):
                per_field = GlyphEncoder.accuracies(scores, labels)[0]
                grounding_diagnostics.update({f"glyph_{prefix}_{name}_accuracy": value
                                              for name, value in zip(GLYPH_FIELDS, per_field)})
    player_cell = batch.get("player_cell")
    if player_cell is not None:
        player_cell = torch.as_tensor(player_cell, device=device).long()
        index = player_cell[:, 1] * GRID_COLS + player_cell[:, 0]
        if bool(((index < 0) | (index >= CELLS)).any()):
            raise ValueError("player_cell must hold (col, row) inside the 12x12 grid")
        losses["player"] = F.cross_entropy(extra["player"], index)
    total = sum(weights[name] * value_ for name, value_ in losses.items())

    with torch.no_grad():
        prediction_metrics = prediction_diagnostics(
            model, current["latent"], predicted, targets, rollout_mask, rollout_actions, _rank_of_true)
        chosen = logits.argmax(-1)
        unreachable = distances.flatten() < 0
        unsafe = unreachable.clone()
        if lost_life is not None:
            unsafe |= lost_life
        life_events = lost_life if lost_life is not None else torch.zeros_like(unsafe)
        deaths = (terminal & ~won).flatten()
        imagined_unsafe = model.value(predicted.flatten(0, 1))[0].softmax(-1)[..., -1] >= .5
        diagnostics = {
            **_risk_diagnostics(model, targets.flatten(0, 1), unreachable, "actual_"),
            **_risk_diagnostics(model, predicted.flatten(0, 1), unsafe, "imagined_"),
            "unsafe_fraction": unsafe.float().mean(),
            "life_loss_event_fraction": (lost_life.float().mean() if lost_life is not None
                                          else distances.new_tensor(0., dtype=torch.float32)),
            "terminal_death_fraction": (terminal & ~won).float().mean(),
            "imagined_life_loss_event_recall": (imagined_unsafe & life_events).sum() / life_events.sum().clamp_min(1),
            "imagined_terminal_death_recall": (imagined_unsafe & deaths).sum() / deaths.sum().clamp_min(1),
            **grounding_diagnostics,
            **prediction_metrics,
            "target_variance_mean": targets.flatten(0, 1).var(0, correction=0).mean(),
            "target_variance_min": targets.flatten(0, 1).var(0, correction=0).min(),
            "set_accuracy": bits.gather(1, chosen[:, None]).sum() / valid_policy.sum().clamp_min(1),
            "optimal_probability": (bits * log_probabilities.exp()).sum() / valid_policy.sum().clamp_min(1),
            "policy_valid_fraction": valid_policy.float().mean(),
            # The last reachable bin is an overflow bin; checkpoint heads retain
            # their dimensions and cannot distinguish distances above this cap.
            "distance_overflow_fraction": (distances > model.bins - 2).float().mean(),
            "distance_accuracy": distance_accuracy, "imagined_distance_accuracy": imagined_accuracy,
            "lookahead_span": (extra["features"][..., 0].max(-1).values
                               - extra["features"][..., 0].min(-1).values).mean(),
        }
        if player_cell is not None:
            diagnostics["player_accuracy"] = (extra["player"].argmax(-1) == index).float().mean()
    metric_weights = (diagnostic_weights(rollout_mask, diagnostics) if rollout_mask is not None else {})
    metric_weights.update({name: valid_policy.sum() for name in
                           ("policy", "set_accuracy", "optimal_probability")})
    for prefix, labels in (("actual_", unreachable), ("imagined_", unsafe)):
        metric_weights[prefix + "unsafe_recall"] = labels.sum()
        metric_weights[prefix + "safe_specificity"] = (~labels).sum()
    metric_weights["imagined_life_loss_event_recall"] = life_events.sum()
    metric_weights["imagined_terminal_death_recall"] = deaths.sum()
    return {"total": total, "losses": losses, "diagnostics": diagnostics, "logits": logits,
            "latent": current["latent"], "targets": targets, "predicted": predicted,
            "glyph_logits": current_glyph,
            "diagnostic_weights": metric_weights}
