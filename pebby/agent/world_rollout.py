"""Optional closing-event K=4 sequence rows for the draft mixed world-model loss.

The first three transitions must stay live; only the fourth may terminate or
reset a life. Collectors exclude level and expert-segment boundaries. Ordinary
rows retain four independent first-action alternatives. No teacher labels are
used to produce current policy logits.
"""
import torch
from torch.nn import functional as F


def rollout_contract(batch, batch_size, device):
    mask, actions = batch.get('rollout_mask'), batch.get('rollout_actions')
    if mask is None and actions is None:
        return None, None
    if mask is None or actions is None:
        raise ValueError('rollout_mask and rollout_actions must be supplied together')
    mask = torch.as_tensor(mask, device=device)
    actions = torch.as_tensor(actions, device=device)
    if mask.dtype != torch.bool or tuple(mask.shape) != (batch_size,):
        raise ValueError('rollout_mask must be boolean [B]')
    if tuple(actions.shape) != (batch_size, 4) or actions.dtype not in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError('rollout_actions must be integer [B, 4]')
    actions = actions.long()
    if bool(((actions[mask] < 0) | (actions[mask] > 3)).any()):
        raise ValueError('sequence rollout_actions must be in 0..3')
    for key in ('terminal', 'won', 'lost_life'):
        if batch.get(key) is not None:
            flags = torch.as_tensor(batch[key], device=device)
            if tuple(flags.shape) != (batch_size, 4):
                raise ValueError(f'{key} must be [B, 4]')
            if bool(flags[mask, :3].bool().any()):
                raise ValueError('rollout interior three transitions forbid terminal or life reset')
    return mask, actions


def chronological_histories(tokens, successors, valid, previous, actions):
    """Reuse current H tokens and four new frames to construct exact future H histories.

    Inputs are sequence rows only: tokens [S,H,P,C], successors [S,4,1,P,C],
    valid/previous [S,H], actions [S,4]. Returns batch-major [S,4,H,...].
    Appending before dropping old history makes this work even for H < K.
    """
    histories, validities, producing = [], [], []
    for horizon in range(4):
        tokens = torch.cat((tokens[:, 1:], successors[:, horizon]), dim=1)
        valid = torch.cat((valid[:, 1:], torch.ones_like(valid[:, :1])), dim=1)
        previous = torch.cat((previous[:, 1:], actions[:, horizon:horizon+1]), dim=1)
        histories.append(tokens)
        validities.append(valid)
        producing.append(previous)
    return tuple(torch.stack(values, dim=1) for values in (histories, validities, producing))


def replace_sequence_histories(target_tokens, target_valid, target_actions,
                               tokens, successors, valid, previous, mask, actions, lost_life=None):
    """Replace only chronological rows; leave legacy counterfactual rows untouched."""
    if mask is None or not bool(mask.any()):
        return target_tokens, target_valid, target_actions
    batch, history = tokens.shape[:2]
    ids = mask.nonzero(as_tuple=True)[0]
    seq = chronological_histories(tokens[ids], successors[ids], valid[ids], previous[ids], actions[ids])
    # Counterfactual reset handling happens before this replacement in world_losses.
    # Reapply a final sequence reset AFTER constructing chronological histories.
    # These observed flags change only targets; autonomous predictions never see them.
    if lost_life is not None:
        flags = torch.as_tensor(lost_life, device=tokens.device).bool()
        if bool(flags[ids, -1].any()):
            reset = torch.zeros((len(ids), 4), dtype=torch.bool, device=tokens.device)
            reset[:, -1] = flags[ids, -1]
            reset_tokens = successors[ids].expand(-1, -1, history, -1, -1)
            reset_valid = torch.zeros_like(seq[1]); reset_valid[:, :, -1] = True
            seq = (torch.where(reset[:, :, None, None, None], reset_tokens, seq[0]),
                   torch.where(reset[:, :, None], reset_valid, seq[1]),
                   torch.where(reset[:, :, None], -torch.ones_like(seq[2]), seq[2]))
    ordinary = (target_tokens.view(batch, 4, history, *tokens.shape[2:]),
                target_valid.view(batch, 4, history), target_actions.view(batch, 4, history))
    return tuple(value.index_copy(0, ids, replacement).flatten(0, 1)
                 for value, replacement in zip(ordinary, seq))


def mixed_predictions(model, latent, first_successors, mask, actions):
    """Four chronological predictions on selected rows, recursively without detach.

    The separate first_successors tensor must still feed the CURRENT policy:
    its four slots are action alternatives even when the loss row is temporal.
    """
    if mask is None or not bool(mask.any()):
        return first_successors
    ids = mask.nonzero(as_tuple=True)[0]
    chosen = actions[ids]
    state = first_successors[ids].gather(
        1, chosen[:, :1, None].expand(-1, 1, latent.size(-1))).squeeze(1)
    predictions = [state]
    for horizon in range(1, 4):
        state = model.predict_successors(state, chosen[:, horizon])
        predictions.append(state)
    return first_successors.index_copy(0, ids, torch.stack(predictions, dim=1))


def prediction_diagnostics(model, current, predicted, targets, mask, actions, rank_function):
    """Exclude time slots from counterfactual ranking; expose temporal errors by horizon.

    Horizon copy baseline repeats the initial latent (an autonomous no-change
    rollout), rather than cheating by copying the previous actual future state.
    Rank keys are absent when there are no counterfactual rows; explicit counts
    prevent an undefined all-sequence rank from being presented as zero/perfect.
    """
    copy = current[:, None, :].expand_as(targets)
    cf = torch.ones(current.size(0), dtype=torch.bool, device=current.device) if mask is None else ~mask
    diagnostics = {'copy_mse': F.mse_loss(copy, targets)}
    if bool(cf.any()):
        rank, top1 = rank_function(predicted[cf], targets[cf])
        copy_rank, copy_top1 = rank_function(copy[cf], targets[cf])
        diagnostics.update(counterfactual_mean_rank=rank, counterfactual_top1=top1,
                           copy_mean_rank=copy_rank, copy_top1=copy_top1)
    shuffled_current = current.roll(1, dims=0)
    shuffled = model.predict_successors(shuffled_current)
    shuffled = mixed_predictions(model, shuffled_current, shuffled, mask, actions)
    diagnostics['shuffled_state_mse'] = F.mse_loss(shuffled, targets)
    if mask is not None:
        diagnostics.update(rollout_rows=mask.sum(), counterfactual_rows=cf.sum())
        if bool(mask.any()):
            for horizon in range(4):
                diagnostics[f'rollout_prediction_mse_h{horizon+1}'] = F.mse_loss(
                    predicted[mask, horizon], targets[mask, horizon])
                diagnostics[f'rollout_current_copy_mse_h{horizon+1}'] = F.mse_loss(
                    current[mask], targets[mask, horizon])
    return diagnostics


def diagnostic_weights(mask, diagnostics):
    """Eligible base-row counts for trainer sum(value*weight)/sum(weight).

    Ordinary diagnostics retain the trainer's normal B weighting. Counts are
    returned outside the diagnostics dict so they cannot become metric values.
    For a counterfactual rank each row has four alternatives; the shared factor
    four cancels in the weighted mean. Temporal metrics have one value per row.
    """
    counts = {}
    for name in ('counterfactual_mean_rank', 'counterfactual_top1', 'copy_mean_rank', 'copy_top1'):
        if name in diagnostics:
            counts[name] = (~mask).sum()
    for name in diagnostics:
        if name.startswith(('rollout_prediction_mse_h', 'rollout_current_copy_mse_h')):
            counts[name] = mask.sum()
    return counts
