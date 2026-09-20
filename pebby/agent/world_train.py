"""Train ``world_model.WorldPolicy`` on counterfactual LS20 transition shards.

Data contract (NPZ, one file per split; ``world_model.REQUIRED_ARRAYS``)
------------------------------------------------------------------------
``frames uint8[N, H, 64, 64]`` chronological public history, left repeat
padded; ``history_valid bool[N, H]``; ``previous_actions int64[N, H]`` (-1 on
the first frame and padding); ``next_frames uint8[N, 4, 64, 64]`` the actual
successor of every action at the current state; ``terminal bool[N, 4]``;
``won bool[N, 4]``; ``optimal uint8[N]`` bitmask of optimal actions;
``distances int16[N, 4]`` exact remaining actions after each action, -1 when
unreachable; ``seeds int32[N]`` level seed per state; optional ``player_cell
int16[N, 2]`` (col, row) of the current state, used only as auxiliary
supervision; optional ``next_optimal uint8[N, 4]`` action masks for the
actual successors (0 means terminal/unreachable/unlabelable; terminal masks
must be 0), required by positive ``--successor-policy-weight`` (default 0).
Successor policy CE and accuracy average only nonzero masks; valid fraction
reports label coverage. These labels supervise the already encoded actual
successors, with no extra image encoder pass or multistep world prediction.
Alternatively, ``--train-successor-labels`` / ``--validation-successor-labels``
attach a small ``pebby.ls20-successor-labels.v1`` NPZ containing ``next_optimal``,
row-ordered ``seeds`` and JSON ``meta``. The loader requires generated-only
source metadata, the exact source NPZ SHA256 and matching integer rows/levels
counts, checks all rows before subsampling, and records both sidecar digests
in checkpoint/report provenance. Existing embedded labels cannot be replaced.
Optional ``meta`` JSON string. ``frames uint8[N, 64, 64]`` is
accepted as ``H = 1``. Longer histories than ``--history`` keep the newest
frames.

What the run refuses
--------------------
Missing arrays, shape mismatches, a validation file sharing any level seed
with training, and an empty split.

What gets reported (per epoch, train and validation)
----------------------------------------------------
Per-component losses (prediction, sigreg, policy, value, imagined_value,
player), the target-embedding variance (mean and min per dimension across
the batch), the counterfactual rank of the true successor among the four
targets for the predictor and for the copy baseline ``z_t`` (top-1 of 0.25 is
chance; copy scores exactly chance), the copy-baseline MSE next to the
prediction MSE, optimal-set accuracy against the constant majority prior,
and the selection criterion that chose the kept epoch.

Supported commands
------------------
    uv run python -m pebby.agent.world_train --train data/world-train.npz \
        --validation data/world-validation.npz --epochs 10 --batch-size 64 \
        --device cuda --checkpoint-out checkpoints/ls20-world.pt \
        --report-out checkpoints/ls20-world.training.json
    uv run python -m pebby.agent.world_train --train shard.npz --epochs 1 \
        --batch-size 8 --device cpu --max-states 64 --channels 16 ...   (smoke)

Learned glyph perception (experimental, off by default)
------------------------------------------------------
``--glyph-recall`` builds ``WorldModelConfig.glyph_recall`` and requires
``current_triple`` / ``next_triple`` in every split (the visual glyph term).
``--initialize-checkpoint`` may add the flag to an existing checkpoint (new
projector columns and the context start at zero, outputs unchanged);
``--initialize-glyph-checkpoint`` then copies ONLY the classifier weights from
a ``pebby.agent.glyph_train`` checkpoint, whose generated-only provenance is
recorded as ``glyph_source`` and must not include a validation level.

Learned query readout (experimental, off by default)
----------------------------------------------------
``--query-readout`` builds ``WorldModelConfig.query_readout``: a task-specific
global attention readout over the current frame's raw and refined tokens
(``pebby.agent.world_readout``) whose zero-initialised residual correction is
added to the direct + ranker logits. ``--initialize-checkpoint`` may add the
flag to an existing checkpoint (alone or together with the flags above); the
migrated network reproduces the source's logits exactly until training.

Memory: batch 64 at the default config is ~4.7 GiB of float32 activations by
the estimate in ``WorldPolicy.activation_estimate_gib``; ``--checkpoint-loops``
recomputes loops in backward and divides that by roughly ``loops``. Only the
root process launches GPU training; this file never picks CUDA unless asked
(``--device auto`` reports what it chose).
"""

from ..ls20.provenance import generated_context, difficulty_provenance

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import uuid

import numpy as np
import torch

from .world_model import (ACTION_COUNT, DEFAULT_WEIGHTS, FRAME_SIZE, OPTIONAL_ARRAYS, REQUIRED_ARRAYS, WorldModelConfig, WorldPolicy, initialize_from_checkpoint, initialize_glyph_encoder, load_world_checkpoint, optimal_bits, parameter_groups, save_world_checkpoint)
from .world_training_objectives import world_losses
from .world_runtime import configure_execution
from .glyph_model import load_glyph_checkpoint
from .curriculum_sampling import CurriculumSampler, DEFAULT_START, DEFAULT_END
from .gameplay_gate import evaluate_sequential, assess_gameplay

CONFIG_FLAGS = ("channels", "blocks", "heads", "expansion", "loops", "history", "temporal_layers",
                "hud_channels", "hud_tokens", "latent", "reduce", "predictor_blocks", "predictor_hidden",
                "value_hidden", "max_distance", "lookahead_depth", "summary", "readout_hidden",
                "ranker_hidden", "sigreg_projections", "sigreg_knots")


def load_dataset(path, history=None, cache_dir=None):
    """Read and validate one NPZ split; returns numpy arrays keyed by name."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"missing data file: {path}")
    if cache_dir is None:
        source = np.load(path, allow_pickle=False)
    else:
        from .world_cache import cached_arrays
        source = cached_arrays(path, cache_dir, (*REQUIRED_ARRAYS, *OPTIONAL_ARRAYS, 'context_index', 'meta'))
    with source as archive:
        missing = [name for name in REQUIRED_ARRAYS if name not in archive.files]
        if missing:
            raise ValueError(f"{path} lacks required arrays: {missing}")
        data = {name: archive[name] for name in REQUIRED_ARRAYS}
        for name in (*OPTIONAL_ARRAYS, "context_index"):
            data[name] = archive[name] if name in archive.files else None
        data["meta"] = {}
        if "meta" in archive.files:
            try:
                data["meta"] = json.loads(str(archive["meta"]))
            except (ValueError, TypeError) as error:  # object array or malformed JSON
                print(f"Warning: {path} meta is not a JSON string ({error}); ignored", flush=True)
    frames = data["frames"]
    if frames.ndim == 3:
        frames = frames[:, None]
    count = len(frames)
    if count == 0:
        raise ValueError(f"{path} holds no states")
    if frames.ndim != 4 or frames.shape[-2:] != (FRAME_SIZE, FRAME_SIZE):
        raise ValueError(f"{path}: frames must be [N, H, 64, 64], got {frames.shape}")
    depth = frames.shape[1]
    expected = {"history_valid": (count, depth), "previous_actions": (count, depth),
                "next_frames": (count, ACTION_COUNT, FRAME_SIZE, FRAME_SIZE),
                "terminal": (count, ACTION_COUNT), "won": (count, ACTION_COUNT),
                "optimal": (count,), "distances": (count, ACTION_COUNT), "seeds": (count,)}
    if data.get("context_index") is not None:
        expected["context_index"] = (count,)
    if data["player_cell"] is not None:
        expected["player_cell"] = (count, 2)
    if data["next_optimal"] is not None:
        expected["next_optimal"] = (count, ACTION_COUNT)
    if data["lost_life"] is not None:
        expected["lost_life"] = (count, ACTION_COUNT)
    for name, shape in {"next_player_cell": (count, 4, 2), "current_triple": (count, 3),
                        "next_triple": (count, 4, 3), "current_steps": (count,),
                        "next_steps": (count, 4), "current_lives": (count,),
                        "next_lives": (count, 4)}.items():
        if data.get(name) is not None:
            expected[name] = shape
    for name, shape in expected.items():
        if data[name].shape != shape:
            raise ValueError(f"{path}: {name} must have shape {shape}, got {data[name].shape}")
    if history is not None and depth > history:
        frames = frames[:, -history:]
        data["history_valid"] = data["history_valid"][:, -history:]
        data["previous_actions"] = data["previous_actions"][:, -history:]
    if not np.asarray(data["history_valid"], dtype=bool)[:, -1].all():
        raise ValueError(f"{path}: the current (last) history slot must always be valid")
    masks = np.asarray(data["optimal"])
    if not np.issubdtype(masks.dtype, np.integer) or np.any((masks < 0) | (masks >= 1 << ACTION_COUNT)):
        raise ValueError(f"{path}: optimal must be an integer 4-bit action mask")
    if np.any(masks == 0):
        # A complete teacher has no surviving action from a dead-end state.
        # Life-loss branches can still reset to a solvable state; their distance
        # is intentionally not overwritten with an unreachable sentinel.
        lost = data.get("lost_life")
        unreachable = np.asarray(data["distances"]) < 0
        if lost is not None:
            unreachable |= np.asarray(lost, dtype=bool)
        if np.any((masks == 0) & ~unreachable.all(axis=1)):
            raise ValueError(f"{path}: empty optimal masks require unreachable or lost-life successors")
    validate_successor_labels(data, path)
    valid = np.asarray(data["history_valid"], dtype=bool)
    actions = data["previous_actions"]
    if np.any((actions < -1) | (actions >= ACTION_COUNT)) or np.any(actions[~valid] != -1):
        raise ValueError(f"{path}: history actions must be -1..3, with -1 in padding")
    if np.any(valid[:, :-1] & ~valid[:, 1:]):
        raise ValueError(f"{path}: history must be left padded with a contiguous valid suffix")
    data["frames"] = frames
    return data


def validate_successor_labels(data, path):
    """The same label contract applies to embedded arrays and attached sidecars."""
    masks = data.get("next_optimal")
    if masks is None:
        return
    if masks.shape != (len(data["frames"]), ACTION_COUNT):
        raise ValueError(f"{path}: next_optimal must have shape [N, 4]")
    if not np.issubdtype(masks.dtype, np.integer) or np.any((masks < 0) | (masks > 15)):
        raise ValueError(f"{path}: next_optimal must contain integer 4-bit masks in 0..15")
    if np.any(np.asarray(data["terminal"], dtype=bool) & (masks != 0)):
        raise ValueError(f"{path}: terminal successors must have next_optimal zero")


def attach_successor_labels(data, source_path, sidecar_path):
    """Attach labels bound to the exact full source archive, before subsampling."""
    sidecar_path = Path(sidecar_path)
    if data.get("next_optimal") is not None:
        raise ValueError(f"{source_path}: cannot replace existing next_optimal with a sidecar")
    try:
        with np.load(sidecar_path, allow_pickle=False) as archive:
            missing = {"next_optimal", "seeds", "meta"} - set(archive.files)
            if missing:
                raise ValueError(f"missing sidecar arrays: {sorted(missing)}")
            masks, seeds = archive["next_optimal"], archive["seeds"]
            meta = json.loads(str(archive["meta"]))
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"{sidecar_path}: invalid successor labels ({error})") from error
    if not isinstance(meta, dict) or meta.get("format") != "pebby.ls20-successor-labels.v1":
        raise ValueError(f"{sidecar_path}: unsupported successor labels format")
    if meta.get("source") != "generated_only":
        raise ValueError(f"{sidecar_path}: successor labels must be generated_only")
    if not np.issubdtype(seeds.dtype, np.integer) or not np.array_equal(seeds, data["seeds"]):
        raise ValueError(f"{sidecar_path}: seeds must match source rows in exact order")
    for name, expected in (("rows", len(data["frames"])), ("levels", len(np.unique(data["seeds"])))):
        if type(meta.get(name)) is not int or meta[name] != expected:
            raise ValueError(f"{sidecar_path}: metadata {name} must equal {expected}")
    # Stream the source digest; never materialize another copy of its pixels.
    if meta.get("source_sha256") != file_digest(source_path):
        raise ValueError(f"{sidecar_path}: source_sha256 does not match {source_path}")
    validate_successor_labels({**data, "next_optimal": masks}, sidecar_path)
    provenance = {"path": str(sidecar_path), "sha256": file_digest(sidecar_path),
                  "format": meta["format"], "source": meta["source"],
                  "source_sha256": meta["source_sha256"], "rows": meta["rows"], "levels": meta["levels"]}
    return {**data, "next_optimal": masks,
            "meta": {**data.get("meta", {}), "successor_labels": provenance}}


def disjoint_seeds(train, validation):
    shared = np.intersect1d(np.unique(train["seeds"]), np.unique(validation["seeds"]))
    if len(shared):
        raise ValueError(f"{len(shared)} level seed(s) appear in both training and validation, "
                         f"e.g. {shared[:5].tolist()}; validation would be meaningless")


def require_verified_data(data):
    """Refuse missing proofs, source truncation, or a different gameplay context."""
    meta = data.get("meta", {})
    if meta.get("source") != "generated_only" or meta.get("oracle_search") != "complete_only":
        raise ValueError("verified training requires generated-only, complete-oracle provenance")
    levels = {int(level["seed"]): level for level in meta.get("levels", [])}
    for seed in np.unique(data["seeds"]):
        proof = levels.get(int(seed), {})
        if proof.get("context_engine_verified") is not True or proof.get("search_truncated") is not False:
            raise ValueError(f"seed {seed} lacks an untruncated contextual engine win proof")
        if proof.get("context_index") != generated_context(proof):
            raise ValueError(f"seed {seed} was verified in a different game context")
        if data.get("context_index") is not None and np.any(
                np.asarray(data["context_index"])[data["seeds"] == seed] != generated_context(proof)):
            raise ValueError(f"seed {seed} row context_index differs from its verified game context")


def require_winning_coverage(data, split="data"):
    """Require every row seed to have a real terminal winning successor.

    A metadata claim that a level is solvable is not enough for this guard:
    the loaded transition rows must contain at least one ``won`` action for
    every seed.  ``won`` is also required to imply ``terminal`` so malformed
    labels cannot satisfy the coverage check.
    """
    try:
        seeds = np.asarray(data["seeds"])
        won = np.asarray(data["won"], dtype=bool)
        terminal = np.asarray(data["terminal"], dtype=bool)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{split}: winning coverage needs seeds, won, and terminal arrays") from error
    if seeds.ndim != 1 or won.ndim != 2 or terminal.shape != won.shape \
            or len(seeds) != len(won):
        raise ValueError(f"{split}: winning coverage arrays have incompatible shapes")
    if np.any(won & ~terminal):
        raise ValueError(f"{split}: won successors must also be terminal")
    row_wins = np.any(won, axis=1)
    seeds_with_wins = np.unique(seeds[row_wins])
    missing = np.setdiff1d(np.unique(seeds), seeds_with_wins)
    if len(missing):
        example = ", ".join(str(int(seed)) for seed in missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise ValueError(f"{split}: level seeds lack an actual winning successor in won: "
                         f"{example}{suffix}")


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def training_objective_source():
    """Bind current training semantics separately from the frozen encoder."""
    path = Path(__file__).with_name("world_training_objectives.py").resolve()
    return {"module": world_losses.__module__, "path": str(path), "sha256": file_digest(path)}


def initial_state_sha256(model):
    """Fingerprint the actual initialization, including deterministic buffers."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def glyph_source_record(path, checkpoint, validation_seeds):
    """Provenance of a pretrained glyph encoder; refuses one trained on validation levels."""
    overlap = sorted(set(checkpoint["train_seeds"]) & set(validation_seeds))
    if overlap:
        raise ValueError(f"glyph checkpoint was trained on {len(overlap)} world validation level(s), "
                         f"e.g. {overlap[:5]}")
    return {"path": str(path), "sha256": file_digest(path), "format": checkpoint["format"],
            "config": checkpoint["config"], "parameters": checkpoint["parameters"],
            "source": checkpoint["source"], "counts": checkpoint["counts"], "results": checkpoint["results"],
            "train_levels": len(checkpoint["train_seeds"]), "validation_levels": len(checkpoint["validation_seeds"]),
            **{key: checkpoint[key] for key in ("train", "validation", "hashes", "steps", "batch_size", "seed")
               if key in checkpoint}}


def curriculum_metadata(curriculum):
    """Serialize a sampler's normalized endpoint ratios for checkpoints."""
    if curriculum is None:
        return None
    return {"start": [float(value) for value in curriculum.ratios(0.).tolist()],
            "end": [float(value) for value in curriculum.ratios(1.).tolist()],
            "unit": "distinct_level", "difficulty_version": curriculum.difficulty_version}


def as_tensors(data):
    def image_tensor(value):
        array = np.asarray(value)
        # Positive-stride history slices can share their disk-backed storage.
        # Torch only requires a copy for unsupported negative strides.
        if any(stride < 0 for stride in array.strides):
            array = np.ascontiguousarray(array)
        return torch.from_numpy(array)
    tensors = {"frames": image_tensor(data["frames"]),
               "history_valid": torch.from_numpy(np.asarray(data["history_valid"], dtype=bool)),
               "previous_actions": torch.from_numpy(np.asarray(data["previous_actions"], dtype=np.int64)),
               "next_frames": image_tensor(data["next_frames"]),
               "terminal": torch.from_numpy(np.asarray(data["terminal"], dtype=bool)),
               "won": torch.from_numpy(np.asarray(data["won"], dtype=bool)),
               "optimal": torch.from_numpy(np.asarray(data["optimal"], dtype=np.uint8)),
               "distances": torch.from_numpy(np.asarray(data["distances"], dtype=np.int64))}
    if data.get("next_optimal") is not None:
        tensors["next_optimal"] = torch.from_numpy(np.ascontiguousarray(data["next_optimal"]))
    if data.get("player_cell") is not None:
        tensors["player_cell"] = torch.from_numpy(np.asarray(data["player_cell"], dtype=np.int64))
    if data.get("lost_life") is not None:
        tensors["lost_life"] = torch.from_numpy(np.asarray(data["lost_life"], dtype=bool))
    from .world_grounding import FIELDS
    for name in FIELDS:
        if data.get(name) is not None:
            tensors[name] = torch.from_numpy(np.asarray(data[name], dtype=np.int64))
    return tensors


def subsample(data, limit, seed):
    if limit is None or limit >= len(data["frames"]):
        return data
    keep = np.sort(np.random.default_rng(seed).choice(len(data["frames"]), limit, replace=False))
    return {name: (value[keep] if isinstance(value, np.ndarray) else value) for name, value in data.items()}


def prior_baseline(train_optimal, validation_optimal=None):
    """Constant predictor fitted to the TRAINING optimal sets; scored on both splits."""
    def targets(masks):
        bits = ((np.asarray(masks, dtype=np.uint8)[:, None] >> np.arange(ACTION_COUNT)) & 1).astype(np.float64)
        return bits, bits / np.maximum(1., bits.sum(1, keepdims=True))

    _, train_target = targets(train_optimal)
    prior = train_target.mean(0)
    guess = int(prior.argmax())

    def score(masks):
        if masks is None:
            return None
        bits, target = targets(masks)
        return {"policy_cross_entropy": float(-(target * np.log(np.clip(prior, 1e-12, None))).sum(1).mean()),
                "set_accuracy": float(bits[:, guess].mean())}

    return {"prior": prior.tolist(), "train": score(train_optimal), "validation": score(validation_optimal)}


def schedule(optimizer, total_steps, warmup=.03):
    warmup_steps = max(1, int(warmup * total_steps))

    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return .5 * (1 + math.cos(math.pi * min(1., progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def batches(tensors, batch_size, generator=None, drop_last=False):
    count = len(tensors["frames"])
    order = torch.randperm(count, generator=generator) if generator is not None else torch.arange(count)
    for start in range(0, count, batch_size):
        index = order[start:start + batch_size]
        if drop_last and len(index) < batch_size:
            break
        yield {name: value[index] for name, value in tensors.items()}


def curriculum_batches(tensors, sampler, batch_size, generator, steps, start_step, total_steps,
                       on_policy_tensors=None):
    """Full batches of distinct levels; schedule advances with optimizer updates."""
    for step in range(steps):
        progress = (start_step + step) / max(1, total_steps - 1)
        index = sampler.indices(batch_size, progress, generator)
        if on_policy_tensors is None:
            yield {name: value[index] for name, value in tensors.items()}
        else:
            if getattr(sampler, 'sequence_index', None) is not None:
                mode = sampler.sequence_index.meta.get('mode')
                if mode == 'closing_only_train':
                    from .world_closing_sequences import closing_mixed_batch
                    yield closing_mixed_batch(tensors, on_policy_tensors, index, sampler.sequence_index,
                                              auxiliary_rows=sampler.auxiliary_indices)
                elif mode == 'on_policy_train':
                    from .world_sequences import four_step_mixed_batch
                    yield four_step_mixed_batch(tensors, on_policy_tensors, index, sampler.sequence_index,
                                                auxiliary_rows=sampler.auxiliary_indices)
                else:
                    raise ValueError('unsupported training sequence index mode')
            else:
                from .on_policy_sampling import mixed_batch
                yield mixed_batch(tensors, on_policy_tensors, index)


def _metric_packet(record, diagnostic_weights, batch, successor_enabled, norm=None):
    """One tensor-scalar readback; Python constants never move to the device.

    Deduplicate tensor objects (not values), retaining references until readback.
    FP64 packing preserves source floating values and physical-batch counts.
    CPU literals stay unchanged and are reconstructed after the transfer.
    """
    size = len(batch["frames"])
    valid = (batch["next_optimal"] != 0).sum() if successor_enabled else 0
    tensors, tensor_ids = [], {}

    def describe(value):
        if not isinstance(value, torch.Tensor):
            return (False, value)
        key = id(value)
        if key not in tensor_ids:
            tensor_ids[key] = len(tensors)
            tensors.append(value)
        return (True, tensor_ids[key])

    descriptors = [describe(valid), describe((norm > 1.) if norm is not None else 0)]
    for name, value in record.items():
        count = valid if name in ("successor_policy", "successor_policy_set_accuracy") else size
        descriptors.extend((describe(value), describe(diagnostic_weights.get(name, count))))
    values = (torch.stack([value.detach().to(torch.float64).reshape(()) for value in tensors])
              .cpu().tolist()) if tensors else []
    packet = [values[value] if is_tensor else value for is_tensor, value in descriptors]
    return int(packet[0]), int(packet[1]), list(zip(packet[2::2], packet[3::2]))


def run_epoch(model, tensors, device, weights, batch_size, optimizer=None, scheduler=None,
              generator=None, loops=None, precision="float32", drop_last=False,
              curriculum=None, steps_per_epoch=None, start_step=0, total_steps=None,
              on_policy_tensors=None):
    """One pass; without an optimizer this is the validation pass."""
    training = optimizer is not None
    if on_policy_tensors is not None and (not training or curriculum is None):
        raise ValueError('on-policy mixing requires training with a curriculum')
    model.train(training)
    sums, denominators, samples, clipped, steps = {}, {}, 0, 0, 0
    successor_valid = 0
    auxiliary_samples = on_policy_samples = 0
    # Deterministic SIGReg directions for validation so the number is repeatable.
    eval_generator = None if training else torch.Generator(device=device.type).manual_seed(0)
    stream = batches(tensors, batch_size, generator if training else None, drop_last=drop_last)
    difficulty_counts = {stage: 0 for stage in (curriculum.difficulties if curriculum else range(1, 6))}
    if curriculum is not None:
        if not training:
            raise ValueError("curriculum sampling is only for training")
        stream = curriculum_batches(tensors, curriculum, batch_size, generator,
                                    steps_per_epoch, start_step, total_steps, on_policy_tensors)
    for batch in stream:
        if curriculum is not None:
            auxiliary_samples += getattr(curriculum, 'last_auxiliary_count', 0)
            on_policy_samples += getattr(curriculum, 'last_on_policy_count', 0)
            for stage, count in curriculum.last_difficulty_counts.items():
                difficulty_counts[stage] += count
        batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
        norm = None
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
            out = world_losses(model, batch, weights, loops=loops, sigreg_generator=eval_generator)
        if training:
            out["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            steps += 1
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        size = len(batch["frames"])
        record = {"total": out["total"], **out["losses"], **out["diagnostics"]}
        valid_count, clip_count, metric_values = _metric_packet(
            record, out.get('diagnostic_weights', {}), batch,
            "successor_policy" in out["losses"], norm)
        successor_valid += valid_count
        clipped += clip_count
        for name, (value, count) in zip(record, metric_values):
            count = int(count)
            if name in ('rollout_rows', 'counterfactual_rows'):
                # These diagnostics are counts, not per-row means.
                sums[name] = sums.get(name, 0.) + value
                denominators[name] = 1
                continue
            sums[name] = sums.get(name, 0.) + count * value
            denominators[name] = denominators.get(name, 0) + count
        samples += size
    stats = {name: value / max(1, denominators[name]) for name, value in sums.items()}
    stats["samples"] = samples
    if training:
        stats["gradient_clip_fraction"] = clipped / max(1, steps)
    if curriculum is not None:
        stats["difficulty_counts"] = difficulty_counts
        stats["distinct_levels_per_batch"] = curriculum.last_distinct_levels
        if on_policy_tensors is not None:
            stats['on_policy_samples'] = on_policy_samples
            stats['on_policy_fraction'] = stats['on_policy_samples'] / samples
            stats['auxiliary_samples'] = auxiliary_samples
            stats['auxiliary_fraction'] = auxiliary_samples / samples
    return stats


def resolve_device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if requested == "cuda":
            raise SystemExit("CUDA was requested but torch.cuda.is_available() is False")
        print("CUDA unavailable -- training on CPU, which is only sane for a smoke test", flush=True)
        return torch.device("cpu")
    return torch.device("cuda")


def selection_score(stats, criterion):
    if criterion != 'gameplay':
        raise ValueError('offline checkpoint selection is disabled; use gameplay or explicit last')
    assessment = assess_gameplay(stats, stats)
    if assessment['status'] == 'no_evidence':
        raise ValueError('checkpoint selection requires complete gameplay evidence: ' + assessment['reason'])
    return -stats['levels_completed']


def selection_mode(value):
    if value not in ('gameplay', 'last'):
        raise argparse.ArgumentTypeError('offline checkpoint selection is disabled; use gameplay or '
                                         'explicit last (unpromoted artifact, gameplay not evaluated)')
    return value


def candidate_selection(entry, best, criterion):
    """Return a new retention record only for more actual progress; stable ties."""
    selection_mode(criterion)
    if criterion == 'last':
        score, assessment = -entry['epoch'], None
    else:
        score = selection_score(entry['gameplay'], criterion)
        assessment = assess_gameplay(best['gameplay'] if best else entry['gameplay'], entry['gameplay'])
        if assessment['status'] == 'no_evidence':
            raise ValueError('checkpoint gameplay protocols differ: ' + assessment['reason'])
        entry['gameplay_assessment'] = assessment
    if best is not None and score >= best['score']:
        return None
    return dict(epoch=entry['epoch'], score=score, criterion=criterion,
                selected_on='actual_sequential_gameplay' if criterion == 'gameplay' else 'last_epoch',
                gameplay=copy.deepcopy(entry.get('gameplay')),
                gameplay_status='evaluated' if criterion == 'gameplay' else 'gameplay_not_evaluated',
                objective_complete=bool(assessment and assessment['objective_evidence']),
                promoted=False, candidate_only=True, offline_diagnostics_only=True)


def save_candidate(path, model, **metadata):
    """Publish a new candidate atomically without replacing existing bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.world-candidate-', suffix='.pt',
                                         delete=False) as stream:
            temporary = Path(stream.name)
        result = save_world_checkpoint(temporary, model, **metadata)
        # Rebuilding the CPU checkpoint must not perturb the next training epoch's RNG.
        with torch.random.fork_rng(devices=[]):
            load_world_checkpoint(temporary)
        os.link(temporary, path)
        return result
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def output_paths(args):
    if args.checkpoint_out is None:
        args.checkpoint_out = Path('checkpoints/candidates') / f'ls20-world-{uuid.uuid4().hex}.pt'
    checkpoint = args.checkpoint_out
    return dict(checkpoint=checkpoint,
                last=checkpoint.with_name(checkpoint.stem + '.last.pt'),
                running=checkpoint.with_name(checkpoint.stem + '.running.pt'),
                progress=checkpoint.with_suffix('.progress.json'),
                report=args.report_out or checkpoint.with_suffix('.training.json'))


def verdict(history, baseline, best):
    """Plain statements about the KEPT epoch; every number next to its bar."""
    chosen = history[best["epoch"] - 1]
    split = "validation" if "validation" in chosen else "train"
    stats, bar = chosen[split], baseline[split]
    return {"split": split, "epoch": best["epoch"], "criterion": best["criterion"],
            "offline_diagnostics_only": True, "promoted": False, "candidate_only": True,
            "gameplay": chosen.get('gameplay'),
            "gameplay_status": 'evaluated' if chosen.get('gameplay') is not None else 'gameplay_not_evaluated',
            "policy_cross_entropy": stats["policy"], "prior_policy_cross_entropy": bar["policy_cross_entropy"],
            "beats_prior_cross_entropy": stats["policy"] < bar["policy_cross_entropy"],
            "set_accuracy": stats["set_accuracy"], "prior_set_accuracy": bar["set_accuracy"],
            "beats_prior_set_accuracy": stats["set_accuracy"] > bar["set_accuracy"],
            "prediction_mse": stats["prediction"], "copy_mse": stats["copy_mse"],
            "beats_copy_baseline": stats["prediction"] < stats["copy_mse"],
            "counterfactual_top1": stats["counterfactual_top1"], "copy_top1": stats["copy_top1"],
            "chance_top1": 1. / ACTION_COUNT,
            "target_variance_mean": stats["target_variance_mean"],
            "target_variance_min": stats["target_variance_min"]}


def describe(entry, baseline):
    parts = []
    for split in ("train", "validation"):
        if split not in entry:
            continue
        stats, bar = entry[split], baseline[split]
        parts.append(f"{split}: policy {stats['policy']:.4f} (prior {bar['policy_cross_entropy']:.4f}) "
                     f"set {stats['set_accuracy']:.3f} (prior {bar['set_accuracy']:.3f}) | "
                     f"pred {stats['prediction']:.4f} vs copy {stats['copy_mse']:.4f} | "
                     f"cf top1 {stats['counterfactual_top1']:.3f} (copy {stats['copy_top1']:.3f}) | "
                     f"sigreg {stats['sigreg']:.3f} var {stats['target_variance_mean']:.3f}"
                     f"/{stats['target_variance_min']:.3f} | value {stats['value']:.3f} "
                     f"imag {stats['imagined_value']:.3f}"
                     + (f" | glyph {stats['glyph']:.3f} cur s/c/r "
                        f"{stats['glyph_current_shape_accuracy']:.3f}/{stats['glyph_current_color_accuracy']:.3f}"
                        f"/{stats['glyph_current_rotation_accuracy']:.3f}" if "glyph" in stats else ""))
    return " || ".join(parts)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train", type=Path, required=True, help="NPZ of the training contract")
    parser.add_argument("--validation", type=Path, help="NPZ built from disjoint level seeds")
    parser.add_argument('--data-cache-dir', type=Path,
                        help='Source-hashed extracted NPY cache; memory-map data instead of holding all images in RAM')
    parser.add_argument('--on-policy-data', type=Path,
                        help='Generated model-visited supplemental rows from base training levels only')
    parser.add_argument('--on-policy-fraction', type=float, default=.25,
                        help='Fraction of each distinct-level batch reserved for model-visited rows')
    parser.add_argument('--on-policy-auxiliary-fraction', type=float, default=.25,
                        help='Chance to replace an eligible base draw with supplemental expert/failure rows; '
                             'keeps the reserved model-visited fraction unchanged')
    parser.add_argument('--closing-rollout-index', type=Path)
    parser.add_argument('--closing-action-attestation', type=Path)
    parser.add_argument('--closing-action-sha256')
    parser.add_argument('--rollout-index', type=Path,
                        help='Source-bound four-step training index for --on-policy-data')
    parser.add_argument("--train-successor-labels", type=Path,
                        help="Generated successor label sidecar bound to the exact training NPZ")
    parser.add_argument("--validation-successor-labels", type=Path,
                        help="Generated successor label sidecar bound to the exact validation NPZ")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--curriculum", action="store_true",
                        help="Sample distinct levels, shifting from easy-heavy to hard-heavy")
    parser.add_argument("--curriculum-start", type=float, nargs="+", default=None,
                        metavar="WEIGHT",
                        help="Curriculum start weights; default follows dataset difficulty_version")
    parser.add_argument("--curriculum-end", type=float, nargs="+", default=None,
                        metavar="WEIGHT",
                        help="Curriculum end weights; default follows dataset difficulty_version")
    parser.add_argument("--min-train-levels", type=int, default=1)
    parser.add_argument('--require-fresh-initialization', action='store_true',
                        help='Refuse all external model/glyph/cell initialization weights')
    parser.add_argument('--expected-initial-state-sha256',
                        help='Require the independently measured random initialization fingerprint')
    parser.add_argument("--require-verified-data", action="store_true",
                        help="Require per-level contextual engine replay proofs in both splits")
    parser.add_argument("--require-winning-coverage", action="store_true",
                        help="Require every split seed to have a terminal won successor")
    parser.add_argument("--drop-last", action="store_true", help="Use full training batches only; reshuffle each epoch")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="auto")
    parser.add_argument("--precision", choices=("float32", "bf16"), default="float32")
    parser.add_argument('--compile-core', action='store_true', help='Compile the repeated core during gradient-enabled training')
    parser.add_argument('--temporal-backend', choices=('auto', 'math', 'cudnn', 'flash'), default='auto')
    parser.add_argument("--checkpoint-out", type=Path,
                        help='New candidate path; defaults to a unique file under checkpoints/candidates')
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=.05, help="Matrices only; no LN/bias/embeddings")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grounding", action="store_true", help="Supervise latent visible state on generated labels")
    parser.add_argument("--state-recall", action="store_true",
                        help="Feed the raw current HUD tokens and the player-weighted refined cell "
                             "to the latent projector next to the reduced tokens")
    parser.add_argument("--glyph-recall", action="store_true",
                        help="Classify the current frame's carried-glyph crop with a learned GlyphEncoder; "
                             "broadcast its probabilities into the source tokens and the latent projector")
    parser.add_argument("--query-readout", action="store_true",
                        help="Add a learned global attention readout over the current frame's raw and "
                             "refined tokens; its residual four-action correction starts at zero")
    parser.add_argument('--cell-recall', action='store_true',
                        help='Add frozen learned cell appearance probabilities to spatial tokens')
    parser.add_argument('--initialize-cell-checkpoint', type=Path)
    parser.add_argument('--cell-appearance-data', type=Path,
                        help='Exact generated bank used to train the cell decoder')
    parser.add_argument('--cell-appearance-proof', type=Path,
                        help='Completed hash-bound proof for the cell decoder bank')
    parser.add_argument("--initialize-checkpoint", type=Path,
                        help="Initialize compatible weights; permits turning on --grounding "
                             "(new head), --state-recall and --glyph-recall (zero-padded projector columns) "
                             "and --query-readout (new head with a zero output layer)")
    parser.add_argument("--initialize-glyph-checkpoint", type=Path,
                        help="With --glyph-recall: copy a pretrained generated-only GlyphEncoder "
                             "(pebby.agent.glyph_train) into glyph_encoder after the world initialization")
    parser.add_argument("--max-states", type=int, help="Random subset of training states (first screen)")
    parser.add_argument("--max-validation-states", type=int)
    parser.add_argument('--require-exact-distances', action='store_true',
                        help='Reject finite distance labels outside the configured value head instead of clipping')
    parser.add_argument("--checkpoint-loops", action="store_true",
                        help="Recompute each loop during backward (same gradients, less memory)")
    parser.add_argument("--checkpoint-encoder", action="store_true",
                        help="Recompute frame stems and temporal/loop encoder to fit larger true batches")
    parser.add_argument("--encoder-chunk-size", type=int, default=0,
                        help="Bound encoder temporary memory; SIGReg still sees the entire batch")
    parser.add_argument("--select-on", type=selection_mode, default="gameplay",
                        help='Retain greatest actual sequential gameplay progress (stable ties); '
                             'explicit last keeps final unpromoted artifact without gameplay')
    for name, default in DEFAULT_WEIGHTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}-weight", type=float, default=default,
                            help=f"Loss weight for {name} (default {default})")
    defaults = WorldModelConfig()
    for flag in CONFIG_FLAGS:
        parser.add_argument(f"--{flag.replace('_', '-')}", type=int, default=getattr(defaults, flag))
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = output_paths(args)
    if len({path.resolve() for path in paths.values()}) != len(paths):
        parser.error('candidate checkpoint, last, running, progress and report paths must differ')
    for path in paths.values():
        if path.exists() or path.is_symlink():
            parser.error(f'output already exists; choose fresh candidate paths: {path}')
    if args.require_fresh_initialization and any((args.initialize_checkpoint,
            args.initialize_glyph_checkpoint,args.initialize_cell_checkpoint,args.cell_recall)):
        parser.error('--require-fresh-initialization forbids pretrained model, glyph, or cell weights')
    if args.expected_initial_state_sha256 and not args.require_fresh_initialization:
        parser.error('--expected-initial-state-sha256 requires --require-fresh-initialization')
    objective_source = training_objective_source()
    if args.validation_successor_labels and args.validation is None:
        parser.error("--validation-successor-labels requires --validation")
    if args.on_policy_data and not args.curriculum:
        parser.error('--on-policy-data requires --curriculum')
    if not args.on_policy_data and args.on_policy_fraction != .25:
        parser.error('--on-policy-fraction requires --on-policy-data')
    closing = (args.closing_rollout_index, args.closing_action_attestation, args.closing_action_sha256)
    if any(closing) and not all(closing):
        parser.error('--closing-rollout-index, --closing-action-attestation and --closing-action-sha256 are required together')
    if args.closing_rollout_index and (args.rollout_index or not args.on_policy_data or not args.curriculum):
        parser.error('closing rollout requires --on-policy-data and --curriculum and excludes --rollout-index')
    if args.rollout_index and not args.on_policy_data:
        parser.error('--rollout-index requires --on-policy-data')
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if args.batch_size > 1024 or args.batch_size & (args.batch_size - 1):
        parser.error("batch-size must be a power of two, capped at 1024")
    if args.encoder_chunk_size < 0:
        parser.error("encoder-chunk-size must not be negative")
    if not 0 < args.lr < 1 or args.weight_decay < 0:
        parser.error("lr must be in (0, 1) and weight-decay must not be negative")
    try:
        config = WorldModelConfig(**{flag: getattr(args, flag) for flag in CONFIG_FLAGS},
                                  grounding=args.grounding, state_recall=args.state_recall,
                                  glyph_recall=args.glyph_recall, query_readout=args.query_readout,
                                  cell_recall=args.cell_recall)
    except ValueError as error:
        parser.error(str(error))
    if args.initialize_glyph_checkpoint and not args.glyph_recall:
        parser.error("--initialize-glyph-checkpoint requires --glyph-recall")
    cell_inputs = (args.initialize_cell_checkpoint, args.cell_appearance_data, args.cell_appearance_proof)
    if any(cell_inputs) and (not all(cell_inputs) or not args.cell_recall):
        parser.error('cell checkpoint, data and proof require each other and --cell-recall')
    weights = {name: getattr(args, f"{name}_weight") for name in DEFAULT_WEIGHTS}
    if not math.isfinite(weights["successor_policy"]) or weights["successor_policy"] < 0:
        parser.error("successor-policy-weight must be finite and nonnegative")
    if weights["sigreg"] <= 0:
        print("Warning: sigreg weight is not positive; nothing prevents latent collapse.", flush=True)
    if args.lookahead_depth < 1:
        parser.error("lookahead-depth must be at least 1 so the predictor is used by the policy")

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = resolve_device(args.device)
    if args.precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        parser.error("bf16 training requires a CUDA device with native bfloat16 support")
    try:
        train_data = load_dataset(args.train, config.history, args.data_cache_dir)
        if args.train_successor_labels:
            train_data = attach_successor_labels(train_data, args.train, args.train_successor_labels)
        train_data = subsample(train_data, args.max_states, args.seed)
        validation_data = None
        if args.validation is not None:
            validation_data = load_dataset(args.validation, config.history, args.data_cache_dir)
            if args.validation_successor_labels:
                validation_data = attach_successor_labels(validation_data, args.validation,
                                                          args.validation_successor_labels)
            validation_data = subsample(validation_data, args.max_validation_states, args.seed)
            disjoint_seeds(train_data, validation_data)
        if len(np.unique(train_data["seeds"])) < args.min_train_levels:
            raise ValueError("training data has fewer distinct levels than --min-train-levels")
        if args.require_exact_distances:
            for split, data in (("training", train_data), ("validation", validation_data)):
                if data is not None and np.any(np.asarray(data['distances']) >= config.max_distance):
                    raise ValueError(f'{split} finite distance exceeds --max-distance; enlarge the fresh value head')
        if weights["successor_policy"] > 0:
            for split, data in (("training", train_data), ("validation", validation_data)):
                if data is not None and data.get("next_optimal") is None:
                    raise ValueError(f"--successor-policy-weight needs next_optimal in the {split} data")
        if config.glyph_recall:
            for split, data in (("training", train_data), ("validation", validation_data)):
                if data is not None and any(data.get(key) is None for key in ("current_triple", "next_triple")):
                    raise ValueError(f"--glyph-recall needs current_triple and next_triple in the {split} data")
        if args.require_verified_data:
            for data in (train_data, validation_data):
                if data is not None:
                    require_verified_data(data)
        if args.require_winning_coverage:
            require_winning_coverage(train_data, "training data")
            if validation_data is not None:
                require_winning_coverage(validation_data, "validation data")
        on_policy_data, on_policy_source = None, None
        if args.on_policy_data:
            from .on_policy_sampling import OnPolicySampler
            from .on_policy_provenance import validate_on_policy_provenance
            on_policy_data = load_dataset(args.on_policy_data, config.history, args.data_cache_dir)
            if args.require_exact_distances and np.any(np.asarray(on_policy_data['distances']) >= config.max_distance):
                raise ValueError('on-policy finite distance exceeds --max-distance; enlarge the fresh value head')
            require_verified_data(on_policy_data)
            require_winning_coverage(on_policy_data, 'on-policy data')
            if validation_data is not None:
                disjoint_seeds(on_policy_data, validation_data)
            meta = on_policy_data['meta']
            behavior_source = validate_on_policy_provenance(on_policy_data)
            on_policy_source = {'path': str(args.on_policy_data), 'sha256': file_digest(args.on_policy_data),
                                **behavior_source, 'fraction': args.on_policy_fraction,
                                'auxiliary_fraction': args.on_policy_auxiliary_fraction,
                                'auxiliary_rows': len(meta.get('auxiliary_rows', [])),
                                'rows': len(on_policy_data['seeds']),
                                'on_policy_rows': len(meta.get('on_policy_rows', [])),
                                'levels': len(np.unique(on_policy_data['seeds']))}
            if args.closing_rollout_index:
                from .world_closing_sequences import load_sidecar, ClosingSampler
                index = load_sidecar(args.closing_rollout_index, args.on_policy_data, on_policy_data,
                                     attestation=args.closing_action_attestation,
                                     attestation_sha256=args.closing_action_sha256)
                on_policy_source['rollout_index'] = {
                    'path': str(args.closing_rollout_index), 'sha256': file_digest(args.closing_rollout_index),
                    **index.meta}
                curriculum = ClosingSampler(train_data, on_policy_data, index,
                                             fraction=args.on_policy_fraction,
                                             auxiliary_fraction=args.on_policy_auxiliary_fraction,
                                             start=args.curriculum_start, end=args.curriculum_end)
            elif args.rollout_index:
                from .world_sequences import load_sidecar, FourStepSampler
                index = load_sidecar(args.rollout_index, args.on_policy_data, on_policy_data,
                                     mode='on_policy_train')
                on_policy_source['rollout_index'] = {
                    'path': str(args.rollout_index), 'sha256': file_digest(args.rollout_index),
                    **index.meta}
                curriculum = FourStepSampler(train_data, on_policy_data, index,
                                             fraction=args.on_policy_fraction,
                                             auxiliary_fraction=args.on_policy_auxiliary_fraction,
                                             start=args.curriculum_start, end=args.curriculum_end)
            else:
                curriculum = OnPolicySampler(train_data, on_policy_data, fraction=args.on_policy_fraction,
                                             auxiliary_fraction=args.on_policy_auxiliary_fraction,
                                             start=args.curriculum_start, end=args.curriculum_end)
        else:
            curriculum = (CurriculumSampler(train_data, start=args.curriculum_start,
                                             end=args.curriculum_end)
                          if args.curriculum else None)
        if curriculum is not None:
            curriculum.check_coverage(args.batch_size)
    except ValueError as error:
        parser.error(str(error))
    if args.batch_size < 16:
        print(f"Warning: batch size {args.batch_size} gives SIGReg a population of "
              f"{args.batch_size} per slot; the Gaussian match is noisy below ~32.", flush=True)

    model = WorldPolicy(config).to(device)
    initialization = (dict(kind='random',seed=args.seed,
                           weights_sha256=initial_state_sha256(model),optimizer_state='new')
                      if args.require_fresh_initialization else None)
    if args.expected_initial_state_sha256 and initialization['weights_sha256']!=args.expected_initial_state_sha256:
        parser.error('random initial weights differ from the independent preflight fingerprint')
    cell_source = None
    validation_seed_list = np.unique(validation_data['seeds']).tolist() if validation_data is not None else []
    if args.initialize_checkpoint:
        try:
            source_model, source_metadata = load_world_checkpoint(args.initialize_checkpoint)
            migrated = initialize_from_checkpoint(model, source_model)
            if source_model.cfg.cell_recall and not args.initialize_cell_checkpoint:
                from .world_cell_recall import verify_cell_source
                cell_source = verify_cell_source(source_metadata.get('cell_source'), validation_seed_list,
                                                 model.cell_appearance)
        except (ValueError, RuntimeError) as error:
            parser.error(f"initialization checkpoint {args.initialize_checkpoint}: {error}")
        print(f"Initialized weights from {args.initialize_checkpoint}"
              + (f" | zero-padded new input columns of {', '.join(migrated)}" if migrated else ""), flush=True)
        del source_model
    glyph_source = None
    if args.initialize_glyph_checkpoint:
        # After the world migration on purpose: only the classifier weights are
        # replaced; glyph_context and the new projector columns stay as migrated.
        try:
            glyph_encoder, glyph_checkpoint = load_glyph_checkpoint(args.initialize_glyph_checkpoint)
            copied = initialize_glyph_encoder(model, glyph_encoder)
            glyph_source = glyph_source_record(
                args.initialize_glyph_checkpoint, glyph_checkpoint,
                np.unique(validation_data["seeds"]).tolist() if validation_data is not None else [])
        except (ValueError, RuntimeError) as error:
            parser.error(f"glyph checkpoint {args.initialize_glyph_checkpoint}: {error}")
        print(f"Initialized {len(copied)} glyph encoder tensors from {args.initialize_glyph_checkpoint} "
              f"({glyph_source['train_levels']} generated train levels, validation results "
              f"{glyph_source['results'].get('validation')})", flush=True)
        del glyph_encoder, glyph_checkpoint
    if args.initialize_cell_checkpoint:
        from .world_cell_recall import load_cell_source, initialize_cell_encoder
        try:
            cell_encoder, cell_source = load_cell_source(args.initialize_cell_checkpoint,
                                                        args.cell_appearance_data,
                                                        args.cell_appearance_proof,
                                                        validation_seed_list)
            initialize_cell_encoder(model, cell_encoder)
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            parser.error(f'cell decoder initialization: {error}')
        del cell_encoder
        print(f"Initialized frozen cell decoder from {args.initialize_cell_checkpoint}; "
              f"{len(cell_source['train_seeds'])} generated training levels", flush=True)
    if args.cell_recall and cell_source is None:
        parser.error('--cell-recall needs a verified pretrained cell decoder or a world checkpoint carrying one')
    model.checkpoint_loops = args.checkpoint_loops
    model.checkpoint_encoder = args.checkpoint_encoder
    model.encoder_chunk_size = args.encoder_chunk_size
    execution = configure_execution(model, compile_core=args.compile_core,
                                    temporal_backend=args.temporal_backend)
    train_tensors = as_tensors(train_data)
    on_policy_tensors = as_tensors(on_policy_data) if on_policy_data is not None else None
    if on_policy_tensors is not None:
        if train_tensors.keys() != on_policy_tensors.keys() or any(
                train_tensors[name].shape[1:] != value.shape[1:] or train_tensors[name].dtype != value.dtype
                for name, value in on_policy_tensors.items()):
            parser.error('on-policy tensor fields and shapes must match base training data')
        print(f"On-policy mixture: {curriculum.reserved_count(args.batch_size)}/{args.batch_size} "
              f"distinct levels per batch from {on_policy_source['levels']} behavior levels; "
              'optimizer step count and majority prior use the base bank', flush=True)
    if args.drop_last and len(train_tensors['frames']) < args.batch_size:
        parser.error("drop-last would leave no training batch")
    validation_tensors = as_tensors(validation_data) if validation_data is not None else None
    baseline = prior_baseline(train_data["optimal"],
                              validation_data["optimal"] if validation_data is not None else None)
    optimizer = torch.optim.AdamW(parameter_groups(model, args.weight_decay), lr=args.lr)
    steps_per_epoch = (len(train_tensors["frames"]) // args.batch_size if args.drop_last or args.curriculum
                       else math.ceil(len(train_tensors["frames"]) / args.batch_size))
    if steps_per_epoch == 0:
        parser.error("no complete training batch")
    scheduler = schedule(optimizer, args.epochs * steps_per_epoch)
    generator = torch.Generator().manual_seed(args.seed)

    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"{device} ({name}) | {args.precision}, TF32 off | execution {execution} | {model.parameter_count():,} parameters | "
          f"estimated activations {model.activation_estimate_gib(args.batch_size):.2f} GiB at batch "
          f"{args.batch_size}{' with loop checkpointing' if args.checkpoint_loops else ''}", flush=True)
    train_seeds = sorted(int(s) for s in np.unique(train_data["seeds"]))
    validation_seeds = (sorted(int(s) for s in np.unique(validation_data["seeds"]))
                        if validation_data is not None else [])
    print(f"{len(train_tensors['frames']):,} training states over {len(train_seeds)} level seeds, "
          f"history {train_tensors['frames'].shape[1]} | "
          + (f"{len(validation_tensors['frames']):,} validation states over {len(validation_seeds)} "
             f"held-out seeds" if validation_tensors is not None else "no validation split")
          + (" | player_cell supervision" if "player_cell" in train_tensors else " | no player_cell"),
          flush=True)
    print(f"Majority prior {np.round(baseline['prior'], 3).tolist()} | train policy CE "
          f"{baseline['train']['policy_cross_entropy']:.4f} set {baseline['train']['set_accuracy']:.3f}"
          + (f" | validation policy CE {baseline['validation']['policy_cross_entropy']:.4f} "
             f"set {baseline['validation']['set_accuracy']:.3f}" if baseline["validation"] else ""), flush=True)
    print(f"Loss weights {weights} | loops {config.loops} x {config.blocks} blocks, full backpropagation"
          f"{' (recomputed per loop)' if args.checkpoint_loops else ''} | lookahead depth "
          f"{config.lookahead_depth}"
          + (f" | state recall: {model.recall_inputs} extra projector inputs" if config.state_recall else "")
          + (f" | glyph recall: {model.glyph_inputs} glyph probabilities into source tokens and projector"
             if config.glyph_recall else "")
          + (f" | query readout: {model.query_head.parameter_count():,} parameters, "
             f"{model.query_head.queries} queries x {len(model.query_head.blocks)} blocks over raw + refined tokens"
             if config.query_readout else ""),
          flush=True)

    successor_labels = {split: data.get("meta", {}).get("successor_labels")
                        for split, data in (("train", train_data), ("validation", validation_data))
                        if data is not None and data.get("meta", {}).get("successor_labels") is not None}
    history, best = [], None
    running_created = progress_created = False
    selection_protocol = dict(select_on=args.select_on, promoted=False, candidate_only=True,
                              trainer_source_sha256=file_digest(Path(__file__)),
                              shipped_gameplay_used_for_checkpoint_selection=args.select_on == 'gameplay',
                              offline_metrics_used_for_selection=False,
                              gameplay_status='evaluated' if args.select_on == 'gameplay' else 'gameplay_not_evaluated',
                              per_level_action_cap=300 if args.select_on == 'gameplay' else None,
                              repeated_target_exposure=args.select_on == 'gameplay',
                              untouched_test_or_generalization_claim=False)
    if args.select_on == 'gameplay':
        print('Checkpoint retention uses actual sequential gameplay after every epoch (300 actions per level). '
              'Shipped games are reused for development selection, not an untouched generalization test. '
              'All saved artifacts remain unpromoted.', flush=True)
    else:
        print('Explicit last artifact mode: gameplay_not_evaluated; final weights remain unpromoted.', flush=True)
    for epoch in range(1, args.epochs + 1):
        entry = {"epoch": epoch, "lr": scheduler.get_last_lr()[0],
                 "train": run_epoch(model, train_tensors, device, weights, args.batch_size,
                                    optimizer, scheduler, generator, precision=args.precision,
                                    drop_last=args.drop_last, curriculum=curriculum,
                                    steps_per_epoch=steps_per_epoch,
                                    start_step=(epoch - 1) * steps_per_epoch,
                                    total_steps=args.epochs * steps_per_epoch,
                                    on_policy_tensors=on_policy_tensors)}
        if validation_tensors is not None:
            with torch.inference_mode():
                entry["validation"] = run_epoch(model, validation_tensors, device, weights, args.batch_size,
                                                 precision=args.precision)
        if args.select_on == 'gameplay':
            weights_sha256 = initial_state_sha256(model)
            entry['gameplay'] = evaluate_sequential(model, device, per_level_cap=300)
            if initial_state_sha256(model) != weights_sha256:
                raise RuntimeError('policy weights changed during gameplay evaluation')
            entry['gameplay']['policy_weights_sha256'] = weights_sha256
            print(f"Epoch {epoch}: actual sequential gameplay {entry['gameplay']['levels_completed']}/7 | "
                  f"{entry['gameplay']['actions']} actions, {entry['gameplay']['lives_left']} lives left", flush=True)
        else:
            entry['gameplay'] = None
            entry['gameplay_status'] = 'gameplay_not_evaluated'
        selected = candidate_selection(entry, best, args.select_on)
        marker = ""
        if selected is not None:
            best = {**selected,
                    "weights": copy.deepcopy({k: v.detach().to("cpu") for k, v in model.state_dict().items()})}
            marker = " *"
        history.append(entry)
        print(f"Epoch {epoch}/{args.epochs} offline diagnostics | {describe(entry, baseline)}{marker}", flush=True)
        if curriculum is not None:
            print(f"Curriculum difficulty counts {entry['train']['difficulty_counts']} | "
                  f"{args.batch_size} distinct levels per batch", flush=True)
        progress_path = paths['progress']
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        with progress_path.open('w' if progress_created else 'x') as handle:
            handle.write(json.dumps({"history": history, "epochs": args.epochs,
                                     **selection_protocol}, indent=2) + "\n")
        progress_created = True
        # Only this run's transient snapshot is replaceable; the first write is exclusive.
        (save_world_checkpoint if running_created else save_candidate)(paths['running'],
                              model, epoch=epoch, train=str(args.train),
                              **selection_protocol, gameplay=entry.get('gameplay'),
                              batch_size=args.batch_size, data_meta=train_data.get("meta"),
                              successor_labels=successor_labels,
                              on_policy_source=on_policy_source,
                              cell_source=cell_source,
                              training_objective_source=objective_source,
                              execution=execution,
                              initialization=initialization,
                              curriculum=curriculum_metadata(curriculum))
        running_created = True
        if device.type == "cuda" and epoch == 1:
            print(f"Peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB", flush=True)
    # Retain final weights as a separate unpromoted artifact, independently of gameplay ranking.
    last_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best.pop("weights"))
    best.pop("score")
    decision = verdict(history, baseline, best)
    print(f"Keeping epoch {best['epoch']} by {best['criterion']} on {best['selected_on']}", flush=True)

    checkpoint = save_candidate(
        args.checkpoint_out, model.to("cpu"),
        train=str(args.train), validation=str(args.validation) if args.validation else None,
        train_seeds=train_seeds, validation_seeds=validation_seeds, epochs=args.epochs,
        best_epoch=best["epoch"], select_on=args.select_on, selected_on=best["selected_on"],
        promoted=False, candidate_only=True, gameplay=best['gameplay'],
        gameplay_status=best['gameplay_status'], objective_complete=best['objective_complete'],
        shipped_gameplay_used_for_checkpoint_selection=args.select_on == 'gameplay',
        selection_protocol=selection_protocol,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay, loss_weights=weights,
        samples=int(len(train_tensors["frames"])), device=str(device), baseline=baseline, verdict=decision,
        data_meta=train_data.get("meta"), successor_labels=successor_labels,
        training_objective_source=objective_source,
        on_policy_source=on_policy_source,
        checkpoint_loops=args.checkpoint_loops,
        execution=execution,
        initialization=initialization,
        initialize_checkpoint=str(args.initialize_checkpoint) if args.initialize_checkpoint else None,
        initialize_glyph_checkpoint=(str(args.initialize_glyph_checkpoint)
                                     if args.initialize_glyph_checkpoint else None),
        glyph_source=glyph_source,
        cell_source=cell_source,
        precision=args.precision, checkpoint_encoder=args.checkpoint_encoder,
        encoder_chunk_size=args.encoder_chunk_size,
        curriculum=curriculum_metadata(curriculum),
        drop_last=args.drop_last, optimizer_steps=steps_per_epoch * args.epochs,
        trained=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    last_path = paths['last']
    model.load_state_dict(last_weights)
    last_selection = {"epoch": args.epochs, "criterion": "last"}
    last_meta = {key: value for key, value in checkpoint.items()
                 if key not in ("format", "config", "parameters", "weights")}
    last_meta.update(best_epoch=args.epochs, select_on="last", selected_on="last_epoch",
                     verdict=verdict(history, baseline, last_selection),
                     gameplay=history[-1].get('gameplay'),
                     gameplay_status='evaluated' if history[-1].get('gameplay') else 'gameplay_not_evaluated',
                     objective_complete=bool(history[-1].get('gameplay', {}) and
                                             history[-1]['gameplay']['completed']),
                     shipped_gameplay_used_for_checkpoint_selection=False,
                     selection_protocol={**selection_protocol, 'select_on': 'last',
                         'shipped_gameplay_used_for_checkpoint_selection': False})
    save_candidate(last_path, model, **last_meta)

    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else None
    report = {"format": checkpoint["format"], "config": checkpoint["config"], "parameters": checkpoint["parameters"],
              "training_objective_source": objective_source,
              "successor_labels": successor_labels,
              "on_policy_source": on_policy_source,
              "loss_weights": weights, "history": history, "baseline": baseline, "verdict": decision,
              "best": best, "train_seeds": train_seeds, "validation_seeds": validation_seeds,
              **selection_protocol, "gameplay": best['gameplay'],
              "objective_complete": best['objective_complete'],
              "checkpoint": str(args.checkpoint_out), "last_checkpoint": str(last_path), "device": str(device),
              "peak_gpu_memory_gib": peak, "checkpoint_loops": args.checkpoint_loops,
              "execution": execution, "initialization": initialization}
    path = paths['report']
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.checkpoint_out}\nReport {path}", flush=True)
    print(f"Unpromoted candidate epoch {best['epoch']}, selected by {best['criterion']}; "
          + (f"actual sequential gameplay {best['gameplay']['levels_completed']}/7."
             if best['gameplay'] else 'gameplay_not_evaluated.'), flush=True)
    print(f"Offline diagnostics for epoch {decision['epoch']} ({decision['split']}): "
          f"policy CE {decision['policy_cross_entropy']:.4f} vs prior {decision['prior_policy_cross_entropy']:.4f}"
          f" ({'beats' if decision['beats_prior_cross_entropy'] else 'does NOT beat'}); set accuracy "
          f"{decision['set_accuracy']:.3f} vs prior {decision['prior_set_accuracy']:.3f}; prediction MSE "
          f"{decision['prediction_mse']:.4f} vs copy {decision['copy_mse']:.4f} "
          f"({'beats' if decision['beats_copy_baseline'] else 'does NOT beat'} copy); counterfactual top-1 "
          f"{decision['counterfactual_top1']:.3f} vs chance {decision['chance_top1']:.2f}", flush=True)
    print('Offline diagnostics did not select or promote this checkpoint.', flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
