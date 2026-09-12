"""Bounded generated-only H4 training for factored carried-glyph transitions.

This is a separate experiment trainer.  It reuses the verified mixed
exploratory/closing chronological caches, but keeps its model construction,
loss additions, checkpoint format, and reports independent of the existing H4
trainer.  The transition receives only public fields and actions.  Exact
triples are detached targets used by the auxiliary loss and never enter the
autoregressive inputs.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch import nn

from pebby.agent.structured_sequence_objective import sequence_objective
from tools.structured_glyph_diagnostics import balanced_glyph_loss
from tools.train_structured_mixed_sequences import (
    InsufficientMixedLevels, event_weights, exposure, load_exploratory_cache,
    mixed_batch, mixed_rows, source_counts,
)
from tools.train_structured_sequences import load_cache
from tools.train_structured_transition import (
    EVENTS, atomic_json, autocast, digest, event_counts, training_scale,
)


BASE_FORMAT = "pebby.structured-transition.v1"
GLOBAL_FORMAT = "pebby.structured-transition-global-glyph.v1"
LOCAL_FORMAT = "pebby.structured-transition-local-global-glyph.v1"
HORIZONS = 4
GLYPH_CHANNELS = (70, 84)


class ObjectiveView(nn.Module):
    """Hide optional model-only outputs from the legacy sequence objective."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, field, action):
        output = self.model(field, action)
        return {key: output[key] for key in ("field", "readout", "events")}

    def readout(self, field):
        return self.model.readout(field)


def _validate_weight(value, name):
    if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _previous_actual(current_triple, next_triple):
    if (not isinstance(current_triple, torch.Tensor) or not isinstance(next_triple, torch.Tensor)
            or tuple(current_triple.shape) != (len(current_triple), 3)
            or tuple(next_triple.shape) != (len(current_triple), HORIZONS, 3)):
        raise ValueError("triple labels must have shapes [B,3] and [B,4,3]")
    return torch.cat((current_triple.detach()[:, None], next_triple.detach()[:, :-1]), dim=1)


def _direct_readout(predicted_fields, horizon):
    """Interpret global/local broadcast probabilities as categorical logits."""
    probabilities = predicted_fields[:, horizon, 0, GLYPH_CHANNELS[0]:GLYPH_CHANNELS[1]].float()
    if not torch.isfinite(probabilities).all():
        raise ValueError("direct glyph probabilities must be finite")
    if bool((probabilities < -5e-3).any()) or bool((probabilities > 1.005).any()):
        raise ValueError("direct glyph probabilities outside [0,1]")
    groups = probabilities.split((6, 4, 4), dim=-1)
    if any(bool((group.sum(-1) - 1.).abs().max() > 5e-2) for group in groups):
        raise ValueError("direct glyph probabilities must be grouped softmax outputs")
    # Global/local heads emit grouped softmax probabilities.  The clamp only
    # protects log() from float underflow and does not alter nonextreme values.
    log_probabilities = probabilities.clamp_min(1e-8).log()
    shape, color, rotation = log_probabilities.split((6, 4, 4), dim=-1)
    return {"carried_shape_logits": shape, "carried_color_logits": color,
            "carried_rotation_logits": rotation}


def h4_glyph_auxiliary_losses(result, labels, *, direct_weight=0.,
                              predicted_readout_weight=0.):
    """Compute optional direct and predicted-readout balanced glyph losses.

    Each horizon compares against the previous *actual* triple: current at
    horizon one, then the preceding actual successor triple.  This function
    never feeds labels back into ``result['outputs']['predicted']``.
    """
    direct_weight = _validate_weight(direct_weight, "direct_weight")
    predicted_readout_weight = _validate_weight(predicted_readout_weight,
                                                "predicted_readout_weight")
    if not isinstance(labels, dict):
        raise ValueError("labels must be a mapping")
    current = labels.get("triple")
    following = labels.get("next_triple")
    previous = _previous_actual(current, following)
    try:
        predicted_fields = result["outputs"]["predicted"]["fields"]
        predicted_readout = result["outputs"]["predicted"]["readout"]
    except (KeyError, TypeError):
        raise ValueError("result must contain predicted fields and readouts") from None
    if (not isinstance(predicted_fields, torch.Tensor)
            or tuple(predicted_fields.shape[1:]) != (HORIZONS, 148, 96)):
        raise ValueError("predicted fields must be [B,4,148,96]")
    if predicted_fields.shape[0] != len(previous):
        raise ValueError("predicted fields and labels have different batch sizes")
    direct_terms, readout_terms = [], []
    direct_clamp_count = 0
    for horizon in range(HORIZONS):
        if direct_weight:
            # Count values protected against log(0); this is diagnostic only and
            # does not alter the grouped probabilities except at the floor.
            probabilities = predicted_fields[:, horizon, 0, GLYPH_CHANNELS[0]:GLYPH_CHANNELS[1]].float()
            direct_clamp_count += int((probabilities < 1e-8).sum().detach().cpu())
            direct_terms.append(balanced_glyph_loss(
                _direct_readout(predicted_fields, horizon), previous[:, horizon],
                following[:, horizon]))
        if predicted_readout_weight:
            readout = {name: value[:, horizon] for name, value in predicted_readout.items()
                       if name.startswith("carried_") and name.endswith("_logits")}
            readout_terms.append(balanced_glyph_loss(
                readout, previous[:, horizon], following[:, horizon]))
    zero = predicted_fields.sum() * 0.
    direct = torch.stack(direct_terms).mean() if direct_terms else zero
    readout = torch.stack(readout_terms).mean() if readout_terms else zero
    return {
        "direct": direct,
        "predicted_readout": readout,
        "total": direct_weight * direct + predicted_readout_weight * readout,
        "direct_per_horizon": direct_terms,
        "predicted_readout_per_horizon": readout_terms,
        "direct_clamp_count": direct_clamp_count,
        "weights": {"direct": direct_weight, "predicted_readout": predicted_readout_weight},
    }


def _model_for_checkpoint(saved, variant, device):
    """Rebuild one arm and accept only an exact compatible warmstart."""
    fmt = saved.get("format")
    if variant == "base":
        from pebby.agent.structured_transition import StructuredTransition
        if fmt != BASE_FORMAT:
            raise ValueError("base arm requires a base structured-transition checkpoint")
        model = StructuredTransition(saved["config"])
        if model.config() != saved["config"]:
            raise ValueError("base checkpoint config mismatch")
        model.load_state_dict(saved["weights"], strict=True)
    elif variant == "global":
        from pebby.agent.structured_global_glyph import GlobalGlyphTransition
        if fmt == BASE_FORMAT:
            from pebby.agent.structured_transition import StructuredTransition
            base = StructuredTransition(saved["config"])
            model = GlobalGlyphTransition()
            if model.cfg.base_config() != base.config():
                raise ValueError("global warmstart base config mismatch")
            model.warmstart_from_base_state_dict(saved["weights"])
        elif fmt == GLOBAL_FORMAT:
            model = GlobalGlyphTransition(saved["config"])
            model.load_state_dict(saved["weights"], strict=True)
        else:
            raise ValueError("global arm requires base or global glyph checkpoint")
    else:
        from pebby.agent.structured_local_glyph import LocalGlobalGlyphTransition
        if fmt == LOCAL_FORMAT:
            model = LocalGlobalGlyphTransition(saved["config"])
            model.load_state_dict(saved["weights"], strict=True)
        elif fmt == GLOBAL_FORMAT:
            local_config = dict(saved["config"])
            local_config["variant"] = "local_global_glyph"
            model = LocalGlobalGlyphTransition(local_config)
            model.warmstart_from_global_state_dict(saved["weights"])
        else:
            raise ValueError("local arm requires global or local glyph checkpoint")
    return model.to(device).train()


def _load_model(path, variant, device):
    before = digest(path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if digest(path) != before:
        raise ValueError("initial checkpoint changed while loading")
    return _model_for_checkpoint(saved, variant, device), saved, before


def _source_records(live, closing, validation, manifests, args):
    sources = {}
    for directory, manifest in ((args.live_cache, manifests["live_train"]),
                                (args.closing_cache, manifests["closing_train"]),
                                (args.validation_cache, manifests["validation"])):
        directory = Path(directory)
        sources[str(directory / "manifest.json")] = digest(directory / "manifest.json")
        sources.update({str(directory / (name + ".npy")): info["sha256"]
                        for name, info in manifest["arrays"].items()})
    for file in (
            __file__, "tools/train_structured_mixed_sequences.py",
            "tools/train_structured_sequences.py", "tools/train_structured_transition.py",
            "tools/structured_sequence_metrics.py", "tools/structured_glyph_diagnostics.py",
            "pebby/agent/structured_transition.py",
            "pebby/agent/structured_objective.py",
            "pebby/agent/structured_sequence_objective.py",
            "pebby/agent/structured_global_glyph.py",
            "pebby/agent/structured_local_glyph.py"):
        sources[str(file)] = digest(file)
    return sources


def _check_encoder(saved, encoder):
    manifests = saved.get("cache_manifests", {})
    if not manifests or any(manifest.get("field_encoder") != encoder
                            for manifest in manifests.values()):
        raise ValueError("initial checkpoint field encoder does not match mixed caches")


def _check_initial_model_sources(saved, variant):
    """Bind the model implementation that produced an initialization checkpoint."""
    required = ["pebby/agent/structured_transition.py"]
    # Validate the files that produced the consumed checkpoint.  The target
    # arm's new implementation is hashed separately in the experiment source
    # manifest; requiring it here would reject valid base->global and
    # global->local warmstarts.
    checkpoint_format = saved.get("format")
    if checkpoint_format == GLOBAL_FORMAT:
        required.append("pebby/agent/structured_global_glyph.py")
    elif checkpoint_format == LOCAL_FORMAT:
        required.extend(("pebby/agent/structured_global_glyph.py",
                         "pebby/agent/structured_local_glyph.py"))
    elif checkpoint_format != BASE_FORMAT:
        raise ValueError("initial checkpoint format has no known source contract")
    declared = saved.get("sources", {})
    for relative in required:
        path = Path(relative)
        expected = declared.get(relative)
        if expected is None:
            expected = declared.get(str(path.resolve()))
        if expected is None:
            raise ValueError(f"initial checkpoint lacks source binding: {relative}")
        if digest(path) != expected:
            raise ValueError(f"initial checkpoint model source changed: {relative}")


def _preflight(model, objective_model, live, closing, scale, positive, args, saved):
    size = 2 ** (args.max_batch.bit_length() - 1)
    attempts = []
    while size:
        disposable = objective_disposable = optimizer = result = extra = total = batch = None
        started = time.monotonic()
        try:
            disposable = _model_for_checkpoint(saved, args.variant, args.device)
            objective_disposable = ObjectiveView(disposable)
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            optimizer = torch.optim.AdamW(disposable.parameters(), lr=args.lr, weight_decay=.01)
            rows = mixed_rows(live, closing, size, 0., np.random.default_rng(args.seed))
            batch = mixed_batch(live, closing, rows, args.device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(args.device):
                result = sequence_objective(objective_disposable, *batch, scale,
                                            pos_weight=positive,
                                            checkpoint_steps=args.checkpoint_steps)
                extra = h4_glyph_auxiliary_losses(result, batch[3],
                                                  direct_weight=args.direct_glyph_weight,
                                                  predicted_readout_weight=args.predicted_readout_glyph_weight)
                total = result["total"] + extra["total"]
            if not bool(torch.isfinite(total)):
                raise ValueError("nonfinite factored preflight loss")
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(disposable.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            if args.device == "cuda":
                torch.cuda.synchronize()
            attempts.append({"batch_size": size, "status": "fits", "loss": float(total.detach()),
                             "auxiliary": float(extra["total"].detach()), "gradient_norm": float(norm),
                             "seconds": time.monotonic() - started,
                             "peak_allocated_bytes": (torch.cuda.max_memory_allocated()
                                                       if args.device == "cuda" else None),
                             "direct_clamp_count": extra["direct_clamp_count"]})
            return size, attempts
        except (torch.cuda.OutOfMemoryError, InsufficientMixedLevels):
            attempts.append({"batch_size": size, "status": "out_of_memory_or_insufficient",
                             "seconds": time.monotonic() - started})
            size //= 2
        finally:
            del objective_disposable, disposable, optimizer, result, extra, total, batch
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
    raise RuntimeError("no mixed factored batch fits")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("live-cache", "closing-cache", "validation-cache", "initialize",
                 "checkpoint", "report"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--variant", choices=("base", "global", "local"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--max-batch", type=int, default=1024)
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=.0003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seconds", type=int, default=1200)
    parser.add_argument("--checkpoint-steps", action="store_true")
    parser.add_argument("--direct-glyph-weight", type=float, default=0.)
    parser.add_argument("--predicted-readout-glyph-weight", type=float, default=0.)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if (not 1 <= args.max_batch <= 1024 or args.max_batch & (args.max_batch - 1)
            or not 1 <= args.updates <= 2000 or not 1 <= args.seconds <= 1800
            or args.eval_batch < 1 or not np.isfinite(args.lr) or args.lr <= 0):
        raise SystemExit("invalid bounded update, batch, timeout, or learning-rate argument")
    _validate_weight(args.direct_glyph_weight, "direct_glyph_weight")
    _validate_weight(args.predicted_readout_glyph_weight, "predicted_readout_glyph_weight")
    if args.variant == "base" and args.direct_glyph_weight:
        raise SystemExit("direct-glyph-weight requires global or local variant")
    for path in (args.checkpoint, args.report):
        if Path(path).exists():
            raise SystemExit(f"refusing existing output: {path}")
    torch.set_num_threads(1)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    report = {
        "status": "running", "pid": os.getpid(), "args": vars(args).copy(),
        "scope": "generated mixed chronological H4 factored glyph experiment",
        "official_inputs_used": False, "policy_integrated": False, "training": [],
        "limitations": [
            "The exploratory and closing caches contain no resets or terminal failures.",
            "Direct glyph loss is derived from predicted global/local broadcast probabilities.",
            "Auxiliary labels are detached loss targets and never autonomous inputs.",
            "This experiment does not establish gameplay or H4 control success.",
        ],
    }
    atomic_json(args.report, report)

    def expired(*_):
        raise TimeoutError("bounded factored H4 deadline")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.seconds)
    try:
        live, live_manifest = load_exploratory_cache(args.live_cache)
        closing, closing_manifest = load_cache(args.closing_cache, "train")
        validation, validation_manifest = load_cache(args.validation_cache, "validation")
        if (live_manifest["field_encoder"] != closing_manifest["field_encoder"]
                or live_manifest["field_encoder"] != validation_manifest["field_encoder"]):
            raise ValueError("all mixed caches must use the same field encoder")
        if np.intersect1d(np.union1d(live["seeds"], closing["seeds"]), validation["seeds"]).size:
            raise ValueError("validation level leakage")
        initial_sha = digest(args.initialize)
        saved = torch.load(args.initialize, map_location="cpu", weights_only=True)
        if digest(args.initialize) != initial_sha:
            raise ValueError("initial checkpoint changed while loading")
        _check_encoder(saved, live_manifest["field_encoder"])
        _check_initial_model_sources(saved, args.variant)
        model = _model_for_checkpoint(saved, args.variant, args.device)
        objective_model = ObjectiveView(model)
        scale = torch.as_tensor(training_scale(live), device=args.device)
        rates, positive_values = event_weights(live, closing)
        positive = torch.as_tensor(positive_values, device=args.device)
        manifests = {"live_train": live_manifest, "closing_train": closing_manifest,
                     "validation": validation_manifest}
        sources = _source_records(live, closing, validation, manifests, args)
        sources[args.initialize] = initial_sha
        report.update(
            sources=sources,
            initial_checkpoint_format=saved.get("format"),
            initial_model_sources_verified=True,
            parameters=model.parameter_count(),
            available_levels={"live": len(live["seeds"]), "closing": len(closing["seeds"]),
                              "validation": len(validation["seeds"])},
            available_train_union=int(len(np.union1d(live["seeds"], closing["seeds"]))),
            event_coverage={name: event_counts(data) for name, data in
                            (("live", live), ("closing", closing), ("validation", validation))},
            feature_scale=scale.tolist(), event_mixture_rates=rates.tolist(),
            event_positive_weights=positive_values.tolist(),
            glyph_auxiliary_weights={"direct": args.direct_glyph_weight,
                                     "predicted_readout": args.predicted_readout_glyph_weight},
        )
        size, attempts = _preflight(model, objective_model, live, closing, scale, positive, args, saved)
        report.update(batch_size=size, batch_source_counts=dict(zip(("live", "closing"), source_counts(size))),
                      batch_probe=attempts)
        atomic_json(args.report, report)
        if args.preflight_only:
            report["status"] = "preflight_complete"
        else:
            # Fresh exact warmstart after the disposable preflight.
            model = _model_for_checkpoint(saved, args.variant, args.device)
            objective_model = ObjectiveView(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
            rng = np.random.default_rng(args.seed)
            seen = {"live": set(), "closing": set()}
            draws = {"live": np.zeros(5, np.int64), "closing": np.zeros(5, np.int64)}
            direct_clamp_total = 0
            for step in range(args.updates):
                rows = mixed_rows(live, closing, size, step / max(args.updates - 1, 1), rng)
                batch = mixed_batch(live, closing, rows, args.device)
                optimizer.zero_grad(set_to_none=True)
                with autocast(args.device):
                    result = sequence_objective(objective_model, *batch, scale,
                                                pos_weight=positive,
                                                checkpoint_steps=args.checkpoint_steps)
                    extra = h4_glyph_auxiliary_losses(
                        result, batch[3], direct_weight=args.direct_glyph_weight,
                        predicted_readout_weight=args.predicted_readout_glyph_weight)
                    total = result["total"] + extra["total"]
                if not bool(torch.isfinite(total)):
                    raise ValueError("nonfinite factored H4 loss")
                total.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                optimizer.step()
                direct_clamp_total += extra["direct_clamp_count"]
                for name, data, indices in (("live", live, rows[0]), ("closing", closing, rows[1])):
                    seen[name].update(map(int, data["seeds"][indices]))
                    draws[name] += np.bincount(data["difficulties"][indices], minlength=6)[1:6]
                if step == 0 or (step + 1) % 20 == 0 or step + 1 == args.updates:
                    entry = {
                        "step": step + 1, "loss": float(total.detach()),
                        "base_loss": float(result["total"].detach()),
                        "direct_glyph_loss": float(extra["direct"].detach()),
                        "predicted_readout_glyph_loss": float(extra["predicted_readout"].detach()),
                        "direct_clamp_count": extra["direct_clamp_count"],
                        "direct_clamp_count_total": direct_clamp_total,
                        "gradient_norm": float(norm), "elapsed_seconds": time.monotonic() - started,
                        "source_counts": {"live": len(rows[0]), "closing": len(rows[1])},
                        "losses": {key: float(value.detach()) for key, value in result["losses"].items()},
                        **exposure(seen),
                    }
                    report["training"].append(entry)
                    report.update(completed_updates=step + 1,
                                  difficulty_draws={name: values.tolist() for name, values in draws.items()},
                                  **exposure(seen))
                    atomic_json(args.report, report)
                    print(json.dumps(entry), flush=True)
                del result, extra, total, batch
            if any(digest(path) != checksum for path, checksum in sources.items()):
                raise ValueError("factored H4 source changed during training")
            seen_union = seen["live"] | seen["closing"]
            checkpoint = {
                "format": model.checkpoint_format if hasattr(model, "checkpoint_format") else BASE_FORMAT,
                "config": model.config(),
                "weights": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "sources": sources, "cache_manifests": manifests,
                "feature_scale": scale.cpu(), "event_positive_weights": positive.cpu(),
                "parameters": model.parameter_count(), "updates": args.updates,
                "batch_size": size, "seed": args.seed,
                "variant": args.variant,
                "objective": "mixed_autoregressive_H4_factored_glyph",
                "direct_glyph_weight": args.direct_glyph_weight,
                "predicted_readout_glyph_weight": args.predicted_readout_glyph_weight,
                "direct_clamp_count": direct_clamp_total,
                "initialize": args.initialize, "seen_train_seeds": sorted(seen_union),
                "exposure": exposure(seen), "policy_integrated": False,
                "official_inputs_used": False,
            }
            temporary = Path(args.checkpoint).with_name(Path(args.checkpoint).name + f".{os.getpid()}.tmp")
            torch.save(checkpoint, temporary)
            os.replace(temporary, args.checkpoint)
            report.update(status="complete", checkpoint=args.checkpoint,
                          checkpoint_sha256=digest(args.checkpoint),
                          seen_train_seeds=sorted(seen_union), **exposure(seen))
        if any(digest(path) != checksum for path, checksum in sources.items()):
            raise ValueError("factored H4 sources changed")
        report["source_unchanged"] = True
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.report, report)


if __name__ == "__main__":
    main()
