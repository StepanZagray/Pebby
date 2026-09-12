"""Paired generated-only H1 dynamics repair from visited public states.

The control and treatment arms start from the same local/global-glyph dynamics
checkpoint and receive the same level, action, and legacy-view schedule.  The
treatment replaces a fixed subset of legacy rows with one uniformly sampled
on-policy row from the same level.  Only the dynamics model is optimized; the
public field encoder and the warm workspace action readout remain frozen.

This is deliberately one-step.  The on-policy cache contains four actual
counterfactual successors per state, but it does not contain chronological H4
action sequences.  Its ``imagined_fields`` array is diagnostic only and is
never used as a training target.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from pebby.agent.structured_factored_policy import state_digest
from pebby.agent.structured_objective import objective
from tools.preflight_structured_workspace import load_inputs
from tools.structured_onpolicy_sampling import PairedRows, PairedStateSampler
from tools.train_structured_onpolicy_comparison import (
    load_trajectory,
    load_validation,
    load_head,
    merge_sources,
    prepare_batch,
    verify_sources,
)
from tools.train_structured_glyph_ablation import evaluate as evaluate_h1
from tools.train_structured_glyph_ablation import losses as glyph_losses
from tools.train_structured_transition import atomic_json, digest


ONPOLICY_CACHE = Path("data/structured-workspace-onpolicy1024-fp32-fields")
WARMSTART = Path("checkpoints/ls20-structured-workspace-comparison-600-evolving.pt")
INITIAL = Path("checkpoints/ls20-factored-local-h4-400.pt")
INITIAL_SHA256 = "dc676f5ac67cc7b5b3cfaf4b133cc6a9c9307d3ca09597f361f1447c9f72385c"
WARMSTART_SHA256 = "6a5d7716fca4403048434a8100e26770dc26fb9d19ce66600fb3eccca7d22e0c"
LOCAL_FORMAT = "pebby.structured-transition-local-global-glyph.v1"
LABELS = (
    "player_cell", "next_player_cell", "triple", "next_triple", "steps",
    "next_steps", "lives", "next_lives", "lost_life", "terminal", "won",
)
EVENTS = ("lost_life", "terminal", "won")


def _digest(path: str | Path) -> str:
    return digest(path)


def _canonical_sources(sources: dict[str, str]) -> dict[str, str]:
    return {str(Path(path).resolve()): value for path, value in sources.items()}


def _verify_sources(sources: dict[str, str]) -> None:
    for path, expected in sources.items():
        if _digest(path) != expected:
            raise ValueError(f"source changed: {path}")


def _source_schedule_digest(schedule: list[tuple[PairedRows, np.ndarray]]) -> str:
    hasher = hashlib.sha256()
    for selection, actions in schedule:
        for value, dtype in (
            (selection.base_rows, "<i8"),
            (selection.base_views, "i1"),
            (selection.trajectory_rows, "<i8"),
            (actions, "<i8"),
        ):
            hasher.update(np.asarray(value, dtype=dtype).tobytes())
    return hasher.hexdigest()


def _used_rows_digest(schedule: list[tuple[PairedRows, np.ndarray]], treatment: bool) -> str:
    """Digest the actual source row/view used by one arm, including actions."""
    hasher = hashlib.sha256()
    for selection, actions in schedule:
        replaced = treatment & (selection.trajectory_rows >= 0)
        rows = np.where(replaced, selection.trajectory_rows, selection.base_rows)
        views = np.where(replaced, 2, selection.base_views)
        for value, dtype in ((rows, "<i8"), (views, "i1"), (actions, "<i8")):
            hasher.update(np.asarray(value, dtype=dtype).tobytes())
    return hasher.hexdigest()


def schedule_event_counts(views, trajectory, schedule, treatment: bool) -> dict[str, int]:
    """Count events in the exact action/row stream consumed by one arm."""
    counts = {name: 0 for name in EVENTS}
    counts.update(on_policy_branches=0, auxiliary_branches=0)
    branches = 0
    for selection, actions in schedule:
        replacement = treatment & (selection.trajectory_rows >= 0)
        rows = np.where(replacement, selection.trajectory_rows, selection.base_rows)
        selected_views = np.where(replacement, 2, selection.base_views)
        for index, (row, view, action) in enumerate(zip(rows, selected_views, actions)):
            source = trajectory if replacement[index] else views[int(view)]
            source_row = int(row)
            action = int(action)
            if replacement[index]:
                counts['on_policy_branches' if source['on_policy'][source_row] else 'auxiliary_branches'] += 1
            for name in EVENTS:
                counts[name] += int(bool(source[name][source_row, action]))
            branches += 1
    counts["branches"] = branches
    counts["terminal_failure"] = counts["terminal"] - counts["won"]
    return counts


def build_schedule(sampler: PairedStateSampler, batch_size: int, replacements: int,
                   updates: int, seed: int = 42) -> list[tuple[PairedRows, np.ndarray]]:
    """Precompute one identical level/action stream for both arms."""
    if type(updates) is not int or updates < 1:
        raise ValueError("updates must be positive")
    rng = np.random.default_rng(seed)
    schedule = []
    for step in range(updates):
        progress = step / max(updates - 1, 1)
        selection = sampler.draw(batch_size, replacements, progress, rng)
        actions = rng.integers(0, 4, size=batch_size, dtype=np.int64)
        if len(np.unique(selection.base_rows)) != batch_size:
            raise ValueError("schedule repeats a level")
        schedule.append((selection, actions))
    return schedule


def prepare_dynamics_batch(views, trajectory, selection: PairedRows,
                           actions: np.ndarray, treatment: bool) -> dict[str, np.ndarray]:
    """Materialize one H1 batch while preserving the paired row contract."""
    actions = np.asarray(actions)
    rows = np.asarray(selection.base_rows)
    which = np.asarray(selection.base_views)
    replacement = np.asarray(selection.trajectory_rows)
    if (rows.ndim != 1 or actions.shape != rows.shape or replacement.shape != rows.shape
            or which.shape != rows.shape or not np.issubdtype(actions.dtype, np.integer)):
        raise ValueError("selection and actions have incompatible shapes")
    if np.any((actions < 0) | (actions > 3)):
        raise ValueError("action outside0..3")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("batch must contain distinct base levels")
    if not len(views) or np.any((which < 0) | (which >= len(views))):
        raise ValueError("invalid legacy view selection")

    # Every field/label is first copied from the selected legacy view.  The
    # treatment then replaces only on-policy rows, never expert anchors.
    current_keys = ("fields", "player_cell", "triple", "steps", "lives",
                    "optimal", "seeds", "difficulties")
    next_keys = ("next_fields", "next_player_cell", "next_triple", "next_steps",
                 "next_lives", "next_optimal", *EVENTS)
    result = {}
    for key in current_keys:
        source = views[0][key]
        result[key] = np.empty((len(rows), *source.shape[1:]), dtype=source.dtype)
        for view in range(len(views)):
            positions = np.flatnonzero(which == view)
            if len(positions):
                result[key][positions] = views[view][key][rows[positions]]
    for key in next_keys:
        source = views[0][key]
        result[key] = np.empty((len(rows), *source.shape[2:]), dtype=source.dtype)
        for view in range(len(views)):
            positions = np.flatnonzero(which == view)
            if len(positions):
                result[key][positions] = views[view][key][rows[positions], actions[positions]]

    positions = np.flatnonzero(replacement >= 0)
    if treatment and len(positions):
        selected = replacement[positions]
        permitted = trajectory['on_policy'] | trajectory.get('auxiliary', np.zeros_like(trajectory['on_policy']))
        if np.any(~permitted[selected]):
            raise ValueError("unmarked expert anchor selected for replacement")
        if not np.array_equal(trajectory["seeds"][selected], result["seeds"][positions]):
            raise ValueError("on-policy replacement crossed level identity")
        for key in current_keys:
            result[key][positions] = trajectory[key][selected]
        for key in next_keys:
            result[key][positions] = trajectory[key][selected, actions[positions]]
    elif treatment and len(positions) == 0:
        raise ValueError("treatment batch has no on-policy replacements")

    # The branch tensors are selected after replacement so C/T share actions
    # and unselected rows exactly while differing only at marked states.
    expected_seeds = np.asarray([
        views[int(view)]["seeds"][int(row)] for row, view in zip(rows, which)
    ])
    if not treatment and not np.array_equal(result["seeds"], expected_seeds):
        raise ValueError("control seed alignment changed")
    result["actions"] = np.asarray(actions, dtype=np.int64).copy()
    return result


def tensors_for_batch(data: dict[str, np.ndarray], device: str):
    fields = torch.as_tensor(np.array(data["fields"], copy=True), device=device).float()
    following = torch.as_tensor(np.array(data["next_fields"], copy=True), device=device).float()
    actions = torch.as_tensor(np.array(data.get("actions", np.zeros(len(fields), dtype=np.int64)),
                                      copy=True), device=device).long()
    labels = {}
    for name in LABELS:
        source = data[name]
        labels[name] = torch.as_tensor(np.array(source, copy=True), device=device)
    return fields, following, actions, labels


def make_local_model(saved: dict, device: str):
    from pebby.agent.structured_local_glyph import LocalGlobalGlyphTransition

    if saved.get("format") != LOCAL_FORMAT:
        raise ValueError("local-global-glyph initialization required")
    model = LocalGlobalGlyphTransition(saved["config"])
    if model.parameter_count() != saved.get("parameters"):
        raise ValueError("initial dynamics parameter count mismatch")
    model.load_state_dict(saved["weights"], strict=True)
    return model.to(device).train()


def _load_initial(path: Path):
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != INITIAL_SHA256:
        raise ValueError(f"unexpected dynamics initializer SHA: {actual}")
    saved = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != actual:
        raise ValueError("dynamics initializer changed while loading")
    if saved.get("format") != LOCAL_FORMAT or saved.get("official_inputs_used") is not False:
        raise ValueError("generated-only local dynamics initializer required")
    if saved.get("parameters") != 294664:
        raise ValueError("unexpected local dynamics capacity")
    for key in ("feature_scale", "event_positive_weights", "cache_manifests", "sources"):
        if key not in saved:
            raise ValueError(f"initializer lacks {key}")
    return saved, actual


def _autocast(device: str):
    return torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda")


def _loss_step(model, data, scale, positive, device: str):
    batch = tensors_for_batch(data, device)
    from tools.structured_glyph_diagnostics import balanced_glyph_loss
    result = objective(model, *batch, scale, pos_weight=positive)
    extra = balanced_glyph_loss(result["readouts"]["predicted"], batch[3]["triple"],
                                batch[3]["next_triple"])
    direct_readout = {f"carried_{name}_logits": result["output"]["glyph_logits"][name]
                      for name in ("shape", "color", "rotation")}
    direct = balanced_glyph_loss(direct_readout, batch[3]["triple"], batch[3]["next_triple"])
    total = result["total"] + extra + direct
    return total, result, {"predicted_readout": extra, "direct": direct}, batch


def _policy_rank(logits: torch.Tensor, masks: np.ndarray) -> dict[str, int | float]:
    masks_t = torch.as_tensor(np.asarray(masks), device=logits.device).long()
    chosen = logits.argmax(-1)
    good = ((masks_t >> chosen) & 1).bool()
    return {"count": int(len(chosen)), "optimal_set_hits": int(good.sum()),
            "optimal_set_rate": float(good.float().mean()) if len(good) else None}


@torch.no_grad()
def evaluate_validation(model, head, data, rows, scale, positive, *, batch_size=32):
    """Fixed validation diagnostics, including the frozen workspace ranking."""
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) < 1:
        raise ValueError("validation rows must be nonempty")
    model.eval(); head.eval()
    # Reuse the established additive H1 metrics for fields, readouts, events,
    # and changed/unchanged glyph groups.
    from tools.train_structured_transition import evaluate as evaluate_h1_transition
    eval_data = {
        key: np.asarray(value)[rows]
        for key, value in data.items()
        if isinstance(value, np.ndarray) and len(value) == len(data["seeds"])
    }
    eval_rows = np.arange(len(rows), dtype=np.int64)
    transition = evaluate_h1_transition(model, eval_data, scale, positive, "cpu", batch_size)
    glyph = evaluate_h1(model, eval_data, eval_rows, "cpu", batch_size)
    rank = {"predicted": {"count": 0, "optimal_set_hits": 0},
            "actual": {"count": 0, "optimal_set_hits": 0}}
    copy_rank = {"count": 0, "optimal_set_hits": 0}
    raw_player = {name: {"count": 0, "correct": 0}
                  for name in ("predicted", "actual", "copy")}
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start + batch_size]
        fields = torch.as_tensor(np.array(data["fields"][selected], copy=True), dtype=torch.float32)
        actual = torch.as_tensor(np.array(data["next_fields"][selected], copy=True), dtype=torch.float32)
        predicted = torch.stack([model.predict(fields, torch.full((len(selected),), action, dtype=torch.long))
                                 for action in range(4)], dim=1)
        current = fields[:, None].expand(-1, 4, -1, -1)
        for target, output in (("predicted", predicted), ("actual", actual)):
            values = _policy_rank(head(output), data["optimal"][selected])
            rank[target]["count"] += values["count"]
            rank[target]["optimal_set_hits"] += values["optimal_set_hits"]
        values = _policy_rank(head(current), data["optimal"][selected])
        copy_rank["count"] += values["count"]
        copy_rank["optimal_set_hits"] += values["optimal_set_hits"]
        target_player = torch.as_tensor(
            np.array(data["next_player_cell"][selected, :, 1] * 12
                     + data["next_player_cell"][selected, :, 0], copy=True),
            dtype=torch.long,
        )
        for name, output in (("predicted", predicted), ("actual", actual), ("copy", current)):
            guessed = output[:, :, :144, 55].argmax(-1)
            raw_player[name]["count"] += int(guessed.numel())
            raw_player[name]["correct"] += int((guessed == target_player).sum())
    for values in (rank["predicted"], rank["actual"], copy_rank):
        values["optimal_set_rate"] = values["optimal_set_hits"] / values["count"]
    for values in raw_player.values():
        values["accuracy"] = values["correct"] / values["count"]
    return {"levels": int(len(rows)), "branches": int(len(rows) * 4),
            "transition": transition, "glyph": glyph,
            "workspace_action_ranking": {**rank, "copy": copy_rank},
            "raw_player_channel": raw_player,
            "scope": "fixed generated validation H1 fidelity; no gameplay claim"}


def _cache_manifests(trajectory_manifest, validation_views):
    result = {"onpolicy_train": trajectory_manifest}
    for name, path in (
        ("legacy_train", Path("data/structured-field-16384/train/manifest.json")),
        ("legacy_additional_train", Path("data/structured-field-additional-state-16384/train/manifest.json")),
    ):
        result[name] = json.loads(path.read_text())
    # load_validation may return one identical view or two independent views.
    result["validation"] = json.loads(Path("data/structured-field-16384/validation/manifest.json").read_text())
    return result


def save_dynamics(path: Path, model, initial_saved, sources, manifests, scale, positive,
                  arm, schedule_sha, args, source_rows_sha,
                  encoder_state_sha256, workspace_state_sha256):
    if path.exists():
        raise ValueError(f"refusing existing checkpoint: {path}")
    _verify_sources(sources)
    payload = {
        "format": LOCAL_FORMAT,
        "config": model.config(),
        "weights": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "parameters": model.parameter_count(),
        "sources": dict(sources),
        "cache_manifests": manifests,
        "feature_scale": scale.detach().cpu(),
        "event_positive_weights": positive.detach().cpu(),
        "updates": args.updates,
        "batch_size": args.batch_size,
        "smoke": bool(args.smoke),
        "seed": args.seed,
        "arm": arm,
        "variant": "local",
        "objective": "paired_onpolicy_h1_dynamics_repair",
        "direct_glyph_weight": 1.0,
        "predicted_readout_glyph_weight": 1.0,
        "initialize": str(args.initialize),
        "initialize_sha256": INITIAL_SHA256,
        "selection_sha256": schedule_sha,
        "used_state_rows_sha256": source_rows_sha,
        "frozen_encoder_state_sha256": encoder_state_sha256,
        "frozen_workspace_state_sha256": workspace_state_sha256,
        "official_inputs_used": False,
        "policy_integrated": False,
        "frozen_encoder": True,
        "frozen_workspace_readout": str(args.warmstart),
        "warmstart_workspace_sha256": WARMSTART_SHA256,
        "initial_metadata": {key: initial_saved.get(key) for key in
                             ("format", "parameters", "updates", "batch_size")},
    }
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    raw = path.read_bytes()
    loaded = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    check = make_local_model(loaded, "cpu")
    model_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if state_digest(check.state_dict()) != state_digest(model_state):
        raise ValueError("saved dynamics reload changed weights")
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "parameters": model.parameter_count(), "strict_reload_exact": True}


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onpolicy-cache", type=Path, default=ONPOLICY_CACHE)
    parser.add_argument("--warmstart", type=Path, default=WARMSTART)
    parser.add_argument("--initialize", type=Path, default=INITIAL)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument('--auxiliary-fraction',type=float,default=.25,
                        help='Probability of a bound expert/failure row in a same-level treatment replacement')
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-prefix", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.report.exists():
        raise SystemExit("refusing existing report")
    if args.batch_size < 2 or args.batch_size > 1024 or args.batch_size & (args.batch_size - 1):
        raise SystemExit("batch size must be a power of two in2..1024")
    if args.smoke:
        if args.device != "cpu" or args.batch_size > 8 or args.updates != 2:
            raise SystemExit("smoke requires CPU, B<=8, and exactly2 updates")
    elif args.preflight_only:
        if args.device != "cuda" or args.batch_size != 1024 or args.updates != 2:
            raise SystemExit("preflight requires CUDA B1024 and exactly2 updates")
    elif args.device != "cuda" or args.batch_size != 1024 or args.updates != 200:
        raise SystemExit("production requires CUDA B1024 and 200 updates")
    if not 1 <= args.seconds <= 1800:
        raise SystemExit("seconds must be1..1800")
    for arm in ("control", "onpolicy"):
        if (args.checkpoint_prefix.with_name(args.checkpoint_prefix.name + f"-{arm}.pt")).exists():
            raise SystemExit("refusing existing checkpoint")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    print("PID", os.getpid(), flush=True)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("paired dynamics deadline")))
    signal.alarm(args.seconds)
    report = {
        "status": "loading", "pid": os.getpid(), "args": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        } | {
            "onpolicy_cache": str(args.onpolicy_cache), "warmstart": str(args.warmstart),
            "initialize": str(args.initialize), "report": str(args.report),
            "checkpoint_prefix": str(args.checkpoint_prefix)},
        "source": "generated_only", "official_inputs_used": False,
        "policy_integrated": False, "arms": {},
        "precision": "CUDA BF16 or CPU FP32; TF32 disabled",
        "limits": [
            "One-step actual successor training; on-policy cache has no chronological H4 actions.",
            "Workspace action readout and public encoder remain frozen.",
            "No terminal-failure positives are added by this cache pair.",
            "Validation and gameplay are diagnostics, never checkpoint selection.",
        ],
    }

    def persist():
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.report, report)

    persist()
    try:
        policy, views, sources = load_inputs()
        warm_sha = _digest(args.warmstart)
        if warm_sha != WARMSTART_SHA256:
            raise ValueError(f"unexpected workspace warmstart SHA: {warm_sha}")
        workspace_head, workspace_saved = load_head(args.warmstart)
        workspace_head.eval()
        for parameter in workspace_head.parameters():
            parameter.requires_grad_(False)
        policy.encoder.eval()
        for parameter in policy.encoder.parameters():
            parameter.requires_grad_(False)
        workspace_state_sha256 = state_digest(workspace_head.state_dict())
        encoder_state_sha256 = state_digest(policy.encoder.state_dict())
        initial_saved, initial_sha = _load_initial(args.initialize)
        expected_dynamics_sha = policy.sources.get("dynamics_state_sha256")
        if expected_dynamics_sha != state_digest(initial_saved["weights"]):
            raise ValueError("initializer weights do not match paired policy dynamics source")
        if initial_saved.get("config") != policy.dynamics.config():
            raise ValueError("initializer config does not match paired policy dynamics source")
        merge_sources(sources, initial_saved["sources"])
        trajectory, trajectory_manifest = load_trajectory(
            args.onpolicy_cache, policy, views[0], args.warmstart, sources)
        sampler = PairedStateSampler(views[0]["seeds"], views[0]["difficulties"],
                                     trajectory["seeds"], trajectory["on_policy"],
                                     auxiliary=trajectory.get('auxiliary'), auxiliary_fraction=args.auxiliary_fraction)
        replacements = min(args.batch_size // 2, len(sampler.eligible))
        if replacements < 1:
            raise ValueError("no verified on-policy replacement levels")
        if not args.smoke and (len(sampler.eligible) < 1024 or replacements != 512):
            raise ValueError("production requires at least1024 eligible levels and512 replacements")
        validations = load_validation(policy, sources)
        val_rows = np.arange(min(8, len(validations[0]["seeds"])) if args.smoke
                             else len(validations[0]["seeds"]), dtype=np.int64)
        if not args.smoke and len(val_rows) != 512:
            raise ValueError("fixed512 validation levels required")
        for value in validations:
            if np.intersect1d(value["seeds"], trajectory["seeds"]).size:
                raise ValueError("on-policy and validation levels overlap")
        merge_sources(sources, {str(args.warmstart): warm_sha, str(args.initialize): initial_sha,
                                str(Path(__file__).resolve()): _digest(__file__)})
        for path in (
            "tools/structured_onpolicy_sampling.py",
            "tools/train_structured_onpolicy_comparison.py",
            "tools/preflight_structured_workspace.py",
            "tools/train_structured_policy.py",
            "tools/train_structured_transition.py",
            "tools/train_structured_glyph_ablation.py",
            "tools/structured_glyph_diagnostics.py",
            "pebby/agent/structured_objective.py",
            "pebby/agent/structured_transition.py",
            "pebby/agent/structured_global_glyph.py",
            "pebby/agent/structured_local_glyph.py",
            "pebby/agent/structured_factored_policy.py",
            "pebby/agent/structured_workspace_policy.py",
            "pebby/agent/structured_workspace_controller.py",
        ):
            merge_sources(sources, {str(Path(path).resolve()): _digest(path)})
        verify_sources(sources)
        schedule = build_schedule(sampler, args.batch_size, replacements, args.updates, args.seed)
        schedule_sha = _source_schedule_digest(schedule)
        manifests = _cache_manifests(trajectory_manifest, validations)
        scale = torch.as_tensor(initial_saved["feature_scale"], dtype=torch.float32)
        positive = torch.as_tensor(initial_saved["event_positive_weights"], dtype=torch.float32)
        report.update(
            status="preflight" if args.preflight_only else "training",
            sources=sources,
            source_schedule_sha256=schedule_sha,
            replacements=replacements,
            trajectory_levels=len(sampler.eligible),
            trajectory_rows=int(trajectory["on_policy"].sum()),
            auxiliary_rows=int(trajectory.get('auxiliary', np.zeros_like(trajectory['on_policy'])).sum()),
            auxiliary_replacement_probability=sampler.auxiliary_fraction,
            validation_levels=len(val_rows),
            initialization={"path": str(args.initialize), "sha256": initial_sha,
                           "parameters": initial_saved["parameters"],
                           "state_sha256": state_digest(initial_saved["weights"])},
            workspace_readout={"path": str(args.warmstart), "sha256": warm_sha,
                              "parameters": workspace_saved["parameters"],
                              "state_sha256": workspace_saved["state_sha256"],
                              "loaded_state_sha256": workspace_state_sha256},
            frozen_encoder_state_sha256=encoder_state_sha256,
            event_coverage={
                "onpolicy_cache": trajectory_manifest.get("event_coverage"),
                "legacy_view0": json.loads(
                    Path("data/structured-field-16384/train/manifest.json").read_text()
                ).get("event_coverage"),
                "legacy_view1": json.loads(
                    Path("data/structured-field-additional-state-16384/train/manifest.json").read_text()
                ).get("event_coverage"),
                "validation": json.loads(
                    Path("data/structured-field-16384/validation/manifest.json").read_text()
                ).get("event_coverage"),
            },
        )
        persist()

        if not args.preflight_only:
            baseline_model = make_local_model(initial_saved, "cpu")
            report["initial_validation"] = evaluate_validation(
                baseline_model, workspace_head, validations[0], val_rows, scale, positive)
            del baseline_model
            gc.collect()
            persist()

        for arm, treatment in (("control", False), ("onpolicy", True)):
            model = make_local_model(initial_saved, args.device)
            initial_state = state_digest(model.state_dict())
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            arm_report = {"status": "training", "parameters": model.parameter_count(),
                          "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                          "initial_state_sha256": initial_state, "training": [],
                          "schedule_sha256": schedule_sha,
                          "clipped_steps": 0, "replacement_rows": 0,
                          "sampled_event_counts": schedule_event_counts(
                              views, trajectory, schedule, treatment),
                          "used_state_rows_sha256": _used_rows_digest(schedule, treatment)}
            report["arms"][arm] = arm_report
            for step, (selection, actions) in enumerate(schedule):
                tick = time.monotonic()
                data = prepare_dynamics_batch(views, trajectory, selection, actions, treatment)
                if treatment:
                    arm_report["replacement_rows"] += int((selection.trajectory_rows >= 0).sum())
                optimizer.zero_grad(set_to_none=True)
                with _autocast(args.device):
                    total, result, extra, batch = _loss_step(model, data, scale.to(args.device),
                                                               positive.to(args.device), args.device)
                if not bool(torch.isfinite(total)):
                    raise ValueError("nonfinite paired dynamics loss")
                total.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                optimizer.step()
                clipped = float(norm) > 10.
                arm_report["clipped_steps"] += int(clipped)
                event = {"step": step + 1, "loss": float(total.detach()),
                         "base_loss": float(result["total"].detach()),
                         "predicted_readout_glyph_loss": float(extra["predicted_readout"].detach()),
                         "direct_glyph_loss": float(extra["direct"].detach()),
                         "gradient_norm": float(norm), "seconds": time.monotonic() - tick,
                         "replaced": int((selection.trajectory_rows >= 0).sum()) if treatment else 0}
                arm_report["training"].append(event)
                if step == 0 or step + 1 == args.updates or (step + 1) % 20 == 0:
                    report["status"] = f"{arm}_training"; persist(); print(json.dumps({"arm": arm, **event}), flush=True)
                del data, result, extra, batch, total
            verify_sources(sources)
            if state_digest(workspace_head.state_dict()) != workspace_state_sha256:
                raise ValueError("workspace readout changed during dynamics fit")
            if state_digest(policy.encoder.state_dict()) != encoder_state_sha256:
                raise ValueError("public field encoder changed during dynamics fit")
            model.eval().to("cpu")
            if not args.preflight_only:
                source_rows_sha = _used_rows_digest(schedule, treatment)
                checkpoint_path = args.checkpoint_prefix.with_name(args.checkpoint_prefix.name + f"-{arm}.pt")
                arm_report["checkpoint"] = save_dynamics(
                    checkpoint_path, model, initial_saved, sources, manifests, scale, positive,
                    arm, schedule_sha, args, source_rows_sha,
                    encoder_state_sha256, workspace_state_sha256)
                arm_report["validation"] = evaluate_validation(
                    model, workspace_head, validations[0], val_rows, scale, positive)
            arm_report["status"] = "complete"
            arm_report["peak_allocated_bytes"] = (
                int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else None
            )
            del optimizer, model
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
            report["status"] = f"{arm}_complete"; persist()
        verify_sources(sources)
        report["status"] = "preflight_complete" if args.preflight_only else "complete"
        report["source_unchanged"] = True
        initial_hashes = [arm.get("initial_state_sha256") for arm in report["arms"].values()]
        used_hashes = [arm.get("used_state_rows_sha256") for arm in report["arms"].values()]
        schedule_hashes = [arm.get("schedule_sha256") for arm in report["arms"].values()]
        report["paired_initializations_exact"] = (
            len(initial_hashes) == 2 and len(set(initial_hashes)) == 1
            and initial_hashes[0] == state_digest(initial_saved["weights"])
        )
        report["paired_schedule_exact"] = (
            len(used_hashes) == 2 and all(value is not None for value in used_hashes)
            and schedule_hashes == [schedule_sha, schedule_sha]
        )
        if not report["paired_initializations_exact"] or not report["paired_schedule_exact"]:
            raise ValueError("paired initialization or schedule witness failed")
    except BaseException as error:
        report.update(status="failed_partial", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        persist()


if __name__ == "__main__":
    main()
