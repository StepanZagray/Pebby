"""Bounded deterministic preparation of split-safe whole-game datasets."""

from __future__ import annotations

from collections import Counter
import copy
from dataclasses import asdict, dataclass, field
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pebby import multigame as M
from pebby.agent.multigame_training import (
    audit_manifest_pair,
    puzzle_identity,
    sha256_file,
)
from pebby.multigame_variants import VariantOptions, WholeGameVariant

DEFAULT_RECOVERY_SECONDS = 20.0
DEFAULT_GAME_SECONDS = 300.0


FORMAT = "pebby-multigame-dataset-preparation-v1"
_ARTIFACT_LABELS = ("public_npz", "teacher_npz", "generated_specs")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class DatasetPreparationConfig:
    output_root: Path
    train_master_seed: int
    validation_master_seed: int
    games: tuple[str, ...] | None = None
    smoke: bool = False
    completed_teacher_games_per_family: int = 1
    mixed_games_per_family: int = 1
    mixed_epsilon: float = 0.2
    difficulties: tuple[int, ...] | None = None
    limits: M.SearchLimits | None = None
    outer_generation_attempts: int = 8
    generator_attempts: int = 50
    max_candidate_attempts: int = 8
    max_game_steps: int = 4096
    variants: VariantOptions = field(default_factory=VariantOptions)
    recovery_seconds: float | None = DEFAULT_RECOVERY_SECONDS
    game_seconds: float | None = DEFAULT_GAME_SECONDS
    perturbation: str = "random"
    learner_checkpoint: Path | None = None
    click_region_probe_limit: int = M.CLICK_REGION_PROBE_LIMIT

    def __post_init__(self) -> None:
        if (isinstance(self.click_region_probe_limit, bool)
                or not isinstance(self.click_region_probe_limit, Integral)
                or self.click_region_probe_limit < 0):
            raise ValueError("click_region_probe_limit must be a non-negative integer")
        for name in ("recovery_seconds", "game_seconds"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Real) or not value > 0:
                raise ValueError(f"{name} must be a positive number of seconds or None")
            object.__setattr__(self, name, float(value))
        if self.perturbation not in M.PERTURBATION_MODES:
            raise ValueError(f"perturbation must be one of {M.PERTURBATION_MODES}")
        if self.perturbation == "learner":
            if self.learner_checkpoint is None:
                raise ValueError("learner perturbation requires learner_checkpoint")
            object.__setattr__(self, "learner_checkpoint", Path(self.learner_checkpoint))
        elif self.learner_checkpoint is not None:
            raise ValueError("learner_checkpoint is only meaningful with perturbation='learner'")
        for name in (
            "train_master_seed", "validation_master_seed",
            "completed_teacher_games_per_family", "mixed_games_per_family",
            "outer_generation_attempts", "generator_attempts",
            "max_candidate_attempts", "max_game_steps",
        ):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if self.train_master_seed == self.validation_master_seed:
            raise ValueError("train and validation master seeds must be different")
        if self.completed_teacher_games_per_family < 1:
            raise ValueError("completed_teacher_games_per_family must be positive")
        if self.mixed_games_per_family < 0:
            raise ValueError("mixed_games_per_family cannot be negative")
        if not isinstance(self.mixed_epsilon, Real) or isinstance(self.mixed_epsilon, bool):
            raise ValueError("mixed_epsilon must be a real number")
        if not 0.0 <= self.mixed_epsilon <= 1.0:
            raise ValueError("mixed_epsilon must be in [0,1]")
        if self.mixed_games_per_family and self.mixed_epsilon <= 0.0:
            raise ValueError("mixed cohorts require a positive mixed_epsilon")
        if any(getattr(self, name) < 1 for name in (
            "outer_generation_attempts", "generator_attempts",
            "max_candidate_attempts", "max_game_steps",
        )):
            raise ValueError("generation, candidate, and step bounds must be positive")
        if self.games is None and self.smoke:
            raise ValueError("--smoke requires an explicit --games subset")
        if self.games is not None and not self.smoke:
            raise ValueError("an explicit --games selection requires --smoke")
        if self.games is None and self.difficulties is not None:
            raise ValueError("full preparation uses each family's complete declared curriculum")
        if self.games is not None and self.difficulties is None:
            object.__setattr__(self, "difficulties", (1, 2, 3))
        if self.difficulties is not None:
            if not self.difficulties or any(
                not isinstance(value, Integral) or isinstance(value, bool) or value < 1
                for value in self.difficulties
            ):
                raise ValueError("difficulties must be a non-empty sequence of positive integers")
            if len(set(self.difficulties)) != len(self.difficulties):
                raise ValueError("difficulties must be distinct")
        if self.limits is None:
            object.__setattr__(
                self,
                "limits",
                M.SearchLimits(
                    max_search_work=(
                        M.MAX_FULL_SEARCH_WORK if self.games is None else 2_000_000
                    ),
                ),
            )

    def parameter_payload(self) -> dict[str, Any]:
        return {
            "train_master_seed": self.train_master_seed,
            "validation_master_seed": self.validation_master_seed,
            "games": None if self.games is None else list(self.games),
            "smoke": self.smoke,
            "completed_teacher_games_per_family": self.completed_teacher_games_per_family,
            "mixed_games_per_family": self.mixed_games_per_family,
            "mixed_epsilon": self.mixed_epsilon,
            "difficulties": (
                None if self.difficulties is None else list(self.difficulties)
            ),
            "limits": asdict(self.limits),
            "outer_generation_attempts": self.outer_generation_attempts,
            "generator_attempts": self.generator_attempts,
            "max_candidate_attempts": self.max_candidate_attempts,
            "max_game_steps": self.max_game_steps,
            "variants": self.variants.to_dict(),
            "recovery_seconds": self.recovery_seconds,
            "game_seconds": self.game_seconds,
            "perturbation": self.perturbation,
            "click_region_probe_limit": self.click_region_probe_limit,
            "learner_checkpoint": (
                None if self.learner_checkpoint is None else str(self.learner_checkpoint)
            ),
        }

    def rollout_options(self, kind: str) -> M.RolloutOptions:
        """Rollout controls for a ``teacher`` or ``mixed`` cohort."""
        if kind not in ("teacher", "mixed"):
            raise ValueError(f"unknown cohort kind {kind!r}")
        return M.RolloutOptions(
            random_action_probability=0.0 if kind == "teacher" else self.mixed_epsilon,
            max_game_steps=self.max_game_steps,
            recovery_seconds=self.recovery_seconds,
            game_seconds=self.game_seconds,
            perturbation=self.perturbation,
            click_region_probe_limit=self.click_region_probe_limit,
            learner_checkpoint=(
                None if self.learner_checkpoint is None else str(self.learner_checkpoint)
            ),
        )


class LearnerPerturber(M.Perturber):
    """Drive a frozen multigame checkpoint as the perturbation policy.

    The model's ``policy_step`` is run on every public transition (teacher or
    perturbation) so its recurrent memory tracks the trajectory it is
    watching; the collector executes the returned action only at perturbation
    points.  Decisions use the same greedy readout as the official evaluator.
    """

    name = "learner"

    def __init__(self, checkpoint: str | Path, *, device: str = "cpu"):
        import torch
        from pebby.agent.multigame_training import model_from_training_checkpoint

        self._torch = torch
        self.checkpoint = Path(checkpoint).resolve()
        self.model, self.payload = model_from_training_checkpoint(self.checkpoint, device=device)
        self.checkpoint_sha256 = sha256_file(self.checkpoint)
        self.device = next(self.model.parameters()).device
        self.memory = None
        config = self.payload.get("training_config", {})
        self.history_mode = config.get("history_mode", self.payload.get("history_mode", "full"))
        if self.history_mode not in ("full", "none"):
            raise ValueError(f"unsupported checkpoint history mode: {self.history_mode!r}")
        self.canonical_inputs = bool(config.get(
            "canonical_inputs", self.payload.get("canonical_inputs", False),
        ))
        self.variant = WholeGameVariant.identity()

    def configure_variant(self, variant: WholeGameVariant) -> None:
        """Bind the public/raw adapter before the collector resets this game."""
        self.variant = variant

    def reset(self) -> None:
        self.memory = self.model.initial_memory(1, device=self.device)

    def step(self, frame, legal_action_mask, previous_action, previous_level_boundary):
        torch = self._torch
        from pebby.agent.multigame_model import CLICK_ACTION

        if self.memory is None:
            self.reset()
        frame = np.asarray(frame, dtype=np.uint8)
        legal = np.asarray(legal_action_mask, dtype=np.bool_)
        if self.canonical_inputs:
            frame = self.variant.raw_frame(frame)
            legal = legal[np.asarray(self.variant.control_raw_to_public)]
            if previous_action is not None:
                previous_action = M.Action(*self.variant.raw_action(*previous_action.as_tuple()))
        previous_id = M.COORDINATE_NONE if previous_action is None else int(previous_action.id)
        previous_x = (
            M.COORDINATE_NONE if previous_action is None or previous_action.x is None
            else int(previous_action.x)
        )
        previous_y = (
            M.COORDINATE_NONE if previous_action is None or previous_action.y is None
            else int(previous_action.y)
        )
        with torch.inference_mode():
            output, self.memory = self.model.policy_step(
                frame=torch.from_numpy(frame)[None].to(self.device),
                previous_action_id=torch.tensor([previous_id], device=self.device),
                previous_action_x=torch.tensor([previous_x], device=self.device),
                previous_action_y=torch.tensor([previous_y], device=self.device),
                previous_level_boundary=torch.tensor(
                    [bool(previous_level_boundary)], device=self.device,
                ),
                terminal=torch.tensor([False], device=self.device),
                won=torch.tensor([False], device=self.device),
                legal_action_mask=torch.from_numpy(legal)[None].to(self.device),
                memory=self.memory,
                **({"history_keep": torch.zeros(1, dtype=torch.bool, device=self.device)}
                   if self.history_mode == "none" else {}),
            )
            action_id = int(output.action_logits.argmax(-1).item())
            if action_id == CLICK_ACTION:
                click_x, click_y = self.model.decode_click(output.click_logits)
                x, y = int(click_x.item()), int(click_y.item())
            else:
                x = y = None
        if self.canonical_inputs:
            action_id, x, y = self.variant.public_action(action_id, x, y)
        return M.Action(action_id, x, y)


def build_perturber(rollout: M.RolloutOptions, *, device: str = "cpu") -> M.Perturber | None:
    """Load the perturbation policy a rollout configuration asks for (``None`` for random)."""
    if rollout.perturbation != "learner" or rollout.random_action_probability == 0.0:
        return None
    if rollout.learner_checkpoint is None:
        raise ValueError("learner perturbation requires a learner checkpoint")
    return LearnerPerturber(rollout.learner_checkpoint, device=device)


@dataclass(frozen=True)
class PreparationResult:
    output_root: Path
    summary_path: Path
    complete: bool
    summary: dict[str, Any]


@dataclass
class _Cohort:
    name: str
    split: str
    kind: str
    master_seed: int
    requested_per_source: int
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    report: dict[str, Any]


def _module_hash(module: Any) -> dict[str, Any]:
    raw_path = getattr(module, "__file__", None)
    if raw_path is None:
        return {"path": None, "sha256": None}
    path = Path(raw_path).resolve()
    return {
        "path": str(path),
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def _provenance(modules: Sequence[M.GameModules]) -> dict[str, Any]:
    sources = {}
    for package in modules:
        source = M.source_for(package.source)
        sources[source.source_id] = {
            "slug": source.slug,
            "solver_adapter": package.solver_adapter,
            "module_provenance_status": (
                None if package.provenance is None else package.provenance.get("status")
            ),
            "env": _module_hash(package.env),
            "generate": _module_hash(package.generate),
            "plan": _module_hash(package.plan),
        }
    return {
        "collector": _module_hash(M),
        "sources": sources,
    }


def _manifest_provenance_override(
    modules: Sequence[M.GameModules],
) -> dict[str, Any] | None:
    statuses = [
        None if package.provenance is None else package.provenance.get("status")
        for package in modules
    ]
    if all(status == "available" for status in statuses):
        # Production preflight modules take the normal M.new_manifest path,
        # which recomputes and requires complete source-code provenance.
        return None
    if any(status == "available" for status in statuses):
        raise ValueError("cannot mix production and synthetic module provenance")
    allowed = {None, "unavailable", "synthetic_test_fixture"}
    if any(status not in allowed for status in statuses):
        raise ValueError(f"unsupported injected module provenance statuses: {statuses}")
    sources = {}
    for package in modules:
        source = M.source_for(package.source)
        sources[source.source_id] = {
            "format": M.PROVENANCE_FORMAT,
            "status": "synthetic_test_fixture",
            "source_id": source.source_id,
            "source": source.slug,
            "reason": "injected GameModules without production preflight provenance",
        }
    return {
        "format": M.PROVENANCE_FORMAT,
        "status": "synthetic_test_fixture",
        "reason": "dataset-preparation dependency injection; not production code provenance",
        "shared_modules": {},
        "installed_packages": {},
        "sources": sources,
    }


def _variant_from_record(record: Mapping[str, Any]) -> WholeGameVariant:
    value = record.get("variant")
    if value is None:
        return WholeGameVariant.identity()
    return WholeGameVariant(
        value["selected"], value["seed"], value["control_raw_to_public"],
        value["spatial"], value["palette_raw_to_public"],
    )


def _candidate_identity(collected: M.CollectedGame):
    kwargs: dict[str, Any] = {}
    if collected.public is not None:
        kwargs = {
            "frames": collected.public["frames"],
            "level_boundaries": collected.public["level_boundary"],
            "variant": _variant_from_record(collected.record),
        }
    return puzzle_identity(collected.specs, label="collected candidate", **kwargs)


def _identity_keys(source_id: str, identity) -> set[tuple[str, str, str]]:
    keys = {
        (source_id, "effective_seed", str(value)) for value in identity.effective_seeds
    }
    keys.update((source_id, kind, digest) for kind, digest in identity.fingerprints)
    return keys


def _fallback_semantic_spec(value: Any) -> Any:
    """Remove generation diagnostics from a smoke spec identity fallback."""
    if isinstance(value, Mapping):
        return {
            str(key): _fallback_semantic_spec(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
            if not any(
                marker in str(key).lower()
                for marker in (
                    "seed", "solution", "proof", "plan", "search", "attempt",
                    "exclusion", "rejection", "diagnostic", "engine_verified",
                    "metadata", "generator_version", "sha256", "hash",
                    "calibration", "statistic", "coverage", "limitation", "note",
                )
            )
        }
    if isinstance(value, list):
        return [_fallback_semantic_spec(child) for child in value]
    return value


def _walk_spec_named(value: Any, names: set[str]) -> list[Any]:
    found = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in names:
                found.append(child)
            found.extend(_walk_spec_named(child, names))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_spec_named(child, names))
    return found


def _whole_game_fingerprint(
    source_id: str,
    specs: Sequence[Mapping[str, Any]],
    *,
    allow_canonical_fallback: bool = False,
) -> tuple[str, str]:
    """Ordered semantic identity for rejecting duplicate games within one cohort.

    Per-level geometry may repeat within a split (small tutorial tiers can have
    finite canonical classes). Prefer generator-authored gameplay/puzzle
    identities. A smoke-only canonical-spec fallback strips seeds, search
    proofs, rejection counters, and other generation diagnostics. Observed
    frames are deliberately absent, so a partial mixed trace cannot make the
    same generated game appear new.
    """
    if not specs or not all(isinstance(spec, Mapping) for spec in specs):
        raise ValueError("whole-game identity requires a non-empty spec sequence")
    ordered = []
    for index, spec in enumerate(specs):
        semantic = None
        for kind, names in (
            ("gameplay", {"gameplay_hash", "gameplay_sha256"}),
            ("puzzle", {"puzzle_hash", "puzzle_sha256"}),
        ):
            values = _walk_spec_named(spec, names)
            if values:
                if any(not isinstance(value, str) or not value for value in values):
                    raise ValueError(
                        f"whole-game spec {index} has a malformed {kind} identity"
                    )
                semantic = [kind, sorted(set(values))]
                break
        if semantic is None:
            if not allow_canonical_fallback:
                raise ValueError(
                    f"whole-game spec {index} lacks an explicit gameplay/puzzle identity"
                )
            canonical = _fallback_semantic_spec(spec)
            if canonical in ({}, []):
                raise ValueError(f"whole-game spec {index} has no semantic identity")
            semantic = ["canonical_spec", _json_hash(canonical)]
        ordered.append(semantic)
    return source_id, _json_hash(ordered)


def _bind_artifact_hashes(root: Path, record: dict[str, Any]) -> None:
    hashes = {}
    for label in _ARTIFACT_LABELS:
        relative = record.get(label)
        if relative is not None:
            hashes[label] = sha256_file(root / relative)
    record["file_hashes"] = hashes
    record_path = record.get("record")
    if record_path is None:
        raise ValueError("persisted record is missing its private record path")
    _atomic_json(root / record_path, record)


def _retained_transitions(collected: M.CollectedGame) -> int:
    if collected.public is None or collected.teacher is None:
        return 0
    actions = np.asarray(collected.public.get("action_id", ()))
    frames = np.asarray(collected.public.get("frames", ()))
    sources = np.asarray(collected.teacher.get("source", ()))
    if actions.ndim != 1 or sources.shape != actions.shape or len(frames) != len(actions) + 1:
        return 0
    return len(actions)


def _completed_teacher_game(collected: M.CollectedGame, requested_levels: int) -> bool:
    record = collected.record
    public = collected.public
    if public is None or collected.teacher is None:
        return False
    transitions = _retained_transitions(collected)
    if transitions <= 0 or len(collected.specs) != requested_levels:
        return False
    try:
        action_sources = np.asarray(collected.teacher["source"])
        return bool(
            record.get("status") == "won"
            and int(record.get("levels_generated", -1)) == requested_levels
            and int(record.get("levels_completed", -1)) == requested_levels
            and str(record.get("final_state", "")).upper() == "WIN"
            and bool(np.asarray(public["terminal"])[-1])
            and bool(np.asarray(public["won"])[-1])
            and int(np.asarray(public["levels_completed"])[-1]) == requested_levels
            and int(np.asarray(public["level_boundary"]).sum()) == requested_levels
            and action_sources.shape == (transitions,)
            and bool(np.all(action_sources == M.TEACHER_SOURCE))
            and int(record.get("random_steps", -1)) == 0
            and int(record.get("teacher_steps", -1)) == transitions
        )
    except (IndexError, KeyError, TypeError, ValueError):
        return False


def _cohort_manifest(
    *,
    name: str,
    split: str,
    kind: str,
    root: Path,
    master_seed: int,
    requested: int,
    sources: Sequence[M.Source],
    config: DatasetPreparationConfig,
    parameter_hash: str,
    provenance_hash: str,
    source_hash: str,
    manifest_provenance: Mapping[str, Any] | None,
    curricula_by_source: Mapping[str, Sequence[M.CurriculumEntry]],
    full_standard_contracts: Mapping[str, M.FullStandardContract] | None,
) -> _Cohort:
    rollout = config.rollout_options(kind)
    manifest = M.new_manifest(
        sources=sources,
        explicit_subset=config.smoke,
        seed=master_seed,
        games_per_source=requested,
        difficulties=config.difficulties,
        limits=config.limits,
        variants=config.variants,
        rollout=rollout,
        provenance=manifest_provenance,
        curricula_by_source=curricula_by_source,
        full_standard_contracts=full_standard_contracts,
    )
    manifest.update({
        "scope": "preparing_smoke_subset" if config.smoke else "preparing_full_24_source_collection",
        "is_full_experiment_collection": False,
        "preparation_complete": False,
        "preparation_format": FORMAT,
        "preparation_cohort": name,
        "preparation_split": split,
        "preparation_kind": kind,
        "preparation_parameter_hash": parameter_hash,
        "preparation_provenance_hash": provenance_hash,
        "preparation_source_hash": source_hash,
        "max_candidate_attempts": config.max_candidate_attempts,
    })
    report = {
        "name": name,
        "split": split,
        "kind": kind,
        "requested_per_source": requested,
        "attempted": 0,
        "accepted": 0,
        "rejected_overlap": 0,
        "rejected_duplicate": 0,
        "failed": 0,
        "retained_partial": 0,
        "random_steps": 0,
        "learner_steps": 0,
        "teacher_steps": 0,
        "live_recovery_steps": 0,
        "recovery_timeouts": 0,
        "game_timeouts": 0,
        "recovery_seconds": rollout.recovery_seconds,
        "game_seconds": rollout.game_seconds,
        "perturbation": rollout.perturbation,
        "sources": {},
    }
    return _Cohort(
        name, split, kind, master_seed, requested, root, root / "manifest.json", manifest, report,
    )


def _record_outcome(source_report: dict[str, Any], outcome: dict[str, Any]) -> None:
    source_report["attempts"].append(outcome)
    source_report["counts"][outcome["outcome"]] += 1


def _collect_cohort(
    cohort: _Cohort,
    modules: Sequence[M.GameModules],
    config: DatasetPreparationConfig,
    curricula_by_source: Mapping[str, Sequence[M.CurriculumEntry]],
    *,
    train_identity_keys: set[tuple[str, str, str]],
    collect_fn: Callable[..., M.CollectedGame],
    progress: Callable[[str], None],
    perturber: M.Perturber | None = None,
) -> bool:
    rollout = config.rollout_options(cohort.kind)
    collect_kwargs: dict[str, Any] = {}
    if cohort.kind == "mixed" and perturber is not None:
        collect_kwargs["perturber"] = perturber
    complete = True
    cohort_whole_games: set[tuple[str, str]] = set()
    for package in modules:
        source = M.source_for(package.source)
        curriculum = tuple(curricula_by_source[source.source_id])
        difficulties = tuple(entry.difficulty for entry in curriculum)
        source_report = {
            "requested": cohort.requested_per_source,
            "accepted": 0,
            "random_steps": 0,
            "learner_steps": 0,
            "live_recovery_steps": 0,
            "recovery_timeouts": 0,
            "game_timeouts": 0,
            "attempts": [],
            "counts": Counter(),
        }
        fingerprint_values: dict[str, set[str]] = {}
        fingerprint_observations: Counter[str] = Counter()
        cohort.report["sources"][source.source_id] = source_report
        candidate_index = 0
        for slot in range(cohort.requested_per_source):
            accepted = False
            for attempt in range(config.max_candidate_attempts):
                game_index = candidate_index
                candidate_index += 1
                cohort.report["attempted"] += 1
                progress(
                    f"{cohort.name} {source.slug} slot={slot + 1}/{cohort.requested_per_source} "
                    f"candidate={game_index} attempt={attempt + 1}/{config.max_candidate_attempts}"
                )
                try:
                    collected = collect_fn(
                        package,
                        master_seed=cohort.master_seed,
                        game_index=game_index,
                        difficulties=difficulties,
                        curriculum=curriculum,
                        split=cohort.split if not config.smoke else None,
                        require_full_standard=not config.smoke,
                        limits=config.limits,
                        outer_generation_attempts=config.outer_generation_attempts,
                        generator_attempts=config.generator_attempts,
                        variants=config.variants,
                        rollout=rollout,
                        **collect_kwargs,
                    )
                except Exception as exc:
                    cohort.report["failed"] += 1
                    _record_outcome(source_report, {
                        "candidate_index": game_index,
                        "outcome": "failed",
                        "reason": f"collector_exception:{type(exc).__name__}:{exc}",
                    })
                    continue
                record = collected.record
                steps = _retained_transitions(collected)
                if cohort.kind == "teacher" and not _completed_teacher_game(
                    collected, len(curriculum),
                ):
                    cohort.report["failed"] += 1
                    _record_outcome(source_report, {
                        "candidate_index": game_index,
                        "outcome": "failed",
                        "status": record.get("status"),
                        "steps": steps,
                        "reason": (
                            "teacher candidate did not prove a public sequential win over every "
                            "requested generated level"
                        ),
                        "errors": list(record.get("errors", ())),
                    })
                    continue
                if cohort.kind == "mixed" and steps <= 0:
                    cohort.report["failed"] += 1
                    _record_outcome(source_report, {
                        "candidate_index": game_index,
                        "outcome": "failed",
                        "status": record.get("status"),
                        "steps": steps,
                        "reason": "mixed candidate retained no transition",
                        "errors": list(record.get("errors", ())),
                    })
                    continue
                try:
                    identity = _candidate_identity(collected)
                except (KeyError, TypeError, ValueError) as exc:
                    cohort.report["failed"] += 1
                    _record_outcome(source_report, {
                        "candidate_index": game_index,
                        "outcome": "failed",
                        "status": record.get("status"),
                        "steps": steps,
                        "reason": f"identity_error:{type(exc).__name__}:{exc}",
                    })
                    continue
                keys = _identity_keys(source.source_id, identity)
                overlap = keys & train_identity_keys if cohort.split == "validation" else set()
                if overlap:
                    cohort.report["rejected_overlap"] += 1
                    sample = [list(value) for value in sorted(overlap)[:3]]
                    _record_outcome(source_report, {
                        "candidate_index": game_index,
                        "outcome": "rejected_overlap",
                        "status": record.get("status"),
                        "steps": steps,
                        "reason": "known puzzle identity overlaps training split",
                        "overlap_sample": sample,
                    })
                    continue
                whole_game = _whole_game_fingerprint(
                    source.source_id,
                    collected.specs,
                    allow_canonical_fallback=config.smoke,
                )
                if whole_game in cohort_whole_games:
                    cohort.report["rejected_duplicate"] += 1
                    _record_outcome(source_report, {
                        "candidate_index": game_index,
                        "outcome": "rejected_duplicate",
                        "status": record.get("status"),
                        "steps": steps,
                        "reason": "duplicate complete whole-game fingerprint within cohort",
                        "whole_game_fingerprint": whole_game[1],
                    })
                    continue

                key = f"{source.slug}-{game_index:06d}"
                if cohort.manifest["provenance"]["status"] == "synthetic_test_fixture":
                    collected.record["provenance"] = copy.deepcopy(
                        cohort.manifest["provenance"]["sources"][source.source_id]
                    )
                    collected.record["provenance_location"] = (
                        "private record JSON only; absent from public rollout arrays"
                    )
                collected.record["preparation_candidate_index"] = game_index
                collected.record["preparation_slot"] = slot
                saved = M.save_collected_game(cohort.root, key, collected)
                _bind_artifact_hashes(cohort.root, saved)
                cohort.manifest["records"].append(saved)
                M.update_manifest_summary(cohort.manifest)
                M.save_manifest(cohort.manifest_path, cohort.manifest)
                if cohort.split == "train":
                    train_identity_keys.update(keys)
                cohort_whole_games.add(whole_game)
                for kind_name, digest in identity.fingerprints:
                    fingerprint_values.setdefault(kind_name, set()).add(digest)
                    fingerprint_observations[kind_name] += 1
                partial = saved.get("status") != "won"
                recovery_timeouts = int(saved.get("recovery_timeouts", 0))
                game_timeout = bool(saved.get("game_timeout", False))
                live_recovery = int(saved.get("live_recovery_teacher_actions", 0))
                cohort.report["accepted"] += 1
                cohort.report["retained_partial"] += int(partial)
                cohort.report["random_steps"] += int(saved.get("random_steps", 0))
                cohort.report["learner_steps"] += int(saved.get("learner_steps", 0))
                cohort.report["teacher_steps"] += int(saved.get("teacher_steps", 0))
                cohort.report["live_recovery_steps"] += live_recovery
                cohort.report["recovery_timeouts"] += recovery_timeouts
                cohort.report["game_timeouts"] += int(game_timeout)
                source_report["accepted"] += 1
                source_report["random_steps"] += int(saved.get("random_steps", 0))
                source_report["learner_steps"] += int(saved.get("learner_steps", 0))
                source_report["live_recovery_steps"] += live_recovery
                source_report["recovery_timeouts"] += recovery_timeouts
                source_report["game_timeouts"] += int(game_timeout)
                _record_outcome(source_report, {
                    "candidate_index": game_index,
                    "outcome": "accepted_partial" if partial else "accepted_won",
                    "status": saved.get("status"),
                    "steps": steps,
                    "random_steps": int(saved.get("random_steps", 0)),
                    "learner_steps": int(saved.get("learner_steps", 0)),
                    "live_recovery_steps": live_recovery,
                    "recovery_timeouts": recovery_timeouts,
                    "game_timeout": game_timeout,
                    "timeout": saved.get("timeout"),
                    "route_source_counts": saved.get("route_source_counts", {}),
                    "record": saved["record"],
                })
                progress(
                    f"accepted {cohort.name} {source.slug} candidate={game_index} "
                    f"status={saved.get('status')} steps={steps} "
                    f"recovery={live_recovery} timeouts={recovery_timeouts}"
                    f"{' game_timeout' if game_timeout else ''}"
                )
                accepted = True
                break
            if not accepted:
                complete = False
                progress(
                    f"incomplete {cohort.name} {source.slug} slot={slot + 1}: "
                    f"exhausted {config.max_candidate_attempts} candidates"
                )
                break
        source_report["counts"] = dict(sorted(source_report["counts"].items()))
        source_report["accepted_fingerprint_diversity"] = {
            kind_name: {
                "unique": len(values),
                "observations": fingerprint_observations[kind_name],
            }
            for kind_name, values in sorted(fingerprint_values.items())
        }
    return complete


def _finalize_manifest(cohort: _Cohort, *, complete: bool, smoke: bool) -> None:
    M.update_manifest_summary(cohort.manifest)
    cohort.manifest["preparation_report"] = cohort.report
    cohort.manifest["preparation_complete"] = bool(complete)
    cohort.manifest["is_full_experiment_collection"] = bool(complete and not smoke)
    if complete:
        cohort.manifest["scope"] = (
            "smoke_subset" if smoke else "full_24_source_collection"
        )
    else:
        cohort.manifest["scope"] = (
            "incomplete_smoke_subset" if smoke else "incomplete_full_24_source_collection"
        )
    M.save_manifest(cohort.manifest_path, cohort.manifest)


def _validate_modules(
    config: DatasetPreparationConfig,
    modules: Sequence[M.GameModules] | None,
) -> tuple[M.GameModules, ...]:
    resolved = tuple(
        M.preflight(config.games, require_full_standard=not config.smoke)
        if modules is None else modules
    )
    if not resolved:
        raise ValueError("dataset preparation requires at least one source")
    sources = tuple(M.source_for(package.source) for package in resolved)
    source_ids = [source.source_id for source in sources]
    if any(source.held_out for source in sources):
        raise ValueError("held-out m0r0 cannot be prepared as training data")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("dataset preparation sources must be distinct")
    if config.games is None:
        if set(source_ids) != set(M.TRAIN_SOURCE_IDS):
            raise ValueError("default dataset preparation requires all 24 training families")
        for package in resolved:
            M.curriculum_for(package, require_full_standard=True)
    else:
        expected = [source.source_id for source in M.resolve_collection_sources(config.games)]
        if source_ids != expected:
            raise ValueError("preflight modules do not match the explicit smoke sources")
    return resolved


def prepare_multigame_dataset(
    config: DatasetPreparationConfig,
    *,
    modules: Sequence[M.GameModules] | None = None,
    collect_fn: Callable[..., M.CollectedGame] = M.collect_generated_game,
    progress: Callable[[str], None] = print,
    perturber_factory: Callable[[M.RolloutOptions], M.Perturber | None] = build_perturber,
) -> PreparationResult:
    """Prepare bounded split-safe cohorts and run the trainer's strict audit.

    Mixed cohorts perturb with uniform random actions or, when the
    configuration selects ``perturbation='learner'``, with the actions of a
    loaded learner checkpoint (``perturber_factory`` builds it from the mixed
    rollout options); the family teacher then recovers under the configured
    wall-clock caps.
    """
    resolved = _validate_modules(config, modules)
    perturber = (
        perturber_factory(config.rollout_options("mixed"))
        if config.mixed_games_per_family else None
    )
    output_root = Path(config.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"output root is not empty: {output_root}")

    sources = tuple(M.source_for(package.source) for package in resolved)
    assert config.limits is not None
    curricula_by_source = {
        package.source.source_id: M.curriculum_for(
            package,
            config.difficulties,
            require_full_standard=not config.smoke,
            smoke_search_work=config.limits.max_search_work,
        )
        for package in resolved
    }
    if any(
        entry.search_work > config.limits.max_search_work
        for curriculum in curricula_by_source.values()
        for entry in curriculum
    ):
        raise ValueError("configured search-work ceiling is below a family curriculum")
    full_standard_contracts = (
        None if config.smoke else {
            package.source.source_id: package.full_standard for package in resolved
        }
    )
    parameters = config.parameter_payload()
    parameter_hash = _json_hash(parameters)
    provenance = _provenance(resolved)
    provenance_hash = _json_hash(provenance)
    manifest_provenance = _manifest_provenance_override(resolved)
    source_payload = [asdict(source) for source in sources]
    source_hash = _json_hash(source_payload)

    definitions = [
        ("train-teacher", "train", "teacher", config.train_master_seed,
         config.completed_teacher_games_per_family),
        ("validation-teacher", "validation", "teacher", config.validation_master_seed,
         config.completed_teacher_games_per_family),
    ]
    if config.mixed_games_per_family:
        definitions[1:1] = [
            ("train-mixed", "train", "mixed", config.train_master_seed,
             config.mixed_games_per_family),
        ]
        definitions.append(
            ("validation-mixed", "validation", "mixed", config.validation_master_seed,
             config.mixed_games_per_family),
        )
    cohorts = [
        _cohort_manifest(
            name=name,
            split=split,
            kind=kind,
            root=output_root / name,
            master_seed=seed,
            requested=requested,
            sources=sources,
            config=config,
            parameter_hash=parameter_hash,
            provenance_hash=provenance_hash,
            source_hash=source_hash,
            manifest_provenance=manifest_provenance,
            curricula_by_source=curricula_by_source,
            full_standard_contracts=full_standard_contracts,
        )
        for name, split, kind, seed, requested in definitions
    ]

    output_root.mkdir(parents=True, exist_ok=True)
    for cohort in cohorts:
        cohort.root.mkdir(parents=True, exist_ok=True)
        M.save_manifest(cohort.manifest_path, cohort.manifest)

    train_keys: set[tuple[str, str, str]] = set()
    cohort_complete: dict[str, bool] = {}
    for cohort in cohorts:
        cohort_complete[cohort.name] = _collect_cohort(
            cohort,
            resolved,
            config,
            curricula_by_source,
            train_identity_keys=train_keys,
            collect_fn=collect_fn,
            progress=progress,
            perturber=perturber,
        )
    collection_complete = all(cohort_complete.values())
    for cohort in cohorts:
        _finalize_manifest(cohort, complete=collection_complete, smoke=config.smoke)

    train_manifests = [cohort.manifest_path for cohort in cohorts if cohort.split == "train"]
    validation_manifests = [
        cohort.manifest_path for cohort in cohorts if cohort.split == "validation"
    ]
    audit_summary = None
    audit_error = None
    if collection_complete:
        try:
            audit_summary = audit_manifest_pair(
                train_manifests, validation_manifests, smoke=config.smoke,
            ).summary()
        except (OSError, TypeError, ValueError) as exc:
            audit_error = f"{type(exc).__name__}: {exc}"
            collection_complete = False
            for cohort in cohorts:
                _finalize_manifest(cohort, complete=False, smoke=config.smoke)

    manifest_hashes = {
        cohort.name: sha256_file(cohort.manifest_path) for cohort in cohorts
    }
    summary = {
        "format": FORMAT,
        "complete": collection_complete,
        "scope": "smoke" if config.smoke else "full_24_family_experiment",
        "parameters": parameters,
        "parameter_hash": parameter_hash,
        "source_ids": [source.source_id for source in sources],
        "source_hash": source_hash,
        "curricula_by_source": {
            source_id: [entry.to_dict() for entry in curriculum]
            for source_id, curriculum in sorted(curricula_by_source.items())
        },
        "full_standard": {
            "required": not config.smoke,
            "ready": bool(not config.smoke),
            "contract_hashes": (
                {} if config.smoke else {
                    source_id: contract.sha256
                    for source_id, contract in sorted(full_standard_contracts.items())
                }
            ),
            "contract_caveats": (
                {} if config.smoke else {
                    source_id: list(contract.caveats)
                    for source_id, contract in sorted(full_standard_contracts.items())
                }
            ),
        },
        "provenance": provenance,
        "provenance_hash": provenance_hash,
        "cohorts": {cohort.name: cohort.report for cohort in cohorts},
        "cohort_complete": cohort_complete,
        "manifests": {
            "train": [str(path.relative_to(output_root)) for path in train_manifests],
            "validation": [
                str(path.relative_to(output_root)) for path in validation_manifests
            ],
        },
        "manifest_hashes": manifest_hashes,
        "strict_audit": audit_summary,
        "strict_audit_error": audit_error,
        "training_cli_arguments": {
            "train_manifest": [str(path) for path in train_manifests],
            "validation_manifest": [str(path) for path in validation_manifests],
            "smoke": config.smoke,
        },
        "limitations": (
            "Initial-frame fingerprints can miss hidden state and canonical specs can over- or "
            "under-identify equality. Per-level tutorial geometry may repeat within a split; "
            "complete whole-game fingerprints are deduplicated within each cohort, and every "
            "detected train/validation identity overlap is rejected. Per-source accepted "
            "fingerprint diversity counts and contract caveats are retained for experiment review."
        ),
    }
    summary_path = output_root / "preparation.json"
    _atomic_json(summary_path, summary)
    progress(
        f"preparation {'complete' if collection_complete else 'incomplete'}: {summary_path}"
    )
    return PreparationResult(output_root, summary_path, collection_complete, summary)


__all__ = [
    "DEFAULT_GAME_SECONDS", "DEFAULT_RECOVERY_SECONDS", "DatasetPreparationConfig", "FORMAT",
    "LearnerPerturber", "PreparationResult", "build_perturber", "prepare_multigame_dataset",
]
