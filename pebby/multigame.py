"""Shared, leak-resistant plumbing for generated ARC-AGI-3 whole games.

The adapters in this module deliberately expose only public gameplay state to
the rollout arrays.  Generator specs, planner results, and teacher provenance
belong in separate metadata files.  ``m0r0`` is registered so evaluation code
can name it, but every collection entry point rejects it.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib
import importlib.util
from importlib import metadata as importlib_metadata
import inspect
import io
import json
from numbers import Integral
import os
from pathlib import Path
import signal
import threading
import time
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .multigame_variants import (
    VariantOptions,
    WholeGameVariant,
    sample_whole_game_variant,
    transform_frame,
)


FORMAT = "pebby-multigame-v1"
MANIFEST_FORMAT = "pebby-multigame-manifest-v1"
PROVENANCE_FORMAT = "pebby-code-provenance-v1"
FULL_STANDARD_FORMAT = "pebby-full-generator-contract-v1"
MAX_FULL_SEARCH_WORK = 32_000_000
ROOT = Path(__file__).resolve().parents[1]
FRAME_SHAPE = (64, 64)
ACTION_COUNT = 8  # RESET plus ACTION1..ACTION7
CLICK_ACTION = 6
COORDINATE_NONE = -1
TEACHER_SOURCE = 1
RANDOM_SOURCE = 0
CERTIFIED_ROUTE_SOURCE = "certified_spec_solution"
RECOVERY_ROUTE_SOURCE = "live_recovery"
LEGACY_ROUTE_SOURCE = "live_search"
RANDOM_ROUTE_SOURCE = "random_action"
LEARNER_ROUTE_SOURCE = "learner_action"
PERTURBATION_ROUTE_SOURCES = (RANDOM_ROUTE_SOURCE, LEARNER_ROUTE_SOURCE)
PERTURBATION_MODES = ("random", "learner")
RECOVERY_TIMEOUT_REASON = "recovery_timeout"
GAME_TIMEOUT_REASON = "game_timeout"
# Equivalent-click regions: every teacher click target is probed on clones of
# the pre-state so the label is the set of pixels whose click yields the same
# public successor.  The probe is bounded to this many candidate pixels.
CLICK_REGION_PROBE_LIMIT = 64
CLICK_REGION_NEIGHBOURHOOD = 2  # Chebyshev radius: a 5x5 window around the target
FULL_STANDARD_EVIDENCE = (
    "official_tier_characterization",
    "solution_mechanics",
    "native_budget",
    "context_engine_replay",
    "novelty_split",
    "bounded_rejections",
)


class MultiGameError(RuntimeError):
    """Base class for explicit multi-game pipeline failures."""


class PreflightError(MultiGameError):
    """The requested source packages do not satisfy the collection contract."""

    def __init__(self, message: str, availability: Sequence["Availability"] = ()):
        super().__init__(message)
        self.availability = tuple(availability)


class GenerationError(MultiGameError):
    """A bounded series of generator attempts produced no usable level."""

    def __init__(
        self,
        message: str,
        partial_specs: Sequence[Mapping[str, Any]] = (),
        partial_metadata: Sequence[Mapping[str, Any]] = (),
    ):
        super().__init__(message)
        self.partial_specs = [dict(value) for value in partial_specs]
        self.partial_metadata = [dict(value) for value in partial_metadata]


class RecoveryTimeout(BaseException):
    """A live family search exceeded its wall-clock cap.

    Derived from ``BaseException`` so a family planner's broad ``except
    Exception`` cannot swallow the interruption.
    """

    def __init__(self, seconds: float):
        super().__init__(f"live search exceeded its {seconds:.3f}s wall-clock cap")
        self.seconds = float(seconds)


def _call_with_deadline(fn: Any, seconds: float | None) -> Any:
    """Run ``fn()`` and raise :class:`RecoveryTimeout` after ``seconds``.

    Family solvers are ordinary Python without any interruption hook, so the
    cap is imposed from outside.  In the main thread a real ``ITIMER_REAL``
    alarm raises inside the running solver at its next bytecode boundary; that
    works for every family, including constructive builders that never consult
    a node budget.  Off the main thread the solver runs in a daemon worker and
    is abandoned on timeout (it keeps running until it finishes on its own).
    """
    if seconds is None:
        return fn()
    if seconds <= 0.0:
        raise RecoveryTimeout(max(seconds, 0.0))
    in_main_thread = threading.current_thread() is threading.main_thread()
    if in_main_thread and hasattr(signal, "setitimer") and hasattr(signal, "SIGALRM"):
        state = {"armed": True}

        def handler(signum, frame):  # noqa: ARG001 - signal handler signature
            if state["armed"]:
                raise RecoveryTimeout(seconds)

        previous = signal.signal(signal.SIGALRM, handler)
        try:
            signal.setitimer(signal.ITIMER_REAL, seconds)
            result = fn()
            state["armed"] = False
            return result
        finally:
            state["armed"] = False
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous)
    outcome: dict[str, Any] = {}

    def worker() -> None:
        try:
            outcome["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            outcome["error"] = exc

    thread = threading.Thread(target=worker, name="pebby-live-search", daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        raise RecoveryTimeout(seconds)
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


class SolverError(MultiGameError):
    """A live level could not be solved within the configured bounds."""


class IllegalActionError(MultiGameError, ValueError):
    """An action is malformed or unavailable in the current public state."""


class FrameError(MultiGameError, ValueError):
    """A game returned something other than a raw 64x64 palette frame."""


@dataclass(frozen=True)
class Source:
    slug: str
    source_id: str
    title: str
    held_out: bool = False

    @property
    def package(self) -> str:
        return f"pebby.games.{self.slug}"


# Canonical ids come from third_party/arc3_games/games.json.  Registry order is
# kept stable so deterministic collection does not depend on directory order.
SOURCES = (
    Source("lf52", "lf52-271a04aa", "LF52"),
    Source("s5i5", "s5i5-18d95033", "S5I5"),
    Source("lp85", "lp85-305b61c3", "LP85"),
    Source("wa30", "wa30-ee6fef47", "WA30"),
    Source("ar25", "ar25-0c556536", "AR25"),
    Source("re86", "re86-8af5384d", "RE86"),
    Source("r11l", "r11l-495a7899", "R11L"),
    Source("sc25", "sc25-635fd71a", "SC25"),
    Source("ft09", "ft09-0d8bbf25", "FT09"),
    Source("ls20", "ls20-9607627b", "LS20"),
    Source("cn04", "cn04-2fe56bfb", "CN04"),
    Source("ka59", "ka59-38d34dbb", "KA59"),
    Source("tu93", "tu93-0768757b", "TU93"),
    Source("tr87", "tr87-cd924810", "TR87"),
    Source("m0r0", "m0r0-492f87ba", "M0R0", held_out=True),
    Source("dc22", "dc22-fdcac232", "DC22"),
    Source("sp80", "sp80-589a99af", "SP80"),
    Source("cd82", "cd82-fb555c5d", "CD82"),
    Source("tn36", "tn36-ef4dde99", "TN36"),
    Source("vc33", "vc33-5430563c", "VC33"),
    Source("bp35", "bp35-0a0ad940", "BP35"),
    Source("sb26", "sb26-7fbdac44", "SB26"),
    Source("su15", "su15-1944f8ab", "SU15"),
    Source("g50t", "g50t-5849a774", "G50T"),
    Source("sk48", "sk48-d8078629", "SK48"),
)
SOURCE_BY_SLUG = {source.slug: source for source in SOURCES}
SOURCE_BY_ID = {source.source_id: source for source in SOURCES}
TRAIN_SOURCES = tuple(source for source in SOURCES if not source.held_out)
TRAIN_SOURCE_IDS = tuple(source.source_id for source in TRAIN_SOURCES)
HELD_OUT_SOURCE_ID = SOURCE_BY_SLUG["m0r0"].source_id

# These are audit findings, not claims that availability implies full coverage.
# Unknown packages remain explicitly unaudited until their game-specific work
# supplies a stronger statement.
KNOWN_COVERAGE = {
    "cd82": "generated difficulties 1-3 and all six official levels verified by package tests",
    "ft09": (
        "all six official levels replay; generator omits tutorial hints/flash, all-special boards, "
        "and deliberate no-op clicks"
    ),
    "tr87": (
        "generator covers plain/alter/double/tree modes; stored isolated proofs are level-index "
        "dependent, and official level 5 reaches the default search cap"
    ),
    "s5i5": (
        "generator omits vertical rails, internal obstacles, shared-colour controls, multiple "
        "pins, and branching/deep child graphs"
    ),
    "sc25": (
        "generator covers movement, shrink, and fire; omits teleport, multiple spells, and "
        "alternate targets; partial spell-grid layouts are unsupported by the exact planner"
    ),
    "wa30": (
        "generator omits helpers, thieves, fences, and bad regions; exact teacher supports only "
        "one of nine official levels"
    ),
    "tu93": (
        "exact planner covers head movement/eating, hunters, patrollers, tails, budgets and exits; "
        "generated d1-d3 progress from plain mazes through hunters, one patroller and one tail"
    ),
    "r11l": (
        "generated d1-d3 cover ordinary full-pixel fragment dragging, walls and hazards; official "
        "pickup/colour-set levels 5-6 are explicitly unsupported"
    ),
    "cn04": (
        "generated d1-d3 cover exact singleton pin matching with 2-4 pieces; alternate stacks and "
        "their shared bounce-direction mechanic are explicitly unsupported"
    ),
    "ls20": (
        "generated tiers 1-3 cover static cyclers, refills, launchers and colour/rotation changes; "
        "moving rails, fog and multi-goal relations from tiers 5-7 are omitted"
    ),
    "sp80": (
        "generated d1-d3 cover horizontal splitters, 2/4/6 cups and rotated presentations; "
        "vertical and embedded-source pipes and corner deflectors are omitted, and guarded or "
        "cyclic flow exhaustion is unsupported/inconclusive"
    ),
    "sk48": (
        "generated d1-d3 cover clickable noncrossing same-orientation chain pairs; fixed blockers, "
        "crossing/mixed-orientation and non-clickable auxiliary chains, and pause/intersection "
        "effects are omitted; only official level 1 is asserted and negative search is inconclusive"
    ),
    "bp35": (
        "generated d1-d3 cover ordinary descending platforms with remote hazards; destructible, "
        "growth, bridge, spike and gravity-switch mechanics are omitted; supported live prefixes "
        "have replayed positive witnesses, while omitted undo branches make live negative results "
        "inconclusive, and only official level 1 is asserted"
    ),
    "ka59": (
        "generated d1-d3 cover clickable size-matched box movement, pushing and selection; internal "
        "walls, player routing, pursuing enemies, explosives and blast pushes are omitted, and only "
        "official level 1 is asserted"
    ),
    "ar25": (
        "generated d1-d3 cover one movable shape and one fixed horizontal or vertical mirror with "
        "live movement/click/cycle/undo replanning; multiple shapes, movable or recursive mirrors, "
        "rotation and reflection-restriction tags are omitted, and later official levels are not claimed"
    ),
    "vc33": (
        "generated d1-d3 cover 1-3 loads, both gravity axes, reversible support transfer and blank-click "
        "completion; swap bars, negative gravity, complex floor limits and shared supports are omitted, "
        "and only official level 1 is asserted"
    ),
    "su15": (
        "generated d1-d3 cover one ordinary fruit, one target zone, magnetic full-pixel movement, "
        "target-centre containment, budgets and source-native undo origins; equal-tier merging, "
        "unequal-tier penalties/automatic undo, pursuers, special fruit and multiple target zones or "
        "required counts are omitted; waypoint-search negatives are inconclusive and only official level 1 is claimed"
    ),
    "dc22": (
        "generated d1-d3 cover stable movement-only perfect-tree mazes, native support checks and budgets; "
        "clickable controls/cycling surfaces, keys/colour gates, fall-triggered undo penalties, crushers, "
        "bridges, carrying and colour cycling are omitted from generation; only official level 1 is claimed, "
        "and live fall/crusher animation states are unsupported"
    ),
    "lp85": (
        "generated d1-d3 cover 1/2/3 disjoint bidirectional rectangle cycles with normal/alternate goals; "
        "overlapping/shared cycles, stacked multi-group controls, moving/duplicated/one-direction-only controls "
        "and large interlocked layouts are omitted; the native-transition planner is broader, but only official "
        "level 1 has a solve/performance claim and capped larger graphs remain inconclusive"
    ),
    "re86": (
        "generated d1-d3 cover 1-3 distinct-colour independently translating rigid shapes, cyclic selection "
        "and sparse target anchors; obstacles/deformation, flexible resizing, fixed-centre pieces and dye/flood "
        "fill are omitted; only official level 1 is claimed, and ambiguous/overlapping anchors, secondary colours, "
        "target-constrained or restored intrinsic selection-marker centres, and terminal states are unsupported"
    ),
    "lf52": (
        "generated d1-d3 cover static single-colour peg jumps in certified trap-safe regions; movable rails, "
        "coloured pegs, blockers, scrolling, scripted trap/reset cells and undo-dependent solutions are omitted; "
        "only official level 1 is claimed, live-history negatives are unsupported/inexact, and external action "
        "caps are truncated"
    ),
    "sb26": (
        "generated d1-d3 cover one ordinary 3/5/7-cell frame with selection, swaps/placement, submit, energy "
        "and undo; multiple frames, recursive link tiles, nested traversal and link-cycle rejection are omitted; "
        "generated colours are distinct, only official level 1 is claimed, and bound failures are inconclusive"
    ),
    "tn36": (
        "generated d1-d3 cover one editable program progressing from translation through rotation, scale and "
        "recolour with live timer replanning; preset selection/history, collision rollback, destructive/toggling "
        "gates, scale-sensitive platforms and manual dual-panel execution are omitted; only official level 1 is claimed"
    ),
    "g50t": (
        "generated d1-d3 cover randomized induced-tree regions joined by a one-cell door bridge, an ordinary "
        "dead-end pressure switch/door, action-5 rewind and one replay ghost; toggle devices, multiple circuits, "
        "additional timeline slots or ghosts, enemies/guide paths and teleports are omitted; only official level 1 "
        "is claimed, and every failed dominance search is truncated/inconclusive"
    ),
}


def source_for(value: str | Source) -> Source:
    """Resolve a canonical source id or short slug without fuzzy matching."""
    if isinstance(value, Source):
        try:
            canonical = SOURCE_BY_SLUG[value.slug.lower()]
        except KeyError as exc:
            raise ValueError(f"unregistered source object {value!r}") from exc
        if value.source_id.lower() != canonical.source_id:
            raise ValueError(
                f"source object id {value.source_id!r} does not match registered {canonical.source_id!r}"
            )
        return canonical
    key = str(value).lower()
    try:
        return SOURCE_BY_SLUG[key]
    except KeyError:
        pass
    try:
        return SOURCE_BY_ID[key]
    except KeyError as exc:
        known = ", ".join(source.slug for source in SOURCES)
        raise ValueError(f"unknown game source {value!r}; expected one of: {known}") from exc


@dataclass(frozen=True)
class Availability:
    source_id: str
    slug: str
    ready: bool
    missing: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    solver_adapter: str | None = None
    full_standard_ready: bool = False
    full_standard_errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CurriculumEntry:
    difficulty: int
    context_index: int
    search_work: int

    def __post_init__(self) -> None:
        for name in ("difficulty", "context_index", "search_work"):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool):
                raise ValueError(f"curriculum {name} must be an integer")
            object.__setattr__(self, name, int(value))
        if self.difficulty < 1 or self.context_index < 0 or self.search_work < 1:
            raise ValueError("curriculum difficulty/search work must be positive and context nonnegative")
        if self.search_work > MAX_FULL_SEARCH_WORK:
            raise ValueError(f"curriculum search_work cannot exceed {MAX_FULL_SEARCH_WORK}")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class FullStandardContract:
    source_id: str
    status: str
    mechanics_inventory_version: str
    quality_profile_version: str
    official_level_count: int
    curriculum: tuple[CurriculumEntry, ...]
    evidence: tuple[tuple[str, str], ...]
    caveats: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FULL_STANDARD_FORMAT,
            "status": self.status,
            "source_id": self.source_id,
            "mechanics_inventory_version": self.mechanics_inventory_version,
            "quality_profile_version": self.quality_profile_version,
            "official_level_count": self.official_level_count,
            "curriculum": [entry.to_dict() for entry in self.curriculum],
            "evidence": dict(self.evidence),
            "caveats": list(self.caveats),
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class GameModules:
    source: Source
    env: ModuleType
    generate: ModuleType
    plan: ModuleType
    solver_adapter: str
    provenance: Mapping[str, Any] | None = None
    full_standard: FullStandardContract | None = None


def _module_name(path: Path) -> str:
    relative = path.relative_to(ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _file_identity(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    try:
        display = resolved.relative_to(ROOT)
    except ValueError:
        display = resolved
    return {
        "path": str(display),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


@lru_cache(maxsize=1)
def _shared_code_provenance() -> dict[str, dict[str, str]]:
    paths = (
        Path(__file__),
        ROOT / "pebby" / "multigame_variants.py",
        ROOT / "tools" / "collect_multigame_games.py",
    )
    return {_module_name(path): _file_identity(path) for path in paths}


@lru_cache(maxsize=1)
def _engine_versions() -> dict[str, dict[str, str | None]]:
    versions = {}
    for distribution, module_name in (("arcengine", "arcengine"), ("arc-agi", "arc_agi")):
        module = importlib.import_module(module_name)
        versions[distribution] = {
            "distribution_version": importlib_metadata.version(distribution),
            "module_version": getattr(module, "__version__", None),
        }
    return versions


def _actual_upstream_path(source: Source) -> Path:
    env_module = importlib.import_module(f"{source.package}.env")
    upstream_path = getattr(env_module, "UPSTREAM", None)
    if upstream_path is None:
        upstream_function = getattr(env_module, "upstream", None)
        owner_name = getattr(upstream_function, "__module__", None)
        if owner_name:
            owner = importlib.import_module(owner_name)
            upstream_path = getattr(owner, "UPSTREAM", None)
    if upstream_path is None:
        raise FileNotFoundError(f"{source.package}.env does not expose its actual vendored source path")
    return Path(upstream_path)


def _family_python_files(source: Source) -> tuple[Path, ...]:
    roots = [ROOT / "pebby" / "games" / source.slug]
    # LS20's compatibility package delegates its engine, layout, planner and
    # generator implementation to this legacy family package.
    if source.slug == "ls20":
        roots.append(ROOT / "pebby" / "ls20")
    paths = {path for root in roots for path in root.rglob("*.py")}
    if not paths:
        raise FileNotFoundError(f"no Python modules found for {source.package}")
    return tuple(sorted(paths))


@lru_cache(maxsize=None)
def _production_provenance(source_id: str) -> dict[str, Any]:
    source = source_for(source_id)
    family = {
        _module_name(path): _file_identity(path)
        for path in _family_python_files(source)
    }
    return {
        "format": PROVENANCE_FORMAT,
        "status": "available",
        "source_id": source.source_id,
        "source": source.slug,
        "vendored_game": _file_identity(_actual_upstream_path(source)),
        "family_modules": family,
        "shared_modules": copy.deepcopy(_shared_code_provenance()),
        "installed_packages": copy.deepcopy(_engine_versions()),
    }


def _unavailable_provenance(source: Source) -> dict[str, Any]:
    return {
        "format": PROVENANCE_FORMAT,
        "status": "unavailable",
        "source_id": source.source_id,
        "source": source.slug,
        "reason": "GameModules was constructed without production preflight provenance",
    }


def _manifest_provenance(sources: Sequence[Source]) -> dict[str, Any]:
    source_entries = {}
    for source in sources:
        provenance = _production_provenance(source.source_id)
        source_entries[source.source_id] = {
            key: copy.deepcopy(provenance[key])
            for key in ("source_id", "source", "vendored_game", "family_modules")
        }
    return {
        "format": PROVENANCE_FORMAT,
        "status": "available",
        "shared_modules": copy.deepcopy(_shared_code_provenance()),
        "installed_packages": copy.deepcopy(_engine_versions()),
        "sources": source_entries,
    }


def _explicit_manifest_provenance(
    sources: Sequence[Source], value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an explicitly labelled synthetic-fixture provenance payload."""
    if value.get("format") != PROVENANCE_FORMAT:
        raise ValueError(f"explicit manifest provenance format must be {PROVENANCE_FORMAT!r}")
    if value.get("status") != "synthetic_test_fixture":
        raise ValueError("explicit manifest provenance must be labelled synthetic_test_fixture")
    entries = value.get("sources")
    if not isinstance(entries, Mapping):
        raise ValueError("explicit manifest provenance sources must be an object")
    expected = {source.source_id: source for source in sources}
    if set(entries) != set(expected):
        raise ValueError("explicit manifest provenance source IDs do not match requested sources")
    for source_id, source in expected.items():
        entry = entries[source_id]
        if not isinstance(entry, Mapping):
            raise ValueError(f"explicit provenance for {source_id} must be an object")
        if (
            entry.get("source_id") != source_id
            or entry.get("source") != source.slug
            or entry.get("status") != "synthetic_test_fixture"
        ):
            raise ValueError(f"explicit provenance for {source_id} is not a canonical fixture entry")
    return copy.deepcopy(dict(value))


def _full_standard_contract(
    source: Source, env: ModuleType | Any, generate: ModuleType | Any,
) -> FullStandardContract:
    raw = getattr(generate, "FULL_STANDARD_CONTRACT", None)
    if not isinstance(raw, Mapping):
        raise ValueError("generate.FULL_STANDARD_CONTRACT is missing or is not an object")
    if raw.get("format") != FULL_STANDARD_FORMAT:
        raise ValueError(f"full-standard format must be {FULL_STANDARD_FORMAT!r}")
    if raw.get("source_id") != source.source_id:
        raise ValueError("full-standard source_id is not canonical for this package")
    status = raw.get("status")
    if status not in ("ready", "pending_audit", "rejected"):
        raise ValueError("full-standard status must be ready, pending_audit, or rejected")
    versions = {}
    for key in ("mechanics_inventory_version", "quality_profile_version"):
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"full-standard {key} must be a non-empty string")
        versions[key] = value

    raw_curriculum = raw.get("curriculum")
    if (
        not isinstance(raw_curriculum, Sequence)
        or isinstance(raw_curriculum, (str, bytes))
        or not raw_curriculum
    ):
        raise ValueError("full-standard curriculum must be a non-empty sequence")
    curriculum = []
    for index, value in enumerate(raw_curriculum):
        if not isinstance(value, Mapping):
            raise ValueError(f"full-standard curriculum[{index}] must be an object")
        try:
            curriculum.append(CurriculumEntry(
                value["difficulty"], value["context_index"], value["search_work"],
            ))
        except KeyError as exc:
            raise ValueError(f"full-standard curriculum[{index}] is missing {exc.args[0]}") from exc
    official_levels = getattr(env, "official_levels", None)
    if not callable(official_levels):
        raise ValueError("env.official_levels() is required by the full-standard contract")
    try:
        official = official_levels()
    except Exception as exc:
        raise ValueError(
            f"env.official_levels() failed: {type(exc).__name__}: {exc}"
        ) from exc
    if (
        not isinstance(official, Sequence)
        or isinstance(official, (str, bytes))
        or not official
    ):
        raise ValueError("env.official_levels() must return a non-empty sequence")
    official_level_count = len(official)
    difficulties = tuple(entry.difficulty for entry in curriculum)
    expected_difficulties = tuple(range(1, official_level_count + 1))
    expected_contexts = tuple(range(official_level_count))
    if difficulties != expected_difficulties:
        raise ValueError(
            "full-standard curriculum must contain exactly one tier per official level "
            "with difficulties 1..N"
        )
    if tuple(entry.context_index for entry in curriculum) != expected_contexts:
        raise ValueError(
            "full-standard curriculum contexts must be exactly 0..N-1 in order"
        )
    declared = getattr(generate, "DIFFICULTIES", None)
    if not isinstance(declared, Sequence) or isinstance(declared, (str, bytes)):
        raise ValueError("generate.DIFFICULTIES must declare the full ordered curriculum")
    if any(
        not isinstance(value, Integral) or isinstance(value, bool)
        for value in declared
    ) or tuple(int(value) for value in declared) != expected_difficulties:
        raise ValueError(
            "generate.DIFFICULTIES must equal 1..N for the official level count"
        )

    raw_evidence = raw.get("evidence")
    if not isinstance(raw_evidence, Mapping):
        raise ValueError("full-standard evidence must be an object")
    missing_evidence = [key for key in FULL_STANDARD_EVIDENCE if key not in raw_evidence]
    if missing_evidence:
        raise ValueError(f"full-standard evidence is missing: {', '.join(missing_evidence)}")
    evidence = []
    for key, value in sorted(raw_evidence.items()):
        if not isinstance(key, str) or not isinstance(value, str) or not value.strip():
            raise ValueError("full-standard evidence names and values must be non-empty strings")
        evidence.append((key, value))
    raw_caveats = raw.get("caveats", ())
    if (
        not isinstance(raw_caveats, Sequence)
        or isinstance(raw_caveats, (str, bytes))
        or any(not isinstance(value, str) or not value.strip() for value in raw_caveats)
    ):
        raise ValueError("full-standard caveats must be a sequence of non-empty strings")
    validator = getattr(generate, "validate_full_standard", None)
    if not callable(validator):
        raise ValueError("generate.validate_full_standard(spec, curriculum_entry) is required")
    parameters = inspect.signature(generate.generate).parameters
    if "split" not in parameters:
        raise ValueError("full-standard generate(...) must expose the explicit split keyword")
    whole_game = getattr(generate, "generate_game", None)
    if not callable(whole_game):
        raise ValueError("generate.generate_game(...) is required for whole-game generation")
    whole_parameters = inspect.signature(whole_game).parameters
    if "split" not in whole_parameters or "difficulties" not in whole_parameters:
        raise ValueError(
            "generate_game(...) must expose split and difficulties keywords"
        )
    if not callable(getattr(generate, "build_game", None)):
        raise ValueError("generate.build_game(specs) is required for whole-game construction")
    return FullStandardContract(
        source.source_id,
        status,
        versions["mechanics_inventory_version"],
        versions["quality_profile_version"],
        official_level_count,
        tuple(curriculum),
        tuple(evidence),
        tuple(raw_caveats),
    )


def curriculum_for(
    modules: GameModules,
    requested: Sequence[int] | None = None,
    *,
    require_full_standard: bool = False,
    smoke_search_work: int = 2_000_000,
) -> tuple[CurriculumEntry, ...]:
    """Resolve either one complete family curriculum or an explicit smoke sequence."""
    if require_full_standard:
        if requested is not None:
            raise ValueError("full-standard collection must use the complete family curriculum")
        contract = modules.full_standard
        if contract is None or not contract.ready:
            raise ValueError(f"{modules.source.slug} has no accepted full-standard contract")
        return contract.curriculum
    if smoke_search_work < 1:
        raise ValueError("smoke_search_work must be positive")
    values = (1, 2, 3) if requested is None else tuple(requested)
    if not values:
        raise ValueError("smoke curriculum cannot be empty")
    declared = tuple(getattr(modules.generate, "DIFFICULTIES", (1, 2, 3)))
    normalized = []
    for index, difficulty in enumerate(values):
        if not isinstance(difficulty, Integral) or isinstance(difficulty, bool):
            raise ValueError("smoke curriculum difficulties must be integers")
        difficulty = int(difficulty)
        if difficulty not in declared:
            raise ValueError(
                f"difficulty {difficulty} is not declared by {modules.source.slug}: {declared}"
            )
        normalized.append(CurriculumEntry(difficulty, index, smoke_search_work))
    if len({entry.difficulty for entry in normalized}) != len(normalized):
        raise ValueError("smoke curriculum difficulties must be distinct")
    return tuple(normalized)


def _full_standard_coverage(contract: FullStandardContract) -> dict[str, Any]:
    """Contract-derived coverage summary for full experiment metadata."""
    return {
        "admission": contract.status,
        "official_level_count": contract.official_level_count,
        "curriculum": [entry.to_dict() for entry in contract.curriculum],
        "mechanics_inventory_version": contract.mechanics_inventory_version,
        "quality_profile_version": contract.quality_profile_version,
        "evidence": dict(contract.evidence),
        "caveats": list(contract.caveats),
    }


_REQUIRED = {
    "env": ("Env",),
    "generate": ("generate", "build_level"),
    "plan": ("search",),
}


def _solver_adapter(search: Any) -> str:
    """Classify supported bounds explicitly from the search call signature."""
    params = inspect.signature(search).parameters
    if "limit" in params and "node_limit" in params:
        return "action_limit+node_limit"
    if "limit" in params and "max_nodes" in params:
        return "action_limit+max_nodes"
    if "limit" in params and "budget" in params:
        return "work_limit+action_budget"
    if "limit" in params:
        return "work_limit"
    raise TypeError(
        "plan.search has no supported bounded signature; expected limit and "
        "optionally node_limit, max_nodes, or budget"
    )


def inspect_source(value: str | Source) -> Availability:
    """Import and inspect one package, returning all missing pieces and errors."""
    source = source_for(value)
    missing: list[str] = []
    errors: list[str] = []
    loaded: dict[str, ModuleType] = {}
    for part, attributes in _REQUIRED.items():
        module_name = f"{source.package}.{part}"
        try:
            found = importlib.util.find_spec(module_name)
        except (ImportError, AttributeError, ValueError) as exc:
            found = None
            errors.append(f"{module_name}: discovery failed: {type(exc).__name__}: {exc}")
        if found is None:
            missing.append(module_name)
            continue
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # package import failures are preflight evidence
            errors.append(f"{module_name}: import failed: {type(exc).__name__}: {exc}")
            continue
        loaded[part] = module
        for attribute in attributes:
            if not hasattr(module, attribute):
                missing.append(f"{module_name}.{attribute}")
    adapter = None
    full_standard_ready = False
    full_standard_errors: list[str] = []
    if "plan" in loaded and hasattr(loaded["plan"], "search"):
        try:
            adapter = _solver_adapter(loaded["plan"].search)
        except (TypeError, ValueError) as exc:
            errors.append(f"{source.package}.plan.search: {exc}")
    if "env" in loaded and hasattr(loaded["env"], "Env"):
        for method in ("reset", "perform", "render"):
            if not hasattr(loaded["env"].Env, method):
                missing.append(f"{source.package}.env.Env.{method}")
    if "generate" in loaded and "env" in loaded:
        try:
            contract = _full_standard_contract(source, loaded["env"], loaded["generate"])
            full_standard_ready = contract.ready
            if not contract.ready:
                full_standard_errors.append(
                    f"full-standard contract status is {contract.status!r}, not 'ready'"
                )
        except (TypeError, ValueError) as exc:
            full_standard_errors.append(str(exc))
    ready = not missing and not errors and adapter is not None
    return Availability(
        source.source_id,
        source.slug,
        ready,
        tuple(missing),
        tuple(errors),
        adapter,
        full_standard_ready,
        tuple(full_standard_errors),
    )


def inspect_availability(values: Iterable[str | Source] = SOURCES) -> tuple[Availability, ...]:
    return tuple(inspect_source(value) for value in values)


def resolve_collection_sources(games: Sequence[str | Source] | None = None) -> tuple[Source, ...]:
    """Resolve a collection request and enforce the m0r0 holdout boundary."""
    sources = TRAIN_SOURCES if games is None else tuple(source_for(game) for game in games)
    if not sources:
        raise ValueError("at least one collection source is required")
    if len({source.source_id for source in sources}) != len(sources):
        raise ValueError("collection sources must be distinct")
    held_out = [source.source_id for source in sources if source.held_out]
    if held_out:
        raise ValueError(
            f"held-out source {HELD_OUT_SOURCE_ID} cannot be used for generation, collection, or training"
        )
    return tuple(sources)


def preflight(
    games: Sequence[str | Source] | None = None,
    *,
    require_full_standard: bool = False,
) -> tuple[GameModules, ...]:
    """Require every requested package before a caller creates output files.

    With no explicit subset, this checks all 24 training sources.  An explicit
    subset is a smoke scope and never relaxes readiness within that subset.
    """
    sources = resolve_collection_sources(games)
    availability = inspect_availability(sources)
    unavailable = tuple(
        item for item in availability
        if not item.ready or (require_full_standard and not item.full_standard_ready)
    )
    if unavailable:
        details = []
        for item in unavailable:
            reasons = list(item.missing) + list(item.errors)
            if require_full_standard:
                reasons.extend(item.full_standard_errors)
            details.append(f"{item.slug}: " + ("; ".join(reasons) or "not ready"))
        scope = "default 24-game" if games is None else "explicit subset"
        standard = " full-standard" if require_full_standard else ""
        raise PreflightError(
            f"{scope}{standard} preflight rejected {len(unavailable)} unavailable package(s): "
            + " | ".join(details),
            unavailable,
        )
    modules = []
    for source, status in zip(sources, availability):
        try:
            provenance = _production_provenance(source.source_id)
        except Exception as exc:
            failed = Availability(
                source.source_id,
                source.slug,
                False,
                errors=(f"code provenance failed: {type(exc).__name__}: {exc}",),
                solver_adapter=status.solver_adapter,
            )
            raise PreflightError(
                f"production preflight rejected unusable code provenance for {source.slug}: "
                f"{type(exc).__name__}: {exc}",
                (failed,),
            ) from exc
        full_standard = None
        try:
            full_standard = _full_standard_contract(
                source,
                importlib.import_module(f"{source.package}.env"),
                importlib.import_module(f"{source.package}.generate"),
            )
        except (TypeError, ValueError):
            if require_full_standard:
                raise AssertionError("full-standard availability and contract loading diverged")
        modules.append(GameModules(
            source=source,
            env=importlib.import_module(f"{source.package}.env"),
            generate=importlib.import_module(f"{source.package}.generate"),
            plan=importlib.import_module(f"{source.package}.plan"),
            solver_adapter=status.solver_adapter or "",
            provenance=copy.deepcopy(provenance),
            full_standard=full_standard,
        ))
    return tuple(modules)


@dataclass(frozen=True)
class Action:
    id: int
    x: int | None = None
    y: int | None = None

    @classmethod
    def parse(cls, value: "Action | Sequence[Any]") -> "Action":
        if isinstance(value, cls):
            action_id, x, y = value.id, value.x, value.y
        else:
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
                raise IllegalActionError("an action must be a triple (action_id, x, y)")
            action_id, x, y = value
        if not isinstance(action_id, Integral) or isinstance(action_id, bool):
            raise IllegalActionError("action_id must be an integer")
        for name, coordinate in (("x", x), ("y", y)):
            if coordinate is not None and (
                not isinstance(coordinate, Integral) or isinstance(coordinate, bool)
            ):
                raise IllegalActionError(f"{name} must be an integer or null")
        return cls(int(action_id), None if x is None else int(x), None if y is None else int(y))

    def as_tuple(self) -> tuple[int, int | None, int | None]:
        return self.id, self.x, self.y


@dataclass(frozen=True)
class Progress:
    state: str
    level_index: int
    levels_completed: int
    terminal: bool
    won: bool


def _state_name(state: Any) -> str:
    value = getattr(state, "value", state)
    return str(value).upper()


def _action_id(value: Any) -> int:
    raw = getattr(value, "value", value)
    if not isinstance(raw, Integral) or isinstance(raw, bool):
        raise IllegalActionError(f"invalid available action value {value!r}")
    action_id = int(raw)
    if not 0 <= action_id < ACTION_COUNT:
        raise IllegalActionError(f"available action id {action_id} is outside 0..7")
    return action_id


class MultiGameEnv:
    """Validated public facade over a generated sequential real-engine game."""

    render_shape = FRAME_SHAPE

    def __init__(self, modules: GameModules, levels: Sequence[Any]):
        if not levels:
            raise ValueError("a whole game needs at least one generated level")
        self.modules = modules
        self.source = source_for(modules.source)
        self._env = modules.env.Env(list(levels))
        self.level_count = len(levels)

    @classmethod
    def from_specs(
        cls,
        modules: GameModules,
        specs: Sequence[Mapping[str, Any]],
        *,
        require_full_standard: bool = False,
    ) -> "MultiGameEnv":
        if require_full_standard:
            if modules.full_standard is None or not modules.full_standard.ready:
                raise ValueError("full-standard game construction requires an accepted contract")
            levels = modules.generate.build_game(specs)
            if len(levels) != modules.full_standard.official_level_count:
                raise ValueError("full-standard builder returned the wrong official level count")
        else:
            levels = [modules.generate.build_level(spec) for spec in specs]
        return cls(modules, levels)

    @property
    def legal_action_ids(self) -> tuple[int, ...]:
        if self.progress.terminal:
            return ()
        values = getattr(self._env, "available_actions")
        if callable(values):
            values = values()
        action_ids = {_action_id(value) for value in values}
        action_ids.discard(0)
        return tuple(sorted(action_ids))

    @property
    def available_actions(self) -> tuple[int, ...]:
        return self.legal_action_ids

    @property
    def state(self) -> str:
        return _state_name(self._env.state)

    @property
    def progress(self) -> Progress:
        state = self.state
        return Progress(
            state=state,
            level_index=int(self._env.level_index),
            levels_completed=int(self._env.levels_completed),
            terminal=state in {"WIN", "GAME_OVER"},
            won=state == "WIN",
        )

    def render(self) -> np.ndarray:
        frame = np.asarray(self._env.render())
        if frame.shape != FRAME_SHAPE:
            raise FrameError(f"expected a 64x64 frame, got {frame.shape}")
        if not np.issubdtype(frame.dtype, np.integer):
            raise FrameError(f"expected integer palette values, got {frame.dtype}")
        if frame.size and (int(frame.min()) < 0 or int(frame.max()) > 15):
            raise FrameError("palette values must be integers in 0..15")
        return frame.astype(np.uint8, copy=True)

    def reset(self) -> np.ndarray:
        self._env.reset()
        return self.render()

    def validate_action(
        self, action: Action | Sequence[Any], *, allow_reset: bool = False
    ) -> Action:
        parsed = Action.parse(action)
        if parsed.id == 0:
            if not allow_reset:
                raise IllegalActionError("RESET is episode control and is not legal in a teacher plan")
            if parsed.x is not None or parsed.y is not None:
                raise IllegalActionError("RESET coordinates must be null")
            return parsed
        if parsed.id not in self.legal_action_ids:
            raise IllegalActionError(
                f"action {parsed.id} is unavailable; legal ids are {self.legal_action_ids}"
            )
        if parsed.id == CLICK_ACTION:
            if parsed.x is None or parsed.y is None:
                raise IllegalActionError("ACTION6 requires display coordinates")
            if not (0 <= parsed.x < FRAME_SHAPE[1] and 0 <= parsed.y < FRAME_SHAPE[0]):
                raise IllegalActionError("ACTION6 coordinates must be in 0..63")
        elif parsed.x is not None or parsed.y is not None:
            raise IllegalActionError("non-click action coordinates must be null")
        return parsed

    def perform(
        self, action: Action | Sequence[Any], *, allow_reset: bool = False
    ) -> Any:
        parsed = self.validate_action(action, allow_reset=allow_reset)
        return self._env.perform(*parsed.as_tuple())

    @property
    def raw_env(self) -> Any:
        """Teacher/planner access. Never serialize this as a public model input."""
        return self._env


@dataclass(frozen=True)
class SearchLimits:
    max_actions_per_level: int = 512
    max_search_work: int = 2_000_000

    def __post_init__(self) -> None:
        if self.max_actions_per_level < 1 or self.max_search_work < 1:
            raise ValueError("search bounds must be positive")


@dataclass(frozen=True)
class RolloutOptions:
    """Bounded teacher/perturbation rollout controls; defaults preserve v1 behavior.

    ``random_action_probability`` is the per-step perturbation probability.
    ``perturbation`` selects what executes at a perturbation point: a uniform
    legal ``random`` action or the ``learner`` policy's action (supplied to the
    collector as a ``perturber``).  ``recovery_seconds`` caps each live
    family search and ``game_seconds`` caps one whole rollout; ``None`` leaves
    the search unbounded in wall-clock time (work bounds still apply).
    """

    random_action_probability: float = 0.0
    max_game_steps: int | None = None
    recovery_seconds: float | None = None
    game_seconds: float | None = None
    perturbation: str = "random"
    learner_checkpoint: str | None = None
    click_region_probe_limit: int = CLICK_REGION_PROBE_LIMIT

    def __post_init__(self) -> None:
        if not 0.0 <= self.random_action_probability <= 1.0:
            raise ValueError("random_action_probability must be in [0,1]")
        if (
            isinstance(self.click_region_probe_limit, bool)
            or not isinstance(self.click_region_probe_limit, Integral)
            or self.click_region_probe_limit < 0
        ):
            raise ValueError("click_region_probe_limit must be a non-negative integer (0 disables)")
        object.__setattr__(self, "click_region_probe_limit", int(self.click_region_probe_limit))
        if self.max_game_steps is not None and self.max_game_steps < 1:
            raise ValueError("max_game_steps must be positive")
        for name in ("recovery_seconds", "game_seconds"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0.0:
                raise ValueError(f"{name} must be a positive number of seconds or None")
            object.__setattr__(self, name, float(value))
        if self.perturbation not in PERTURBATION_MODES:
            raise ValueError(f"perturbation must be one of {PERTURBATION_MODES}")
        if self.learner_checkpoint is not None:
            object.__setattr__(self, "learner_checkpoint", str(self.learner_checkpoint))

    @property
    def mode(self) -> str:
        if self.random_action_probability == 0.0:
            return "teacher_only"
        return f"mixed_teacher_{self.perturbation}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "random_action_probability": self.random_action_probability,
            "max_game_steps": self.max_game_steps,
            "recovery_seconds": self.recovery_seconds,
            "game_seconds": self.game_seconds,
            "perturbation": self.perturbation,
            "learner_checkpoint": self.learner_checkpoint,
            "click_region_probe_limit": self.click_region_probe_limit,
        }


@dataclass(frozen=True)
class PlanResult:
    actions: tuple[Action, ...] | None
    truncated: bool
    unsupported: bool
    exact: bool | None
    work: int | None
    reason: str
    adapter: str
    timed_out: bool = False
    elapsed_seconds: float | None = None


def solve_live_level(
    game: MultiGameEnv, limits: SearchLimits, *, deadline_seconds: float | None = None,
) -> PlanResult:
    """Solve the current live level using its package's explicit limit mapping.

    ``deadline_seconds`` is a wall-clock cap on the family search itself.  A
    capped search returns a plan-less result with ``timed_out=True`` and
    ``reason == RECOVERY_TIMEOUT_REASON`` instead of raising.
    """
    search = game.modules.plan.search
    adapter = game.modules.solver_adapter
    if adapter == "action_limit+node_limit":
        kwargs = {"limit": limits.max_actions_per_level, "node_limit": limits.max_search_work}
    elif adapter == "action_limit+max_nodes":
        kwargs = {"limit": limits.max_actions_per_level, "max_nodes": limits.max_search_work}
    elif adapter == "work_limit+action_budget":
        # WA30's default budget is the live level's remaining steps.  Never
        # replace it with a larger external cap.  If the wrapper exposes the
        # remaining value, a smaller external cap can safely tighten it.
        remaining = getattr(game.raw_env, "steps_left", None)
        remaining = remaining() if callable(remaining) else remaining
        kwargs = {"limit": limits.max_search_work}
        if isinstance(remaining, Integral) and not isinstance(remaining, bool):
            kwargs["budget"] = min(int(remaining), limits.max_actions_per_level)
    elif adapter == "work_limit":
        kwargs = {"limit": limits.max_search_work}
    else:
        raise SolverError(f"unsupported solver adapter {adapter!r} for {game.source.slug}")
    started = time.monotonic()
    try:
        result = _call_with_deadline(lambda: search(game.raw_env, **kwargs), deadline_seconds)
    except RecoveryTimeout as exc:
        return PlanResult(
            None, False, False, None, None, RECOVERY_TIMEOUT_REASON, adapter,
            timed_out=True, elapsed_seconds=time.monotonic() - started,
        )
    except Exception as exc:
        raise SolverError(f"{game.source.slug} live search failed: {type(exc).__name__}: {exc}") from exc
    elapsed = time.monotonic() - started
    raw_actions = getattr(result, "actions", result if isinstance(result, (list, tuple)) else None)
    truncated = bool(getattr(result, "truncated", False))
    unsupported = bool(getattr(result, "unsupported", False))
    raw_exact = getattr(result, "exact", None)
    exact = None if raw_exact is None else bool(raw_exact)
    work = next((int(getattr(result, name)) for name in ("expanded", "explored", "nodes")
                 if getattr(result, name, None) is not None), None)
    reason = str(getattr(result, "reason", ""))
    if raw_actions is None:
        return PlanResult(
            None, truncated, unsupported, exact, work,
            reason or "search returned no actions", adapter, elapsed_seconds=elapsed,
        )
    try:
        actions = tuple(Action.parse(action) for action in raw_actions)
    except (IllegalActionError, TypeError, ValueError) as exc:
        raise SolverError(f"{game.source.slug} planner returned malformed actions: {exc}") from exc
    if not actions:
        raise SolverError("planner returned an empty plan, which cannot trigger engine completion")
    if len(actions) > limits.max_actions_per_level:
        raise SolverError(
            f"planner returned {len(actions)} actions, above cap {limits.max_actions_per_level}"
        )
    return PlanResult(
        actions, truncated, unsupported, exact, work, reason or "solved", adapter,
        elapsed_seconds=elapsed,
    )


def _certified_spec_plan(
    game: MultiGameEnv,
    spec: Mapping[str, Any],
    *,
    expected_level: int,
    max_actions: int,
) -> PlanResult:
    """Parse and preflight one validated full-standard route in live context.

    Family validation establishes the certificate's mechanics and quality
    claims. This boundary independently checks the stored triples and action
    cap, then replays the raw route on a clone of the current sequential engine
    state. A bad certificate is a collection failure; it never falls back to a
    newly searched route.
    """
    raw_actions = spec.get("solution")
    if (
        not isinstance(raw_actions, Sequence)
        or isinstance(raw_actions, (str, bytes))
        or not raw_actions
    ):
        raise SolverError(
            f"level {expected_level} certified solution must be a non-empty sequence"
        )
    declared_length = spec.get("solution_length")
    if (
        not isinstance(declared_length, Integral)
        or isinstance(declared_length, bool)
        or int(declared_length) != len(raw_actions)
    ):
        raise SolverError(
            f"level {expected_level} certified solution_length does not match its route"
        )
    try:
        actions = tuple(Action.parse(action) for action in raw_actions)
    except (IllegalActionError, TypeError, ValueError) as exc:
        raise SolverError(
            f"level {expected_level} certified solution has malformed action triples: {exc}"
        ) from exc
    if len(actions) > max_actions:
        raise SolverError(
            f"level {expected_level} certified solution has {len(actions)} actions, "
            f"above native collection cap {max_actions}"
        )

    clone = getattr(game.raw_env, "clone", None)
    if not callable(clone):
        raise SolverError(
            f"level {expected_level} cannot preflight its certified solution: "
            "the engine does not expose clone()"
        )
    try:
        raw_clone = clone()
    except Exception as exc:
        raise SolverError(
            f"level {expected_level} certified solution clone failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    probe = _probe_from_raw(game, raw_clone)
    if probe.progress.levels_completed != expected_level:
        raise SolverError(
            f"level {expected_level} certified solution preflight started at "
            f"{probe.progress.levels_completed} completed levels"
        )
    for offset, action in enumerate(actions):
        try:
            validated = probe.validate_action(action)
            probe.perform(validated)
        except Exception as exc:
            raise SolverError(
                f"level {expected_level} certified solution action {offset} failed native "
                f"preflight: {type(exc).__name__}: {exc}"
            ) from exc
        after = probe.progress
        if after.state == "GAME_OVER":
            raise SolverError(
                f"level {expected_level} certified solution reached GAME_OVER at action {offset}"
            )
        if after.levels_completed > expected_level:
            if after.levels_completed != expected_level + 1:
                raise SolverError(
                    f"level {expected_level} certified solution advanced more than one level"
                )
            if offset != len(actions) - 1:
                raise SolverError(
                    f"level {expected_level} certified solution completes before its final action"
                )
    if probe.progress.levels_completed != expected_level + 1:
        raise SolverError(
            f"level {expected_level} certified solution ends without engine completion"
        )
    return PlanResult(
        actions, False, False, True, None,
        "validated spec solution replayed in live sequential context",
        CERTIFIED_ROUTE_SOURCE,
    )


def deterministic_level_seed(
    master_seed: int, source: str | Source, game_index: int, level_index: int, attempt: int
) -> int:
    """Stable 63-bit seed independent of Python's randomized hash."""
    resolved = source_for(source)
    material = f"{int(master_seed)}:{resolved.source_id}:{game_index}:{level_index}:{attempt}".encode()
    return int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def deterministic_rollout_seed(master_seed: int, source: str | Source, game_index: int) -> int:
    resolved = source_for(source)
    material = f"rollout:{int(master_seed)}:{resolved.source_id}:{int(game_index)}".encode()
    return int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def _generate_kwargs(
    function: Any, attempts: int, work_limit: int, split: str | None,
) -> dict[str, int | str]:
    params = inspect.signature(function).parameters
    kwargs: dict[str, int | str] = {}
    if "attempts" in params:
        kwargs["attempts"] = attempts
    if "node_limit" in params:
        kwargs["node_limit"] = work_limit
    elif "limit" in params:
        kwargs["limit"] = work_limit
    if split is not None:
        if "split" not in params:
            raise TypeError("full-standard generator does not expose the split keyword")
        kwargs["split"] = split
    return kwargs


def _generator_declarations(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Retain package-authored coverage and omission fields without one fixed schema."""
    markers = (
        "coverage", "limitation", "omitted", "unsupported", "supported_mechanic",
        "generator_note", "mechanic",
    )
    return {
        str(key): value
        for key, value in spec.items()
        if any(marker in str(key).lower() for marker in markers)
    }


def generate_game(
    modules: GameModules,
    *,
    master_seed: int,
    game_index: int,
    difficulties: Sequence[int],
    curriculum: Sequence[CurriculumEntry] | None = None,
    split: str | None = None,
    require_full_standard: bool = False,
    outer_attempts: int = 8,
    generator_attempts: int = 50,
    search_work_limit: int = 2_000_000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate one ordered game of specs or fail without constructing an Env.

    Full-standard calls require the complete official curriculum. Explicit
    reduced sequences remain smoke data and are expanded one level at a time,
    so native context indices are never silently shifted by a family builder.
    """
    source = source_for(modules.source)
    if source.held_out:
        raise ValueError(f"held-out source {source.source_id} cannot be used for generation")
    if not difficulties:
        raise ValueError("a game needs at least one difficulty")
    if outer_attempts < 1 or generator_attempts < 1 or search_work_limit < 1:
        raise ValueError("generation bounds must be positive")
    if split is not None and split not in ("train", "validation", "test"):
        raise ValueError("generation split must be train, validation, or test")
    if require_full_standard and split is None:
        raise ValueError("full-standard generation requires an explicit split")
    if curriculum is None:
        entries = tuple(
            CurriculumEntry(int(difficulty), index, search_work_limit)
            for index, difficulty in enumerate(difficulties)
        )
    else:
        entries = tuple(curriculum)
        if not all(isinstance(entry, CurriculumEntry) for entry in entries):
            raise ValueError("curriculum entries must be CurriculumEntry values")
        if tuple(entry.difficulty for entry in entries) != tuple(difficulties):
            raise ValueError("curriculum entries do not match requested difficulties")
    if require_full_standard:
        expected = curriculum_for(modules, require_full_standard=True)
        if entries != expected:
            raise ValueError("full-standard generation requires the complete declared curriculum")
    if any(entry.search_work > search_work_limit for entry in entries):
        raise ValueError("configured search-work ceiling is below the requested curriculum")

    specs: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for level_index, entry in enumerate(entries):
        difficulty = entry.difficulty
        kwargs = _generate_kwargs(
            modules.generate.generate, generator_attempts, entry.search_work, split,
        )
        failures = []
        accepted = None
        accepted_seed = None
        accepted_attempt = None
        for attempt in range(outer_attempts):
            seed = deterministic_level_seed(
                master_seed, modules.source, game_index, level_index, attempt
            )
            try:
                candidate = modules.generate.generate(seed, int(difficulty), **kwargs)
            except Exception as exc:
                failures.append(f"seed {seed}: {type(exc).__name__}: {exc}")
                continue
            if candidate is None:
                failures.append(f"seed {seed}: generator returned None")
                continue
            if not isinstance(candidate, Mapping):
                failures.append(f"seed {seed}: generator returned {type(candidate).__name__}, not a mapping")
                continue
            if require_full_standard:
                if candidate.get("split") != split:
                    failures.append(
                        f"seed {seed}: generated spec split {candidate.get('split')!r} "
                        f"does not match requested {split!r}"
                    )
                    continue
                try:
                    validation = modules.generate.validate_full_standard(
                        candidate, entry.to_dict(),
                    )
                except Exception as exc:
                    failures.append(
                        f"seed {seed}: full-standard validator failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                if (
                    not isinstance(validation, Sequence)
                    or isinstance(validation, (str, bytes))
                ):
                    failures.append(
                        f"seed {seed}: full-standard validator did not return a sequence"
                    )
                    continue
                validation_errors = [str(value) for value in validation if str(value)]
                if validation_errors:
                    failures.append(
                        f"seed {seed}: full-standard validation: "
                        + "; ".join(validation_errors)
                    )
                    continue
            accepted = dict(candidate)
            accepted_seed = seed
            accepted_attempt = attempt
            break
        if accepted is None:
            joined = " | ".join(failures)
            raise GenerationError(
                f"{modules.source.slug} level {level_index} difficulty {difficulty} failed after "
                f"{outer_attempts} bounded seed attempts: {joined}",
                specs,
                metadata,
            )
        specs.append(accepted)
        metadata.append({
            "level_index": level_index,
            "difficulty": int(difficulty),
            "context_index": entry.context_index,
            "search_work": entry.search_work,
            "generation_split": split,
            "full_standard_validated": bool(require_full_standard),
            "seed": int(accepted_seed),
            "outer_attempt": int(accepted_attempt),
            "generator_version": accepted.get("generator_version"),
            "engine_verified_in_isolation": bool(accepted.get("engine_verified", False)),
            "declared_coverage": _generator_declarations(accepted),
        })
    return specs, metadata


def generate_levels(*args, **kwargs):
    """Backward-compatible name for :func:`generate_game`."""
    return generate_game(*args, **kwargs)


def _probe_from_raw(game: MultiGameEnv, raw_env: Any) -> MultiGameEnv:
    """Wrap a cloned raw engine in the public facade without re-running reset."""
    probe = object.__new__(MultiGameEnv)
    probe.modules = game.modules
    probe.source = game.source
    probe._env = raw_env
    probe.level_count = game.level_count
    return probe


def _clone_raw_env(raw_env: Any) -> Any:
    clone = getattr(raw_env, "clone", None)
    if callable(clone):
        return clone()
    return copy.deepcopy(raw_env)


def _successor_signature(probe: MultiGameEnv) -> tuple[Any, ...]:
    """Everything the public successor state exposes after one action."""
    progress = probe.progress
    return (
        probe.render().tobytes(),
        progress.state,
        progress.level_index,
        progress.levels_completed,
        progress.terminal,
        progress.won,
    )


def _sprite_display_pixels(raw_env: Any, x: int, y: int) -> list[tuple[int, int]]:
    """Display pixels covered by the engine sprite under raw click ``(x, y)``.

    Uses the arcengine camera/level hit-test when the family exposes it
    (``raw_env.game.camera`` / ``raw_env.game.current_level``); otherwise, or on
    any engine error, returns nothing.  These are only *candidates*: every
    pixel is still verified by clone+step before it enters a region.
    """
    engine = getattr(raw_env, "game", None)
    camera = getattr(engine, "camera", None)
    level = getattr(engine, "current_level", None)
    if camera is None or level is None:
        return []
    try:
        grid = camera.display_to_grid(x, y)
        if grid is None:
            return []
        sprite = level.get_sprite_at(*grid)
        if sprite is None:
            sprite = level.get_sprite_at(*grid, ignore_collidable=True)
        if sprite is None:
            return []
        left, top = int(sprite.x), int(sprite.y)
        right, bottom = left + int(sprite.width), top + int(sprite.height)
        # display_to_grid is separable in x and y, so scan each axis once
        # (holding the other at the known on-camera target coordinate).
        xs = []
        for px in range(FRAME_SHAPE[1]):
            cell = camera.display_to_grid(px, y)
            if cell is not None and left <= cell[0] < right:
                xs.append(px)
        ys = []
        for py in range(FRAME_SHAPE[0]):
            cell = camera.display_to_grid(x, py)
            if cell is not None and top <= cell[1] < bottom:
                ys.append(py)
    except Exception:
        return []
    return [(px, py) for py in ys for px in xs]


def click_region_candidates(
    raw_env: Any, x: int, y: int, *, limit: int = CLICK_REGION_PROBE_LIMIT,
) -> list[tuple[int, int]]:
    """Ordered, de-duplicated raw display pixels to probe for click ``(x, y)``.

    The target pixel comes first, then its 5x5 neighbourhood, then the pixels
    of the sprite under it, each group ordered by Chebyshev distance from the
    target; the list is capped at ``limit`` entries.
    """
    def distance(pixel: tuple[int, int]) -> tuple[int, int, int]:
        return (max(abs(pixel[0] - x), abs(pixel[1] - y)), abs(pixel[1] - y), abs(pixel[0] - x))

    radius = CLICK_REGION_NEIGHBOURHOOD
    neighbourhood = [
        (px, py)
        for py in range(max(0, y - radius), min(FRAME_SHAPE[0], y + radius + 1))
        for px in range(max(0, x - radius), min(FRAME_SHAPE[1], x + radius + 1))
    ]
    ordered: list[tuple[int, int]] = [(x, y)]
    seen = {(x, y)}
    for group in (neighbourhood, _sprite_display_pixels(raw_env, x, y)):
        for pixel in sorted(group, key=distance):
            if pixel not in seen:
                seen.add(pixel)
                ordered.append(pixel)
    return ordered[:max(1, int(limit))]


@dataclass(frozen=True)
class ClickRegion:
    """Engine-verified equivalent-click set for one teacher click target."""

    mask: np.ndarray  # bool [64,64] in the coordinate space of the probed engine
    candidates: int
    seconds: float
    sprite_candidates: int

    @property
    def size(self) -> int:
        return int(self.mask.sum())


def equivalent_click_region(
    game: MultiGameEnv,
    target: Action,
    *,
    limit: int = CLICK_REGION_PROBE_LIMIT,
) -> ClickRegion:
    """Pixels whose click on a clone of the current state matches ``target``'s.

    Two clicks are equivalent when their successors agree on the rendered
    frame, state name, level index, levels completed and terminal/won flags.
    The target pixel is always in the region; every other candidate is
    admitted only after a clone+step comparison.  The live ``game`` is never
    stepped.
    """
    target = Action.parse(target)
    if target.id != CLICK_ACTION or target.x is None or target.y is None:
        raise ValueError("equivalent_click_region requires a click target")
    target = game.validate_action(target)
    started = time.monotonic()
    mask = np.zeros(FRAME_SHAPE, dtype=np.bool_)
    mask[target.y, target.x] = True
    candidates = click_region_candidates(game.raw_env, target.x, target.y, limit=limit)
    sprite_candidates = len(_sprite_display_pixels(game.raw_env, target.x, target.y))
    reference_probe = _probe_from_raw(game, _clone_raw_env(game.raw_env))
    reference_probe.perform(target)
    reference = _successor_signature(reference_probe)
    for px, py in candidates:
        if mask[py, px]:
            continue
        probe = _probe_from_raw(game, _clone_raw_env(game.raw_env))
        try:
            probe.perform(Action(CLICK_ACTION, px, py))
            signature = _successor_signature(probe)
        except Exception:
            continue
        if signature == reference:
            mask[py, px] = True
    return ClickRegion(
        mask=mask,
        candidates=len(candidates),
        seconds=time.monotonic() - started,
        sprite_candidates=sprite_candidates,
    )


def legal_mask(game: MultiGameEnv, variant: WholeGameVariant | None = None) -> np.ndarray:
    mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
    for action_id in game.legal_action_ids:
        mask[action_id] = True
    return mask if variant is None else variant.public_legal_mask(mask)


@dataclass
class CollectedGame:
    record: dict[str, Any]
    public: dict[str, np.ndarray] | None
    teacher: dict[str, np.ndarray] | None
    specs: list[dict[str, Any]]


def _empty_columns() -> dict[str, list[Any]]:
    return {
        "frames": [], "legal_action_mask": [], "state": [], "level_index": [],
        "levels_completed": [], "terminal": [], "won": [], "action_id": [],
        "action_x": [], "action_y": [], "level_boundary": [],
        "teacher_action_id": [], "teacher_action_x": [], "teacher_action_y": [],
        "teacher_source": [], "teacher_plan_level": [], "teacher_plan_offset": [],
        "teacher_plan_length": [], "teacher_route_source": [],
        "teacher_click_region_mask": [], "teacher_click_region_size": [],
    }


def _append_public_state(
    columns: dict[str, list[Any]], game: MultiGameEnv, variant: WholeGameVariant | None = None
) -> None:
    variant = variant or WholeGameVariant.identity()
    progress = game.progress
    # Compute every value first: a render/legal-mask failure must not leave a
    # partially appended state column that violates the N+1/N contract.
    values = {
        "frames": variant.public_frame(game.render()),
        "legal_action_mask": legal_mask(game, variant),
        "state": progress.state,
        "level_index": progress.level_index,
        "levels_completed": progress.levels_completed,
        "terminal": progress.terminal,
        "won": progress.won,
    }
    for key, value in values.items():
        columns[key].append(value)


def _arrays(columns: dict[str, list[Any]]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    public = {
        "frames": np.asarray(columns["frames"], dtype=np.uint8).reshape(-1, *FRAME_SHAPE),
        "legal_action_mask": np.asarray(columns["legal_action_mask"], dtype=np.bool_).reshape(-1, ACTION_COUNT),
        "state": np.asarray(columns["state"], dtype="<U16"),
        "level_index": np.asarray(columns["level_index"], dtype=np.int16),
        "levels_completed": np.asarray(columns["levels_completed"], dtype=np.int16),
        "terminal": np.asarray(columns["terminal"], dtype=np.bool_),
        "won": np.asarray(columns["won"], dtype=np.bool_),
        "action_id": np.asarray(columns["action_id"], dtype=np.int8),
        "action_x": np.asarray(columns["action_x"], dtype=np.int16),
        "action_y": np.asarray(columns["action_y"], dtype=np.int16),
        "level_boundary": np.asarray(columns["level_boundary"], dtype=np.bool_),
    }
    teacher = {
        "target_action_id": np.asarray(columns["teacher_action_id"], dtype=np.int8),
        "target_action_x": np.asarray(columns["teacher_action_x"], dtype=np.int16),
        "target_action_y": np.asarray(columns["teacher_action_y"], dtype=np.int16),
        "source": np.asarray(columns["teacher_source"], dtype=np.int8),
        "plan_level": np.asarray(columns["teacher_plan_level"], dtype=np.int16),
        "plan_offset": np.asarray(columns["teacher_plan_offset"], dtype=np.int16),
        "plan_length": np.asarray(columns["teacher_plan_length"], dtype=np.int16),
        "route_source": np.asarray(columns["teacher_route_source"], dtype="<U32"),
        "click_region_mask": np.asarray(
            columns["teacher_click_region_mask"], dtype=np.uint8,
        ).reshape(-1, *FRAME_SHAPE),
        "click_region_size": np.asarray(columns["teacher_click_region_size"], dtype=np.int16),
    }
    transitions = public["action_id"].shape[0]
    if public["frames"].shape[0] != transitions + 1:
        raise AssertionError("public timeline must have one more state than action")
    if any(array.shape[0] != transitions for array in teacher.values()):
        raise AssertionError("teacher arrays must align one-to-one with executed actions")
    return public, teacher


_TRANSITION_COLUMN_KEYS = (
    "action_id", "action_x", "action_y", "teacher_action_id", "teacher_action_x",
    "teacher_action_y", "teacher_source", "teacher_plan_level", "teacher_plan_offset",
    "teacher_plan_length", "teacher_route_source", "teacher_click_region_mask",
    "teacher_click_region_size",
)


def _coordinate(value: int | None) -> int:
    return COORDINATE_NONE if value is None else int(value)


def _public_action(variant: WholeGameVariant, action: Action) -> Action:
    return Action(*variant.public_action(action.id, action.x, action.y))


def _append_transition(
    columns: dict[str, list[Any]],
    *,
    executed: Action,
    target: Action,
    source: int,
    level: int,
    plan_offset: int,
    plan_length: int,
    route_source: str,
    click_region: np.ndarray | None = None,
) -> None:
    if click_region is None:
        click_region = np.zeros(FRAME_SHAPE, dtype=np.bool_)
    elif target.id == CLICK_ACTION and target.x is not None and target.y is not None:
        if click_region.shape != FRAME_SHAPE or not click_region[target.y, target.x]:
            raise ValueError("a click region must be [64,64] and contain its exact teacher target")
    columns["action_id"].append(executed.id)
    columns["action_x"].append(_coordinate(executed.x))
    columns["action_y"].append(_coordinate(executed.y))
    columns["teacher_action_id"].append(target.id)
    columns["teacher_action_x"].append(_coordinate(target.x))
    columns["teacher_action_y"].append(_coordinate(target.y))
    columns["teacher_source"].append(int(source))
    columns["teacher_plan_level"].append(int(level))
    columns["teacher_plan_offset"].append(int(plan_offset))
    columns["teacher_plan_length"].append(int(plan_length))
    columns["teacher_route_source"].append(str(route_source))
    columns["teacher_click_region_mask"].append(click_region.astype(np.uint8, copy=False))
    columns["teacher_click_region_size"].append(int(click_region.sum()))


def _pop_transition(columns: dict[str, list[Any]]) -> None:
    for key in _TRANSITION_COLUMN_KEYS:
        columns[key].pop()


def _sample_random_action(
    game: MultiGameEnv, variant: WholeGameVariant, rng: np.random.Generator
) -> tuple[Action, Action]:
    """Uniform public legal ID/full click pixel and its inverse raw action."""
    public_ids = np.flatnonzero(legal_mask(game, variant))
    if not len(public_ids):
        raise IllegalActionError("live state has no legal public action IDs")
    public_id = int(rng.choice(public_ids))
    if public_id == CLICK_ACTION:
        public_x = int(rng.integers(0, FRAME_SHAPE[1]))
        public_y = int(rng.integers(0, FRAME_SHAPE[0]))
    else:
        public_x = public_y = None
    public = Action(public_id, public_x, public_y)
    raw = Action(*variant.raw_action(*public.as_tuple()))
    return public, game.validate_action(raw)


class Perturber:
    """Learner-state perturbation source driven once per executed transition.

    ``reset()`` is called at the start of every whole game.  ``step`` receives
    the public frame/legal mask of the state about to be acted on plus the
    previously executed public action (``None`` before the first step) and
    whether that action completed a level.  It must return a public
    :class:`Action`; the collector only executes it at perturbation points but
    calls ``step`` on every transition so a stateful policy keeps a coherent
    memory of the trajectory it is watching.
    """

    name = "perturber"

    def configure_variant(self, variant: WholeGameVariant) -> None:
        """Receive the game's adapter; ordinary public policies need no conversion."""
        pass

    def reset(self) -> None:  # pragma: no cover - protocol default
        pass

    def step(
        self,
        frame: np.ndarray,
        legal_action_mask: np.ndarray,
        previous_action: "Action | None",
        previous_level_boundary: bool,
    ) -> "Action":  # pragma: no cover - protocol default
        raise NotImplementedError


def collect_generated_game(
    modules: GameModules,
    *,
    master_seed: int,
    game_index: int,
    difficulties: Sequence[int],
    curriculum: Sequence[CurriculumEntry] | None = None,
    split: str | None = None,
    require_full_standard: bool = False,
    limits: SearchLimits = SearchLimits(),
    outer_generation_attempts: int = 8,
    generator_attempts: int = 50,
    variants: VariantOptions = VariantOptions(),
    rollout: RolloutOptions = RolloutOptions(),
    perturber: Perturber | None = None,
) -> CollectedGame:
    """Generate and play one bounded sequential game in one real ``Env``.

    Identity variants and zero random probability preserve the original exact
    teacher rollout.  Perturbations (uniform random or learner actions)
    invalidate the cached plan and force a live-state recovery search before
    another transition can be recorded.  Every live search is capped by
    ``rollout.recovery_seconds`` and the whole rollout by
    ``rollout.game_seconds``; a capped search ends the game with its usable
    partial trace retained and the timeout counted in the record.
    """
    if rollout.perturbation == "learner" and rollout.random_action_probability > 0.0 \
            and perturber is None:
        raise ValueError("learner perturbation requires a perturber (loaded learner policy)")
    source = source_for(modules.source)
    if source.held_out:
        raise ValueError(f"held-out source {source.source_id} cannot be collected")
    entries = (
        curriculum_for(
            modules,
            None if require_full_standard else difficulties,
            require_full_standard=require_full_standard,
            smoke_search_work=limits.max_search_work,
        )
        if curriculum is None else tuple(curriculum)
    )
    if not entries or not all(isinstance(entry, CurriculumEntry) for entry in entries):
        raise ValueError("collection curriculum must contain CurriculumEntry values")
    if tuple(entry.difficulty for entry in entries) != tuple(difficulties):
        raise ValueError("collection curriculum does not match requested difficulties")
    if require_full_standard:
        if entries != curriculum_for(modules, require_full_standard=True):
            raise ValueError("full-standard collection requires the complete family curriculum")
        if split not in ("train", "validation", "test"):
            raise ValueError("full-standard collection requires an explicit split")
    if any(entry.search_work > limits.max_search_work for entry in entries):
        raise ValueError("configured search-work ceiling is below the requested curriculum")

    max_game_steps = rollout.max_game_steps
    if max_game_steps is None:
        max_game_steps = max(1, len(difficulties) * limits.max_actions_per_level)
    rollout_mode = rollout.mode
    provenance = (
        copy.deepcopy(modules.provenance)
        if modules.provenance is not None
        else _unavailable_provenance(source)
    )
    record: dict[str, Any] = {
        "format": FORMAT,
        "source_id": source.source_id,
        "source": source.slug,
        "game_index": int(game_index),
        "master_seed": int(master_seed),
        "requested_difficulties": [int(value) for value in difficulties],
        "requested_curriculum": [entry.to_dict() for entry in entries],
        "generation_split": split,
        "full_standard_required": bool(require_full_standard),
        "full_standard_validated": bool(require_full_standard),
        "full_standard_contract_hash": (
            modules.full_standard.sha256
            if require_full_standard and modules.full_standard is not None
            else None
        ),
        "status": "generating",
        "rollout_mode": rollout_mode,
        "random_action_probability": rollout.random_action_probability,
        "perturbation": rollout.perturbation,
        "learner_checkpoint": rollout.learner_checkpoint,
        "max_game_steps": max_game_steps,
        "recovery_seconds": rollout.recovery_seconds,
        "game_seconds": rollout.game_seconds,
        "recovery_timeouts": 0,
        "search_timeouts": 0,
        "game_timeout": False,
        "timeout": None,
        "rollout_wall_clock_seconds": None,
        "rollout_seed": deterministic_rollout_seed(master_seed, source, game_index),
        "teacher_steps": 0,
        "random_steps": 0,
        "learner_steps": 0,
        "live_recovery_teacher_actions": 0,
        "learner_illegal_fallbacks": 0,
        "click_region_probe_limit": rollout.click_region_probe_limit,
        "click_region_stats": None,
        "route_source_counts": {},
        "random_metric_eligible_steps": 0,
        "random_transition_scoring_available": False,
        "whole_game_rule_variants": "requested" if variants.enabled else "absent_baseline",
        "variant_request": variants.to_dict(),
        "mechanics_coverage_claim": (
            "certificate-only accepted full-standard coverage; actual executed-route coverage "
            "is reported separately"
            if require_full_standard else
            "historical core/smoke adapter coverage only; not full-standard admission"
        ),
        "stored_solution_certificate_scope": (
            "private generated-spec evidence only; random and live-recovery transitions never "
            "inherit its mechanic-use claims"
            if require_full_standard else None
        ),
        "known_coverage": (
            _full_standard_coverage(modules.full_standard)
            if require_full_standard and modules.full_standard is not None else
            KNOWN_COVERAGE.get(source.slug, "not audited")
        ),
        "full_standard_coverage": (
            _full_standard_coverage(modules.full_standard)
            if require_full_standard and modules.full_standard is not None else None
        ),
        "historical_core_coverage_note": (
            KNOWN_COVERAGE.get(source.slug, "not audited")
            if require_full_standard else None
        ),
        "provenance": provenance,
        "provenance_location": "private record JSON only; absent from public rollout arrays",
        "errors": [],
    }
    try:
        specs, level_metadata = generate_game(
            modules,
            master_seed=master_seed,
            game_index=game_index,
            difficulties=difficulties,
            curriculum=entries,
            split=split,
            require_full_standard=require_full_standard,
            outer_attempts=outer_generation_attempts,
            generator_attempts=generator_attempts,
            search_work_limit=limits.max_search_work,
        )
    except GenerationError as exc:
        record["levels"] = exc.partial_metadata
        record.update(
            status="generation_failed",
            levels_generated=len(exc.partial_specs),
            levels_completed=0,
            steps=0,
        )
        record["errors"].append(str(exc))
        return CollectedGame(record, None, None, exc.partial_specs)

    record["levels"] = level_metadata
    record["levels_generated"] = len(specs)
    try:
        game = MultiGameEnv.from_specs(
            modules, specs, require_full_standard=require_full_standard,
        )
        game.reset()
    except Exception as exc:
        record.update(status="engine_init_failed", levels_completed=0, steps=0)
        record["errors"].append(f"engine construction/reset failed: {type(exc).__name__}: {exc}")
        return CollectedGame(record, None, None, specs)

    variant = sample_whole_game_variant(
        variants,
        source_id=source.source_id,
        game_index=game_index,
        legal_action_ids=game.legal_action_ids,
    )
    record["variant"] = variant.private_metadata()
    record["whole_game_rule_variants"] = (
        "absent_baseline" if not variants.enabled else
        ("transformed" if not variant.is_identity else "identity_mix_member")
    )
    rng = np.random.default_rng(record["rollout_seed"])
    columns = _empty_columns()
    _append_public_state(columns, game, variant)
    searches: list[dict[str, Any]] = []
    failed = False
    use_learner = rollout.perturbation == "learner" and perturber is not None
    if use_learner:
        perturber.configure_variant(variant)
        perturber.reset()
        if getattr(perturber, "checkpoint_sha256", None):
            record["learner_checkpoint_sha256"] = perturber.checkpoint_sha256
            record["learner_input_contract"] = {
                "history_mode": perturber.history_mode,
                "canonical_inputs": perturber.canonical_inputs,
            }
    previous_public: Action | None = None
    previous_boundary = False
    rollout_started = time.monotonic()
    game_deadline = (
        None if rollout.game_seconds is None else rollout_started + rollout.game_seconds
    )
    region_sizes: list[int] = []
    region_candidates = 0
    region_seconds = 0.0
    region_failures = 0

    def game_seconds_left() -> float | None:
        return None if game_deadline is None else game_deadline - time.monotonic()

    def search_deadline() -> tuple[float | None, str | None]:
        """Wall-clock cap for the next live search and which cap is binding."""
        remaining = game_seconds_left()
        if rollout.recovery_seconds is None:
            return remaining, (None if remaining is None else "game")
        if remaining is None or rollout.recovery_seconds <= remaining:
            return rollout.recovery_seconds, "recovery"
        return remaining, "game"

    for expected_level, entry in enumerate(entries):
        before_level = game.progress
        if before_level.terminal:
            record["errors"].append(
                f"engine became terminal before requested level {expected_level}: {before_level.state}"
            )
            failed = True
            break
        if before_level.levels_completed != expected_level:
            record["errors"].append(
                f"expected {expected_level} completed levels, engine reports {before_level.levels_completed}"
            )
            failed = True
            break
        plan: PlanResult | None = None
        plan_route_source = LEGACY_ROUTE_SOURCE
        offset = 0
        level_steps = 0
        search_trigger = "level_start"
        while game.progress.levels_completed == expected_level:
            if len(columns["action_id"]) >= max_game_steps:
                record["errors"].append(
                    f"whole-game step cap {max_game_steps} reached during level {expected_level}"
                )
                failed = True
                break
            if level_steps >= limits.max_actions_per_level:
                record["errors"].append(
                    f"level {expected_level} action cap {limits.max_actions_per_level} reached"
                )
                failed = True
                break
            remaining_game_seconds = game_seconds_left()
            if remaining_game_seconds is not None and remaining_game_seconds <= 0.0:
                elapsed = time.monotonic() - rollout_started
                record["errors"].append(
                    f"whole-game wall-clock cap {rollout.game_seconds}s reached during level "
                    f"{expected_level} after {elapsed:.1f}s"
                )
                record["game_timeout"] = True
                record["timeout"] = {
                    "kind": GAME_TIMEOUT_REASON,
                    "level_index": expected_level,
                    "trigger": search_trigger,
                    "elapsed_seconds": elapsed,
                    "cap_seconds": rollout.game_seconds,
                }
                failed = True
                break
            if plan is None:
                search_deadline_used, binding_cap = search_deadline()
                try:
                    remaining_actions = limits.max_actions_per_level - level_steps
                    if require_full_standard and search_trigger == "level_start":
                        plan = _certified_spec_plan(
                            game,
                            specs[expected_level],
                            expected_level=expected_level,
                            max_actions=remaining_actions,
                        )
                        plan_route_source = CERTIFIED_ROUTE_SOURCE
                    else:
                        plan = solve_live_level(
                            game,
                            SearchLimits(
                                max_actions_per_level=remaining_actions,
                                max_search_work=entry.search_work,
                            ),
                            deadline_seconds=search_deadline_used,
                        )
                        plan_route_source = (
                            RECOVERY_ROUTE_SOURCE
                            if search_trigger == "after_perturbation"
                            else LEGACY_ROUTE_SOURCE
                        )
                except SolverError as exc:
                    record["errors"].append(str(exc))
                    failed = True
                    break
                searches.append({
                    "level_index": expected_level,
                    "difficulty": entry.difficulty,
                    "context_index": entry.context_index,
                    "search_work_limit": (
                        None if plan_route_source == CERTIFIED_ROUTE_SOURCE
                        else entry.search_work
                    ),
                    "trigger": search_trigger,
                    "adapter": plan.adapter,
                    "actions": 0 if plan.actions is None else len(plan.actions),
                    "truncated": plan.truncated,
                    "unsupported": plan.unsupported,
                    "exact": plan.exact,
                    "work": plan.work,
                    "reason": plan.reason,
                    "route_source": plan_route_source,
                    "timed_out": plan.timed_out,
                    "elapsed_seconds": plan.elapsed_seconds,
                    "deadline_seconds": (
                        None if plan_route_source == CERTIFIED_ROUTE_SOURCE
                        else search_deadline_used
                    ),
                })
                if plan.timed_out:
                    record["search_timeouts"] += 1
                    recovery = search_trigger == "after_perturbation"
                    if binding_cap == "game":
                        kind = GAME_TIMEOUT_REASON
                        record["game_timeout"] = True
                    elif recovery:
                        kind = RECOVERY_TIMEOUT_REASON
                        record["recovery_timeouts"] += 1
                    else:
                        kind = "live_search_timeout"
                    record["timeout"] = {
                        "kind": kind,
                        "level_index": expected_level,
                        "trigger": search_trigger,
                        "elapsed_seconds": plan.elapsed_seconds,
                        "cap_seconds": search_deadline_used,
                        "binding_cap": binding_cap,
                    }
                    record["errors"].append(
                        f"level {expected_level} {'recovery' if recovery else 'live'} search "
                        f"exceeded its {search_deadline_used}s wall-clock cap "
                        f"({kind}; binding cap: {binding_cap}_seconds); partial trace retained"
                    )
                    failed = True
                    break
                if plan.actions is None:
                    record["errors"].append(
                        f"level {expected_level} search returned no plan; "
                        f"truncated={plan.truncated}; unsupported={plan.unsupported}; "
                        f"exact={plan.exact}; {plan.reason}"
                    )
                    failed = True
                    break
                if plan.unsupported or plan.exact is False:
                    record["errors"].append(
                        f"level {expected_level} rejected non-exact plan; "
                        f"unsupported={plan.unsupported}; exact={plan.exact}; {plan.reason}"
                    )
                    failed = True
                    break
                offset = 0

            prior = game.progress
            try:
                target_raw = game.validate_action(plan.actions[offset])
            except IllegalActionError as exc:
                record["errors"].append(f"level {expected_level} illegal teacher action {offset}: {exc}")
                failed = True
                break
            target_public = _public_action(variant, target_raw)
            learner_public: Action | None = None
            if use_learner:
                try:
                    learner_public = Action.parse(perturber.step(
                        np.asarray(columns["frames"][-1]),
                        np.asarray(columns["legal_action_mask"][-1]),
                        previous_public,
                        previous_boundary,
                    ))
                except Exception as exc:
                    record["errors"].append(
                        f"level {expected_level} learner policy failed: {type(exc).__name__}: {exc}"
                    )
                    failed = True
                    break
            random_step = bool(
                rollout.random_action_probability > 0.0
                and rng.random() < rollout.random_action_probability
            )
            if random_step:
                transition_route_source = RANDOM_ROUTE_SOURCE
                executed_public = executed_raw = None
                if use_learner and learner_public is not None:
                    try:
                        executed_raw = game.validate_action(
                            Action(*variant.raw_action(*learner_public.as_tuple()))
                        )
                        executed_public = learner_public
                        transition_route_source = LEARNER_ROUTE_SOURCE
                    except IllegalActionError:
                        # An illegal learner proposal still perturbs: fall back
                        # to a uniform legal action and count the fallback.
                        record["learner_illegal_fallbacks"] += 1
                if executed_raw is None:
                    try:
                        executed_public, executed_raw = _sample_random_action(game, variant, rng)
                    except IllegalActionError as exc:
                        record["errors"].append(f"level {expected_level} random action failed: {exc}")
                        failed = True
                        break
                action_source = RANDOM_SOURCE
            else:
                executed_raw = target_raw
                executed_public = target_public
                action_source = TEACHER_SOURCE
                transition_route_source = plan_route_source
            public_region: np.ndarray | None = None
            if target_raw.id == CLICK_ACTION and rollout.click_region_probe_limit > 0:
                # Label the teacher click with every pixel the engine treats
                # identically, probing clones of the pre-state only.
                try:
                    region = equivalent_click_region(
                        game, target_raw, limit=rollout.click_region_probe_limit,
                    )
                except Exception as exc:
                    region_failures += 1
                    if region_failures == 1:
                        record["errors"].append(
                            f"level {expected_level} click-region probe failed at plan offset "
                            f"{offset}: {type(exc).__name__}: {exc}; exact target retained"
                        )
                else:
                    public_region = transform_frame(region.mask, variant.spatial)
                    region_sizes.append(region.size)
                    region_candidates += region.candidates
                    region_seconds += region.seconds
            _append_transition(
                columns,
                executed=executed_public,
                target=target_public,
                source=action_source,
                level=expected_level,
                plan_offset=offset,
                plan_length=len(plan.actions),
                route_source=transition_route_source,
                click_region=public_region,
            )
            try:
                game.perform(executed_raw)
                after = game.progress
                _append_public_state(columns, game, variant)
            except Exception as exc:
                # Remove the action without a matching after-state.
                _pop_transition(columns)
                try:
                    raw = game.progress
                    record["unrecorded_engine_state_after_error"] = {
                        "state": raw.state,
                        "level_index": raw.level_index,
                        "levels_completed": raw.levels_completed,
                        "terminal": raw.terminal,
                        "won": raw.won,
                    }
                except Exception:
                    pass
                record["errors"].append(
                    f"level {expected_level} action {offset} engine failure: {type(exc).__name__}: {exc}"
                )
                failed = True
                break
            boundary = after.levels_completed > prior.levels_completed
            columns["level_boundary"].append(boundary)
            level_steps += 1
            previous_public = executed_public
            previous_boundary = boundary
            if after.state == "GAME_OVER":
                kind = (
                    "teacher" if not random_step
                    else "learner" if transition_route_source == LEARNER_ROUTE_SOURCE
                    else "random"
                )
                record["errors"].append(
                    f"level {expected_level} reached GAME_OVER after {kind} action at plan offset {offset}"
                )
                failed = True
                break
            if random_step:
                # Even when the perturbation equals the target, source stays
                # non-teacher and recovery must be planned from the actual result.
                plan = None
                search_trigger = "after_perturbation"
                if boundary:
                    break
                continue

            offset += 1
            if boundary and offset != len(plan.actions):
                record["errors"].append(
                    f"level {expected_level} completed before the end of its teacher plan"
                )
                failed = True
                break
            if boundary:
                break
            if offset == len(plan.actions):
                record["errors"].append(
                    f"level {expected_level} teacher plan ended without engine completion"
                )
                failed = True
                break
        if failed:
            break
        if game.progress.levels_completed != expected_level + 1:
            record["errors"].append(f"level {expected_level} did not advance exactly once")
            failed = True
            break

    public, teacher = _arrays(columns)
    if rollout.click_region_probe_limit == 0:
        # An absent region array invokes the loader's legacy exact-pixel
        # fallback. An all-zero array would falsely claim invalid stored labels.
        teacher.pop("click_region_mask")
        teacher.pop("click_region_size")
    # The retained public timeline is authoritative.  An engine may mutate and
    # then fail while rendering its successor; that unrecorded raw state is
    # diagnostic only and cannot define this trajectory's final public state.
    final = Progress(
        state=str(public["state"][-1]),
        level_index=int(public["level_index"][-1]),
        levels_completed=int(public["levels_completed"][-1]),
        terminal=bool(public["terminal"][-1]),
        won=bool(public["won"][-1]),
    )
    won = (
        not failed
        and final.won
        and final.levels_completed == len(specs)
        and int(public["level_boundary"].sum()) == len(specs)
    )
    if not won and not record["errors"]:
        record["errors"].append(
            f"engine did not report WIN after all requested levels: state={final.state}, "
            f"completed={final.levels_completed}/{len(specs)}"
        )
    route_sources, route_counts = np.unique(teacher["route_source"], return_counts=True)
    click_targets = int((teacher["target_action_id"] == CLICK_ACTION).sum())
    record["click_region_stats"] = {
        "click_targets": click_targets,
        "labelled": len(region_sizes),
        "probe_failures": region_failures,
        "mean_size": float(np.mean(region_sizes)) if region_sizes else None,
        "median_size": float(np.median(region_sizes)) if region_sizes else None,
        "min_size": int(min(region_sizes)) if region_sizes else None,
        "max_size": int(max(region_sizes)) if region_sizes else None,
        "candidates_probed": int(region_candidates),
        "probe_seconds": float(region_seconds),
        "probe_seconds_per_click": (
            float(region_seconds) / len(region_sizes) if region_sizes else None
        ),
    }
    record.update(
        status="won" if won else "rollout_failed",
        levels_completed=final.levels_completed,
        final_state=final.state,
        steps=int(public["action_id"].shape[0]),
        teacher_steps=int((teacher["source"] == TEACHER_SOURCE).sum()),
        random_steps=int((teacher["source"] == RANDOM_SOURCE).sum()),
        learner_steps=int(np.sum(teacher["route_source"] == LEARNER_ROUTE_SOURCE)),
        rollout_wall_clock_seconds=time.monotonic() - rollout_started,
        route_source_counts={
            str(name): int(count) for name, count in zip(route_sources, route_counts)
        },
        certified_teacher_actions=int(
            np.sum(teacher["route_source"] == CERTIFIED_ROUTE_SOURCE)
        ),
        live_recovery_teacher_actions=int(
            np.sum(teacher["route_source"] == RECOVERY_ROUTE_SOURCE)
        ),
        certified_route_levels_completed=int(np.sum(
            public["level_boundary"]
            & (teacher["route_source"] == CERTIFIED_ROUTE_SOURCE)
        )),
        random_metric_eligible_steps=int((teacher["source"] == RANDOM_SOURCE).sum()),
        random_transition_scoring_available=bool((teacher["source"] == RANDOM_SOURCE).any()),
        level_boundaries=int(public["level_boundary"].sum()),
        searches=searches,
        public_keys=sorted(public),
        teacher_keys=sorted(teacher),
    )
    record["certified_solution_completed"] = bool(
        require_full_standard
        and won
        and record["certified_route_levels_completed"] == len(specs)
        and record["random_steps"] == 0
        and record["live_recovery_teacher_actions"] == 0
    )
    record["actual_trajectory_coverage"] = {
        "certified_teacher_actions": record["certified_teacher_actions"],
        "live_recovery_teacher_actions": record["live_recovery_teacher_actions"],
        "random_actions": record["random_steps"] - record["learner_steps"],
        "learner_actions": record["learner_steps"],
        "certified_route_levels_completed": record["certified_route_levels_completed"],
        "all_levels_completed_by_unperturbed_certified_solutions": record[
            "certified_solution_completed"
        ],
        "stored_mechanic_certificate_inherited_by_recovery": False,
    }
    return CollectedGame(record, public, teacher, specs)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    _atomic_bytes(path, buffer.getvalue())


def save_collected_game(root: str | Path, key: str, collected: CollectedGame) -> dict[str, Any]:
    """Persist one status record, with public and teacher arrays physically separate."""
    root = Path(root)
    record = dict(collected.record)
    record_source = source_for(record.get("source_id", record.get("source", "")))
    if record_source.held_out:
        raise ValueError(f"held-out source {record_source.source_id} cannot be persisted as training data")
    if record.get("source") != record_source.slug or record.get("source_id") != record_source.source_id:
        raise ValueError("collected record source/source_id must be canonical and consistent")
    if collected.public is not None:
        public_path = root / "games" / f"{key}.npz"
        teacher_path = root / "teacher" / f"{key}.npz"
        _atomic_npz(public_path, collected.public)
        _atomic_npz(teacher_path, collected.teacher or {})
        record["public_npz"] = str(public_path.relative_to(root))
        record["teacher_npz"] = str(teacher_path.relative_to(root))
    if collected.specs:
        specs_path = root / "teacher" / f"{key}.levels.json"
        _atomic_json(specs_path, collected.specs)
        record["generated_specs"] = str(specs_path.relative_to(root))
    record_path = root / "records" / f"{key}.json"
    record["record"] = str(record_path.relative_to(root))
    _atomic_json(record_path, record)
    return record


def save_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    path = Path(path)
    value = dict(manifest)
    if value.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"manifest format must be {MANIFEST_FORMAT!r}")
    requested = value.get("requested_source_ids", ())
    resolved = resolve_collection_sources(requested)
    if [source.source_id for source in resolved] != list(requested):
        raise ValueError("manifest requested_source_ids must use canonical source ids")
    requested_ids = set(requested)
    for record in value.get("records", ()):
        record_source = source_for(record.get("source_id", record.get("source", "")))
        if record_source.held_out or record_source.source_id not in requested_ids:
            raise ValueError(
                f"manifest record source {record_source.source_id} is held out or was not requested"
            )
        if (record.get("source") != record_source.slug
                or record.get("source_id") != record_source.source_id):
            raise ValueError("manifest record source/source_id must be canonical and consistent")
    _atomic_json(path, value)
    return path


def load_manifest(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if value.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"unsupported manifest format {value.get('format')!r}")
    requested = value.get("requested_source_ids", ())
    resolved = resolve_collection_sources(requested)
    if [source.source_id for source in resolved] != list(requested):
        raise ValueError("manifest requested_source_ids must use canonical source ids")
    requested_ids = set(requested)
    for record in value.get("records", ()):
        record_source = source_for(record.get("source_id", record.get("source", "")))
        if record_source.held_out or record_source.source_id not in requested_ids:
            raise ValueError(
                f"manifest record source {record_source.source_id} is held out or was not requested"
            )
        if (record.get("source") != record_source.slug
                or record.get("source_id") != record_source.source_id):
            raise ValueError("manifest record source/source_id must be canonical and consistent")
    return value


def new_manifest(
    *,
    sources: Sequence[Source],
    explicit_subset: bool,
    seed: int,
    games_per_source: int,
    difficulties: Sequence[int] | None,
    limits: SearchLimits,
    variants: VariantOptions = VariantOptions(),
    rollout: RolloutOptions = RolloutOptions(),
    provenance: Mapping[str, Any] | None = None,
    curricula_by_source: Mapping[str, Sequence[CurriculumEntry]] | None = None,
    full_standard_contracts: Mapping[str, FullStandardContract] | None = None,
) -> dict[str, Any]:
    sources = resolve_collection_sources(sources)
    full_source_set = {source.source_id for source in sources} == set(TRAIN_SOURCE_IDS)
    if not explicit_subset and not full_source_set:
        raise ValueError("non-explicit collection scope must contain all 24 training sources")
    source_ids = {source.source_id for source in sources}
    if curricula_by_source is None:
        if difficulties is None:
            raise ValueError("smoke manifests require explicit difficulties")
        curricula = {
            source.source_id: tuple(
                CurriculumEntry(int(value), index, limits.max_search_work)
                for index, value in enumerate(difficulties)
            )
            for source in sources
        }
    else:
        if set(curricula_by_source) != source_ids:
            raise ValueError("curricula_by_source must exactly match requested sources")
        curricula = {key: tuple(value) for key, value in curricula_by_source.items()}
        if any(
            not value or not all(isinstance(entry, CurriculumEntry) for entry in value)
            for value in curricula.values()
        ):
            raise ValueError("every source curriculum must contain CurriculumEntry values")
    full_contracts: dict[str, FullStandardContract] = {}
    if not explicit_subset:
        if difficulties is not None:
            raise ValueError("full-standard manifests cannot use one shared difficulty sequence")
        if full_standard_contracts is None or set(full_standard_contracts) != source_ids:
            raise ValueError("full-standard manifests require one contract per requested source")
        for source in sources:
            contract = full_standard_contracts[source.source_id]
            if (
                not isinstance(contract, FullStandardContract)
                or not contract.ready
                or contract.source_id != source.source_id
            ):
                raise ValueError(f"{source.slug} does not have an accepted full-standard contract")
            if curricula[source.source_id] != contract.curriculum:
                raise ValueError(f"{source.slug} manifest curriculum differs from its contract")
            full_contracts[source.source_id] = contract
    max_game_steps = rollout.max_game_steps
    if max_game_steps is None:
        max_game_steps = max(
            1, max(len(value) for value in curricula.values()) * limits.max_actions_per_level,
        )
    rollout_mode = rollout.mode
    return {
        "format": MANIFEST_FORMAT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "smoke_subset" if explicit_subset else "full_24_source_collection",
        "is_full_experiment_collection": bool(
            not explicit_subset and full_source_set and len(full_contracts) == len(sources)
        ),
        "requested_source_ids": [source.source_id for source in sources],
        "held_out_source_id": HELD_OUT_SOURCE_ID,
        "seed": int(seed),
        "games_per_source": int(games_per_source),
        "difficulties": (
            None if difficulties is None else [int(value) for value in difficulties]
        ),
        "curricula_by_source": {
            source_id: [entry.to_dict() for entry in curricula[source_id]]
            for source_id in sorted(curricula)
        },
        "full_standard": {
            "format": FULL_STANDARD_FORMAT,
            "required": not explicit_subset,
            "ready": bool(not explicit_subset and len(full_contracts) == len(sources)),
            "contract_hashes": {
                source_id: full_contracts[source_id].sha256
                for source_id in sorted(full_contracts)
            },
            "contracts": {
                source_id: full_contracts[source_id].to_dict()
                for source_id in sorted(full_contracts)
            },
        },
        "max_actions_per_level": limits.max_actions_per_level,
        "max_search_work": limits.max_search_work,
        "rollout_mode": rollout_mode,
        "random_action_probability": rollout.random_action_probability,
        "perturbation": rollout.perturbation,
        "learner_checkpoint": rollout.learner_checkpoint,
        "max_game_steps": max_game_steps,
        "recovery_seconds": rollout.recovery_seconds,
        "game_seconds": rollout.game_seconds,
        "wall_clock_caps": (
            "recovery_seconds caps each live family search after a perturbation; "
            "game_seconds caps one whole rollout; a capped search ends the game as "
            "rollout_failed with its usable partial trace retained and counted in "
            "recovery_timeouts/game_timeouts"
        ),
        "variant_request": variants.to_dict(),
        "random_transition_scoring_available": False,
        "whole_game_rule_variants": "requested" if variants.enabled else "absent_baseline",
        "variant_metadata_location": (
            "private record JSON only; public NPZ contains transformed observations/actions "
            "without a variant ID"
        ),
        "provenance": (
            _manifest_provenance(sources)
            if provenance is None
            else _explicit_manifest_provenance(sources, provenance)
        ),
        "provenance_location": (
            "private manifest/record JSON only; public NPZ and model inputs contain no code identity"
        ),
        "coverage_caveat": (
            "Full readiness is contract-derived and retains reviewed caveats/evidence per source. "
            "Historical core coverage notes are labelled separately. Public transforms are "
            "inexpensive representational variants, not new mechanics. Random-action transition "
            "metrics are available only when records contain actual source=0 rows."
            if full_contracts else
            "Package readiness proves the historical adapter contract only. Generator-declared "
            "mechanics coverage and omissions are recorded per level. Public transforms are "
            "inexpensive representational variants, not new mechanics."
        ),
        "source_coverage_notes": {
            source.source_id: (
                _full_standard_coverage(full_contracts[source.source_id])
                if source.source_id in full_contracts else
                KNOWN_COVERAGE.get(source.slug, "not audited")
            )
            for source in sources
        },
        "historical_core_coverage_notes": (
            {
                source.source_id: KNOWN_COVERAGE.get(source.slug, "not audited")
                for source in sources
            }
            if full_contracts else {}
        ),
        "public_schema": {
            "frames": (
                "uint8 [transitions+1,64,64], public palette/display space after any fixed transform"
            ),
            "legal_action_mask": (
                "bool [transitions+1,8], legal IDs in transformed public control space"
            ),
            "state": "unicode [transitions+1]",
            "level_index": "int16 [transitions+1]",
            "levels_completed": "int16 [transitions+1]",
            "terminal": "bool [transitions+1]",
            "won": "bool [transitions+1]",
            "action_id": "int8 [transitions]",
            "action_x/action_y": (
                "int16 [transitions], public full-display coordinates; -1 for non-click"
            ),
            "level_boundary": "bool [transitions], true only when that action completed a level",
        },
        "teacher_schema": {
            "target_action_id/x/y": "teacher target triples aligned with transitions",
            "source": (
                f"int8 [transitions], {RANDOM_SOURCE}=perturbation execution (random or "
                f"learner) and {TEACHER_SOURCE}=exact-teacher execution"
            ),
            "plan_level/plan_offset/plan_length": "teacher plan provenance",
            "route_source": (
                f"unicode [transitions], one of {CERTIFIED_ROUTE_SOURCE!r}, "
                f"{RECOVERY_ROUTE_SOURCE!r}, {LEGACY_ROUTE_SOURCE!r}, "
                f"{RANDOM_ROUTE_SOURCE!r}, {LEARNER_ROUTE_SOURCE!r}"
            ),
            "click_region_mask": (
                "uint8 [transitions,64,64], public display pixels whose click on a clone of "
                "the pre-state yields the same public successor as the teacher click target "
                "(engine-verified, bounded probe); all-zero for non-click targets"
            ),
            "click_region_size": (
                "int16 [transitions], pixel count of click_region_mask; 0 for non-click targets"
            ),
        },
        "click_region_probe_limit": rollout.click_region_probe_limit,
        "records": [],
        "status_counts": {},
        "games_requested": len(sources) * int(games_per_source),
        "games_recorded": 0,
        "games_won": 0,
        "levels_requested": int(games_per_source) * sum(
            len(value) for value in curricula.values()
        ),
        "levels_completed": 0,
        "steps": 0,
        "teacher_steps": 0,
        "random_steps": 0,
        "learner_steps": 0,
        "live_recovery_steps": 0,
        "route_source_counts": {},
        "recovery_timeouts": 0,
        "search_timeouts": 0,
        "game_timeouts": 0,
        "random_metric_eligible_steps": 0,
        "click_region_summary": {},
        "rollout_failures": [],
    }


def route_source_counts_from_teacher(teacher: Mapping[str, Any], steps: int) -> dict[str, int]:
    """Count aligned NPZ provenance, preserving unknown legacy provenance honestly."""
    targets = np.asarray(teacher["target_action_id"])
    if targets.shape != (steps,):
        raise ValueError("teacher targets do not align with record steps")
    if "route_source" not in teacher:
        return {"unknown": steps} if steps else {}
    routes = np.asarray(teacher["route_source"])
    if routes.shape != (steps,) or routes.dtype.kind not in "US":
        raise ValueError("teacher route_source must be a string array aligned with record steps")
    values, counts = np.unique(routes.astype(str), return_counts=True)
    if any(not value for value in values):
        raise ValueError("teacher route_source contains empty provenance")
    return {str(value): int(count) for value, count in zip(values, counts)}


def update_manifest_summary(manifest: dict[str, Any]) -> None:
    records = manifest["records"]
    statuses: dict[str, int] = {}
    for record in records:
        status = record["status"]
        statuses[status] = statuses.get(status, 0) + 1
    manifest["status_counts"] = statuses
    manifest["games_recorded"] = len(records)
    manifest["games_won"] = sum(record["status"] == "won" for record in records)
    manifest["levels_completed"] = sum(int(record.get("levels_completed", 0)) for record in records)
    manifest["steps"] = sum(int(record.get("steps", 0)) for record in records)
    manifest["teacher_steps"] = sum(int(record.get("teacher_steps", 0)) for record in records)
    manifest["random_steps"] = sum(int(record.get("random_steps", 0)) for record in records)
    manifest["learner_steps"] = sum(int(record.get("learner_steps", 0)) for record in records)
    manifest["live_recovery_steps"] = sum(
        int(record.get("live_recovery_teacher_actions", 0)) for record in records
    )
    route_counts: dict[str, int] = {}
    for record in records:
        steps = int(record.get("steps", 0))
        counts = record.get("route_source_counts") or ({"unknown": steps} if steps else {})
        if any(int(count) < 0 for count in counts.values()) or sum(map(int, counts.values())) != steps:
            raise ValueError("record route_source_counts must sum to steps")
        for name, count in counts.items():
            route_counts[str(name)] = route_counts.get(str(name), 0) + int(count)
    manifest["route_source_counts"] = dict(sorted(route_counts.items()))
    manifest["recovery_timeouts"] = sum(
        int(record.get("recovery_timeouts", 0)) for record in records
    )
    manifest["search_timeouts"] = sum(int(record.get("search_timeouts", 0)) for record in records)
    manifest["game_timeouts"] = sum(bool(record.get("game_timeout", False)) for record in records)
    manifest["random_metric_eligible_steps"] = sum(
        int(record.get("random_metric_eligible_steps", 0)) for record in records
    )
    manifest["random_transition_scoring_available"] = manifest["random_steps"] > 0
    region_stats = [
        record["click_region_stats"] for record in records
        if isinstance(record.get("click_region_stats"), Mapping)
    ]
    labelled = sum(int(item.get("labelled", 0)) for item in region_stats)
    probe_seconds = sum(float(item.get("probe_seconds", 0.0)) for item in region_stats)
    manifest["click_region_summary"] = {
        "click_targets": sum(int(item.get("click_targets", 0)) for item in region_stats),
        "labelled": labelled,
        "probe_failures": sum(int(item.get("probe_failures", 0)) for item in region_stats),
        "mean_size": (
            sum(float(item["mean_size"]) * int(item["labelled"]) for item in region_stats
                if item.get("mean_size") is not None) / labelled
            if labelled else None
        ),
        "max_size": max(
            (int(item["max_size"]) for item in region_stats if item.get("max_size") is not None),
            default=None,
        ),
        "probe_seconds": probe_seconds,
        "probe_seconds_per_click": probe_seconds / labelled if labelled else None,
    }
    manifest["rollout_failures"] = [
        record.get("record", f"{record.get('source', 'unknown')}:{record.get('game_index', '?')}")
        for record in records
        if record.get("status") != "won"
    ]
