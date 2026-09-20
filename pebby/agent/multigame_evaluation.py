"""Frozen public-policy evaluation on sequential official ARC-AGI-3 games."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from pebby import multigame as M
from .multigame_model import CLICK_ACTION, MultiGameModel
from .multigame_training import model_from_training_checkpoint, sha256_file


REPORT_FORMAT = "pebby-multigame-official-evaluation-v1"
PINNED_M0R0_SHA256 = "fc3954236f712759c2a5baaada09bb3244e217e022086dc9ec8f6e6fe101d797"


def _hash_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _state_name(value: Any) -> str:
    value = getattr(value, "value", value)
    return str(value).upper()


def _action_id(value: Any) -> int:
    return int(getattr(value, "value", value))


def _public_legal_mask(available_actions: Any) -> np.ndarray:
    """Legal mask over public action IDs; ``ValueError`` names the contract breach.

    RESET (0) is never offered to the policy.  IDs outside ``0..7`` or not
    integer-like, and sets with no playable public action, are adapter faults.
    """
    legal = np.zeros(M.ACTION_COUNT, dtype=np.bool_)
    for value in available_actions:
        try:
            action_id = _action_id(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid_legal_action_id:{value!r}") from exc
        if not 0 <= action_id < M.ACTION_COUNT:
            raise ValueError(f"invalid_legal_action_id:{action_id}")
        if action_id:
            legal[action_id] = True
    if not legal[1:].any():
        raise ValueError("no_legal_public_action")
    return legal


def _validated_public_frame(value: Any) -> np.ndarray:
    frame = np.asarray(value)
    if frame.shape != M.FRAME_SHAPE:
        raise ValueError(f"invalid public frame shape: {frame.shape}")
    if frame.dtype == np.bool_ or not np.issubdtype(frame.dtype, np.integer):
        raise ValueError(f"invalid public frame dtype: {frame.dtype}; expected non-bool integers")
    if frame.size and (int(frame.min()) < 0 or int(frame.max()) >= 16):
        raise ValueError("invalid public frame palette outside 0..15")
    return frame.astype(np.uint8, copy=True)


@dataclass(frozen=True)
class OfficialEvaluationConfig:
    max_actions_per_level: int = 256
    max_game_actions: int = 2048
    voluntary_resets: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.max_actions_per_level < 1 or self.max_game_actions < 1:
            raise ValueError("official action budgets must be positive")
        if self.voluntary_resets != 0:
            raise ValueError("the v1 official protocol permits zero voluntary resets only")

    @property
    def hash(self) -> str:
        value = asdict(self)
        value.pop("device", None)
        return _hash_json(value)


@torch.inference_mode()
def run_policy_game(
    model: MultiGameModel,
    env: Any,
    *,
    source_id: str,
    config: OfficialEvaluationConfig,
    history_mode: str = "full",
) -> dict[str, Any]:
    """Run one sequential game using only public observations and policy memory."""
    device = next(model.parameters()).device
    if history_mode not in {"full", "none"}:
        raise ValueError(f"invalid inference history mode: {history_mode!r}")
    env.reset()  # required game initialization, not a voluntary retry
    memory = model.initial_memory(1, device=device)
    previous_id = previous_x = previous_y = -1
    previous_boundary = False
    actions = 0
    level_actions = 0
    failures: list[str] = []
    initial_completed = int(getattr(env, "levels_completed", 0))
    while actions < config.max_game_actions:
        state = _state_name(getattr(env, "state"))
        if state in {"WIN", "GAME_OVER"}:
            break
        if level_actions >= config.max_actions_per_level:
            failures.append("per_level_action_budget")
            break
        # Validate the advertised action set before any inference: an empty
        # or malformed set is reported as an adapter failure, never raised.
        try:
            legal = _public_legal_mask(getattr(env, "available_actions"))
        except ValueError as exc:
            failures.append(str(exc))
            break
        try:
            frame = _validated_public_frame(env.render())
        except ValueError as exc:
            failures.append(f"invalid_public_frame:{exc}")
            break
        output, memory = model.policy_step(
            frame=torch.from_numpy(frame)[None].to(device),
            previous_action_id=torch.tensor([previous_id], device=device),
            previous_action_x=torch.tensor([previous_x], device=device),
            previous_action_y=torch.tensor([previous_y], device=device),
            previous_level_boundary=torch.tensor([previous_boundary], device=device),
            terminal=torch.tensor([False], device=device),
            won=torch.tensor([False], device=device),
            legal_action_mask=torch.from_numpy(legal)[None].to(device),
            memory=memory,
            **({"history_keep": torch.zeros(1, dtype=torch.bool, device=device)}
               if history_mode == "none" else {}),
        )
        action_id = int(output.action_logits.argmax(-1).item())
        if action_id == CLICK_ACTION:
            click_x, click_y = model.decode_click(output.click_logits)
            x, y = int(click_x.item()), int(click_y.item())
        else:
            x = y = None
        before = int(getattr(env, "levels_completed"))
        try:
            env.perform(action_id, x, y)
        except Exception as exc:
            failures.append(f"engine_error:{type(exc).__name__}:{exc}")
            break
        after = int(getattr(env, "levels_completed"))
        previous_boundary = after > before
        previous_id = action_id
        previous_x = -1 if x is None else x
        previous_y = -1 if y is None else y
        actions += 1
        level_actions = 0 if previous_boundary else level_actions + 1
    state = _state_name(getattr(env, "state"))
    if actions >= config.max_game_actions and state not in {"WIN", "GAME_OVER"}:
        failures.append("whole_game_action_budget")
    completed = int(getattr(env, "levels_completed")) - initial_completed
    level_count = getattr(env, "level_count", None)
    return {
        "source_id": source_id,
        "levels_completed": completed,
        "levels_total": None if level_count is None else int(level_count),
        "whole_game_win": state == "WIN",
        "actions": actions,
        "initial_resets": 1,
        "voluntary_resets": 0,
        "final_state": state,
        "failures": failures,
        "unsupported_adapter_pieces": M.KNOWN_COVERAGE.get(
            M.source_for(source_id).slug,
            "game-specific official adapter coverage has not been audited",
        ),
    }


def make_training_family_official_env(source: M.Source) -> Any:
    if source.held_out:
        raise ValueError("held-out source requires the pinned m0r0 adapter")
    module = importlib.import_module(f"{source.package}.env")
    return module.Env()


class PinnedM0R0PublicEnv:
    """Public-response wrapper over the pinned vendored m0r0 engine."""

    def __init__(self, game: Any):
        self._game = game
        self._last = None
        self.level_count = None

    def reset(self) -> np.ndarray:
        from arcengine import ActionInput, GameAction

        self._last = self._game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        self.level_count = int(self._last.win_levels)
        return self.render()

    def perform(self, action_id: int, x: int | None = None, y: int | None = None) -> Any:
        from arcengine import ActionInput, GameAction

        if action_id == CLICK_ACTION:
            if x is None or y is None:
                raise ValueError("click action requires full display coordinates")
            data = {"x": int(x), "y": int(y)}
        else:
            if x is not None or y is not None:
                raise ValueError("non-click action cannot have coordinates")
            data = {}
        self._last = self._game.perform_action(
            ActionInput(id=GameAction.from_id(int(action_id)), data=data), raw=True,
        )
        return self._last

    def render(self) -> np.ndarray:
        if self._last is None or not self._last.frame:
            raise RuntimeError("m0r0 has no public frame; call reset first")
        return np.asarray(self._last.frame[-1])

    @property
    def available_actions(self):
        return tuple(self._last.available_actions)

    @property
    def levels_completed(self) -> int:
        return int(self._last.levels_completed)

    @property
    def state(self) -> Any:
        return self._last.state


def make_pinned_m0r0_env() -> PinnedM0R0PublicEnv:
    """Import the exact vendored source locally; never download or use Arcade.make."""
    path = Path(__file__).resolve().parents[2] / "third_party" / "arc3_games" / "m0r0.py"
    actual = sha256_file(path)
    if actual != PINNED_M0R0_SHA256:
        raise ValueError(f"pinned m0r0 source hash mismatch: {actual}")
    name = "pebby_pinned_official_m0r0"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load pinned m0r0 source")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return PinnedM0R0PublicEnv(module.M0r0())


def _validate_holdout_gate(
    report_path: str | Path,
    *,
    checkpoint_sha256: str,
    evaluation_config_hash: str,
) -> dict[str, Any]:
    report = json.loads(Path(report_path).read_text())
    if report.get("format") != REPORT_FORMAT:
        raise ValueError("training-family report has the wrong format")
    if report.get("phase") != "training_families" or not report.get("phase_complete"):
        raise ValueError("heldout requires a completed training-family official report")
    if report.get("scope") != "full_24_family_experiment" or report.get("smoke"):
        raise ValueError("a smoke/partial report cannot unlock heldout evaluation")
    source_ids = report.get("source_ids", ())
    if len(source_ids) != len(M.TRAIN_SOURCE_IDS) or len(set(source_ids)) != len(source_ids):
        raise ValueError("training-family report contains duplicate or missing source IDs")
    if set(source_ids) != set(M.TRAIN_SOURCE_IDS):
        raise ValueError("training-family report does not cover all 24 intended families")
    results = report.get("results")
    if not isinstance(results, list) or len(results) != len(M.TRAIN_SOURCE_IDS):
        raise ValueError("training-family report has missing evaluation results")
    result_ids = [item.get("source_id") for item in results if isinstance(item, Mapping)]
    if (
        len(result_ids) != len(M.TRAIN_SOURCE_IDS)
        or set(result_ids) != set(M.TRAIN_SOURCE_IDS)
        or len(set(result_ids)) != len(result_ids)
        or any(item.get("status") != "evaluated" for item in results)
    ):
        raise ValueError("training-family report includes duplicate, missing, or unsupported results")
    if report.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("training-family report used a different checkpoint")
    if report.get("evaluation_config_hash") != evaluation_config_hash:
        raise ValueError("training-family report used a different evaluation configuration")
    return report


def evaluate_official(
    checkpoint: str | Path,
    output: str | Path,
    *,
    phase: str,
    config: OfficialEvaluationConfig = OfficialEvaluationConfig(),
    sources: Sequence[str | M.Source] | None = None,
    smoke: bool = False,
    training_report: str | Path | None = None,
    env_factory: Callable[[M.Source], Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a frozen checkpoint with phase ordering and heldout gates."""
    if phase not in {"training_families", "heldout"}:
        raise ValueError("phase must be training_families or heldout")
    checkpoint = Path(checkpoint).resolve()
    output = Path(output)
    if output.exists():
        raise ValueError(f"refusing to overwrite existing official report {output}")
    checkpoint_sha = sha256_file(checkpoint)
    config_hash = config.hash
    if phase == "heldout":
        if smoke:
            raise ValueError("smoke evaluation can never open heldout m0r0")
        resolved = (M.source_for("m0r0"),) if sources is None else tuple(M.source_for(item) for item in sources)
        if len(resolved) != 1 or not resolved[0].held_out:
            raise ValueError("heldout phase evaluates exactly m0r0")
        if training_report is None:
            raise ValueError("heldout phase requires --training-report")
        _validate_holdout_gate(
            training_report,
            checkpoint_sha256=checkpoint_sha,
            evaluation_config_hash=config_hash,
        )
    else:
        resolved = M.TRAIN_SOURCES if sources is None else tuple(M.source_for(item) for item in sources)
        resolved_ids = [source.source_id for source in resolved]
        if not resolved_ids or len(set(resolved_ids)) != len(resolved_ids):
            raise ValueError("training-family sources must be non-empty and distinct")
        if any(source.held_out for source in resolved):
            raise ValueError("training-family phase cannot include m0r0")
        if not smoke and set(resolved_ids) != set(M.TRAIN_SOURCE_IDS):
            raise ValueError("full training-family evaluation requires all 24 intended families")

    model, checkpoint_payload = model_from_training_checkpoint(checkpoint, device=config.device)
    history_mode = checkpoint_payload.get("training_config", {}).get(
        "history_mode", checkpoint_payload.get("history_mode", "full"),
    )
    if history_mode not in {"full", "none"}:
        raise ValueError(f"checkpoint has invalid inference history mode: {history_mode!r}")
    checkpoint_scope = checkpoint_payload.get("scope")
    if checkpoint_payload.get("smoke") and not smoke:
        raise ValueError("a smoke checkpoint cannot be reported as a full official evaluation")
    if phase == "heldout" and checkpoint_scope != "full_24_family_experiment":
        raise ValueError("heldout requires a full-experiment checkpoint")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    results = []
    for source in resolved:
        try:
            if env_factory is not None:
                env = env_factory(source)
            elif source.held_out:
                env = make_pinned_m0r0_env()
            else:
                env = make_training_family_official_env(source)
            result = run_policy_game(
                model, env, source_id=source.source_id, config=config, history_mode=history_mode,
            )
            adapter_failure = any(
                str(failure).startswith((
                    "invalid_public_frame", "no_legal_public_action",
                    "invalid_legal_action_id", "engine_error",
                ))
                for failure in result["failures"]
            )
            result["status"] = "adapter_failure" if adapter_failure else "evaluated"
        except Exception as exc:
            result = {
                "source_id": source.source_id,
                "status": "unsupported_adapter",
                "levels_completed": 0,
                "levels_total": None,
                "whole_game_win": False,
                "actions": 0,
                "initial_resets": 0,
                "voluntary_resets": 0,
                "final_state": "NOT_STARTED",
                "failures": [f"{type(exc).__name__}: {exc}"],
                "unsupported_adapter_pieces": M.KNOWN_COVERAGE.get(source.slug, "not audited"),
            }
        results.append(result)
    report = {
        "format": REPORT_FORMAT,
        "phase": phase,
        "phase_complete": (
            len(results) == len(resolved)
            and all(item.get("status") == "evaluated" for item in results)
        ),
        "scope": "smoke" if smoke else checkpoint_scope,
        "smoke": bool(smoke),
        "source_ids": [source.source_id for source in resolved],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_config_hash": _hash_json(checkpoint_payload["model_config"]),
        "evaluation_config": asdict(config),
        "evaluation_config_hash": config_hash,
        "protocol": {
            "history_mode": history_mode,
            "policy_inputs": "public frames/legal IDs/previous action/public events/causal memory only",
            "teacher_or_planner_fallback": False,
            "forced_level_advance": False,
            "voluntary_resets": 0,
            "m0r0_source_inspection_caveat": (
                "the pinned public source was inspected earlier; heldout means excluded from "
                "training/tuning, not unseen source design"
            ),
        },
        "results": results,
        "games_won": sum(item["whole_game_win"] for item in results),
        "levels_completed": sum(item["levels_completed"] for item in results),
        "actions": sum(item["actions"] for item in results),
        "failures": {
            item["source_id"]: item["failures"] for item in results if item["failures"]
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


__all__ = [
    "OfficialEvaluationConfig", "PINNED_M0R0_SHA256", "PinnedM0R0PublicEnv",
    "REPORT_FORMAT", "evaluate_official", "make_pinned_m0r0_env",
    "make_training_family_official_env", "run_policy_game",
]
