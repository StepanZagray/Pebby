"""Pebby-specific declarations for the reusable HostAI provider SDK.

The SDK owns wire/UI validation; Pebby owns operation semantics and docs.
Constructing this module or provider never loads a checkpoint.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path

from hostai import CustomUI, InputError, Interaction, Model, Provider


def interaction() -> Interaction:
    return Interaction(
        instructions=(
            "Pebby exposes stateless LS20 operations, not text chat. op=boot returns info, banks, the "
            "first page of the first bank's training split and that page's first level in one call; it is "
            "the viewer's opening request and composes the four operations below without changing them. "
            "op=banks lists accepted level banks; "
            "bank_levels takes bank, optional split/difficulty/offset/limit (max100); bank_level takes bank and id. "
            "Bank rows are exact accepted records, not regenerated levels; bank_status distinguishes partial/complete. "
            "op=info is the same metadata on its own, for callers that want nothing else: "
            "it reports actions, palette, supported levels/difficulties and agent.loaded/reason. "
            "info may lazily load the configured policy checkpoint. op=generate accepts seed "
            "(default 0) and difficulty (default 1); op=shipped requires zero-based index 0..6. "
            "Both return level, frame and status. Retain level unchanged and your own actions list. "
            "op=play, oracle and agent require level and accept actions (default []). Every call "
            "replays the complete action history; there is no server-side game session. Actions "
            "are 1=up, 2=down, 3=left, 4=right; at most 2048. Undo by dropping the last action, "
            "reset with an empty list. play returns frames, frame and status. Frames are arrays "
            "of palette-index rows; status includes completion information. oracle returns "
            "action, available, remaining and reason: unavailable/search-limited is not proof "
            "that the level is unsolvable. agent returns action, probabilities, loaded, reason "
            "and sometimes status. Append a non-null advised action to your history before "
            "calling play. Without usable weights, agent returns loaded=false and action=null; "
            "environment play still works. Level generation, replay and oracle search consume "
            "CPU/memory; oracle search is capped by Pebby. Unexpected fields, malformed levels "
            "or out-of-range values fail validation. The schemas describe the operation envelope; "
            "Pebby performs the detailed generated-level geometry checks."
        ),
        input_schema={
            "type": "object",
            "description": "One stateless Pebby operation; send no extra fields.",
            "properties": {
                "op": {
                    "type": "string",
                    "enum": ["boot", "info", "generate", "shipped", "play", "oracle", "agent", "banks", "bank_levels", "bank_level"],
                    "description": "Operation to perform.",
                },
                "bank": {"type": "string", "description": "Allowlisted bank id from banks."},
                "id": {"type": "string", "description": "Stable accepted row id from bank_levels."},
                "split": {"type": "string", "enum": ["train", "validation"], "description": "Bank split; default train."},
                "offset": {"type": "integer", "minimum": 0, "maximum": 1000000, "description": "Filtered row offset; default zero."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Page size; default50, maximum100."},
                "seed": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 4294967295,
                    "description": "Deterministic generation seed; default 0.",
                },
                "difficulty": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 7,
                    "description": "Reference-calibrated LS20 tier 1..7; default 1.",
                },
                "index": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 6,
                    "description": "Zero-based shipped level index.",
                },
                "level": {
                    "type": "object",
                    "description": "Return the level object from generate/shipped unchanged, or use {shipped: index}. Pebby validates generated geometry.",
                    "minProperties": 1,
                },
                "actions": {
                    "type": "array",
                    "maxItems": 2048,
                    "items": {"type": "integer", "minimum": 1, "maximum": 4},
                    "description": "Full action history, not just the newest action; default empty.",
                },
            },
            "required": ["op"],
            "additionalProperties": False,
            "oneOf": [
                {"properties": {"op": {"const": "boot", "description": "Fetch the viewer's whole opening screen."}},
                 "propertyNames": {"enum": ["op", "limit"]}},
                {"properties": {"op": {"const": "banks", "description": "List allowlisted accepted banks."}}, "propertyNames": {"enum": ["op"]}},
                {"properties": {"op": {"const": "bank_levels", "description": "List accepted rows in a bank."}}, "required": ["bank"],
                 "propertyNames": {"enum": ["op", "bank", "split", "difficulty", "offset", "limit"]}},
                {"properties": {"op": {"const": "bank_level", "description": "Render an exact accepted bank row."}}, "required": ["bank", "id"],
                 "propertyNames": {"enum": ["op", "bank", "id"]}},
                {
                    "properties": {"op": {"const": "info", "description": "Inspect capabilities."}},
                    "maxProperties": 1,
                },
                {
                    "properties": {"op": {"const": "generate", "description": "Generate a level."}},
                    "propertyNames": {"enum": ["op", "seed", "difficulty"]},
                },
                {
                    "properties": {
                        "op": {"const": "shipped", "description": "Load a shipped level."}
                    },
                    "required": ["index"],
                    "propertyNames": {"enum": ["op", "index"]},
                },
                {
                    "properties": {
                        "op": {
                            "enum": ["play", "oracle", "agent"],
                            "description": "Replay or request advice.",
                        }
                    },
                    "required": ["level"],
                    "propertyNames": {"enum": ["op", "level", "actions"]},
                },
            ],
        },
        output_schema={
            "type": "object",
            "description": "Operation-dependent result. info returns metadata; generate/shipped return level/frame/status; play returns frames/frame/status; oracle/agent return advice or an explicit unavailable reason.",
            "minProperties": 1,
            "properties": {
                "info": {"type": "object", "description": "boot only: the same object op=info returns."},
                "page": {"type": ["object", "null"],
                         "description": "boot only: the same object op=bank_levels returns, or null when no bank is readable."},
                "banks": {"type": "array", "description": "Allowlisted banks with accepted/requested counts and publication status."},
                "levels": {"type": "array", "description": "Filtered accepted-row summaries with stable ids."},
                "bank": {"type": "string", "description": "Allowlisted bank identifier."},
                "bank_status": {"type": "string", "enum": ["partial", "complete", "unavailable"],
                                "description": "Bank publication status, separate from game status."},
                "id": {"type": "string", "description": "Stable accepted bank row id."},
                "total": {"type": "integer", "description": "Number of accepted rows matching the filters."},
                "offset": {"type": "integer", "description": "Filtered pagination offset."},
                "level": {
                    "type": ["object", "null"],
                    "description": "Opaque level specification to preserve for later requests. "
                                   "Under boot this is the whole bank_level result, or null when the bank has no rows.",
                },
                "frame": {"type": "array", "description": "Current palette-index grid, row-major."},
                "frames": {"type": "array", "description": "Rendered frames produced by replay."},
                "status": {
                    "type": "object",
                    "description": "Current game progress and completion information.",
                },
                "action": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                    "maximum": 4,
                    "description": "Advised next action, or null if unavailable/finished.",
                },
                "probabilities": {
                    "type": ["array", "null"],
                    "description": "Learned-policy action probabilities in action-ID order, or null.",
                },
                "loaded": {
                    "type": "boolean",
                    "description": "Whether learned-policy weights are loaded, not a success/quality score.",
                },
                "available": {
                    "type": "boolean",
                    "description": "Whether oracle advice is available for this request.",
                },
                "reason": {
                    "type": ["string", "null"],
                    "description": "Explanation of unavailable advice or failure, otherwise null.",
                },
                "agent": {
                    "type": "object",
                    "description": "info metadata: loaded, parameters, checkpoint and reason.",
                },
            },
        },
        examples=[
            {
                "description": "Illustrative agent-unavailable result. The actual reason depends on the checkpoint; this is not a trained-policy prediction.",
                "input": {"op": "agent", "level": {"shipped": 0}, "actions": []},
                "output": {
                    "action": None,
                    "probabilities": None,
                    "loaded": False,
                    "reason": "No agent checkpoint is available.",
                },
            }
        ],
    )


def make_provider(engine, ui_dir: Path) -> Provider:
    from inference import DEFAULT_CHECKPOINT

    path = engine.agent.path or DEFAULT_CHECKPOINT
    stat = path.stat() if path.is_file() else None
    lock = threading.Lock()

    def dispatch(request):
        # Protect lazy policy initialization as well as the shared model instance.
        with lock:
            try:
                return engine.dispatch(request)
            except ValueError as error:
                # Pebby explicitly classifies geometry/operation failures as bad input.
                raise InputError(str(error)) from None

    return Provider(
        "pebby",
        [
            Model(
                "pebby:latest",
                dispatch,
                interaction(),
                size_bytes=stat.st_size if stat else 0,
                parameter_size=str(engine.agent.parameters or 0),
                quantization="F32",
                modified_at=datetime.fromtimestamp(stat.st_mtime, UTC).isoformat()
                if stat
                else datetime.now(UTC).isoformat(),
            )
        ],
        ui=CustomUI(ui_dir),
    )
