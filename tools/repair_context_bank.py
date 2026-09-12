"""Re-prove a generated bank in training contexts and publish a new copy only.

The source and its generator versions are preserved. A context_proof_version
marks repaired proof metadata; repair does not claim new geometry was generated.
Any failed row prevents publication. No existing output can be overwritten.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from pebby.ls20 import names
from pebby.ls20.env import Ls20Scenario
from pebby.ls20.generate import FORMAT, build_level
from pebby.ls20.layout import extract
from pebby.ls20.plan import Oracle, simulate
from pebby.ls20.generation_quality import route_budget_slack


PROOF_VERSION = "context-complete-v1"


def repair_row(spec, *, search_limit=600_000):
    """Recompute only the proof, with a complete oracle and real engine replay."""
    if spec.get("format") != FORMAT or not isinstance(spec.get("seed"), int):
        raise ValueError("expected a seeded generated level spec")
    context = spec["seed"] % 7
    if context == 0 and spec.get("launchers"):
        raise ValueError("context-zero launcher hints are outside the oracle state")
    env = Ls20Scenario(build_level(spec), context)
    layout = extract(env)
    oracle = Oracle(layout, limit=search_limit)
    if oracle.truncated or not oracle.solvable:
        raise ValueError("incomplete or unsolvable contextual search")
    solution = oracle.solution(seed=spec["seed"])
    state, result = oracle.start, None
    minimum_slack = layout.max_steps // layout.step_cost
    mechanics = {"moving_cycler": False, "launcher": False, "refill": False}
    for action in solution or ():
        before = state
        index = names.ACTION_IDS.index(action)
        state, outcome = simulate(layout, state, index, oracle.refills)
        minimum_slack = min(minimum_slack, route_budget_slack(
            layout, before, state, action=index, outcome=outcome))
        mechanics["moving_cycler"] |= before[1:4] != state[1:4] and (
            state[0] in layout.moving_cyclers[state[7]])
        mechanics["launcher"] |= outcome == "launched"
        mechanics["refill"] |= before[5] != state[5]
        result = env.perform(action)
        if env.lives() != 3:
            raise ValueError("contextual engine replay lost a life")
    if result is None or not result.won or env.levels_completed != 1:
        raise ValueError("contextual engine replay did not win")
    # Old route-dependent annotations cannot survive a different solution.
    cleaned = {key: value for key, value in spec.items() if key not in (
        "proof", "solution_mechanics", "minimum_route_slack_moves",
        "distractor_count", "distractor_cells", "non_required_distractor_count")}
    return {**cleaned, "context_proof_version": PROOF_VERSION,
            "solution": solution, "context_solution": solution,
            "optimal_actions": oracle.optimal_actions,
            "context_optimal_actions": oracle.optimal_actions,
            "context_index": context, "training_context_index": context,
            "verification_level_index": context,
            "verification_match_hint": layout.match_hint,
            "engine_verified": True, "context_engine_verified": True,
            "engine_win": True, "replay_lives": env.lives(),
            "levels_completed": env.levels_completed,
            "search_limit": search_limit, "search_truncated": False,
            "reachable_states": oracle._reachable,
            "slack_moves": state[6] // layout.step_cost,
            "minimum_slack_moves": minimum_slack,
            "solution_mechanics": mechanics}


def repair_bank(source, target, *, search_limit=600_000, max_levels=1000):
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve() or target.exists() or target.is_symlink():
        raise ValueError("output must be a new path distinct from the source")
    if not 1 <= search_limit <= 600_000 or max_levels < 1:
        raise ValueError("search_limit must be 1..600000 and max_levels positive")
    content = source.read_bytes()
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    if not rows or len(rows) > max_levels:
        raise ValueError("source is empty or exceeds the explicit max_levels bound")
    report = {"source_sha256": hashlib.sha256(content).hexdigest(),
              "source_rows": len(rows), "original_claimed_context_verified": sum(
                  row.get("context_engine_verified") is True for row in rows),
              "reverified": 0, "changed_solutions": 0, "failures": [],
              "published": False, "context_proof_version": PROOF_VERSION}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=target.parent,
                                         prefix=f".{target.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            for index, row in enumerate(rows):
                try:
                    fixed = repair_row(row, search_limit=search_limit)
                except (ValueError, RuntimeError) as error:
                    report["failures"].append({"row": index + 1, "seed": row.get("seed"),
                                               "reason": str(error)})
                    continue
                report["reverified"] += 1
                report["changed_solutions"] += row.get("solution") != fixed["solution"]
                handle.write(json.dumps(fixed, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if not report["failures"]:
            # Atomic publication with no overwrite, including a racing writer.
            os.link(temporary, target)
            descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            report["published"] = True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return report


def repair_mismatched_bank(source, target, *, search_limit=600_000, max_seconds=120):
    """Repair known stored/context distance contradictions, preserving other bytes.

    This deliberately does not certify untouched routes or whole-bank optimality.
    The report contains independent replay outcomes and complete new proof rows.
    """
    started = time.monotonic()
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve() or target.exists() or target.is_symlink():
        raise ValueError("output must be a new path distinct from the source")
    if not 1 <= search_limit <= 600_000 or max_seconds <= 0:
        raise ValueError("search_limit must be 1..600000 and max_seconds positive")
    content = source.read_bytes()
    lines = content.splitlines(keepends=True)
    report = {"source": str(source), "output": str(target),
              "source_sha256": hashlib.sha256(content).hexdigest(),
              "source_rows": sum(bool(line.strip()) for line in lines),
              "selection": "stored/context optimal_actions disagree",
              "scope": "Selected routes replayed and fully re-proved; untouched rows retain original proof claims and are not newly certified.",
              "selected": 0, "old_routes_failed_replay": 0, "repaired": 0,
              "untouched_rows": 0, "failures": [], "corrections": [], "published": False}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent,
                                         prefix=f".{target.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            for line in lines:
                if not line.strip():
                    handle.write(line)
                    continue
                spec = json.loads(line)
                if (spec.get("context_optimal_actions") is None
                        or spec.get("optimal_actions") == spec.get("context_optimal_actions")):
                    handle.write(line)
                    report["untouched_rows"] += 1
                    continue
                report["selected"] += 1
                try:
                    if time.monotonic() - started > max_seconds:
                        raise RuntimeError("diagnostic time bound exceeded")
                    replay = Ls20Scenario(build_level(spec), spec["seed"] % 7)
                    result = None
                    for action in spec.get("solution", ()):
                        result = replay.perform(action)
                    old_won = result is not None and result.won
                    old_lives = replay.lives()
                    report["old_routes_failed_replay"] += not old_won or old_lives != 3
                    fixed = repair_row(spec, search_limit=search_limit)
                except (ValueError, RuntimeError) as error:
                    report["failures"].append({"seed": spec.get("seed"), "reason": str(error)})
                    continue
                report["repaired"] += 1
                report["corrections"].append({"seed": spec["seed"],
                                               "old_optimal_actions": spec.get("optimal_actions"),
                                               "old_context_optimal_actions": spec.get("context_optimal_actions"),
                                               "old_route_won": old_won,
                                               "old_route_lives": old_lives,
                                               "repaired_row": fixed})
                separator = b"\n" if line.endswith(b"\n") else b""
                handle.write(json.dumps(fixed, separators=(",", ":")).encode() + separator)
            handle.flush()
            os.fsync(handle.fileno())
        if not report["failures"]:
            report["output_sha256"] = hashlib.sha256(temporary.read_bytes()).hexdigest()
            os.link(temporary, target)
            descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            report["published"] = True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--search-limit", type=int, default=600_000)
    parser.add_argument("--max-levels", type=int, default=1000)
    parser.add_argument("--mismatched-only", action="store_true",
                        help="Repair only stored/context distance contradictions; preserve other rows verbatim")
    parser.add_argument("--max-seconds", type=float, default=120)
    args = parser.parse_args()
    if args.mismatched_only:
        report = repair_mismatched_bank(args.source, args.out, search_limit=args.search_limit,
                                       max_seconds=args.max_seconds)
    else:
        report = repair_bank(args.source, args.out, search_limit=args.search_limit,
                             max_levels=args.max_levels)
    print(json.dumps(report, indent=2))
    if not report["published"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
