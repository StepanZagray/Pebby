"""Audit generated source banks using the collector's exact context verifier."""

from pebby.ls20.provenance import generated_context, difficulty_provenance

import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import tempfile

from tools.extend_curriculum_bank import _bounded_results


SOURCES = ("data/levels-train.jsonl", "data/levels-validation.jsonl",
           "data/ls20-curriculum-large-train.jsonl",
           "data/ls20-curriculum-large-validation.jsonl")
CODE = ("pebby/agent/world_data.py", "pebby/ls20/env.py", "pebby/ls20/plan.py",
        "pebby/ls20/fastplan.py", "pebby/ls20/_fastplan.c", "pebby/ls20/generate.py")
UNSUPPORTED = "context-zero launcher can leave pending hint outside oracle state"


def digest(content):
    return hashlib.sha256(content).hexdigest()


def gameplay_hash(spec):
    gameplay = {"walls": sorted(spec["walls"]), "start": spec["start"],
                "start_triple": spec["start_triple"],
                "goals": sorted(spec["goals"], key=lambda x: x["cell"]),
                "cyclers": sorted(spec["cyclers"], key=lambda x: (x["cell"], x["kind"])),
                "rails": sorted([sorted(r["cells"]) for r in spec.get("rails", [])]),
                "launchers": sorted(spec.get("launchers", []), key=lambda x: x["cell"]),
                "refills": sorted(spec["refills"]), "step_counter": spec["step_counter"],
                "step_cost": spec["step_cost"], "fog": spec["fog"]}
    return digest(json.dumps(gameplay, sort_keys=True, separators=(",", ":")).encode())


def worker_started():
    print(f"Context audit worker PID {os.getpid()}", flush=True)


def audit_one(job):
    from pebby.agent.world_data import verified_context

    row, spec, search_limit = job
    try:
        env, oracle, metadata = verified_context(spec, context_index=row["context_index"],
                                                 search_limit=search_limit)
        if env is None:
            return {**row, "status": "failed", "reason": metadata["excluded"],
                    "optimal_actions": None, "engine_win": False, "replay_lives": None}
        if (oracle.truncated or not oracle.solvable
                or env.level_index != row["context_index"]
                or metadata.get("context_engine_verified") is not True
                or metadata["context_optimal_actions"] != oracle.optimal_actions):
            raise AssertionError("context verifier returned an inconsistent success")
        # verified_context returns success only after its actual replay reports
        # WIN, one completed level and exactly three lives. These fields record
        # that checked contract, not a second potentially different replay.
        return {**row, "status": "verified", "optimal_actions": oracle.optimal_actions,
                "reachable_states": oracle._reachable, "search_truncated": False,
                "oracle_backend": metadata["oracle_backend"], "engine_win": True,
                "replay_lives": 3, "levels_completed": 1}
    except Exception as error:
        return {**row, "status": "failed", "reason": f"{type(error).__name__}: {error}",
                "optimal_actions": None, "engine_win": False, "replay_lives": None}


def atomic_json(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, separators=(",", ":"), sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def summarize(rows):
    summary = {}
    for split in ("train", "validation"):
        selected = [row for row in rows if row["split"] == split]
        status = collections.Counter(row["status"] for row in selected)
        verified = [row for row in selected if row["status"] == "verified"]
        summary[split] = {"source_rows": len(selected), **dict(status),
                          "distinct_source_seeds": len({r["seed"] for r in selected}),
                          "distinct_source_gameplays": len({r["gameplay_sha256"] for r in selected}),
                          "distinct_verified_gameplays": len({r["gameplay_sha256"] for r in verified}),
                          "verified_difficulty_counts": dict(collections.Counter(r["difficulty"] for r in verified))}
    for status in (None, "verified"):
        selected = rows if status is None else [row for row in rows if row["status"] == status]
        train = {r["gameplay_sha256"] for r in selected if r["split"] == "train"}
        validation = {r["gameplay_sha256"] for r in selected if r["split"] == "validation"}
        train_seeds = {r["seed"] for r in selected if r["split"] == "train"}
        validation_seeds = {r["seed"] for r in selected if r["split"] == "validation"}
        summary["all_sources" if status is None else "verified_sources"] = {
            "rows": len(selected), "unique_gameplays": len(train | validation),
            "train_validation_gameplay_overlap": sorted(train & validation),
            "train_validation_seed_overlap": sorted(train_seeds & validation_seeds)}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/world-full-context-validity.json"))
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--search-limit", type=int, default=600_000)
    parser.add_argument("--resume-after-eligibility-guard", action="store_true",
                        help="Retain prior replay evidence and apply the new context-zero launcher exclusion")
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    source_hashes, entries = {}, []
    for source in SOURCES:
        content = Path(source).read_bytes()
        source_hashes[source] = digest(content)
        for spec in (json.loads(line) for line in content.splitlines()):
            row = {"source": source, "split": "validation" if "validation" in source else "train",
                   "seed": spec["seed"], **difficulty_provenance(spec),
                   "context_index": generated_context(spec), "gameplay_sha256": gameplay_hash(spec),
                   "status": "skipped_source" if spec.get("search_truncated") else "pending"}
            if row["status"] == "skipped_source":
                row.update(reason="truncated source proof", optimal_actions=None,
                           source_search_truncated=True)
            elif row["context_index"] == 0 and spec.get("launchers"):
                row.update(status="excluded_unsupported", reason=UNSUPPORTED,
                           optimal_actions=None, engine_win=False, replay_lives=None)
            entries.append((row, spec))
    code_hashes = {source: digest(Path(source).read_bytes()) for source in CODE}
    run_id = digest(json.dumps(code_hashes, sort_keys=True).encode())[:16]
    manifest = {"format": "pebby.world-full-context-validity.v1", "source_hashes": source_hashes,
                "code_hashes": code_hashes, "context_rule": "difficulty - 1 for ls20-reference-v1; seed % 7 for legacy", "search_limit": args.search_limit,
                "verification": "world_data.verified_context: complete oracle, engine WIN, one level, lives=3",
                "eligibility_exclusion": UNSUPPORTED,
                "active_verification_run": run_id, "verification_runs": {run_id: code_hashes},
                "created": datetime.now(timezone.utc).isoformat()}
    rows = [row for row, _ in entries]
    slots = {(row["source"], row["seed"]): index for index, row in enumerate(rows)}
    if len(slots) != len(rows):
        raise ValueError("duplicate source/seed pairs in input banks")
    journal = args.out.with_suffix(".rows.jsonl")
    if args.out.exists():
        previous = json.loads(args.out.read_text())
        for key in ("source_hashes", "search_limit"):
            if previous[key] != manifest[key]:
                raise ValueError(f"cannot resume changed audit {key}; choose a new output")
        if previous["code_hashes"] != code_hashes:
            changed = {key for key in code_hashes if previous["code_hashes"].get(key) != code_hashes[key]}
            if not args.resume_after_eligibility_guard or changed != {"pebby/agent/world_data.py"}:
                raise ValueError(f"cannot resume changed verifier code: {sorted(changed)}")
        previous_run = previous.get("active_verification_run") or digest(
            json.dumps(previous["code_hashes"], sort_keys=True).encode())[:16]
        manifest["verification_runs"].update(previous.get("verification_runs", {}))
        manifest["verification_runs"][previous_run] = previous["code_hashes"]
        prior_rows = {(row["source"], row["seed"]): row for row in previous["levels"]}
        if journal.exists():
            with journal.open("rb+") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        handle.truncate(offset)
                        break
                    result = json.loads(line)
                    key = (result["source"], result["seed"])
                    slot = slots[key]
                    if rows[slot]["status"] == "excluded_unsupported":
                        rows[slot]["initial_route_previously_verified"] = result["status"] == "verified"
                        continue
                    result.setdefault("verification_run", prior_rows.get(key, {}).get(
                        "verification_run", previous_run))
                    rows[slot] = result
    elif journal.exists():
        raise ValueError("orphaned audit journal; choose a new output")

    def checkpoint(status):
        atomic_json(args.out, {**manifest, "status": status,
                               "summary": summarize(rows), "levels": rows})

    for row in rows:
        if row["status"] == "pending":
            row["verification_run"] = run_id
    jobs = [(rows[slots[(row["source"], row["seed"])]], spec, args.search_limit)
            for row, spec in entries if rows[slots[(row["source"], row["seed"])]] ["status"] == "pending"]
    checkpoint("running")
    print(f"Audit PID {os.getpid()}: {len(jobs)} pending eligible levels; "
          f"{sum(r['status'] == 'skipped_source' for r in rows)} source exclusions", flush=True)
    # Preload before fork to pin this verifier and planner for every worker.
    from pebby.agent import world_data  # noqa: F401

    completed = 0
    with journal.open("ab") as handle:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=multiprocessing.get_context("fork"),
                                 initializer=worker_started) as pool:
            for result in _bounded_results(pool, audit_one, jobs, args.workers):
                rows[slots[(result["source"], result["seed"])]] = result
                handle.write(json.dumps(result, separators=(",", ":")).encode() + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
                completed += 1
                if result["status"] != "verified":
                    print(f"FAILURE {json.dumps(result, sort_keys=True)}", flush=True)
                if completed % 200 == 0:
                    checkpoint("running")
                    failures = sum(row["status"] == "failed" for row in rows)
                    print(f"{completed}/{len(jobs)} newly audited; failures={failures}", flush=True)
    checkpoint("complete")
    print(json.dumps(summarize(rows), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
