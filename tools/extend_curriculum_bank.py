"""Extend verified generated banks with durable, resumable, ordered appends.

Run from the repository root with ``python -m tools.extend_curriculum_bank``.
Two workers maximum; each candidate keeps the curriculum's complete-search and
real-engine checks. Failed seeds stop the run without discarding saved rows.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile


def read_bank(path, *, repair_tail=False):
    """Read complete rows; recover only an interrupted, unterminated final row."""
    path = Path(path)
    content = path.read_bytes()
    rows, offset = [], 0
    lines = content.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if repair_tail and index == len(lines) - 1 and not line.endswith(b"\n"):
                with path.open("r+b") as handle:
                    handle.truncate(offset)
                    handle.flush()
                    os.fsync(handle.fileno())
                print(f"Recovered interrupted final write at byte {offset}: {path}", flush=True)
                break
            raise ValueError(f"invalid bank row {index + 1}: {path}") from None
        rows.append(value)
        offset += len(line)
    return rows


def check_row(row, seed, difficulty):
    from pebby.ls20.curriculum import CURRICULUM_VERSION
    from pebby.ls20.generate import GENERATOR_VERSION

    if row.get("seed") != seed or row.get("difficulty") != difficulty:
        raise ValueError(f"seed/difficulty collision: expected {seed}/d{difficulty}")
    if (row.get("generator_version") != GENERATOR_VERSION
            or row.get("curriculum_version") != CURRICULUM_VERSION
            or row.get("search_truncated") is not False or row.get("engine_verified") is not True
            or row.get("context_engine_verified") is not True
            or row.get("context_index") != seed % 7
            or row.get("verification_level_index") != seed % 7
            or row.get("verification_match_hint") is not (seed % 7 == 0)
            or row.get("engine_win") is not True or row.get("replay_lives") != 3
            or row.get("levels_completed") != 1
            or not row.get("solution") or row.get("optimal_actions") != len(row["solution"])
            or row.get("context_optimal_actions") != row.get("optimal_actions")
            or row.get("context_solution") != row.get("solution")):
        raise ValueError(f"seed {seed} lacks complete curriculum verification")


def prepare_bank(source, target, total):
    """Validate source and target prefixes before generating or appending anything."""
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve():
        raise ValueError("source and output must be different paths")
    original = read_bank(source)
    if not original or total < len(original):
        raise ValueError("source must be nonempty and total must include its existing rows")
    offset = original[0]["seed"]
    if offset // 1_000_000 != (offset + total - 1) // 1_000_000:
        raise ValueError("extension would cross a train/validation/test seed boundary")
    for index, row in enumerate(original):
        check_row(row, offset + index, index % 5 + 1)
    if target.exists():
        existing = read_bank(target, repair_tail=True)
        if len(existing) < len(original) or existing[:len(original)] != original:
            raise ValueError("output does not contain the exact verified source prefix")
        if len(existing) > total:
            raise ValueError("output already exceeds requested total; refusing to truncate it")
        for index, row in enumerate(existing):
            check_row(row, offset + index, index % 5 + 1)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        content = source.read_bytes()
        if not content.endswith(b"\n"):
            content += b"\n"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.",
                                             delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        existing = original
    # A valid but unterminated JSON row survived its write; add its separator.
    with target.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) != b"\n":
            handle.seek(0, os.SEEK_END)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    return offset, len(existing)


def generate_job(job):
    from pebby.ls20.curriculum import generate_legacy_level

    seed, difficulty = job
    return generate_legacy_level(seed, difficulty)


def extend_bank(source, target, total, *, workers=2, generator=None, progress_every=25):
    """Append only contiguous verified rows; a failed candidate leaves a checkpoint."""
    target = Path(target)
    if workers not in (1, 2):
        raise ValueError("workers must be one or two")
    target.parent.mkdir(parents=True, exist_ok=True)
    # Keep the lock inode stable across resumptions; unlinking it introduces a
    # race between a waiting writer and another process creating a new inode.
    with target.with_suffix(target.suffix + ".lock").open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another writer holds the output lock: {target}") from None
        offset, completed = prepare_bank(source, target, total)
        print(f"Resuming {target}: {completed}/{total}, next seed {offset + completed}", flush=True)
        jobs = [(offset + index, index % 5 + 1) for index in range(completed, total)]
        make = generator or generate_job
        pool = None
        try:
            if workers == 1:
                results = map(make, jobs)
            else:
                # Preloaded modules survive fork: every worker uses the same
                # reviewed planner even if its source changes during this run.
                pool = ProcessPoolExecutor(max_workers=workers,
                                           mp_context=multiprocessing.get_context("fork"))
                results = _bounded_results(pool, make, jobs, workers)
            with target.open("ab") as handle:
                for (seed, difficulty), row in zip(jobs, results):
                    check_row(row, seed, difficulty)
                    handle.write(json.dumps(row, separators=(",", ":")).encode() + b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    completed += 1
                    if completed % progress_every == 0 or completed == total:
                        print(f"{target}: {completed}/{total} saved; seed={seed}", flush=True)
        except BaseException:
            print(f"STOPPED: {completed}/{total} rows preserved at {target}; "
                  f"resume with the same command (next seed {offset + completed})", file=sys.stderr, flush=True)
            raise
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
        return completed


def _bounded_results(pool, make, jobs, workers):
    """At most two in-flight candidates, in seed order (Python 3.12/3.13)."""
    iterator = iter(jobs)
    pending = []
    for _ in range(workers):
        job = next(iterator, None)
        if job is not None:
            pending.append(pool.submit(make, job))
    while pending:
        future = pending.pop(0)
        yield future.result()
        job = next(iterator, None)
        if job is not None:
            pending.append(pool.submit(make, job))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", action="store_true", help="Explicit extension of historical five-tier banks")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--planner-sha256", help="Require this reviewed planner before loading it")
    parser.add_argument("--planner-source", type=Path,
                        help="Load an exact reviewed source snapshot; requires --planner-sha256")
    args = parser.parse_args()
    if not args.legacy:
        parser.error("historical bank extension requires --legacy; use regenerate_mechanism_banks for seven-tier generation")
    if args.total < 1:
        parser.error("total must be positive")
    if args.planner_source and not args.planner_sha256:
        parser.error("--planner-source requires --planner-sha256")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    if args.planner_sha256:
        path = args.planner_source or Path(__file__).resolve().parents[1] / "pebby/ls20/plan.py"
        source_bytes = path.read_bytes()
        actual = hashlib.sha256(source_bytes).hexdigest()
        if actual != args.planner_sha256:
            parser.error(f"planner changed before import: expected {args.planner_sha256}, found {actual}")
        if args.planner_source:
            module_spec = importlib.util.spec_from_file_location("pebby.ls20.plan", path)
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_spec.name] = module
            # Execute the bytes whose digest was checked, avoiding a later
            # filesystem read or a stale bytecode cache for the source snapshot.
            exec(compile(source_bytes, str(path), "exec"), module.__dict__)
    # Import before any fork; workers never reload a planner edited mid-run.
    from pebby.ls20 import curriculum  # noqa: F401

    print(f"Bank extension PID {os.getpid()}, workers={args.workers}", flush=True)
    # Report an actual search, not merely whether a native library can load.
    # Loading here also pins the native library before workers can fork.
    probe_env = curriculum.Ls20Env([curriculum.build_level(read_bank(args.source)[0])])
    probe = curriculum.Oracle(curriculum.extract(probe_env))
    print(f"Planner engine={getattr(probe, 'engine', 'reference')}, "
          f"fallback_reason={getattr(probe, 'fallback_reason', None)!r}", flush=True)
    root = Path(__file__).resolve().parents[1]
    source_hashes = {name: hashlib.sha256((root / "pebby/ls20" / name).read_bytes()).hexdigest()
                     for name in ("plan.py", "fastplan.py", "_fastplan.c")
                     if (root / "pebby/ls20" / name).exists()}
    print(f"Planner source hashes: {json.dumps(source_hashes, sort_keys=True)}", flush=True)
    try:
        extend_bank(args.source, args.out, args.total, workers=args.workers)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
