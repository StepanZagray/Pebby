#!/usr/bin/env python3
"""Manage a detached Linux training process: start, status, stop, or resume.

State and append-only logs live beside the run, in .<run-name>.launcher/, so a
fresh training output directory stays empty. Stop requests a graceful checkpoint;
it never escalates to SIGKILL. Epochs on resume mean the total epoch target.
"""
from __future__ import annotations

import argparse
import ctypes
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "tools" / "train_multigame.py"


def _libc_call(name: str, *arguments) -> int:
    # uv's portable Python builds may omit Python's pidfd wrappers even on a
    # supporting Linux host. Use the libc API without falling back to racy kill.
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, name, None)
    if function is None:
        raise OSError(f"{name} is unavailable; safe process signalling requires Linux pidfd support")
    result = function(*arguments)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def pidfd_open(pid: int) -> int:
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    return _libc_call("pidfd_open", ctypes.c_int(pid), ctypes.c_uint(0))


def pidfd_signal(fd: int, sig: int) -> None:
    if hasattr(signal, "pidfd_send_signal"):
        signal.pidfd_send_signal(fd, sig)
    else:
        _libc_call("pidfd_send_signal", ctypes.c_int(fd), ctypes.c_int(sig),
                   ctypes.c_void_p(), ctypes.c_uint(0))


def state_dir(run_dir: Path) -> Path:
    run_dir = run_dir.absolute()
    return run_dir.parent / f".{run_dir.name}.launcher"


@contextmanager
def locked(run_dir: Path):
    directory = state_dir(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield directory


def read_state(directory: Path) -> dict | None:
    path = directory / "state.json"
    return json.loads(path.read_text()) if path.exists() else None


def process_identity(pid: int) -> dict | None:
    try:
        proc = Path(f"/proc/{pid}")
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {
            "starttime": fields[19],
            "command": [s.decode() for s in (proc / "cmdline").read_bytes().split(b"\0") if s],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }
    except OSError:
        # A reused PID may belong to another user and be unreadable. It cannot
        # be identified as our recorded child and must never be signalled.
        return None


def active(state: dict | None) -> bool:
    return bool(state and process_identity(state["pid"]) == state["identity"])


def checkpoint_info(run_dir: Path) -> dict | None:
    path = run_dir / "latest.pt"
    if not path.is_file():
        return None
    stat = path.stat()
    return {"path": str(path), "target": str(path.resolve()), "bytes": stat.st_size,
            "mtime_unix": stat.st_mtime}


def status(run_dir: Path) -> dict:
    state = read_state(state_dir(run_dir))
    return {"status": "running" if active(state) else "stopped", "run_dir": str(run_dir),
            "pid": state["pid"] if state else None,
            "log": state["log"] if state else None,
            "checkpoint": checkpoint_info(run_dir)}


def launch(run_dir: Path, command: list[str], *, cwd: Path = ROOT,
           startup_wait: float = 0.5) -> dict:
    """Launch exactly command; exposed separately to test with a tiny fixture."""
    with locked(run_dir) as directory:
        prior = read_state(directory)
        if active(prior):
            raise ValueError(f"run already active as PID {prior['pid']}")
        log_path = directory / "training.log"
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        # Unique argv is unnecessary: boot + PID start ticks + complete argv identify
        # this process. A separate session avoids terminal hangup on shell logout.
        with log_path.open("ab", buffering=0) as log:
            log.write((f"\n--- launch {uuid.uuid4()} ---\n").encode())
            process = subprocess.Popen(command, cwd=cwd, env=environment,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       start_new_session=True)
        temporary = directory / "state.json.tmp"
        try:
            time.sleep(startup_wait)
            # /proc/cmdline can still be empty during exec even after Popen has
            # returned. Persist only a fully initialized, expected command line.
            deadline = time.monotonic() + 1.0
            while True:
                code = process.poll()
                if code is not None:
                    raise ValueError(f"trainer exited during startup (code {code}); inspect {log_path}")
                identity = process_identity(process.pid)
                if identity is not None and identity["command"] == command:
                    break
                if time.monotonic() >= deadline:
                    raise ValueError(f"trainer command identity could not be verified; inspect {log_path}")
                time.sleep(0.01)
            state = {"pid": process.pid, "identity": identity, "command": command,
                     "cwd": str(cwd), "log": str(log_path), "run_dir": str(run_dir)}
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(directory / "state.json")
            return status(run_dir)
        except BaseException:
            # This is rollback of an unsuccessful launch, not the stop command.
            # Popen owns the unreaped child, so its PID cannot be reused here.
            # Do not leave untracked training running if state persistence fails.
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            else:
                process.wait()
            temporary.unlink(missing_ok=True)
            raise


def stop(run_dir: Path, timeout: float = 60.0) -> dict:
    with locked(run_dir) as directory:
        state = read_state(directory)
        if not active(state):
            return status(run_dir)
        # pidfd pins the specific process, preventing a PID-reuse race between
        # checking /proc and delivering the signal.
        try:
            fd = pidfd_open(state["pid"])
        except ProcessLookupError:
            return status(run_dir)
        try:
            if active(state):
                try:
                    pidfd_signal(fd, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        finally:
            os.close(fd)
    deadline = time.monotonic() + timeout
    while active(state) and time.monotonic() < deadline:
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    result = status(run_dir)
    if active(state):
        result["status"] = "stopping"
        result["message"] = "SIGTERM requested; trainer is finishing its current game. No force kill sent."
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("start", "status", "stop", "resume"):
        sub = commands.add_parser(name)
        sub.add_argument("--run-dir", required=True, type=Path)
        if name == "start":
            sub.add_argument("training_args", nargs=argparse.REMAINDER)
        elif name == "stop":
            sub.add_argument("--timeout", type=float, default=60.0)
        elif name == "resume":
            sub.add_argument("--epochs", required=True, type=int, help="total epoch target")
            sub.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    try:
        if args.action == "start":
            extra = args.training_args
            if extra[:1] == ["--"]:
                extra = extra[1:]
            if any(a.split("=", 1)[0] in {"--out-dir", "--resume"} for a in extra):
                raise ValueError("start owns --out-dir; use the resume command to continue a run")
            # Resolve relative manifest/initialization paths from the caller's cwd.
            result = launch(run_dir, [sys.executable, "-u", str(TRAINER),
                                     "--out-dir", str(run_dir), *extra], cwd=Path.cwd())
        elif args.action == "resume":
            if args.epochs < 1:
                raise ValueError("--epochs must be positive")
            if checkpoint_info(run_dir) is None:
                raise ValueError(f"no checkpoint at {run_dir / 'latest.pt'}")
            prior = read_state(state_dir(run_dir))
            command = [sys.executable, "-u", str(TRAINER), "--resume",
                       str(run_dir / "latest.pt"), "--epochs", str(args.epochs)]
            if args.device:
                command += ["--device", args.device]
            result = launch(run_dir, command, cwd=Path(prior["cwd"]) if prior else ROOT)
        elif args.action == "stop":
            if args.timeout < 0:
                raise ValueError("--timeout must be nonnegative")
            result = stop(run_dir, args.timeout)
        else:
            result = status(run_dir)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
