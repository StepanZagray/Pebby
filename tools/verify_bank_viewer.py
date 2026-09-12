#!/usr/bin/env python3
"""Run one of the viewer browser tests in a proved private headless display.

The browser test is deliberately kept in JavaScript.  This module owns the
process boundary around it: a copied, capability-free Sway is started in a
headless pixman bubblewrap namespace, the server is a disposable loopback
instance, and Chromium is started in a second bubblewrap namespace which only
shares the network namespace for CDP and the test server.

This command does not use or restart the application's live server.  It is
safe to import for tests; no process is started until ``main`` is called.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
from typing import Iterable
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
TEST = TESTS / "ui_bank_viewer.cjs"
SWAY = Path("/usr/bin/sway")
SWAYMSG = Path("/usr/bin/swaymsg")
GRIM = Path("/usr/bin/grim")
BWRAP = Path("/usr/bin/bwrap")
CHROMIUM = Path("/usr/bin/chromium")
NODE = Path(shutil.which("node") or "/usr/bin/node")
PLAYWRIGHT = Path("/tmp/pebby-uitest/node_modules/playwright-core")

FORBIDDEN_ENV = {
    "WAYLAND_DISPLAY",
    "WAYLAND_SOCKET",
    "DISPLAY",
    "HYPRLAND_INSTANCE_SIGNATURE",
    "SWAYSOCK",
    "XDG_SESSION_ID",
    "XDG_VTNR",
    "XDG_SEAT",
    "DBUS_SESSION_BUS_ADDRESS",
    "XDG_ACTIVATION_TOKEN",
    "WLR_DRM_DEVICES",
    "LIBSEAT_BACKEND",
}
FORBIDDEN_PATH_PARTS = (
    "/dev/dri",
    "/dev/input",
    "/dev/uinput",
    "/dev/fb",
    "/dev/tty",
    "seatd",
    "logind",
    "/run/user",
    "/run/dbus",
    "dbus/system_bus",
)


class HarnessError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _write_report(path: Path, report: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_env(*, private_keys: Iterable[str] = (), **values: str) -> dict[str, str]:
    """Construct an allow-listed environment, never inheriting a display bus."""
    env = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "HOME": values.pop("HOME", "/tmp/home"),
        "PWD": values.pop("PWD", "/tmp/home"),
    }
    env.update(values)
    leaked = FORBIDDEN_ENV.difference(private_keys).intersection(env)
    if leaked:
        raise AssertionError(f"forbidden environment keys: {sorted(leaked)}")
    return env


def _reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _proc_ppid(pid: int) -> int | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    end = text.rfind(")")
    if end < 0:
        return None
    fields = text[end + 2 :].split()
    try:
        return int(fields[1])
    except (IndexError, ValueError):
        return None


def _children(root: int) -> list[int]:
    """Return exact descendants of a process by walking /proc parent links."""
    parent_to_children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        ppid = _proc_ppid(pid)
        if ppid is not None:
            parent_to_children.setdefault(ppid, []).append(pid)
    result: list[int] = []
    pending = [root]
    while pending:
        parent = pending.pop()
        for child in parent_to_children.get(parent, ()):
            if child not in result:
                result.append(child)
                pending.append(child)
    return result


def _exe(pid: int) -> str:
    try:
        return os.readlink(f"/proc/{pid}/exe")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return ""


def _cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return []
    return [item.decode(errors="replace") for item in raw.split(b"\0") if item]


def _wait_for(predicate, timeout: float, description: str):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise HarnessError(f"timed out waiting for {description}")


def _start_ticks(pid: int) -> int | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    end = text.rfind(")")
    if end < 0:
        return None
    fields = text[end + 2 :].split()
    try:
        return int(fields[19])
    except (IndexError, ValueError):
        return None


def _proc_state(pid: int) -> str | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    end = text.rfind(")")
    if end < 0:
        return None
    fields = text[end + 2 :].split()
    return fields[0] if fields else None


def _alive(pid: int) -> bool:
    # A zombie has no running process to terminate and should not be reported
    # as a cleanup leak once its parent has been reaped.
    return _proc_state(pid) not in (None, "Z")


def _kill_exact(
    pids: Iterable[int],
    expected_starts: dict[int, int | None] | None = None,
    *,
    term_timeout: float = 2.0,
) -> dict[str, list[int]]:
    """Terminate only the supplied PIDs, with inner bwraps before outer ones."""
    unique = list(dict.fromkeys(int(pid) for pid in pids if int(pid) > 0))
    inner = [pid for pid in unique if Path(_exe(pid)).name == "bwrap"]
    others = [pid for pid in unique if pid not in inner]
    # A nested bwrap is PID 1 in the private namespace; killing it tears down
    # that namespace.  Do this before its outer bwrap, as required by the
    # isolation proof, then clean any still-visible children explicitly.
    ordered = list(reversed(inner)) + list(reversed(others))
    def matches(pid: int) -> bool:
        if expected_starts is None or pid not in expected_starts:
            return True
        expected = expected_starts[pid]
        current = _start_ticks(pid)
        return expected is None or current == expected

    skipped_reused = [pid for pid in unique if _alive(pid) and not matches(pid)]
    sent_term: list[int] = []
    for pid in ordered:
        if _alive(pid) and matches(pid):
            try:
                os.kill(pid, signal.SIGTERM)
                sent_term.append(pid)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + term_timeout
    while time.monotonic() < deadline and any(_alive(pid) for pid in unique):
        time.sleep(0.03)
    sent_kill: list[int] = []
    for pid in ordered:
        if _alive(pid) and matches(pid):
            try:
                os.kill(pid, signal.SIGKILL)
                sent_kill.append(pid)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + term_timeout
    while time.monotonic() < deadline and any(_alive(pid) for pid in unique):
        time.sleep(0.03)
    return {
        "tracked": unique,
        "sigterm": sent_term,
        "sigkill": sent_kill,
        "skipped_reused": skipped_reused,
        "remaining": [pid for pid in unique if _alive(pid) and matches(pid)],
    }


def _make_runtime(root: Path) -> dict[str, Path]:
    runtime = root / "r"
    runtime.mkdir(mode=0o700)
    for name in ("home", "cache", "config", "data", "browser", "screenshots"):
        (root / name).mkdir(mode=0o700)
    config = root / "sway.conf"
    config.write_text(
        "output HEADLESS-1 mode 1280x900@60Hz\n"
        "xwayland disable\n"
        "default_border none\n"
        "focus_follows_mouse no\n",
        encoding="utf-8",
    )
    return {"root": root, "runtime": runtime, "config": config}


def _copy_sway(root: Path) -> Path:
    """Copy bytes only so cap_sys_nice xattrs cannot follow the system binary."""
    target = root / "sway"
    shutil.copyfile(SWAY, target)
    target.chmod(0o700)
    if _sha256(target) != _sha256(SWAY):
        raise HarnessError("copied Sway bytes differ")
    try:
        if shutil.which("getcap"):
            caps = subprocess.check_output(["getcap", "-n", str(target)], text=True).strip()
            if caps:
                raise HarnessError(f"capabilities copied to Sway: {caps}")
    except subprocess.CalledProcessError:
        pass
    return target


def _common_bwrap(work: Path, *, share_net: bool) -> list[str]:
    # Both clients use only the system runtime trees needed by their binaries.
    # Home/root/sys/var are hidden and /run/tmp are private tmpfs mounts; no
    # command receives the live home or host cache.
    args = [str(BWRAP), "--unshare-all"]
    if share_net:
        args.append("--share-net")
    args += ["--cap-drop", "ALL", "--die-with-parent", "--new-session"]
    for path in ("/usr", "/etc", "/lib", "/lib64", "/bin"):
        args += ["--ro-bind", path, path]
    args += [
        "--tmpfs", "/dev",
        "--dev-bind", "/dev/null", "/dev/null",
        "--dev-bind", "/dev/zero", "/dev/zero",
        "--dev-bind", "/dev/full", "/dev/full",
        "--dev-bind", "/dev/random", "/dev/random",
        "--dev-bind", "/dev/urandom", "/dev/urandom",
        # Explicitly mask tty with /dev/null; do not let bwrap expose a real tty.
        "--dev-bind", "/dev/null", "/dev/tty",
        "--tmpfs", "/dev/shm",
        "--tmpfs", "/run",
        "--tmpfs", "/tmp",
        "--tmpfs", "/var",
        "--tmpfs", "/home",
        "--tmpfs", "/root",
        "--tmpfs", "/sys",
        "--proc", "/proc",
        "--bind", str(work), str(work),
        "--clearenv",
    ]
    return args


def _setenv(command: list[str], env: dict[str, str]) -> list[str]:
    result = list(command)
    clearenv = result.index("--clearenv")
    for key, value in env.items():
        result[clearenv + 1:clearenv + 1] = ["--setenv", key, value]
        clearenv += 3
    return result


def _sway_command(runtime: dict[str, Path], sway_copy: Path) -> tuple[list[str], dict[str, str]]:
    work = runtime["root"]
    private = str(work)
    env = _clean_env(private_keys={"SWAYSOCK"},
        HOME=f"{private}/home",
        PWD=f"{private}/home",
        XDG_RUNTIME_DIR=f"{private}/r",
        XDG_CACHE_HOME=f"{private}/cache",
        XDG_CONFIG_HOME=f"{private}/config",
        XDG_DATA_HOME=f"{private}/data",
        SWAYSOCK=f"{private}/sway.sock",
        WLR_BACKENDS="headless",
        WLR_RENDERER="pixman",
        WLR_HEADLESS_OUTPUTS="1",
        LIBGL_ALWAYS_SOFTWARE="1",
        XKB_DEFAULT_LAYOUT="us",
    )
    command = _setenv(_common_bwrap(work, share_net=False), env) + [
        "--chdir", f"{private}/home", str(sway_copy), "-d", "-c", f"{private}/sway.conf"
    ]
    return command, env


def _browser_command(runtime: dict[str, Path], cdp_port: int) -> tuple[list[str], dict[str, str]]:
    work = runtime["root"]
    private = str(work)
    env = _clean_env(private_keys={"WAYLAND_DISPLAY"},
        HOME=f"{private}/home",
        PWD=f"{private}/home",
        XDG_RUNTIME_DIR=f"{private}/r",
        XDG_CACHE_HOME=f"{private}/cache",
        XDG_CONFIG_HOME=f"{private}/config",
        XDG_DATA_HOME=f"{private}/data",
        WAYLAND_DISPLAY="wayland-1",
        LIBGL_ALWAYS_SOFTWARE="1",
    )
    command = _setenv(_common_bwrap(work, share_net=True), env) + [
        "--chdir", f"{private}/home",
        str(CHROMIUM),
        "--ozone-platform=wayland",
        "--disable-gpu",
        "--disable-gpu-compositing",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--test-type",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        "--disable-sync",
        "--disable-component-update",
        "--disable-breakpad",
        "--disable-crash-reporter",
        "--password-store=basic",
        "--remote-debugging-address=127.0.0.1",
        f"--remote-debugging-port={cdp_port}",
        f"--user-data-dir={private}/browser",
        "--app=about:blank",
    ]
    return command, env


def _sandbox_processes(outer: subprocess.Popen) -> list[int]:
    return [outer.pid] + _children(outer.pid)


def _wait_sway(outer: subprocess.Popen, sway_copy: Path, timeout: float) -> int:
    def find() -> int | None:
        for pid in _children(outer.pid):
            if _exe(pid) == str(sway_copy):
                return pid
        return None

    return _wait_for(find, timeout, "Sway inside the private namespace")


def _read_env(pid: int) -> dict[str, str]:
    raw = Path(f"/proc/{pid}/environ").read_bytes()
    return {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in raw.decode(errors="replace").split("\0")
        if "=" in item
    }


def _fd_targets(pid: int) -> dict[str, str]:
    result: dict[str, str] = {}
    for fd in Path(f"/proc/{pid}/fd").iterdir():
        try:
            result[fd.name] = os.readlink(fd)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result


def _namespace_ids(pid: int) -> dict[str, str]:
    result = {}
    for name in ("user", "mnt", "pid", "net", "ipc", "uts", "cgroup"):
        try:
            result[name] = os.readlink(f"/proc/{pid}/ns/{name}")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return result


def _assert_private_process(pid: int, *, expected_env: dict[str, str], runtime: Path, share_net: bool = False) -> dict:
    env = _read_env(pid)
    private_keys = {key for key in ("SWAYSOCK", "WAYLAND_DISPLAY") if key in expected_env}
    forbidden = FORBIDDEN_ENV.difference(private_keys).intersection(env)
    if forbidden:
        raise HarnessError(f"forbidden environment in pid {pid}: {sorted(forbidden)}")
    for key, value in expected_env.items():
        if env.get(key) != value:
            raise HarnessError(f"pid {pid} environment {key}={env.get(key)!r}, expected {value!r}")
    fds = _fd_targets(pid)
    bad = {
        fd: target for fd, target in fds.items()
        if any(part in target for part in FORBIDDEN_PATH_PARTS)
    }
    if bad:
        raise HarnessError(f"forbidden descriptors in pid {pid}: {bad}")
    root = Path(f"/proc/{pid}/root")
    blocked = [str(root / item) for item in ("dev/dri", "dev/input", "dev/uinput", "dev/fb0", "dev/tty0", "dev/console", "run/user", "run/dbus") if (root / item).exists()]
    if blocked:
        raise HarnessError(f"forbidden paths visible to pid {pid}: {blocked}")
    # /dev/tty is deliberately present only as a null-device mask.
    tty = root / "dev/tty"
    null = root / "dev/null"
    if not tty.exists() or not null.exists() or os.stat(tty).st_rdev != os.stat(null).st_rdev:
        raise HarnessError(f"pid {pid} does not have the required /dev/tty null mask")
    for hidden in (root / "home", root / "sys"):
        if hidden.exists() and any(hidden.iterdir()):
            raise HarnessError(f"pid {pid} can see nonempty {hidden}")
    namespaces = _namespace_ids(pid)
    host_namespaces = _namespace_ids(os.getpid())
    if namespaces.get("mnt") == host_namespaces.get("mnt") or namespaces.get("pid") == host_namespaces.get("pid"):
        raise HarnessError(f"pid {pid} did not receive private mount/pid namespaces")
    if share_net:
        if namespaces.get("net") != host_namespaces.get("net"):
            raise HarnessError(f"pid {pid} did not retain the host network namespace for CDP/server")
    elif namespaces.get("net") == host_namespaces.get("net"):
        raise HarnessError(f"pid {pid} retained the host network namespace")
    return {"pid": pid, "exe": _exe(pid), "env": env, "fds": fds, "namespaces": namespaces}


def _run_json(command: list[str], env: dict[str, str], timeout: float) -> object:
    completed = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout, check=True)
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise HarnessError(f"invalid JSON from {' '.join(command)}: {completed.stdout!r}") from error


def _sway_proof(pid: int, runtime: dict[str, Path], env: dict[str, str], log: Path) -> dict:
    output = _run_json([str(SWAYMSG), "-s", env["SWAYSOCK"], "-t", "get_outputs", "-r"], env, 10)
    inputs = _run_json([str(SWAYMSG), "-s", env["SWAYSOCK"], "-t", "get_inputs", "-r"], env, 10)
    if not isinstance(output, list) or len(output) != 1 or output[0].get("name") != "HEADLESS-1" or not output[0].get("active"):
        raise HarnessError(f"unexpected private outputs: {output!r}")
    if inputs != []:
        raise HarnessError(f"private compositor exposed input devices: {inputs!r}")
    proof = _assert_private_process(pid, expected_env=env, runtime=runtime["runtime"], share_net=False)
    proof.update({"outputs": output, "inputs": inputs})
    if log.exists():
        text = log.read_text(errors="replace")
        required = ("headless", "pixman", "xwayland disable")
        missing = [item for item in required if item.lower() not in text.lower()]
        if missing:
            raise HarnessError(f"Sway log lacks isolation evidence {missing}: {log}")
        if any(item in text.lower() for item in ("drm backend", "libinput", "seatd", "xwayland started")):
            # wlroots emits a harmless "no DRM backend supplied" diagnostic for
            # headless mode; only actual backend/session acquisition is rejected.
            actual = [line for line in text.splitlines() if any(term in line.lower() for term in ("drm backend opened", "libinput", "seatd", "xwayland started"))]
            if actual:
                raise HarnessError(f"Sway log suggests a forbidden backend/session: {actual[:3]}")
        proof["log_sha256"] = _sha256(log)
    return proof


def _browser_pid(outer: subprocess.Popen) -> int | None:
    candidates = []
    for pid in _children(outer.pid):
        exe = _exe(pid)
        args = _cmdline(pid)
        if Path(exe).name in {"chromium", "chromium-browser"} and not any(arg.startswith("--type=") for arg in args):
            candidates.append(pid)
    return candidates[0] if candidates else None


def _wait_cdp(port: int, timeout: float) -> str:
    def find() -> str | None:
        try:
            with urlopen(f"http://127.0.0.1:{port}/json/version", timeout=1) as response:
                data = json.load(response)
            if data.get("webSocketDebuggerUrl"):
                return f"http://127.0.0.1:{port}"
        except (OSError, URLError, ValueError):
            return None
        return None

    return _wait_for(find, timeout, "Chromium CDP")


def _wait_private_sockets(env: dict[str, str]) -> bool:
    runtime = Path(env["XDG_RUNTIME_DIR"])
    if not (runtime / "wayland-1").exists() or not Path(env["SWAYSOCK"]).exists():
        return False
    try:
        version = _run_json([str(SWAYMSG), "-s", env["SWAYSOCK"], "-t", "get_version", "-r"], env, 2)
    except (OSError, subprocess.SubprocessError, HarnessError):
        return False
    return isinstance(version, dict) and version.get("human_readable")


def _grim(path: Path, env: dict[str, str], timeout: float) -> None:
    capture_env = dict(env)
    capture_env.pop("SWAYSOCK", None)
    capture_env["WAYLAND_DISPLAY"] = "wayland-1"
    completed = subprocess.run([str(GRIM), "-o", "HEADLESS-1", str(path)], env=capture_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if completed.returncode != 0:
        raise HarnessError(f"grim failed: {completed.stderr.strip()}")
    if not path.is_file() or path.stat().st_size == 0:
        raise HarnessError(f"grim produced no screenshot: {path}")


def _server_command(port: int, checkpoint: Path) -> list[str]:
    code = (
        "from inference import Engine; from serve import make_server; "
        f"e=Engine({str(checkpoint)!r}); s=make_server(e, {port}); "
        "print('isolated bank-viewer server', s.server_address, flush=True); "
        "s.serve_forever()"
    )
    return [sys.executable, "-c", code]


def _resolve_test(name: str) -> Path:
    """Name one of this repository's browser tests, and nothing else.

    The harness hands the test a private display and a disposable server, so
    what it runs has to stay inside ``tests/`` rather than being any path the
    caller names."""
    test = (TESTS / name).resolve() if "/" not in name else Path(name).resolve()
    if test.parent != TESTS or test.suffix != ".cjs":
        raise HarnessError(f"browser test must be a .cjs file in {TESTS}: {name}")
    return test


def _assert_test_contract(test: Path) -> None:
    if not test.is_file():
        raise HarnessError(f"missing browser test: {test}")
    source = test.read_text(encoding="utf-8")
    for name in ("PEBBY_TEST_CDP", "PEBBY_TEST_ORIGIN", "PEBBY_SCREENSHOT_DIR", "PEBBY_PLAYWRIGHT"):
        if name not in source:
            raise HarnessError(f"browser test does not declare the required environment contract: {name}")


def run(args: argparse.Namespace) -> int:
    test = _resolve_test(args.test)
    _assert_test_contract(test)
    for required in (BWRAP, SWAY, SWAYMSG, GRIM, CHROMIUM, NODE, PLAYWRIGHT):
        if not required.exists():
            raise HarnessError(f"required UI isolation executable/path missing: {required}")
    evidence = Path(args.evidence).resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    screenshot_dir = Path(args.screenshot_dir).resolve() if args.screenshot_dir else evidence / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "format": "pebby.bank-viewer-isolation.v1",
        "status": "running",
        "started_at": _now(),
        "repo": str(ROOT),
        "test": str(test),
        "test_sha256": _sha256(test),
        "bwrap": str(BWRAP),
        "sway": {"path": str(SWAY), "sha256": _sha256(SWAY)},
        "browser": {"path": str(CHROMIUM)},
        "evidence_dir": str(evidence),
        "screenshot_dir": str(screenshot_dir),
        "processes": {},
        "commands": {},
    }
    report_path = evidence / "bank-viewer-isolation.json"
    _write_report(report_path, report)
    server: subprocess.Popen | None = None
    compositor: subprocess.Popen | None = None
    browser: subprocess.Popen | None = None
    tmp: tempfile.TemporaryDirectory[str] | None = None
    tracked: list[int] = []
    try:
        tmp = tempfile.TemporaryDirectory(prefix="pbuv-", dir="/tmp")
        runtime = _make_runtime(Path(tmp.name))
        sway_copy = _copy_sway(runtime["root"])
        report["sway"]["copy"] = str(sway_copy)
        report["sway"]["copy_sha256"] = _sha256(sway_copy)
        report["sway"]["copy_mode"] = oct(stat.S_IMODE(sway_copy.stat().st_mode))
        sway_cmd, sway_env = _sway_command(runtime, sway_copy)
        report["commands"]["compositor"] = sway_cmd
        sway_log = evidence / "sway.log"
        with sway_log.open("w", encoding="utf-8") as log:
            compositor = subprocess.Popen(sway_cmd, env=sway_env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        tracked += _sandbox_processes(compositor)
        report["logs"] = {"sway": str(sway_log)}
        report["processes"]["compositor"] = {
            "outer": compositor.pid,
            "outer_start_ticks": _start_ticks(compositor.pid),
            "started_at": _now(),
        }
        _write_report(report_path, report)
        sway_pid = _wait_sway(compositor, sway_copy, args.start_timeout)
        _wait_for(lambda: _wait_private_sockets(sway_env), args.start_timeout, "private Wayland and Sway IPC sockets")
        report["processes"]["compositor"].update({
            "descendants": _children(compositor.pid),
            "sway": sway_pid,
            "sway_start_ticks": _start_ticks(sway_pid),
        })
        report["proof_compositor"] = _sway_proof(sway_pid, runtime, sway_env, sway_log)
        _write_report(report_path, report)

        port = _reserve_port()
        missing_checkpoint = runtime["root"] / "missing-checkpoint.pt"
        server_cmd = _server_command(port, missing_checkpoint)
        server_env = _clean_env(
            HOME=str(runtime["root"] / "home"), PWD=str(ROOT), PYTHONPATH=str(ROOT),
            OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        )
        report["commands"]["server"] = server_cmd
        server_log = evidence / "server.log"
        with server_log.open("w", encoding="utf-8") as log:
            server = subprocess.Popen(server_cmd, cwd=str(ROOT), env=server_env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        tracked.append(server.pid)
        origin = _wait_for(lambda: f"http://127.0.0.1:{port}" if _alive(server.pid) and _health(port) else None, args.start_timeout, "disposable loopback server")
        report["server"] = {"pid": server.pid, "origin": origin, "checkpoint": str(missing_checkpoint), "checkpoint_exists": missing_checkpoint.exists()}
        report["logs"]["server"] = str(server_log)
        report["server"]["start_ticks"] = _start_ticks(server.pid)
        _write_report(report_path, report)

        cdp_port = _reserve_port()
        browser_cmd, browser_env = _browser_command(runtime, cdp_port)
        report["commands"]["browser"] = browser_cmd
        browser_log = evidence / "browser.log"
        with browser_log.open("w", encoding="utf-8") as log:
            browser = subprocess.Popen(browser_cmd, env=browser_env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        tracked += _sandbox_processes(browser)
        report["logs"]["browser"] = str(browser_log)
        report["processes"]["browser"] = {
            "outer": browser.pid,
            "outer_start_ticks": _start_ticks(browser.pid),
            "started_at": _now(),
        }
        _write_report(report_path, report)
        browser_pid = _wait_for(lambda: _browser_pid(browser) if _alive(browser.pid) else None, args.start_timeout, "Chromium inside its private namespace")
        cdp = _wait_cdp(cdp_port, args.start_timeout)
        report["processes"]["browser"].update({
            "descendants": _children(browser.pid),
            "chromium": browser_pid,
            "chromium_start_ticks": _start_ticks(browser_pid),
        })
        report["proof_browser"] = _assert_private_process(browser_pid, expected_env=browser_env, runtime=runtime["runtime"], share_net=True)
        _write_report(report_path, report)

        test_env = _clean_env(
            HOME=str(runtime["root"] / "home"), PWD=str(ROOT),
            PEBBY_TEST_CDP=cdp, PEBBY_TEST_ORIGIN=origin,
            PEBBY_SCREENSHOT_DIR=str(screenshot_dir), PEBBY_PLAYWRIGHT=str(PLAYWRIGHT),
            # The JS test may capture the compositor while its page is still
            # open.  These are explicit private-runtime values, never inherited
            # from the user's display session.
            PEBBY_TEST_GRIM=str(GRIM),
            PEBBY_TEST_WAYLAND_DISPLAY="wayland-1",
            PEBBY_TEST_XDG_RUNTIME_DIR=str(runtime["runtime"]),
        )
        report["commands"]["test"] = [str(NODE), str(test)]
        test_log = evidence / "test.log"
        started = time.monotonic()
        with test_log.open("w", encoding="utf-8") as log:
            completed = subprocess.run([str(NODE), str(test)], cwd=str(ROOT), env=test_env, stdout=log, stderr=subprocess.STDOUT, timeout=args.test_timeout)
        report["test_result"] = {"returncode": completed.returncode, "elapsed_seconds": time.monotonic() - started, "log": str(test_log)}
        _write_report(report_path, report)
        preferred = screenshot_dir / "compositor.png"
        screenshot = preferred if preferred.is_file() and preferred.stat().st_size else screenshot_dir / "harness-after-test.png"
        if screenshot == screenshot_dir / "harness-after-test.png":
            _grim(screenshot, sway_env, args.start_timeout)
        report["screenshot"] = {"path": str(screenshot), "sha256": _sha256(screenshot), "bytes": screenshot.stat().st_size}
        if completed.returncode != 0:
            raise HarnessError(f"browser test exited {completed.returncode}; see {test_log}")
        report["status"] = "complete"
        _write_report(report_path, report)
        return 0
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        return 1
    finally:
        # Refresh descendant lists before each teardown.  This records and kills
        # inner bwrap PID 1 as well as the outer Popen process.
        cleanup: dict[str, dict] = {}
        for label, process in (("browser", browser), ("compositor", compositor)):
            if process is None:
                continue
            pids = _sandbox_processes(process)
            starts = {pid: _start_ticks(pid) for pid in pids}
            cleanup[label] = _kill_exact(pids, starts)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                cleanup[label]["popen_returncode"] = None
            else:
                cleanup[label]["popen_returncode"] = process.returncode
            cleanup[label]["remaining_after_wait"] = [pid for pid in pids if _alive(pid) and _start_ticks(pid) == starts.get(pid)]
            cleanup[label]["remaining"] = cleanup[label]["remaining_after_wait"]
        if server is not None:
            starts = {server.pid: _start_ticks(server.pid)}
            cleanup["server"] = _kill_exact([server.pid], starts)
            try:
                server.wait(timeout=2)
            except subprocess.TimeoutExpired:
                cleanup["server"]["popen_returncode"] = None
            else:
                cleanup["server"]["popen_returncode"] = server.returncode
            cleanup["server"]["remaining_after_wait"] = [pid for pid in (server.pid,) if _alive(pid) and _start_ticks(pid) == starts.get(pid)]
            cleanup["server"]["remaining"] = cleanup["server"]["remaining_after_wait"]
        report["cleanup"] = cleanup
        report["finished_at"] = _now()
        report["processes_clean"] = all(not item.get("remaining") for item in cleanup.values())
        _write_report(report_path, report)
        if tmp is not None:
            tmp.cleanup()


def _health(port: int) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", default=TEST.name, help=f"browser test in {TESTS} to run (default: {TEST.name})")
    parser.add_argument("--evidence", default="artifacts/ui-bank-viewer", help="persistent logs/report directory")
    parser.add_argument("--screenshot-dir", default=None, help="persistent screenshot directory")
    parser.add_argument("--start-timeout", type=float, default=30.0)
    parser.add_argument("--test-timeout", type=float, default=300.0)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as error:
        print(f"verify_bank_viewer.py: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
