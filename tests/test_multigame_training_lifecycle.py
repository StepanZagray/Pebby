"""Exercise the real background CLI and signal/resume path on a tiny CPU fixture."""
import os
from pathlib import Path
import signal
import time

from pebby.agent.multigame_training import load_training_checkpoint
from tools import manage_multigame_training as manager
from test_multigame_training import _pair


def test_real_training_start_stop_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("MKL_NUM_THREADS", "2")
    train, validation = _pair(tmp_path)
    run = tmp_path / "run"
    owned = []

    def remember():
        state = manager.read_state(manager.state_dir(run))
        owned.append(state)
        return state

    try:
        assert manager.main([
            "start", "--run-dir", str(run), "--",
            "--train-manifest", str(train), "--validation-manifest", str(validation),
            "--smoke", "--cpu-test-model", "--architecture", "v2", "--device", "cpu",
            "--epochs", "100", "--update-mode", "game", "--checkpoint-every-games", "1",
            "--closed-loop-games", "1", "--closed-loop-train-games", "0",
            "--history-mode", "none", "--frame-weight", "0", "--event-weight", "0",
        ]) == 0
        state = remember()
        deadline = time.monotonic() + 20
        while not (run / "latest.pt").is_file() and time.monotonic() < deadline:
            assert manager.active(state), Path(state["log"]).read_text()
            time.sleep(0.02)
        assert (run / "latest.pt").is_file()
        assert manager.stop(run, timeout=20)["status"] == "stopped"
        os.waitpid(state["pid"], 0)
        saved = load_training_checkpoint(run / "latest.pt")
        assert saved["epoch_progress"] is not None or saved["epoch"] < 99
        total_epochs = max(1, saved["epoch"] + 2)
        assert manager.main([
            "resume", "--run-dir", str(run), "--epochs", str(total_epochs), "--device", "cpu",
        ]) == 0
        resumed = remember()
        deadline = time.monotonic() + 30
        while manager.active(resumed) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not manager.active(resumed), Path(resumed["log"]).read_text()
        _, wait_status = os.waitpid(resumed["pid"], 0)
        assert os.waitstatus_to_exitcode(wait_status) == 0, Path(resumed["log"]).read_text()
        completed = load_training_checkpoint(run / "latest.pt")
        assert completed["epoch"] == total_epochs - 1
        assert completed["epoch_progress"] is None
        assert completed["training_config"]["history_mode"] == "none"
        assert completed["global_step"] > saved["global_step"]
        assert (run / "best.pt").is_file()
    finally:
        for state in owned:
            if manager.active(state):
                fd = manager.pidfd_open(state["pid"])
                try:
                    manager.pidfd_signal(fd, signal.SIGKILL)
                finally:
                    os.close(fd)
            try:
                os.waitpid(state["pid"], 0)
            except ChildProcessError:
                pass
            assert not Path(f"/proc/{state['pid']}").exists()
