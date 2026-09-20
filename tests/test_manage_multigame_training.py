"""Detached launcher tests use tiny Python children, never real training."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import manage_multigame_training as manager


class LauncherTests(unittest.TestCase):
    def test_inaccessible_reused_pid_is_not_our_process(self):
        with patch.object(Path, "read_text", side_effect=PermissionError("other user")):
            self.assertIsNone(manager.process_identity(12345))
            self.assertFalse(manager.active({"pid": 12345, "identity": {"starttime": "old"}}))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / 'training'
        self.children = []
        self.addCleanup(self.cleanup_children)

    def cleanup_children(self):
        for state in self.children:
            if manager.active(state):
                fd = manager.pidfd_open(state['pid'])
                try:
                    manager.pidfd_signal(fd, signal.SIGKILL)
                finally:
                    os.close(fd)
            try:
                os.waitpid(state['pid'], 0)
            except ChildProcessError:
                pass
            self.assertIsNone(manager.process_identity(state['pid']))

    def launch_fixture(self, source):
        result = manager.launch(self.run, [sys.executable, '-u', '-c', source])
        self.children.append(manager.read_state(manager.state_dir(self.run)))
        return result

    def test_detached_lifecycle_and_checkpoint_symlink(self):
        source = (
            'import os,signal,time; from pathlib import Path\n'
            f'run = Path({str(self.run)!r})\n'
            'assert not run.exists()\n'
            'assert os.getsid(0) == os.getpid()\n'
            'run.mkdir()\n'
            'def stop(*args):\n'
            ' (run / "saved.pt").write_text("checkpoint")\n'
            ' (run / "latest.pt").symlink_to("saved.pt")\n'
            ' raise SystemExit(0)\n'
            'signal.signal(signal.SIGTERM, stop)\n'
            'print("ready", flush=True)\n'
            'while True: time.sleep(.01)\n'
        )
        result = self.launch_fixture(source)
        self.assertEqual(result['status'], 'running')
        self.assertEqual(Path(result['log']).parent, manager.state_dir(self.run))
        with self.assertRaisesRegex(ValueError, 'already active'):
            manager.launch(self.run, [sys.executable, '-c', 'pass'])
        stopped = manager.stop(self.run, 2)
        self.assertEqual(stopped['status'], 'stopped')
        self.assertEqual(stopped['checkpoint']['target'], str(self.run / 'saved.pt'))
        self.assertIn('ready', Path(result['log']).read_text())

    def test_stale_identity_does_not_signal(self):
        self.launch_fixture('import time; time.sleep(30)')
        path = manager.state_dir(self.run) / 'state.json'
        state = json.loads(path.read_text())
        state['identity']['starttime'] = 'wrong'
        path.write_text(json.dumps(state))
        with patch.object(manager, 'pidfd_signal') as send:
            result = manager.stop(self.run, 0)
        send.assert_not_called()
        self.assertEqual(result['status'], 'stopped')
        self.assertTrue(manager.active(self.children[0]))

    def test_timeout_never_force_kills(self):
        self.launch_fixture('import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)')
        result = manager.stop(self.run, 0.1)
        self.assertEqual(result['status'], 'stopping')
        self.assertTrue(manager.active(self.children[0]))

    def test_failed_start_is_reported_with_log(self):
        with self.assertRaisesRegex(ValueError, 'code 7.*training.log'):
            manager.launch(self.run, [sys.executable, '-c', 'print("failed"); raise SystemExit(7)'])
        self.assertEqual(manager.status(self.run)['status'], 'stopped')
        self.assertIn('failed', (manager.state_dir(self.run) / 'training.log').read_text())

    def test_transient_empty_cmdline_is_not_persisted(self):
        real_identity = manager.process_identity
        calls = []

        def transient_identity(pid):
            identity = real_identity(pid)
            calls.append(pid)
            if len(calls) == 1 and identity:
                return {**identity, 'command': []}
            return identity

        with patch.object(manager, 'process_identity', side_effect=transient_identity):
            result = self.launch_fixture('import time; time.sleep(30)')
        self.assertEqual(result['status'], 'running')
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(self.children[0]['identity']['command'], self.children[0]['command'])

    def test_state_write_failure_rolls_back_and_reaps_child(self):
        real_popen = manager.subprocess.Popen
        processes = []

        def capture_process(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process

        with patch.object(manager.subprocess, 'Popen', side_effect=capture_process):
            with patch.object(Path, 'write_text', side_effect=OSError('state disk full')):
                with self.assertRaisesRegex(OSError, 'state disk full'):
                    manager.launch(self.run, [sys.executable, '-c', 'import time; time.sleep(30)'])
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].returncode)
        self.assertFalse(Path(f'/proc/{processes[0].pid}').exists())
        self.assertFalse((manager.state_dir(self.run) / 'state.json.tmp').exists())

    def test_start_command_owns_output_and_preserves_training_flags(self):
        with patch.object(manager, 'launch', return_value={'status': 'running'}) as launch:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(manager.main(['start', '--run-dir', str(self.run), '--',
                                               '--smoke', '--epochs', '2']), 0)
        launch.assert_called_once_with(
            self.run, [sys.executable, '-u', str(manager.TRAINER), '--out-dir',
                       str(self.run), '--smoke', '--epochs', '2'], cwd=Path.cwd())

    def test_resume_command_uses_checkpoint_and_saved_working_directory(self):
        self.run.mkdir()
        (self.run / 'latest.pt').write_bytes(b'fixture')
        directory = manager.state_dir(self.run)
        directory.mkdir()
        directory.joinpath('state.json').write_text(json.dumps({'cwd': self.temp.name}))
        with patch.object(manager, 'launch', return_value={'status': 'running'}) as launch:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(manager.main(['resume', '--run-dir', str(self.run),
                                               '--epochs', '8', '--device', 'cpu']), 0)
        launch.assert_called_once_with(
            self.run, [sys.executable, '-u', str(manager.TRAINER), '--resume',
                       str(self.run / 'latest.pt'), '--epochs', '8', '--device', 'cpu'],
            cwd=Path(self.temp.name))


if __name__ == '__main__':
    unittest.main()
