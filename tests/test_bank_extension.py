"""Durable extension and collision checks without expensive generation."""

import fcntl
import json
from pathlib import Path
import tempfile
import unittest

from tools.extend_curriculum_bank import extend_bank


def verified(job):
    seed, difficulty = job
    return {"seed": seed, "difficulty": difficulty, "generator_version": 3,
            "curriculum_version": 2, "search_truncated": False, "engine_verified": True,
            "solution": [1, 2, 3], "optimal_actions": 3,
            "context_solution": [1, 2, 3], "context_optimal_actions": 3,
            "context_index": seed % 7, "verification_level_index": seed % 7,
            "verification_match_hint": seed % 7 == 0, "context_engine_verified": True,
            "engine_win": True, "replay_lives": 3, "levels_completed": 1}


class BankExtensionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="pebby-bank-extension-test-")
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / "source.jsonl"
        self.target = Path(self.directory.name) / "extended.jsonl"
        self.source.write_text(json.dumps(verified((10000, 1))) + "\n")

    def rows(self):
        return [json.loads(line) for line in self.target.read_text().splitlines()]

    def test_failure_preserves_contiguous_progress_and_resume_skips_saved_seeds(self):
        def failing(job):
            if job[0] == 10003:
                raise RuntimeError("bounded candidate verification failed")
            return verified(job)

        with self.assertRaisesRegex(RuntimeError, "verification failed"):
            extend_bank(self.source, self.target, 5, workers=1, generator=failing)
        self.assertEqual([row["seed"] for row in self.rows()], [10000, 10001, 10002])
        calls = []

        def recording(job):
            calls.append(job)
            return verified(job)

        extend_bank(self.source, self.target, 5, workers=1, generator=recording)
        self.assertEqual(calls, [(10003, 4), (10004, 5)])
        self.assertEqual(self.rows()[0], json.loads(self.source.read_text()))
        unchanged = self.target.read_bytes()
        extend_bank(self.source, self.target, 5, workers=1, generator=recording)
        self.assertEqual(self.target.read_bytes(), unchanged)
        self.assertEqual(len(calls), 2)

    def test_unterminated_partial_row_is_recovered_before_resume(self):
        extend_bank(self.source, self.target, 2, workers=1, generator=verified)
        with self.target.open("ab") as handle:
            handle.write(b'{"seed":10002')
        extend_bank(self.source, self.target, 3, workers=1, generator=verified)
        self.assertEqual([row["seed"] for row in self.rows()], [10000, 10001, 10002])

    def test_wrong_source_prefix_and_duplicate_seed_are_rejected(self):
        self.target.write_text(json.dumps(verified((20000, 1))) + "\n")
        with self.assertRaisesRegex(ValueError, "source prefix"):
            extend_bank(self.source, self.target, 3, workers=1, generator=verified)
        self.target.write_bytes(self.source.read_bytes() * 2)
        with self.assertRaisesRegex(ValueError, "collision"):
            extend_bank(self.source, self.target, 3, workers=1, generator=verified)

    def test_concurrent_writer_and_incomplete_proof_are_rejected(self):
        lockpath = self.target.with_suffix(".jsonl.lock")
        with lockpath.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "another writer"):
                extend_bank(self.source, self.target, 3, workers=1, generator=verified)
        with self.assertRaisesRegex(ValueError, "complete curriculum verification"):
            extend_bank(self.source, self.target, 2, workers=1,
                        generator=lambda job: {**verified(job), "search_truncated": True})
        self.assertEqual(len(self.rows()), 1)


if __name__ == "__main__":
    unittest.main()
