import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools.build_structured_event_cache import (
    _cache_arrays,
    _metadata_file_hashes,
    load_event_source,
    validate_event_arrays,
)
from tools.build_structured_field_cache import actual_histories


def _event_arrays():
    n = 2
    frames = np.zeros((n, 8, 64, 64), dtype=np.uint8)
    frames[0, -1] = 1
    frames[1, -1] = 2
    history_valid = np.zeros((n, 8), dtype=bool)
    history_valid[:, -1] = True
    previous_actions = np.full((n, 8), -1, dtype=np.int64)
    next_frames = np.zeros((n, 4, 64, 64), dtype=np.uint8)
    next_frames[0, 0] = 3
    next_frames[0, 1] = 4
    next_frames[0, 2] = 5
    next_frames[0, 3] = 6
    next_frames[1] = 7
    terminal = np.zeros((n, 4), dtype=bool)
    won = np.zeros((n, 4), dtype=bool)
    lost = np.zeros((n, 4), dtype=bool)
    # Row zero contains one reset loss and one win. Row one is a doomed state
    # with a terminal third-life failure; its current mask must stay zero.
    lost[0, 1] = True
    terminal[0, 2] = True
    won[0, 2] = True
    lost[0, 3] = True
    lost[1, 2] = True
    terminal[1, 2] = True
    lost[1, 3] = True
    terminal[1, 3] = True
    branch_events = np.where(won, 3, np.where(lost & terminal, 2, np.where(lost, 1, 0))).astype(np.uint8)
    current_reachable = np.array([True, False])
    current_distance = np.array([1, -1], dtype=np.int16)
    optimal = np.array([1, 0], dtype=np.uint8)
    next_distance = np.array([[2, 2, 0, -1], [2, 2, -1, -1]], dtype=np.int16)
    next_reachable = next_distance > 0
    next_optimal = np.where(next_reachable, 1, 0).astype(np.uint8)
    next_optimal[0, 2:] = 0
    next_optimal[1, 2:] = 0
    selected_action = np.array([1, 2], dtype=np.int8)
    data = {
        "frames": frames, "history_valid": history_valid,
        "previous_actions": previous_actions, "next_frames": next_frames,
        "terminal": terminal, "won": won, "lost_life": lost,
        "optimal": optimal, "seed": np.array([123, 1_000_123], dtype=np.int64),
        "context_index": np.array([4, 5], dtype=np.int8),
        "split_id": np.array([0, 1], dtype=np.int8),
        "player_cell": np.zeros((n, 2), dtype=np.int16),
        "next_player_cell": np.zeros((n, 4, 2), dtype=np.int16),
        "current_triple": np.zeros((n, 3), dtype=np.int16),
        "next_triple": np.zeros((n, 4, 3), dtype=np.int16),
        "current_steps": np.full(n, 20, dtype=np.int16),
        "next_steps": np.full((n, 4), 19, dtype=np.int16),
        "current_lives": np.array([3, 1], dtype=np.int16),
        "next_lives": np.array([[3, 2, 3, 2], [1, 1, 0, 0]], dtype=np.int16),
        "current_reachable": current_reachable,
        "current_distance": current_distance,
        "next_distance": next_distance,
        "next_reachable": next_reachable,
        "next_optimal": next_optimal,
        "branch_events": branch_events,
        "selected_action": selected_action,
        "actual_lost_life": lost[np.arange(n), selected_action],
        "actual_terminal": terminal[np.arange(n), selected_action],
        "actual_won": won[np.arange(n), selected_action],
        "actual_event": branch_events[np.arange(n), selected_action],
    }
    validate_event_arrays(data)
    return data


class StructuredEventCacheTests(unittest.TestCase):
    def test_metadata_hashes_include_checkpoint_style_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            world = Path(directory) / "world.pt"
            visibility = Path(directory) / "visibility.pt"
            world.write_bytes(b"world")
            visibility.write_bytes(b"visibility")
            world_hash = hashlib.sha256(world.read_bytes()).hexdigest()
            visibility_hash = hashlib.sha256(visibility.read_bytes()).hexdigest()
            metadata = {
                "sources": {
                    "world_checkpoint": {"path": str(world), "sha256": world_hash},
                    "visibility": {"checkpoint": str(visibility), "sha256": visibility_hash},
                }
            }
            hashes = _metadata_file_hashes(metadata)
            self.assertEqual(hashes[str(world)], world_hash)
            self.assertEqual(hashes[str(visibility)], visibility_hash)

    def test_actual_histories_resets_only_loss_branch_and_keeps_action_alternatives(self):
        source = _event_arrays()
        histories, validity, actions = actual_histories({
            "frames": source["frames"][:1], "history_valid": source["history_valid"][:1],
            "previous_actions": source["previous_actions"][:1],
            "next_frames": source["next_frames"][:1], "lost_life": source["lost_life"][:1],
        })
        self.assertEqual(histories.shape, (1, 4, 8, 64, 64))
        self.assertEqual(validity.shape, (1, 4, 8))
        self.assertEqual(actions.shape, (1, 4, 8))
        # Branch 1 is a nonterminal loss: its target history is the real reset
        # frame with no causal pre-reset actions.
        self.assertTrue(torch.all(histories[0, 1] == torch.from_numpy(source["next_frames"][0, 1])))
        self.assertEqual(validity[0, 1].tolist(), [False] * 7 + [True])
        self.assertEqual(actions[0, 1].tolist(), [-1] * 8)
        # Branch 0 is live, so the prior history shifts and action 0 is appended.
        self.assertTrue(torch.equal(histories[0, 0, -1], torch.from_numpy(source["next_frames"][0, 0])))
        self.assertTrue(validity[0, 0, -1])
        self.assertEqual(actions[0, 0, -1], 0)
        # A terminal WIN is still a normal causal successor: it appends the
        # action and retains the preceding history.  A normal life loss is a
        # reset even when it is not terminal.
        self.assertTrue(torch.equal(histories[0, 2, -1], torch.from_numpy(source["next_frames"][0, 2])))
        self.assertEqual(validity[0, 2].tolist(), [False] * 6 + [True, True])
        self.assertEqual(actions[0, 2, -1], 2)
        self.assertTrue(torch.all(histories[0, 3] == torch.from_numpy(source["next_frames"][0, 3])))
        self.assertEqual(validity[0, 3].tolist(), [False] * 7 + [True])
        self.assertEqual(actions[0, 3].tolist(), [-1] * 8)
        # A third-life GAME_OVER is also a reset, but only a 1 -> 0 branch is
        # valid terminal-failure data.
        terminal_histories, terminal_validity, terminal_actions = actual_histories({
            "frames": source["frames"][1:2], "history_valid": source["history_valid"][1:2],
            "previous_actions": source["previous_actions"][1:2],
            "next_frames": source["next_frames"][1:2], "lost_life": source["lost_life"][1:2],
        })
        self.assertTrue(torch.all(terminal_histories[0, 2] == torch.from_numpy(source["next_frames"][1, 2])))
        self.assertEqual(terminal_validity[0, 2].tolist(), [False] * 7 + [True])
        self.assertEqual(terminal_actions[0, 2].tolist(), [-1] * 8)

    def test_event_source_accepts_doomed_zero_optimal_and_preserves_split_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            data = _event_arrays()
            meta = {
                "format": "pebby.ls20-structured-event-transitions-fast.v1",
                "source": "generated_only", "oracle_search": "complete_only", "history": 8,
                "alternatives_per_state": 4, "official_inputs_used": False,
                "levels": [
                    {"seed": 123, "difficulty": 2, "context_index": 4, "context_engine_verified": True},
                    {"seed": 1_000_123, "difficulty": 3, "context_index": 5, "context_engine_verified": True},
                ],
            }
            source = tmp_path / "events.npz"
            np.savez_compressed(source, **data, meta=np.array(json.dumps(meta)))
            report = tmp_path / "events.json"
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            code_path = str(Path(__file__).resolve())
            code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            report.write_text(json.dumps({
                "status": "complete", "source_unchanged": True, "code_unchanged": True,
                "output": {"sha256": source_hash}, "source_code_hashes": {code_path: code_hash},
                "sources": {"train": {"path": str(source), "selected": [123],
                                       "sha256_before": source_hash, "sha256_after": source_hash},
                            "validation": {"path": str(source), "selected": [1_000_123],
                                           "sha256_before": source_hash, "sha256_after": source_hash}},
            }))
            train = load_event_source(source, report, "train")
            validation = load_event_source(source, report, "validation")
            self.assertEqual(train["seeds"].tolist(), [123])
            self.assertEqual(validation["seeds"].tolist(), [1_000_123])
            self.assertEqual(int(validation["optimal"][0]), 0)
            self.assertEqual(validation["difficulties"].tolist(), [3])

    def test_event_cache_mapping_keeps_four_branch_events_and_zero_masks(self):
        data = _event_arrays()
        fields = np.zeros((2, 148, 96), dtype=np.float16)
        future = np.zeros((2, 4, 148, 96), dtype=np.float16)
        arrays = _cache_arrays(data | {
            "seeds": data["seed"], "difficulties": np.array([2, 3], dtype=np.int8),
            "source_rows": np.array([4, 9]),
        }, fields, future)
        self.assertEqual(arrays["branch_actions"].tolist(), [[0, 1, 2, 3], [0, 1, 2, 3]])
        self.assertEqual(arrays["next_fields"].shape, (2, 4, 148, 96))
        self.assertEqual(arrays["optimal"].tolist(), [1, 0])
        self.assertEqual(arrays["next_optimal"][1, 2], 0)
        self.assertEqual(arrays["branch_events"][0].tolist(), [0, 1, 3, 1])

    def test_build_split_writes_distinct_event_manifest_without_one_row_level_filter(self):
        from tools.build_structured_event_cache import build_split

        class FakeEncoder(torch.nn.Module):
            def eval(self):
                return self

            def parameters(self):
                return iter(())

            def metadata(self):
                return {"config": {"tokens": 148, "width": 96}, "sources": {}}

            def forward(self, frames, history_valid, previous_actions):
                return torch.zeros((len(frames), 148, 96), dtype=torch.float32)

        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            data = _event_arrays()
            meta = {
                "format": "pebby.ls20-structured-event-transitions-fast.v1",
                "source": "generated_only", "oracle_search": "complete_only", "history": 8,
                "alternatives_per_state": 4, "official_inputs_used": False,
                "levels": [
                    {"seed": 123, "difficulty": 2, "context_index": 4, "context_engine_verified": True},
                    {"seed": 1_000_123, "difficulty": 3, "context_index": 5, "context_engine_verified": True},
                ],
            }
            source = tmp_path / "events.npz"
            np.savez_compressed(source, **data, meta=np.array(json.dumps(meta)))
            report = tmp_path / "events.json"
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            code_path = str(Path(__file__).resolve())
            code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            report.write_text(json.dumps({
                "status": "complete", "source_unchanged": True, "code_unchanged": True,
                "output": {"sha256": source_hash}, "source_code_hashes": {code_path: code_hash},
                "sources": {"train": {"path": str(source), "selected": [123],
                                       "sha256_before": source_hash, "sha256_after": source_hash},
                            "validation": {"path": str(source), "selected": [1_000_123],
                                           "sha256_before": source_hash, "sha256_after": source_hash}},
            }))
            out = tmp_path / "cache"
            (tmp_path / "world.pt").write_bytes(b"world")
            (tmp_path / "vis.pt").write_bytes(b"visibility")
            with patch("pebby.agent.structured_field.load_structured_field_encoder",
                       return_value=FakeEncoder()):
                manifest = build_split(source, report, out, tmp_path / "world.pt", tmp_path / "vis.pt",
                                       "train", batch_size=1, max_encoder_batch=2)
            self.assertEqual(manifest["rows"], 1)
            self.assertEqual(manifest["event_coverage"]["terminal_failure"], 0)
            self.assertEqual(manifest["event_coverage"]["doomed_current_rows"], 0)
            self.assertEqual(np.load(out / "next_fields.npy").shape, (1, 4, 148, 96))
            self.assertEqual(json.loads((out / "manifest.json").read_text())["split"], "train")

    def test_event_source_rejects_report_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            data = _event_arrays()
            meta = {"format": "pebby.ls20-structured-event-transitions-fast.v1", "source": "generated_only",
                    "oracle_search": "complete_only", "history": 8, "alternatives_per_state": 4,
                    "official_inputs_used": False, "levels": []}
            source = tmp_path / "events.npz"
            np.savez_compressed(source, **data, meta=np.array(json.dumps(meta)))
            report = tmp_path / "events.json"
            report.write_text(json.dumps({"status": "running"}))
            with self.assertRaisesRegex(ValueError, "not complete"):
                load_event_source(source, report, "train")
