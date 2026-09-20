"""Full-standard SP80 generation and native replay tests."""

from dataclasses import replace
from collections import Counter
import importlib
import json
from pathlib import Path
import random
import tempfile
import time
import unittest
from unittest.mock import patch

from arcengine import GameState, Level

from pebby.games.sp80 import names
from pebby.games.sp80.bank import build as build_bank, load as load_bank, save as save_bank
from pebby.games.sp80.env import Env, official_levels, replay, upstream
from pebby.games.sp80.generate import (
    CURRICULUM,
    DIFFICULTIES,
    FORMAT,
    FULL_STANDARD_CONTRACT,
    REFERENCE_PROFILES,
    build_game,
    build_level,
    generate,
    generate_game,
    last_game_generation_report,
    last_generation_report,
    validate_full_standard,
)
from pebby.games.sp80.layout import extract
from pebby.games.sp80.plan import _selection_clicks, _state_key, search, solve
from pebby.multigame import source_for


generation = importlib.import_module("pebby.games.sp80.generate")


def checked_replay(env, actions):
    start_score = env.levels_completed
    observation = None
    for action_id, x, y in actions:
        assert action_id in env.available_actions
        if action_id == names.ACTION_CLICK:
            assert isinstance(x, int) and isinstance(y, int)
            assert 0 <= x < 64 and 0 <= y < 64
        else:
            assert x is None and y is None
        observation = env.perform(action_id, x, y)
        if env.levels_completed > start_score or observation.state == GameState.WIN:
            return True, observation
        if observation.state == GameState.GAME_OVER:
            return False, observation
    return False, observation


def hidden_by_sinks(level):
    """Piece, cup and deflector pixels that a later-drawn edge sink overwrites.

    Sprites are painted at grid resolution in level order, which is the order
    the native camera composites them, so the last painter owns each cell.
    """
    size = int(level.grid_size[0])
    sprites = list(level.get_sprites())
    owner, cells = {}, {}
    for index, sprite in enumerate(sprites):
        pixels = sprite.render()
        cells[index] = set()
        for row in range(pixels.shape[0]):
            for column in range(pixels.shape[1]):
                x, y = int(sprite.x) + column, int(sprite.y) + row
                if int(pixels[row, column]) >= 0 and 0 <= x < size and 0 <= y < size:
                    cells[index].add((x, y))
                    owner[(x, y)] = index
    hidden = []
    for index, sprite in enumerate(sprites):
        if not set(sprite.tags) & {names.TAG_PIPE, names.TAG_DEFLECTOR, names.TAG_CUP}:
            continue
        hidden.extend((sprite.name, cell) for cell in sorted(cells[index])
                      if names.TAG_SINK in sprites[owner[cell]].tags)
    return hidden


class OfficialGame(unittest.TestCase):
    def test_official_levels_draw_no_piece_or_cup_under_an_edge_sink(self):
        for index, level in enumerate(official_levels(), 1):
            env = Env([level])
            env.reset()
            self.assertEqual(hidden_by_sinks(env.level), [], index)

    def test_all_six_official_teachers_replay_in_real_engine(self):
        env = Env()
        env.reset()
        lengths = []
        for index in range(6):
            env.set_level(index)
            result = search(env, limit=80_000)
            self.assertTrue(result.solved, (index, result.reason))
            self.assertFalse(result.truncated)
            self.assertFalse(result.unsupported)
            self.assertTrue(result.exact)
            lengths.append(len(result.actions))
            completed, _ = checked_replay(env.clone(), result.actions)
            self.assertTrue(completed, index)
        self.assertEqual(lengths, [4, 18, 32, 49, 42, 43])

    def test_snapshot_layout_and_clone_are_independent(self):
        env = Env()
        env.reset()
        layout = extract(env)
        twin = env.clone()
        before = twin.render()
        env.perform(names.ACTION_RIGHT)
        self.assertEqual(twin.render(), before)
        self.assertNotEqual(env.render(), twin.render())
        completed, _ = checked_replay(layout.snapshot.clone(), solve(layout))
        self.assertTrue(completed)

    def test_truncation_and_unsupported_are_distinct(self):
        env = Env()
        env.reset()
        capped = search(env, limit=0)
        self.assertTrue(capped.truncated)
        self.assertFalse(capped.unsupported)
        unsupported = replace(extract(env), unsupported=("synthetic mechanic",))
        result = search(unsupported)
        self.assertTrue(result.unsupported)
        self.assertFalse(result.exact)

    def test_official_level_inventory_is_fresh(self):
        first = official_levels()
        second = official_levels()
        self.assertEqual(len(first), 6)
        self.assertIsNot(first[0], second[0])
        self.assertEqual(Env().level_count, 6)

    def test_failed_flow_auto_selection_and_frame_guard_remain_explicit(self):
        module = upstream()
        p = module.sprites
        level = Level(
            sprites=[
                p[names.FRAME[16]].clone().set_position(-1, -1),
                p[names.WATER].clone().set_position(10, 1),
                p[names.PIPE[5]].clone().set_position(-1, 4),
                p[names.PIPE[3]].clone().set_position(0, 4),
                p[names.CUP].clone().set_position(6, 13),
                p[names.SOURCE].clone().set_position(10, 0),
                p[names.SINK].clone().set_position(0, 15),
            ],
            grid_size=(16, 16),
            data={names.KEY_STEPS: 12, names.KEY_ROTATION: 0},
        )
        env = Env([level])
        env.reset()
        clicks = dict(_selection_clicks(env))
        self.assertNotIn(1, clicks)
        env.perform(names.ACTION_CLICK, *clicks[0])
        before = _state_key(env)
        env.perform(names.ACTION_FLOW)
        after = _state_key(env)
        self.assertEqual(after[0], before[0])
        self.assertEqual(after[1:], (1, 1))
        with patch.object(env.module.Sp80, "step", lambda game: None):
            guarded = search(env, limit=10, budget=1)
        self.assertTrue(guarded.unsupported)
        self.assertIn("frame guard", guarded.reason)


class FullGeneratedGames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = [generate(0, d, search_work=80_000) for d in DIFFICULTIES]
        if any(spec is None for spec in cls.specs):
            raise AssertionError("one of the six baseline generated tiers was rejected")

    def test_contract_and_reference_profile_cover_all_six_ordered_tiers(self):
        self.assertEqual(DIFFICULTIES, (1, 2, 3, 4, 5, 6))
        self.assertEqual(tuple(item["difficulty"] for item in CURRICULUM), DIFFICULTIES)
        self.assertEqual(tuple(item["context_index"] for item in CURRICULUM), tuple(range(6)))
        self.assertEqual(FULL_STANDARD_CONTRACT["status"], "ready")
        self.assertEqual(FULL_STANDARD_CONTRACT["format"], "pebby-full-generator-contract-v1")
        self.assertEqual(FULL_STANDARD_CONTRACT["source_id"], source_for("sp80").source_id)
        self.assertEqual(set(REFERENCE_PROFILES), set(DIFFICULTIES))

    def test_every_tier_validates_and_stored_solution_is_legal_native_replay(self):
        for difficulty, spec in zip(DIFFICULTIES, self.specs):
            self.assertEqual(validate_full_standard(spec, CURRICULUM[difficulty - 1]), [])
            self.assertTrue(spec["engine_verified"])
            self.assertEqual(spec["format"], FORMAT)
            self.assertEqual(spec["solution_length"], len(spec["solution"]))
            self.assertLessEqual(spec["solution_length"], spec["steps"])
            self.assertEqual(
                spec["critical_piece_indices"],
                [i for i, piece in enumerate(spec["pieces"])
                 if (piece["x"], piece["y"]) != (piece["target_x"], piece["target_y"])],
            )
            env = Env([build_level(json.loads(json.dumps(spec)))])
            env.reset()
            completed, observation = checked_replay(env, spec["solution"])
            self.assertTrue(completed, difficulty)
            self.assertEqual(observation.state, GameState.WIN)

    def test_mechanic_certificates_cover_later_reference_features(self):
        d4, d5, d6 = self.specs[3:]
        self.assertGreater(d4["mechanic_use"]["embedded_source_emissions"], 0)
        self.assertGreater(d4["mechanic_use"].get("source_pipe_incoming_hits", 0), 0)
        self.assertTrue(d4["mechanic_use"]["source_pipe_tag_ablation_prevents_win"])
        self.assertGreater(d5["mechanic_use"]["deflector_right_hits"], 0)
        self.assertEqual(sum(c["rotation"] != 0 for c in d5["cups"]), 1)
        self.assertGreater(d6["mechanic_use"]["embedded_source_emissions"], 0)
        self.assertGreater(d6["mechanic_use"]["vertical_pipe_hits"], 0)
        self.assertGreater(d6["mechanic_use"]["deflector_left_hits"], 0)
        self.assertGreater(d6["mechanic_use"]["deflector_right_hits"], 0)
        self.assertEqual(d6["mechanic_use"]["source_pipe_incoming_hits"], 0)
        self.assertFalse(d6["mechanic_use"]["source_pipe_tag_ablation_prevents_win"])
        self.assertEqual(sum(c["rotation"] != 0 for c in d6["cups"]), 3)
        self.assertEqual(len(d6["sinks"]), 3)

    def test_tier_four_source_pipe_combines_incoming_split_and_emission(self):
        spec = self.specs[3]
        env = Env([build_level(spec)])
        env.reset()
        for action in spec["solution"][:-1]:
            env.perform(*action)
        source_indices = [
            index for index, piece in enumerate(spec["pieces"])
            if piece["kind"] == "source_pipe"
        ]
        self.assertEqual(source_indices, [4])
        source_pipe = env.movables()[source_indices[0]]
        self.assertIn(names.TAG_SOURCE, source_pipe.tags)
        self.assertIn(names.TAG_PIPE, source_pipe.tags)
        source_pipe.tags.remove(names.TAG_PIPE)
        self.assertNotEqual(env.perform(names.ACTION_FLOW).state, GameState.WIN)

    def test_three_difficulty_one_seeds_finish_within_sixty_seconds(self):
        started = time.perf_counter()
        specs = [generate(seed, 1, search_work=80_000) for seed in range(3)]
        self.assertTrue(all(spec is not None for spec in specs))
        self.assertLess(time.perf_counter() - started, 60)
        self.assertGreaterEqual(len({spec["geometry_d4_sha256"] for spec in specs}), 2)
        self.assertTrue(all(len(spec["gameplay_sha256"]) == 64 for spec in specs))

    def test_live_random_edit_and_failed_flow_can_be_replanned(self):
        spec = self.specs[5]
        env = Env([build_level(spec)])
        env.reset()
        for action in (names.ACTION_UP, names.ACTION_RIGHT, names.ACTION_DOWN, names.ACTION_LEFT):
            env.perform(action)
        clicks = list(_selection_clicks(env))
        self.assertTrue(clicks)
        env.perform(names.ACTION_CLICK, *clicks[-1][1])
        observation = env.perform(names.ACTION_FLOW)
        self.assertEqual(observation.state, GameState.NOT_FINISHED)
        self.assertEqual(env.failed_flows, 1)
        result = search(env, limit=80_000)
        self.assertTrue(result.solved, result.reason)
        self.assertTrue(checked_replay(env, result.actions)[0])

    def test_full_game_mode_replays_sequential_context_and_rejects_reduced_build(self):
        specs = generate_game(23, split="validation", search_work=80_000)
        self.assertIsNotNone(specs)
        self.assertEqual([s["difficulty"] for s in specs], list(DIFFICULTIES))
        self.assertEqual([s["whole_game_context_replay"]["score_after"] for s in specs],
                         [1, 2, 3, 4, 5, 6])
        env = Env(build_game(specs))
        env.reset()
        for index, spec in enumerate(specs):
            self.assertEqual(env.level_index, index)
            self.assertTrue(checked_replay(env, spec["solution"])[0])
        self.assertEqual(env.state, GameState.WIN)
        reduced = generate_game(23, split="validation", difficulties=(1, 2), search_work=80_000)
        self.assertEqual(len(reduced), 2)
        with self.assertRaisesRegex(ValueError, "complete six-tier"):
            build_game(reduced)

    def test_split_identity_determinism_bounds_and_tamper_detection(self):
        first = generate(11, 2, split="train", search_work=80_000)
        self.assertEqual(first, generate(11, 2, split="train", search_work=80_000))
        validation = generate(11, 2, split="validation", search_work=80_000)
        self.assertEqual(first["geometry_split"], "train")
        self.assertEqual(validation["geometry_split"], "validation")
        self.assertNotEqual(first["geometry_d4_sha256"], validation["geometry_d4_sha256"])
        self.assertIsNone(generate(11, 2, attempts=1, node_limit=1))

        wrong_route = json.loads(json.dumps(first))
        wrong_route["solution"][0][0] = names.ACTION_FLOW
        errors = validate_full_standard(wrong_route, CURRICULUM[1])
        self.assertTrue(any("route" in error or "replay" in error for error in errors), errors)
        forged = json.loads(json.dumps(first))
        forged["mechanic_use"]["horizontal_pipe_hits"] += 1
        errors = validate_full_standard(forged, CURRICULUM[1])
        self.assertTrue(any("mechanic-use" in error for error in errors), errors)
        malformed = json.loads(json.dumps(first))
        malformed["pieces"] = None
        self.assertTrue(validate_full_standard(malformed, CURRICULUM[1]))

    def test_identity_uses_initial_semantics_and_excludes_private_target_certificate(self):
        original = json.loads(json.dumps(self.specs[0]))
        retargeted = json.loads(json.dumps(original))
        retargeted["pieces"][0]["target_y"] += 1
        self.assertEqual(generation._identities(original), generation._identities(retargeted))

        reordered = json.loads(json.dumps(self.specs[1]))
        reordered["pieces"][0], reordered["pieces"][1] = (
            reordered["pieces"][1], reordered["pieces"][0])
        initial_exact, initial_d4, initial_gameplay = generation._identities(self.specs[1])
        reordered_exact, reordered_d4, reordered_gameplay = generation._identities(reordered)
        self.assertEqual((initial_exact, initial_d4), (reordered_exact, reordered_d4))
        self.assertNotEqual(initial_gameplay, reordered_gameplay)

    def test_validator_recomputes_official_exclusion_and_guarded_criticality_is_unknown(self):
        spec = self.specs[0]
        official = generation._OFFICIAL_D4_GEOMETRY | {spec["geometry_d4_sha256"]}
        with patch.object(generation, "_OFFICIAL_D4_GEOMETRY", official):
            errors = validate_full_standard(spec, CURRICULUM[0])
        self.assertTrue(any("official" in error for error in errors), errors)

        original_perform = generation.Env.perform

        def guarded_perform(env, action, x=None, y=None):
            if action == names.ACTION_FLOW:
                raise ValueError("too many frames for action")
            return original_perform(env, action, x, y)

        counts = Counter()
        with patch.object(generation.Env, "perform", guarded_perform):
            self.assertIsNone(generation.verify(spec, 80_000, counts))
        self.assertEqual(counts["criticality_frame_guard"], 1)

    def test_public_generation_inputs_are_strict_and_smoke_tiers_are_ordered_unique(self):
        for value in (True, 1.0):
            with self.assertRaises(ValueError):
                generate(1, value)
            for field in ("attempts", "max_attempts", "limit", "node_limit", "search_work"):
                with self.subTest(value=value, field=field), self.assertRaises(ValueError):
                    generate(1, 1, **{field: value})
        with self.assertRaises(ValueError):
            generate(True, 1)
        with self.assertRaises(ValueError):
            generate_game(1.0, split="train")
        with self.assertRaises(ValueError):
            generate_game(1, split="train", difficulties=(2, 1))
        with self.assertRaises(ValueError):
            generate_game(1, split="train", difficulties=(1, 1))

    def test_bounded_whole_game_failure_retains_typed_tier_diagnostics(self):
        self.assertIsNone(generate_game(888, split="train"))
        game_report = last_game_generation_report()
        self.assertEqual(game_report["status"], "failed")
        self.assertEqual(game_report["failed_difficulty"], 3)
        self.assertEqual(game_report["completed_difficulties"], [1, 2])
        level_report = game_report["level_report"]
        self.assertEqual(level_report, last_generation_report())
        self.assertEqual(level_report["status"], "exhausted")
        self.assertEqual(level_report["attempts_used"], 24)
        self.assertEqual(level_report["terminal_reason"], "bounded_attempts_exhausted")
        self.assertEqual(level_report["rejections"], {
            "geometry_split": 17,
            "mechanic_use": 3,
            "noncritical_moves": 1,
            "quality_profile": 3,
        })

    def test_first_frame_of_every_generated_tier_keeps_objects_clear_of_sinks(self):
        for difficulty in DIFFICULTIES:
            drafts = 0
            for seed in range(40):
                rng = random.Random(f"sink-clearance:{seed}:{difficulty}")
                spec = generation._draft_tier(rng, seed, difficulty, "train", 0)
                if not generation._scramble(rng, spec):
                    continue
                drafts += 1
                env = Env([build_level(spec)])
                env.reset()
                self.assertEqual(hidden_by_sinks(env.level), [], (difficulty, seed))
            self.assertGreaterEqual(drafts, 20, difficulty)
        for difficulty, spec in zip(DIFFICULTIES, self.specs):
            env = Env([build_level(spec)])
            env.reset()
            self.assertEqual(hidden_by_sinks(env.level), [], difficulty)
            for piece in spec["pieces"]:
                self.assertTrue(generation._position_allowed(
                    spec, piece, piece["target_x"], piece["target_y"]))

    def test_validator_rejects_pieces_cups_and_sources_placed_under_edge_sinks(self):
        tier5, tier6 = self.specs[4], self.specs[5]
        self.assertEqual(sorted(generation._sink_cells(tier5)),
                         [(x, 19) for x in range(19)] + [(19, y) for y in range(20)])
        self.assertEqual(sorted(generation._sink_cells(self.specs[0])),
                         [(x, 15) for x in range(16)])

        cup_on_sink = json.loads(json.dumps(tier5))
        side_cup = next(cup for cup in cup_on_sink["cups"] if cup["rotation"] == 270)
        side_cup["x"] = 18
        with self.assertRaisesRegex(ValueError, "cup overlaps an edge sink"):
            generation._validate_structure(cup_on_sink)

        piece_on_sink = json.loads(json.dumps(tier6))
        deflector = next(p for p in piece_on_sink["pieces"] if p["kind"] == "deflector_left")
        self.assertFalse(generation._position_allowed(piece_on_sink, deflector, 12, 18))
        self.assertFalse(generation._position_allowed(piece_on_sink, deflector, 18, 10))
        deflector["x"], deflector["y"] = 12, 18
        with self.assertRaisesRegex(ValueError, "position is illegal"):
            generation._validate_structure(piece_on_sink)

        source_on_sink = json.loads(json.dumps(tier5))
        source_on_sink["sources"][-1] = 19
        with self.assertRaisesRegex(ValueError, "source sits on an edge sink"):
            generation._validate_structure(source_on_sink)

    def test_json_bank_io_and_build_game_mismatch_checks(self):
        spec = self.specs[2]
        self.assertEqual(json.loads(json.dumps(spec)), spec)
        with tempfile.TemporaryDirectory() as directory:
            specs, tried = build_bank(2, seed=30, difficulty=1)
            self.assertEqual(len(specs), 2)
            self.assertGreaterEqual(tried, 2)
            path = save_bank(specs, Path(directory) / "sp80.jsonl")
            self.assertEqual(load_bank(path), specs)
        game = generate_game(7, split="test", search_work=80_000)
        broken = json.loads(json.dumps(game))
        broken[1]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "same split"):
            build_game(broken)


if __name__ == "__main__":
    unittest.main()
