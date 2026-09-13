import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from time import perf_counter

import inference
from inference import MAX_ACTIONS, SHIPPED_LEVELS, Engine, validate_actions, validate_level
from pebby.ls20 import shipped

ROOT = Path(__file__).resolve().parents[1]

# Read straight out of the official toolkit (arcprize/ARC-AGI, arc_agi/rendering.py
# COLOR_MAP) rather than imported, so swapping in the 10-colour ARC-AGI-1 palette
# — whose indices disagree almost everywhere — fails here instead of in the UI.
ARC_AGI_3_PALETTE = [
    "#FFFFFF", "#CCCCCC", "#999999", "#666666", "#333333", "#000000",
    "#E53AA3", "#FF7BCC", "#F93C31", "#1E93FF", "#88D8F1", "#FFDC00",
    "#FF851B", "#921231", "#4FCC30", "#A356D6",
]

STATUS_KEYS = {"state", "level_index", "level_count", "levels_completed", "steps_left",
               "step_cost", "lives", "triple", "goal_triples", "goals_solved",
               "player_cell", "fog", "finished", "won"}


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(None)
        cls.spec = json.loads((ROOT / "examples/level.json").read_text())

    def dispatch(self, request):
        return self.engine.dispatch(request)

    def example(self, name):
        return json.loads((ROOT / "examples" / name).read_text())

    def spec_with(self, **changes):
        level = copy.deepcopy(self.spec)
        level.update(changes)
        return {"op": "play", "level": level, "actions": []}

    def test_importing_inference_does_not_load_torch(self):
        # The server must boot and serve the UI on a machine with no torch at
        # all, so the import graph is checked in a fresh interpreter.
        result = subprocess.run([sys.executable, "-c", "import sys, inference; print('torch' in sys.modules)"],
                                cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "False")

    def test_palette_is_the_sixteen_arc_agi_3_colours(self):
        self.assertEqual(list(inference.PALETTE), ARC_AGI_3_PALETTE)
        self.assertEqual(len(set(inference.PALETTE)), 16)
        for index, colour in enumerate(inference.PALETTE):
            with self.subTest(index=index, colour=colour):
                self.assertRegex(colour, r"^#[0-9A-F]{6}$")
                self.assertEqual(len(colour), 7)
        self.assertEqual(len(inference.PALETTE_NAMES), 16)
        self.assertEqual(len(set(inference.PALETTE_NAMES)), 16)
        info = self.dispatch({"op": "info"})
        self.assertEqual(info["palette"], ARC_AGI_3_PALETTE)
        self.assertEqual(info["palette_names"], list(inference.PALETTE_NAMES))

    def test_info_describes_the_whole_contract(self):
        info = self.dispatch({"op": "info"})
        self.assertEqual(info["game"], "ls20")
        self.assertEqual(info["ruleset"], "pebby.ls20.level.v1")
        self.assertEqual(info["actions"], [{"id": 1, "name": "up"}, {"id": 2, "name": "down"},
                                           {"id": 3, "name": "left"}, {"id": 4, "name": "right"}])
        self.assertEqual(info["shipped_levels"], SHIPPED_LEVELS)
        self.assertEqual(SHIPPED_LEVELS, 7)
        self.assertEqual(info["max_actions"], MAX_ACTIONS)
        self.assertEqual(info["max_actions"], 2048)
        self.assertEqual(info["difficulties"], [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(info["grid"], {"cols": 12, "rows": 12, "cell": 5,
                                        "x_origin": 4, "y_origin": 0, "frame_size": 64})
        self.assertEqual(info["triple"], {"shapes": 6, "colors": [12, 9, 14, 8],
                                          "rotations": [0, 90, 180, 270]})
        self.assertEqual(info["agent"], {"loaded": False, "parameters": None,
                                         "checkpoint": None, "reason": info["agent"]["reason"]})

    def test_replay_is_stateless(self):
        first = {"op": "play", "level": {"shipped": 0}, "actions": [3, 3, 3, 1, 1]}
        second = {"op": "play", "level": {"shipped": 1}, "actions": [4, 4, 2]}
        alone = [json.dumps(self.dispatch(first)), json.dumps(self.dispatch(second))]
        # Same question twice, and two levels interleaved: neither may drift.
        self.assertEqual(json.dumps(self.dispatch(first)), alone[0])
        self.dispatch(second)
        self.dispatch({"op": "play", "level": {"shipped": 4}, "actions": [1, 2, 3, 4]})
        self.assertEqual(json.dumps(self.dispatch(first)), alone[0])
        self.assertEqual(json.dumps(self.dispatch(second)), alone[1])
        self.assertNotEqual(alone[0], alone[1])

    def test_frames_are_sixty_four_square_grids_of_palette_indices(self):
        result = self.dispatch({"op": "play", "level": {"shipped": 0}, "actions": [3, 3]})
        self.assertEqual(set(result), {"frames", "frame", "status"})
        self.assertEqual(set(result["status"]), STATUS_KEYS)
        for grid in result["frames"] + [result["frame"]]:
            self.assertEqual(len(grid), 64)
            for row in grid:
                self.assertEqual(len(row), 64)
                self.assertTrue(all(type(value) is int and 0 <= value < 16 for value in row))

    def test_example_solution_completes_shipped_level_one(self):
        request = self.example("play.json")
        self.assertEqual(len(request["actions"]), 13)
        status = self.dispatch(request)["status"]
        self.assertEqual(status["levels_completed"], 1)
        self.assertEqual(status["level_index"], 1)
        self.assertEqual(status["level_count"], 7)
        self.assertFalse(status["finished"])
        # One action short of the goal the level is still open.
        short = self.dispatch({**request, "actions": request["actions"][:-1]})["status"]
        self.assertEqual(short["levels_completed"], 0)
        self.assertEqual(short["level_index"], 0)

    def test_generated_level_round_trips_and_its_solution_wins(self):
        request = self.example("generate.json")
        result = self.dispatch(request)
        self.assertEqual(set(result), {"level", "frame", "status"})
        level = result["level"]
        self.assertEqual(level["format"], "pebby.ls20.level.v1")
        self.assertEqual(level["seed"], request["seed"])
        self.assertEqual(level["difficulty"], request["difficulty"])
        self.assertEqual(len(result["frame"]), 64)
        self.assertEqual(result["status"]["state"],
                         "NOT_FINISHED" if level['training_context_index'] else "NOT_PLAYED")
        # Through JSON and back through the validator, unchanged both ways: the
        # client hands this exact object back on every later request.
        reread = json.loads(json.dumps(level))
        self.assertEqual(reread, level)
        self.assertEqual(validate_level(reread), level)
        self.assertEqual(json.dumps(validate_level(reread)), json.dumps(level))
        solution = level["solution"]
        self.assertEqual(len(solution), level["optimal_actions"])
        self.assertEqual(level['difficulty_version'], 'ls20-reference-v1')
        self.assertEqual(level['training_context_index'], level['difficulty'] - 1)
        with self.assertRaisesRegex(ValueError, 'context'):
            validate_level({**level, 'training_context_index': (level['difficulty'] % 7)})
        status = self.dispatch({"op": "play", "level": reread, "actions": solution})["status"]
        self.assertEqual(status["state"], "WIN")
        self.assertTrue(status["won"])
        self.assertTrue(status["finished"])
        self.assertEqual(status["levels_completed"], 1)
        self.assertFalse(self.dispatch({"op": "play", "level": reread,
                                        "actions": solution[:-1]})["status"]["won"])

    def test_versioned_vanishing_goal_survives_public_validation(self):
        from tests.test_reference_generation_v2 import small_spec
        spec = {"format": inference.RULESET, "size": 64, **small_spec()}
        spec["mechanics_version"] = "ls20-reference-relations-v2"
        canonical = validate_level(spec)
        self.assertIs(canonical["goals"][0]["vanishing_ring"], True)
        self.assertEqual(validate_level(canonical), canonical)
        self.assertEqual(canonical["mechanics_version"], spec["mechanics_version"])
        result = self.dispatch({"op": "play", "level": canonical, "actions": [4]})
        self.assertEqual(result["status"]["goals_solved"], [True, False])
        for version, flag in ((3, True), (4, 1), (4, "yes")):
            bad = copy.deepcopy(spec)
            bad["generator_version"] = version
            bad["goals"][0]["vanishing_ring"] = flag
            with self.assertRaises(ValueError):
                validate_level(bad)

    def test_hint_context_changes_the_oracle_cache_key(self):
        # Context-key regression uses an explicitly historical fixture; calibrated
        # levels separately enforce the tier's single permitted context.
        level = copy.deepcopy(self.spec)
        hint = {**level, 'training_context_index': 0, 'context_index': 0,
                'verification_level_index': 0}
        later = {**level, 'training_context_index': 1, 'context_index': 1,
                 'verification_level_index': 1}
        self.assertNotEqual(inference.cache_key(validate_level(hint)),
                            inference.cache_key(validate_level(later)))
        for context in (True, -1, 7):
            with self.assertRaises(ValueError):
                validate_level({**hint, 'training_context_index': context})
        # seed and difficulty are both optional, and both default.
        default = self.dispatch({"op": "generate"})["level"]
        self.assertEqual((default["seed"], default["difficulty"]), (0, 1))

    def test_a_spec_without_launchers_loads_as_one_with_none(self):
        # Launcher pads arrived with generator version 2. A client still holding
        # a version-1 spec must not be told its level is missing a field.
        self.assertTrue(self.spec["launchers"], "examples/level.json needs a launcher to test this")
        legacy = {key: value for key, value in self.spec.items() if key != "launchers"}
        self.assertEqual(validate_level(legacy), validate_level({**self.spec, "launchers": []}))
        self.assertNotEqual(validate_level(legacy), validate_level(self.spec))
        self.assertNotEqual(self.dispatch({"op": "play", "level": legacy})["frame"],
                            self.dispatch({"op": "play", "level": self.spec})["frame"])

    def test_undo_is_replaying_one_action_fewer(self):
        level, actions = {"shipped": 0}, self.example("play.json")["actions"]
        history = [json.dumps(self.dispatch({"op": "play", "level": level, "actions": actions[:step]}))
                   for step in range(len(actions) + 1)]
        for step in range(len(actions), 0, -1):
            with self.subTest(undo_from=step):
                undone = self.dispatch({"op": "play", "level": level, "actions": actions[:step][:-1]})
                self.assertEqual(json.dumps(undone), history[step - 1])
        self.assertNotEqual(history[-1], history[-2])

    def test_action_history_is_capped(self):
        self.assertEqual(len(validate_actions([1] * MAX_ACTIONS)), MAX_ACTIONS)
        with self.assertRaises(ValueError) as error:
            validate_actions([1] * (MAX_ACTIONS + 1))
        self.assertEqual(str(error.exception), "actions must hold at most 2048 ids.")
        # The cap is a cost bound, not a rule: a full history really does replay.
        status = self.dispatch({"op": "play", "level": {"shipped": 0},
                                "actions": [1, 2] * (MAX_ACTIONS // 2)})["status"]
        self.assertEqual(status["state"], "GAME_OVER")
        self.assertEqual(status["lives"], 0)
        self.assertTrue(status["finished"])
        self.assertFalse(status["won"])

    def test_rejects_malformed_requests(self):
        wall = list(self.spec["walls"][0])
        goal = copy.deepcopy(self.spec["goals"][0])
        cycler = copy.deepcopy(self.spec["cyclers"][0])
        launcher = copy.deepcopy(self.spec["launchers"][0])
        shipped = {"op": "play", "level": {"shipped": 0}}
        cases = [
            ("request is a string", "not a dict", "Request must be a JSON object."),
            ("request is a list", [], "Request must be a JSON object."),
            ("request is a number", 7, "Request must be a JSON object."),
            ("request is null", None, "Request must be a JSON object."),
            ("op missing", {}, 'Request must contain a string "op".'),
            ("op not a string", {"op": 5}, 'Request must contain a string "op".'),
            ("op unknown", {"op": "nope"}, "Unknown operation or unexpected fields."),
            ("info with extra field", {"op": "info", "extra": 1},
             "Unknown operation or unexpected fields."),
            ("seed negative", {"op": "generate", "seed": -1},
             "seed must be an integer between 0 and 4294967295."),
            ("seed a string", {"op": "generate", "seed": "7"},
             "seed must be an integer between 0 and 4294967295."),
            ("seed a bool", {"op": "generate", "seed": True},
             "seed must be an integer between 0 and 4294967295."),
            ("seed too large", {"op": "generate", "seed": 0x100000000},
             "seed must be an integer between 0 and 4294967295."),
            ("difficulty zero", {"op": "generate", "difficulty": 0},
             "difficulty must be one of [1, 2, 3, 4, 5, 6, 7]."),
            ("difficulty too high", {"op": "generate", "difficulty": 9},
             "difficulty must be one of [1, 2, 3, 4, 5, 6, 7]."),
            ("difficulty a string", {"op": "generate", "difficulty": "1"},
             "difficulty must be one of [1, 2, 3, 4, 5, 6, 7]."),
            ("difficulty a bool", {"op": "generate", "difficulty": True},
             "difficulty must be one of [1, 2, 3, 4, 5, 6, 7]."),
            ("generate with extra field", {"op": "generate", "seed": 0, "difficulty": 1, "extra": 1},
             "Unknown operation or unexpected fields."),
            ("shipped without index", {"op": "shipped"}, "Unknown operation or unexpected fields."),
            ("shipped with extra field", {"op": "shipped", "index": 0, "extra": 1},
             "Unknown operation or unexpected fields."),
            ("shipped index negative", {"op": "shipped", "index": -1},
             "shipped must be an integer between 0 and 6."),
            ("shipped index past the last", {"op": "shipped", "index": 7},
             "shipped must be an integer between 0 and 6."),
            ("shipped index a string", {"op": "shipped", "index": "0"},
             "shipped must be an integer between 0 and 6."),
            ("shipped index a bool", {"op": "shipped", "index": True},
             "shipped must be an integer between 0 and 6."),
            ("play without level", {"op": "play"}, "level is required."),
            ("oracle without level", {"op": "oracle"}, "level is required."),
            ("agent without level", {"op": "agent", "actions": []}, "level is required."),
            ("play with extra field", {**shipped, "extra": 1},
             "Unknown operation or unexpected fields."),
            ("level is a string", {"op": "play", "level": "shipped"}, "level must be a JSON object."),
            ("level is a list", {"op": "play", "level": []}, "level must be a JSON object."),
            ("shipped reference with extra key", {"op": "play", "level": {"shipped": 0, "extra": 1}},
             'A shipped level is exactly {"shipped": index}.'),
            ("level with an unknown field", self.spec_with(bogus=1), "Unknown level fields: bogus."),
            ("level with two unknown fields", self.spec_with(bogus=1, another=2),
             "Unknown level fields: another, bogus."),
            ("level missing a field", {"op": "play", "level": {k: v for k, v in self.spec.items()
                                                               if k != "walls"}},
             "Level is missing: walls."),
            ("level missing two fields", {"op": "play", "level": {k: v for k, v in self.spec.items()
                                                                  if k not in ("walls", "goals")}},
             "Level is missing: goals, walls."),
            ("wrong format", self.spec_with(format="pebby.grid.level.v1"),
             "level format must be pebby.ls20.level.v1."),
            ("wall off the right edge", self.spec_with(walls=[[12, 0]]),
             "Each wall must be inside the 12x12 lattice."),
            ("wall off the top edge", self.spec_with(walls=[[0, -1]]),
             "Each wall must be inside the 12x12 lattice."),
            ("goal cell off the lattice", self.spec_with(goals=[{**goal, "cell": [0, 12]}]),
             "Each goal cell must be inside the 12x12 lattice."),
            ("cycler cell off the lattice", self.spec_with(cyclers=[{**cycler, "cell": [12, 12]}]),
             "Each cycler cell must be inside the 12x12 lattice."),
            ("refill off the lattice", self.spec_with(refills=[[0, 99]]),
             "Each refill must be inside the 12x12 lattice."),
            ("cell holds a string", self.spec_with(walls=[["0", 0]]),
             "Each wall must be a [column, row] pair of integers."),
            ("cell holds a float", self.spec_with(walls=[[0.0, 0]]),
             "Each wall must be a [column, row] pair of integers."),
            ("cell holds a bool", self.spec_with(walls=[[True, 0]]),
             "Each wall must be a [column, row] pair of integers."),
            ("cell is one number", self.spec_with(walls=[[1]]),
             "Each wall must be a [column, row] pair of integers."),
            ("cell is a string", self.spec_with(walls=["11"]),
             "Each wall must be a [column, row] pair of integers."),
            ("walls not a list", self.spec_with(walls="none"), "walls must be a list."),
            ("goals not a list", self.spec_with(goals="none"), "goals must be a list."),
            ("cyclers not a list", self.spec_with(cyclers="none"), "cyclers must be a list."),
            ("refills not a list", self.spec_with(refills="none"), "refills must be a list."),
            ("too many walls", self.spec_with(walls=[[0, 0]] * 145),
             "walls must hold at most 144 entries."),
            ("too many goals", self.spec_with(goals=[goal] * 9), "goals must hold at most 8 entries."),
            ("too many cyclers", self.spec_with(cyclers=[cycler] * 33),
             "cyclers must hold at most 32 entries."),
            ("too many refills", self.spec_with(refills=[[1, 1]] * 17),
             "refills must hold at most 16 entries."),
            ("no goals at all", self.spec_with(goals=[]), "A level needs at least one goal."),
            ("goal without a triple", self.spec_with(goals=[{"cell": goal["cell"]}]),
             'Each goal needs cell and triple, with an optional boolean vanishing_ring.'),
            ("goal with an extra key", self.spec_with(goals=[{**goal, "extra": 1}]),
             'Each goal needs cell and triple, with an optional boolean vanishing_ring.'),
            ("cycler without a kind", self.spec_with(cyclers=[{"cell": cycler["cell"]}]),
             'Each cycler must be {"cell": [c, r], "kind": "shape"|"color"|"rotation"}.'),
            ("cycler kind unknown", self.spec_with(cyclers=[{**cycler, "kind": "size"}]),
             'Each cycler kind must be "shape", "color" or "rotation".'),
            # An unhashable kind used to raise TypeError out of `in`, which the
            # HTTP layer does not catch, so it dropped the connection.
            ("cycler kind a list", self.spec_with(cyclers=[{**cycler, "kind": []}]),
             'Each cycler kind must be "shape", "color" or "rotation".'),
            ("cycler kind an object", self.spec_with(cyclers=[{**cycler, "kind": {"a": 1}}]),
             'Each cycler kind must be "shape", "color" or "rotation".'),
            ("launchers not a list", self.spec_with(launchers="none"), "launchers must be a list."),
            ("too many launchers", self.spec_with(launchers=[launcher] * 9),
             "launchers must hold at most 8 entries."),
            ("launcher without a delta", self.spec_with(launchers=[{"cell": launcher["cell"]}]),
             'Each launcher must be {"cell": [c, r], "delta": [dx, dy]}.'),
            ("launcher with an extra key", self.spec_with(launchers=[{**launcher, "extra": 1}]),
             'Each launcher must be {"cell": [c, r], "delta": [dx, dy]}.'),
            ("launcher delta diagonal", self.spec_with(launchers=[{**launcher, "delta": [1, 1]}]),
             "Each launcher delta must be one of [[0, -1], [0, 1], [-1, 0], [1, 0]]."),
            ("launcher delta oversized", self.spec_with(launchers=[{**launcher, "delta": [0, -1, 0]}]),
             "Each launcher delta must be one of [[0, -1], [0, 1], [-1, 0], [1, 0]]."),
            # `[True, 0] == [1, 0]` and `[1.0, 0] == [1, 0]` in Python, so plain
            # membership would let a non-integer delta through and give one level
            # two canonical forms.
            ("launcher delta holds a bool", self.spec_with(launchers=[{**launcher, "delta": [True, 0]}]),
             "Each launcher delta must be one of [[0, -1], [0, 1], [-1, 0], [1, 0]]."),
            ("launcher delta holds a float", self.spec_with(launchers=[{**launcher, "delta": [1.0, 0]}]),
             "Each launcher delta must be one of [[0, -1], [0, 1], [-1, 0], [1, 0]]."),
            ("launcher cell off the lattice",
             self.spec_with(launchers=[{**launcher, "cell": [12, 3]}]),
             "Each launcher cell must be inside the 12x12 lattice."),
            ("launcher inside a wall", self.spec_with(launchers=[{**launcher, "cell": wall}]),
             "An interacting tile sits inside a wall."),
            ("launcher on a goal", self.spec_with(launchers=[{**launcher, "cell": goal["cell"]}]),
             "Two interacting tiles share one cell."),
            ("goal triple out of range", self.spec_with(goals=[{**goal, "triple": [0, 0, 4]}]),
             "Each goal triple entry 2 must be between 0 and 3."),
            ("start triple shape too high", self.spec_with(start_triple=[6, 0, 0]),
             "start_triple entry 0 must be between 0 and 5."),
            ("start triple colour too high", self.spec_with(start_triple=[0, 4, 0]),
             "start_triple entry 1 must be between 0 and 3."),
            ("start triple rotation negative", self.spec_with(start_triple=[0, 0, -1]),
             "start_triple entry 2 must be between 0 and 3."),
            ("start triple too short", self.spec_with(start_triple=[0, 0]),
             "start_triple must be a [shape, colour, rotation] triple of integers."),
            ("start triple holds a bool", self.spec_with(start_triple=[True, 0, 0]),
             "start_triple must be a [shape, colour, rotation] triple of integers."),
            ("step counter zero", self.spec_with(step_counter=0),
             "step_counter must be an integer between 1 and 1000."),
            ("step counter too high", self.spec_with(step_counter=1001),
             "step_counter must be an integer between 1 and 1000."),
            ("step counter a bool", self.spec_with(step_counter=True),
             "step_counter must be an integer between 1 and 1000."),
            ("step cost zero", self.spec_with(step_cost=0),
             "step_cost must be an integer between 1 and 10."),
            ("step cost too high", self.spec_with(step_cost=11),
             "step_cost must be an integer between 1 and 10."),
            ("step cost a bool", self.spec_with(step_cost=True),
             "step_cost must be an integer between 1 and 10."),
            ("fog a string", self.spec_with(fog="yes"), "fog must be true or false."),
            ("fog an integer", self.spec_with(fog=1), "fog must be true or false."),
            ("two tiles on one cell", self.spec_with(cyclers=[{**cycler, "cell": goal["cell"]}]),
             "Two interacting tiles share one cell."),
            ("goal inside a wall", self.spec_with(goals=[{**goal, "cell": wall}]),
             "An interacting tile sits inside a wall."),
            ("refill inside a wall", self.spec_with(refills=[wall]),
             "An interacting tile sits inside a wall."),
            ("start inside a wall", self.spec_with(start=wall), "start sits inside a wall."),
            ("actions a string", {**shipped, "actions": "1234"},
             "actions must be a list of action ids."),
            ("actions an object", {**shipped, "actions": {"0": 1}},
             "actions must be a list of action ids."),
            ("action zero", {**shipped, "actions": [1, 0]},
             "Each action must be one of [1, 2, 3, 4]."),
            ("action five", {**shipped, "actions": [5]}, "Each action must be one of [1, 2, 3, 4]."),
            ("action negative", {**shipped, "actions": [-1]},
             "Each action must be one of [1, 2, 3, 4]."),
            ("action a string", {**shipped, "actions": ["1"]},
             "Each action must be one of [1, 2, 3, 4]."),
            ("action a bool", {**shipped, "actions": [True]},
             "Each action must be one of [1, 2, 3, 4]."),
            ("action a float", {**shipped, "actions": [1.0]},
             "Each action must be one of [1, 2, 3, 4]."),
            ("one action too many", {**shipped, "actions": [1] * (MAX_ACTIONS + 1)},
             "actions must hold at most 2048 ids."),
        ]
        self.assertEqual(len(cases), len({name for name, _, _ in cases}))
        for name, request, message in cases:
            with self.subTest(case=name):
                with self.assertRaises(ValueError) as error:
                    self.dispatch(request)
                self.assertEqual(str(error.exception), message)

    def test_oracle_answers_every_shipped_level(self):
        # info()["oracle_levels"] says which levels the oracle can search inside
        # a request, which since rail support landed is the only thing that
        # separates them: the planner understands all seven. It is not a claim
        # about solvability in either direction.
        advertised = self.dispatch({"op": "info"})["oracle_levels"]
        self.assertEqual(advertised, list(shipped.CHEAP_LEVELS))
        self.assertLessEqual(set(advertised), set(range(SHIPPED_LEVELS)))
        for index in range(SHIPPED_LEVELS):
            with self.subTest(shipped=index):
                advice = self.dispatch({"op": "oracle", "level": {"shipped": index}, "actions": []})
                self.assertEqual(set(advice), {"action", "available", "remaining", "reason",
                                               "optimal", "human_baseline"})
                # Known without searching, so it is present whatever the answer.
                self.assertEqual(advice["optimal"], shipped.OPTIMAL_ACTIONS[index])
                self.assertEqual(advice["human_baseline"], shipped.HUMAN_BASELINE[index])
                if index in advertised:
                    # A level it can search, from the start, must produce the
                    # cached optimum. That is the live planner and the cache
                    # checking each other.
                    self.assertTrue(advice["available"], advice)
                    self.assertIn(advice["action"], (1, 2, 3, 4))
                    self.assertEqual(advice["remaining"], shipped.OPTIMAL_ACTIONS[index])
                    self.assertIsNone(advice["reason"])
                else:
                    self.assertFalse(advice["available"], advice)
                    self.assertIsNone(advice["action"])
                    self.assertIsNone(advice["remaining"])
                    self.assertIn(f"cached {shipped.OPTIMAL_ACTIONS[index]} actions", advice["reason"])
                    self.assertNotIn("unsolvable", advice["reason"])

    def test_expensive_shipped_levels_are_answered_without_searching(self):
        # Levels 6 and 7 need 13M and 22M planner states and several gigabytes.
        # A request must never start that search, so the answer has to come back
        # immediately and from the cache, not after a truncated search.
        expensive = [index for index in range(SHIPPED_LEVELS) if index not in shipped.CHEAP_LEVELS]
        self.assertTrue(expensive, "shipped.CHEAP_LEVELS covering everything makes this test vacuous")
        for index in expensive:
            with self.subTest(shipped=index):
                self.assertGreater(shipped.search_limit(index), inference.ORACLE_STATE_LIMIT)
                started = perf_counter()
                advice = self.dispatch({"op": "oracle", "level": {"shipped": index}, "actions": []})
                elapsed = perf_counter() - started
                # Replaying the level is all this may cost. A truncated search
                # at the request ceiling would take seconds.
                self.assertLess(elapsed, 1.0, f"took {elapsed:.2f}s, which means it searched")
                self.assertEqual(advice["optimal"], shipped.OPTIMAL_ACTIONS[index])
                self.assertIn(f"{inference.ORACLE_STATE_LIMIT:,}-state limit", advice["reason"])

    def test_the_planner_default_search_limit_is_never_used(self):
        # The planner's own default is 600,000 states, which truncates silently
        # on shipped level 5. Passing a limit explicitly is what keeps that from
        # turning into "no plan found" on a level that has one.
        self.assertEqual(shipped.search_limit(4), 1_000_000)
        self.assertGreater(shipped.search_limit(4), 600_000)
        self.assertGreaterEqual(inference.ORACLE_STATE_LIMIT, shipped.search_limit(4))
        advice = self.dispatch({"op": "oracle", "level": {"shipped": 4}, "actions": []})
        self.assertTrue(advice["available"], advice)
        self.assertEqual(advice["remaining"], shipped.OPTIMAL_ACTIONS[4])

    def test_oracle_reports_an_unfinishable_level_without_refusing_it(self):
        # The planner models every mechanic in this level and still cannot
        # finish it, because the player is walled into one cell. That is the
        # third oracle answer, distinct from "I do not model this level" and
        # from "you have walked into a dead end", and it is the one the UI has
        # to degrade on now that nothing is refused outright.
        walls = ([[column, 0] for column in range(12)] + [[column, 11] for column in range(12)]
                 + [[0, row] for row in range(12)] + [[11, row] for row in range(12)]
                 + [[1, 10], [2, 1], [1, 2]])
        sealed = validate_level({
            "format": "pebby.ls20.level.v1", "walls": walls, "start": [1, 1],
            "start_triple": [0, 0, 0], "goals": [{"cell": [5, 5], "triple": [1, 1, 1]}],
            "cyclers": [], "launchers": [], "refills": [], "step_counter": 42,
            "step_cost": 1, "fog": False})
        self.assertEqual(self.dispatch({"op": "oracle", "level": sealed, "actions": []}),
                         {"action": None, "available": False, "remaining": None,
                          "optimal": None, "human_baseline": None,
                          "reason": "The planner found no plan for level 1."})
        # Unfinishable is not unplayable: the level still replays by hand, which
        # is what the UI leaves the player able to do.
        status = self.dispatch({"op": "play", "level": sealed, "actions": [1, 2, 3, 4]})["status"]
        self.assertEqual(status["state"], "NOT_FINISHED")
        self.assertEqual(status["player_cell"], [1, 1])
        self.assertEqual(status["goals_solved"], [False])

    def test_oracle_refuses_a_finished_game(self):
        level = validate_level(self.spec)
        solution = level["solution"]
        self.assertTrue(self.dispatch({"op": "play", "level": level,
                                       "actions": solution})["status"]["won"])
        self.assertEqual(self.dispatch({"op": "oracle", "level": level, "actions": solution}),
                         {"action": None, "available": False, "remaining": None,
                          "optimal": level["optimal_actions"], "human_baseline": None,
                          "reason": "The game is over."})
        # One action from the end it is still advising, so the refusal above is
        # the game being over rather than the level being unplannable.
        advice = self.dispatch({"op": "oracle", "level": level, "actions": solution[:-1]})
        self.assertEqual(advice, {"action": solution[-1], "available": True, "remaining": 1,
                                  "optimal": level["optimal_actions"], "human_baseline": None,
                                  "reason": None})

    def test_agent_degrades_instead_of_failing(self):
        missing = str(inference.DEFAULT_CHECKPOINT)
        self.assertFalse(inference.DEFAULT_CHECKPOINT.exists(), "this test needs an untrained tree")
        request = self.example("agent.json")
        for name, checkpoint, expected in [("default", None, missing),
                                           ("explicit", "/nonexistent/path.pt", "/nonexistent/path.pt")]:
            with self.subTest(checkpoint=name):
                result = Engine(checkpoint).dispatch(request)
                self.assertEqual(result["action"], None)
                self.assertEqual(result["probabilities"], None)
                self.assertFalse(result["loaded"])
                self.assertTrue(result["reason"].startswith(f"No agent checkpoint at {expected}."),
                                result["reason"])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "garbage.pt"
            checkpoint.write_bytes(b"not a checkpoint")
            # The message quotes torch's own loader, which changes between torch
            # releases; only the part inference.py writes is asserted.
            result = Engine(str(checkpoint)).dispatch(request)
            self.assertFalse(result["loaded"])
            self.assertIsNone(result["action"])
            self.assertTrue(result["reason"].startswith(f"Could not load {checkpoint}: "),
                            result["reason"])

    def test_cli_runs_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "bad.json"
            bad.write_text(json.dumps({"op": "play", "level": {"shipped": 9}}))
            for name in ("info.json", "shipped.json", "play.json", "agent.json"):
                with self.subTest(request=name):
                    result = subprocess.run([sys.executable, str(ROOT / "predict.py"),
                                             str(ROOT / "examples" / name)],
                                            cwd=directory, capture_output=True, text=True, timeout=120)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(json.loads(result.stdout),
                                     json.loads(json.dumps(self.dispatch(self.example(name)))))
            result = subprocess.run([sys.executable, str(ROOT / "predict.py"), str(bad)],
                                    cwd=directory, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, "")
            self.assertEqual(result.stderr,
                             "Request failed: shipped must be an integer between 0 and 6.\n")


if __name__ == "__main__":
    unittest.main()
