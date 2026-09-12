"""Causal history integration on generated engine levels, with CPU-only policies."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import inference
from pebby.agent.evaluate import rollout
from pebby.agent.history import PolicyHistory, for_policy
from pebby.agent.model import build_policy, load_checkpoint, save_checkpoint
from pebby.agent.world_data import history_arrays
from pebby.ls20 import names
from pebby.ls20.curriculum import generate_level
from pebby.ls20.env import Ls20Env
from pebby.ls20.generate import FORMAT, GENERATOR_VERSION, build_level


def corridor(row=3, budget=20):
    """An independently constructed three-action lesson, never an official map."""
    free = {(col, row) for col in range(3, 9)}
    return {"format": FORMAT, "generator_version": GENERATOR_VERSION,
            "walls": sorted({(col, r) for col in range(names.GRID_COLS)
                             for r in range(names.GRID_ROWS)} - free),
            "start": (3, row), "start_triple": [0, 0, 0],
            "goals": [{"cell": (6, row), "triple": [0, 0, 0]}],
            "cyclers": [], "launchers": [], "refills": [],
            "step_counter": budget, "step_cost": 1, "fog": False}


class RecordingPolicy(torch.nn.Module):
    def __init__(self, history=3):
        super().__init__()
        self.length = history
        self.calls = []

    def config(self):
        return {"architecture": "world", "history": self.length}

    def forward(self, frames, history_valid=None, previous_actions=None):
        self.calls.append((frames.detach().clone(), history_valid.detach().clone(),
                           previous_actions.detach().clone()))
        return torch.tensor([[0., 0., 0., 4.]], device=frames.device).expand(frames.size(0), -1)


def tiny_world():
    return build_policy({"architecture": "world", "channels": 8, "heads": 2,
                         "blocks": 1, "loops": 1, "history": 3, "expansion": 1,
                         "temporal_layers": 1, "hud_channels": 4, "hud_tokens": 2,
                         "latent": 8, "reduce": 1, "predictor_blocks": 1,
                         "predictor_hidden": 8, "value_hidden": 8, "max_distance": 8,
                         "lookahead_depth": 1, "summary": 2, "readout_hidden": 8,
                         "ranker_hidden": 8, "sigreg_projections": 4, "sigreg_knots": 3})


class PolicyHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.spec = generate_level(0, 1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_history(self, recorded, frames, actions, length=3):
        expected = history_arrays(frames, actions, length)
        for actual, value in zip(recorded, expected):
            np.testing.assert_array_equal(actual.cpu().numpy()[0], value)

    def assert_reset(self, recorded, frame):
        self.assert_history(recorded, [frame], [-1])
        self.assertEqual(recorded[1].tolist(), [[False, False, True]])
        self.assertEqual(recorded[2].tolist(), [[-1, -1, -1]])

    def test_history_matches_training_arrays_with_actions_that_produced_each_frame(self):
        env = Ls20Env([build_level(self.spec)])
        policy = RecordingPolicy()
        frames, actions = [env.render()], [-1]
        history = for_policy(policy, frames[0], "cpu")
        for action in self.spec["solution"][:5]:
            history.scores()
            self.assert_history(policy.calls[-1], frames, actions)
            result = env.perform(action)
            index = names.ACTION_IDS.index(action)
            frames.append(result.frame)
            actions.append(index)
            history.observe(result.frame, index)
        history.scores()
        self.assert_history(policy.calls[-1], frames, actions)
        self.assertEqual(policy.calls[-1][2].tolist()[0], actions[-3:])

    def test_empty_history_refuses_scores_and_non_world_policy_gets_no_history(self):
        with self.assertRaisesRegex(ValueError, "observe an initial frame"):
            PolicyHistory(RecordingPolicy()).scores()
        self.assertIsNone(for_policy(lambda frame: frame, [[0]]))

    def test_masked_padding_cannot_change_real_world_policy_output(self):
        model = tiny_world().eval()
        env = Ls20Env([build_level(self.spec)])
        frames, valid, actions = history_arrays([env.render()], [-1], 3)
        inputs = torch.tensor(frames[None], dtype=torch.long)
        valid = torch.tensor(valid[None])
        actions = torch.tensor(actions[None])
        changed = inputs.clone()
        changed[:, :2] = (changed[:, :2] + 7) % 16
        changed_actions = actions.clone()
        changed_actions[:, :2] = 3
        with torch.inference_mode():
            first = model(inputs, history_valid=valid, previous_actions=actions)
            second = model(changed, history_valid=valid, previous_actions=changed_actions)
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)

    def test_evaluator_history_resets_after_actual_loss_of_life(self):
        env = Ls20Env([build_level(corridor(budget=1))])
        policy = RecordingPolicy()
        result = rollout(policy, env, max_actions=3, on_stall="repeat")
        self.assertEqual(result["lives_left"], 2)
        self.assertEqual(len(policy.calls), 3)
        expected = Ls20Env([build_level(corridor(budget=1))])
        expected.perform(4)
        observation = expected.perform(4)
        self.assertEqual(expected.lives(), 2)
        self.assert_reset(policy.calls[2], observation.frame)

    def test_evaluator_history_resets_on_real_generated_level_transition(self):
        specs = [corridor(3), corridor(5)]
        env = Ls20Env([build_level(spec) for spec in specs])
        policy = RecordingPolicy()
        result = rollout(policy, env, max_actions=4, on_stall="repeat")
        self.assertEqual(result["levels_completed"], 1)
        self.assertEqual(len(policy.calls), 4)
        expected = Ls20Env([build_level(spec) for spec in specs])
        for _ in range(3):
            observation = expected.perform(4)
        self.assertEqual(expected.level_index, 1)
        self.assert_reset(policy.calls[3], observation.frame)

    def test_replay_resets_history_after_lost_life_and_transition(self):
        for specs, actions in (([corridor(budget=1)], [4, 4]),
                               ([corridor(3), corridor(5)], [4, 4, 4])):
            with self.subTest(levels=len(specs)):
                policy = RecordingPolicy()
                history = PolicyHistory(policy)
                env = Ls20Env([build_level(spec) for spec in specs])
                with patch("inference.build_env", return_value=env):
                    _, _, frame = inference.replay(specs[0], actions, history)
                history.scores()
                self.assert_reset(policy.calls[-1], frame)
                self.assertTrue(env.lives() == 2 or env.level_index == 1)

    def test_engine_rebuilds_request_history_without_cross_client_state(self):
        policy = RecordingPolicy()
        engine = inference.Engine(None)
        engine.agent._model = policy
        engine.agent.loaded = engine.agent._attempted = True
        first, second = corridor(3), corridor(5)
        requests = ((first, [4, 4]), (second, []), (first, [4]), (first, [4, 4]))
        outputs = []
        for spec, actions in requests:
            result = engine.dispatch({"op": "agent", "level": spec, "actions": actions})
            self.assertIsNone(result["reason"])
            self.assertEqual(result["action"], 4)
            outputs.append(result)
            env = Ls20Env([build_level(spec)])
            frames, indices = [env.render()], [-1]
            for action in actions:
                frames.append(env.perform(action).frame)
                indices.append(names.ACTION_IDS.index(action))
            self.assert_history(policy.calls[-1], frames, indices)
        self.assertEqual(outputs[0], outputs[-1])
        for before, after in zip(policy.calls[0], policy.calls[-1]):
            torch.testing.assert_close(before, after)

    def test_generic_world_checkpoint_roundtrip_and_actual_engine_inference(self):
        model = tiny_world().eval()
        frame = Ls20Env([build_level(corridor())]).render()
        history = for_policy(model, frame)
        with torch.inference_mode():
            expected = history.scores()
        with tempfile.TemporaryDirectory(prefix="pebby-world-history-") as directory:
            path = Path(directory) / "tiny.pt"
            saved = save_checkpoint(path, model, note="generated-only CPU history test")
            loaded, metadata = load_checkpoint(path)
            self.assertEqual(loaded.config(), model.config())
            self.assertEqual(metadata["format"], saved["format"])
            self.assertEqual(metadata["config"]["architecture"], "world")
            with torch.inference_mode():
                actual = for_policy(loaded, frame).scores()
            torch.testing.assert_close(expected, actual, atol=0, rtol=0)
            engine = inference.Engine(path)
            first = engine.dispatch({"op": "agent", "level": corridor(), "actions": []})
            engine.dispatch({"op": "agent", "level": corridor(5), "actions": [4, 4]})
            repeated = engine.dispatch({"op": "agent", "level": corridor(), "actions": []})
            self.assertIsNone(first["reason"])
            self.assertTrue(first["loaded"])
            np.testing.assert_allclose(first["probabilities"], expected.softmax(-1).tolist(),
                                       rtol=1e-6, atol=1e-7)
            self.assertEqual(first, repeated)


if __name__ == "__main__":
    unittest.main()
