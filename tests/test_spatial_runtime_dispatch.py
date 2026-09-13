"""Generic checkpoint/runtime contracts using synthetic models, no game instances."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from pebby.agent.model import load_checkpoint
from pebby.agent.history import PolicyHistory
from pebby.agent.spatial_outcome_policy import FORMAT as SPATIAL_FORMAT
from pebby.agent.spatial_route_outcome_policy import FORMAT as ROUTE_FORMAT, checkpoint_from_parent
from pebby.agent.spatial_route_outcome_planner import SpatialRouteOutcomePlanner
from pebby.agent.spatial_semantic_outcome_policy import FORMAT as SEMANTIC_FORMAT
from tests.test_spatial_outcome_policy import checkpoint, make_policy


class SpatialRuntimeDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        assert not torch.cuda.is_initialized()

    def test_explicit_formats_dispatch_to_exact_loader_with_path_and_device(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            for kind, module in ((SPATIAL_FORMAT, 'spatial_outcome_policy'),
                                 (ROUTE_FORMAT, 'spatial_route_outcome_policy'),
                                 (SEMANTIC_FORMAT, 'spatial_semantic_outcome_policy')):
                with self.subTest(format=kind):
                    torch.save({'format': kind}, path)
                    sentinel = (object(), {'validated': True})
                    with patch('pebby.agent.' + module + '.load_checkpoint', return_value=sentinel) as loader:
                        self.assertIs(load_checkpoint(path, 'cpu'), sentinel)
                        loader.assert_called_once_with(path, 'cpu')
                    with patch('pebby.agent.' + module + '.load_checkpoint', side_effect=ValueError('bad lineage')):
                        with self.assertRaisesRegex(ValueError, 'bad lineage'):
                            load_checkpoint(path)

    def test_real_synthetic_envelopes_load_through_agent_and_preserve_h8_scores(self):
        from inference import AgentPolicy, DEFAULT_CHECKPOINT
        torch.manual_seed(41)
        original = make_policy()
        parent = checkpoint(original)
        route = SpatialRouteOutcomePlanner.from_parent(original.planner, route_channels=16, route_heads=4)
        envelopes = (parent, checkpoint_from_parent(parent, route, parent_checkpoint_sha256='a' * 64))
        frames = torch.randint(0, 16, (10, 64, 64))
        with tempfile.TemporaryDirectory() as directory:
            for index, envelope in enumerate(envelopes):
                with self.subTest(format=envelope['format']):
                    path = Path(directory) / f'{index}.pt'
                    torch.save(envelope, path)
                    agent = AgentPolicy(path)
                    self.assertTrue(agent.ensure(), agent.reason)
                    self.assertEqual(agent.parameters, sum(p.numel() for p in agent._model.parameters()))
                    self.assertGreater(agent.parameters, envelope['planner_weights']['value_head.weight'].numel())
                    self.assertTrue({'weights', 'encoder_weights', 'planner_weights'}.isdisjoint(agent.metadata))
                    self.assertEqual(agent._model.config()['architecture'], 'world')
                    self.assertEqual(agent._model.config()['history'], 8)
                    history = PolicyHistory(agent._model)
                    for i, frame in enumerate(frames):
                        history.observe(frame.tolist(), i % 4 if i else -1)
                    self.assertEqual(len(history.frames), 8)
                    with torch.inference_mode():
                        expected = agent._model(frames[-8:][None], torch.ones(1, 8, dtype=torch.bool),
                                                torch.tensor([[i % 4 for i in range(2, 10)]]))[0]
                        torch.testing.assert_close(history.scores(), expected, atol=0, rtol=0)
                    action, probabilities = agent.act(frames[-1], history)
                    self.assertEqual(action, int(expected.argmax()) + 1)
                    torch.testing.assert_close(torch.tensor(probabilities), expected.softmax(-1), atol=0, rtol=0)
                    history.observe(frames[0].tolist(), reset=True)
                    self.assertEqual(len(history.frames), 1)
                    self.assertEqual(history.actions, [-1])
                    # Exercise the Engine's world-history dispatch without constructing a game.
                    import inference
                    engine = inference.Engine(path, banks=object())
                    engine.agent = agent
                    def fake_replay(level, actions, active_history):
                        self.assertIsInstance(active_history, PolicyHistory)
                        for i, frame in enumerate(frames):
                            active_history.observe(frame.tolist(), i % 4 if i else -1, reset=i == 0)
                        return object(), [frames[-1]], frames[-1]
                    with patch.object(inference, 'replay', side_effect=fake_replay), \
                         patch.object(inference, 'status_of', return_value={}):
                        response = engine.act({}, [])
                    self.assertIsNone(response['reason'])
                    self.assertEqual(response['action'], action)
                    self.assertEqual(response['probabilities'], probabilities)
        from inference import DEFAULT_CHECKPOINT as after
        self.assertEqual(DEFAULT_CHECKPOINT, after)

    def test_semantic_envelope_uses_generic_runtime_and_excludes_perceptor_tensors(self):
        from inference import AgentPolicy
        from pebby.agent import spatial_semantic_outcome_policy as semantic
        from pebby.agent.spatial_semantic_outcome_planner import SpatialSemanticOutcomePlanner
        from pebby.agent.cell_appearance import CellAppearance
        from pebby.agent.neural_outcome_policy import weights_sha256
        original, teacher = make_policy(), CellAppearance()
        planner = SpatialSemanticOutcomePlanner.from_parent(original.planner, actor=True,
                                                             route_channels=16, actor_channels=16)
        with patch.object(semantic, 'TEACHER_WEIGHTS_SHA256', weights_sha256(teacher.state_dict())):
            envelope = semantic.checkpoint_from_parent(checkpoint(original), planner, teacher)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'semantic.pt'
                torch.save(envelope, path)
                agent = AgentPolicy(path)
                self.assertTrue(agent.ensure(), agent.reason)
                self.assertTrue({'weights', 'encoder_weights', 'planner_weights', 'perceptor_weights'}.isdisjoint(agent.metadata))
                self.assertEqual(agent.parameters, sum(p.numel() for p in agent._model.parameters()))
                action, probabilities = agent.act(torch.zeros(64, 64, dtype=torch.long))
                self.assertIn(action, (1, 2, 3, 4))
                self.assertEqual(len(probabilities), 4)

    def test_actual_replay_resets_history_on_life_loss_without_a_real_game(self):
        import inference
        frame = torch.zeros(64, 64, dtype=torch.long).tolist()
        class FakeEnv:
            level_index = 0
            def __init__(self):
                self.steps = 0
            def render(self):
                return frame
            def lives(self):
                return 3 if self.steps < 5 else 2
            def perform(self, action):
                self.steps += 1
                return SimpleNamespace(frames=[frame], frame=frame)
        history = PolicyHistory(make_policy())
        with patch.object(inference, 'build_env', return_value=FakeEnv()):
            inference.replay({}, [1] * 7, history)
        self.assertEqual(len(history.frames), 3)
        self.assertEqual(history.actions, [-1, 0, 0])

    def test_underlying_lineage_and_weight_digest_validation_cannot_be_bypassed(self):
        parent = checkpoint(make_policy())
        route = SpatialRouteOutcomePlanner.from_parent(make_policy().planner, route_channels=16, route_heads=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.pt'
            for envelope in (parent, checkpoint_from_parent(parent, route, parent_checkpoint_sha256='a' * 64)):
                for changed in ({'encoder_frozen': False}, {'encoder_weights_sha256': '0' * 64},
                                {'encoder_runtime': {}}, {'official_training_inputs': True}):
                    with self.subTest(format=envelope['format'], changed=changed):
                        torch.save({**envelope, **changed}, path)
                        with self.assertRaises(ValueError):
                            load_checkpoint(path)

    def test_existing_cnn_format_still_roundtrips_and_unknown_format_fails(self):
        from pebby.agent.model import Ls20Policy, save_checkpoint
        model = Ls20Policy(channels=8, blocks=1, hud_channels=8, reduce_channels=2, hidden=16).eval()
        frame = torch.zeros(1, 64, 64, dtype=torch.long)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.pt'
            save_checkpoint(path, model)
            restored, _ = load_checkpoint(path)
            with torch.inference_mode():
                torch.testing.assert_close(restored(frame), model(frame), atol=0, rtol=0)
            torch.save({'format': 'unknown'}, path)
            with self.assertRaisesRegex(ValueError, 'unsupported'):
                load_checkpoint(path)


if __name__ == '__main__':
    unittest.main()
