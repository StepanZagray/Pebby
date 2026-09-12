"""Independent regressions for audited world-model data and metric hazards."""
import tempfile
import copy
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

from pebby.agent import world_model as wm, world_train as train
from pebby.agent.history import PolicyHistory
from pebby.ls20.env import Ls20Env
from pebby.ls20.generate import build_level
from tests.test_policy_history import tiny_world, corridor
from pebby.agent.world_data import collect_level
from pebby.ls20.generate import generate_legacy_level as generate_level


class WorldContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_collapsed_retrieval_scores_chance_and_average_tie_rank(self):
        rank, top = wm._rank_of_true(torch.zeros(3, 4, 8), torch.zeros(3, 4, 8))
        self.assertAlmostEqual(float(rank), 2.5)
        self.assertAlmostEqual(float(top), .25)
        targets = torch.eye(4)[None]
        rank, top = wm._rank_of_true(targets, targets)
        self.assertEqual(float(rank), 1.)
        self.assertEqual(float(top), 1.)
        rank, top = wm._rank_of_true(torch.zeros_like(targets), targets)
        self.assertAlmostEqual(float(top), .25)

    def test_life_loss_target_equals_fresh_inference_history(self):
        model = tiny_world().eval()
        spec = corridor(budget=1)
        env = Ls20Env([build_level(spec)])
        first = env.render()
        second = env.perform(4).frame
        successors = []
        for action in (1, 2, 3, 4):
            branch = Ls20Env([build_level(spec)])
            branch.perform(4)
            successors.append(branch.perform(action).frame)
            self.assertEqual(branch.lives(), 2)
        batch = {'frames':torch.tensor([[first, first, second]]),
                 'history_valid':torch.tensor([[False, True, True]]),
                 'previous_actions':torch.tensor([[-1,-1,3]]),
                 'next_frames':torch.tensor([successors]),
                 'terminal':torch.zeros(1,4,dtype=torch.bool),
                 'won':torch.zeros(1,4,dtype=torch.bool),
                 'optimal':torch.tensor([15]), 'distances':torch.ones(1,4,dtype=torch.long),
                 'lost_life':torch.ones(1,4,dtype=torch.bool)}
        out = wm.world_losses(model, batch)
        for action, frame in enumerate(successors):
            expected = model.encode(torch.tensor([[frame,frame,frame]]),
                                    torch.tensor([[False,False,True]]),
                                    torch.tensor([[-1,-1,-1]]))['latent'][0]
            torch.testing.assert_close(out['targets'][0, action],expected,atol=1e-6,rtol=1e-5)
        self.assertTrue(torch.isfinite(out['total']))

    def test_policy_selection_uses_actual_loss_key(self):
        self.assertEqual(train.selection_score({'policy':1.2}, 'policy_cross_entropy'),1.2)

    def test_failed_checkpoint_write_preserves_previous_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            wm.save_world_checkpoint(path, tiny_world(), epoch=1)
            previous = path.read_bytes()
            with patch.object(torch, 'save', side_effect=OSError('simulated disk failure')):
                with self.assertRaises(OSError):
                    wm.save_world_checkpoint(path, tiny_world(), epoch=2)
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ['checkpoint.pt'])

    def test_curriculum_training_stream_changes_difficulty_and_keeps_full_unique_batches(self):
        seeds = np.arange(500)
        data = {'seeds': seeds, 'meta': {'levels': [
            {'seed': int(seed), 'difficulty': int(seed // 100 + 1)} for seed in seeds]}}
        sampler = train.CurriculumSampler(data)
        tensors = {'frames': torch.from_numpy(seeds[:, None]), 'id': torch.from_numpy(seeds)}
        generator = torch.Generator().manual_seed(12)
        early = next(train.curriculum_batches(tensors, sampler, 64, generator, 1, 0, 2))
        late = next(train.curriculum_batches(tensors, sampler, 64, generator, 1, 1, 2))
        for batch in (early, late):
            self.assertEqual(len(batch['id'].unique()), 64)
            torch.testing.assert_close(batch['frames'][:, 0], batch['id'])
        self.assertGreater(int((early['id'] < 100).sum()), int((late['id'] < 100).sum()))
        self.assertLess(int((early['id'] >= 400).sum()), int((late['id'] >= 400).sum()))

    def test_training_refuses_missing_or_wrong_context_proofs(self):
        proof = {'seed': 8, 'context_index': 1, 'context_engine_verified': True,
                 'search_truncated': False}
        data = {'seeds': np.array([8, 8]), 'meta': {'source': 'generated_only',
                'oracle_search': 'complete_only', 'levels': [proof]}}
        train.require_verified_data(data)
        for field, value in [('context_index', 0), ('context_engine_verified', False),
                             ('search_truncated', True)]:
            malformed = copy.deepcopy(data)
            malformed['meta']['levels'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                train.require_verified_data(malformed)

    def test_grounding_terminal_budget_handles_all_generated_step_costs(self):
        from pebby.agent.world_grounding import labels
        batch = {'player_cell':[[1,2]]*4, 'current_triple':[[0,0,0]]*4,
                 'current_steps':[-3,-2,-1,42], 'current_lives':[3]*4}
        self.assertEqual(labels(batch)[4].tolist(),[0,0,0,43])

    def test_grounding_reaches_predictor_and_encoder_with_identical_recomputed_gradients(self):
        config = {**tiny_world().config(), 'grounding':True}
        model = wm.build_world_policy(config)
        other = copy.deepcopy(model)
        other.checkpoint_encoder = other.checkpoint_loops = True
        other.encoder_chunk_size = 1
        rows, _ = collect_level(generate_level(0,1), history=3, samples=2)
        batch = {key:torch.as_tensor(np.stack([row[key] for row in rows])) for key in rows[0]}
        for policy in (model,other):
            torch.manual_seed(1)
            out = wm.world_losses(policy,batch)
            self.assertIn('latent_rotation_accuracy',out['diagnostics'])
            out['losses']['grounding'].backward()
            for part in (policy.stem,policy.projector,policy.predictor,policy.grounding_head):
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in part.parameters()
                                       if p.grad is not None),0.)
        for first,second in zip(model.parameters(),other.parameters()):
            if first.grad is not None:
                torch.testing.assert_close(first.grad,second.grad,atol=1e-6,rtol=1e-5)
        # Earlier checkpoint configurations still load exactly without new heads.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.pt'
            saved = wm.save_world_checkpoint(path,tiny_world())
            saved['config'].pop('grounding')
            torch.save(saved,path)
            self.assertFalse(wm.load_world_checkpoint(path)[0].cfg.grounding)

    def test_dataset_rejects_zero_masks_and_malformed_padding(self):
        arrays = {'frames':np.zeros((2,3,64,64),dtype=np.uint8),
                  'history_valid':np.array([[False,True,True]]*2),
                  'previous_actions':np.array([[-1,-1,0]]*2),
                  'next_frames':np.zeros((2,4,64,64),dtype=np.uint8),
                  'terminal':np.zeros((2,4),dtype=bool),'won':np.zeros((2,4),dtype=bool),
                  'optimal':np.ones(2,dtype=np.uint8),'distances':np.ones((2,4),dtype=np.int16),
                  'seeds':np.array([1,2]), 'lost_life':np.zeros((2,4),dtype=bool)}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'shard.npz'
            np.savez(path,**arrays)
            loaded=train.load_dataset(path)
            self.assertIn('lost_life',train.as_tensors(loaded))
            for key,value in [('optimal',np.zeros(2,dtype=np.uint8)),
                              ('previous_actions',np.array([[0,-1,0]]*2)),
                              ('history_valid',np.array([[True,False,True]]*2))]:
                np.savez(path,**{**arrays,key:value})
                with self.subTest(key=key),self.assertRaises(ValueError):
                    train.load_dataset(path)
