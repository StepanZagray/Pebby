"""The HUD decoder must equal the engine's own counters on frames it rendered.

Frames come from oracle routes (with refills), random walks that exhaust
tanks and lives, and the shipped levels; no privileged state is decoded, the
engine values are only the assertion target.
"""
from pathlib import Path
import unittest

import numpy as np
import torch

from pebby.agent.evaluate import shipped_levels
from pebby.agent.hud_decoder import (HUD_SCALARS, MAX_STEPS, PIP_LIT, STEP_FILLED,
                                     decode_counts, decode_hud, decode_hud_numpy)
from pebby.ls20.bank import load
from pebby.ls20.env import Ls20Scenario
from pebby.ls20.generate import build_level

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / 'data/ls20-reference-unequal-v1/validation.jsonl'


def walk(env, actions, frames, truth, events):
    """Perform actions, keeping every observation frame with the engine's counters."""
    for action in actions:
        before = env.steps_left(), env.lives()
        observation = env.perform(int(action))
        if observation.frame is None:
            break
        steps, lives = env.steps_left(), env.lives()
        frames.append(np.array(observation.frame))
        truth.append((max(steps, 0), lives))
        if steps > before[0]:
            events['refill' if lives == before[1] else 'life_lost'] += 1
        elif lives < before[1]:
            events['life_lost'] += 1
        if steps < 0:
            events['negative'] += 1
        if observation.finished:
            break


@unittest.skipUnless(BANK.exists(), 'reference validation bank not present')
class HudDecoderEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        cls.frames, cls.truth = [], []
        cls.events = dict(refill=0, life_lost=0, negative=0)
        rows = [row for row in load(BANK)][:4]
        for row in rows:
            env = Ls20Scenario(build_level(row), row['training_context_index'])
            cls.frames.append(np.array(env.render()))
            cls.truth.append((env.steps_left(), env.lives()))
            walk(env, row['solution'], cls.frames, cls.truth, cls.events)
            env = Ls20Scenario(build_level(row), row['training_context_index'])
            walk(env, rng.integers(1, 5, size=160), cls.frames, cls.truth, cls.events)
        for index, level in enumerate(shipped_levels()[:3]):
            env = Ls20Scenario(level, index)
            cls.frames.append(np.array(env.render()))
            cls.truth.append((env.steps_left(), env.lives()))
            walk(env, rng.integers(1, 5, size=100), cls.frames, cls.truth, cls.events)
        cls.frames = np.stack(cls.frames)
        cls.steps = np.array([steps for steps, _ in cls.truth])
        cls.lives = np.array([lives for _, lives in cls.truth])

    def test_corpus_is_diverse(self):
        self.assertGreaterEqual(len(self.frames), 300)
        self.assertEqual(set(np.unique(self.lives)) & {1, 2, 3}, {1, 2, 3})
        self.assertIn(0, self.steps)
        self.assertIn(MAX_STEPS, self.steps)
        self.assertGreater(len(np.unique(self.steps)), 20)
        self.assertGreater(self.events['refill'], 0)
        self.assertGreater(self.events['life_lost'], 0)

    def test_exact_equality_with_engine_counters(self):
        steps, lives = decode_counts(torch.as_tensor(self.frames))
        np.testing.assert_array_equal(steps.numpy(), self.steps)
        np.testing.assert_array_equal(lives.numpy(), self.lives)
        expected = np.zeros((len(self.frames), HUD_SCALARS), dtype=np.float32)
        expected[:, 0] = self.steps / MAX_STEPS
        for count in (1, 2, 3):
            expected[:, count] = self.lives == count
        decoded = decode_hud(torch.as_tensor(self.frames))
        self.assertEqual(decoded.dtype, torch.float32)
        self.assertEqual(decoded.shape, (len(self.frames), HUD_SCALARS))
        np.testing.assert_array_equal(decoded.numpy(), expected)
        np.testing.assert_array_equal(decode_hud_numpy(self.frames), expected)

    def test_input_shapes_and_last_frame_selection(self):
        single = decode_hud(self.frames[0])
        self.assertEqual(single.shape, (1, HUD_SCALARS))
        torch.testing.assert_close(single, decode_hud(torch.as_tensor(self.frames[:1])), atol=0, rtol=0)
        window = torch.as_tensor(self.frames[:16]).reshape(2, 8, 64, 64)
        torch.testing.assert_close(decode_hud(window), decode_hud(window[:, -1]), atol=0, rtol=0)
        torch.testing.assert_close(decode_hud(self.frames[0].tolist()), single, atol=0, rtol=0)
        self.assertEqual(decode_hud(self.frames[:5].astype(np.uint8)).shape, (5, HUD_SCALARS))
        for bad in (torch.zeros(64, 64), torch.zeros(3, 64, 63, dtype=torch.long),
                    torch.zeros(2, 2, 2, 64, 64, dtype=torch.long)):
            with self.subTest(shape=tuple(bad.shape), dtype=bad.dtype), self.assertRaises(ValueError):
                decode_hud(bad)
        with self.assertRaises(ValueError):
            decode_hud_numpy(np.zeros((2, 64, 64), dtype=np.float32))

    def test_negative_budget_clamps_and_flash_frames_decode_without_pips(self):
        frame = self.frames[0].copy()
        frame[61:63, 13:55] = 3  # empty bar: the engine draws this for steps_left <= 0
        self.assertEqual(decode_counts(frame)[0].item(), 0)
        torch.testing.assert_close(decode_hud(frame)[0, 0], torch.tensor(0.), atol=0, rtol=0)
        flash = np.full((64, 64), STEP_FILLED, dtype=np.int64)
        torch.testing.assert_close(decode_hud(flash), torch.tensor([[1., 0., 0., 0.]]), atol=0, rtol=0)
        frame[61:63, 56:64] = PIP_LIT  # a lit inter-pip column must not count as a pip
        self.assertEqual(decode_counts(frame)[1].item(), 3)
        frame[61:63, 58] = PIP_LIT
        frame[61:63, 62:64] = 3
        self.assertEqual(decode_counts(frame)[1].item(), 2)


if __name__ == '__main__':
    unittest.main()
