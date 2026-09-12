"""Regression tests for the LS20 looped world-model policy and its trainer.

Everything runs on the CPU in well under two minutes. The synthetic gridworld
below follows the NPZ data contract of ``pebby.agent.world_train`` exactly
(walls, a goal, BFS distances, four counterfactual successors per state and
left-padded histories) but is not LS20: no official level, layout or label is
read here.
"""

from collections import deque
from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pebby.agent import world_model as wm
from pebby.agent import world_train as trainer
from pebby.agent.world_model import (ACTION_COUNT, CELLS, GRID_COLS, GRID_ROWS, WORLD_MODEL_FORMAT,
                                     WorldModelConfig, WorldPolicy, world_losses)

TINY = dict(channels=16, blocks=1, heads=4, expansion=2, loops=2, history=4, temporal_layers=1,
            hud_channels=8, hud_tokens=4, latent=16, reduce=2, predictor_blocks=1, predictor_hidden=32,
            value_hidden=16, max_distance=16, lookahead_depth=1, summary=4, readout_hidden=8,
            ranker_hidden=8, sigreg_projections=32, sigreg_knots=9)

DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0))  # up, down, left, right in (dx, dy)


def make_model(**overrides):
    return WorldPolicy(WorldModelConfig(**{**TINY, **overrides}))


# ------------------------------------------------------------ synthetic world
def render(walls, goal, player, remaining, token_colour, fog_radius=None):
    frame = np.zeros((64, 64), dtype=np.uint8)
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            y, x = wm.Y_ORIGIN + 5 * row, wm.X_ORIGIN + 5 * col
            if walls[row, col]:
                frame[y:y + 5, x:x + 5] = 1
            if (col, row) == goal:
                frame[y + 1:y + 4, x + 1:x + 4] = 9
    if fog_radius is not None:
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                if abs(col - player[0]) + abs(row - player[1]) > fog_radius:
                    y, x = wm.Y_ORIGIN + 5 * row, wm.X_ORIGIN + 5 * col
                    frame[y:y + 5, x:x + 5] = 3
    y, x = wm.Y_ORIGIN + 5 * player[1], wm.X_ORIGIN + 5 * player[0]
    frame[y:y + 5, x:x + 5] = 12
    frame[52, :] = 4
    frame[55:61, 3:9] = token_colour
    frame[61:63, 13:13 + min(41, remaining)] = 5
    return frame


def bfs(walls, goal):
    distance = np.full((GRID_ROWS, GRID_COLS), -1, dtype=np.int64)
    distance[goal[1], goal[0]] = 0
    queue = deque([goal])
    while queue:
        col, row = queue.popleft()
        for dx, dy in DELTAS:
            ncol, nrow = col + dx, row + dy
            if 0 <= ncol < GRID_COLS and 0 <= nrow < GRID_ROWS and not walls[nrow, ncol] \
                    and distance[nrow, ncol] < 0:
                distance[nrow, ncol] = distance[row, col] + 1
                queue.append((ncol, nrow))
    return distance


def step(walls, player, action):
    dx, dy = DELTAS[action]
    col, row = player[0] + dx, player[1] + dy
    if 0 <= col < GRID_COLS and 0 <= row < GRID_ROWS and not walls[row, col]:
        return (col, row)
    return player


def make_synthetic(seed=0, levels=4, steps=12, history=4, fog_radius=None, budget=40, player_cell=True):
    """States along random walks in random wall mazes, in the NPZ contract."""
    rng = np.random.default_rng(seed)
    rows = {name: [] for name in ("frames", "history_valid", "previous_actions", "next_frames",
                                  "terminal", "won", "optimal", "distances", "player_cell", "seeds")}
    for level in range(levels):
        while True:
            walls = rng.random((GRID_ROWS, GRID_COLS)) < .18
            free = np.argwhere(~walls)
            goal = tuple(int(v) for v in free[rng.integers(len(free))][::-1])
            distance = bfs(walls, goal)
            reachable = np.argwhere(distance > 0)
            if len(reachable) >= 20:
                break
        token = int(rng.integers(8, 15))
        player = tuple(int(v) for v in reachable[rng.integers(len(reachable))][::-1])
        remaining = budget
        frames, actions = [render(walls, goal, player, remaining, token, fog_radius)], [-1]
        for _ in range(steps):
            window = frames[-history:]
            padding = history - len(window)
            rows["frames"].append(np.stack([window[0]] * padding + window))
            rows["history_valid"].append(np.array([False] * padding + [True] * len(window)))
            rows["previous_actions"].append(np.array([-1] * padding + actions[-history:], dtype=np.int64))
            successors, terminal, won, dists = [], [], [], []
            for action in range(ACTION_COUNT):
                nxt = step(walls, player, action)
                successors.append(render(walls, goal, nxt, remaining - 1, token, fog_radius))
                dists.append(int(distance[nxt[1], nxt[0]]))
                won.append(nxt == goal)
                terminal.append(nxt == goal)
            valid = [d for d in dists if d >= 0]
            best = min(valid) if valid else None
            rows["optimal"].append(sum(1 << a for a, d in enumerate(dists) if best is not None and d == best))
            rows["next_frames"].append(np.stack(successors))
            rows["terminal"].append(np.array(terminal))
            rows["won"].append(np.array(won))
            rows["distances"].append(np.array(dists, dtype=np.int16))
            rows["player_cell"].append(np.array(player, dtype=np.int16))
            rows["seeds"].append(level)
            # Mostly-optimal walk with noise so the states cover the maze.
            choices = [a for a, d in enumerate(dists) if d == best] if rng.random() < .7 else list(range(4))
            action = int(rng.choice(choices))
            player = step(walls, player, action)
            remaining -= 1
            if player == goal:
                player = tuple(int(v) for v in reachable[rng.integers(len(reachable))][::-1])
                frames, actions, remaining = [render(walls, goal, player, budget, token, fog_radius)], [-1], budget
                continue
            frames.append(render(walls, goal, player, remaining, token, fog_radius))
            actions.append(action)
    data = {name: np.stack(values) for name, values in rows.items()}
    data["frames"] = data["frames"].astype(np.uint8)
    data["next_frames"] = data["next_frames"].astype(np.uint8)
    data["optimal"] = data["optimal"].astype(np.uint8)
    data["seeds"] = data["seeds"].astype(np.int32)
    if not player_cell:
        del data["player_cell"]
    return data


def to_batch(data, index=None):
    tensors = trainer.as_tensors({**data, "player_cell": data.get("player_cell")})
    if index is None:
        return tensors
    return {name: value[index] for name, value in tensors.items()}


class WorldModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._threads = torch.get_num_threads()
        torch.set_num_threads(4)
        cls.data = make_synthetic(seed=1, levels=3, steps=10, history=4)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls._threads)

    def setUp(self):
        torch.manual_seed(11)

    # ------------------------------------------------------------- contracts
    def test_geometry_constants_match_ls20_names(self):
        from pebby.ls20 import names
        self.assertEqual((wm.FRAME_SIZE, wm.CELL, wm.X_ORIGIN, wm.Y_ORIGIN, wm.GRID_ROWS, wm.GRID_COLS),
                         (names.FRAME_SIZE, names.CELL, names.X_ORIGIN, names.Y_ORIGIN,
                          names.GRID_ROWS, names.GRID_COLS))
        self.assertEqual(ACTION_COUNT, len(names.ACTION_IDS))
        self.assertEqual(DELTAS, names.ACTION_DELTAS)

    def test_config_validation_and_weight_sharing_across_loops(self):
        for bad in (dict(channels=15, heads=4), dict(loops=0), dict(history=True), dict(sigreg_knots=2),
                    dict(grounding=1), dict(state_recall="yes")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                WorldModelConfig(**{**TINY, **bad})
        with self.assertRaisesRegex(ValueError, "unknown config keys"):
            WorldModelConfig.from_dict({**TINY, "architecture": "world", "bogus": 1})
        with self.assertRaisesRegex(ValueError, "not a world-policy config"):
            WorldModelConfig.from_dict({"architecture": "looped"})
        counts = {make_model(loops=depth).parameter_count() for depth in (1, 8)}
        self.assertEqual(len(counts), 1)
        model = make_model().eval()
        calls = []
        handle = model.core[0].register_forward_hook(lambda module, args, output: calls.append(args[1]))
        batch = to_batch(self.data, slice(0, 2))
        with torch.inference_mode():
            shallow = model(batch["frames"], batch["history_valid"], batch["previous_actions"], loops=1)
            deep = model(batch["frames"], batch["history_valid"], batch["previous_actions"], loops=5)
        handle.remove()
        self.assertEqual(len(calls), 6)
        self.assertTrue(all(source is calls[-1] for source in calls[1:]), "every loop recalls one source")
        self.assertEqual(tuple(shallow.shape), (2, ACTION_COUNT))
        self.assertFalse(torch.allclose(shallow, deep))
        self.assertEqual(model.config()["architecture"], "world")
        self.assertEqual(WorldModelConfig.from_dict(model.config()), model.cfg)

    def test_forward_contract_single_frame_batch_independence_and_no_hidden_state(self):
        model = make_model().eval()
        batch = to_batch(self.data, slice(0, 3))
        with torch.inference_mode():
            full = model(batch["frames"], batch["history_valid"], batch["previous_actions"])
            single = model(batch["frames"][1:2], batch["history_valid"][1:2], batch["previous_actions"][1:2])
            self.assertTrue(torch.allclose(full[1:2], single, atol=1e-5))
            current_only = model(batch["frames"][:, -1])  # [B, 64, 64] -> history of one
            self.assertEqual(tuple(current_only.shape), (3, ACTION_COUNT))
            other = model(torch.full((3, 64, 64), 15, dtype=torch.uint8))
            again = model(batch["frames"], batch["history_valid"], batch["previous_actions"])
        self.assertTrue(torch.equal(full, again), "no state may survive between calls")
        self.assertFalse(torch.allclose(current_only, other))
        for bad in ((64, 64), (2, 63, 64), (2, 5, 64, 64)):
            with self.subTest(shape=bad), self.assertRaises(ValueError):
                model(torch.zeros(bad, dtype=torch.uint8))
        with self.assertRaisesRegex(ValueError, "current .* must be valid"):
            model(batch["frames"], torch.zeros(3, 4, dtype=torch.bool), batch["previous_actions"])
        with self.assertRaisesRegex(ValueError, "previous_actions"):
            model(batch["frames"], batch["history_valid"], torch.full((3, 4), 4))

    def test_temporal_history_and_previous_actions_change_logits_but_padding_does_not(self):
        model = make_model().eval()
        data = make_synthetic(seed=5, levels=1, steps=8, history=4, fog_radius=2)
        batch = to_batch(data, slice(6, 8))  # histories with all four slots valid
        self.assertTrue(bool(batch["history_valid"].all()))
        with torch.inference_mode():
            base = model(batch["frames"], batch["history_valid"], batch["previous_actions"])
            frames = batch["frames"].clone()
            frames[:, 0] = torch.full_like(frames[:, 0], 7)  # oldest, valid
            self.assertFalse(torch.allclose(base, model(frames, batch["history_valid"],
                                                        batch["previous_actions"])))
            valid = batch["history_valid"].clone()
            valid[:, 0] = False
            masked = model(batch["frames"], valid, batch["previous_actions"])
            masked_again = model(frames, valid, batch["previous_actions"])
            self.assertTrue(torch.allclose(masked, masked_again, atol=1e-6),
                            "a padded slot must not influence the output")
            actions = batch["previous_actions"].clone()
            actions[:, -2] = (actions[:, -2] + 1) % ACTION_COUNT
            self.assertFalse(torch.allclose(base, model(batch["frames"], batch["history_valid"], actions)))

    # ------------------------------------------------------------- LeWM parts
    def test_sigreg_prefers_gaussian_over_collapse_and_restores_variance(self):
        regularizer = wm.SIGReg(projections=128, knots=17)
        generator = torch.Generator().manual_seed(0)
        gaussian = torch.randn(2, 512, 32, generator=generator)
        collapsed = torch.zeros(2, 512, 32)
        squeezed = gaussian * .05
        low_rank = gaussian[..., :1].expand(-1, -1, 32).clone()
        torch.manual_seed(0)
        gaussian_score = regularizer(gaussian).item()
        for name, sample in (("collapsed", collapsed), ("squeezed", squeezed), ("low_rank", low_rank)):
            torch.manual_seed(0)
            with self.subTest(name=name):
                self.assertGreater(regularizer(sample).item(), 5 * gaussian_score)
        # Under the null the scaled Epps-Pulley statistic has expectation of order one.
        self.assertLess(gaussian_score, 3.)
        # Free embeddings starting near a constant: gradient steps must spread them.
        embeddings = (torch.randn(1, 256, 8, generator=generator) * .01).requires_grad_()
        optimizer = torch.optim.Adam([embeddings], lr=.05)
        before = embeddings.var(1).mean().item()
        for _ in range(60):
            optimizer.zero_grad()
            regularizer(embeddings).backward()
            optimizer.step()
        self.assertGreater(embeddings.var(1).mean().item(), 50 * before)
        with self.assertRaises(ValueError):
            regularizer(torch.zeros(2, 3, 4, 5))

    def test_target_encoder_gets_gradients_and_there_is_no_teacher_copy(self):
        model = make_model().train()
        names = [name for name, _ in model.named_modules()]
        self.assertFalse(any("ema" in name or "teacher" in name or "target" in name for name in names))
        self.assertEqual(sum(1 for name in names if name.startswith("stem")), 1 + len(model.stem))
        out = world_losses(model, to_batch(self.data, slice(0, 4)), {"sigreg": .1})
        # Gradient of the prediction term through the TARGET branch only.
        prediction = torch.nn.functional.mse_loss(out["predicted"].detach(), out["targets"])
        grads = torch.autograd.grad(prediction, [model.stem[0].weight, model.core[0].mlp[0].weight,
                                                 model.projector[0].weight, model.temporal[0].mlp[0].weight],
                                    retain_graph=True, allow_unused=True)
        for name, grad in zip(("stem", "core", "projector", "temporal"), grads):
            with self.subTest(name=name):
                self.assertIsNotNone(grad)
                self.assertGreater(grad.abs().sum().item(), 0.)
        # ...and through the prediction branch into the predictor and the encoder.
        prediction = torch.nn.functional.mse_loss(out["predicted"], out["targets"].detach())
        grads = torch.autograd.grad(prediction, [model.predictor.blocks[0].modulation[-1].weight,
                                                 model.stem[0].weight], retain_graph=True)
        self.assertTrue(all(g.abs().sum().item() > 0 for g in grads))
        # SIGReg itself reaches the encoder through every slot.
        slots = torch.cat((out["latent"][None], out["targets"].transpose(0, 1)))
        self.assertEqual(tuple(slots.shape)[:2], (5, 4))
        grad = torch.autograd.grad(model.sigreg(slots), model.stem[0].weight)[0]
        self.assertGreater(grad.abs().sum().item(), 0.)

    def test_labels_and_next_frames_never_reach_the_logits(self):
        model = make_model().eval()
        batch = to_batch(self.data, slice(0, 4))
        with torch.no_grad():
            reference = model(batch["frames"], batch["history_valid"], batch["previous_actions"])
            out = world_losses(model, batch)
            self.assertTrue(torch.allclose(out["logits"], reference, atol=1e-5))
            shuffled = dict(batch)
            for name in ("next_frames", "terminal", "won", "optimal", "distances", "player_cell"):
                shuffled[name] = batch[name].flip(0)
            shuffled["next_frames"] = torch.full_like(batch["next_frames"], 6)
            leaked = world_losses(model, shuffled)
            self.assertTrue(torch.allclose(leaked["logits"], reference, atol=1e-5))
            self.assertFalse(torch.allclose(leaked["losses"]["policy"], out["losses"]["policy"]))
            self.assertFalse(torch.allclose(leaked["targets"], out["targets"]))
        without_player = {name: value for name, value in batch.items() if name != "player_cell"}
        self.assertNotIn("player", world_losses(model, without_player)["losses"])

    def test_lost_life_targets_restart_the_history_at_the_reset_frame(self):
        model = make_model().eval()
        batch = to_batch(make_synthetic(seed=2, levels=1, steps=8, history=4), slice(4, 8))
        batch["lost_life"] = torch.zeros(4, ACTION_COUNT, dtype=torch.bool)
        batch["lost_life"][:, 2] = True
        with torch.no_grad():
            out = world_losses(model, batch)
            plain = world_losses(model, {k: v for k, v in batch.items() if k != "lost_life"})
            solo = model.encode(batch["next_frames"][:, 2])["latent"]  # reset frame alone, H = 1
        self.assertTrue(torch.allclose(out["targets"][:, 2], solo, atol=1e-5))
        self.assertTrue(torch.allclose(out["targets"][:, 0], plain["targets"][:, 0], atol=1e-6))
        self.assertFalse(torch.allclose(out["targets"][:, 2], plain["targets"][:, 2]))
        with self.assertRaisesRegex(ValueError, "lost_life"):
            world_losses(model, {**batch, "lost_life": torch.zeros(4, 3, dtype=torch.bool)})

    def test_successors_differ_per_action_and_learned_dynamics_rank_actions(self):
        model = make_model(lookahead_depth=2).eval()
        with torch.no_grad():  # AdaLN-zero starts as the identity; wake the gates up.
            for block in model.predictor.blocks:
                torch.nn.init.normal_(block.modulation[-1].weight, std=.5)
        batch = to_batch(self.data, slice(0, 3))
        with torch.no_grad():
            encoding = model.encode(batch["frames"], batch["history_valid"], batch["previous_actions"])
            latent = encoding["latent"]
            successors = model.predict_successors(latent)
            self.assertEqual(tuple(successors.shape), (3, ACTION_COUNT, TINY["latent"]))
            for a in range(ACTION_COUNT):
                for b in range(a + 1, ACTION_COUNT):
                    self.assertFalse(torch.allclose(successors[:, a], successors[:, b]))
                    self.assertFalse(torch.allclose(successors[0, a], successors[1, a]))
            single = model.predict_successors(latent, torch.tensor([2, 0, 3]))
            self.assertTrue(torch.allclose(single[0], successors[0, 2]))
            two_step = model.predict_successors(successors[:, 1], torch.tensor([[0, 1]] * 3))
            self.assertEqual(tuple(two_step.shape), (3, 2, TINY["latent"]))
            logits, extra = model.logits_from(encoding)
            self.assertEqual(tuple(extra["features"].shape), (3, ACTION_COUNT, 2 * model.feature_size))
            self.assertEqual(tuple(logits.shape), (3, ACTION_COUNT))
            shallow, _ = model.lookahead(latent, depth=1)
            self.assertEqual(tuple(shallow.shape), (3, ACTION_COUNT, model.feature_size))
            self.assertTrue(torch.allclose(shallow, extra["features"][..., :model.feature_size]))
        # Changing only the predictor must change the ranking: dynamics influence the policy.
        altered = copy.deepcopy(model)
        with torch.no_grad():
            for block in altered.predictor.blocks:
                block.modulation[-1].weight.mul_(-1.)
        with torch.no_grad():
            altered_logits = altered(batch["frames"], batch["history_valid"], batch["previous_actions"])
            direct = extra["direct"]
        self.assertFalse(torch.allclose(altered_logits, logits))
        self.assertTrue(torch.allclose(altered.direct_logits(encoding["cells"])[0], direct))
        self.assertFalse(torch.allclose(logits - direct, altered_logits - direct))
        # The lookahead term differs across actions and across inputs, not just a bias.
        ranking = logits - direct
        self.assertGreater((ranking - ranking.mean(-1, keepdim=True)).abs().max().item(), 1e-6)
        self.assertFalse(torch.allclose(ranking[0], ranking[1]))
        # Player-relative readout: no parameter is indexed by an absolute cell position.
        self.assertEqual(model.player_head.weight.shape, (1, TINY["channels"]))
        self.assertEqual(model.move_head[-1].weight.shape[0], ACTION_COUNT)

    # --------------------------------------------------------------------- IO
    def test_checkpoint_round_trip_and_rejections(self):
        model = make_model().eval()
        batch = to_batch(self.data, slice(0, 2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "world.pt"
            saved = wm.save_world_checkpoint(path, model, epochs=1, train_seeds=[1])
            restored, loaded = wm.load_world_checkpoint(path)
            self.assertEqual(loaded["format"], WORLD_MODEL_FORMAT)
            self.assertEqual(loaded["config"], model.config())
            self.assertEqual(loaded["parameters"], model.parameter_count())
            self.assertFalse(restored.training)
            with torch.inference_mode():
                self.assertTrue(torch.equal(
                    model(batch["frames"], batch["history_valid"], batch["previous_actions"]),
                    restored(batch["frames"], batch["history_valid"], batch["previous_actions"])))
            with self.assertRaisesRegex(ValueError, "reserved"):
                wm.save_world_checkpoint(path, model, config={})
            foreign = dict(saved, format="pebby.ls20-looped-policy.v1")
            torch.save(foreign, Path(directory) / "foreign.pt")
            with self.assertRaisesRegex(ValueError, "unsupported checkpoint format"):
                wm.load_world_checkpoint(Path(directory) / "foreign.pt")
            mismatched = dict(saved, config={**saved["config"], "latent": 32})
            torch.save(mismatched, Path(directory) / "mismatched.pt")
            with self.assertRaises((ValueError, RuntimeError)):
                wm.load_world_checkpoint(Path(directory) / "mismatched.pt")

    def test_parameter_groups_decay_matrices_only(self):
        model = make_model()
        groups = wm.parameter_groups(model, .05)
        decayed = {id(p) for p in groups[0]["params"]}
        for name, parameter in model.named_parameters():
            with self.subTest(name=name):
                is_matrix = parameter.ndim >= 2 and "position" not in name and "embedding" not in name
                self.assertEqual(id(parameter) in decayed, is_matrix)
        self.assertEqual(groups[1]["weight_decay"], 0.)
        self.assertEqual(sum(len(g["params"]) for g in groups), len(list(model.parameters())))

    # ------------------------------------------------------------- optimisation
    def test_real_optimizer_steps_fit_a_tiny_world(self):
        torch.manual_seed(3)
        data = make_synthetic(seed=7, levels=2, steps=16, history=2)
        model = make_model(history=2, loops=2, latent=32, max_distance=16).train()
        tensors = to_batch(data)
        optimizer = torch.optim.AdamW(wm.parameter_groups(model, .01), lr=3e-3)
        weights = {"prediction": 1., "sigreg": .1, "policy": 1., "value": .5, "imagined_value": .5, "player": .1}
        first, last = None, None
        # Measured: the predictor overtakes the copy baseline after ~100 steps here.
        # With only the two LeWM terms on this tiny two-level set the encoder settles on
        # successor == current (prediction == copy MSE); the value heads break that tie.
        for step_index in range(150):
            out = world_losses(model, tensors, weights)
            optimizer.zero_grad(set_to_none=True)
            out["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            record = {name: float(value.detach())
                      for name, value in {**out["losses"], **out["diagnostics"]}.items()}
            first = first or record
            last = record
        prior = trainer.prior_baseline(data["optimal"])["train"]
        self.assertLess(last["policy"], first["policy"])
        self.assertLess(last["policy"], prior["policy_cross_entropy"])
        self.assertGreater(last["set_accuracy"], prior["set_accuracy"])
        self.assertLess(last["prediction"], last["copy_mse"], "the predictor must beat copying z_t")
        self.assertGreater(last["counterfactual_top1"], .5)
        self.assertGreater(last["target_variance_mean"], 10 * first["target_variance_mean"])
        self.assertGreater(last["lookahead_span"], 0.)
        self.assertLess(last["sigreg"], first["sigreg"])

    def test_loop_checkpointing_matches_full_backpropagation(self):
        model = make_model().train()
        batch = to_batch(self.data, slice(0, 3))
        torch.manual_seed(0)
        world_losses(model, batch, sigreg_generator=torch.Generator().manual_seed(1))["total"].backward()
        reference = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        model.checkpoint_loops = True
        torch.manual_seed(0)
        world_losses(model, batch, sigreg_generator=torch.Generator().manual_seed(1))["total"].backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.allclose(parameter.grad, reference[name], atol=1e-6), name)
        self.assertGreater(model.activation_estimate_gib(64), 0.)

    # ------------------------------------------------------------------ trainer
    def test_trainer_cli_trains_reports_and_refuses_bad_data(self):
        train = make_synthetic(seed=21, levels=2, steps=8, history=4)
        validation = make_synthetic(seed=22, levels=1, steps=6, history=4)
        validation["seeds"] = validation["seeds"] + 100
        overlapping = dict(validation, seeds=np.zeros_like(validation["seeds"]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, payload in (("train", train), ("validation", validation), ("overlap", overlapping)):
                np.savez(root / f"{name}.npz", meta=np.array(json.dumps({"source": "synthetic"})), **payload)
            truncated = {name: value for name, value in train.items() if name != "distances"}
            np.savez(root / "truncated.npz", **truncated)
            flags = ["--epochs", "2", "--batch-size", "8", "--device", "cpu", "--seed", "0",
                     "--checkpoint-out", str(root / "out" / "world.pt"), "--report-out", str(root / "report.json")]
            torch.manual_seed(0)
            initial_hash = trainer.initial_state_sha256(make_model())
            flags += ['--require-fresh-initialization','--expected-initial-state-sha256',initial_hash,
                      '--temporal-backend','math']
            for key, value in TINY.items():
                flags += [f"--{key.replace('_', '-')}", str(value)]
            buffer = io.StringIO()
            with redirect_stdout(buffer), redirect_stderr(buffer):
                code = trainer.main(["--train", str(root / "train.npz"), "--validation",
                                     str(root / "validation.npz")] + flags)
            self.assertEqual(code, 0, buffer.getvalue())
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["format"], WORLD_MODEL_FORMAT)
            self.assertEqual(len(report["history"]), 2)
            for split in ("train", "validation"):
                stats = report["history"][0][split]
                for key in ("prediction", "sigreg", "policy", "value", "imagined_value", "player",
                            "target_variance_mean", "counterfactual_top1", "copy_top1", "copy_mse",
                            "set_accuracy"):
                    self.assertIn(key, stats)
            self.assertEqual(report["verdict"]["criterion"], "set_accuracy")
            self.assertEqual(report["verdict"]["split"], "validation")
            self.assertIn("beats_copy_baseline", report["verdict"])
            self.assertIn("prior", report["baseline"])
            self.assertEqual(report["validation_seeds"], [100])
            model, checkpoint = wm.load_world_checkpoint(root / "out" / "world.pt")
            self.assertEqual(checkpoint["best_epoch"], report["best"]["epoch"])
            self.assertEqual(checkpoint["config"], report["config"])
            self.assertEqual(checkpoint['initialization'],report['initialization'])
            self.assertEqual(checkpoint['initialization']['kind'],'random')
            self.assertEqual(checkpoint['initialization']['weights_sha256'],initial_hash)
            self.assertEqual(checkpoint['execution']['temporal_backend'],'math')
            with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    trainer.main(['--train',str(root/'train.npz'),*flags,
                                  '--expected-initial-state-sha256','0'*64])
            with torch.inference_mode():
                self.assertEqual(tuple(model(torch.from_numpy(train["frames"][:2])).shape), (2, 4))
            for name, extra in (("overlap", ["--validation", str(root / "overlap.npz")]),
                                ("truncated", []), ("missing", [])):
                source = {"overlap": "train", "truncated": "truncated", "missing": "nope"}[name]
                with self.subTest(name=name), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as failure:
                        trainer.main(["--train", str(root / f"{source}.npz")] + extra + flags)
                    self.assertNotEqual(failure.exception.code, 0)
            # No validation split: selection falls back to the training split.
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = trainer.main(["--train", str(root / "train.npz"), "--max-states", "10"] + flags)
            self.assertEqual(code, 0)
            self.assertEqual(json.loads((root / "report.json").read_text())["verdict"]["split"], "train")


if __name__ == "__main__":
    unittest.main()
