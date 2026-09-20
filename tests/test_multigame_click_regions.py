"""Engine-verified equivalent-click regions: collector labels, loss, loading, training."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from arcengine import Camera, Level, Sprite

from pebby import multigame as M
from pebby.agent import multigame_training as T
from pebby.agent.multigame_model import (
    ACTION_COUNT,
    CLICK_ACTION,
    FRAME_SIZE,
    LossWeights,
    MultiGameModel,
    MultiGameModelConfig,
    click_region_nll,
    click_region_or_exact,
    collate_game_sequences,
    compute_multigame_loss,
    load_supervised_game,
)
from pebby.multigame_variants import WholeGameVariant, transform_frame, transform_xy


# --------------------------------------------------------------------------
# A synthetic clickable family: a 16x16 grid of 4x4-display-pixel cells.
# Clicking any pixel of a cell toggles that cell, so every cell is one
# equivalence class of 16 display pixels; a level completes when its target
# cell is toggled on.  The engine exposes a real arcengine camera/level so the
# collector's sprite hit-test candidate path is exercised, not mocked.
# --------------------------------------------------------------------------

GRID = 16
CELL = FRAME_SIZE // GRID  # 4 display pixels per cell


def _cell_centre(gx: int, gy: int) -> tuple[int, int]:
    return gx * CELL + CELL // 2, gy * CELL + CELL // 2


class _ClickGridEnv:
    def __init__(self, levels):
        self.levels = [dict(level) for level in levels]
        camera = Camera(width=GRID, height=GRID)
        sprites = [
            Sprite([[1]], name=f"cell-{gx}-{gy}", x=gx, y=gy, tags=["cell"])
            for gy in range(GRID) for gx in range(GRID)
        ]
        self.game = SimpleNamespace(camera=camera, current_level=Level(sprites=sprites))
        self.reset()

    def reset(self):
        self.state = "NOT_FINISHED"
        self.level_index = 0
        self.levels_completed = 0
        self.grid = np.zeros((GRID, GRID), dtype=np.uint8)

    @property
    def available_actions(self):
        return (CLICK_ACTION,)

    def render(self):
        return np.kron(self.grid, np.ones((CELL, CELL), dtype=np.uint8)) + 1

    def perform(self, action_id, x=None, y=None):
        assert action_id == CLICK_ACTION
        gx, gy = x // CELL, y // CELL
        self.grid[gy, gx] ^= 1
        target = tuple(self.levels[self.level_index]["target"])
        if self.grid[target[1], target[0]] == 1:
            self.levels_completed += 1
            if self.levels_completed == len(self.levels):
                self.state = "WIN"
            else:
                self.level_index = self.levels_completed
                self.grid[:] = 0

    def clone(self):
        other = object.__new__(type(self))
        other.levels = [dict(level) for level in self.levels]
        other.game = self.game  # static geometry; safe to share
        other.state = self.state
        other.level_index = self.level_index
        other.levels_completed = self.levels_completed
        other.grid = self.grid.copy()
        return other


def _click_modules(env_cls=_ClickGridEnv):
    def generate(seed, difficulty):
        return {
            "seed": seed,
            "difficulty": difficulty,
            "target": [3 + difficulty, 5],
            "coverage": {"click_grid": True},
            "omitted_mechanics": [],
            "generator_notes": "click-grid fixture",
            "limitations": ["test only"],
        }

    def search(env, limit):
        x, y = _cell_centre(*env.levels[env.level_index]["target"])
        return SimpleNamespace(
            actions=[(CLICK_ACTION, x, y)], truncated=False, unsupported=False,
            exact=True, expanded=1, reason="click the target cell",
        )

    return M.GameModules(
        source=M.source_for("cd82"),
        env=SimpleNamespace(Env=env_cls),
        generate=SimpleNamespace(generate=generate, build_level=lambda spec: dict(spec)),
        plan=SimpleNamespace(search=search),
        solver_adapter="work_limit",
    )


def _true_equivalence_set(game: M.MultiGameEnv, target: M.Action) -> np.ndarray:
    """Brute-force reference: every display pixel with the same clone+step successor."""
    reference = M._probe_from_raw(game, game.raw_env.clone())
    reference.perform(target)
    expected = M._successor_signature(reference)
    truth = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.bool_)
    for y in range(FRAME_SIZE):
        for x in range(FRAME_SIZE):
            probe = M._probe_from_raw(game, game.raw_env.clone())
            probe.perform(M.Action(CLICK_ACTION, x, y))
            truth[y, x] = M._successor_signature(probe) == expected
    return truth


def test_region_collects_every_engine_equivalent_pixel_and_rejects_neighbours():
    modules = _click_modules()
    game = M.MultiGameEnv.from_specs(modules, [modules.generate.generate(0, 1)])
    game.reset()
    target = M.Action(CLICK_ACTION, *_cell_centre(4, 5))
    before = game.render()

    region = M.equivalent_click_region(game, target)

    truth = _true_equivalence_set(game, target)
    assert truth.sum() == CELL * CELL  # one whole cell, several pixels -> one sprite
    assert region.size == CELL * CELL
    assert np.array_equal(region.mask, truth)
    # The 5x5 neighbourhood reaches into adjacent cells; clone+step rejected them.
    assert region.candidates > CELL * CELL
    assert region.sprite_candidates == CELL * CELL
    assert region.mask[target.y, target.x]
    assert not region.mask[target.y, target.x + CELL]
    # Probing never stepped the live engine.
    assert np.array_equal(game.render(), before)
    assert game.progress.levels_completed == 0


def test_region_probe_is_bounded_and_always_contains_the_target():
    modules = _click_modules()
    game = M.MultiGameEnv.from_specs(modules, [modules.generate.generate(0, 1)])
    game.reset()
    target = M.Action(CLICK_ACTION, *_cell_centre(4, 5))
    tiny = M.equivalent_click_region(game, target, limit=1)
    assert tiny.size == 1 and tiny.candidates == 1 and tiny.mask[target.y, target.x]
    capped = M.equivalent_click_region(game, target, limit=9)
    assert capped.candidates == 9 and 1 <= capped.size <= 9
    assert len(M.click_region_candidates(game.raw_env, target.x, target.y, limit=1000)) <= 25 + CELL * CELL
    with pytest.raises(ValueError, match="click target"):
        M.equivalent_click_region(game, M.Action(1, None, None))


def test_collector_writes_region_masks_sizes_and_stats_for_click_targets_only():
    modules = _click_modules()
    collected = M.collect_generated_game(
        modules, master_seed=0, game_index=0, difficulties=(1, 2),
    )
    assert collected.record["status"] == "won"
    teacher = collected.teacher
    mask = teacher["click_region_mask"]
    size = teacher["click_region_size"]
    assert mask.dtype == np.uint8 and mask.shape == (2, FRAME_SIZE, FRAME_SIZE)
    assert size.dtype == np.int16 and size.tolist() == [CELL * CELL] * 2
    assert np.array_equal(mask.sum(axis=(1, 2)), size)
    for row in range(2):
        tx, ty = int(teacher["target_action_x"][row]), int(teacher["target_action_y"][row])
        assert mask[row, ty, tx] == 1
        cell = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
        cell[(ty // CELL) * CELL:(ty // CELL + 1) * CELL, (tx // CELL) * CELL:(tx // CELL + 1) * CELL] = 1
        assert np.array_equal(mask[row], cell)
    stats = collected.record["click_region_stats"]
    assert stats["click_targets"] == stats["labelled"] == 2
    assert stats["probe_failures"] == 0
    assert stats["mean_size"] == stats["median_size"] == stats["max_size"] == CELL * CELL
    assert stats["probe_seconds"] > 0 and stats["probe_seconds_per_click"] > 0
    assert collected.record["click_region_probe_limit"] == M.CLICK_REGION_PROBE_LIMIT

    # The stored label survives the loader and its exact-target consistency checks.
    manifest = M.new_manifest(
        sources=[modules.source], explicit_subset=True, seed=0, games_per_source=1,
        difficulties=(1, 2), limits=M.SearchLimits(),
    )
    assert "click_region_mask" in manifest["teacher_schema"]


def test_non_click_targets_store_empty_regions_and_disabled_probe_stores_exact_only():
    modules = _click_modules()
    disabled = M.collect_generated_game(
        modules, master_seed=0, game_index=0, difficulties=(1,),
        rollout=M.RolloutOptions(click_region_probe_limit=0),
    )
    assert disabled.record["status"] == "won"
    assert "click_region_mask" not in disabled.teacher
    assert "click_region_size" not in disabled.teacher
    assert disabled.record["click_region_stats"]["labelled"] == 0
    with pytest.raises(ValueError, match="click_region_probe_limit"):
        M.RolloutOptions(click_region_probe_limit=-1)

    class NonClickEnv(_ClickGridEnv):
        @property
        def available_actions(self):
            return (1, CLICK_ACTION)

        def perform(self, action_id, x=None, y=None):
            if action_id == 1:
                self.levels_completed += 1
                self.state = "WIN" if self.levels_completed == len(self.levels) else self.state
                self.level_index = min(self.levels_completed, len(self.levels) - 1)
                return
            super().perform(action_id, x, y)

    non_click_modules = _click_modules(env_cls=NonClickEnv)
    non_click_modules = M.GameModules(
        source=non_click_modules.source, env=non_click_modules.env,
        generate=non_click_modules.generate,
        plan=SimpleNamespace(search=lambda env, limit: SimpleNamespace(
            actions=[(1, None, None)], truncated=False, unsupported=False,
            exact=True, expanded=1, reason="key press",
        )),
        solver_adapter="work_limit",
    )
    collected = M.collect_generated_game(
        non_click_modules, master_seed=0, game_index=0, difficulties=(1,),
    )
    assert collected.record["status"] == "won"
    assert collected.teacher["target_action_id"].tolist() == [1]
    assert collected.teacher["click_region_size"].tolist() == [0]
    assert collected.record["click_region_stats"]["click_targets"] == 0


def test_public_region_follows_the_whole_game_spatial_variant(tmp_path):
    modules = _click_modules()
    variants = M.VariantOptions(enabled=True, controls=False, spatial=True, palette=False, seed=3)
    found_non_identity = False
    for game_index in range(6):
        collected = M.collect_generated_game(
            modules, master_seed=0, game_index=game_index, difficulties=(1,), variants=variants,
        )
        assert collected.record["status"] == "won"
        spatial = collected.record["variant"]["spatial"]
        found_non_identity |= spatial != "identity"
        mask = collected.teacher["click_region_mask"][0].astype(bool)
        tx, ty = int(collected.teacher["target_action_x"][0]), int(collected.teacher["target_action_y"][0])
        assert mask[ty, tx] and mask.sum() == CELL * CELL
        # Undo the stored spatial map: the raw region must be the raw target cell.
        raw_x, raw_y = _cell_centre(4, 5)
        assert transform_xy(spatial, raw_x, raw_y) == (tx, ty)
        raw_cell = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
        raw_cell[(raw_y // CELL) * CELL:(raw_y // CELL + 1) * CELL, (raw_x // CELL) * CELL:(raw_x // CELL + 1) * CELL] = True
        assert np.array_equal(transform_frame(raw_cell, spatial), mask)
        # And the persisted files load with the region intact.
        record = M.save_collected_game(tmp_path / f"g{game_index}", "g", collected)
        loaded = load_supervised_game(
            tmp_path / f"g{game_index}" / record["public_npz"],
            tmp_path / f"g{game_index}" / record["teacher_npz"],
        )
        assert loaded.has_click_regions and np.array_equal(loaded.target_click_region[0], mask)
    assert found_non_identity


# --------------------------------------------------------------------------
# Loss and loader.
# --------------------------------------------------------------------------


def _one_hot(rows: int, xs, ys) -> torch.Tensor:
    region = torch.zeros((rows, FRAME_SIZE, FRAME_SIZE), dtype=torch.bool)
    for row, (x, y) in enumerate(zip(xs, ys)):
        region[row, y, x] = True
    return region


def test_region_nll_equals_exact_cross_entropy_for_single_pixel_regions():
    torch.manual_seed(0)
    logits = torch.randn(5, FRAME_SIZE, FRAME_SIZE)
    xs, ys = [3, 60, 0, 17, 63], [9, 1, 0, 40, 63]
    region = _one_hot(5, xs, ys)
    exact = F.cross_entropy(
        logits.flatten(1), torch.tensor(ys) * FRAME_SIZE + torch.tensor(xs), reduction="none",
    )
    assert torch.allclose(click_region_nll(logits, region), exact, atol=1e-5)
    with pytest.raises(ValueError, match="at least one pixel"):
        click_region_nll(logits, torch.zeros_like(region))


def test_region_nll_is_lower_when_the_argmax_lands_elsewhere_inside_the_region():
    logits = torch.zeros(1, FRAME_SIZE, FRAME_SIZE)
    logits[0, 10, 10] = 20.0  # confident click on a pixel that is NOT the exact target
    target_x, target_y = 12, 10
    exact = click_region_nll(logits, _one_hot(1, [target_x], [target_y]))
    region = torch.zeros(1, FRAME_SIZE, FRAME_SIZE, dtype=torch.bool)
    region[0, 8:13, 8:13] = True  # a 5x5 equivalent set containing both pixels
    inside = click_region_nll(logits, region)
    assert inside.item() < exact.item()
    assert inside.item() < 0.1  # nearly all mass is on the region
    assert exact.item() > 19.0  # the exact-pixel label would punish an equivalent click
    # Mass on the region is a superset of mass on the exact pixel, so never worse.
    torch.manual_seed(1)
    random_logits = torch.randn(8, FRAME_SIZE, FRAME_SIZE)
    superset = torch.zeros_like(region).expand(8, -1, -1).clone()
    superset[:, 8:13, 8:13] = True
    assert torch.all(
        click_region_nll(random_logits, superset)
        <= click_region_nll(random_logits, _one_hot(8, [target_x] * 8, [target_y] * 8)) + 1e-6
    )


def _write_game(
    root: Path,
    name: str,
    *,
    actions=(1, 6, 6),
    click_xy=(30, 20),
    region_radius: int | None = None,
    corrupt: str | None = None,
) -> tuple[Path, Path]:
    steps = len(actions)
    frames = np.zeros((steps + 1, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
    for index in range(steps + 1):
        frames[index, index, index] = index + 1
    legal = np.zeros((steps + 1, ACTION_COUNT), dtype=np.bool_)
    legal[:-1, 1:] = True
    x = np.full(steps, -1, dtype=np.int16)
    y = np.full(steps, -1, dtype=np.int16)
    actions = np.asarray(actions, dtype=np.int8)
    x[actions == CLICK_ACTION], y[actions == CLICK_ACTION] = click_xy
    boundary = np.zeros(steps, dtype=np.bool_)
    boundary[-1] = True
    terminal = np.zeros(steps + 1, dtype=np.bool_)
    terminal[-1] = True
    public = dict(
        frames=frames, legal_action_mask=legal,
        state=np.asarray(["NOT_FINISHED"] * steps + ["WIN"], dtype="<U16"),
        level_index=np.zeros(steps + 1, dtype=np.int16),
        levels_completed=np.concatenate((np.zeros(steps, dtype=np.int16), [1])).astype(np.int16),
        terminal=terminal, won=terminal.copy(), action_id=actions, action_x=x, action_y=y,
        level_boundary=boundary,
    )
    teacher = dict(
        target_action_id=actions.copy(), target_action_x=x.copy(), target_action_y=y.copy(),
        source=np.ones(steps, dtype=np.int8),
    )
    if region_radius is not None:
        mask = np.zeros((steps, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
        cx, cy = click_xy
        for row in np.flatnonzero(actions == CLICK_ACTION):
            mask[row, cy - region_radius:cy + region_radius + 1, cx - region_radius:cx + region_radius + 1] = 1
        if corrupt == "missing_target":
            mask[actions == CLICK_ACTION, cy, cx] = 0
        elif corrupt == "non_click_row":
            mask[0, 1, 1] = 1
        teacher["click_region_mask"] = mask
        teacher["click_region_size"] = mask.sum(axis=(1, 2)).astype(np.int16)
        if corrupt == "size":
            teacher["click_region_size"] = teacher["click_region_size"] + 1
    public_path, teacher_path = root / f"{name}.public.npz", root / f"{name}.teacher.npz"
    np.savez(public_path, **public)
    np.savez(teacher_path, **teacher)
    return public_path, teacher_path


def test_old_teacher_files_without_regions_load_and_collate_to_the_exact_pixel(tmp_path):
    old = load_supervised_game(*_write_game(tmp_path, "old"))
    assert old.supervised and not old.has_click_regions and old.target_click_region is None
    fallback = click_region_or_exact(old)
    assert fallback.shape == (3, FRAME_SIZE, FRAME_SIZE)
    assert fallback.sum(axis=(1, 2)).tolist() == [0, 1, 1]
    assert fallback[1, 20, 30] and fallback[2, 20, 30]
    batch = collate_game_sequences((old,))
    assert batch["target_click_region"].dtype == torch.bool
    assert torch.equal(batch["target_click_region"][0], torch.from_numpy(fallback))

    torch.manual_seed(2)
    model = MultiGameModel(MultiGameModelConfig.cpu_test()).train()
    loss = compute_multigame_loss(model, batch, transition_indices=torch.tensor([[0, 0]]))
    # Identical to the plain exact-pixel cross-entropy the old loss computed.
    policy = model.policy_from_history(model.encode_history(batch), batch["legal_action_mask"])
    destination = torch.tensor([20 * FRAME_SIZE + 30] * 2)
    exact = F.cross_entropy(policy.click_logits[0, 1:].flatten(1), destination)
    assert torch.allclose(loss.click, exact, atol=1e-5)
    assert loss.click_targets == 2
    loss.total.backward()
    assert torch.isfinite(loss.total)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_region_labels_load_validate_and_lower_the_click_loss(tmp_path):
    new = load_supervised_game(*_write_game(tmp_path, "new", region_radius=2))
    assert new.has_click_regions
    assert new.target_click_region.dtype == np.bool_
    assert new.target_click_region.sum(axis=(1, 2)).tolist() == [0, 25, 25]
    old = load_supervised_game(*_write_game(tmp_path, "old"))

    torch.manual_seed(3)
    model = MultiGameModel(MultiGameModelConfig.cpu_test()).eval()
    with_region = compute_multigame_loss(
        model, collate_game_sequences((new,)), transition_indices=torch.tensor([[0, 0]]),
    )
    exact_only = compute_multigame_loss(
        model, collate_game_sequences((old,)), transition_indices=torch.tensor([[0, 0]]),
    )
    assert with_region.click.item() < exact_only.click.item()
    assert with_region.click_region_correct >= with_region.click_exact_correct
    assert exact_only.click_region_correct == exact_only.click_exact_correct
    # Mixed batches (one old, one new game) collate: the old game gets its exact pixel.
    mixed = collate_game_sequences((new, old))
    assert mixed["target_click_region"][0].flatten(1).sum(1).tolist() == [0, 25, 25]
    assert mixed["target_click_region"][1].flatten(1).sum(1).tolist() == [0, 1, 1]

    for corrupt, message in (
        ("missing_target", "exact teacher target"),
        ("non_click_row", "non-click rows"),
        ("size", "click_region_size disagrees"),
    ):
        with pytest.raises(ValueError, match=message):
            load_supervised_game(*_write_game(tmp_path, corrupt, region_radius=1, corrupt=corrupt))


def test_region_moves_with_load_time_variants_and_survives_chunking(tmp_path):
    game = load_supervised_game(*_write_game(tmp_path, "new", region_radius=1))
    variant = replace(WholeGameVariant.identity(), spatial="rot90")
    moved = T.apply_variant_to_game(game, variant, T.game_legal_action_ids(game))
    assert moved.target_click_region.sum(axis=(1, 2)).tolist() == [0, 9, 9]
    for row in (1, 2):
        tx, ty = int(moved.target_action_x[row]), int(moved.target_action_y[row])
        assert (tx, ty) == transform_xy("rot90", 30, 20)
        assert moved.target_click_region[row, ty, tx]
        assert np.array_equal(
            moved.target_click_region[row], transform_frame(game.target_click_region[row], "rot90"),
        )
    back = T.canonical_sequence(moved, variant)
    assert np.array_equal(back.target_click_region, game.target_click_region)
    chunk = T.slice_game_sequence(moved, 1, 3)
    assert chunk.target_click_region.shape == (2, FRAME_SIZE, FRAME_SIZE)
    assert chunk.target_click_region.sum() == 18


def test_policy_metrics_report_region_hits_next_to_exact_hits():
    accumulator = T._empty_policy_accumulator()
    region = torch.zeros((1, 2, FRAME_SIZE, FRAME_SIZE), dtype=torch.bool)
    region[0, 0, 10:13, 10:13] = True
    region[0, 1, 5, 5] = True
    T._accumulate_policy_metrics(
        accumulator,
        action_guess=torch.tensor([[CLICK_ACTION, CLICK_ACTION]]),
        click_x=torch.tensor([[12, 6]]), click_y=torch.tensor([[12, 5]]),
        target_id=torch.tensor([[CLICK_ACTION, CLICK_ACTION]]),
        target_x=torch.tensor([[11, 5]]), target_y=torch.tensor([[11, 5]]),
        previous_id=torch.tensor([[-1, CLICK_ACTION]]),
        selected=torch.tensor([[True, True]]),
        click_region=region,
    )
    summary = T._policy_summary(accumulator)
    assert summary["clicks"] == 2
    assert summary["click_accuracy"] == 0.0  # neither argmax is the exact pixel
    assert summary["click_region_accuracy"] == 0.5  # the first lands inside its region
    assert summary["click_region_mean_size"] == 5.0
    assert summary["click_region_labelled"] == 1
    line = T.format_epoch_line({
        "epoch": 0, "train": {"total": 1.0},
        "generated_validation_offline": {"click_accuracy": 0.0, "click_region_accuracy": 0.5},
    })
    assert "click acc exact 0.000 region 0.500" in line


# --------------------------------------------------------------------------
# Training smoke on a synthetic manifest with region labels.
# --------------------------------------------------------------------------


def _write_manifest(root: Path, name: str, *, master_seed: int, puzzle: str) -> Path:
    folder = root / name
    (folder / "games").mkdir(parents=True)
    (folder / "teacher").mkdir()
    source = M.source_for("cd82")
    public_path, teacher_path = _write_game(folder, "game", region_radius=2)
    public_path.rename(folder / "games" / "game.npz")
    teacher_path.rename(folder / "teacher" / "game.npz")
    with np.load(folder / "games" / "game.npz") as public:
        frames = public["frames"].copy()
    frames[:, 3:6, 5:8] = (sum(puzzle.encode()) % 14) + 1
    with np.load(folder / "games" / "game.npz") as public:
        arrays = {key: public[key] for key in public.files}
    arrays["frames"] = frames
    np.savez(folder / "games" / "game.npz", **arrays)
    specs = [{"generator_version": 1, "effective_seed": master_seed * 7, "layout": {"puzzle_token": puzzle}}]
    (folder / "teacher" / "game.levels.json").write_text(json.dumps(specs))
    record = {
        "format": M.FORMAT, "source": source.slug, "source_id": source.source_id,
        "master_seed": master_seed, "game_index": 0, "status": "won", "steps": 3,
        "public_npz": "games/game.npz", "teacher_npz": "teacher/game.npz",
        "generated_specs": "teacher/game.levels.json", "record": "records/game.json",
        "levels": [{"generator_version": 1}],
    }
    manifest = {
        "format": M.MANIFEST_FORMAT, "requested_source_ids": [source.source_id],
        "held_out_source_id": M.HELD_OUT_SOURCE_ID, "is_full_experiment_collection": False,
        "records": [record],
    }
    M.save_manifest(folder / "manifest.json", manifest)
    return folder / "manifest.json"


def test_two_update_training_smoke_on_region_labelled_manifest(tmp_path, monkeypatch):
    train = _write_manifest(tmp_path, "train", master_seed=11, puzzle="train")
    validation = _write_manifest(tmp_path, "validation", master_seed=22, puzzle="validation")
    bundle = T.audit_manifest_pair(train, validation, smoke=True)
    assert bundle.train.games[0].sequence.has_click_regions
    monkeypatch.setattr(T, "evaluate_generated_closed_loop", lambda *args, **kwargs: {
        "games_won": 0, "levels_completed": 0, "games": [{"failure": None}],
        "panel_records": ["fake"], "actions": 1,
    })
    result = T.train_multigame(
        bundle, tmp_path / "run",
        T.TrainingConfig(
            epochs=1, model=MultiGameModelConfig.cpu_test(),
            loss=LossWeights(next_frame=0.0, events=0.0),
            chunk_steps=2, auxiliary_transitions_per_chunk=1, metric_transitions_per_game=1,
            closed_loop_games=1, closed_loop_train_games=0,
            validation_max_actions_per_level=1, validation_max_game_actions=1, device="cpu",
        ),
    )
    log = result.logs[0]
    assert log["global_step"] == 2
    assert log["train"]["click_targets"] == 2
    assert log["train"]["click_region_accuracy"] is not None
    assert log["train"]["click_exact_accuracy"] <= log["train"]["click_region_accuracy"]
    offline = log["generated_validation_offline"]
    assert offline["click_targets"] == 2
    assert offline["click_region_mean_size"] == 25.0
    assert offline["click_region_labelled"] == 2
    assert offline["click_accuracy"] <= offline["click_region_accuracy"]
    assert offline["policy"]["teacher"]["click_region_accuracy"] == offline["click_region_accuracy"]
    assert np.isfinite(offline["policy_click_loss_per_target"])
    assert "region" in T.format_epoch_line(log)
