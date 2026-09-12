#!/usr/bin/env python3
"""Audit the cell-appearance decoder on generated gameplay states.

The model sees only freshly rendered public frames.  Engine state is used for
diagnostic targets and masks, never as an input.  This audit replays a fixed
100-level generated validation subset, scores route states and bounded
four-action branches, and separates active public goals from solved or covered
underlays whose attributes are not observable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from pebby.agent.cell_appearance import CellAppearance, FORMAT
from pebby.ls20 import generate, names
from pebby.ls20.env import Ls20Scenario


ROLE_BITS = {
    "wall": 1,
    "goal": 2,
    "cycler_shape": 4,
    "cycler_color": 8,
    "cycler_rotation": 16,
    "launcher": 32,
    "refill": 64,
    "player": 128,
}
ROLE_NAMES = tuple(ROLE_BITS)
ATTRIBUTE_SIZES = (6, 4, 4)
GRID = names.GRID_COLS * names.GRID_ROWS
ACTION_IDS = tuple(names.ACTION_IDS)


def is_terminal(env: Ls20Scenario) -> bool:
    return env.state.name in ("WIN", "GAME_OVER")


def is_won(env: Ls20Scenario) -> bool:
    return env.state.name == "WIN"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_specs(bank: Path, selected_seeds: set[int]) -> dict[int, dict]:
    found = {}
    with bank.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            spec = json.loads(line)
            seed = int(spec["seed"])
            if seed in selected_seeds:
                if seed in found:
                    raise ValueError(f"duplicate validation seed {seed}")
                found[seed] = spec
    missing = sorted(selected_seeds - set(found))
    if missing:
        raise ValueError(f"validation bank missing selected seeds: {missing[:5]}")
    return found


def cell_index(col: int, row: int) -> int:
    return row * names.GRID_COLS + col


def support_status(env: Ls20Scenario, frame: np.ndarray, col: int, row: int, padding: int = 1) -> str | None:
    """Return None only when the current 7x7/9x9 support is public."""
    x, y = names.cell_to_pixel(col, row)
    left, top = x - padding, y - padding
    right, bottom = x + names.CELL + padding, y + names.CELL + padding
    if left < 0 or top < 0 or right > frame.shape[1] or bottom > frame.shape[0]:
        return "boundary"
    if bottom > 52:
        return "hud"
    if not env.fog():
        return None
    player = getattr(env.game, names.ATTR_PLAYER)
    for py in range(top, bottom):
        for px in range(left, right):
            if math.dist((py, px), (player.y + 1.5, player.x + 1.5)) > 20.0:
                return "fog"
    return None


def current_masks(env: Ls20Scenario, frame: np.ndarray):
    support7 = np.zeros((names.GRID_ROWS, names.GRID_COLS), dtype=bool)
    support9 = np.zeros_like(support7)
    reasons = np.empty(support7.shape, dtype=object)
    for row in range(names.GRID_ROWS):
        for col in range(names.GRID_COLS):
            reasons[row, col] = support_status(env, frame, col, row, 1)
            support7[row, col] = reasons[row, col] is None
            support9[row, col] = support_status(env, frame, col, row, 2) is None
    return support7, support9, reasons


def sprite_cells(env: Ls20Scenario, tag: str) -> set[tuple[int, int]]:
    return {
        tuple(map(int, names.pixel_to_cell(sprite.x, sprite.y)))
        for sprite in env.game.current_level.get_sprites_by_tag(tag)
        if getattr(sprite, "is_visible", True)
    }


def current_surface_labels(spec: dict, env: Ls20Scenario, support7: np.ndarray, animation: bool):
    """Build labels from currently active sprites, excluding hidden goal underlay."""
    roles = np.zeros((names.GRID_ROWS, names.GRID_COLS), dtype=np.uint16)
    attrs = np.full((names.GRID_ROWS, names.GRID_COLS, 3), -1, dtype=np.int8)
    goal_cells = np.zeros((names.GRID_ROWS, names.GRID_COLS), dtype=bool)
    role_exclude = np.zeros((names.GRID_ROWS, names.GRID_COLS), dtype=bool)
    solved = list(map(bool, env.goals_solved()))
    player = tuple(map(int, env.player_cell()))

    def add(cell, bit):
        col, row = cell
        if 0 <= col < names.GRID_COLS and 0 <= row < names.GRID_ROWS:
            roles[row, col] |= bit

    # These are public sprite surfaces at the current time.  Rails remain an
    # engine-only invisible support and are intentionally not labelled.
    for cell in sprite_cells(env, names.TAG_WALL):
        add(cell, ROLE_BITS["wall"])
    for tag, kind in names.CYCLER_TAGS.items():
        for cell in sprite_cells(env, tag):
            add(cell, ROLE_BITS[f"cycler_{kind}"])
    for cell in sprite_cells(env, names.TAG_STEP_REFILL):
        add(cell, ROLE_BITS["refill"])
    # Launcher artwork is offset by one pixel, so use the verified semantic
    # cells from the generated spec while requiring their current public mask.
    for entry in spec.get("launchers", []):
        add(tuple(map(int, entry["cell"])), ROLE_BITS["launcher"])
    add(player, ROLE_BITS["player"])
    raw_overlap = int(roles[player[1], player[0]]) & ~ROLE_BITS["player"]

    active_pad_objects = [pad for pad in env.game.current_level.get_sprites_by_tag(names.TAG_GOAL_PAD)
                          if getattr(pad, "is_visible", True)]
    active_pad_ids = {id(pad) for pad in active_pad_objects}
    goal_status = []
    for index, goal in enumerate(spec["goals"]):
        spec_cell = tuple(map(int, goal["cell"]))
        pad = env.game.plrpelhym[index]
        active = not solved[index] and id(pad) in active_pad_ids
        cell = tuple(map(int, names.pixel_to_cell(pad.x, pad.y))) if active else spec_cell
        col, row = cell
        if solved[index] or not active:
            status = "solved_or_removed"
        elif cell == player:
            status = "covered_by_player"
            role_exclude[row, col] = True
            raw_overlap = True
        elif animation:
            status = "transient_animation"
            role_exclude[row, col] = True
        elif not support7[row, col]:
            status = "hidden_" + ("fog" if env.fog() else "nonpublic")
        else:
            status = "active_public"
            add(cell, ROLE_BITS["goal"])
            attrs[row, col] = np.asarray(goal["triple"], dtype=np.int8)
            goal_cells[row, col] = True
        goal_status.append({"cell": list(cell), "spec_cell": list(spec_cell), "status": status,
                            "solved": solved[index], "active": active, "moving": active and cell != spec_cell})

    # A player sprite can cover part or all of an object.  Retain raw engine
    # underlay bits for diagnostics, but the scorer excludes this cell from
    # aggregate role accuracy and reports player/underlay predictions apart.
    return roles, attrs, goal_cells, goal_status, role_exclude.reshape(-1), bool(raw_overlap)


def capture_state(env: Ls20Scenario, spec: dict, collection: str, route_index: int,
                  action: int | None, transition: dict | None = None):
    frame = np.asarray(env.render(), dtype=np.uint8)
    if frame.shape != (64, 64) or frame.min() < 0 or frame.max() > 15:
        raise ValueError(f"invalid current frame shape/range for seed {spec['seed']}")
    support7, support9, reasons = current_masks(env, frame)
    game = env.game
    death_flash = bool(getattr(game, names.ATTR_DEATH_FLASH))
    animation = bool(getattr(game, names.ATTR_ACTIVE_ANIMATIONS)) or death_flash or bool(getattr(game, names.ATTR_REJECT_FLASH))
    roles, attrs, goal_cells, goal_status, role_exclude, player_overlap = current_surface_labels(spec, env, support7, animation)
    if death_flash:
        # The engine omits normal fog composition during the death flash and
        # paints an opaque overlay. No semantic role is scoreable in that frame.
        role_exclude = support7.reshape(-1).copy()
    player = tuple(map(int, env.player_cell()))
    overlap_cells = np.zeros(GRID, dtype=bool)
    if player_overlap:
        overlap_cells[cell_index(player[0], player[1])] = True
    return {
        "frame": frame,
        "seed": int(spec["seed"]),
        "difficulty": int(spec["difficulty"]),
        "collection": collection,
        "route_index": int(route_index),
        "action": None if action is None else int(action),
        "support7": support7.reshape(-1),
        "support9": support9.reshape(-1),
        "reasons": reasons.reshape(-1),
        "roles": roles.reshape(-1),
        "attrs": attrs.reshape(-1, 3),
        "goal_cells": goal_cells.reshape(-1),
        "overlap_cells": overlap_cells,
        "role_exclude": role_exclude,
        "goal_status": goal_status,
        "player_cell": player,
        "lives": int(env.lives()),
        "goals_solved": int(sum(env.goals_solved())),
        "fog": bool(env.fog()),
        "animation": animation,
        "death_flash": death_flash,
        "terminal": env.state.name in ("WIN", "GAME_OVER"),
        "won": env.state.name == "WIN",
        "player_overlap": player_overlap,
        "goal_covered": any(item["status"] == "covered_by_player" for item in goal_status),
        "goal_active_public": any(item["status"] == "active_public" for item in goal_status),
        "goal_solved_removed": any(item["status"] == "solved_or_removed" for item in goal_status),
        "moving_goal": any(item["moving"] for item in goal_status),
        "transition": transition or {},
    }


def transition(before, env: Ls20Scenario):
    solved_after = int(sum(env.goals_solved()))
    return {
        "life_lost": int(env.lives()) < int(before["lives"]),
        "goal_progress": solved_after > int(before["goals_solved"]),
        "won": is_won(env),
        "game_over": env.state.name == "GAME_OVER",
        "player_reset_to_spawn": int(env.lives()) < int(before["lives"]) and tuple(env.player_cell()) == tuple(before["spawn"]),
    }


def snapshot(env: Ls20Scenario, spec: dict):
    return {"lives": int(env.lives()), "goals_solved": int(sum(env.goals_solved())),
            "spawn": tuple(map(int, spec["start"]))}


def make_env(spec):
    context = spec.get("training_context_index")
    if isinstance(context, bool) or not isinstance(context, (int, np.integer)) or int(context) < 0:
        raise ValueError(f"missing/invalid training context for seed {spec.get('seed')}")
    env = Ls20Scenario(generate.build_level(spec), context_index=int(context))
    expected_goals = [tuple(map(int, goal["triple"])) for goal in spec.get("goals", [])]
    actual_goals = [tuple(map(int, triple)) for triple in env.goal_triples()]
    if actual_goals != expected_goals:
        raise ValueError(f"goal identity mismatch for seed {spec.get('seed')}: {actual_goals} != {expected_goals}")
    return env


def replay_level(spec, samples, branch_samples, transition_counts):
    route = [int(action) for action in spec.get("solution", [])]
    if not route:
        raise ValueError(f"seed {spec['seed']} has no verified route")
    env = make_env(spec)
    samples.append(capture_state(env, spec, "route_initial", 0, None))
    for index, action in enumerate(route):
        before = snapshot(env, spec)
        env.perform(action)
        trans = transition(before, env)
        transition_counts["route_" + ("life_loss" if trans["life_lost"] else "ordinary")] += 1
        transition_counts["route_goal_progress"] += int(trans["goal_progress"])
        transition_counts["route_win"] += int(trans["won"])
        transition_counts["route_game_over"] += int(trans["game_over"])
        sample = capture_state(env, spec, "route_post_action", index + 1, action, trans)
        samples.append(sample)
        if is_terminal(env):
            break
    if not is_won(env):
        transition_counts["route_failures"] += 1
        return False

    # Four actual action branches at four deterministic route anchors.  Prefix
    # replay uses the real engine, so branch states include animation, goal,
    # overlap, budget, and reset effects rather than synthetic edits.
    anchors = sorted({0, len(route) // 3, (2 * len(route)) // 3, max(0, len(route) - 1)})
    for anchor in anchors:
        for action in ACTION_IDS:
            branch = make_env(spec)
            for prefix_action in route[:anchor]:
                if is_terminal(branch):
                    break
                branch.perform(prefix_action)
            if is_terminal(branch):
                continue
            before = snapshot(branch, spec)
            branch.perform(action)
            trans = transition(before, branch)
            trans["branch_anchor"] = anchor
            transition_counts["branch_count"] += 1
            transition_counts["branch_action_" + str(action)] += 1
            transition_counts["branch_life_loss"] += int(trans["life_lost"])
            transition_counts["branch_goal_progress"] += int(trans["goal_progress"])
            transition_counts["branch_win"] += int(trans["won"])
            transition_counts["branch_game_over"] += int(trans["game_over"])
            branch_samples.append(capture_state(branch, spec, "branch_post_action", anchor, action, trans))
    return True


def random_reset_probe(specs: dict[int, dict], seeds: list[int]) -> dict:
    """Probe real life-loss/reset behavior without synthetic state edits.

    Each of ten fixed validation levels receives a deterministic random action
    stream, stopping at the first observed life loss, terminal state, or 500
    actions.  This is a coverage probe, not a policy-quality evaluation.
    """
    by_difficulty = defaultdict(list)
    for seed in sorted(seeds):
        by_difficulty[int(specs[seed]["difficulty"])].append(seed)
    probe_seeds = [seed for difficulty in sorted(by_difficulty) for seed in by_difficulty[difficulty][:2]]
    if len(probe_seeds) != 10:
        raise ValueError(f"expected ten reset-probe levels across difficulties, got {probe_seeds}")
    rng = np.random.default_rng(20260912)
    levels = []
    for seed in probe_seeds:
        spec = specs[seed]
        env = make_env(spec)
        initial_lives = int(env.lives())
        life_loss = False
        terminal_before_loss = False
        actions = 0
        reset = False
        lives_before = initial_lives
        lives_after = initial_lives
        player_before = tuple(map(int, env.player_cell()))
        player_after = player_before
        for _ in range(500):
            if is_terminal(env):
                terminal_before_loss = True
                break
            action = int(rng.choice(np.asarray(ACTION_IDS, dtype=np.int64)))
            lives_before = int(env.lives())
            player_before = tuple(map(int, env.player_cell()))
            env.perform(action)
            actions += 1
            lives_after = int(env.lives())
            player_after = tuple(map(int, env.player_cell()))
            if lives_after < lives_before:
                life_loss = True
                reset = not is_terminal(env) and player_after == tuple(map(int, spec["start"]))
                break
        levels.append({
            "seed": seed,
            "difficulty": int(spec["difficulty"]),
            "actions": actions,
            "life_loss": life_loss,
            "reset_to_start": reset,
            "terminal_before_loss": terminal_before_loss,
            "state_after": env.state.name,
            "initial_lives": initial_lives,
            "lives_before": lives_before,
            "lives_after": lives_after,
            "player_before": list(player_before),
            "player_after": list(player_after),
            "goals_solved_after": int(sum(env.goals_solved())),
        })
    return {
        "seed": 20260912,
        "max_actions_per_level": 500,
        "stop_condition": "first life loss, terminal state, or action cap",
        "levels": levels,
        "levels_with_life_loss": sum(int(level["life_loss"]) for level in levels),
        "levels_with_reset_to_start": sum(int(level["reset_to_start"]) for level in levels),
        "actions_total": sum(level["actions"] for level in levels),
    }


@torch.no_grad()
def infer(model, samples):
    predictions = []
    for start in range(0, len(samples), 256):
        batch = np.stack([sample["frame"] for sample in samples[start:start + 256]])
        role_logits, shape_logits, color_logits, rotation_logits = model(torch.from_numpy(batch))
        role_pred = role_logits.cpu().numpy() >= 0
        attr_pred = [shape_logits.cpu().numpy().argmax(-1), color_logits.cpu().numpy().argmax(-1),
                     rotation_logits.cpu().numpy().argmax(-1)]
        for row in range(len(batch)):
            predictions.append((role_pred[row], [scores[row] for scores in attr_pred]))
    return predictions


def summarize(samples, predictions, indices):
    cells = roles_correct = roles_bits = 0
    public_support7_cells = 0
    role_excluded_unknown_cells = 0
    tp = np.zeros(8, dtype=np.int64); fp = np.zeros(8, dtype=np.int64); fn = np.zeros(8, dtype=np.int64)
    goal_count = 0; goal_correct = np.zeros(3, dtype=np.int64); goal_joint = 0
    goal_role_correct = 0
    overlap_cell_count = overlap_player_correct = 0
    overlap_target_underlay = overlap_predicted_underlay = 0
    overlap_predicted_bits = np.zeros(7, dtype=np.int64)
    for index in indices:
        sample = samples[index]; role_pred, attrs_pred = predictions[index]
        public_mask = sample["support7"]
        public_support7_cells += int(public_mask.sum())
        role_exclude = sample["role_exclude"]
        role_excluded_unknown_cells += int((public_mask & role_exclude).sum())
        mask = public_mask & ~sample["overlap_cells"] & ~role_exclude
        target = sample["roles"][:, None].astype(np.uint16)
        target_bits = (target & (1 << np.arange(8, dtype=np.uint16))) != 0
        pred_bits = role_pred
        cells += int(mask.sum())
        roles_correct += int(np.all(pred_bits[mask] == target_bits[mask], axis=1).sum())
        roles_bits += int(mask.sum() * 8)
        tp += np.logical_and(pred_bits[mask], target_bits[mask]).sum(0)
        fp += np.logical_and(pred_bits[mask], ~target_bits[mask]).sum(0)
        fn += np.logical_and(~pred_bits[mask], target_bits[mask]).sum(0)
        overlap_mask = sample["overlap_cells"] & public_mask
        if np.any(overlap_mask):
            overlap_cell_count += int(overlap_mask.sum())
            overlap_player_correct += int((pred_bits[overlap_mask, 7] == target_bits[overlap_mask, 7]).sum())
            overlap_target_underlay += int(target_bits[overlap_mask, :7].any(1).sum())
            overlap_predicted_underlay += int(pred_bits[overlap_mask, :7].any(1).sum())
            overlap_predicted_bits += pred_bits[overlap_mask, :7].sum(0)
        goal_mask = sample["goal_cells"] & public_mask & ~role_exclude
        if np.any(goal_mask):
            attr_target = sample["attrs"][goal_mask]
            attr_ok = np.stack([attrs_pred[column][goal_mask] == attr_target[:, column] for column in range(3)], axis=1)
            goal_count += int(goal_mask.sum())
            goal_correct += attr_ok.sum(0)
            goal_joint += int(attr_ok.all(1).sum())
            goal_role_correct += int(pred_bits[goal_mask, 1].sum())
    role_f1 = (2 * tp / np.maximum(1, 2 * tp + fp + fn)).tolist()
    tn = (cells - tp - fp - fn).clip(min=0)
    return {
        "states": len(indices), "public_support7_cells": int(public_support7_cells),
        "role_scored_cells": int(cells), "role_excluded_unknown_cells": int(role_excluded_unknown_cells),
        "role_cell_exact_accuracy": roles_correct / max(1, cells),
        "role_bit_accuracy": float((sum(tp) + sum(tn)) / max(1, roles_bits)),
        "role_macro_f1": float(np.mean(role_f1)), "role_f1": dict(zip(ROLE_NAMES, role_f1)),
        "overlap_cells_excluded_from_role_score": int(overlap_cell_count),
        "overlap_player_bit_accuracy": overlap_player_correct / max(1, overlap_cell_count),
        "overlap_target_underlay_positive_cells": int(overlap_target_underlay),
        "overlap_predicted_underlay_positive_cells": int(overlap_predicted_underlay),
        "overlap_predicted_underlay_bits": dict(zip(ROLE_NAMES[:7], overlap_predicted_bits.tolist())),
        "goal_cells": int(goal_count),
        "goal_attribute_accuracy": (goal_correct / goal_count).tolist() if goal_count else None,
        "goal_joint_accuracy": float(goal_joint / goal_count) if goal_count else None,
        "goal_role_detected": float(goal_role_correct / goal_count) if goal_count else None,
        "support7_cells": int(cells),
    }


def export_public_frames(samples: list[dict], path: Path) -> str:
    """Persist the exact public inputs and masks used by this audit.

    Engine labels are deliberately absent.  The export is for cheap, frozen
    visibility follow-up only; role scoring masks are included separately so
    consumers cannot confuse public support with a scored target cell.
    """
    if path.exists():
        raise FileExistsError(path)
    frames = np.stack([sample["frame"] for sample in samples]).astype(np.uint8, copy=False)
    support7 = np.stack([sample["support7"] for sample in samples]).astype(bool, copy=False)
    support9 = np.stack([sample["support9"] for sample in samples]).astype(bool, copy=False)
    overlap = np.stack([sample["overlap_cells"] for sample in samples]).astype(bool, copy=False)
    role_exclude = np.stack([sample["role_exclude"] for sample in samples]).astype(bool, copy=False)
    role_scored = support7 & ~overlap & ~role_exclude
    kwargs = {
        "schema_version": np.asarray("pebby.cell-appearance-dynamic-public.v1"),
        "frames": frames,
        "support7": support7,
        "support9": support9,
        "role_scored": role_scored,
        "overlap_cells": overlap,
        "role_exclude": role_exclude,
        "seed": np.asarray([sample["seed"] for sample in samples], dtype=np.int64),
        "difficulty": np.asarray([sample["difficulty"] for sample in samples], dtype=np.int8),
        "collection": np.asarray([sample["collection"] for sample in samples]),
        "route_index": np.asarray([sample["route_index"] for sample in samples], dtype=np.int32),
        "action": np.asarray([-1 if sample["action"] is None else sample["action"] for sample in samples], dtype=np.int16),
        "fog": np.asarray([sample["fog"] for sample in samples], dtype=bool),
        "animation": np.asarray([sample["animation"] for sample in samples], dtype=bool),
        "death_flash": np.asarray([sample["death_flash"] for sample in samples], dtype=bool),
        "terminal": np.asarray([sample["terminal"] for sample in samples], dtype=bool),
        "won": np.asarray([sample["won"] for sample in samples], dtype=bool),
        "player_overlap": np.asarray([sample["player_overlap"] for sample in samples], dtype=bool),
        "goal_covered": np.asarray([sample["goal_covered"] for sample in samples], dtype=bool),
        "goal_solved_removed": np.asarray([sample["goal_solved_removed"] for sample in samples], dtype=bool),
        "goal_active_public": np.asarray([sample["goal_active_public"] for sample in samples], dtype=bool),
        "moving_goal": np.asarray([sample["moving_goal"] for sample in samples], dtype=bool),
        "lives": np.asarray([sample["lives"] for sample in samples], dtype=np.int8),
        "goals_solved": np.asarray([sample["goals_solved"] for sample in samples], dtype=np.int8),
        "life_lost": np.asarray([sample["transition"].get("life_lost", False) for sample in samples], dtype=bool),
        "player_reset_to_spawn": np.asarray([sample["transition"].get("player_reset_to_spawn", False) for sample in samples], dtype=bool),
        "goal_progress": np.asarray([sample["transition"].get("goal_progress", False) for sample in samples], dtype=bool),
        "game_over": np.asarray([sample["transition"].get("game_over", False) for sample in samples], dtype=bool),
        "branch_anchor": np.asarray([sample["transition"].get("branch_anchor", -1) for sample in samples], dtype=np.int32),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with tmp.open("wb") as stream:
            np.savez_compressed(stream, **kwargs)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return digest(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/cell-appearance-2k-400.pt"))
    parser.add_argument("--validation-data", type=Path, default=Path("data/ls20-visible-cell-labels.npz"))
    parser.add_argument("--validation-bank", type=Path, default=Path("data/ls20-verified-validation.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/cell-appearance-dynamic-audit.json"))
    parser.add_argument("--public-export", type=Path,
                        default=Path("data/cell-appearance-dynamic-public-frames.npz"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    print("PID", os.getpid(), flush=True)

    with np.load(args.validation_data, allow_pickle=False) as archive:
        seeds = archive["seeds"][archive["split"] == "validation"].astype(np.int64).tolist()
    if len(seeds) != 100 or len(set(seeds)) != 100:
        raise ValueError(f"expected fixed100 validation seeds, got {len(seeds)}")
    specs = load_specs(args.validation_bank, set(map(int, seeds)))
    paths = [args.checkpoint, args.validation_data, args.validation_bank,
             Path(__file__), Path("pebby/agent/cell_appearance.py"), Path("pebby/ls20/env.py"),
             Path("pebby/ls20/generate.py"), Path("pebby/ls20/names.py"), Path("third_party/ls20/ls20.py")]
    hashes_before = {str(path): digest(path) for path in paths}
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("unsupported appearance checkpoint")
    architecture_hash = digest(Path("pebby/agent/cell_appearance.py"))
    if checkpoint.get("source_hashes", {}).get("pebby/agent/cell_appearance.py") != architecture_hash:
        raise ValueError("checkpoint architecture provenance mismatch")
    model = CellAppearance(); model.load_state_dict(checkpoint["weights"]); model.eval()

    route_samples, branch_samples = [], []
    transitions = Counter()
    levels = []
    for seed in sorted(map(int, seeds)):
        if seed < 1_000_000 or seed >= 2_000_000:
            raise ValueError(f"selected seed outside verified validation range: {seed}")
        spec = specs[seed]
        if spec.get("search_truncated", False) or not spec.get("context_engine_verified", False):
            raise ValueError(f"selected route lacks complete context proof: seed {seed}")
        before_count = len(route_samples); before_branches = len(branch_samples)
        if not replay_level(spec, route_samples, branch_samples, transitions):
            raise RuntimeError(f"verified route failed for seed {seed}")
        levels.append({"seed": seed, "difficulty": int(spec["difficulty"]),
                       "route_actions": len(spec.get("solution", [])),
                       "route_samples": len(route_samples) - before_count,
                       "branch_samples": len(branch_samples) - before_branches,
                       "engine_verified": bool(spec.get("context_engine_verified"))})

    samples = route_samples + branch_samples
    predictions = infer(model, samples)
    groups = {
        "all_states": list(range(len(samples))),
        "route_states": [i for i, s in enumerate(samples) if s["collection"].startswith("route")],
        "branch_states": [i for i, s in enumerate(samples) if s["collection"].startswith("branch")],
        "route_initial": [i for i, s in enumerate(samples) if s["collection"] == "route_initial"],
        "route_post_action": [i for i, s in enumerate(samples) if s["collection"] == "route_post_action"],
        "branch_post_action": [i for i, s in enumerate(samples) if s["collection"] == "branch_post_action"],
    }
    for flag, label in (("fog", "fog_states"), ("animation", "animation_states"),
                        ("player_overlap", "player_overlap_states"), ("goal_covered", "covered_goal_states"),
                        ("goal_solved_removed", "solved_or_removed_goal_states"),
                        ("goal_active_public", "active_public_goal_states"), ("terminal", "terminal_states")):
        groups[label] = [i for i, s in enumerate(samples) if s[flag]]
    metrics = {name: summarize(samples, predictions, indices) for name, indices in groups.items()}

    surface_status = Counter()
    for sample in samples:
        surface_status.update(item["status"] for item in sample["goal_status"])
    overlap_states = sum(bool(s["player_overlap"]) for s in samples)
    moving_goal_states = sum(bool(sample["moving_goal"]) for sample in samples)
    reset_probe = random_reset_probe(specs, sorted(map(int, seeds)))
    if any(digest(Path(path)) != value for path, value in hashes_before.items()):
        raise ValueError("audit source changed during run")
    public_export_sha256 = export_public_frames(samples, args.public_export)
    report = {
        "status": "complete", "format": "pebby.cell-appearance-dynamic-audit.v1",
        "training_performed": False, "device": "cpu", "cpu_threads": 1,
        "checkpoint_parameters": model.parameter_count(), "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashes_before[str(args.checkpoint)],
        "source_hashes": hashes_before,
        "fixed_validation": {"levels": 100, "seed_count": len(seeds), "seed_sha256": hashlib.sha256("\n".join(map(str, sorted(map(int, seeds)))).encode()).hexdigest(),
                              "selection_source": str(args.validation_data), "engine_bank": str(args.validation_bank)},
        "levels": levels,
        "sample_counts": {"route_states": len(route_samples), "branch_states": len(branch_samples), "total_states": len(samples)},
        "public_export": {
            "path": str(args.public_export),
            "sha256": public_export_sha256,
            "rows": len(samples),
            "schema": "pebby.cell-appearance-dynamic-public.v1",
            "contains_engine_labels": False,
        },
        "metrics_by_state_category": metrics,
        "goal_surface_status_counts": dict(sorted(surface_status.items())),
        "transition_counts": dict(sorted(transitions.items())),
        "dynamic_coverage": {"player_object_overlap_states": overlap_states,
                              "moving_goal_states": moving_goal_states,
                              "solved_goal_states": sum(1 for s in samples if s["goal_solved_removed"]),
                              "life_reset_transitions": transitions["route_life_loss"] + transitions["branch_life_loss"],
                              "terminal_states": sum(1 for s in samples if s["terminal"]),
                              "animation_states": sum(1 for s in samples if s["animation"]),
                              "death_flash_states": sum(1 for s in samples if s["death_flash"]),
                              "covered_goal_states": sum(1 for s in samples if s["goal_covered"]),
                              "random_probe_levels_with_life_loss": reset_probe["levels_with_life_loss"],
                              "random_probe_levels_with_reset_to_start": reset_probe["levels_with_reset_to_start"]},
        "random_reset_probe": reset_probe,
        "label_contract": {
            "role_mask": "current-player 7x7 support must be fully inside current fog aperture and outside HUD",
            "goal_attributes": "scored only for active unsolved goals with current public 7x7 support and no player coverage or transient animation",
            "solved_or_covered": "goal attributes are excluded and counted by status; player-overlap cells are excluded from aggregate role scores while player and raw underlay predictions are reported separately",
            "engine_labels_diagnostic_only": True,
        },
        "limitations": [
            "Generated validation only; no official layouts, frames, routes, or labels.",
            "The bank has static goals; moving_goal_states is therefore zero and does not test moving goals.",
            "Four actions are sampled at four route anchors per level; this is not exhaustive branch rollout.",
            "Life/reset transitions may be unavailable in one-action branches; counts are reported rather than synthesized.",
            "Engine-derived labels and current-player/fog masks are diagnostic only and are never model inputs.",
            "A public 7x7 support proves pixel visibility, not that every semantic role is identifiable from appearance.",
            "State categories and branch counts are frequency-weighted replay samples, not unique game-state populations.",
            "Role scores exclude overlap, covered/transient goal cells, and opaque death-flash cells; those exclusions are counted explicitly.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"route_states": len(route_samples), "branch_states": len(branch_samples), "metrics": metrics["all_states"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
