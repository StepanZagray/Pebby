"""Generated-only, counterfactual transition data for the looped world model.

Every row has causal observation/action history and four actual engine branches.
The oracle supplies supervision only after a COMPLETE search. It is never used
by the policy at inference. No official level is constructed by this module.

Three coverage modes share one row contract:

* ``prefix`` (default): an epsilon-greedy episode from the initial state,
  recording every visited state until the sample budget runs out. Long levels
  therefore never show the model their endings.
* ``mixed``: about half the budget is the same epsilon-greedy prefix; the rest
  is spread over the complete verified expert trajectory and always includes
  the expert's last pre-WIN state, so every accepted level carries a real
  winning successor. Unselected expert steps cost one engine action; only
  selected states are branched four ways.
* ``mixed_failure`` (CLI default): mixed coverage with at least one expert
  anchor per four route actions, plus up to three actual pre-death states from
  a bounded exhaustion episode. This mode can exceed the base sample budget.
  Unreachable current states have optimal=0 and train dynamics/value only.
"""

import argparse
import copy
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import random

import numpy as np

from ..ls20 import names
from ..ls20.env import Ls20Scenario
from ..ls20.generate import build_level
from ..ls20.layout import extract
from ..ls20.plan import Oracle, advance, simulate

FORMAT = "pebby.ls20-world-transitions.v1"
STATE_SUPERVISION_VERSION = 1
SUCCESSOR_POLICY_SUPERVISION_VERSION = 1
COVERAGE_MODES = ("prefix", "mixed", "mixed_failure")


def history_arrays(frames, actions, length):
    """Action at each slot produced that frame; -1 means padding/initial frame."""
    count = min(len(frames), length)
    padding = length - count
    recent_frames = frames[-count:]
    recent_actions = actions[-count:]
    return (np.asarray([recent_frames[0]] * padding + recent_frames, dtype=np.uint8),
            np.asarray([False] * padding + [True] * count, dtype=bool),
            np.asarray([-1] * padding + recent_actions, dtype=np.int64))


def history_key(observed, valid, previous):
    """Exact identity of public input; hidden states can still differ under fog."""
    return observed.tobytes() + valid.tobytes() + previous.tobytes()


def state_history_key(env, oracle, observed, valid, previous):
    """Deduplicate only equal histories AND predictive states, never just pixels."""
    return history_key(observed, valid, previous), oracle.state_of(env), env.lives()


def spread_indices(candidates, count):
    """Evenly spaced picks from an ascending list, always keeping its last entry."""
    if count <= 0 or not candidates:
        return []
    if count >= len(candidates):
        return list(candidates)
    if count == 1:
        return [candidates[-1]]
    last = len(candidates) - 1
    # Spacing exceeds one slot whenever count < len(candidates), so the rounded
    # positions are distinct and the final one lands exactly on `last`.
    picks = sorted({round(last * j / (count - 1)) for j in range(count)})
    return [candidates[i] for i in picks]


def clone_env(env):
    """Copy live state while sharing generated scenarios' untouched templates.

    The engine never mutates _clean_levels: full_reset, level_reset and LS20's
    on_set_level clone them before use. Ls20Scenario's earlier _levels entries
    are skipped context copies; on_set_level replaces one before playing it.
    Keep each list itself private and deepcopy the active level plus every
    runtime object (HUD, camera, animations, patrollers and sprite references).
    Sharing these templates avoids copying up to thirteen unused sprite sets
    for each of the four actions at every sampled state.
    """
    memo = {id(env.module): env.module}
    if isinstance(env, Ls20Scenario):
        for level in env.game._clean_levels:
            memo[id(level)] = level
        for index, level in enumerate(env.game._levels):
            if index != env.game.level_index:
                memo[id(level)] = level
    return copy.deepcopy(env, memo)


def verified_context(spec, context_index=None, search_limit=600_000):
    """Complete contextual oracle AND a winning real-engine replay, or refusal."""
    if spec.get("search_truncated"):
        return None, None, {"seed": spec["seed"], "excluded": "truncated source proof"}
    context_index = spec["seed"] % 7 if context_index is None else context_index
    # A context-zero launcher landing on a matching cycler can leave a hint
    # pending beyond perform_action. The next submitted action only clears that
    # animation. The logical oracle has no pending-hint state, so its distance
    # would undercount actions even when its initial solution happens to win.
    # Conservatively exclude all context-zero launcher layouts from supervision.
    if context_index == 0 and spec.get("launchers"):
        return None, None, {"seed": spec["seed"], "excluded":
                           "context-zero launcher can leave pending hint outside oracle state"}
    env = Ls20Scenario(build_level(spec), context_index)
    oracle = Oracle(extract(env), limit=search_limit)
    if oracle.truncated or not oracle.solvable:
        return None, None, {"seed": spec["seed"], "excluded": (
            "incomplete contextual oracle (search limit)" if oracle.truncated
            else "unsolved contextual oracle"), "search_truncated": bool(oracle.truncated),
            "reachable_states": oracle._reachable, "search_limit": search_limit}
    solution = oracle.solution(seed=spec["seed"])
    replay = clone_env(env)
    result = None
    for action in solution or ():
        result = replay.perform(action)
    if result is None or not result.won or replay.lives() != 3 or replay.levels_completed != 1:
        return None, None, {"seed": spec["seed"], "excluded": "contextual solution failed real-engine replay"}
    return env, oracle, {"seed": spec["seed"], "context_index": context_index,
                        "context_engine_verified": True, "search_truncated": False,
                        "context_optimal_actions": len(solution),
                        "oracle_backend": getattr(oracle, "engine", "reference")}


def successor_optimal_mask(oracle, state, *, terminal=False):
    """Exact action set at an actual successor; zero means no supervised action.

    Reuse the completed graph and its cheap logical transitions. This does not
    render another sixteen engine branches. Finished and unreachable states
    have no target. A life loss that resets to a live, solvable state does.
    """
    if oracle.truncated:
        raise ValueError("successor policy labels require a complete oracle")
    if terminal:
        return 0
    distance = oracle.distance_for(state)
    if distance is None or distance == 0:
        return 0
    mask = 0
    for action in range(4):
        following = advance(oracle.layout, state, action, oracle.refills)
        if following is not None and oracle.distance_for(following) == distance - 1:
            mask |= 1 << action
    if not mask:
        raise ValueError("reachable successor has no distance-decreasing action")
    return mask


def _expand(env, oracle, before, seed, step):
    """Every real engine action from one live state.

    Returns the per-action targets, the four post-action branches, their
    observations and the optimal-action mask. `before` is the complete
    teacher's distance at `env`; the mask marks actions that shorten it by
    exactly one without costing a life.
    """
    next_frames, distances, terminal, won, lost_life, next_optimal = [], [], [], [], [], []
    next_player_cell, next_triple, next_steps, next_lives = [], [], [], []
    branches, results = [], []
    for action in names.ACTION_IDS:
        branch = clone_env(env)
        result = branch.perform(action)
        branches.append(branch)
        results.append(result)
        if result.frame is None:
            raise RuntimeError("a live-state action returned no public successor observation")
        next_frames.append(result.frame)
        terminal.append(result.finished)
        won.append(result.won)
        lost_life.append(branch.lives() < env.lives())
        # Read from the same post-action branch that produced next_frames.
        # These are training targets only; inference still receives pixels
        # and the actions that produced them, never privileged state.
        next_player_cell.append(branch.player_cell())
        next_triple.append(branch.triple())
        next_steps.append(branch.steps_left())
        next_lives.append(branch.lives())
        successor_state = oracle.state_of(branch)
        after = 0 if result.won else oracle.distance_for(successor_state)
        distances.append(-1 if result.finished and not result.won or after is None else after)
        next_optimal.append(successor_optimal_mask(oracle, successor_state, terminal=result.finished))
    mask = (sum(1 << index for index, distance in enumerate(distances)
                if distance == before - 1 and not lost_life[index]) if before is not None else 0)
    if not mask and before is not None:
        raise ValueError(f"complete teacher has no optimal action at seed {seed}, step {step}")
    targets = {"next_frames": np.asarray(next_frames, dtype=np.uint8),
               "terminal": np.asarray(terminal, dtype=bool), "won": np.asarray(won, dtype=bool),
               "lost_life": np.asarray(lost_life, dtype=bool), "optimal": np.uint8(mask),
               "next_optimal": np.asarray(next_optimal, dtype=np.uint8),
               "distances": np.asarray(distances, dtype=np.int16),
               "player_cell": np.asarray(env.player_cell(), dtype=np.int16),
               "next_player_cell": np.asarray(next_player_cell, dtype=np.int16),
               "current_triple": np.asarray(env.triple(), dtype=np.int16),
               "next_triple": np.asarray(next_triple, dtype=np.int16),
               "current_steps": np.int16(env.steps_left()),
               "next_steps": np.asarray(next_steps, dtype=np.int16),
               "current_lives": np.int16(env.lives()),
               "next_lives": np.asarray(next_lives, dtype=np.int16)}
    return targets, branches, results, mask


def _row(targets, observed, valid, previous, seed, context_index):
    return {"frames": observed, "history_valid": valid, "previous_actions": previous, **targets,
            "seeds": np.int32(seed), "context_index": np.int8(context_index)}


def _explore(env, oracle, spec, context_index, rng, history, budget, epsilon, max_steps):
    """The epsilon-greedy prefix episode from the initial state.

    Returns its rows, their history keys and the action indices it took.
    Histories reset after loss of a life; they never cross a level boundary.
    The passed `env` is never mutated: every step adopts an isolated branch.
    """
    rows, keys, taken = [], set(), []
    frames, actions = [env.render()], [-1]
    for step in range(max_steps):
        state = oracle.state_of(env)
        before = oracle.distance_for(state)
        if before is None or env.state.value in ("WIN", "GAME_OVER"):
            break
        targets, branches, results, mask = _expand(env, oracle, before, spec["seed"], step)
        observed, valid, previous = history_arrays(frames, actions, history)
        keys.add(state_history_key(env, oracle, observed, valid, previous))
        rows.append(_row(targets, observed, valid, previous, spec["seed"], context_index))
        if len(rows) >= budget:
            break
        choice = (rng.randrange(4) if rng.random() < epsilon
                  else rng.choice([index for index in range(4) if mask & (1 << index)]))
        old_lives = env.lives()
        # This exact action was already executed above. Adopt its isolated
        # branch rather than rendering/animating a fifth engine action.
        env, result = branches[choice], results[choice]
        taken.append(choice)
        if result.finished:
            break
        if env.lives() < old_lives:
            frames, actions = [env.render()], [-1]
        else:
            frames.append(result.frame)
            actions.append(choice)
            frames, actions = frames[-history:], actions[-history:]
    return rows, keys, taken


def _expert(env, oracle, solution, spec, context_index, history, budget, seen, covered):
    """Walk the verified expert route once, branching only at selected states.

    Path indices below `covered` already appear verbatim among the explorer's
    rows; `budget` states are spread over the rest and the last pre-WIN state
    is always selected. Every step is checked against the complete teacher
    and the final action must win in the real engine; any disagreement raises
    rather than producing a row. Only rows with both the same public history
    and logical state are skipped; fog can hide distinct predictive states.
    """
    seed, length = spec["seed"], len(solution)
    planned = set(spread_indices(list(range(covered, length)), budget))
    frames, actions = [env.render()], [-1]
    start_lives = env.lives()
    rows, emitted = [], []
    for index, action in enumerate(solution):
        remaining = length - index
        if oracle.distance_for(oracle.state_of(env)) != remaining:
            raise ValueError(f"expert path state disagrees with the complete teacher "
                             f"at seed {seed}, step {index}")
        choice = names.ACTION_IDS.index(action)
        observed, valid, previous = history_arrays(frames, actions, history)
        key = state_history_key(env, oracle, observed, valid, previous)
        recorded = False
        if index in planned:
            if key in seen:
                # The explorer already holds this exact row (it rejoined the
                # route). Spend the slot on the next unplanned route state.
                substitute = next((j for j in range(index + 1, length) if j not in planned), None)
                if substitute is not None:
                    planned.add(substitute)
            else:
                targets, branches, results, mask = _expand(env, oracle, remaining, seed, index)
                if not mask & (1 << choice):
                    raise ValueError(f"expert action is not optimal for the complete teacher "
                                     f"at seed {seed}, step {index}")
                rows.append(_row(targets, observed, valid, previous, seed, context_index))
                seen.add(key)
                emitted.append(index)
                recorded = True
        if recorded:
            # Reuse the branch that already executed the expert's action.
            env, result = branches[choice], results[choice]
        else:
            result = env.perform(action)
        if result.frame is None:
            raise RuntimeError("an expert action returned no public successor observation")
        if index + 1 < length:
            if result.finished or env.lives() != start_lives:
                raise ValueError(f"expert path ended early or lost a life at seed {seed}, step {index}")
        elif not (result.won and env.lives() == start_lives and env.levels_completed == 1):
            raise ValueError(f"expert path did not win in the real engine at seed {seed}")
        frames.append(result.frame)
        actions.append(choice)
        frames, actions = frames[-history:], actions[-history:]
    return rows, emitted


def expert_budget(length):
    """At least start/middle/end, then one anchor per four route actions."""
    return min(length, max(3, (length + 3) // 4))


def _failures(initial, oracle, spec, context_index, history):
    """Play a bounded budget-exhaustion episode, retaining each pre-loss state.

    These are actual actions from the verified initial state, never edited lives
    or budgets. The complete oracle labels unreachable states with optimal=0;
    dynamics, reset distances and terminal labels remain fully supervised.
    The action trace permits independent replay. Failure coverage is measured,
    not assumed: pathological free-action traps can hit the explicit limit.
    """
    env = clone_env(initial)
    frames, actions = [env.render()], [-1]
    rows, taken, emitted = [], [], []
    # Avoid an unbounded walk on layouts with free refusals or refills.
    limit = 12 * (oracle.layout.max_steps // oracle.layout.step_cost + 2)
    stop = "action_limit"
    for step in range(limit):
        state = oracle.state_of(env)
        choices = []
        for action in range(4):
            following, outcome = simulate(oracle.layout, state, action, oracle.refills)
            if outcome != "won":
                # Spend budget, avoid free refusals and refills, and prefer death.
                choices.append(((outcome != "died", following[6], action), action, outcome))
        if not choices:
            stop = "only_winning_actions"
            break
        _, choice, outcome = min(choices)
        old_lives = env.lives()
        if outcome == "died":
            before = oracle.distance_for(state)
            targets, branches, results, _ = _expand(env, oracle, before, spec["seed"], step)
            observed, valid, previous = history_arrays(frames, actions, history)
            rows.append(_row(targets, observed, valid, previous, spec["seed"], context_index))
            emitted.append(step)
            env, result = branches[choice], results[choice]
            if env.lives() != old_lives - 1:
                raise ValueError("exhaustion trajectory disagrees with real-engine life loss")
        else:
            result = env.perform(names.ACTION_IDS[choice])
            if env.lives() != old_lives or result.finished:
                raise ValueError("exhaustion trajectory disagrees with real-engine transition")
        taken.append(choice)
        if result.finished:
            stop = "won" if result.won else "game_over"
            break
        if env.lives() < old_lives:
            frames, actions = [result.frame], [-1]
        else:
            frames, actions = (frames + [result.frame])[-history:], (actions + [choice])[-history:]
    return rows, {"failure_samples": len(rows), "failure_actions": taken,
                  "failure_indices": emitted, "failure_stop": stop}


def collect_level(spec, history=8, samples=32, epsilon=.15, context_index=None,
                  search_limit=600_000, coverage="prefix"):
    """A bounded episode, including all actions from each sampled live state.

    Prefix copies in Ls20Scenario are generated too and are never played. They
    preserve later-level rules (no free matching-cycle hint) in the real engine.
    Histories reset after loss of a life; they never cross a level boundary.

    ``coverage="mixed"`` returns at most `samples` rows: `samples // 2` from
    the epsilon-greedy prefix, the remainder spread over the verified expert
    trajectory including its last pre-WIN state, without duplicate rows.
    ``mixed_failure`` raises the expert budget to at least ceil(route/4),
    with three anchors for routes of length >=3, and appends up to three
    pre-death rows. The older modes retain their fixed-budget contract.
    """
    if coverage not in COVERAGE_MODES:
        raise ValueError(f"coverage must be one of {COVERAGE_MODES}")
    env, oracle, verification = verified_context(spec, context_index, search_limit)
    if env is None:
        return [], verification
    context_index = verification["context_index"]
    rng = random.Random(f"{FORMAT}:{spec['seed']}")
    failure_rows, failure_proof = [], {"failure_samples": 0}
    if coverage == "prefix":
        rows, _, _ = _explore(env, oracle, spec, context_index, rng, history, samples, epsilon,
                              max(samples * 3, 64))
        expert_rows, emitted = [], []
    else:
        if samples < 1:
            raise ValueError("mixed coverage needs a positive sample budget")
        explore_budget = samples // 2
        rows, keys, taken = [], set(), []
        if explore_budget:
            rows, keys, taken = _explore(env, oracle, spec, context_index, rng, history,
                                         explore_budget, epsilon, max(explore_budget * 3, 64))
        # Deterministic, the same route verified_context replayed to a WIN.
        solution = oracle.solution(seed=spec["seed"])
        matched = 0
        for choice, action in zip(taken, solution):
            if names.ACTION_IDS[choice] != action:
                break
            matched += 1
        # The explorer's row t is the expert's row t exactly while its actions
        # match the route; later coincidences are caught by history keys.
        covered = min(len(rows), matched + 1)
        anchor_budget = samples - len(rows)
        if coverage == "mixed_failure":
            anchor_budget = max(anchor_budget, expert_budget(len(solution)))
        expert_rows, emitted = _expert(clone_env(env), oracle, solution, spec, context_index, history,
                                       anchor_budget, keys, covered)
        rows = rows + expert_rows
        if not any(bool(row["won"].any()) for row in rows):
            raise ValueError(f"mixed coverage recorded no winning counterfactual at seed {spec['seed']}")
        if coverage == "mixed_failure":
            failure_rows, failure_proof = _failures(env, oracle, spec, context_index, history)
            rows.extend(failure_rows)
    win_rows = sum(bool(row["won"].any()) for row in rows)
    return rows, {**verification, "samples": len(rows), "context_index": context_index,
                  "fog": bool(spec.get("fog")), "difficulty": spec.get("difficulty"),
                  "reachable_states": oracle._reachable, "search_truncated": False,
                  "coverage": coverage, **failure_proof,
                  "explore_samples": len(rows) - len(expert_rows) - len(failure_rows),
                  "expert_samples": len(expert_rows), "expert_indices": emitted,
                  "life_loss_branches": sum(int(row["lost_life"].sum()) for row in rows),
                  "terminal_death_branches": sum(int((row["terminal"] & ~row["won"]).sum()) for row in rows),
                  "policy_unlabelled_rows": sum(int(row["optimal"]) == 0 for row in rows),
                  "win_rows": win_rows, "win_covered": win_rows > 0}


def _worker(task):
    spec, kwargs = task
    return collect_level(spec, **kwargs)


def build(specs, *, workers=1, progress=False, **kwargs):
    rows, levels = [], []
    tasks = [(spec, kwargs) for spec in specs]
    if workers > 1:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            for result in pool.imap(_worker, tasks):
                rows.extend(result[0])
                levels.append(result[1])
                if progress and len(levels) % 25 == 0:
                    print(f"{len(levels)}/{len(specs)} generated levels, {len(rows)} states", flush=True)
    else:
        for task in tasks:
            result = _worker(task)
            rows.extend(result[0])
            levels.append(result[1])
    if not rows:
        raise ValueError("no complete-oracle transition samples produced")
    arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    seeds = sorted(int(seed) for seed in np.unique(arrays["seeds"]))
    accepted = [level for level in levels if "excluded" not in level]
    arrays["meta"] = {"format": FORMAT, "source": "generated_only", "seeds": seeds,
                      "history": kwargs.get("history", 8), "samples": len(rows),
                      "alternatives_per_state": 4, "oracle_search": "complete_only",
                      "state_supervision_version": STATE_SUPERVISION_VERSION,
                      "successor_policy_supervision_version": SUCCESSOR_POLICY_SUPERVISION_VERSION,
                      "coverage": kwargs.get("coverage", "prefix"),
                      "samples_per_level": kwargs.get("samples", 32),
                      "failure_samples": sum(level.get("failure_samples", 0) for level in accepted),
                      "life_loss_branches": sum(level.get("life_loss_branches", 0) for level in accepted),
                      "terminal_death_branches": sum(level.get("terminal_death_branches", 0) for level in accepted),
                      "accepted_levels": len(accepted),
                      "win_rows": sum(level.get("win_rows", 0) for level in accepted),
                      "win_covered_levels": sum(bool(level.get("win_covered")) for level in accepted),
                      "levels": levels, "created": datetime.now(timezone.utc).isoformat()}
    return arrays


def save(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **{key: value for key, value in arrays.items() if key != "meta"},
                            meta=np.array(json.dumps(arrays["meta"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--samples-per-level", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=.15)
    parser.add_argument("--search-limit", type=int, default=600_000)
    parser.add_argument("--coverage", choices=COVERAGE_MODES, default="mixed_failure",
                        help="mixed_failure (default): mixed with route-proportional anchors "
                             "and up to three extra actual pre-death states; "
                             "prefix: epsilon-greedy episode from the start; "
                             "mixed: half prefix, half spread over the expert route "
                             "including its winning state")
    args = parser.parse_args()
    if min(args.limit, args.history, args.samples_per_level, args.workers, args.search_limit) < 1:
        parser.error("counts must be positive")
    if not 0 <= args.epsilon <= 1:
        parser.error("epsilon must be in [0, 1]")
    from ..ls20.bank import load
    specs = load(args.bank)[:args.limit]
    arrays = build(specs, workers=args.workers, history=args.history, samples=args.samples_per_level,
                   epsilon=args.epsilon, search_limit=args.search_limit, coverage=args.coverage,
                   progress=True)
    arrays["meta"]["bank"] = str(args.bank)
    save(args.out, arrays)
    meta = arrays["meta"]
    print(f"Saved {len(arrays['optimal'])} states with four actual alternatives each to {args.out} "
          f"({meta['coverage']} coverage; {meta['win_covered_levels']}/{meta['accepted_levels']} "
          f"accepted levels carry a real winning successor)")


if __name__ == "__main__":
    main()
