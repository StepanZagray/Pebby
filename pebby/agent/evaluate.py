"""Roll the trained policy out in the real LS20 game and count finished levels.

Validation accuracy is a proxy; this is the number. The policy plays greedily
from `reset()` in the same `Ls20Env` the training data came from, and the report
says how many levels it actually cleared, how many individual goal pads it lit,
and how its action count compares with the planner's optimum.

Two details of the game decide how the counting has to work, and both are easy
to get wrong:

* `goals_solved()` only ever describes the level the game is standing on. It is
  wiped when a level is cleared *and* when a life is lost, so accumulating it
  across a run either double-counts or silently loses progress. Cleared levels
  are credited instead from a per-level goal census taken before play starts.
* Clearing the last level sets the score to the level count and the state to
  WIN without moving `level_index`, and leaves that level's goals flagged
  solved. The final level must therefore not be credited twice.

The action that finishes a game still carries a frame; it is the action *after*
a finished game that returns an empty frame list. So the rollout stops on a
finished observation and never has a `None` frame to encode.

One inference rule earns its keep, and the reason is not the obvious one. A
move into a WALL is charged: the budget drops by the step cost and two pixels of
the step bar (rows 61-62) change, so the frame does move. A move into a GOAL PAD
whose triple does not match is refused *before* the budget is charged, so that
frame is byte-identical -- and a memoryless deterministic argmax fed an
identical frame must emit the identical action, forever. That is the absorbing
state, and it is the dominant failure: over 3,684 actions on 50 unseen levels,
2,734 (74.2%) produced a byte-identical frame, in 43 of the 50 episodes, and
every one of the 1,763 checked was a zero-budget bump against a goal pad cell.

So an action that leaves the frame unchanged is masked out of the next decision,
and the mask clears the moment the frame moves. It reads only the frame and uses
no game internals. On 300 unseen difficulty-1 levels it lifted completion from
14.7% +/- 2.0% to 20.7% +/- 2.3%. It is a controller heuristic bolted onto the
policy rather than something the policy learned, so `--on-stall repeat`
reproduces unmodified greedy argmax and every report records which rule produced
its numbers; quote both.
"""

from ..ls20.provenance import difficulty_version, generated_context

import argparse
import json
from pathlib import Path
import sys

from arcengine import GameState
import torch

from ..ls20 import names
from ..ls20.env import Ls20Env, Ls20Scenario, upstream
from .model import frames_to_tensor, load_checkpoint

REPORT_FORMAT = "pebby.ls20-evaluation.v1"


def _context_index(value, where="context"):
    """Validate the finite LS20 context namespace used by generated banks."""
    if type(value) is not int or not 0 <= value <= 6:
        raise ValueError(f"{where} must be an integer in 0..6")
    return value


def level_goal_counts(env):
    """How many goal pads each level in `env` has.

    `goal_triples()` describes one level at a time, so the denominator has to be
    collected by walking them. `set_level` re-clones the level from the pristine
    copy and the `reset()` that follows re-clones every level again, so the walk
    leaves no trace on the game that is about to be played.
    """
    counts = []
    for index in range(env.level_count):
        env.set_level(index)
        counts.append(len(env.goal_triples()))
    return counts


def optimal_actions(env):
    """Fewest actions that finish everything in `env`, or None if not known.

    Only ever asked of an env the caller did not build from a known spec, and
    only for levels the planner can search cheaply: `plan.Oracle` will happily
    spend two minutes and 6.8 GiB on shipped level 7 and then truncate silently.
    A level it will not search cheaply makes the whole total unknown rather than
    partial, which is the honest answer -- completion is still reported. Callers
    with specs should use `spec["optimal_actions"]`, and the shipped set has
    `pebby.ls20.shipped.OPTIMAL_ACTIONS`; both are exact and free.
    """
    try:
        from ..ls20.plan import Unplannable, oracle_for
    except ImportError:
        return None
    if env.level_count == 7:
        from ..ls20 import shipped
        return sum(shipped.OPTIMAL_ACTIONS)  # The shipped sequence, from cache.
    total = 0
    for index in range(env.level_count):
        env.set_level(index)
        try:
            solution = oracle_for(env).solution()
        except (Unplannable, ValueError):
            return None
        if solution is None:
            return None
        total += len(solution)
    return total


def shipped_levels():
    """The seven levels the real game ships, as the objects `Ls20Env` installs.

    Playing them one per env is the only way to see level 5 when the policy dies
    on level 2; the sequential seven-level run stays the headline, because that
    is the game as a player meets it, but it cannot produce a per-level table.
    """
    return list(upstream().levels)


def level_optimum(index):
    """(optimal action count, provenance) for shipped level `index`, from cache.

    Never plan a shipped level here. All seven are plannable now, which is worse
    than it sounds: level 6 searches 12.0M states at 4.3 GiB and level 7 21.7M at
    6.8 GiB, taking about two minutes each, and `Oracle`'s default 600k limit
    truncates them SILENTLY -- `solution()` returns None and it reads like
    "unsolvable" when it means "ran out of budget". `pebby.ls20.shipped` holds
    the verified answers, so an evaluation costs nothing and cannot be fooled.
    """
    from ..ls20 import shipped
    return shipped.optimal(index), "cached exact optimum"


def shipped_table(policy, max_actions, device=None, on_stall="next-best"):
    """One isolated run per shipped level, against its cached optimum."""
    from ..ls20 import shipped
    rows = []
    for index, level in enumerate(shipped_levels()):
        optimum, reason = level_optimum(index)
        run = rollout(policy, Ls20Scenario(level, index), max_actions, device, optimum, on_stall)
        rows.append({"level": index + 1, "completed": run["completed"], "actions": run["actions"],
                     "ending": run["ending"], "goals_cleared": run["goals_cleared"],
                     "goals_total": run["goals_total"], "optimal": optimum,
                     "human_baseline": shipped.HUMAN_BASELINE[index],
                     "actions_vs_optimal": run["actions_vs_optimal"], "optimal_reason": reason})
    return rows


def choose_action(policy, frame, device=None, blocked=(), temperature=0., generator=None,
                  history=None):
    """Pick an action from the four logits; returns (game action id, index).

    `blocked` holds action indices already shown to do nothing from this exact
    frame. Scoring on a list rather than in-place on the tensor keeps this valid
    under `torch.inference_mode`, where an in-place write is refused.

    `temperature` 0 is argmax, the strict protocol. Above 0 the action is
    sampled from the tempered softmax, which matters for retries: a
    deterministic policy that is reset replays the identical failing trajectory
    forever, so retrying is only meaningful if the policy can vary.
    """
    scores = history.scores() if history is not None else policy(frames_to_tensor(frame, device))[0].float()
    if blocked:
        scores = scores.masked_fill(torch.tensor([i in blocked for i in range(len(scores))],
                                                 device=scores.device), float("-inf"))
    if temperature > 0:
        index = int(torch.multinomial((scores / temperature).softmax(0), 1, generator=generator))
    else:
        index = int(scores.argmax())
    return names.ACTION_IDS[index], index


def greedy_action(policy, frame, device=None, blocked=()):
    """Argmax only. Kept as the name the strict protocol is written against."""
    return choose_action(policy, frame, device, blocked)


@torch.inference_mode()
def rollout(policy, env, max_actions, device=None, oracle_length=None, on_stall="next-best",
            temperature=0., generator=None):
    """Play one greedy game from `env.reset()` and report what it achieved.

    `policy` is anything with `Ls20Policy`'s call signature -- a frames tensor
    in, logits [B, 4] out -- so a stub is as playable as a checkpoint. Greedy
    argmax over those logits is the whole policy at inference time; there is no
    oracle at inference. World policies receive causal observation/action history.
    """
    if hasattr(policy, "eval"):
        policy.eval()
    goals_per_level = level_goal_counts(env)
    frame = env.reset()
    from .history import for_policy
    history = for_policy(policy, frame, device)
    ending, actions, blocked, stalls = "capped", 0, set(), 0
    while actions < max_actions:
        old_lives, old_level = env.lives(), env.level_index
        action, index = choose_action(policy, frame, device, blocked, temperature, generator, history)
        observation = env.perform(action)
        actions += 1
        if history is not None and observation.frame is not None:
            history.observe(observation.frame, index,
                            reset=env.lives() < old_lives or env.level_index != old_level)
        if observation.finished:
            ending = "win" if observation.won else "game_over"
            break
        if observation.frame is None:
            break  # Defensive: an empty frame is never worth encoding.
        # An unchanged frame means the move changed nothing at all -- in
        # practice a goal pad refusing an unmatched triple, which costs no
        # budget -- so repeating it loops forever. Anything else clears the
        # mask: the policy is entitled to that action again from a state it has
        # not yet tried.
        if observation.frame == frame:
            stalls += 1
            blocked = blocked | {index} if on_stall == "next-best" else blocked
            if len(blocked) == len(names.ACTION_IDS):
                ending = "stuck"  # Every direction did nothing: nothing left to try.
                break
        else:
            blocked = set()
        frame = observation.frame
    cleared_levels = env.levels_completed
    goals_cleared = sum(goals_per_level[:cleared_levels])
    if cleared_levels < env.level_count:
        # Cleared levels are credited in full by the census above; only a level
        # still in play contributes its live tally, which a lost life truncates.
        goals_cleared += sum(env.goals_solved())
    return {"completed": ending == "win",
            "levels_completed": cleared_levels,
            "levels_total": env.level_count,
            "goals_cleared": goals_cleared,
            "goals_total": sum(goals_per_level),
            "actions": actions,
            "ending": ending,
            "stalls": stalls,
            "on_stall": on_stall,
            "temperature": temperature,
            "optimal": oracle_length,
            "actions_vs_optimal": actions / oracle_length if oracle_length else None,
            "final_level": env.level_index,
            "lives_left": env.lives()}


@torch.inference_mode()
def budgeted_rollout(policy, env, budget, device=None, temperature=.5, on_stall="next-best",
                     generator=None):
    """Play the way the benchmark does: one action budget, RESET and retry inside it.

    ARC-AGI-3 scores against a budget of roughly 5x the human median action
    count, not one flawless attempt, and LS20 hands the player a RESET action and
    three lives precisely so that failing is recoverable. Humans take two to four
    times optimal on these levels, so a single greedy attempt under an action cap
    is a much stricter test than the game applies. Retries only mean anything
    with a non-zero temperature: a deterministic policy that resets replays the
    same failing trajectory forever.
    """
    attempts, used = [], 0
    while used < budget:
        attempt = rollout(policy, env, budget - used, device, None, on_stall, temperature, generator)
        used += attempt["actions"]
        attempts.append({"attempt": len(attempts) + 1, "actions": attempt["actions"],
                         "ending": attempt["ending"], "levels_completed": attempt["levels_completed"],
                         "goals_cleared": attempt["goals_cleared"]})
        if attempt["completed"]:
            break
    best = max(attempts, key=lambda a: (a["levels_completed"], a["goals_cleared"]))
    return {"completed": attempts[-1]["ending"] == "win", "budget": budget, "actions": used,
            "attempts": len(attempts), "levels_completed": best["levels_completed"],
            "goals_cleared": best["goals_cleared"], "goals_total": attempt["goals_total"],
            "levels_total": attempt["levels_total"], "temperature": temperature,
            "endings": [a["ending"] for a in attempts]}


@torch.inference_mode()
def diverse_rollout(policy, env, max_actions, device=None, oracle_length=None, *, resets=True,
                    **knobs):
    """Play one env through `DiverseLivesController`, the way `inference.py` serves it.

    Native lives, and after a GAME_OVER a level-only RESET (charged as one
    action, exactly as the competition session charges it) until `max_actions`
    is spent or the game is won. `resets=False` stops at the first GAME_OVER
    instead, which is the isolated three-lives-only reading. The stall mask,
    penalties and per-life seeded sampling are runtime heuristics bolted onto
    the policy, never something it learned; `knobs` are the controller's.
    """
    from types import MethodType
    from .diverse_controller import DiverseLivesController
    if hasattr(policy, "eval"):
        policy.eval()
    goals_per_level = level_goal_counts(env)
    frame = env.reset()
    # RESET after a GAME_OVER restarts the current level only, as the competition
    # session enforces; the engine's own handler would fully restart the game
    # whenever its action counter reads zero.
    env.game.handle_reset = MethodType(lambda game: game.level_reset(), env.game)
    controller = DiverseLivesController(policy, device, **knobs)
    controller.start(frame, level_index=env.level_index)
    ending, actions, reset_count, lives_lost, stalls = "capped", 0, 0, 0, 0
    while actions < max_actions:
        if env.state == GameState.GAME_OVER:
            if not resets:
                ending = "game_over"
                break
            observation = env.perform(0)
            actions += 1
            reset_count += 1
            controller.observe(observation.frame, None, reset=True)
            frame = observation.frame
            continue
        old_lives, old_level = env.lives(), env.level_index
        index = controller.decide(frame, level_index=env.level_index, lives=old_lives, state=env.state)
        observation = env.perform(names.ACTION_IDS[index])
        actions += 1
        if observation.frame is None:
            break  # Defensive: an empty frame is never worth encoding.
        life_lost = env.lives() < old_lives
        lives_lost += life_lost
        stalls += observation.frame == frame
        controller.observe(observation.frame, index, life_lost=life_lost,
                           level_changed=env.level_index != old_level, reset=False)
        frame = observation.frame
        if observation.won:
            ending = "win"
            break
    cleared_levels = env.levels_completed
    goals_cleared = sum(goals_per_level[:cleared_levels])
    if cleared_levels < env.level_count:
        goals_cleared += sum(env.goals_solved())
    return {"completed": ending == "win", "levels_completed": cleared_levels,
            "levels_total": env.level_count, "goals_cleared": goals_cleared,
            "goals_total": sum(goals_per_level), "actions": actions, "resets": reset_count,
            "lives_lost": lives_lost, "stalls": stalls, "ending": ending, "protocol": "diverse",
            "controller": controller.metadata, "optimal": oracle_length,
            "actions_vs_optimal": actions / oracle_length if oracle_length else None,
            "final_level": env.level_index, "lives_left": env.lives()}


def diverse_completion(policy, levels, optima, multiplier=5, device=None, context_indices=None,
                       **knobs):
    """Aggregate `diverse_rollout` over levels, one env each, under `multiplier` x optimum."""
    runs = []
    for index, (level, optimum) in enumerate(zip(levels, optima)):
        env = (Ls20Scenario(level, context_indices[index]) if context_indices is not None
               else Ls20Env(levels=[level]))
        budget = max(1, int(multiplier * optimum))
        runs.append({"run": index, "budget": budget,
                     **diverse_rollout(policy, env, budget, device, optimum, **knobs)})
    total = sum(run["levels_total"] for run in runs)
    completed = sum(run["levels_completed"] for run in runs)
    goals_total = sum(run["goals_total"] for run in runs)
    return {"protocol": "diverse", "multiplier": multiplier, "learned": False,
            "runtime_heuristics": True,
            "controller": runs[0]["controller"] if runs else None,
            "levels": total, "completed": completed,
            "completion_rate": completed / total if total else None,
            "goals_cleared": sum(run["goals_cleared"] for run in runs), "goals_total": goals_total,
            "goal_rate": sum(run["goals_cleared"] for run in runs) / goals_total if goals_total else None,
            "mean_actions": sum(run["actions"] for run in runs) / len(runs) if runs else None,
            "resets": sum(run["resets"] for run in runs),
            "lives_lost": sum(run["lives_lost"] for run in runs),
            "budget_exhausted": sum(not run["completed"] for run in runs),
            "runs": runs}


def budgeted_completion(policy, levels, optima, multiplier=5, device=None, temperature=.5,
                        on_stall="next-best", sample_seed=0, context_indices=None):
    """Aggregate `budgeted_rollout` over levels, one env each."""
    generator = torch.Generator(device="cpu" if device is None else device).manual_seed(sample_seed)
    runs = []
    for index, (level, optimum) in enumerate(zip(levels, optima)):
        env = (Ls20Scenario(level, context_indices[index]) if context_indices is not None
               else Ls20Env(levels=[level]))
        budget = max(1, int(multiplier * optimum))
        runs.append({"run": index, **budgeted_rollout(policy, env, budget, device, temperature,
                                                      on_stall, generator)})
    total = sum(run["levels_total"] for run in runs)
    completed = sum(run["levels_completed"] for run in runs)
    goals_total = sum(run["goals_total"] for run in runs)
    return {"protocol": "budgeted", "multiplier": multiplier, "temperature": temperature,
            "levels": total, "completed": completed,
            "completion_rate": completed / total if total else None,
            "goals_cleared": sum(run["goals_cleared"] for run in runs), "goals_total": goals_total,
            "goal_rate": sum(run["goals_cleared"] for run in runs) / goals_total if goals_total else None,
            "mean_actions": sum(run["actions"] for run in runs) / len(runs) if runs else None,
            "mean_attempts": sum(run["attempts"] for run in runs) / len(runs) if runs else None,
            "budget_exhausted": sum(not run["completed"] for run in runs),
            "runs": runs}


def completion_rate(policy, levels=None, max_actions=200, device=None, oracles=None, on_stall="next-best",
                    context_indices=None):
    """Roll `policy` out over every level and aggregate the runs.

    `levels=None` is the real game exactly as shipped: ONE env running all seven
    levels in sequence, so the headline is how many of the seven it cleared
    before it ran out of lives. A list of `Level` objects is played one level per
    env instead, which is the only way per-level completion is unambiguous -- in
    a shared env a failure on level 2 hides whatever the policy could have done
    on level 3.

    `oracles` are optimal action counts aligned with the envs; when omitted the
    planner is asked, and every entry it cannot answer stays None.
    """
    envs = ([Ls20Env(levels=None)] if levels is None else
            [Ls20Scenario(level, context_indices[index]) if context_indices is not None
             else Ls20Env(levels=[level]) for index, level in enumerate(levels)])
    if oracles is None:
        oracles = [optimal_actions(env) for env in envs]
    elif len(oracles) != len(envs):
        raise ValueError(f"got {len(oracles)} oracle lengths for {len(envs)} envs")
    runs = [{"run": index, **rollout(policy, env, max_actions, device, oracle, on_stall)}
            for index, (env, oracle) in enumerate(zip(envs, oracles))]
    # Summing levels_total/levels_completed makes both layouts read identically:
    # seven levels inside one run, or N runs of one level each.
    total = sum(run["levels_total"] for run in runs)
    completed = sum(run["levels_completed"] for run in runs)
    goals_cleared = sum(run["goals_cleared"] for run in runs)
    goals_total = sum(run["goals_total"] for run in runs)
    ratios = [run["actions_vs_optimal"] for run in runs if run["actions_vs_optimal"] is not None]
    return {"format": REPORT_FORMAT,
            "levels": total,
            "runs_played": len(runs),
            "max_actions": max_actions,
            "completed": completed,
            "completion_rate": completed / total if total else None,
            "goals_cleared": goals_cleared,
            "goals_total": goals_total,
            "goal_rate": goals_cleared / goals_total if goals_total else None,
            "mean_actions": sum(run["actions"] for run in runs) / len(runs) if runs else None,
            "mean_actions_vs_optimal": sum(ratios) / len(ratios) if ratios else None,
            "runs_with_oracle": len(ratios),
            "won": sum(run["completed"] for run in runs),
            "game_over": sum(run["ending"] == "game_over" for run in runs),
            "capped": sum(run["ending"] == "capped" for run in runs),
            "stuck": sum(run["ending"] == "stuck" for run in runs),
            "stalled_actions": sum(run["stalls"] for run in runs),
            # A run that spends most of its actions on frame-preserving bumps is
            # in a pad loop, which is a different failure from wandering.
            "stall_dominated": sum(run["stalls"] * 2 >= run["actions"] for run in runs),
            "on_stall": on_stall,
            "runs": runs}


@torch.inference_mode()
def optimality_rate(specs, policy, max_actions=120, device=None, on_stall="next-best",
                    context_indices=None):
    """How often the policy's move is ONE OF the optimal moves, not the oracle's pick.

    Imitation accuracy is measured against a single label, but 30% of states
    have more than one equally optimal action (measured: mean 1.397 optimal
    actions per state over 981 on-path test-bank states), so a perfect player
    scores only 83.6% imitation accuracy. What actually decides completion is
    whether the distance to completion fell by one, which the level's own Oracle
    answers directly. This is the per-step number worth steering on.

    Costs one Oracle per level, so it is measured on a slice, not on everything.
    """
    from ..ls20.generate import build_level
    from ..ls20.plan import oracle_for
    if context_indices is None:
        context_indices = [spec.get("training_context_index", 0) for spec in specs]
    if len(context_indices) != len(specs):
        raise ValueError(f"got {len(context_indices)} context indices for {len(specs)} levels")
    context_indices = [_context_index(value, f"context index for level {index}")
                       for index, value in enumerate(context_indices)]
    optimal = measured = 0
    for index, spec in enumerate(specs):
        level = build_level(spec)
        env = Ls20Scenario(level, context_indices[index])
        frame = env.reset()
        from .history import for_policy
        history = for_policy(policy, frame, device)
        oracle = oracle_for(env)
        if oracle.truncated:
            raise ValueError("optimality requires a complete oracle search; this level truncated")
        blocked = set()
        for _ in range(max_actions):
            before = oracle.distance_for(oracle.state_of(env))
            old_lives = env.lives()
            action, index = choose_action(policy, frame, device, blocked, history=history)
            observation = env.perform(action)
            if history is not None and observation.frame is not None:
                history.observe(observation.frame, index, reset=env.lives() < old_lives)
            done = observation.finished or env.levels_completed
            # A finished level is distance zero; an unreachable state has no
            # distance at all and is not scored either way.
            after = 0 if done else oracle.distance_for(oracle.state_of(env))
            if before is not None:
                measured += 1
                optimal += after is not None and after == before - 1
            if done or observation.frame is None:
                break
            blocked = blocked | {index} if (observation.frame == frame and on_stall == "next-best") \
                else (blocked if observation.frame == frame else set())
            frame = observation.frame
    return {"optimal_moves": optimal, "measured_moves": measured,
            "optimality_rate": optimal / measured if measured else None, "levels": len(specs)}


def bank_levels(path, limit=None):
    """Levels and optimal lengths from a bank file, the split of record.

    The bank is the honest test set: its seeds start at 2,000,000 and cannot
    collide with the training seeds, and `bank.load` refuses a bank built by an
    older generator, so a stale file fails loudly instead of quietly measuring
    the wrong game.
    """
    from ..ls20.bank import load
    from ..ls20.generate import build_level
    specs = load(path)[:limit]
    levels, optima = [], []

    def integer(value, field, index):
        if isinstance(value, bool):
            raise ValueError(f"bank level {index} has an invalid {field}")
        try:
            result = int(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"bank level {index} has an invalid {field}") from error
        if isinstance(value, float) and value != result:
            raise ValueError(f"bank level {index} has an invalid {field}")
        return result

    for index, spec in enumerate(specs):
        has_context = "training_context_index" in spec
        if difficulty_version(spec) and (not has_context or
                spec["training_context_index"] != generated_context(spec)):
            raise ValueError(f"bank level {index} has a mismatched calibrated context")
        has_context_optimum = "context_optimal_actions" in spec
        if has_context and not has_context_optimum:
            raise ValueError(f"bank level {index} has training_context_index but no "
                             "context_optimal_actions")
        if has_context_optimum and not has_context:
            raise ValueError(f"bank level {index} has context_optimal_actions but no "
                             "training_context_index")
        try:
            optimum = integer(spec["context_optimal_actions"] if has_context
                              else spec["optimal_actions"],
                              "context_optimal_actions" if has_context else "optimal_actions", index)
        except (KeyError, TypeError, ValueError) as error:
            field = "context_optimal_actions" if has_context else "optimal_actions"
            raise ValueError(f"bank level {index} has no valid {field}") from error
        if optimum < 1:
            raise ValueError(f"bank level {index} has non-positive optimum {optimum}")
        if has_context:
            _context_index(spec["training_context_index"],
                           f"bank level {index} training_context_index")
        levels.append(build_level(spec))
        optima.append(optimum)
    return levels, optima, specs


def generated_levels(count, difficulty, seed):
    """`count` generated levels from consecutive seeds, with their optimal lengths.

    Returns (levels, optimal action counts). The generator already proved each
    level completable by replaying an optimal solution in the real engine, so
    `spec["optimal_actions"]` is exact and free -- re-deriving it by building an
    Oracle here would cost up to a second per level for the same number.

    Imported lazily and deliberately not caught: the CLI turns a missing
    generator into a readable error, and a caller in a notebook wants the real
    ImportError.
    """
    from ..ls20.generate import build_level, generate_level
    specs = [generate_level(seed + index, difficulty) for index in range(count)]
    return [build_level(spec) for spec in specs], [spec["optimal_actions"] for spec in specs]


def default_max_actions(source, shipped_level=None):
    """Return the CLI cap for a source when the user leaves it unspecified.

    Function callers retain the historical ``max_actions=200`` defaults.  The
    command-line evaluator uses caps that cover the protocol it selected: a
    generated strict episode gets 300 actions, one isolated shipped level gets
    five times its human baseline, and the sequential shipped run gets the sum
    of those seven per-level caps.
    """
    from ..ls20 import shipped
    if source == "generated":
        return 300
    if source.startswith("bank "):
        return 300
    if source == "shipped":
        return sum(5 * baseline for baseline in shipped.HUMAN_BASELINE)
    if source == "shipped_level":
        if shipped_level not in range(1, shipped.LEVEL_COUNT + 1):
            raise ValueError("shipped_level must be in 1..7")
        return 5 * shipped.HUMAN_BASELINE[shipped_level - 1]
    if source.startswith("shipped level "):
        try:
            level = int(source.rsplit(" ", 1)[1])
        except ValueError as error:
            raise ValueError(f"unknown evaluation source {source!r}") from error
        return default_max_actions("shipped_level", level)
    raise ValueError(f"unknown evaluation source {source!r}")


def parameter_count(policy):
    """Count loaded parameters when an older checkpoint omitted metadata."""
    method = getattr(policy, "parameter_count", None)
    if callable(method):
        return int(method())
    return sum(parameter.numel() for parameter in policy.parameters())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--loops", type=int,
                        help="Fixed inference depth for a looped checkpoint; does not change weights")
    parser.add_argument("--shipped", action="store_true",
                        help="Play the seven shipped levels in one env (the default)")
    parser.add_argument("--shipped-level", type=int, choices=range(1, 8),
                        help="Evaluate exactly one official level, numbered 1..7")
    parser.add_argument("--levels", type=int, help="Play this many generated levels, one env each")
    parser.add_argument("--bank", type=Path, help="Play the levels in this bank file, one env each")
    parser.add_argument("--limit", type=int, help="Use only the first N levels of --bank")
    parser.add_argument("--protocol", choices=("strict", "budgeted", "both", "diverse"), default="both",
                        help="strict: one greedy attempt under --max-actions. budgeted: the "
                             "benchmark's own terms, a total budget with RESET and retries. "
                             "diverse: native lives and level-only RESET through the same "
                             "DiverseLivesController inference.py serves (runtime heuristics, "
                             "not learning); --temperature and --sample-seed feed it")
    parser.add_argument("--temperature", type=float, default=.5,
                        help="Sampling temperature for the budgeted protocol; 0 is argmax, which "
                             "makes retries pointless because the trajectory repeats exactly")
    parser.add_argument("--budget-multiplier", type=float, default=5.,
                        help="Total budget as a multiple of optimal, or of the human median "
                             "for the shipped set, which is the benchmark convention")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--optimality", type=int, metavar="N",
                        help="Also measure per-step optimality on the first N bank levels; "
                             "costs one planner build per level")
    parser.add_argument("--difficulty", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1_000_000, help="First generated-level seed")
    parser.add_argument("--max-actions", type=int, default=None,
                        help="Strict action cap. Defaults to 300 for generated levels, "
                             "5x the selected human baseline for one shipped level, "
                             "or 5x the sum of shipped human baselines for the sequential set.")
    parser.add_argument("--on-stall", choices=("next-best", "repeat"), default="next-best",
                        help="What to do when an action leaves the frame unchanged. 'repeat' is "
                             "unmodified greedy argmax, which measurably loops against walls")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="auto")
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args()
    chosen = [name for name, value in (("--shipped", args.shipped),
                                       ("--shipped-level", args.shipped_level), ("--levels", args.levels),
                                       ("--bank", args.bank)) if value]
    if len(chosen) > 1:
        parser.error(f"{', '.join(chosen)} are mutually exclusive")
    if args.levels is not None and args.levels < 1:
        parser.error("--levels must be positive")
    if args.limit is not None and (args.bank is None or args.limit < 1):
        parser.error("--limit is positive and only applies to --bank")
    if args.bank is not None and not args.bank.exists():
        parser.error(f"bank not found: {args.bank}")
    if ((args.max_actions is not None and args.max_actions < 1) or args.seed < 0):
        parser.error("max-actions must be positive and seed must not be negative")
    if args.temperature < 0 or args.budget_multiplier <= 0:
        parser.error("temperature must not be negative and budget-multiplier must be positive")
    if not args.checkpoint.exists():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but torch.cuda.is_available() is False")
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    try:
        policy, checkpoint = load_checkpoint(args.checkpoint, device)
    except (ValueError, KeyError) as error:
        parser.error(f"{args.checkpoint} is not a usable LS20 policy checkpoint: {error}")

    if args.loops is not None:
        if (args.loops < 1 or policy.config().get("architecture") not in ("looped", "world")
                or not hasattr(policy, "loops")):
            parser.error("--loops must be positive and requires a checkpoint with mutable inference loops; "
                         "spatial policies keep their encoder depth fixed")
        policy.loops = args.loops

    levels, oracles, specs, source = None, None, None, "shipped"
    context_indices = None
    try:
        if args.bank is not None:
            levels, oracles, specs = bank_levels(args.bank, args.limit)
            # Context-tagged banks were solved in a non-default game context;
            # ordinary banks retain the historical context-0 behavior.
            context_indices = [int(spec.get("training_context_index", 0)) for spec in specs]
            source = f"bank {args.bank.name}"
        elif args.levels is not None:
            levels, oracles = generated_levels(args.levels, args.difficulty, args.seed)
            source = "generated"
        elif args.shipped_level is not None:
            levels = [shipped_levels()[args.shipped_level - 1]]
            oracles = [level_optimum(args.shipped_level - 1)[0]]
            source = f"shipped level {args.shipped_level}"
    except ImportError as error:
        parser.error("generated levels need pebby.ls20.generate, which is not implemented yet "
                     f"({error}); run with --shipped instead")
    except (ValueError, RuntimeError) as error:
        # The generator and the bank loader own what counts as a valid level and
        # a current bank; repeat their reason rather than a traceback.
        parser.error(f"could not load levels: {error}")
    if levels is not None and not levels:
        parser.error("no levels to evaluate")
    if args.max_actions is None:
        args.max_actions = default_max_actions(source, args.shipped_level)

    report = {"format": REPORT_FORMAT}
    if args.protocol == "diverse":
        from ..ls20 import shipped as shipped_cache
        diverse_levels = shipped_levels() if levels is None else levels
        diverse_optima = shipped_cache.HUMAN_BASELINE if levels is None else oracles
        if args.shipped_level is not None:
            diverse_optima = [shipped_cache.HUMAN_BASELINE[args.shipped_level - 1]]
        report["diverse"] = diverse_completion(policy, diverse_levels, diverse_optima,
                                               args.budget_multiplier, device,
                                               list(range(7)) if levels is None else
                                               ([args.shipped_level - 1] if args.shipped_level
                                                else context_indices),
                                               temperature=args.temperature,
                                               base_seed=args.sample_seed)
    if args.protocol in ("strict", "both"):
        contexts = ([args.shipped_level - 1] if args.shipped_level is not None
                    else context_indices)
        report = completion_rate(policy, levels, args.max_actions, device, oracles, args.on_stall, contexts)
        report["protocol"] = "strict"
    if args.protocol in ("budgeted", "both"):
        # The shipped set is budgeted against the human median, which is what the
        # benchmark scores against; generated levels have an exact optimum.
        from ..ls20 import shipped as shipped_cache
        budget_levels = shipped_levels() if levels is None else levels
        budget_optima = shipped_cache.HUMAN_BASELINE if levels is None else oracles
        if args.shipped_level is not None:
            budget_optima = [shipped_cache.HUMAN_BASELINE[args.shipped_level - 1]]
        report["budgeted"] = budgeted_completion(policy, budget_levels, budget_optima,
                                                 args.budget_multiplier, device, args.temperature,
                                                 args.on_stall, args.sample_seed,
                                                 list(range(7)) if levels is None else
                                                 ([args.shipped_level - 1] if args.shipped_level
                                                  else context_indices))
    if levels is None and args.protocol in ("strict", "both"):
        # The sequential run answers "how far into the real game does it get";
        # the table answers "which levels can it do at all", which the sequence
        # physically cannot show once the policy dies early.
        report["per_level"] = shipped_table(policy, args.max_actions, device, args.on_stall)
    if args.optimality and specs:
        report["optimality"] = optimality_rate(specs[:args.optimality], policy, args.max_actions,
                                               device, args.on_stall,
                                               context_indices[:args.optimality]
                                               if context_indices is not None else None)
    report.update({"checkpoint": str(args.checkpoint), "device": str(device),
                   "parameters": (checkpoint.get("parameters")
                                  if checkpoint.get("parameters") is not None
                                  else parameter_count(policy)),
                   "architecture": policy.config().get("architecture", "cnn"),
                   "inference_loops": getattr(policy, "loops", policy.config().get("loops")),
                   "checkpoint_loops": checkpoint.get("config", {}).get("loops"),
                   "levels_source": source,
                   "difficulty": args.difficulty if source == "generated" else None,
                   "seed": args.seed if source == "generated" else None,
                   "bank": str(args.bank) if args.bank else None})

    diverse = report.get("diverse")
    if diverse:
        print(f"DIVERSE ({diverse['multiplier']:g}x optimal, T={diverse['controller']['temperature']}, "
              f"runtime heuristics, not learning) | {diverse['completed']}/{diverse['levels']} levels "
              f"completed ({diverse['completion_rate']:.1%}) | goals {diverse['goals_cleared']}/"
              f"{diverse['goals_total']} | {diverse['mean_actions']:.1f} actions, "
              f"{diverse['resets']} resets, {diverse['lives_lost']} lives lost", flush=True)
    budgeted = report.get("budgeted")
    if budgeted:
        print(f"BUDGETED ({budgeted['multiplier']:g}x optimal, T={budgeted['temperature']}) | "
              f"{budgeted['completed']}/{budgeted['levels']} levels completed "
              f"({budgeted['completion_rate']:.1%}) | goals {budgeted['goals_cleared']}/"
              f"{budgeted['goals_total']} | {budgeted['mean_actions']:.1f} actions over "
              f"{budgeted['mean_attempts']:.2f} attempts", flush=True)
    if args.protocol in ("budgeted", "diverse"):
        text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.report_out:
            args.report_out.parent.mkdir(parents=True, exist_ok=True)
            args.report_out.write_text(text)
            print(f"Report {args.report_out}", flush=True)
        return
    ratio = report["mean_actions_vs_optimal"]
    print(f"STRICT (one greedy attempt, cap {args.max_actions}) | "
          f"{args.checkpoint} | {device} | {report['levels_source']} | "
          f"{report['completed']}/{report['levels']} levels completed "
          f"({report['completion_rate']:.1%})", flush=True)
    print(f"Goals {report['goals_cleared']}/{report['goals_total']} ({report['goal_rate']:.1%}) | "
          f"{report['mean_actions']:.1f} actions per run" +
          (f" | {ratio:.2f}x optimal over {report['runs_with_oracle']} run(s)" if ratio else
           " | no oracle length available") +
          f" | won {report['won']}, game over {report['game_over']}, capped {report['capped']}, "
          f"stuck {report['stuck']} | {report['stall_dominated']} runs stuck in a pad loop "
          f"| on-stall {args.on_stall}", flush=True)
    if report.get("optimality"):
        print(f"Per-step optimality {report['optimality']['optimality_rate']:.1%} over "
              f"{report['optimality']['measured_moves']} scored moves on "
              f"{report['optimality']['levels']} levels (a perfect player scores 100% here, "
              f"but only 83.6% on imitation accuracy)", flush=True)
    for row in report.get("per_level", []):
        ratio = (f"{row['actions_vs_optimal']:.2f}x optimal ({row['optimal']})"
                 if row["actions_vs_optimal"] else row["optimal_reason"])
        print(f"  level {row['level']} | {'completed' if row['completed'] else row['ending']:>10} | "
              f"goals {row['goals_cleared']}/{row['goals_total']} | {row['actions']:3d} actions | {ratio}",
              flush=True)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(text)
        print(f"Report {args.report_out}", flush=True)
    print(text)


if __name__ == "__main__":
    sys.exit(main())
