# Game and proof

This document records the durable game rules Pebby drives and the distinction
between planner proofs and learned-controller results.

## The game

The player is a 5×5 avatar on a 12×12 lattice inside a 64×64 frame of colour
indices 0–15. The carried state is a **(shape, colour, rotation)** triple.

| Tile | Rule |
|---|---|
| Wall | Blocks movement, but a blocked move still spends budget. |
| Shape, colour, or rotation cycler | Entering advances that attribute by one, wrapping around its domain. |
| Goal pad | Solid until the carried triple matches it exactly; a matching entry clears the goal. A rejected entry is free. |
| Refill | Restores the step budget and is consumed; the move is free. |
| Launcher pad | Flings the player in a fixed direction to the last free cell before the next wall or goal pad; the move is free and cannot complete a level. |
| Rail-riding cycler | Moves along a fixed circuit and advances once per accepted move. |

The level starts with a budget of 42 units. Moves cost one or two units. Exhausting
the budget loses one of three lives and restarts the level; losing all three lives
ends the game. Clearing every goal advances to the next level, and clearing level 7
wins. The four actions are up, down, left, and right. The full 64×64 frame is the
public observation; level 7 also includes fog of war.

## What the planner proves

Pebby's exact transition model is checked against the vendored game by driving both
implementations with the same action sequence and comparing the observable state,
carried triple, cleared goals, and budget after every step. Run the check with:

```bash
PYTHONPATH=. uv run python tools/differential.py
```

The exact planner searches the game state and returns an optimal route when its
search completes. A generated level is retained only when the route is replayed in
the real game and the game reports a win. Search caps or an incomplete search are
not proofs of solvability or optimality.

## Proof boundaries

The stored route for a generated level is an oracle solution, not a learned policy
output. Training data may use oracle-labelled counterfactuals, but a model must be
evaluated by rolling it out from reset in the real environment. Per-step imitation
accuracy, planner agreement, or a successful viewer replay is not a completion rate.

The vendored implementation is the source of truth for behavior. Read
[`third_party/ls20/PROVENANCE.md`](../third_party/ls20/PROVENANCE.md) before changing
or redistributing it.
