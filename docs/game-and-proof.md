# Game and proof

This document describes the game model Pebby drives and the evidence behind the
generated levels and planner results.

## The game

You are a 5×5 avatar on a 12×12 lattice inside a 64×64 frame of colour indices 0–15.
You carry a **(shape, colour, rotation)** triple — 6 shapes, 4 colours, 4 rotations.

| Tile | Rule |
|---|---|
| Wall | Blocks. A blocked move still spends budget. |
| Shape / colour / rotation cycler | Entering it advances that one attribute by +1 (mod 6 or 4). |
| Goal pad | **Solid unless your triple matches it exactly.** Step on it matching and that goal clears. A rejected bump is free — it spends no budget. |
| Refill | Restores the step budget to full, and that move is free. Consumed. |
| Launcher pad | Flings you in a fixed direction to the last free cell before the next wall or goal pad. Free, and it cannot complete a level. |
| Rail-riding cycler | A cycler that walks a fixed circuit, advancing once per accepted move. |

Budget is 42 units at 1 or 2 per move. Running it out costs one of 3 lives and
restarts the level; three losses is `GAME_OVER`. Clearing every goal advances;
clearing level 7 wins. Actions are `ACTION1..4` = up/down/left/right. The whole
64×64 frame is observable — LS20's `Camera(16, 16)` is dead config, overridden
because every level declares `grid_size=(64, 64)`. Level 7 adds fog of war, drawn into
the frame.

## What is proved

The transition model is exact for the entire game. `tools/differential.py` drives the
real engine and Pebby's model in lockstep over random action sequences and compares
cell, triple, cleared goals, and remaining budget at every step:

```
L1 16918/16918   L2 8848/8848    L3 9003/9003    L4 20507/20507
L5 14399/14399   L6 22961/22961  L7 20880/20880     (12267 of these standing on a moving cycler)
```

That is 113,516 transitions with zero mismatches. Rerun it with:

```bash
PYTHONPATH=. uv run python tools/differential.py
```

The planner solves all seven shipped levels optimally, and the engine confirms each
plan wins. It beats the published human median on every level:

| Level | Pebby optimal | Human baseline | Search states | Peak RAM | Mechanics |
|---|---:|---:|---:|---:|---|
| 1 | **13** | 22 | 4,763 | — | one rotation cycler |
| 2 | **45** | 123 | 7,417 | — | refills, 2 budget per move |
| 3 | **39** | 73 | 13,133 | — | colour cycler, launcher pads |
| 4 | **43** | 84 | 158,891 | 0.1 GiB | shape cycler, 8 launchers |
| 5 | **44** | 96 | 847,459 | 0.3 GiB | first rail-riding cycler |
| 6 | **72** | 192 | 12,040,252 | 4.3 GiB | two goals, all three cyclers on rails |
| 7 | **53** | 186 | 21,746,093 | 6.8 GiB | fog of war, 6 refills |

Levels 6 and 7 cost minutes and gigabytes, so nothing at request time re-derives
them: the values are cached in `pebby/ls20/shipped.py` and regenerated with
`tools/shipped_optima.py`. `Oracle`'s default limit of 600,000 states is sized for
generated levels and **truncates silently on shipped levels 6 and 7** — pass
`shipped.search_limit(index)` there.

Every generated level is completable. `generate_level` returns a level only after the
exact planner finds a solution and that solution is replayed in the real game, which
must report the level completed. A 6,600-level bank generated with a 100% success rate
had a median optimum of 25 actions.

## Proof boundaries

These guarantees do not imply controller success. The generator's stored route is an
oracle solution, not a learned policy output. Collection checks four actual engine
branches at retained states; it does not exhaustively check every reachable engine
state. Independent rule-model checks and bank audits provide additional evidence, but
they do not turn a small calibration pilot into a production dataset. See the
[seven-tier contract](seven-reference-difficulties.md),
[world-data coverage notes](world-failure-coverage.md), and the
[agent experiment record](agent-and-training.md) for the exact bounds.
