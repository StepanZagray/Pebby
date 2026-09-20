# Pebby

Pebby is a learned controller for **LS20**, one of the ARC-AGI-3 games. It trains
on procedurally generated levels, then plays the vendored game implementation in
the same environment used for data collection.

**The seven-level target is still unmet.** The retained model completes only the
first shipped level. See [current model status](MODEL_STATUS.md) for the selected
checkpoint, measured limits, and the command to run it.

## What Pebby is

Pebby is a **planner working with an ideal world model**. The world model is
learned rigorously on one fixed set of game rules (LS20's), so that once trained
it predicts the consequences of any action in any state as faithfully as the game
itself. The planner sits on top of that model and is built to be independent of
the particular goal: it takes a goal as a test ("is this state a success?") and
finds actions that reach a state passing the test. It must not be tied to one
kind of objective such as distance to a pad, because ARC-AGI-3 games pose goals
of many kinds. Solving LS20 is the first proof that the pairing works; the
architecture is judged by whether the same planner would plan for a different
goal on the same rules without being rewritten.

The exact rule-based planner in `pebby/ls20/plan.py` is a teacher and a test
instrument. Generated training levels are accepted only after it finds a route
and the real game replays that route successfully. It is not the learned
controller's inference policy. An ablation that runs the goal-agnostic search
over the real game (`tools/ablate_engine_search.py`) completes all seven levels
optimally when given an exact progress estimate, which fixes the target the
learned components have to meet; see [current model status](MODEL_STATUS.md).

## Start here

Install the locked environment:

```bash
uv sync --locked
```

Start the local viewer:

```bash
uv run serve.py --port 11435
# open http://127.0.0.1:11435/ui/index.html
```

To train, evaluate, or serve a checkpoint, see
[Training and model operation](docs/agent-and-training.md). For the viewer and
HostAI runtime contract, see [Viewer and HostAI integration](docs/viewer.md).

## Documentation

The [documentation index](docs/README.md) contains only durable project
documentation:

- [Game and proof](docs/game-and-proof.md) — LS20 mechanics, planning, and proof
  boundaries.
- [Training and model operation](docs/agent-and-training.md) — the stable workflow
  for data, training, evaluation, and serving.
- [Viewer and HostAI integration](docs/viewer.md) — local viewer and runtime use.
- [Development notes](docs/development.md) — repository layout, checks, and
  provenance.

Experiment history stays in local artifacts. The separate
[model status](MODEL_STATUS.md) records which checkpoint the current claims refer to.

## Repository shape

```
third_party/ls20/     verbatim upstream game, licence, and provenance
pebby/ls20/           environment, planner, generator, and level banks
pebby/agent/          models, data collection, training, and evaluation
serve.py/inference.py stateless HTTP runtime used by the viewer and HostAI
ui/                   generated-level viewer
tools/                differential checks, audits, collection, and training helpers
tests/                engine, data, server, agent, and UI checks
docs/                 durable project contracts and workflows
```

## Checks

Pytest runs both the unittest classes and the plain pytest functions. The Python
suite runs on CPU; some integration tests require local data or checkpoints.

```bash
CUDA_VISIBLE_DEVICES='' uv run --group dev python -m pytest tests -q
PYTHONPATH=. uv run python tools/differential.py
node --check ui/app.js ui/board.js
```

## Scope

Pebby is not an ARC-AGI-3 benchmark result. It is trained on LS20's known rules;
it does not perform unknown-rule discovery or online adaptation. Nothing here
should be interpreted as an official scorecard result.
