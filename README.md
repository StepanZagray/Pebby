# Pebby

Pebby is a learned controller for **LS20**, one of the ARC-AGI-3 games. It trains
on procedurally generated levels, then plays the vendored game implementation in
the same environment used for data collection.

**The seven-level target is still unmet.** The retained model completes only the
first shipped level. See [current model status](MODEL_STATUS.md) for the selected
checkpoint, measured limits, and the command to run it.

Generated training levels are accepted only after an exact planner finds a route
and the real game replays that route successfully. The planner is a teacher and
is not used as the learned controller's inference policy.

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

```bash
uv run python -m unittest discover -s tests
PYTHONPATH=. uv run python tools/differential.py
node --check ui/app.js ui/board.js
```

## Scope

Pebby is not an ARC-AGI-3 benchmark result. It is trained on LS20's known rules;
it does not perform unknown-rule discovery or online adaptation. Nothing here
should be interpreted as an official scorecard result.
