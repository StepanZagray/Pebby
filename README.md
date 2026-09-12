# Pebby

An agent that learns to complete **LS20**, one of the ARC-AGI-3 games, by training
on procedurally generated levels that are proved completable.

Pebby vendors the official game verbatim and plays it, so generated levels and the
seven shipped levels run the same code. Retained training levels must pass a complete
oracle search and a winning replay in the game's own implementation, in the context
used for training. An independent rule model provides additional verification for
generated-bank proofs.

The current research target is a learned controller that completes all seven official
levels. The best official checkpoint completes **1/7** official levels (level 1),
15/100 compact generated monitor levels, and 0/20 harder generated levels. The
seven-level target has not been achieved. Detailed architecture, experiment results,
known limitations, and reproduction commands live in
[Agent and training](docs/agent-and-training.md).

## Start here

Install the locked environment:

```bash
uv sync --locked
```

Open the generated-level viewer:

```bash
uv run serve.py --port 11435
# open http://127.0.0.1:11435/ui/index.html
```

The viewer lets you inspect generated levels, their proof metadata, overlays, and
stored winning routes. See [Viewer and HostAI integration](docs/viewer.md) for the
standalone and HostAI-backed modes.

For new banks, use the [seven-tier generation workflow](docs/seven-reference-difficulties.md#entry-points-and-durable-generation).
Historical five-tier banks and their explicit legacy commands are documented in
[generation-quality-v2.md](docs/generation-quality-v2.md). Existing banks, arrays,
and checkpoints keep their original difficulty meaning until rebuilt and retrained.

## Documentation

The [documentation index](docs/README.md) groups the project by task:

- [Game and proof](docs/game-and-proof.md) — LS20 rules, exact planning, and what is
  verified versus cached or not established.
- [Agent and training](docs/agent-and-training.md) — policy architectures, datasets,
  world-model experiments, training protocols, metrics, and research status.
- [Viewer and HostAI integration](docs/viewer.md) — the UI, server contract, and
  HostAI SDK setup.
- [Seven reference difficulties](docs/seven-reference-difficulties.md) — the current
  public generation contract and durable generation workflow.
- [Generated-level and collection fixes](docs/generation-quality-v2.md) — the
  historical five-tier mechanism supplement and its collection caveats.
- [World-data failure and route coverage](docs/world-failure-coverage.md) — failure
  supervision, on-policy rows, and sampling semantics.
- [UI and server validation](docs/validation.md) — automated/browser coverage,
  isolated testing, screenshots, and known gaps.
- [Development notes](docs/development.md) — repository layout, checks, provenance,
  and licensing notes.

## Repository shape

```
third_party/ls20/     verbatim upstream game, licence, and provenance
pebby/ls20/           environment, layout, rails, planner, generator, and banks
pebby/agent/          models, data collection, training, and evaluation
serve.py/inference.py stateless HTTP runtime used by the viewer and HostAI
ui/                   generated-level viewer
tools/                differential checks, audits, collection, and training helpers
tests/                engine, data, server, agent, and UI checks
docs/                 detailed contracts, experiments, and validation records
```

## Checks

```bash
uv run python -m unittest discover -s tests
PYTHONPATH=. uv run python tools/differential.py
node --check ui/app.js ui/board.js
```

The full UI validation procedure and its isolation requirements are in
[docs/validation.md](docs/validation.md). The complete test suite is under active
development; check that document for the latest measured result and known failures.

## Scope

This is not an ARC-AGI-3 benchmark result. Pebby is trained on LS20's rules rather
than discovering them: there is no unknown-rule discovery, online adaptation, or
human-relative scoring here. The planner has full knowledge of the rules and is used
as a teacher, not as the agent. Nothing here has been run against the official
scorecard.
