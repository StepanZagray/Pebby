# Development notes

## Repository layout

```
third_party/ls20/     verbatim upstream game, its licence, and provenance
pebby/ls20/           environment, names, layout, rails, planner, generator, and banks
pebby/agent/          models, data collection, training, and evaluation
serve.py/inference.py stateless HTTP runtime used by the viewer and HostAI
ui/                   generated-level viewer and its overlays
tools/                differential checks, audits, collection, and training helpers
tests/                engine, data, server, agent, and UI checks
docs/                 durable project contracts and workflows
```

`pebby/ls20/names.py` is a useful first read: upstream uses randomized identifiers,
and this module provides Pebby's translation table. The vendored game is not
modified.

## Checks

Run these commands from the repository root:

```bash
uv run python -m unittest discover -s tests
PYTHONPATH=. uv run python tools/differential.py
node --check ui/app.js ui/board.js
```

Use the module `--help` output as the authoritative reference for other commands;
training and generation flags are intentionally not duplicated here because they
change with the implementation.

## Licence and provenance

`third_party/ls20/ls20.py` is the unmodified upstream game source and carries its
own licence header. The accompanying notices are retained in
`third_party/ls20/LICENSE`. Read [`third_party/ls20/PROVENANCE.md`](../third_party/ls20/PROVENANCE.md)
before reusing the vendored game. For commercial use, confirm the licence with the
copyright holder rather than relying on repository notes.
