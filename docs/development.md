# Development notes

## Repository layout

```
third_party/ls20/     the verbatim upstream game, its licence, and provenance
pebby/ls20/            env, names, layout, rails, planner, generator, and bank code
pebby/agent/           models, data collection, training, and evaluation
serve.py/inference.py  stateless HTTP runtime used by the viewer and HostAI
ui/                    generated-level viewer and its overlays/scrubber
tools/                 differential checks, audits, collection, and training helpers
tests/                 engine, data, server, agent, and UI checks
docs/                  contracts, experiment records, and validation evidence
```

`pebby/ls20/names.py` is worth reading first: upstream ships with randomised
identifiers, and that module is the translation table. The vendored file is never
modified.

## Checks

Run the broad suite and the engine-equivalence check from the repository root:

```bash
uv run python -m unittest discover -s tests
PYTHONPATH=. uv run python tools/differential.py
node --check ui/app.js ui/board.js
```

The measured browser procedure is intentionally separate because it needs an isolated
headless compositor and Chromium. See [validation.md](validation.md) before running it;
never attach UI tests to the live desktop or an existing browser.

## Licence and provenance

`third_party/ls20/ls20.py` is the official `ls20-9607627b`, retrieved 2026-09-11,
SHA-256 `298c810d…`, **unmodified**. It carries an MIT header in the file itself,
© 2026 ARC Prize Foundation, reproduced in `third_party/ls20/LICENSE`. `arcengine` is
MIT from the same holder. Both notices are retained as MIT requires.

Read `third_party/ls20/PROVENANCE.md` before reusing this: the licence grant exists
**only** in the delivered file's header. The toolkit repositories, docs site, and PyPI
packages say nothing about the licensing of downloaded game content, and
`arcprize.org/terms` is generic proprietary boilerplate with no open-source carve-out
(and names a different legal entity than the copyright holder). For commercial use,
confirm with the ARC Prize Foundation rather than relying on this note.
