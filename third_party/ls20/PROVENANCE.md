# Vendored ARC-AGI-3 game: LS20

`ls20.py` and `metadata.json` are **verbatim, unmodified** copies of the official
ARC-AGI-3 game `ls20`, published by the ARC Prize Foundation.

| | |
|---|---|
| Game id | `ls20-9607627b` |
| `ls20.py` SHA-256 | `298c810da2850d557c95d92a2cbd846df29a45d7134e20888617bedf5dafcd92` |
| Size | 105,874 bytes, 2,060 lines |
| Retrieved | 2026-09-11 |
| Upstream | `https://three.arcprize.org/api/games/ls20-9607627b/source` |

## Licence

`ls20.py` carries an MIT licence header in the file itself (lines 1-21),
`Copyright (c) 2026 ARC Prize Foundation`, byte-identical to canonical MIT.
That header is reproduced standalone in `LICENSE`. The game imports `arcengine`,
also MIT and also copyright ARC Prize Foundation; its licence is in
`LICENSE.arcengine`. Both notices are retained here to satisfy the MIT
attribution condition.

Note for anyone reusing this: the licence grant lives **only in the delivered
file's own header**. The `arcprize/ARC-AGI` toolkit repository, the
`arcprize/docs` site and the PyPI packages say nothing about the licensing of
downloaded game content, and `https://arcprize.org/terms` is generic proprietary
boilerplate with no open-source carve-out. The in-file MIT grant from the
copyright holder is what permits this copy. For commercial use, confirm with
`team@arcprize.org` rather than relying on this note.

## Re-fetching

No credentials are required; an anonymous key is issued on request.

```bash
KEY=$(curl -sS https://three.arcprize.org/api/games/anonkey \
      | python3 -c "import sys,json;print(json.load(sys.stdin)['api_key'])")
curl -sS -H "X-Api-Key: $KEY" \
     https://three.arcprize.org/api/games/ls20-9607627b/source -o ls20.py
```

## How Pebby uses it

This copy is the **authority on the rules**. Pebby does not re-implement LS20:
`pebby/ls20/env.py` loads this exact module and drives the real `Ls20` game
class, so generated levels and the shipped levels run under identical logic.
Identifiers in the upstream file are obfuscated; `pebby/ls20/names.py` maps them
to readable names without altering the file.
