# Viewer and HostAI integration

`serve.py` plus `ui/` is a viewer for the levels the generator produces. It answers
“what did the generator actually build?” by eye, on the real frame the engine renders,
rather than by reading a JSON spec.

## Standalone viewer

```bash
uv run serve.py --port 11435
# open http://127.0.0.1:11435/ui/index.html
```

One panel, “Level”, chooses what to play. Its three sources — the generated bank, a
seed generated on the spot, and the shipped LS20 levels — differ only in where the level
comes from, so a segmented control shows one of them at a time, and loading a level from
any source switches the panel to that source. The seven difficulties are labelled Tier 1
to Tier 7 there, each calibrated to the shipped LS20 level of the same number. The die
beside the seed box only fills the box: Generate and the seed arrows are the only controls
that fetch, so a stray click never replaces the level you are reading. The page shows:

- the 64×64 frame, nearest-neighbour scaled, exactly as an ARC-AGI-3 agent receives it;
- overlays drawn from the level's own spec in its logical 12×12 lattice: the lattice,
  features, and the stored solution route;
- anatomy such as tier, seed, step budget and cost, fog, wall count, and the
  generator's proof metadata — optimal actions, budget slack, reachable states, and
  whether the search truncated;
- a solution scrubber. Every generated level carries a solution the generator proved
  optimal and replayed to a win, so the scrubber is a proof you can watch;
- hand play through arrow keys or the d-pad, with exact undo and reset.

“Play solution” replays every prefix once and caches it, which is also what draws the
route. The server is stateless: the page owns the level and action list, and each board
is the server replaying that whole list from the start. That makes undo exact and lets
the scrubber jump to any step. Shipped levels load too; they carry no stored solution
here, and the page says so rather than implying one.

Finishing a level puts a card over the board. On a win its first button offers the next
level in the source this one came from — the next row of the bank listing as it is
filtered right now, turning the page at a boundary, or seed + 1 at the same tier — and
reads “Play again” when nothing follows. The shipped campaign advances inside its own
replay, finishing one level straight into the next without a win, so a win there is the
end of it and there is nothing after. Enter takes the card's offer; Escape dismisses it
and leaves the coloured frame behind.

## HostAI SDK integration

Every request the page makes goes through HostAI. Embedded in HostAI the page calls the
injected `window.hostai` bridge; standalone it posts the same `{model, input}` envelope
to `/hostai/infer` itself. There is no second transport, so what you see standalone is
what HostAI shows.

HostAI is an installed Python dependency, not copied server glue. `uv sync --locked`
installs the versioned SDK wheel from `vendor/`; no separate HostAI checkout path is
needed. `pebby/hostai_provider.py` declares the model, custom UI, operation
instructions, input/output schemas, and examples. The SDK validates those declarations,
serves the manifest/assets, and validates both inference routes. `serve.py` retains
only Pebby's health route, standalone bridge placeholder, and process startup.

Start the engine with `uv run serve.py` and start HostAI separately, configured with
`HOSTAI_RUNTIME_URLS=http://127.0.0.1:11435`. A nonvisual client can use HostAI without
loading the UI (commands run from the HostAI checkout):

```sh
pnpm cli describe pebby:latest --json
pnpm cli infer pebby:latest '{"op":"boot"}'
pnpm cli infer pebby:latest '{"op":"info"}'
pnpm cli infer pebby:latest '{"op":"play","level":{"shipped":0},"actions":[4]}'
```

The contract describes all ten operations and makes caller-owned level/action history
explicit. `op=boot` is a read-only composite: it returns `info`, the bank catalogue, the
first page of the first bank's training split and that page's first level in one
response, so the viewer paints its opening screen without a chain of dependent requests.
It adds no server-side state and every part it returns stays reachable on its own. Generated-level geometry remains validated by Pebby. The SDK schemas
describe operation envelopes and common result fields, not every internal level field.
Missing weights return explicit unavailable agent results; successful environment
playback is not evidence of a trained policy.

`/predict` remains a raw-input alias of the SDK's inference path, `/health` remains
available, and the standalone UI continues to work. Invalid requests still return
`{"error":"..."}` with 400, though schema validation messages differ from the old
hand-written server. Provider failures return generic 500. Custom UI declarations are
validated at server construction, so missing bridge scripts or unsafe/missing assets
fail at startup. Single-response inference is bounded to 256 KiB and UI assets to 8 MiB,
matching HostAI's wire limits.

For SDK development, use `uv run --with-editable /path/to/hostai/python serve.py`
explicitly. Normal runs use the locked wheel. Future SDK releases should update the
versioned wheel, dependency, and lockfile together; the SDK is not yet published to
PyPI. This relative wheel source follows
[uv's documented path dependency workflow](https://docs.astral.sh/uv/concepts/projects/dependencies/#path).

## Validation

The browser checks, isolated-compositor procedure, screenshots, and known gaps are in
[UI and server validation](validation.md). The viewer's stored route is a generator
proof replay, not a learned controller evaluation.
