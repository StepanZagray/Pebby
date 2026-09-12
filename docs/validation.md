# LS20 UI and server validation

What was checked, how, and — as important — what was not.

Everything below describes Pebby's HTTP server (`serve.py`, `inference.py`,
`predict.py`) and its browser UI (`ui/`). It does not cover the LS20 wrapper,
the generator, the planner or the agent, which have their own tests in
`tests/test_ls20.py` and `tests/test_agent.py`.

## Automated checks

`uv run python -m unittest tests.test_serve tests.test_inference`:
**40 tests, all passing, 7 s.**

- `tests/test_serve.py` (21 tests) starts a real server on port 0 in a thread
  and drives it over a socket: `/health`, the HostAI manifest, every operation
  on both `/predict` and `/hostai/infer` compared against the engine's own
  answer, `Cache-Control: no-store` everywhere, 415 for four wrong content
  types, 413 for empty and oversize bodies, 400 for malformed JSON, a wrong
  model name, a missing `input`, chunked bodies and every engine `ValueError`,
  404 for unknown paths, the action-history cap at both ends against a real
  socket, the static files byte-for-byte, the synthesised `hostai-bridge.js`,
  the no-inline-script rule, and a traversal matrix against a temporary UI
  directory (dot segments, a symlink escaping the root, a hidden file, a
  non-whitelisted extension, directory paths).
- `tests/test_inference.py` (19 tests) covers the engine directly: the palette
  against the 16 ARC-AGI-3 colours written out literally, the whole `info`
  contract, statelessness under interleaved levels, 64x64 frames of indices
  0..15, the shipped level-1 solution, a generated level through
  `json.dumps`/`loads` and back through the validator unchanged, undo as a
  prefix replay, the action cap, 98 malformed requests each asserted against
  its exact error string, the oracle's four distinct answers, that the two
  expensive shipped levels are answered from cache in under a second instead of
  being searched, that the planner's silently-truncating default search limit is
  never used, and the agent degrading with no checkpoint, a missing path and a
  corrupt one. One test runs a fresh interpreter to prove `import inference`
  does not pull in torch, and one runs `predict.py` from another directory.

`uv run python -m unittest discover -s tests`: **659 tests, 1 skipped, 1
failure** (measured 2026-09-12 12:05). The failure is
`test_structured_sequence_metrics.test_actual_readout_is_reported_separately_from_dynamics`,
a wording assertion against a notes string in `tools/structured_sequence_metrics.py`.
It is unrelated to the server or the UI and predates this UI work. The other
suites are under active development, so re-measure rather than quoting this.

`node --check` passes on `ui/app.js`, `ui/board.js` and `tests/ui_viewer.cjs`.

An independent read-only review (GPT-5.6 Sol, high reasoning) of `inference.py`,
`serve.py`, `predict.py` and `ui/app.js` found five defects, all reproduced and
fixed: an unhashable cycler `kind` raised `TypeError` instead of returning 400;
launcher deltas accepted `[true, 0]` and `[1.0, 0]` because Python equality;
`cache_key` was not canonical across permuted or duplicated wall lists; the UI's
run/stop used a string token with an ABA race; and stale `play` and `agent`
responses could commit because the guards compared revision or length rather
than action-history identity.

## Browser checks

### Isolation

Never on the live desktop. A separate **Sway 1.12 / wlroots** compositor with the
`headless` backend and the `pixman` software renderer, inside **bubblewrap
0.11.2**, started from `env -i` with a private mode-0700 `XDG_RUNTIME_DIR` at a
short path. Its executable was a mode-0700 copy of `/usr/bin/sway` with the
capability xattrs not copied, because Arch's binary carries `cap_sys_nice=ep`,
which would make `/proc/<pid>/environ` and descriptor inspection unavailable.
The system binary was not modified.

The full proof is in [`artifacts/ui-isolation-proof.txt`](../artifacts/ui-isolation-proof.txt),
captured from the running compositor before any client started. It records:

- the exact host PIDs, the compositor's `exe` link, `getcap` on both the copy
  and the system binary, and `CapPrm`/`CapEff`/`CapBnd` all zero with
  `NoNewPrivs: 1`;
- the compositor's complete environment, and a count of zero for
  `WAYLAND_DISPLAY`, `WAYLAND_SOCKET`, `DISPLAY`, `HYPRLAND_INSTANCE_SIGNATURE`,
  `DBUS_SESSION_BUS_ADDRESS`, `XDG_SESSION_ID`, `XDG_SEAT`, `XDG_VTNR`,
  `XDG_ACTIVATION_TOKEN`, `WLR_DRM_DEVICES` and `LIBSEAT_BACKEND`;
- every open descriptor, with zero matching `/dev/dri`, `/dev/input`,
  `/dev/tty`, `/dev/fb`, seatd, logind, `/run/user` or dbus, and both unix
  sockets it holds resolved to paths inside the private runtime directory;
- what the sandbox shows it: a minimal `/dev` with no `dri` and no `input`, and
  an empty `/run` tmpfs, so seatd, logind and the session bus are unreachable
  whatever the compositor tries;
- the startup lines proving the backend and renderer (`headless`, `pixman`, no
  DRM backend, `xwayland disable`), plus `swaymsg -s <private socket>` reporting
  one output, `HEADLESS-1` at 1280x800@60 Hz, and no input devices at all.

Chromium 151 was then launched with `env -i`, `--ozone-platform=wayland`, a
fresh profile inside the private runtime directory and CDP on 127.0.0.1:9333;
its own environment is recorded in the same file. Screenshots come from the page
via CDP, and one compositor-level capture from `grim -o HEADLESS-1` on the
private socket.

Teardown killed only the five tracked PIDs and then verified each was gone from
`/proc`, that no process anywhere still had an `exe` under the test directory,
that no test socket remained in `/proc/net/unix`, that nothing was listening on
the test ports, and that both temporary directories were removed.

### What the browser tests check

`tests/ui_viewer.cjs` — the whole viewer, in one pass, **passing** in the
isolated Chromium described above (re-measured 2026-09-12 17:55 against the
single Level panel and the next-level card, through
`tools/verify_bank_viewer.py --test ui_viewer.cjs`; report, logs and screenshots
in `artifacts/ui-level-panel/viewer/`):

- startup reports its transport and model, the tier menu reads Tier 1 to Tier 7,
  and asking for generated seed 7 at tier 3 lands there with an empty
  history and a 64x64 canvas;
- the three sources share one panel and only one of them is on show: choosing
  Shipped puts the Generate pane away, and choosing Generate brings it back;
- the anatomy panel is checked field by field against the spec the server
  returns for the same seed — seed, tier, optimal actions, wall count,
  step budget, fog — plus the goal, cycler and launcher counts, so the panel
  cannot drift from the level it claims to describe;
- the overlays put ink on the overlay canvas and take it away again: features
  alone draw, unchecking clears to zero lit pixels, the lattice draws on its
  own, and features on top of the lattice strictly increase the ink;
- the seed box states an intent and nothing more: the die beside it fills in a
  random seed without loading anything, a changed tier does not load
  anything either, and both raise the "inputs changed" hint until "Generate"
  fetches exactly what is in the boxes; the die and "Generate" together are the
  random-level path, and the level that arrives is the seed the die drew;
- a request in flight is reported rather than greyed out: with the response held
  back deliberately, the Level panel carries a "generating" marker naming the
  work it was asked for and a spinner on the button that was pressed, its fields
  stay at full opacity and readable, and both retire with the request;
- seed stepping moves to seed 8 and back, and a new level starts empty;
- arrow keys, the d-pad, undo and reset, with undo checked against a direct API
  replay of the same prefix rather than against an assumed inverse move;
- arrow keys inside the seed field still belong to the field;
- **the stored solution finishes the level** — "Jump to end" reaches the
  `Completed` state, which is the generator's completability proof replayed in
  the browser;
- a finished run announces itself over the board: the finish card names the
  state and the action count, its first button offers the level after this one,
  the board frame carries the win colour, the card can be dismissed while the
  frame stays, and loading another level clears both;
- the route overlay replays every prefix, caches them, draws a path, and a
  cached scrub step is checked against the engine replaying that same prefix;
- a shipped level loads and says plainly that it carries no stored solution,
  with the scrubber and its buttons disabled;
- a rejected seed is reported in the panel's one error slot without destroying
  the loaded level, and any working request retires the error;
- the finish card leads on rather than only back: on a won generated level it
  reads "Next level", pressing it loads seed + 1 at the same tier with an empty
  history and retires the card, and Enter on the card does the same;
- the board stays square and fits 1280x800, there is no horizontal overflow at
  390px, and the d-pad stays tappable there;
- the page raises no uncaught errors;
- finally, a second page is loaded with a **stand-in `window.hostai` injected**,
  and the UI must prefer it: the transport reads "HostAI bridge", `ready()` is
  awaited before any inference, the entire opening screen arrives over the bridge
  in a single `boot` call — info, the bank catalogue, the first page of rows and
  that page's first level — and a move still lands. See the limits on this below.

`tests/ui_bank_viewer.cjs` — the bank source, **passing** in the same harness
(measured 2026-09-12 17:57; evidence in `artifacts/ui-level-panel/bank/`). It
plays one accepted row from each of the seven tiers in both splits by its stored
route — 14 levels, each checked against the row the server returns for that id —
walks pagination and the split and tier filters without disturbing the board,
holds a `bank_levels` response back to watch the Level panel report the work,
drives a failed catalogue and an empty filtered page into the panel's one error
slot and recovers from both, proves the finish card walks the listing (a
completed row offers the one after it, and taking the offer moves the panel's
own selection with it), checks 320, 768, 1024 and 1440 px for horizontal
overflow, and asserts that browsing accepted levels never issues a `generate`
request.

`tools/verify_bank_viewer.py` owns the process boundary for both suites and is
the one-command route: it starts the capability-free Sway copy on the headless
pixman backend inside bubblewrap, a disposable loopback server on a port it
reserves itself, and Chromium in its own namespace; it writes the isolation
proof and teardown check into a JSON report, and kills only the PIDs it started.
It never touches a running application server.

```
uv run python tools/verify_bank_viewer.py --test ui_viewer.cjs --evidence artifacts/ui-level-panel/viewer
uv run python tools/verify_bank_viewer.py --test ui_bank_viewer.cjs --evidence artifacts/ui-level-panel/bank
```

Driving a suite by hand instead requires `playwright-core` and explicit `PEBBY_TEST_CDP` and
`PEBBY_TEST_ORIGIN` pointing at a dedicated browser and server.
`PEBBY_PLAYWRIGHT` names an external `playwright-core` install and
`PEBBY_SCREENSHOT_DIR` saves the evidence images. **The temporary
`playwright-core` install is removed during teardown**, so re-running it means
installing it again and pointing at it:

```
mkdir -p /tmp/pebby-uitest && cd /tmp/pebby-uitest && npm init -y && npm install playwright-core
export PEBBY_PLAYWRIGHT=/tmp/pebby-uitest/node_modules/playwright-core
export PEBBY_TEST_CDP=http://127.0.0.1:9333 PEBBY_TEST_ORIGIN=http://127.0.0.1:11477
export PEBBY_SCREENSHOT_DIR="$PWD/artifacts"
node tests/ui_viewer.cjs
```

Prove display isolation before connecting; never point `PEBBY_TEST_CDP` at a
normal desktop browser's debugging endpoint.

### What they do not check

- **The HostAI bridge against a real gateway.** Both of the UI's code paths are
  now exercised — the standalone `fetch("/hostai/infer")` fallback and, through
  an injected stand-in, the `window.hostai.ready()` / `infer()` seam — and the
  server still synthesises the inert `/ui/hostai-bridge.js`. But **no HostAI
  gateway was run.** The stand-in is a few lines of test code that resolves
  `ready()` and forwards `infer()` to the same endpoint; it proves the UI
  prefers and uses the bridge, and says nothing about a real host's streaming
  records, its `onEvent` behaviour, its model-UI proxying or its guest
  boundary. The `--hostai` and `--guest` paths in `predict.py` were likewise not
  exercised against a real gateway. `git show HEAD:test_hostai.py` is the
  integration test that did this against the pre-rewrite Pebby; it needs porting
  to `Engine.dispatch` and the LS20 operations before it can run again.
- **Anything about the agent or the oracle, from this page.** The viewer does
  not call the `agent` or `oracle` operations at all — it has no policy panel and
  no planner panel. Both operations are still served and still covered by
  `tests/test_serve.py` and `tests/test_inference.py`, but nothing in the browser
  exercises them. The "solution" the viewer replays is the one the **generator**
  stored on the level, proved optimal and replayed to a win at generation time;
  it is not a policy's output and says nothing about how any trained controller
  performs. Completion rates come from `pebby/agent/evaluate.py`, not from this
  page.
- **Browsers other than Chromium**, touch input, and screen readers. The UI is
  keyboard- and pointer-tested only, and the accessibility work (live regions, a
  described canvas, `aria-busy`) was written but not verified with assistive
  technology.
- **Concurrency between clients.** The server is stateless and binds to
  loopback, and oracle planning is serialised behind a lock, but no multi-client
  test was run. The viewer does issue up to six concurrent `play` requests of
  its own when it caches a solution's every prefix, which the tests do cover,
  but only from a single page.
- **Large solutions.** Caching a solution costs one `play` per prefix, so the
  work is quadratic in the solution length. The levels exercised here are 14 to
  33 actions and cache in well under a second. Nothing establishes where that
  becomes unpleasant.
- **The cached optima for levels 6 and 7.** Levels 1-5 are cross-checked on
  every test run, because the live planner's distance-to-finish from the start
  must equal `shipped.OPTIMAL_ACTIONS`. Levels 6 and 7 cannot be re-derived here
  for the reasons above, so their 72 and 53 are taken on trust from
  `pebby/ls20/shipped.py` and are surfaced as such.

## Cost measurements

These set the action cap and the UI's non-blocking behaviour, and are worth
re-measuring if either changes.

- Replay: ~0.3 ms per action that lands, ~1.8 ms per action that is blocked,
  because a blocked move still renders a rejection animation. `MAX_ACTIONS` is
  2048 (~3.7 s worst case) rather than 4096 (~7.5 s), which is still far more
  than an honest game can use.
- Planning a shipped level, cold: levels 1-3 under 0.1 s, level 4 ~1.0 s,
  level 5 ~5.8 s. Levels 6 and 7 are **not searched at all**. Exhausting them
  takes 13M and 22M planner states, 4.3 and 6.8 GiB, and about two minutes each
  (`pebby/ls20/shipped.py`), which is not something an unauthenticated loopback
  server should do on request, so the oracle answers them from that module's
  cached, replay-verified optima (72 and 53 actions) in ~0.01 s and says it
  cannot step through them. Whether those two levels have a shorter solution or
  none at all has not been established here, and neither the server nor the UI
  claims either way.
- `inference.ORACLE_STATE_LIMIT` is 1,000,000 and is passed to every search
  explicitly. The planner's own default is 600,000, which **truncates silently**
  — `solvable` becomes False and `solution()` None with no exception — and
  shipped level 5 needs a million. A test asserts the default is never relied on.
  The whole seven-level sweep peaks at 0.34 GiB.
- The UI never waits on a plan. Loading shipped level 5 made the page usable
  again after 110 ms and accepted a hand move at 216 ms while the plan arrived at
  6.4 s. Serialising concurrent identical searches behind one lock cut that
  arrival from 21 s to 6.4 s.
- `oracle_levels()` in `info` is `shipped.CHEAP_LEVELS`: the levels whose search
  fits inside a request. Since rail-riding cycler support landed the planner
  understands all seven, so "would it refuse this level" no longer separates
  anything — cost does. It is **not** a claim about solvability either way.

## Screenshots

All captured in the isolated Chromium described above, by `tests/ui_viewer.cjs`
when `PEBBY_SCREENSHOT_DIR` is set, and re-captured from the 2026-09-12 17:55
run of the single Level panel.

- [Desktop](../artifacts/viewer-desktop.png) — generated seed 11 at tier 3
  with the feature overlay on: the start ring, the goal diamond, cyclers by
  silhouette, and the launcher arrow pointing the way it flings.
- [Route](../artifacts/viewer-route.png) — seed 7 scrubbed to action 15 of 31.
  The walked part of the solution is drawn bright, the rest dim.
- [Solved](../artifacts/viewer-solved.png) — the same level after "Jump to end":
  31 actions, `Completed`, with the finish card offering the next level.
- [Dark theme](../artifacts/viewer-dark.png).
- [Mobile](../artifacts/viewer-mobile.png) — 390x844.
