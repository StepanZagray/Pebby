# Astra BP35 continuous-chamber closure review

Review began 2026-09-19 at 02:45:29 CEST; completed 02:51:07 CEST. Local machine timezone verified with `date`. Source reviewed read-only in `/home/stepan/Projects/code/Pebby-full-bp35`; root retains final acceptance.

## Recommendation

**Pass this focused structural closure.** No native counterexample or blocking source defect was found in the bounded review. The previous per-shelf and underside transfer problems are replaced by genuine continuous chamber separation, not merely historical-route blacklisting. This is not independent whole-family acceptance, a shortest-path proof, or population-equivalence evidence.

## Structural reasoning beyond the selected witness

`generate.py:405–444` installs immutable partitions at x=3 and x=7 across the full 33-row height. The left partition permits a protected bottom transfer at y=6 above its initial open bridge at y=5 and immutable spike at y=4, plus the ceiling transfer y=31. The final partition permits only the ceiling transfer y=31. In tier 7, its cell (7,30) is an immutable up-spike instead of an ordinary wall; it is not a traversable opening. Central cap spikes occupy x=4..6, y=30 (also x=7 for tier 7). Initial rows and native entity names agree.

These relations constrain all ordinary native movement, irrespective of the chosen stored click list:

1. The player starts in the middle chamber; the goal is in the final chamber. Lateral movement advances one column and cannot pass wall/spike cells. Therefore entry to the final chamber requires the y=31 ceiling aperture. Top and bottom border rows close any outside detour.
2. The middle cap prevents a safe direct ascent to that ceiling. Visible gravity switches can be clicked remotely, but they do not move or erase the cap. A safe ceiling approach comes from the separate x=2 ascent chamber.
3. The left chamber's lower entrance is the y=6 transfer above a spike-protected open bridge. Entering this under downward gravity without closing the bridge loses. Available middle supports do not provide a reversible-gravity state at y=6 that bypasses this entrance: the next support is at least y=8, so a premature reversal stops at y>=7 or encounters the cap, behind the continuous partition. This closes the prior under-shelf route.
4. Ascending to y=31 requires upward gravity, while reaching the final lower gem requires downward gravity again. Remote toggles can consume/reorder resources but cannot remove these roles. Bridge construction and both gravity directions are therefore grounded in the chamber geometry, not just event counts on the teacher path.
5. Native click dispatch (`third_party/arc3_games/bp35.py:4289–4399`) changes bridges, destructibles, growers and gravity switches; walls and spikes are not mutable click targets. Native UNDO restores previous world/entity/gravity state rather than preserving a newer mutation with an older player position. Thus neither adds an aperture through the immutable partitions.

The new `_gravity_chamber_certificate` (`generate.py:1138`) reconstructs these expected rows and checks native entity identities. Full validation independently reconstructs the deterministic grammar (`:1845`) and recomputes the chamber certificate (`:1956`). This supports a structural role argument. The separate restricted adjacent-action search and fixed shortcut templates retain their narrower meaning; neither is promoted to a complete solver.

I did not formally exhaust every action/UNDO history or prove the numeric 28/34 floors globally. The check establishes the specific required traversal/mechanic roles and closes the demonstrated shortcut classes. The finite grammar and prior whole-family caveats remain.

## Independent native execution

Fresh accepted specs `generate(5006,6,split="train")` and `generate(5007,7,split="train")` both used attempt 0 and returned `[]` from full validation. Stored routes were 28 and 36 actions. Generation plus fresh full validation took approximately 5.89 and 9.25 seconds.

Every exact historical 6/13/26/22/28-action attack was issued publicly on both specs, stopping at native terminal states. None won:

| Historical route length | Tier 6 | Tier 7 |
|---:|---|---|
| 6 | NOT_FINISHED | GAME_OVER |
| 13 | GAME_OVER | NOT_FINISHED |
| 26 | NOT_FINISHED | GAME_OVER |
| 22 | NOT_FINISHED, x=6,y=7, gravity up | GAME_OVER |
| 28 | GAME_OVER | NOT_FINISHED, x=6,y=8, gravity up |

Direct public replay of both accepted routes verified the following native roles:

| Tier | Bottom x=3 crossing | Ceiling x=3 / x=7 crossings | Bridge closures | Gravity reversals |
|---:|---|---|---|---|
| 6 | action 11 at y=6 | actions 14 / 18 at y=31 | (3,5), (2,5) | actions 13 / 20 |
| 7 | action 17 at y=6 | actions 20 / 24 at y=31 | (5,5), (4,5), (3,5), (2,5) | actions 19 / 26 |

Both replays reached native WIN. Each immutable wall/spike in the initial layout was checked for its exact native name throughout the nonterminal route states.

At every bridge closure and gravity reversal, an independent UNDO followed by re-execution restored the exact prior native player/entity/bridge/gravity key and preserved all immutable cells: four checked events in tier 6, six in tier 7. Added UNDO actions were diagnostic interventions, not advertised as the original route length or a minimal solution.

Visible wall cells (3,28), (7,28), and cap spike (5,30) were clicked and undone from fresh states. All remained immutable, with exact state restoration.

Remote top-bank reversal at spawn is possible but loses in both tiers at (5,29), gravity up. A remote bottom-bank reversal in the first shaft also loses under the cap. Shifted bottom reversals corresponding to the prior underside approach survive only below a shelf, then repeated RIGHT stays behind x=7: tier 6 remains (6,7), tier 7 (6,8), both gravity up. These are actual public camera-relative clicks, not injected player states.

Mutation checks removed one cell each at (3,0), (7,32), (3,12), (7,12), and (5,30), separately for both tiers. Border holes were rejected by native-level construction's solid-border requirement; interior partition/cap holes failed the recomputed chamber certificate. No source or persistent test fixture was changed.

## Minor wording issue and scope caveats

`generate.py:541–543` says top switches cannot be reached before a legitimate first reversal. Taken as click accessibility, this is inaccurate: a top-bank switch can be clicked remotely at spawn, as independently demonstrated here. The correct statement is that this premature use cannot safely bypass the cap. This is a nonblocking comment issue; the structural defense does not rely on camera inaccessibility.

Author-reported exhaustive enumeration of the 18 declared support layouts was not duplicated. This review freshly executed two representative accepted native contexts, read the complete partition logic, and exercised decisive adversarial/public-action probes. It does not freshly cover tiers 1–5/8–9, whole-nine collector runs, rendering, official-copy checks or all malformed schemas. Prior reports remain intact.

The official tier-6/7 within-level gravity counts are 3/11, whereas the generated grammar uses two required reversals. Official arbitrary live-prefix recovery remains unsupported; generated UNDO recovery is separate. Previous documentation limitations and root's final visual/pipeline checks remain outside this closure.

## Frozen hashes, resources and cleanup

- `pebby/games/bp35/generate.py`: `f4f49c20173cf0b4d92dbeddebc368a1e0ebcf9a2ddde8d316d158e138bb26f7`
- `tests/games/test_bp35.py`: `d1ca059758738f64cfc62ab7d61badd44b9419940e35c80347b1030f57308fb1`

Initial hashes, end-of-compute hashes, and final report-time recomputation all match.

One substantive native compute process ran with primary `.venv/bin/python`, worker cwd and PYTHONPATH, `PYTHONDONTWRITEBYTECODE=1`, BLAS/OpenMP threads 1, a 115-second timeout, and 1,900 MiB address-space ceiling. Peak actual RSS was 100,472 KiB. PID 1214440 / tool session 90853 exited normally and no longer exists.

An initial launch exited immediately before any native generation because the package exports `generate` as a function; importing the module explicitly corrected the harness. That preliminary PID 1213159 also no longer exists. There was no second native batch, repeated all-18 run, broad search, official search, training, held-out evaluation, GPU work, UI/browser activity, subagent or commit.

No source/tests/historical notes were edited. Only this new review report was written; no retained process is intended.
