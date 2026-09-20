# SU15 enriched-sequence closure implementation

## Outcome and scope

Both remaining closure findings are corrected in the SU15-owned package and tests. The contract remains `pending_audit`; root owns acceptance. Work ran from 2026-09-19 01:12 to 01:18 CEST.

Only these files were changed in this closure pass:

- `pebby/games/su15/generate.py`
- `tests/games/test_su15.py`

No shared code, other game family, canonical status, native source, grammar, RNG, geometry/gameplay identity, planner/work meter, legend, composition, or rendering code changed. No commit was made.

## P1: complete enriched whole-game provenance and replay

Sequence admission now has three explicit modes:

1. Standalone tier specs have no sequence fields and no `proof.full_game_replay`. Nine independently generated standalone specs remain accepted by `build_game`.
2. `explicit-smoke-subset` rows retain exactly the reduced fields `parent_game_seed`, `game_position`, `child_seed`, and `sequence_kind`, without a full-game hash or replay claim. Child seeds are still derived from the smoke ordinal and difficulty.
3. `full-official-context` rows require all five sequence fields plus a non-null `proof.full_game_replay`. Partial or mixed enrichment is rejected.

For each full row, validation now requires:

- `game_position == difficulty - 1`, alongside the existing exact context mirror;
- `child_seed == seed == _child_seed(parent_game_seed, game_position, difficulty)`;
- a syntactically valid full-game digest;
- the exact tier replay fact `{context_index: difficulty - 1, levels_completed: difficulty, state: NOT_FINISHED}`, except tier 9 must be `WIN`.

At the nine-row `build_game` boundary, all rows must be wholly standalone or wholly full-enriched. Full rows must share one parent seed, preserve ordered positions 0..8, retain the exact parent/ordinal/difficulty-derived child seeds, and share the SHA-256 recomputed from the ordered, freshly recomputed gameplay identities. The builder now retains the result of the real native sequential replay and compares it exactly with all nine supplied replay rows rather than discarding it.

Focused seed-889 regressions reject each reviewed mutation: row-0 position 8, an all-zero sequence digest, parent 890, row-0 `WIN`, null replay, tier-9 `NOT_FINISHED`, missing digest, and missing replay. A mixed-parent case whose modified row is internally self-consistent is rejected at the whole-game common-parent boundary. A monkeypatched sequential replay mismatch confirms that the builder checks, rather than discards, native replay facts.

Positive coverage retains all required modes: the valid enriched seed-889 train game builds all nine contexts; nine separately generated test-split specs build as a standalone game; and a validation-split explicit smoke subset validates with its deliberately reduced metadata.

## P2: malformed nested values fail before dependent operations

Invalid fruit tiers now stop processing that object before sprite-width arithmetic. The exact adjacent values `'2'`, `None`, `[]`, and `{}` all return finite string diagnostics without reaching native construction.

`sequence_kind` is type-checked before supported-mode comparison uses a tuple rather than hash-based set membership. The exact `[]` and `{}` mutations return finite diagnostics rather than raising `TypeError`. Existing malformed `fruits=[None]`, `enemies=[None]`, `targets=[None]`, `requirements=[None]`, string solution length, oversized fruit list, and malformed proof work values remain covered.

## Verification

All subprocesses used the primary venv, worker-tree `PYTHONPATH`, numerical thread count 1, a 1.9 GiB virtual-memory ceiling for heavy runs, and a 120-second timeout.

- Syntax compilation of the two changed Python files: passed.
- Focused malformed-input test: `1 passed, 16 deselected in 0.22s`.
- Focused enriched/standalone/smoke sequence test: `1 passed, 16 deselected in 8.62s`.
- Bounded SU15 integration over `tests/games/test_su15.py` and `tests/games/test_su15_quality.py`, excluding teacher and rendered-frame cases: `18 passed, 2 deselected in 62.49s`.
- `git diff --check`: passed.

The integration run includes the exact closure counterexamples, all nine enriched contexts, a standalone nine-level composition, smoke semantics, clone/live suffix recovery, native undo/collision behavior, and the existing identity/order/work regressions. No full-72 census, all-nine official search, official-teacher repeat, held-out access, or desktop/rendered-frame run was performed.

Root's existing three-split nine-level WIN evidence remains 100/98/103 actions in 10.08 seconds, and the existing native-frame review remains the visual evidence. Those were intentionally not rerun. Since this closure changes only validation/build semantics and not generated grammar, routes, or rendering, their behavioral result remains applicable; their recorded `generate.py` source hash necessarily predates this final closure patch and root must bind final acceptance to the new hash below.

## Preserved closed findings and interpretation limits

The four independently cleared findings remain unchanged:

- Width-aware native horizontal reflection remains protected in geometry/gameplay identity.
- Gameplay identity retains native ordered fruit and enemy dispatch; target order remains normalized as a native union.
- Official-copy exclusion remains semantic and independent of header art/provenance.
- One honest planner transition meter remains consumed before every planner-controlled native candidate transition, including lookahead, commits, parking, recovery, suffix work, and final verification.

Work accounting still measures planner-controlled settled `Env.perform` calls, not native animation frames, clone cost, rendering, initial-state extraction, generation admission replay, validator replay, or full-game replay. Bounded failures remain inconclusive.

All nine actual official tiers and contexts, both single-tier and game generators, full native compositions, and existing meaningful late interactions remain intact. Installed target zones and witnessed occupied zones remain distinct: tier 9 currently witnesses two of three installed zones; neither that fact nor any witness proves every installed zone or enemy necessary. The evidence still has one shipped reference per tier, broad engineering bands rather than population confidence intervals, a finite placement/cluster grammar, constructive non-optimal witnesses, route hashes rather than proof of novel puzzle families, and no arbitrary graph-isomorphism or minimum-hardness claim.

## Integrity and cleanup

Initial closure hashes:

- `generate.py`: `43c1048a409a1ff044923172068f3060b232cf8f6fab5d128e2940413930fa21`
- `test_su15.py`: `8ec1b25ad8e90d2e4801956fd0b590a728920c6e16ea53ce39ff14613b5abc5d`

Final hashes:

- `generate.py`: `a6981d03ab189834e7c6636fb3ea804abb217100f01f60126dc3361970c3455e`
- `test_su15.py`: `8202d4137a2ba83b1161f676415365f60b7af11358ede78d8a44b26f1c31d0e6`

All other scoped package/test hashes are unchanged from the initial check:

- `__init__.py`: `3f2047df7b0d90104a85667b3078ae468ec1d2fbaa6817c4a073e27bad8976e1`
- `bank.py`: `ca78215d679c9d48da8fb781213089f96833e64bcb42023c2135083888d98e33`
- `env.py`: `3d9d8858410ef2079946833729746cd2f12653f8e5050ff541f797a5dfe53e10`
- `layout.py`: `48dae4b3ec91e4f0442150e4debebd973857bcd2b3c0030fbb1096f590e62089`
- `names.py`: `5b932923f856d1834710866657063187c7c86e8ad85a922d6de4618fc8be2041`
- `plan.py`: `e1c272cca21c30fc086e5734f233e380d770dab792e4188df44fe97213bf3612`
- `quality.py`: `b49a4b2f7ed8d16d043c00badf8def97f3a4ecdce7e5137f06cb6dfa289eff0d`
- `test_su15_quality.py`: `aa9cb2efcc8243dbdaad815df45076e54a9b6d7c17a2e2a290ad8edfd975f11c`

Heavy test session 82928 completed normally. Exact command-line process checks found no retained pytest or timeout process. No subagents, training, held-out data, desktop/browser, external API, or commit was used. Mailbox ready/design/phase updates were sent; all inbox polls were empty.
