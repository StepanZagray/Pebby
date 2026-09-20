# SU15 v3 frozen closure review

## Scope and decision

**Four old findings close; two remain partially open. Recommend keep full-standard acceptance pending.** Root owns acceptance. No status/source/test/canonical-note changes were made.

Read the entire prior su15-review-detailed.md, external-su15-re86-followup/su15-detailed.md, current worker notes, current scoped source/tests and the refreshed root integration artifact. Review started2026-09-19 at01:03 CEST; local machine timezone was verified with date. The author explicitly labels old72-row and earlier all9-teacher evidence as pre-correction. Those audits were not rerun here.

Frozen current versions: format/generator v3, mechanics su15-full-mechanics-v3, profile su15-nine-reference-v2, geometry su15-width-aware-hreflection-v2, gameplay su15-native-object-order-v2, proof su15-native-proof-v2 and work su15-native-transition-work-v2. The contract remains pending_audit.

## Remaining findings

### P1 — enriched sequence identity and completion claims are not checked against their facts

Locations: pebby/games/su15/generate.py:772-781,793-810,994-1011.

Reproduction starts with generate_game(889, split='train'), which returns a valid complete game. Independently modify only row0 in a deep copy with each of the following:

| Mutation | validate_full_standard(row0, curriculum[0]) | build_game(all9) |
|---|---|---|
| game_position=8 | [] | returns9 levels |
| game_sequence_sha256='0'*64 | [] | returns9 levels |
| parent_game_seed=890 | [] | returns9 levels |
| proof.full_game_replay.state='WIN' | [] | returns9 levels |
| proof.full_game_replay=None | [] | returns9 levels |

The schema validates game_position only as an integer0..8, sequence hash only as hex syntax and parent seed only as an integer domain. It does not require full-game position to equal context, recompute the sequence digest, or bind child seed to parent/ordinal/difficulty. The full_game_replay check explicitly skips None, while the full-sequence requirement checks only key presence. Non-null replay state is accepted as either NOT_FINISHED or WIN for every tier. build_game performs a real replay but discards its returned rows instead of checking the supplied certificate or sequence metadata.

Native gameplay still completes correctly in these counterexamples. The finding concerns false accepted proof/provenance, not an inability to win. Standalone specs without sequence enrichment must remain supported; when sequence claims are present, they must be checked. For complete enrichment, require row/context position agreement, consistent parent and recomputed child-seed derivation, recomputed ordered sequence hash, and exact non-null replay facts including WIN only on tier9. Compare actual full replay rows at the builder boundary. Preserve explicit smoke-subset semantics separately.

This leaves original finding4 partially open. Original top-level type/flag/work/route-mirror tampering is fixed; the advertised complete optional full-sequence proof schema is not yet complete.

### P2 — the schema gate still executes operations on invalid nested values

Locations: generate.py:670-680 and804.

Starting from generate(7001,1), mutate fruits[0].tier independently to '2', None, [] or {}. validate_full_standard raises TypeError in width=value.get('tier',8)+1. It appends an invalid-tier diagnostic but continues into arithmetic instead of skipping dependent checks.

Mutate sequence_kind independently to [] or {}. validate_full_standard raises TypeError: unhashable type in membership against the two-string set. This occurs even if the rest of the required sequence enrichment is missing; accumulating earlier diagnostics does not protect later operations.

Original cases fruits=[None], targets=[None] and solution_length='bad' now return finite error lists. The author tests cover those exact shapes but not these adjacent nested scalar/container shapes. Validate type before dependent operations or return/continue after a shape failure. The schema entry point itself is outside the native rebuild exception handler, so these exceptions propagate to collectors.

This leaves original finding5 partially open. No broad malformed-input fuzzing or large-allocation test was needed to establish the issue.

## Closed original findings and independent evidence

### 1. Width-aware reflection and public split identity

quality.py:108 now transforms top-left x as64-native_width-x. Independent probes used _draft(Random(7100+d),d) for each d1..9; applying _mirror_values to fruits/enemies/targets preserved geometry_identity, gameplay_identity and geometry_partition in every case. Translating the reflected objects by[3,2] and changing solution, proof, private interaction_plan, seed and a cosmetic label also preserved both canonical identities. This tests pure identity functions; translated drafts were not claimed native-playable when close to boundaries.

The earlier reflected-tutorial train/validation leakage no longer reproduces. Native horizontal reflection remains protected, rather than being removed from the contract. Raw geometry remains a separate absolute-position representation. No fullD4 or arbitrary graph-isomorphism claim is inferred.

### 2. Native dispatch order

The regression test reruns the exact earlier fruit-tie and enemy-knockback counterexamples. Each starts with equal native rendered arrays, reverses one relevant input list, then applies the same far click. Native outcomes remain different, and gameplay hashes now differ. Geometry stays an order-insensitive identity; gameplay separately retains ordered fruit/enemy families. Target order is normalized because the native predicate treats target zones as a union. No code-level control ordering was altered.

### 3. Header-independent official semantic denial

Independent native initial-state extraction covered all9 official contexts, without teacher search. Each exact semantic transplant and its native-width reflection was passed to _verify at limit1. All18 returned no row and reason official_gameplay_copy before search. Requirements/budgets and typed objects came from native starts; artwork/provenance did not enter the comparison.

Separately, a fully schema-valid tutorial row with its gameplay replaced by the official tutorial passed _schema_errors=[] and was rejected by validate_full_standard with the explicit 'official gameplay-equivalent start is forbidden' diagnostic. It also had stale geometry/replay evidence, which the validator independently rejected; this was a denial test, not a recertification or a claimed official solution.

### 6. One planner transition meter, including final verification

Source inspection finds the only direct Env.perform call in plan.py inside _WorkMeter.perform. Each probe wrapper inspected its caller to assert that the caller was that meter and that0<used<=limit before the native transition began. Reported SearchResult.work equaled actual Env.perform calls.

For generated tier6 seed7006 with stored solution removed:

| Limit | Actual/reported calls | Result |
|---:|---:|---|
|0|0|cutoff, no witness|
|1|1|cutoff, no witness|
|2|2|cutoff, no witness|
|5|5|cutoff, no witness|
|10|10|cutoff, no witness|
|500|86|verified positive witness|
|85|85|cutoff during final verification, no witness|

The86-call positive run breaks down into20 degradation lookahead,4 committed click calls,46 candidate-placement lookahead,6 committed placement calls and10 final replay calls. With limit85, the final replay uses9 calls and correctly returns no certificate even though construction had reached a win. That failure is inconclusive, not proof of impossibility.

Both _park_threat_preserving_mass and _park_pursuer were exercised directly with fresh constructors at limits0/1/2. Each used exactly0/1/2 calls. Stored-suffix recovery at limits0/1/5 used exactly those limits and returned cutoff; limit500 returned a positive suffix using55 actual calls, all through _verified.

Merge-phase probes add coverage beyond the old tier6 case. Tier2 seed7002 at cap500 returned a positive witness with206 calls:121 merge lookahead,8 clicks,55 candidate-placement,7 committed placement and15 final verification. Tier9 seed7009 returned a positive witness with41 calls:11 merge lookahead,6 clicks,4 degradation lookahead,10 candidate-placement,2 committed placement and8 final verification. Both tiers at caps1/5 used exactly1/5 and returned cutoff.

### Work accounting convention and limits

Search work includes every planner-controlled public native action, including candidate lookahead, commits, parking, recovery and the planner's final positive verification. It counts settled Env.perform actions, not internal animation frames, clone/copy cost, initial-state extraction or rendering.

Generation/admission have additional bounded evidence replays outside that search meter. Instrumented generate(7101,1) records search_work_used91 for a9-action witness, but executes109 total Env.perform calls:91 search plus9 mechanic_trace replay in _verify plus9 validation replay. The91 is therefore honest search work, not total generation work. Full-game replay and later validation likewise add work;72 attempts remain a separate per-call bound. This explicit convention is required when reporting resources. No additional cap violation was found in the planner itself.

## Regressions, full-nine integration and retained limitations

Independent command: primary venv Python invokes pytest with bytecode disabled and cacheprovider disabled on tests/games/test_su15.py, selecting 'not teacher_solves and not rendered_frames'. **15 passed,2 deselected in16.79s.** It reruns the new identity/schema/official-copy/work regressions plus original generated/native integration. It includes full default generate_game(888), deterministic regeneration, build_game of enriched rows and separately generated standalone all9 rows, sequential contexts, first completion on stored final action, native undo/collision behavior and live-prefix recovery. The adjacent seed889 episode also builds and replays all9 through the mutation probes above; enrichment admission is what fails closed incorrectly. git diff --check passed.

The author's9 focused checks,27-tier/split smoke and16-test integration subset are read evidence, not newly rerun totals here. No full72 census or expensive all9 official search was repeated. Later-tier causal native interactions from the first review remain positive evidence; this closure does not add a new necessity census.

Root's refreshed source-bound su15-root-integration.json reports current v3 train/validation/test all9 WIN in100/98/103 actions, no reported canonical overlap, elapsed10.084s and an explicit pending-admission override. Its package hashes match this frozen review. The coordinator reports that refreshed native frames were visually inspected and restored colored progression cues are clear. Root owns those visual checks; this reviewer did not use desktop/browser or make an independent visual-quality claim.

Keep all interpretation limits: one reference per tier, broad engineering bands rather than population confidence intervals, finite cluster/placement grammar, constructive witnesses rather than optimal routes, route hashes rather than proof of novel puzzle families, and inconclusive bounded failures. Tier9 currently witnesses two of three installed target zones. Witnessed occupied zones are not a proof that every installed zone or enemy is necessary. Existing native exact-type completion still governs wins.

## Integrity and cleanup

Frozen primary hashes:

- generate.py:43c1048a409a1ff044923172068f3060b232cf8f6fab5d128e2940413930fa21
- plan.py:e1c272cca21c30fc086e5734f233e380d770dab792e4188df44fe97213bf3612
- quality.py:b49a4b2f7ed8d16d043c00badf8def97f3a4ecdce7e5137f06cb6dfa289eff0d
- native source:a5f91f7c963d6ca6447dae0ab21342b48a3f511601c40dfa8e972bdc59b4651e

All8 package Python files,2 scoped test files and native source were hashed initially and rechecked before final handoff. All11 remain unchanged. Heavy sessions64747,23000,14906,79120 completed with exit0 and were checked closed. Every heavy run was sequential, timeout120s, address-space ceiling1900MiB, numerical threads1 and primary-venv/worker-PYTHONPATH. No processes are retained.

Only these two closure reports and the original su15-review mailbox were written. No source/test/canonical-note edits, commits, training, held-out access, subagents or desktop/browser. No new style-only finding; the remaining two issues are specification correctness, P1 enriched proof and P2 malformed schema.
