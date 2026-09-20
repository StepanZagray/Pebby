# Handoff: multi-game training programme (updated 19 September 2026)

## State
- All 24 public-game generators accepted (`.scratch/multigame-resume/full-standard/STATUS.md`),
  m0r0 held out. Independent review of every family with native replays and frame
  comparisons: `artifacts/multigame-v1/checks/REVIEW.md` (+ `frames/<family>/`).
- Fixes applied after review: sp80 (objects off edge sinks), ar25 (4-connected
  polyominoes with pinned component counts), tn36 (public teacher route carries
  required mechanic events; DEFAULT_ATTEMPTS 600), ls20 (official D4 geometry
  exclusion, `official_copy` field, DEFAULT_ATTEMPTS 200).
- Family test suite: 672 passed, 7 failed, 7 errors (`artifacts/multigame-v1/checks/tests-games.log`);
  triage in progress (test-only edits allowed).
- Corpus runs (both resumable by re-running the same command; see the .log next to each):
  - `data/multigame-v1/teacher`: 6 certified teacher games per family (train), validation
    cohort by the tool default; variants on (controls, spatial, palette, mix 0.5). The
    preparation tool is NOT resumable (refuses a non-empty root); an earlier 12-game run
    was stopped and its partial output kept at `data/multigame-v1/teacher-partial-12`.
    Command: `tools/prepare_multigame_dataset.py --output-root data/multigame-v1/teacher
    --train-master-seed 1000 --validation-master-seed 2000
    --completed-teacher-games-per-family 6 --mixed-games-per-family 0 --variants
    --variant-components controls spatial palette --variant-mix 0.5 --variant-seed 7`.
    Caveat: s5i5 tier 4 was generated before the seed-variety fix landed.
  - `data/multigame-v1/mixed`: 1 teacher + 3 mixed (epsilon 0.2, max 800 steps) games per
    family. The dc22 mixed cohort was very slow in the pilot (live recovery); if it stalls,
    train on the teacher manifests alone.
- Pilot timing (1 teacher + 1 mixed per family) is in `artifacts/multigame-v1/prepare-pilot.log`.

## Next steps (protocol from docs/multigame-generators.md)
1. `tools/train_multigame.py --train-manifest <teacher train manifest> [<mixed train manifest>]
   --validation-manifest <teacher validation manifest> --out-dir artifacts/multigame-v1/train-v1
   --device cuda` (defaults: 1 epoch; raise epochs; generated validation selects the checkpoint).
2. `tools/evaluate_multigame.py --checkpoint <frozen> --phase training-families --out ...`
   (24 official games), then, without tuning, `--phase heldout` for m0r0.
3. Record both in MODEL_STATUS.md with the review caveats (generated tiers are easier than
   official; undo never demonstrated; several finite-grammar tiers).

## Parallel corpus generation (rule: always parallelise)
`tools/prepare_multigame_dataset.py` is single-process and non-resumable. Run
several instances with different master seeds and 1 game per family each,
into `data/multigame-v1/teacher-p<i>`, and pass all their manifests to the
trainer (`--train-manifest a b c ... --validation-manifest x y z ...`). The
train/validation split is a deterministic geometry-hash partition, so separate
runs cannot leak into each other. Six such runs were launched 09:26 with
seeds 5001..5006 / 6001..6006. Training command to use once manifests exist:
`tools/train_multigame.py --train-manifest data/multigame-v1/teacher*/train-teacher/manifest.json
--validation-manifest data/multigame-v1/teacher*/validation-teacher/manifest.json
--out-dir artifacts/multigame-v1/train-v1 --device cuda --load-variants
--load-variant-mix 1.0 --load-variant-components controls spatial palette --epochs N`.

## Cross-run manifest collisions (found 10:10)
The trainer's audit rejects train manifests from one preparation combined with
validation manifests from another: bp35 produces identical initial frames from
different seeds ("train/validation overlap in puzzle fingerprints"). Also, any
preparation whose validation cohort misses a family (bp35/tn36 exhaust their 8
candidates) has is_full_experiment_collection=False on BOTH its manifests.
Resolution in progress: tools/merge_multigame_manifests.py unions all runs,
drops colliding validation games, and re-audits; train on data/multigame-v1/merged.
Smoke training on teacher-p1 alone: artifacts/multigame-v1/train-smoke (3 epochs,
2 minutes; pipeline works).

## Merged corpus and training v1 (10:30)
`tools/merge_multigame_manifests.py --keep-validation bp35:2 tn36:2 --drop-validation-collisions --link`
over `teacher` + `teacher-p1..p12` (partial-12 excluded) -> `data/multigame-v1/merged/{train,validation}`,
strict audit passed: 406 train / 397 validation games; 18/18 per family except
bp35 7/2, tn36 3/2 (their tutorial tiers have only a few distinct layouts, so
validation games had to be bought by dropping colliding train games), ka59 18/17,
re86 18/17, s5i5 18/17. Training v1: `tools/train_multigame.py --train-manifest
data/multigame-v1/merged/train/manifest.json --validation-manifest
data/multigame-v1/merged/validation/manifest.json --out-dir artifacts/multigame-v1/train-v1
--device cuda --epochs 30 --load-variants --load-variant-mix 1.0 --load-variant-seed 1
--load-variant-components controls spatial palette --seed 1` (log: train-v1.log).
Then: evaluate_multigame --phase training-families (all 24, ~10 min), then --phase heldout (m0r0).
The mixed-cohort preparation was stopped (dc22 live-recovery stall); not used.

## Results (11:05)
Training v1 done (30 epochs, 28 min). Phase 1 official: 0/24 games, 0 levels.
Phase 2 m0r0: 0/6 levels. Generated closed-loop validation never exceeded 1
level. Recorded in MODEL_STATUS.md. Reports: artifacts/multigame-v1/eval-v1-*.json.
Note: `evaluate_multigame.py --phase heldout --training-report` expects the
PHASE-1 OFFICIAL REPORT json, not the training log.

## Independent investigation (Codex GPT-6 Astra, xhigh, 16:43) and fix plan
Report: artifacts/multigame-v1/codex/investigation.md. Confirmed by its own measurements:
- frozen policy completes 0/58 levels on its OWN training games (not a transfer gap);
  four directional families emit action 3 on every step; click families repeat coordinates.
- hidden control permutation at game start is irreducible before interaction (24 LS20
  permutations give identical model inputs; NLL floor ln 4); with teacher-only histories
  the model learns to imitate the previous action (repeat baseline 67.45% vs policy 68.62%).
- click head can address every pixel; supervision, not resolution, is the gap.
- zero recovery transitions; zero native undo in the corpus.
- checkpoint selection ranks GAME_OVER above budget exhaustion (tie-break bug).
- all suspected mechanical skews (frames, previous actions, masks, clicks, memory, budgets) checked out.
Fix order in progress: (1) canonical baseline (invert stored variants, no load-time variants,
train-game closed-loop panel, honest selection) -> retrain and require completion of training
games; (2) bounded learner-state recovery collection (wall-clock caps; dc22 stall); (3) then
click-region supervision, diversity, and an interaction curriculum for control identification.

## State at 17:55
- Canonical baseline trained (`artifacts/multigame-v1/train-canonical-v1`, frozen as
  `frozen-canonical-v1.pt`): offline acc 0.79 vs repeat 0.674, closed-loop val best 8/177
  levels, train panel best 3/55, official 0/24, m0r0 0/6. Recorded in MODEL_STATUS.md.
- Recovery collection running (PID 373433, `data/multigame-v1/mixed-bounded`, log
  `data/multigame-v1/mixed-bounded.log`): teacher cohorts done, train-mixed done, on
  validation-mixed. When finished: merge its four cohort manifests with
  `data/multigame-v1/merged` via `tools/merge_multigame_manifests.py --keep-validation
  bp35:2 tn36:2 --drop-validation-collisions --link` into `data/multigame-v1/merged-recovery`,
  then train `--canonical` (same CLI as train-canonical-v1) into
  `artifacts/multigame-v1/train-canonical-recovery-v1`, then both evaluation phases.
- Then Codex fix item 3: set-valued (engine-verified equivalent) click targets in the
  collector, GameSequence and compute_multigame_loss; do not edit pebby/multigame.py while
  a collection is running.

## 18:35 state
- Training v3 (canonical + recovery, `artifacts/multigame-v1/train-canonical-recovery-v1`)
  running; both official phases chained after it (`eval-canonical-recovery-*.json`).
- Click-region supervision landed (collector stores engine-verified equivalent-click
  masks; region-mass loss; 172 multigame tests pass). Old corpora have no region labels.
- Region-labelled corpus v2 collecting: 12 preparations `data/multigame-v2/prep-<i>`
  (seeds 9001..9012 / 10001..10012), each 1 teacher + 2 bounded-recovery games per family.
  When done: merge all their cohort manifests (train-teacher, train-mixed / validation-*)
  with `tools/merge_multigame_manifests.py --keep-validation bp35:2 tn36:2
  --drop-validation-collisions --link` into `data/multigame-v2/merged`, train
  `--canonical` into `artifacts/multigame-v2/train-canonical-regions-v1`, evaluate both phases.

## 20:40 state
v3 done: official 2 levels / 0 games on the 24 training families, m0r0 0/6 (MODEL_STATUS.md).
v2 region-labelled preparations running (12 procs, ~30 of ~144 games each at 20:38).
Next: merge v2, train v4 canonical with region loss, evaluate; GPU idle until then.

## Second independent investigation (Codex GPT-6 Astra, high, 22:18) and implementation
Report: artifacts/multigame-v1/codex/investigation2.md. Measured on training games:
feeding the model its own predicted previous actions (teacher frames kept) drops action
accuracy 84 -> 49 percent (exposure bias via action-history input); freezing frames within
a level changes only 27/9,864 action argmaxes (action path blind to the image); clicks hit
the functional region 58.6 percent; policy diverges from teacher at median step 1 of a level;
empty legal mask raises uncaught ValueError in closed-loop eval; trailing 1-8 step chunks
get full optimizer steps. Implementation split: model agent (architecture "v2": 8x8 spatial
action path + conv click decoder; `history_keep` mask; loss reduction="sum"; aux skip),
trainer agent (per-level/switch/BOS/region metrics, 24-family train panel, adapter_error
guard, `--history-dropout`, `--history-mode`, `--history-free` diagnostic,
`--require-click-regions`, `--update-mode game`, `--frame-weight/--event-weight`).
Planned runs after both land (matched seed 1, canonical, 30 epochs, cuda):
 A. v2 arch + history-dropout 0.5 + update-mode game on data/multigame-v1/merged-recovery
 B. same on data/multigame-v2/merged (region labels) when the v2 collection is merged
 then both official phases on the selected checkpoint of each.

## 00:30 (20 Sep) state
- Run A (`artifacts/multigame-v1/train-v2arch-A`, v2 arch, history-dropout 0.5, game
  updates, canonical, merged-recovery corpus) running; epoch 16: train-panel 8 levels (best
  before this round: 3/55), val 8/177, evals chained (`eval-v2arch-A-*.json`).
- v2 region-labelled corpus: 12 preparations done (~137 games each, all four cohorts),
  merged into `data/multigame-v2/merged` (see merge-report.json). Run B: same flags as A
  plus `--require-click-regions`, out-dir `artifacts/multigame-v2/train-v2arch-B`, then both phases.
lf52 teacher games in v2 preps hit the 300 s game cap with region probing; union with merged-recovery used instead (data/multigame-v2/merged-all)

## 04:00 (20 Sep) results
Run A: train panel 27/177 levels, 1 game; val 9/177; official 0/24; m0r0 0/6.
Run B (union corpus, regions): train panel 51/177, 2 games; val 10/177; official 3 levels
(lf52/lp85/su15 level 1), 0 games; m0r0 0/6. Full table in MODEL_STATUS.md.
Open: generalisation across layouts. Candidates: longer training with early stopping on
the validation panel (train panel still rising), more layouts per family (parallel
preps are cheap: 12 procs ~6 h for ~1,600 games), curriculum on level index, and a
planning/search component over the transition head (never used for control so far).
