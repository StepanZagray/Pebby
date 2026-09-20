# Current model status

Latest multi-game checkpoint (verified 20 September 2026): Run B,
`artifacts/multigame-v2/frozen-v2arch-B.pt`, epoch index 19. The **same frozen
weights** complete 25/177 generated training levels (1/24 games) and 18/177
generated validation levels (1/24 games). Training-task mastery and unseen-layout
transfer both remain incomplete. The auxiliary dynamics head is not used for
planning. The earlier LS20 controller below remains separately retained.
See [the audit](artifacts/model-audit-20260920/REPORT.md) and
[training and remediation commands](docs/multigame-training-operations.md).
Regenerate checkpoint-specific metrics with
`tools/summarize_multigame_checkpoint.py`; fixes to the training code do not change
these frozen weights or establish a new capability result.

Verified 13 September 2026. Seven-level completion remains unachieved. The retained
checkpoint is `artifacts/spatial-recovery-v1/quality-fit/recovery.pt`, SHA256
`ff88327214b6dc2d4278167e0d61edcc37c683292b788be6927b71286331a5f8`.

Under deterministic public-history inference it completes shipped level 1 in 23
actions, then fails level 2. Separate fresh-start evaluations of shipped levels
2–7 clear no goals and exhaust all three lives. On the reused generated development
panel it wins 7/70, all in difficulty 1. The v3 control and route-readout candidates win 6/70 and 7/70;
neither is promoted. V4's disposable qualifications pass tiny memorization but
fail the predeclared added-feature reliance test; no full v4 training or candidate
promotion has occurred. These are local measurements, not an official scorecard.

The model receives the last 8 public frames. Its encoder performs 6 refinement
passes, but its action-conditioned predictor is only one action ahead. Persistent
game memory, learned voluntary RESET, and multi-step latent planning are absent
from this retained controller. A frozen pixel perceptor and direct actor exist as experimental
components, without demonstrated overall gameplay improvement.

V5 tested four matched training variants at two training seeds. Their seed-42
development results were 6/70, 7/70, 7/70 and 7/70, with no difficulty-2–7 wins.
Joint training with uniform action targets cleared shipped level 2 in isolation,
but lost level 1; that isolated win does not establish sequential progress. No candidate was promoted. Seed-43
replication covers cached decisions, not another gameplay sweep.

A subsequent basic-navigation diagnostic found only 30/128 correct first moves
toward an already-matching goal in an empty room; the semantic candidate scored
31/128. All 128 teacher paths won in the real engine. The retained model eventually
won 11/16 selected closed-loop cases, often taking long detours. These deliberately
simplified rooms differ from its complex training distribution, so this is evidence
of a basic transfer failure, not a new benchmark score or an architecture ceiling.
Position and carried-glyph decoding were correct on every case; the additional
goal perceptor also decoded every goal correctly and is wired into the candidate.
Detailed hypotheses, negative results and component checks are recorded in
`artifacts/spatial-repair-v5/REPORT.md`.

New controlled navigation training establishes that frozen encoder features can
support these primitives: a freshly trained outcome controller won all 24 selected
generated development episodes at both training seeds 42 and 43. Finetuned and
direct-readout controls also learned the lessons, with some rollout failures despite
near-perfect expert-state decisions. However, all five seed-42 fresh/continuation
candidates scored 0/7 sequentially under a common 300-actions-per-level reset
protocol; the retained model scored 1/7 under that same protocol. These small,
policy-only fits cover matching-goal navigation, not attribute changes, resources
or multiple-goal ordering. No candidate is promoted, and confirmation remains
unscored. See `artifacts/navigation-research-implementation-v1/REPORT.md` and the
[experiment guide](docs/navigation-research.md). The new experimental checkpoint
format is supported by its dedicated numerical evaluator, not the production viewer.

Planning remains the active engineering priority. All tested neural-imagination
candidates still score0/7 in the shipped sequential game; the latest repaired
K1/K4 candidates each win3/10 actual generated development games. No candidate
is promoted over the retained1/7 controller. The latest offline action-accuracy
gain did not translate into gameplay improvement.

Testing is now gameplay-first: the planning trainer automatically compares each
candidate's actual sequential run with the retained baseline; cached-state
scores are opt-in diagnostics and cannot promote a model. Basic/world trainer
checkpoint selection now uses gameplay rather than offline metrics.
The new dynamics fit includes reset/ending supervision, but predictions remain
unreliable and persistent game memory is absent. Explicit search over the same
learned model remains a later comparison. See `artifacts/seven-level-learning-v1/`
and [planning guide](docs/neural-planning.md).

The matched 1,500-update ranking-loss experiment is complete. Both fits used the
same earlier checkpoint, fresh optimizer state and frozen visual encoder; neither
was trained from scratch. The control scored 0/7 sequentially and 8/70 generated;
ranking scored 0/7 and 7/70. All generated wins were tier 1. Both are rejected
against the retained 1/7 baseline. See
[comparison evidence](artifacts/seven-level-learning-v1/ranking-real-recipe/README.md).

At two actual K4 cycler-failure states, perception and decoded immediate action
consequences are correct, yet the policy chooses to spoil a matching appearance.
Deeper replay finds correct decoded consequences on the selected destructive
trajectory, alongside genuine rollout errors on other branches. This supports
work on both continuation/value and dynamics; it does not establish a general
encoder failure. The fresh joint qualification model has now been trained from
random initialization with all 1,524,873 parameters trainable: its dynamics has
H1 transition supervision and its planner imagines K4. The 2,000-update run
reached 8/8 generated TRAIN games at selected step 900, but its real shipped
sequential run completed 0/7 levels. The 500-update run reached 4/8 TRAIN and
also completed 0/7 shipped levels. This eight-game qualification is a tiny
trainability check, not generalization; across these two bounded runs the gate
was 0/1 then 1/1. Neither candidate was promoted, so the retained baseline
remains 1/7. There is no persistent cross-life memory or learned voluntary
RESET. The repeated-prefix check found the longer run shared initialization and
the first 500 samples but was not a bit-exact optimizer continuation. Source
files for the old qualification evidence are frozen; the broader collector,
objective and trainer work remains under construction and has not been trained.

The completed CPU gameplay comparison confirms poor transfer for `fresh900`:
the retained controller scored 1/7 sequentially and 7/70 on the generated
development panel, while `fresh900` scored 0/7 and 0/70. The gameplay gate
rejected the candidate, sources were unchanged, offline metrics were not used,
and no checkpoint was promoted. Its level-1 trace is a charged budget-exhaustion
loop that reaches a state proven unreachable before each life loss, then repeats
the same bad first action after reset; it is not a no-charge goal-pad loop. The
mask audit finds valid post-loss successor policy targets, but sparse direct
post-loss roots, so this is a recovery-coverage gap to address rather than a
claim that it caused all failures.

The opt-in generator version 4 adds goals that restore some initial glyph
components, disappearing goal rings, and a distinct test partition. Existing
banks and model weights have not been regenerated with these mechanics.

Runtime loading now accepts the spatial, route, and semantic checkpoint formats.
Select the retained checkpoint explicitly; `checkpoints/ls20-policy.pt` is not
automatically populated or pointed at an experimental candidate:

```bash
uv run serve.py --port 11435 \
  --checkpoint artifacts/spatial-recovery-v1/quality-fit/recovery.pt
```

The default missing-checkpoint behavior remains useful on machines without local
training artifacts. The numerical evaluator accepts the same explicit checkpoint.
Detailed local evidence is in `artifacts/spatial-repair-v3/REPORT.md` and
`artifacts/spatial-repair-v4/`; artifacts and checkpoint weights are not committed.


A dedicated diagnosis of the joint-goal planner's learning failure is recorded in
`artifacts/learning-failure-diagnosis-v1/REPORT.md`. On the 14-level replay pilot
the action head reaches 0.1412 held-out train policy CE by update 2,400 while its
held-out validation CE rises to 2.9494 — worse than the 1.3863 uniform baseline
and the 1.3832 best-constant baseline, against an achievable floor of 0.1204. The
failure is memorisation with negative transfer, not an inability to learn. The
head also sits at chance for its first ~600 updates because it receives 16
supervised labels per batch against 18,432 for the semantic term, which is why
runs evaluated at step 500 saw nothing. Value-loss scaling, loss reweighting,
gradient accumulation, larger batches and label degeneracy were each tested and
rejected as causes.

Rebuilding the replay dataset at 126 train levels / 8,635 roots and 28 validation
levels / 1,975 roots (`data/joint-goal-replay-fullroute-v2`) cut the validation
policy CE to 0.6938 and the generalisation gap ~17x. The resulting run
`artifacts/seven-level-learning-v1/joint-replay-v2-scaled-seed42` still completed
0/7 shipped levels and won 2/28 generated validation games, all tier 1, so it was
not promoted and the retained 1/7 controller stands. The gap resumes growing after
update 800 and validation CE stalled between updates 3,200 and 4,000, so the
128-shard reader cap (`pebby/agent/joint_goal_replay.py:429`) is now the binding
limit on further level scaling. Learner post-life-loss roots are now collected;
no training has yet isolated their effect.

## Seven-level fixes, round 1 (18 September 2026)

A step-by-step trace of the retained checkpoint against the exact planner on
all seven shipped levels (`artifacts/seven-level-fixes-v1/`) found that every
one of its 18 lost lives on levels 2–7 ends by fuel exhaustion, never by a
stall, and that it never steps onto a goal cell on those levels. Levels 2, 3, 5
and 7 charge two units per move, so a tank is 21 moves and the optimal routes
have zero slack in at least one tank. The wasted moves come from three
behaviours: re-entering a cycler after the attribute already matches (16 spoils
over 18 lives, including destroying the exact goal triple on level 2 at step 84
with a winnable route left), corridor reversals, and one low-confidence wrong
turn per life followed by no recovery. Thirty percent of all actions are taken
after the planner already says the life is unwinnable.

New engine-verified development gates in `tools/probe_ladder.py` reproduce the
diagnosis on the retained checkpoint (96 cases each; a gate, not a benchmark):

| Check | Retained | Threshold |
|---|---:|---:|
| Empty-room first action accuracy | 0.375 | >= 0.75 |
| Cycler approach accuracy | 1.000 | informative |
| Cycler leave accuracy | 0.708 | >= 0.80 |
| Cycler avoid accuracy (attribute already matches) | 0.031 | informative |
| Cycler spoil rate | 0.542 | <= 0.10 |

Training-side causes confirmed by audit: 8 oracle-chosen snapshots per level
with recovery demonstrations capped at 16 actions; every training root filtered
to have a valid optimal action so unwinnable states were never seen; loss
weights computed on the unfiltered set and applied to the filtered one; 780
fixed updates with the loss still falling and no gameplay selection; fuel read
only through a 16-column pooled HUD; deterministic argmax replaying identical
lives; tiers 6 and 7 at 3 percent and 1 percent of the bank with forced goal
order and vanishing rings never generated.

Implemented in this round (all under the branch, none promoted):

- `pebby/agent/cycler_probe.py`, `tools/probe_ladder.py`: the gates above.
- `pebby/agent/full_route_data.py`, `tools/collect_full_routes.py`: every
  oracle-route state, every learner-visited state and uncapped recoveries,
  unwinnable states retained. `data/full-routes-v1` holds 1,900 train levels
  (439,197 rows, 80,387 unwinnable) and 210 validation levels (50,042 rows).
- `pebby/agent/hud_decoder.py`, `pebby/agent/spatial_v2_policy.py`: exact
  fuel and lives scalars into the planner; v2 checkpoint format; warm start
  from v1 reproduces v1 logits to 4e-6 and the same 1/7 gameplay.
- `pebby/agent/spatial_v2_training.py`, `tools/train_spatial_v2.py`: weights
  on the trained rows, unwinnable rows kept, plateau stop, gameplay selection,
  encoder fine-tuning.
- `pebby/agent/diverse_controller.py` and `DiverseLivesDecision`: runtime
  heuristics, not learning. First life is strict argmax plus stall mask, so the
  baseline cannot regress; later lives sample. On the retained checkpoint the
  gated defaults reproduce 1/7; temperature 0.8 reached 3/7 on one seed of
  three and 1/7 on the other two, so treat that as luck, not a gain. The serving
  engine now answers RESET on game over and reports finished games.
- `tools/build_balanced_bank.py`: equal-quota generator version 4 banks with a
  forced-goal-order quota; the natural forced-order rate is 12.5 percent of
  accepted tier-6 levels and tiers 6–7 cost 1.5–3 minutes per level.

A preliminary 1,500-update fine-tune on the first 342 collected levels moved
the cycler avoid spoil rate from 0.969 to 0.719–0.760 and left the empty-room
score unchanged, while overfitting (train set accuracy 0.845, validation
0.747) and losing level 1 in gameplay after update 500. Full runs on all 1,900
levels are recorded below as they complete.

Full run 1, encoder fine-tuned, 1,900 levels, 6,000 updates at batch 96
(`artifacts/seven-level-fixes-v1/full-finetune`): validation total loss fell
from 7.45 to 6.19 and policy set accuracy rose only from 0.750 to 0.774 (train
0.859). Gameplay selection picked update 3,500, which wins shipped level 1 in
45 actions and fails level 2; the final weights lose level 1 as well. Probe
ladder: cycler avoid accuracy 0.021, spoil rate 0.583, empty room 0.375, all
unchanged from the retained model. A step trace on level 1 shows the same
on-off cycler bounce as the retained model. Under the diverse controller the
best checkpoint scores 1/7 and the final 0/7. Conclusion: with action scores
produced only by the outcome comparator, this data and objective do not teach
the leave-the-cycler decision, because the outcome heads that decide it (glyph
change, distance bins at 0.13 accuracy) stay inaccurate. A zero-initialised
direct per-action readout (`direct_readout` in the planner config, warm start
reproduces v1 exactly) was added to test that explanation; its run is recorded
below.

Full run 2, frozen encoder, batch 256, 6,000 updates
(`artifacts/seven-level-fixes-v1/full-frozen`): validation total 5.99, policy
set accuracy 0.766; gameplay 1/7 with level 1 in 135 actions; probe ladder
cycler avoid accuracy 0.010, empty room 0.312. Full run 3, fine-tuned with the
direct readout (`full-direct`): validation 6.19, set accuracy 0.773, best
update 3,500 wins level 1 in 45 actions, final loses it; probe unchanged. The
direct head alone did not change the decision.

Why, measured on 48 engine-verified avoid cases (player next to a cycler,
attribute already matching, oracle excludes entering):

| Model | Predicts glyph changes on entry | Predicted next triple correct | Value head says entering is worse | Chooses to enter |
|---|---:|---:|---:|---:|
| Retained | 0.94 | 0.27 | 0.12 | 0.94 |
| Run 3 best | 0.92 | 0.10 | 0.19 | 0.96 |

Both models know the cycler changes the glyph but not to what, and their
distance head predicts that entering brings the goal closer (mean predicted
distance 9.2 versus 10.1 for the correct move). The comparator then prefers
entry. The 130-bin distance head sits at 0.13 accuracy in every run. So the
remaining basic defect is the value supervision, not perception, data volume or
the runtime: the policy is forced to trust a distance predictor that has not
learned, and the policy term alone cannot override it. The next experiment
that follows from this is a value target the head can learn (a coarse
ordinal or relative "closer/farther than the best sibling" target, or the
existing `--distance-ordering` pairwise loss turned on), evaluated by the same
probe ladder before any gameplay claim.

Run 4 (`full-direct-lr3e4`: fine-tuned, direct readout, learner rows weighted
0.25, learning rate 3e-4) is the only run that changed gameplay: at updates
5,000 and 6,000 it wins shipped level 1 in 15 actions against an optimum of
13 (retained: 23), then loses level 2 after 108 actions and level 3 after
110 in isolation. Validation total 5.29, set accuracy 0.793. Its probe ladder
is unchanged (cycler avoid 0.042, empty room 0.250), so the level-1 gain
comes from learning the tier-1 pattern, not the general leave-the-cycler
primitive. Run 5 (`full-ordering`: as run 4 at learning rate 1e-4 with the
pairwise distance-ordering loss at weight 1) reduced that loss only from 1.08
to 0.97 and wins level 1 in 42 actions; probe unchanged. Value accuracy stays
at 0.20–0.23 in every run.

Standing result after this round: retained baseline 1/7 unchanged; no
candidate promoted. Every implemented fix is verified in isolation (data,
fuel input, weights, selection, runtime, bank tooling), and the remaining
blocker is now localised to one learned quantity: the successor distance
estimate that the action comparator relies on. Until a value target is found
that this network learns past chance, further fine-tuning of the retained
weights on more of the same data is not expected to pass the probe ladder.

Final result of run 4 (`full-direct-lr3e4`) at 12,000 updates: validation
total 5.05, policy set accuracy 0.80, value accuracy 0.24; selected update
10,000. Shipped strict: level 1 in 15 actions, level 2 game over after 102,
level 3 after 142 in isolation. Probe ladder: cycler avoid accuracy 0.021,
spoil rate 0.656, empty room 0.250. The longer run improved every offline
number and did not change the cycler decision. This is the strongest
evidence in the round that more of the same fine-tuning cannot pass the
probe ladder, and that the value target must change first. No candidate is
promoted; the retained checkpoint remains the reference at 1/7.

The balanced generator version 4 bank is complete at `data/ls20-balanced-v4`:
300 train, 60 validation and 60 test levels per tier (2,100 / 420 / 420), with
50 / 8 / 10 tier-6 levels carrying a proven forced goal order and zero seed,
geometry or gameplay overlap between splits. No model has been trained on it
yet; it is the bank to use once the value target question above is settled.

### Ablation: exact engine as world model, learned network as value (18 September 2026)

`tools/ablate_engine_search.py` searches every action sequence to a fixed
depth on cloned copies of the real game, scores leaves with a pluggable value,
plays the best first action and replans. This is tool-assisted and is not a
controller result. Isolated shipped levels, three lives, cap 300
(`artifacts/seven-level-fixes-v1/engine-search/`):

| Leaf value | Depth 3 | Depth 5 |
|---|---:|---:|
| Exact oracle distance | 7/7, every level at its optimum | 7/7, every level at its optimum |
| None (depth alone) | 0/7 | not run |
| Retained network's distance head | 0/7 | 0/7 |
| Best new checkpoint's distance head (`full-direct-lr3e4`) | 1/7 (level 1 in 13) | 0/7 |

Reading: the search is correct and a perfect world model with a correct value
solves the game optimally at depth 3. Swapping only the value for the learned
distance head drops this to 0–1/7, and deeper search makes the learned value
worse, not better, because it finds more states the head wrongly rates as
close. The learned value, not the dynamics and not the search horizon, is the
component that blocks planning. This confirms the diagnosis from the cycler
avoid cases and fixes the priority for the next round: a value the network can
learn (two-hot or ranking targets, with the goal glyph visible to the scorer),
gated by rerunning this ablation until the learned-value row approaches the
oracle row.

## Rule-inference experiment on LS20 variants (18 September 2026)

Protocol in `docs/agent-and-training.md`. Data: 1,080 exploratory games
(epsilon 0.1 around the oracle) and 240 pure-oracle games over all 24
permutations, 447k steps, plus 120 pure-random-action games on the 6 held-out
permutations. Models trained on the 18 training permutations for 4,000
updates (`artifacts/variant-inference-v1/`).

First scoring on all steps was misleading: the memoryless control reached
0.80 movement accuracy on held-out permutations because 90 percent of actions
were oracle moves toward the goal, so the outcome was predictable from the
board without knowing the mapping. All arms were therefore re-scored on
random-action steps only (`tools/score_variant_arms.py`).

Held-out permutations, random-action steps, movement accuracy:

| Arm | Mixed data (9,374 steps) | Fully random games (17,718 steps) | First 10 steps | After step 30 | Median steps to 20 correct in a row |
|---|---:|---:|---:|---:|---:|
| Identity baseline (memorised standard controls) | 0.181 | 0.228 | 0.21 | 0.23 | never |
| Memoryless network | 0.269 | 0.276 | 0.26 | 0.28 | never |
| Transformer without previous-outcome inputs | 0.965 | 0.338 | 0.32 | 0.34 | 28 |
| Transformer with previous action and outcome | 0.985 | 0.942 | 0.59 | 0.98 | 11 |
| Explicit hypothesis inference (no training) | 0.972 | 0.962 | 0.76 | 0.98 | 5 |

Readings. A 175k-parameter transformer trained on 18 permutations does infer
an unseen permutation in-context: it starts near chance and reaches 0.98
after about thirty steps, so for this small rule family the update procedure
was learnable from 18 rule sets, which corrects the earlier expectation that
it would only memorise. It needs the previous step's action and observed
outcome as explicit inputs; the variant without them only works on the
oracle-heavy action distribution it was trained on and collapses to 0.34 on
random actions, a distribution-dependent shortcut. The explicit inferrer,
which knows the fixed rules and eliminates permutations, needs no training,
identifies the permutation in a median of 5 to 6 actions in every game, and
is the strongest early in the game. A learned prior over permutation identity
changed nothing on held-out permutations, as expected. The exact-engine rule
model matched the hand-written local rule (0.977 movement, zero abstentions).

Limits: the varied rule is a permutation of four actions, a far smaller
family than real game mechanics; the shipped-game data shortcut shows how
easily an evaluation on oracle-driven actions overstates inference. Both
findings carry to real public games once their engines are available.

## Multi-game imitation run v1 on 24 public families (19 September 2026)

Protocol: `docs/multigame-generators.md`. Data: 406 train / 397 validation
generated whole games from all 24 public families (18 per family except bp35
7/2 and tn36 3/2; see `data/multigame-v1/merged/merge-report.json`), teacher
routes only, stored variants at mix 0.5 plus load-time variants at mix 1.0
(controls, D4 spatial, palette; one per game per epoch). Model: the repo's
GRU visual imitation policy with a one-step auxiliary predictor
(`pebby/agent/multigame_model.py`), 30 epochs, 28 minutes on the RTX 5060,
checkpoint selected by generated closed-loop validation (epoch 18), frozen as
`artifacts/multigame-v1/frozen-v1.pt` (sha256 a9052912...).

| Evaluation | Result |
|---|---|
| Offline validation action loss / click loss | 1.02 to 0.87 nats / 8.1 to 4.6 nats over 30 epochs |
| Generated closed-loop validation (24 games per epoch) | best 1 level completed, 0 games won, at any epoch |
| Official training-family games, all 24, frozen checkpoint | 0 levels completed, 0 games won; every game ran to its own action budget |
| Held-out m0r0, no tuning after phase 1 | 0 of 6 levels, 151 actions |

Reading: the imitation losses fall but the policy does not complete generated
levels in closed loop, so the official 0/24 and the m0r0 0/6 are bounded by
that and carry no information about transfer. The corpus is teacher-route
only (the mixed recovery cohort stalled on dc22 and was dropped), so the
policy has never seen a state after its own mistake; this is the same
recovery gap diagnosed on LS20 earlier. The generator review
(`artifacts/multigame-v1/checks/REVIEW.md`) also records that generated
tiers are systematically easier than official ones and that undo is never
demonstrated. Next diagnostic before any larger run: closed-loop play on the
training games themselves (does the policy complete levels it was fitted on),
then add recovery data with a bounded live-recovery search.

### Canonical baseline (v2, 19 September 2026, after the independent investigation)

The Codex GPT-6 Astra investigation (`artifacts/multigame-v1/codex/investigation.md`)
showed the v1 policy completed 0 of 58 levels on its own training games and had
learned to repeat the previous action (repeat baseline 0.674 vs policy 0.686), with
the hidden per-game control permutation making the first action irreducible and the
checkpoint selection ranking game overs above survival. Fixes applied: canonical
inputs (stored variants inverted, no augmentation), honest outcome-classified
selection, a training-game closed-loop panel, per-family and repeat-baseline offline
metrics (`pebby/agent/multigame_training.py`, `--canonical`); bounded live-recovery
collection with wall-clock caps and learner-state perturbations (`pebby/multigame.py`).

Canonical run (`artifacts/multigame-v1/train-canonical-v1`, 30 epochs, same 406/397
corpus): offline action accuracy 0.657 -> 0.79 (repeat baseline 0.674), click loss per
target 3.72 -> 1.95 nats, generated closed-loop validation 1 -> best 8 of 177 levels,
training-game panel best 3 of 55 levels, 0 games won anywhere. Selected epoch 24.
Reading: canonicalisation removed the action-history shortcut and doubled click
quality, but the imitation policy still does not master its own training games in
closed loop; train loss keeps falling (0.68) while offline accuracy plateaus at 0.79,
so the remaining gap is recovery data (none in this corpus) and click supervision,
in that order per the investigation. Official evaluation of this checkpoint follows.
Official evaluation of the canonical checkpoint (`frozen-canonical-v1.pt`): 0 of
24 training-family games and 0 levels; held-out m0r0 0 of 6. Consistent with the
generated closed-loop numbers; the gate "complete the training games in closed
loop" is not met, so official results remain uninformative about transfer. Next
run adds bounded live-recovery data (`data/multigame-v1/mixed-bounded`, in
collection) to the canonical corpus.

### v3: canonical inputs plus bounded recovery data (19 September 2026, 20:37)

Corpus `data/multigame-v1/merged-recovery`: 473 train / 459 validation games; train
routes 5,402 certified, 4,516 live-recovery, 1,372 random-action transitions (bounded
live recovery: 20 s per search, 300 s per game; dc22 collected with two capped
timeouts and no stall). Training as in v2 (canonical, 30 epochs). Offline action
accuracy 0.78 (repeat baseline 0.670), click loss per target about 2.1 nats,
generated closed-loop validation best 8 of 177 levels, training-game panel best
3 of 55, 0 games won. Official: 2 levels completed across the 24 training-family
games (lp85 level 1 in 67 actions and sc25 level 1 in 72; the first official
levels ever completed by this pipeline), 0 games won;
held-out m0r0 0 of 6. Reading: recovery data did not change closed-loop mastery
of training games; the gate is still unmet. Click supervision was the next item
on the investigation's list and is now implemented; the next run uses a
region-labelled corpus (`data/multigame-v2`).

### Run A: architecture v2 + history dropout + per-game updates (20 September 2026, 01:52)

Second independent investigation (`artifacts/multigame-v1/codex/investigation2.md`):
feeding the policy its own predicted previous actions drops accuracy 84 -> 49 percent;
freezing frames within a level changes 27 of 9,864 action decisions (the action path
ignored the image); clicks hit the functional region 58.6 percent; divergence from the
teacher at median step 1. Implemented: architecture "v2" (8x8 spatial feature grid on
the action path, convolutional 64x64 click decoder, 2.58M parameters), per-game history
dropout 0.5, one optimiser step per whole game with per-term target normalisation,
adapter-error guard for empty legal masks, per-level/switch/region metrics, a 24-family
training-game closed-loop panel, and `--history-free` diagnostics.

Run A (`artifacts/multigame-v1/train-v2arch-A`, canonical, merged-recovery corpus,
30 epochs, seed 1): action-switch accuracy 0.09 -> 0.60, click exact 0.18 -> 0.58,
offline action accuracy 0.79 (repeat baseline 0.670). Training-game closed-loop panel:
27 of 177 levels and 1 of 24 games won at its best (previous best 3 of 55 levels, 0
games). Generated validation closed-loop best 9 of 177 levels, 0 games won. Official:
0 of 24 games and 0 levels; m0r0 0 of 6.
Reading: the model now fits its training games in closed loop far better, so the
diagnosed shortcuts were real; the remaining gap is generalisation to unseen layouts
(train panel 27 vs validation 9 on the same recipe). Run B applies the same recipe
to the 2.7x larger region-labelled corpus (`data/multigame-v2/merged-all`, 1,286 train
games, 45k region-labelled clicks) to test whether data volume closes that gap.

### Run B: same recipe on the region-labelled union corpus (20 September 2026, 03:53)

Corpus `data/multigame-v2/merged-all`: 1,286 train / 1,238 validation games (12 new
region-labelled preparations with bounded recovery plus the earlier recovery corpus;
lf52 teacher games in the new preparations hit the 300 s game cap under region probing
and were kept as partials, so the union supplies its completed games). Train routes:
150,008 certified, 68,398 live-recovery, 20,006 random (238,412 total, counted from
NPZs); zero learner-policy transitions. The old manifest summary omitted 81,702
certified rows from legacy records. There are 44,959 region-labelled clicks out of
68,749 click targets; 23,790 still use exact-pixel fallback.
Recipe as run A (`--canonical --architecture v2 --history-dropout 0.5 --update-mode game
--require-click-regions`), 30 epochs, closed-loop panels every second epoch.

| Metric | v1 | canonical | +recovery (v3) | run A | run B |
|---|---:|---:|---:|---:|---:|
| Offline action accuracy (repeat baseline) | 0.69 (0.67) | 0.79 (0.67) | 0.78 (0.67) | 0.79 (0.67) | 0.80 (0.65) |
| Action-switch accuracy | n/a | n/a | n/a | 0.60 | 0.62 |
| Click exact / region accuracy | n/a | n/a | n/a | 0.58 / 0.58 | 0.61 / 0.62 |
| Training-game closed-loop, best (levels / games of 177 / 24) | 0 / 0 (of 58 lvls) | 3 / 0 (of 55) | 3 / 0 (of 55) | 27 / 1 | 51 / 2 |
| Generated validation closed-loop, best levels of 177 | 1 | 8 | 8 | 9 | 18 |
| Official 24 games: levels / games | 0 / 0 | 0 / 0 | 2 / 0 | 0 / 0 | 3 / 0 |
| Held-out m0r0 levels of 6 | 0 | 0 | 0 | 0 | 0 |

The table contains historical maxima, not one matched set of checkpoint weights.
The 51 training levels belong to epoch index 29; the frozen checkpoint is epoch
index 19 and completes **25/177 training levels and 18/177 validation levels**,
one whole game on each 24-game panel. Nine validation levels come from BP35;
BP35's offline validation trajectories contain targets only for its first level.

Official levels completed by Run B: lf52, lp85 and su15, each level 1. No checkpoint
is promoted; the protocol result stands at zero games on both official phases.
The audit also measured residual recorded-history dependence, weak functional
click prediction, and poor prediction of changed pixels and positive events.
These establish a training-task control problem as well as a transfer gap.
Learner-state recovery data and revised objectives are experiments to address
those deficits; their effectiveness must be measured after new training.
