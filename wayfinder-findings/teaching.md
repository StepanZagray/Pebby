# Teaching goal-directed and compositional LS20 behavior

Status: **DONE_WITH_CONCERNS**. Research completed 13 September 2026, Europe/Dublin (IST). No model, data collector, or training job was run. Proposed budgets below are decision probes, not convergence estimates or authorization to spend resources.

## Answer

Test whether the existing model can learn reliable goal-dependent primitives before scaling the curriculum or introducing online RL. Then test **repeated learner-state relabeling** against equally sized static replay, followed by compositional scheduling and explicit subgoal/process targets. Sequence imitation becomes a distinct intervention only when information or gradients persist across decisions. RL is worth comparing after a controller produces meaningful successful trajectories; it is not a substitute for establishing what the current supervised targets teach.

This ordering is an LS20 hypothesis, not a published result. The existing failure does not establish insufficient parameter capacity. Nor is Pebby missing all agent-oriented supervision: it already learns optimal actions, remaining route distance, and outcomes on learner-generated states.

## Verified local evidence packets

### L1 — Behavior and objective differ from a purely predictive world model

- Claim/status/confidence: **Established, high**. Retained H8 inference has a one-action predictor plus a learned comparator; its objectives include native action imitation and remaining-distance classification, in addition to physical/events predictions and teacher-outcome comparator supervision.
- Evidence/source tier: **local-primary**. [Outcome loss](/home/stepan/Projects/code/Pebby/pebby/agent/neural_outcome_planner.py:186), lines 186–240; [spatial additions](/home/stepan/Projects/code/Pebby/pebby/agent/spatial_outcome_objective.py:37), lines 37–80.
- Revision/retrieval: Dirty working source over commit `e16b224eb0648177619599025174e332743528a3`, read 2026-09-13; hashes below.
- Scope/limits: Remaining distance contains long-horizon teacher knowledge despite one-step prediction. Calling this “only learning physics” is incorrect. Current status remains 1/7 sequential and 7/70 exposed development, all D1; persistent memory, voluntary RESET and multistep planning are absent. [Status](/home/stepan/Projects/code/Pebby/MODEL_STATUS.md:7), lines 7–40. Primitive first-action results, 30/128 retained and 31/128 semantic, are deliberate distribution-shift diagnostics. Later failure-supervision repairs and V5 do not establish improved gameplay; V5's isolated L2 gain loses L1. [V5 evidence](/home/stepan/Projects/code/Pebby/artifacts/spatial-repair-v5/REPORT.md:23), lines 23–27 and 58–77. No architecture ceiling follows.

### L2 — DAgger components and scheduling already exist

- Claim/status/confidence: **Established, high**. The collector executes the policy on public H8, labels all four engine branches, and adds expert continuations after selected mistakes. V3 recollects from the retained checkpoint. Each collection job fixes its policy; recovery is bounded and within-life.
- Evidence/source tier: **local-primary**. [Recovery collector](/home/stepan/Projects/code/Pebby/tools/collect_spatial_recovery.py:63), lines 63–179 and 263–265; [V3 orchestration](/home/stepan/Projects/code/Pebby/tools/collect_spatial_repair_v3.py:77), lines 77–104.
- Revision/retrieval: Dirty working source, read 2026-09-13; hashes below.
- Scope/limits: This is substantial learner-state teaching, not a full continuously refreshed DAgger program. The original [tier scheduler](/home/stepan/Projects/code/Pebby/pebby/agent/curriculum_sampling.py:150), lines 150–160, interpolates weights with elapsed training progress; [recovery continuation](/home/stepan/Projects/code/Pebby/tools/train_spatial_recovery_comparison.py:163), lines 163–201, uses uniform levels with reserved replay. Neither is demonstrated primitive mastery scheduling.

## Comparison: what would actually change

| Method | New teaching signal or distribution | Fit to Pebby and main failure mode |
|---|---|---|
| Primitive curriculum | Controlled examples isolate navigation, matching, obstacles, one transformation, then compositions. | Tests missing coverage; overly simple training can damage complex behavior or teach shortcuts. |
| Automatic curriculum | Choose task families from measured learning progress; retain a coverage floor. | Changes sampling, not model intelligence; loss decline can favor easy auxiliaries over gameplay. |
| Repeated expert relabeling | Roll out each updated learner, label its new states, aggregate experience. | Builds on existing collectors; fails if the teacher labels unobservable distinctions or merely repeats attractors. |
| Goal/subgoal supervision | Explicit matching predicates, next required operation, waypoint or goal-conditioned value. | More specific than the existing global distance target; auxiliary correctness may remain unused by the actor. |
| Algorithmic/process teaching | Supervise intermediate route propagation or state transitions along a computation. | Specifies how to derive an answer; incorrect state abstraction or teacher-only inference inputs invalidate the gain. |
| Sequence imitation | Train contiguous demonstrations while preserving recurrent state/gradients; optionally predict short action sequences. | Distinct from shuffled H8 decisions; teacher-forced success can fail when predictions feed back. |
| Online RL | Update from consequences of the learner's own executed actions and episode returns. | Direct behavior optimization; exploration, credit assignment and observation limits remain. |

## Owning-primary evidence packets

### P1 — DAgger targets the learner's evolving state distribution

- Claim/status/confidence: **Established, high**. DAgger iterates collection under the current policy, expert labeling, dataset aggregation and supervised refitting.
- Evidence/source tier: **owning-primary**. [Ross, Gordon and Bagnell, Algorithm 3.1 and §3, printed pp. 629–631](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf).
- Revision/retrieval: AISTATS 2011 proceedings; retrieved 2026-09-13.
- Scope/limits: The guarantees require the stated reduction/learner assumptions; they do not guarantee this neural architecture can fit a policy or recover information absent from H8. “DAgger already failed conclusively” and “DAgger is entirely new here” are both unsupported.

### P2 — Adaptive curricula need a uniform comparator

- Claim/status/confidence: **Established, high**. Graves et al. use a nonstationary bandit driven by learning-progress signals. Some signals improve training; others underperform uniform task sampling.
- Evidence/source tier: **owning-primary**. [Graves et al., §§2–3, §4.2–4.3 and §5, PDF pp. 2–8](https://proceedings.mlr.press/v70/graves17a/graves17a.pdf).
- Revision/retrieval: ICML 2017 proceedings; retrieved 2026-09-13.
- Scope/limits: LSTM experiments on their three curricula, not LS20. The bAbI experiment reports training performance after generating a large corpus; it is not evidence for unseen LS20 composition. This supports comparing adaptive scheduling with uniform sampling, not presuming a benefit.

### P3 — Goal-conditioned values can be supervised directly

- Claim/status/confidence: **Established, high**. UVFAs represent value jointly as a function of state and goal; the paper includes supervised fitting and tests unseen goal locations.
- Evidence/source tier: **owning-primary**. [Schaul et al., §3.1 and §4.2–4.3, PDF pp. 3–6](https://proceedings.mlr.press/v37/schaul15.pdf).
- Revision/retrieval: ICML 2015 proceedings; retrieved 2026-09-13.
- Scope/limits: Their goal-transfer results do not establish compositional planning. The LS20 proposal is paired training on valid scenes with different goals, using goal data derived from public pixels; it does not require adopting their matrix-factorization implementation.

### P4 — Gridworld teaching is feasible but curriculum and RL are not free wins

- Claim/status/confidence: **Established, high**. BabyAI trains recurrent imitation policies with truncated backpropagation and compares PPO. Its tested IL settings use fewer episodes than RL; useful pretraining depends on the chosen base competencies.
- Evidence/source tier: **owning-primary**. [BabyAI, §4.1, Tables 3–4 and §4.3–4.4, printed pp. 7–10](https://arxiv.org/pdf/1810.08272v4).
- Revision/retrieval: arXiv v4, 2019-12-19; retrieved 2026-09-13.
- Scope/limits: Grounded language and different actions, observations and models. Episode counts do not equal annotation or compute cost. Its interactive protocol selects failed missions, which is different from DAgger labeling intermediate learner states. It supports the relevance of controlled competencies and recurrent imitation, not a transferable LS20 sample requirement.

### P5 — Process supervision has a concrete algorithmic meaning and a deployment caveat

- Claim/status/confidence: **Established, high**. CLRS records intermediate algorithm states as hints. Its baselines predict and feed back hints; the published setup uses the number of hints to determine evaluation computation length.
- Evidence/source tier: **owning-primary**. [Veličković et al., §3.2, §4.1, and Appendix C Bellman–Ford example, PDF pp. 5, 7 and 14](https://proceedings.mlr.press/v162/velickovic22a/velickovic22a.pdf).
- Revision/retrieval: ICML 2022 proceedings; retrieved 2026-09-13.
- Scope/limits: Algorithm traces on structured inputs, not game pixels. An LS20 experiment must choose a fixed deployment-valid computation budget or learn termination. Supplying oracle path length at inference would test a different system. Generic verbal chain-of-thought is unnecessary.

## Smallest discriminating experiments — proposals only

**1. Primitive teaching before new architecture.** Use one common semantic-policy architecture and identical initialization, trainable scope and primitive-compatible policy/distance loss. Compare staged primitive teaching against the **same examples shuffled**, with equal complex replay and total updates. Stage empty-room/matching-goal navigation, obstacles, then one glyph change; reserve 50% complex replay in both arms. Start with 2,048 training rooms, 256 validation and 512 sealed test rooms, grouping opposite-goal pairs within partitions. Two seeds, 400 updates and batch 256 per arm are a bounded probe. A complex-only control would test added coverage separately. Record actual cost before extending; these counts are not a promise of mastery.

Use natural, engine-valid opposite-goal scenes as the key control; simple input masking is weaker evidence because it changes the distribution. Measure both members of each pair, optimal-set first actions, wins from reset, route inefficiency and complex-task regression. A proposed primitive admission target is ≥95% first-action correctness and ≥95% closed-loop success within twice oracle route length on fresh validation. A tiny semantic policy using public-decoded coordinates is a diagnostic positive control for input/label learnability, not a candidate for claiming seven-level success. Failure to fit training examples directs the next check toward optimization/interface/capacity; fitting without transfer directs it toward coverage and inductive bias.

**Required prerequisite:** The current [weight builder](/home/stepan/Projects/code/Pebby/pebby/agent/spatial_outcome_objective.py:18), lines 18–33, rejects datasets without both changed/unchanged glyph examples and positive events. Pure empty rooms intentionally lack these. A dedicated primitive path can optimize native optimal-action imitation plus exact remaining-distance classification, masking zero-optimal rows only for policy loss. Both schedule arms must use that same objective. Alternatively preserve documented original auxiliary weights and explicitly support absent families; do not launch the existing full trainer unchanged on this bank.

**2. Test refresh, not more of the same replay.** After primitive competence, compare three collect/refit rounds against one static aggregate with equal optimizer updates, labeled-root budget and expert-query accounting. Proposal: 128 TRAIN levels, at most 64 learner actions per round, 200 updates per round, two seeds; keep at most the existing three short recovery contexts per level. Refresh from the newly fitted policy, retaining old examples and action-set labels. Report unique histories, policy-valid roots, teacher search failures and wins. If the new state distribution barely changes, more DAgger rounds cannot answer whether refresh matters. Zero optimal masks must not become invented action labels; life-reset successor labels are not voluntary RESET teaching.

**3. Separate curriculum content from ordering.** After the primitive test, build a fixed union of obstacle, single-attribute, two-attribute and resource/refill families. Compare uniform sampling, a mastery-based schedule with replay, and a learning-progress schedule, using the same total examples and updates. Initially cap each at three blocks of 400 updates for two seeds. Include held-out combinations of individually seen mechanics and larger route lengths. Reserve ≥20% old-family replay as a proposed forgetting control. Neither increased generator difficulty nor a training-loss threshold alone counts as mastery. Restoration/disappearing-goal mechanics require newly generated verified data; retained caches lack them.

**4. Add goal/subgoal targets only when the previous test identifies their need.** On opposite-goal and one-transformation scenes, compare native policy loss alone against the same policy plus supervised match predicates and next-required-operation/waypoint predictions that the actor actually consumes. Keep an equally sized unsupervised auxiliary-head control. Keep all valid next subgoals where paths tie. Stop at the same 400-update/two-seed probe. Better subgoal accuracy without better actions rejects the claimed behavioral mechanism; correct oracle subgoals substituted diagnostically measure an upper bound, not deployable performance.

**5. Escalate temporal/process teaching selectively.** For route reasoning, test intermediate reachable-frontier or distance-relaxation targets on static obstacle rooms before augmented glyph/resource state graphs. Compare final-answer-only supervision with intermediate supervision using identical computation and parameters; evaluate beyond trained path lengths without oracle-derived iteration counts. For persistent state, compare contiguous sequence imitation against shuffled decisions only after a recurrent state API exists. Otherwise sequence batching leaves the retained H8 information boundary unchanged. A proposed first budget is 256 training trajectories, 64 validation trajectories and two seeds, with equal supervised action tokens. Long hidden-state tasks must not be used to declare failure of a model deliberately lacking memory.

**6. RL as a later bounded comparison.** From the same competent imitation checkpoint, compare frozen-policy evaluation, further imitation, and online RL on matched primitive/composition families. Proposal: at most 100,000 real-engine actions per seed initially, two seeds, plus logged optimizer compute. Audit reward, termination, legal-action and history handling first. Report success per environment step and total cost. Sparse successes or no measured learning are a reason to reassess the pilot, not proof of an RL ceiling. Any shaped-reward variant needs an unshaped terminal-success evaluation and a control separating shaping from RL itself.

## Verification, search record and remaining uncertainty

Fresh adversarial self-review: L1/L2 and P1–P5 **verified within scope**; all LS20 effectiveness predictions **qualified, empirical unknown**. Specifically rejected: “no action teaching exists,” “DAgger is absent,” “goal decoding proves goal use,” “curriculum always helps,” “hint supervision alone proves deployable planning,” and “a short failed fit establishes a capacity ceiling.” This worker review was not agent-independent; the integrator should retain these qualifications.

Library-first retrieval covered the research and ML indexes, Pebby index/boundary note and shared run brief. Web search used **2 of 4 query rounds**, **5 owning primary works**, no secondary evidence: round 1 searched DAgger, automatic curriculum, BabyAI and CLRS; round 2 located UVFA. OpenReview's BabyAI PDF redirected to a browser challenge; the exact arXiv v4 PDF supplied the primary text. All five final URLs were reopened, and cited page/section locators checked. No new canonical note, project implementation, branch commit, or long-running process was created.

The unresolved decision is empirical: whether controlled teaching fixes goal dependence in the existing model. Published papers cannot answer that. No proposed experiment establishes all seven sequential levels until evaluated from reset under the actual public-input contract.

### Local revision hashes

All paths refer to the dirty working repository `/home/stepan/Projects/code/Pebby`, read 2026-09-13.

| File | SHA256 |
|---|---|
| `MODEL_STATUS.md` | `f6d1012ed26e5d44fff0799c4438c4207d93392d06356dabcdf583dd12f804cc` |
| `artifacts/spatial-repair-v5/REPORT.md` | `cdab239f896bdf055e9908ff18d62797801d0d5615f2815a786f1b7859798904` |
| `pebby/agent/neural_outcome_planner.py` | `8dd99aeaa18f0bf420af51a2ec5573217338023113204595803fff1861c45dc6` |
| `pebby/agent/spatial_outcome_objective.py` | `1b95248d6462c6bb6d548f8d0b02cb32d4a1af50e37ef4d841296431e97f66ed` |
| `tools/collect_spatial_recovery.py` | `2be197c752839a16c449c15d96ed2eadb758ce5ab3316420470dfc2a86332a80` |
| `tools/collect_spatial_repair_v3.py` | `f303ae966d1893be5754165ba12f69e8b4364753525fe4211c372db3c7867f14` |
| `pebby/agent/curriculum_sampling.py` | `c6285c3ffd6246571e63818397c24e5a73c3fd7d7a9daf20ec0a3423a1f8197a` |
| `tools/train_spatial_recovery_comparison.py` | `28520da1b26a5c1ab99489e04e5ec4a0202feedf6120aa2a4a8437ba3f553a8d` |
