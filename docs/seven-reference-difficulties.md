# Seven LS20 reference difficulties

New public generation uses `difficulty_version: ls20-reference-v1`. Difficulty
**N corresponds to LS20 level N**, including its actual engine context N−1.
These are seven reference profiles, not an assertion that action length increases
monotonically. Legacy five-tier metadata is preserved as a different contract;
old data is never silently relabeled.

| Tier | Reference optimum | Accepted optimal actions | Free cells | Goals | Launchers | Refills | Moving cyclers | Cost | Fog |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| 1 | 13 | 10–17 | 32–44 | 1 | 0 | 0 | none | 1 | no |
| 2 | 45 | 36–54 | 49–65 | 1 | 0 | 2 | none | 2 | no |
| 3 | 39 | 31–47 | 57–73 | 1 | 2 | 2 | none | 2 | no |
| 4 | 43 | 34–52 | 59–75 | 1 | 8 | 2 | none | 1 | no |
| 5 | 44 | 35–53 | 61–77 | 1 | 8 | 3 | rotation, 3-cell line | 2 | no |
| 6 | 72 | 58–86 | 61–77 | 2 distinct | 2 | 3 | shape/rotation 5-cell lines; color 8-cell ring | 1 | no |
| 7 | 53 | 42–64 | 58–74 | 1 | 3 | 6 | rotation, 6-cell line | 2 | yes |

Every level starts with the real budget of 42. Required changed attributes are
rotation for tiers 1–2; color and rotation for tier 3; shape and color for tier 4;
all three for tiers 5–7. Cycler counts follow those references: the additional
non-required distractors in historical mechanism supplements are not mandated
where the official reference has none. Moving cyclers must all be exercised.
Verified routes must use at least 2/5/4 distinct launcher pads in tiers 3/4/5,
and at least one in tiers 6–7. Installed counts are not reported as used counts.
The tier-6/7 usage floors are conservative proxies: their official route-use
counts were not measured in the reference characterization.
Tiers 2–7 must consume at least 2/1/1/1/1/2 refill tanks respectively.

The bounds are explicit calibration tolerances around one official reference per
tier, not estimated confidence intervals. Tier 1 uses a 7–9-cell bounding extent;
other tiers use 9–10. Each tier also gates corridor fraction (free cells with at
most two orthogonal free neighbors), preventing sparse mazes from substituting
for large rooms merely by having a similar route length. Coordinates, connected
geometry, targets, rail positions and launcher positions are newly sampled.
The generator does not rotate or copy a shipped layout or reuse its solution.
The aggregate source measurements and their hashes are recorded in
`artifacts/official-seven-tier-characterization.json`.

Complete contextual Oracle searches certify optimal lengths and labels. Caps are
600k/600k/1M/2M/4M/24M/32M states for tiers 1–7, allowing the larger state spaces
required by the later references. Incomplete searches remain exclusions, with
reasons reported. The real engine must replay the full winning route with all
three lives; intermediate state agreement is checked. Tier 1 retains eight spare
charged moves. Other tiers are explicitly challenge profiles and permit zero
margin. Official level 5 can legally win at negative final budget because winning
precedes exhaustion; this generator currently selects nonnegative-margin routes
within the reference action range. It does not reproduce that final-action edge
case's frequency or claim identical error-recovery behavior.

Geometry is partitioned under translation, all rotations and reflections.
Balanced per-tier quotas and independent seeded final shuffling replace the
periodic mode schedule. The irregular proposals avoid allocating a tiny handful
of rectangle shapes wholesale to one split. Audits still report measured
structural distributions; a common range does not by itself prove identical
train/validation distributions or hold out whole topology families.

## Entry points and durable generation

`pebby.ls20.generate.generate_level`, `curriculum.generate_level`, and
`extended_curriculum.generate_level` default to the seven-tier contract. The
extended API retains its `(row, exclusions)` return shape. Historical algorithms
are retained only as explicitly named `generate_legacy_level` APIs; historical
extension commands require `--legacy`. Their old floor/bounds behavior is not
presented as current generation quality.

```bash
uv run python -m tools.regenerate_mechanism_banks \
  --profile ls20-reference-v1 --train-count 196 --validation-count 56 \
  --workers 2 --out-dir data/ls20-reference-v1
# Resume with the same configuration, source code and input inventory:
uv run python -m tools.regenerate_mechanism_banks \
  --profile ls20-reference-v1 --train-count 196 --validation-count 56 \
  --workers 2 --out-dir data/ls20-reference-v1 --resume
```

These examples build a calibration pilot, not the recommended ≥500-level final
validation bank. Later tiers need substantially more CPU and memory per level;
two workers are the maximum for this profile. Each accepted job is saved in an
atomic, checksummed checkpoint before it counts toward progress. Ordinary bounded
rejections and duplicates retry inside reserved seed blocks. Contract disagreements
are quarantined for investigation; successful sibling jobs are retained. Completed
banks are shuffled reproducibly. Incomplete banks are not mislabeled complete.

The auditor now gates profile coverage, geometric diversity, nested proofs and
cross-split D4 overlap. Its default seven spotchecks per split are stratified across
tiers. Same-planner re-search checks reproducibility and reachable counts; engine
replay checks actual wins. Neither is described as an independent optimality proof. A separate
`tools.audit_independent_ls20` command uses a rules transcription independent of
`plan`, `fastplan` and `layout`, validates it against the engine, then performs
bounded BFS. Its budget-dominance optimization has explicit domain guards; plain
BFS comparison is available with `--plain-bfs`. State/time/memory caps are reported
as inconclusive, never as proofs.
Use `--generation-report` to bind rejection evidence to the exact bank hashes.
Small calibration runs must explicitly lower `--min-validation`; sample-only
coverage policy is for incomplete fixture collections, not a production exemption.

## Training and caches

Calibrated proofs carry both the difficulty version and tier through collection,
cache building and sampling. Collection honors the declared complete-search budget
(up to 32M), rather than dropping tiers 6–7 at the historical 600k default. Context
is provenance, not privileged policy input. General world and structured training
paths support seven tiers. Mixed legacy/calibrated contracts are rejected rather
than silently interpreted as the same labels. Frozen historical experiments with
fixed five-tier data refuse calibrated inputs and identify configurable alternatives.

Existing arrays, structured caches and checkpoints remain historical until rebuilt
and retrained. In particular, the earlier `mechanism-bank-v2` and its 51,515
transition states are five-tier mechanism supplements, not seven-tier reference
training data. The default value head still saturates reachable distances at 63.
No improvement in learned gameplay is established by generator and pipeline tests.

## Verified calibration pilot

`data/ls20-reference-calibration-v1/{train,validation}.jsonl` contains 28 training
and 14 validation levels: four and two per tier respectively. The full coverage
audit passes with zero shared seeds, gameplay hashes or geometry hashes, including
rotations and reflections. Independent BFS proves the stored optimum for all 42
levels. Engine comparisons cover 1,766 solution actions, 1,344 random off-route
actions and 444 controlled exhaustion actions, including 42 life losses, with no
transition mismatches. Evidence is in
`artifacts/ls20-reference-calibration-{audit,independent}.json`.

| Tier | Pilot optimal actions: min / median / max |
|---|---|
| 1 | 10 / 12.5 / 17 |
| 2 | 36 / 42 / 50 |
| 3 | 35 / 36.5 / 39 |
| 4 | 36 / 39 / 41 |
| 5 | 36 / 49 / 53 |
| 6 | 60 / 69.5 / 85 |
| 7 | 44 / 45 / 51 |

`data/ls20-reference-calibration-world-{train,validation}.npz` contains 622 and
317 transition rows, preserving every requested level and all seven contexts.
Production loading, proof validation, winning coverage and seed-disjointness
guards pass. Actual life-loss branches number 330/168 and terminal-death branches
109/56. These are small calibration archives, unsuitable for 1,024-distinct-level
batches or production-performance conclusions.

Exact successor distances are retained up to 80/90 in train/validation. The
unchanged default 64-bin value head would saturate 122/99 targets (4.9%/7.8%).
For fresh world-model training, `pebby.agent.world_train --max-distance 128`
supports exact finite distances 0–127 plus a separate unreachable bin. This
covers the observed pilot distances; it is not a bound on all future states.
Changing that head size requires a fresh compatible model: existing checkpoints
retain their saved configuration and cannot initialize a differently sized head.
The archive build did not encode feature caches or train a model. Its detailed
provenance and verification are in `artifacts/reference-calibration-world-build.json`.
