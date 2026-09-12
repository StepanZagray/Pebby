# Generated-level and collection fixes

This document records the historical five-tier v2 mechanism supplement. Current
public generation uses the [seven LS20 reference profiles](seven-reference-difficulties.md).
The files and measurements below retain their original five-tier meaning.

This revision follows T3 thread `1187f7d9-ebae-4177-aa8e-0b67f6d4b59b`, including
its corrections about long-route memorization, context-specific proofs and
failure supervision. Existing level banks, arrays and checkpoints do not acquire
these fixes automatically.

## Generation contracts

Lessons generated under this historical v2 contract have a complete contextual Oracle search and a winning replay
in the actual engine with three lives. The stored route and distance use the
same `seed % 7` context as collection. Optimal ties are randomized using a
per-level seed; distances and full optimal-action masks remain exact. Small
generated fixtures additionally compare native, reference and independent BFS
optimal lengths. A search cap is an exclusion, never a successful full proof;
reports preserve rejection reasons and counts.

The sprite generator is version 3; the mechanism and extended curricula are
version 2. Historical sprite-version-2 banks remain structurally readable, but
reading a file does not validate its proof. Legacy collection replans stale
routes, and extenders require current aligned contextual proof metadata.

Mechanism and extended learning lessons require at least ten actions and eight spare charged moves
throughout the verified route, including before refill resets and the charge
before a launcher landing. Explicit `challenge` lessons permit zero margin,
with complete search and a life-preserving winning route still required. This
separates recovery practice from tight-budget/refill-chain practice. Headroom
alone cannot guarantee recovery from arbitrary errors or patroller phase changes.

New mechanism and extended rows use the real initial budget of 42 and costs
1 or 2. Row 10 is available except reserved HUD cell `(1,10)`. Every goal differs
from the initial carried triple. Extra cyclers use a non-required kind when one
exists and must actually be avoided by the verified route. With all three kinds
required, unused extra cyclers are alternatives, not a nonexistent fourth kind.

## Coverage and remaining bounds

Mechanism quotas include one/two/three attributes, three/four goals, varied long
rails and rings, three distinct moving kinds that must all be contacted, three
distinct used launcher pads, mixed rails/launchers/goals, an eight-pad network,
long navigation, and cost-2 refill chains consuming at least three tanks.
Installed and actually used mechanic counts are reported separately.

The old fixed serpentine has been replaced by randomized tree mazes with varied
starts, goals, branches, lattice symmetries and refill placements. Long-route
lessons require at least 72 actions and targets needing two or three presses.
They have a much smaller state space than freely composed large rooms, but
remain a procedural topology family, not the full diversity of hand-designed
levels.

The 600,000-state cap still excludes combinations approaching the largest
official state spaces. Four-goal lessons use compact rooms, color/rotation and
cost 1. Three-moving-kind mechanism lessons use synchronized short rails.
Extended learning tiers also bound high-attribute room and rail complexity;
the challenge profile proposes larger mechanic counts, but a proposal is not
proof that every maximum-count combination is accepted. Difficulty names are
lesson categories, not a claim that empirical medians increase monotonically.

A deterministic hash partitions translation-normalized geometry between new
train and validation banks. Moving a room cannot change its split. This does
not hold out topology families or retroactively remove overlap with old banks.
Use at least 500 validation levels with matching curriculum quality versions.

`tools.audit_generated_banks` checks proof consistency, split leakage, version
alignment, actual coverage and long-route diversity. It reports action marginals
and a previous-action baseline descriptively: route autocorrelation by itself
does not establish that an observation-aware policy learned nothing. Optional
spotchecks repeat the generator's planner search and replay stored routes in the
real engine, independently of their randomized tie choices. Planner agreement
alone is not an independent optimality proof. The current auditor also checks
D4 overlap; the historical translation-only split below does not satisfy that
stronger holdout merely because its original audit passed.

## Regeneration and historical repair

```bash
uv run python -m tools.regenerate_mechanism_banks \
  --profile legacy-mechanism-v2 --train-count 2004 --validation-count 504 --workers 4 \
  --out-dir data/mechanism-bank-v2
uv run python -m tools.audit_generated_banks \
  --train data/mechanism-bank-v2/train.jsonl \
  --validation data/mechanism-bank-v2/validation.jsonl \
  --report artifacts/mechanism-bank-v2-audit.json --spotcheck 2
```

These are explicit legacy commands. New reference-calibrated generation uses the
[seven-tier workflow](seven-reference-difficulties.md#entry-points-and-durable-generation).
The historical `tools.extend_curriculum_bank` and `tools.extend_extended_bank`
commands now require `--legacy`; their existing banks are not seven-tier inputs.
Rebuilding legacy proposals does not guarantee that the current D4 audit passes.

The rebuilt bank at `data/mechanism-bank-v2` contains 2,004 training and 504
validation levels, with 167/42 examples of each mechanism mode. The audit found
no shared seeds, gameplay specifications or translation-normalized geometries
between these splits. Training includes 167 distinct long-route geometries with
73–100-action solutions; learning routes all retain at least eight spare
charged moves, while challenge routes can use their entire budget. The bank's
167 examples in each of difficulties 1–3 are insufficient on their own for
some 1,024-distinct-level curriculum schedules; these are mechanism supplements,
not a drop-in replacement for every existing base-curriculum schedule.
The subsequent independent census confirmed winning replays and minimal action
counts for all 2,508 historical rows, with no disagreements. That result certifies
those rows; it does not automatically certify later generator outputs.

Regeneration refuses an existing output directory unless `--resume` is explicit
and its configuration, source hashes and input inventory match. It records every
worker PID, bounds time/search/attempts, excludes existing seeds and gameplay,
and checkpoints successful jobs before publishing completed split quotas. It does
not replace running training inputs or retrain a model.

The 43 training and eight validation stored routes known to be invalid in their
recorded contexts were repaired in `data/ls20-context-repaired-{train,validation}.jsonl`.
`artifacts/ls20-context-route-repairs.json` records source/output hashes and every
replacement proof. Original files and the other 12,243 rows are preserved; those
unchanged rows were not freshly searched and do not become v2-quality lessons.

## Training-data consumption

Use `mixed_failure` collection for new world-model archives. It adds real
three-life exhaustion trajectories and route-proportional expert anchors, with
zero optimal-action masks only for hopeless states. Policy objectives mask
those rows while dynamics and outcome objectives retain them. On-policy
supplemental expert/failure rows have an explicit auxiliary sampling path.
See [world-failure-coverage.md](world-failure-coverage.md) for the tested pipeline,
event/value semantics, sampling counts and architecture limitations.

Current training losses live in `pebby/agent/world_training_objectives.py`, with
their source hash recorded in new checkpoints and reports. The frozen encoder
source remains byte-identical to existing checkpoint bindings; strict source
validation remains enabled.

Rebuilt transition archives are `data/ls20-world-mechanism-v2-train.npz` and
`data/ls20-world-mechanism-v2-validation.npz`. Their source hashes, measured
failure counts and full collection coverage are recorded in
`artifacts/world-mechanism-v2-regeneration.json`; the checked 12-level pilot
independently verifies every retained action branch. Structured caches must
still be rebuilt from the new data before training.
The existing default distance head still saturates at 63; overflow is now
reported. Hint-on context coverage is restored where the exact oracle supports
it, but context remains provenance rather than a privileged model input.
Check distinct-level coverage before enabling a 1,024-level curriculum batch;
the old roughly 205-per-difficulty on-policy subset cannot satisfy the default
late-stage requirement of 410 distinct difficulty-5 levels by itself.

These changes validate data generation and consumption. They do not establish
improved safety recall, long-horizon completion, or a new official-level score
without training and evaluation.
