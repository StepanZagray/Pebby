# Generated-level and collection fixes

This revision follows T3 thread `1187f7d9-ebae-4177-aa8e-0b67f6d4b59b`, including
its corrections about long-route memorization, context-specific proofs and
failure supervision. Existing level banks, arrays and checkpoints do not acquire
these fixes automatically.

## Generation contracts

All new lessons have a complete contextual Oracle search and a winning replay
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

Learning lessons require at least ten actions and eight spare charged moves
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
spotchecks re-search the optimum and replay stored routes independently of
their randomized tie choices.

## Regeneration and historical repair

```bash
uv run python -m tools.regenerate_mechanism_banks \
  --train-count 2004 --validation-count 504 --workers 4 \
  --out-dir data/mechanism-bank-v2
uv run python -m tools.audit_generated_banks \
  --train data/mechanism-bank-v2/train.jsonl \
  --validation data/mechanism-bank-v2/validation.jsonl \
  --report artifacts/mechanism-bank-v2-audit.json --spotcheck 2
```

Regeneration refuses an existing output directory, records every worker PID,
bounds time/search/attempts, excludes existing seeds and gameplay, and publishes
only after both quotas and source-hash checks pass. It does not replace running
training inputs or retrain a model.

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

Rebuild arrays and any structured caches from the new banks before training.
The existing default distance head still saturates at 63; overflow is now
reported. Hint-on context coverage is restored where the exact oracle supports
it, but context remains provenance rather than a privileged model input.
Check distinct-level coverage before enabling a 1,024-level curriculum batch;
the old roughly 205-per-difficulty on-policy subset cannot satisfy the default
late-stage requirement of 410 distinct difficulty-5 levels by itself.

These changes validate data generation and consumption. They do not establish
improved safety recall, long-horizon completion, or a new official-level score
without training and evaluation.
