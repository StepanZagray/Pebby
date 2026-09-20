# Full-standard multi-game generators

Pebby keeps one generator per public training family at `pebby/games/<slug>/generate.py`.
Rules stay family-native: shared code validates contracts, seeds generation, constructs sequential
games, and records evidence; it does not replace 24 engines with one symbolic generator.

Admission changes as reviews close. Treat
[`STATUS.md`](../.scratch/multigame-resume/full-standard/STATUS.md) as authoritative for current
family readiness and
[`ACCEPTANCE-CAVEATS.md`](../.scratch/multigame-resume/full-standard/ACCEPTANCE-CAVEATS.md) as
authoritative for accepted limitations. Run the current preflight before collection rather than
inferring readiness from this guide or a historical snapshot.

## Family interface

An accepted family exports:

```python
DIFFICULTIES = (1, ..., N)

generate(seed, difficulty, ..., split=...) -> dict | None
generate_game(seed, *, split=..., difficulties=None, ...) -> list[dict] | None
build_level(spec) -> native_level
build_game(specs) -> list[native_level]
validate_full_standard(spec, curriculum_entry) -> list[str]
FULL_STANDARD_CONTRACT = {...}
```

`N` is the exact official level count. With `difficulties=None`, `generate_game` must generate
exactly difficulties `1..N` in increasing native contexts `0..N-1`. A shorter sequence is
inspection/smoke data, not a full game. `build_game` accepts only the exact ordered curriculum,
revalidates independently generated specs, and reconstructs them without shifting contexts.

Each accepted spec is JSON-ready and binds difficulty, native context, split, generated identity,
stored action witness and length, and family proof/mechanic metadata. Other fields are
family-specific. Reconstruct with that family's `build_level` or `build_game`.

Here is a bounded, tested single-level CD82 example:

```python
from pebby.games.cd82 import generate as cd82

spec = cd82.generate(7, 1, attempts=120, limit=200_000, split="train")
assert spec is not None
assert cd82.validate_full_standard(spec, cd82.FULL_STANDARD_CONTRACT["curriculum"][0]) == []
level = cd82.build_level(spec)
```

The corresponding whole-game API defaults to all six CD82 tiers:

```python
specs = cd82.generate_game(0, split="train")
assert specs is not None
assert [row["difficulty"] for row in specs] == list(cd82.DIFFICULTIES)
levels = cd82.build_game(specs)
```

Whole-game generation may be expensive. For an interface check, use one low tier or family tests.

## Bounds, rejection, and diagnostics

Optional names are not universal. Families variously expose `attempts`, `limit`, `node_limit`,
and sometimes `search_limit`, `search_work`, or an action cap. Inspect its signature or bank help.

The collector passes `attempts` when present and then `node_limit`, otherwise `limit`. Its bounds are:

- `--max-generation-attempts`: deterministic outer seed attempts per level;
- `--generator-attempts`: inner family draft attempts when supported;
- `--max-search-work`: global ceiling, default 2,000,000 for smoke and
  32,000,000 for full collection;
- `--max-actions-per-level` and `--max-game-steps`: native rollout caps.

Each tier declares its own `search_work`; full collection rejects a ceiling below any requested
tier. A normal bounded rejection returns `None`; malformed inputs and broken proof/engine
invariants may raise. Diagnostics are family-specific: `stats`, `record_rejection`, `diagnostics`,
or `*.last_report`; bank CLIs may use `--max-seeds`, `--attempts`, `--max-attempts`, or `--node-limit`:

```bash
uv run python -m pebby.games.cd82.bank --help
uv run python -m pebby.games.cd82.bank \
  --levels 1 --seed 7 --difficulty 1 --split train --out /tmp/cd82-d1.jsonl
```

The collector never retries without a bound. It retains failures, partial metadata, search flags,
and status. Preparation records failed candidates, duplicates, split overlaps, and exhausted
slots; an incomplete preparation remains explicitly incomplete.

### LS20's evidence boundary

LS20 maps arbitrary requested seeds into finite native split ranges, so audit both requested and
effective identities. Tier caps are 600k, 600k, 1M, 2M, 4M, 24M, and 32M. Accepted high-tier
evidence is frozen v4-bank proof replay plus current adapter/validator/native replay, not fresh
24M/32M searches. It certifies stored rows and wiring, not newly re-proved optimality.

## Certified collection versus recovery

For full-standard collection, the authoritative teacher at level start is the validated spec's
stored `solution`. The collector enforces the cap and clone-replays it from the actual sequential
state before real execution. These actions are labelled `certified_spec_solution`.

After a random legal action, the certificate is invalidated and the family planner searches from
the actual resulting state. Those teacher actions are `live_recovery`; the random transition is
`random_action`. Recovery does not inherit certified mechanic-use claims. A record sets
`certified_solution_completed` only for unperturbed certified completion; smoke may use `live_search`.

Public arrays live in `games/*.npz`. Private teacher targets and route-source
labels live in `teacher/*.npz`; generated specs live beside them as
`*.levels.json`; status, limits, identities, errors, and code provenance live
in `records/*.json` and `manifest.json`. The public timeline is authoritative
if an engine mutates and then fails while producing a successor.

## Split and provenance identities

Shared full collection requires an explicit `train`, `validation`, or `test` split.
The pipeline records requested seeds, effective seeds where applicable,
family-authored geometry/gameplay/puzzle hashes, canonical spec fingerprints,
and variant-undone raw initial-frame hashes. Validation candidates matching any
known training identity are rejected. Complete whole-game semantic fingerprints
are also deduplicated within each cohort. Smoke mode can fall back to a stripped
canonical-spec identity; production full-standard data requires authored
semantic identities.

These checks are conservative controls, not population-equivalence proofs.
Initial frames can miss hidden state, canonical specs can over- or under-match,
and D4 canonicalization does not establish arbitrary graph isomorphism.
Likewise, a native positive witness proves solvability in its stated context;
it does not establish universal optimality or impossibility outside a bounded
search.

Manifests bind hashes for the shared collector, each family module, vendored
game source, installed engine packages, full-standard contracts, records, and
artifacts. Preserve those files with any result.

## Official curriculum cardinalities

This table is copied from the verified
`official-curriculum-cardinalities.json` artifact. Every row is a required
contract, not a readiness claim.

| Family | Official N | Required difficulties | Native contexts |
|---|---:|---|---|
| LF52 | 10 | 1..10 | 0..9 |
| S5I5 | 8 | 1..8 | 0..7 |
| LP85 | 8 | 1..8 | 0..7 |
| WA30 | 9 | 1..9 | 0..8 |
| AR25 | 8 | 1..8 | 0..7 |
| RE86 | 8 | 1..8 | 0..7 |
| R11L | 6 | 1..6 | 0..5 |
| SC25 | 6 | 1..6 | 0..5 |
| FT09 | 6 | 1..6 | 0..5 |
| LS20 | 7 | 1..7 | 0..6 |
| CN04 | 6 | 1..6 | 0..5 |
| KA59 | 7 | 1..7 | 0..6 |
| TU93 | 9 | 1..9 | 0..8 |
| TR87 | 6 | 1..6 | 0..5 |
| DC22 | 6 | 1..6 | 0..5 |
| SP80 | 6 | 1..6 | 0..5 |
| CD82 | 6 | 1..6 | 0..5 |
| TN36 | 7 | 1..7 | 0..6 |
| VC33 | 7 | 1..7 | 0..6 |
| BP35 | 9 | 1..9 | 0..8 |
| SB26 | 8 | 1..8 | 0..7 |
| SU15 | 9 | 1..9 | 0..8 |
| G50T | 7 | 1..7 | 0..6 |
| SK48 | 8 | 1..8 | 0..7 |

## Collection and dataset preparation

A safe labelled smoke subset can run before 24/24 acceptance:

```bash
uv run python tools/collect_multigame_games.py \
  --games cd82 --difficulties 1 --games-per-source 1 --seed 7 \
  --max-search-work 200000 --out-dir /tmp/pebby-cd82-smoke
```

Omitting `--games` switches to the full protocol: all 24 packages are
preflighted with accepted contracts before the output directory is created,
and each family uses its own complete curriculum. Do not pass `--difficulties`
in full mode. The preferred experiment-data entry point is:

```bash
uv run python tools/prepare_multigame_dataset.py \
  --output-root data/multigame-v1 \
  --train-master-seed 1000 --validation-master-seed 2000 \
  --completed-teacher-games-per-family 1 \
  --mixed-games-per-family 1 --mixed-epsilon 0.2
```

Run that only after 24/24 admission and after the experiment owner chooses the
actual corpus size and compute budget; the defaults are wiring values, not a
scientifically sufficient design. For a subset, `--games ...` requires
`--smoke`. Output is split into train/validation teacher cohorts and optional
mixed cohorts, followed by the same strict manifest audit used by training.

Training data is generated-only and covers the 24 public training families.
Official levels, layouts, frames, trajectories, routes, answers, calibration
artifacts, and reference artifacts are excluded from model data. Official
source mechanics and aggregate reference statistics informed generator
implementation and calibration only. `m0r0` is outside the public 24-family
generator and training registries: it is excluded from generated collection,
dataset preparation, training, generated validation, and training-family
official evaluation. The evaluation adapter exposes it only through the
separately gated heldout phase after the frozen 24-family official report.

Generated validation selects and freezes the checkpoint/configuration. Only
then does evaluation run on the 24 official training-family games, followed by
the separately held-out official `m0r0` stage, with no tuning between stages.
Human developers previously inspected `m0r0`, so the holdout claim is model-
data/tuning exclusion, not an unseen design to humans.

The existing GRU imitation baseline with a one-step auxiliary predictor does not by itself test
the broader transfer-of-reasoning hypothesis: it performs no learned multi-step search. The later
experiment agent must read the full thread and choose the design and budget.

## Evidence limits

Acceptance means bounded source, generator, native-engine, split, and
integration checks passed. It is neither learned-generalization evidence nor
an exhaustive novelty/optimality proof. The authoritative current limitations
are in
[`ACCEPTANCE-CAVEATS.md`](../.scratch/multigame-resume/full-standard/ACCEPTANCE-CAVEATS.md);
the experiment gate and handoff are in
[`OBJECTIVE.md`](../.scratch/multigame-resume/full-standard/OBJECTIVE.md) and
[`EXPERIMENT-THREAD-HANDOFF.md`](../.scratch/multigame-resume/full-standard/EXPERIMENT-THREAD-HANDOFF.md).
