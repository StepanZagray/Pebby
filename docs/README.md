# Pebby documentation

The root [README](../README.md) is the project entry point. This directory keeps
only durable documentation: domain rules, proof boundaries, operational workflows,
and repository conventions. Checkpoint metrics, experiment ledgers, generated-bank
snapshots, and browser evidence are deliberately not maintained here.

## Documents

- [Game and proof](game-and-proof.md) — LS20 mechanics and what the planner and
  generated-level pipeline establish.
- [Training and model operation](agent-and-training.md) — create data, train a
  checkpoint, evaluate it, and start the runtime.
- [Navigation research experiments](navigation-research.md) — compare encoder
  adaptation and action readouts, measure primitive mastery, and collect learner states.
- [Viewer and HostAI integration](viewer.md) — run the local viewer and connect
  the HostAI runtime.
- [Development notes](development.md) — repository layout, checks, and provenance.
