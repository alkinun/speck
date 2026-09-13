# 110 — Finalist collector and analysis implementation qualified

## Implementation

The finalist-specific collector now validates exactly one final step-23,496 checkpoint, resolved
seed/data/geometry, five frozen validation points, complete-batch token arithmetic, timing, metadata,
summary, and all contract hashes. The target-lock path accepts exactly six dense controls and rejects
candidate contamination.

The analyzer requires all twelve unique cells and candidate record creation after the lock. It computes
df=5 one-sided paired bounds under unchanged aggregate/source margins, enforces identical source keys,
retains any pair censoring, and reports fixed-token, analytic-FLOP, steady-time, and time-to-quality
views.

Six focused fixture tests pass: clear non-inferiority, complete censoring, control-only locking, missing/
duplicate cell rejection, contract identity, and final-checkpoint collection. Ruff also passes. Module,
CLI, tests, analysis, materialization contract, and generated manifest are SHA-pinned.

## Decision

Collector, target lock, analysis, and stopping-rule implementations qualify. This does not qualify the
CUDA runtime or release dependencies and does not authorize training, automatic launch, attribution,
promotion, or paper-scale execution.

## Artifact

- [Finalist analysis qualification v1](../results/Speck-Paper1/finalist-analysis-qualified-v1.json)
