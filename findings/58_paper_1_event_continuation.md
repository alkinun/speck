# 58 — Event-driven Paper 1 baseline continuation

## Requirement

The remaining fixed baseline sequence must continue after successful runs without requiring a manual
status request, but it must not use a periodic polling loop or make decisions from interim quality.

## Design

Each training run receives a one-shot systemd path event for its final `run_summary.json`. The invoked
service immediately disables that path before doing any work, then waits on the training process's
Linux pidfd. This avoids both timer polling and the persistent-`PathExists` failure loop observed in
the first control collector.

After a successful process exit, the transition requires a clean repository, runs the frozen
collector in the isolated CPU environment, updates the machine-readable program pins, validates the
Paper 1 program, and commits the result. Dense control 2 additionally locks the time-to-quality target
from all three controls before any candidate output exists. Candidate 2 additionally runs the frozen
six-cell analysis.

Successor launches use one monotonic 15-minute cooldown timer rather than temperature polling. The
launcher then performs one live check: exact GPU/UUID, zero compute processes and utilization,
temperature at most 50 C, minimum host memory and storage, pinned volume identity, inactive HELMET
transfer, clean repository, correct prior result sequence, and absent output target. A failed check
stops the sequence and launches nothing.

## Decision integrity

Candidate pairs 0, 1, and 2 run in frozen order regardless of observed loss. There are zero interim
efficacy or futility branches. Each result is committed before its successor is scheduled. The final
analysis has proxy-screen authority only; the automation cannot promote an architecture or authorize
paper-scale training.

## Artifacts

- [Frozen automation contract](../research/paper-1/baseline_automation_v1.json)
- [Continuation runner](../scripts/paper_baseline_continue.py)
- [Continuation tests](../tests/test_paper_baseline_continue.py)
