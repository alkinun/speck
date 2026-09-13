# 122 — Identical finalist control 0 restart frozen

## Rerun scope

Attempt 2 is exactly the same first dense cell: pair 0, seed 42, data offset 0, 1,539,833,856 tokens,
23,496 steps, and the same hashed run config. There is no checkpoint to resume, so it must restart from
step 0. No model, optimizer, data, schedule, validation, analysis, margin, pair, or order field changes.

The step-0/step-1 values observed from attempt 1 are explicitly listed and forbidden from influencing
the rerun or analysis. Attempt 1 remains permanently ineligible; only a complete attempt-2 final
checkpoint may create the dense pair-0 result.

## Execution boundary

This is a manually preregistered recovery, not an automatic retry. The program must pin the failed
attempt and this contract, then pass a fresh live gate. If attempt 2 fails, the chain stops again and
requires another explicit recovery decision.

## Decision

The identical restart becomes eligible only after the fresh gate. Training is false now, and no
attribution, promotion, or paper-scale authority follows.

## Artifact

- [Finalist rerun v1](../research/paper-1/finalist_rerun_v1.json)
