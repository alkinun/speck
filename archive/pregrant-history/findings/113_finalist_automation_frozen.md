# 113 — Event-driven finalist automation frozen

## Implementation

The finalist continuation runner and five state-machine tests are pinned before any checkpoint or result
exists. The exact order is six controls, target lock, then six candidates. Each final summary triggers
collection, program validation, one result/transition commit, and a 15-minute one-shot successor timer.
The last candidate triggers the frozen twelve-cell analysis instead of another launch.

Every launch rechecks the entire unfinished output suffix, repository/program validity, GPU identity,
temperature and idleness, host memory, filesystem identity/options/free space, and HELMET inactivity.
The state machine rejects candidates before the control lock, duplicate/missing order, premature
analysis, output overwrite, quality branches, retries, and cell skipping.

## Decision

Automation source and state logic qualify, but training remains false until a live qualification is
recorded after HELMET acquisition becomes inactive. No checkpoint/output exists. The chain retains no
attribution, promotion, novelty, release, or paper-scale authority.

## Artifact

- [Finalist automation v1](../research/paper-1/finalist_automation_v1.json)
