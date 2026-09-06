# 117 — Crossed-factor finalist automation v2

## Bound implementation

Automation v2 pins the corrected analysis module, v2 plan and qualification, v2 launch boundary, and
the updated event runner. Its five state tests and seven analysis tests pass. The twelve-run order,
control-only target lock, per-result commits, 15-minute events, and no polling/branch/retry behavior do
not change.

The last candidate now invokes only the v2 analyzer and requires the crossed-factor final status. V1
automation is explicitly unauthorized. All finalist output paths remain absent.

## Decision

V2 implementation, state machine, and analysis binding qualify. Initial launch and training remain
false until a new live qualification is recorded after the HELMET transfer becomes inactive. No
attribution, promotion, novelty, release, or paper-scale authority follows.

## Artifact

- [Finalist automation v2](../research/paper-1/finalist_automation_v2.json)
