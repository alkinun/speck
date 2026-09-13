# 77 — Adaptive cache safeguard apportionment reference

## Problem

Ada-KV's paper and pinned code attach the reported 0.2 safeguard coefficient to opposite sides of the
adaptive/uniform mixture as written. The code also truncates a per-head floor and independently rounds
mixed budgets without checking capacity afterward. The frozen `(4,1,1)` witness consequently produces
`(3,1,1)`, losing one of six physical slots.

## Reference rule

The successor avoids the overloaded name `alpha`. An exact rational `uniform_fraction` lambda names the
fraction moved from an already-valid adaptive allocation toward the real-valued uniform target `B/H`.
Each target is `(1-lambda)*a_h + lambda*B/H`. Floors are taken exactly, then the remaining slots go to
the largest fractional remainders with physical-head index as the deterministic tie-breaker.

This is Hamilton-style apportionment over exact rational values. It preserves all `B` slots, keeps every
head within its physical capacity, returns a floor or ceiling of each target, recovers the adaptive arm
at lambda zero, and recovers the existing quotient/remainder uniform control at lambda one.

## Qualification

Every allocation for one through four heads and one through five tokens per head was checked at uniform
fractions 0, 1/5, 1/2, 4/5, and 1. The 14,120 cases required 952,660 exhaustive feasible-allocation
comparisons. The result has zero conservation error and exactly matches the joint L1/squared-L2 nearest
integer oracle with deterministic ties. All 5,648 endpoint checks and all 14,120 minimum-uniform-floor
checks pass. The registered upstream-style negative witness remains five slots; the reference preserves
six.

## Decision

This qualifies only a conservation-safe apportionment control for later paired empirical evaluation.
It does not select the paper's coefficient semantics, reproduce upstream behavior, establish a quality
benefit, authorize primary safeguard use, integrate a model, authorize training, affect novelty, or
promote an architecture.

## Artifacts

- [Frozen safeguard protocol](../research/paper-1/adaptive_cache_safeguard_v1.json)
- [Safeguard qualification](../results/Speck-Paper1/adaptive-cache-safeguard-qualified.json)
