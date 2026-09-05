# 75 — Adaptive cache budget clean-room reference

## Boundary

Ada-KV's pinned upstream implementation remains unauthorized for execution or reuse. A v1 protocol was
therefore frozen before local implementation and limits this work to one layer of physical KV-cache
heads. It excludes query-to-KV-head reduction, model integration, training, and architecture claims.

## Qualification

The clean-room allocator sorts every `(weight, head, token)` by descending salience with ascending head
and token identity as deterministic tie-breakers, selects the global top budget, and derives each
physical head's retained-token count. An exhaustive oracle checks every feasible integer head-budget
composition for the frozen small-case domain.

All 200 normalized random matrices, 1,700 random budget cases, 28,240 oracle allocation comparisons, and
84 adversarial budget cases pass. Capacity is conserved; retained identities are per-head prefixes;
ties are deterministic; adaptive retained mass is never below the quotient/remainder uniform control;
and the normalized L1 output-error bound is monotone. The largest optimality and uniform-control gaps
are only `8.88e-16` floating-point noise.

## Decision

The reference is qualified as an independent baseline primitive. It does not qualify upstream code,
select a GQA aggregation rule, authorize integration or training, alter the novelty gate, or support an
architecture claim. Any of those steps requires a separately frozen successor experiment.

## Artifacts

- [Frozen protocol](../research/paper-1/adaptive_cache_budget_v1.json)
- [Qualification result](../results/Speck-Paper1/adaptive-cache-budget-qualified.json)
