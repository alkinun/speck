# 159 — Midpoint resolves fixed-score QB ties, not the training policy

## Pre-output correction

The first validation contract was rejected before producing a result. Its bracket required
`count(r < τ) ≤ q < count(r ≤ τ)`, which describes the upper order-statistic endpoint but falsely
rejects a midpoint strictly inside a nonzero minimizer interval, where both counts equal `q`. V2 uses
the general coordinate-subgradient condition `count(r < τ) ≤ q ≤ count(r ≤ τ)`.

The 256 intended v1 holdout cells were inspected while finding that error. They are recorded as
consumed diagnostics with no selection authority. V2 froze 256 fresh cells—64 seeds across each of
four `(tokens, experts, top-k)` shapes—before the corrected validator revealed them.

## Fresh result

The interval-midpoint candidate balanced all 256/256 fixed-score cases within 33 updates. It had zero
cycles, unresolved cases, coordinate-subgradient failures, interval violations, and non-finite values.

The source controls did not establish the same finite-batch property under the local stable top-k:

- the exact upper endpoint balanced 28/256 and cycled in 228;
- the 1,000-bin histogram balanced 241/256 and cycled in 15;
- the 10,000-bin sensitivity balanced 254/256 and cycled in 2.

This does not contradict QB's large-batch empirical use. It isolates a finite, fixed-score interaction
among order-statistic representative, histogram interpolation, and deterministic tie behavior. The
midpoint is a local extension, not a claim about Moonshot's implementation.

## Decision boundary

Midpoint is qualified only as the preferred fixed-score float64 CPU tie-safe oracle. A training policy
is not selected. Hardware top-k tie semantics, changing router scores, optimizer coupling, distributed
parity, quality/specialization, route churn, histogram precision, and communication remain mandatory.
Any future training comparison must retain source upper, source 1,000-bin histogram, and midpoint arms.
No active model, experiment program, finalist output, checkpoint, service, GPU, or runtime was changed.

## Artifacts

- [V1 failed gate](../research/paper-1/quantile_balancing_tie_readiness_v1.json)
- [Corrected v2 gate](../research/paper-1/quantile_balancing_tie_readiness_v2.json)
- [V2 qualification](../results/Speck-Paper1/quantile-balancing-tie-v2-qualified.json)
- [Standalone tie reference](../speck/quantile_balancing_tie_reference.py)
