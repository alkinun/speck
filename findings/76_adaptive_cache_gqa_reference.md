# 76 — Adaptive cache GQA reduction reference

## Question

When several query heads share one physical KV cache head, which reduction preserves the retained-mass
objective before adaptive physical-slot allocation? Ada-KV's paper specifies the arithmetic mean; its
pinned implementation also exposes maximum aggregation. The choices were therefore separated rather
than treated as interchangeable.

## Derivation and qualification

One physical eviction indicator applies to every query head sharing that KV head. Under the paper's
single shared maximum transformed-value norm, the relevant retained score for a physical token is the
sum of its query-head attention weights. For equal-size groups, the mean divides every score by the same
positive group size, so it preserves the global ordering and allocation exactly. This is a Speck
derivation from the paper's bound, not an explicit theorem in the paper.

The clean-room reference passes 600 normalized random tensors, all 4,200 physical budgets, 22,000
exhaustive feasible budget allocations, and 500 cases with Speck's exact three-physical-head/four-query-
heads-per-group geometry. Mean and sum always select identical physical identities; mean allocation
maximizes original query-head retained mass; and the direct query-level bound equals the group-scaled
physical bound and is monotone. Maximum numerical discrepancies are `5.33e-15` or smaller.

## Negative controls

With two physical heads, two query heads per group, two tokens, and a one-slot budget, the frozen
counterexample makes mean retain physical slot `(1,0)` with query mass 1.2. Max instead retains `(0,1)`
with mass 1.1. Max is deterministic but is not theorem-equivalent.

The safeguard is also excluded. For adaptive budgets `(4,1,1)`, average capacity two, and ratio 0.2,
the pinned code's independent code-like rounding produces `(3,1,1)`: five slots instead of six. This
also preserves the observed coefficient-direction ambiguity between the paper equation and code.

## Decision

Equal-group arithmetic mean is qualified only as a clean-room GQA reduction primitive. Unequal groups,
head-specific norm weighting, observation-window acquisition, max as a primary reducer, safeguard
integerization, model integration, training, novelty, and architecture promotion all remain blocked.

## Artifacts

- [Frozen GQA protocol](../research/paper-1/adaptive_cache_gqa_v1.json)
- [GQA qualification result](../results/Speck-Paper1/adaptive-cache-gqa-qualified.json)
