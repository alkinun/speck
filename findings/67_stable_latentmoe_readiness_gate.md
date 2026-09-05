# 67 — Stable LatentMoE readiness gate

## Evidence limit

The local Kimi K3 review supports a stability package—latent normalization, bounded SiTU-GLU,
Quantile Balancing, extreme expert sparsity, and optimizer/system changes—but does not contain
implementation-complete equations for LatentMoE, SiTU-GLU, or Quantile Balancing. Writing local formulas
from those names would manufacture evidence. The first required action is therefore a pinned primary
specification/code audit, not implementation.

The sequence and depth parents are also unselected, and no named expert-parallel hardware, checkpoint,
optimizer-memory, or communication envelope exists. A single RTX 3090 can qualify reference semantics
and small conventional MoE geometry but cannot establish the required expert-parallel serving claim.

## Frozen decomposition

The width program is strictly ordered:

1. dense SwiGLU versus direct conventional top-k SwiGLU experts plus one shared expert;
2. direct experts versus exact LatentMoE factorization;
3. add latent normalization only;
4. plain versus one exact bounded activation;
5. auxiliary-loss, bias, and exact quantile balancing; then
6. total expert count/top-k geometry.

No stage may silently absorb the next. The initial conventional reference must freeze expert placement,
one shared expert, selected-weight normalization, deterministic ties, router precision, initialization,
and optimizer roles. It is dropless; capacity, padding, token dropping, or overflow fallback form separate
arms. Active and total parameters explicitly include routers, shared/selected experts, latent projections,
normalization, and balancing state.

## Required evidence

Correctness covers dense/one-expert reductions, dispatch forward/backward parity, token conservation,
zero-token and overloaded experts, checkpoint/export, single-device/expert-parallel semantics, mixed
precision, and full memory accounting. Stability evidence includes load tails, dead experts, routing
margins/churn, specialization/redundancy, activation/gradient quantiles, loss-spike history, dispatch,
all-to-all, GEMM utilization, and small-batch latency.

Any rescue policy must be frozen before training. Every trigger, duration, old-router identity, added
compute, and trajectory is reported; frequent rescue is a failed stability result rather than invisible
infrastructure.

## Decision

No Stable LatentMoE component or geometry is selected. Implementation, training, and promotion remain
blocked. A sparse arm eventually needs all quality floors plus at least 20% realized primary systems
improvement on both single-device and expert-parallel paths after expert memory, communication, and
fragmentation.

## Artifact

- [Stable LatentMoE readiness gate](../research/paper-1/stable_latentmoe_readiness_v1.json)
