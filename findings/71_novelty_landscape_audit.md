# 71 — Recent novelty landscape and surviving hypotheses

## Scope

FlashMorph and Sparse Prefix Caching now have full-text v1 audits. Four additional sources remain
primary arXiv metadata/abstract audits. Exact versions are pinned, but released code, the other full
texts, backward references, forward citations, proceedings, patents, technical reports, deployed
systems, and independent expert review remain mandatory.

## Direct overlaps

Several easy Speck novelty stories are already occupied:

- [FlashMorph, arXiv:2606.30562v1](https://arxiv.org/abs/2606.30562v1) treats hybrid full/linear layer
  placement as a budget-constrained interdependent subset problem and jointly learns layer gates.
- [Sparse Prefix Caching, arXiv:2605.05219v1](https://arxiv.org/abs/2605.05219v1) gives an exact
  distribution-aware dynamic program for sparse recurrent-state prefix checkpoints.
- [Ada-KV, arXiv:2407.11550v5](https://arxiv.org/abs/2407.11550v5) and
  [SqueezeAttention, arXiv:2404.04793v2](https://arxiv.org/abs/2404.04793v2) allocate KV budgets
  non-uniformly across heads, layers, and sequence.
- [Budgeted Attention Allocation, arXiv:2605.05697v1](https://arxiv.org/abs/2605.05697v1) conditions
  attention-head usage on a requested compute budget in one checkpoint.
- [Latent/local-global NSA, arXiv:2511.00819v1](https://arxiv.org/abs/2511.00819v1) already combines
  alternating local/global sparse attention with latent representations.

Therefore joint hybrid placement, adaptive state allocation, sparse recurrent prefix checkpointing,
budget-conditioned attention, local/global alternation, and a bundle of named published components are
not novel by themselves.

The full FlashMorph audit strengthens the placement overlap: it trains a linear replacement for every
layer against a frozen full-attention teacher, jointly optimizes scalar full/linear gates using
answer-token hidden alignment plus a linearization penalty and synthetic retrieval, then discretizes a
preset budget before recovery training. Its boundary is conversion, not a prospective from-scratch law.
The full Sparse Prefix audit confirms an exact `O(NM)` distribution-aware dynamic program while retaining
all attention KV. It isolates recurrent checkpoint positions under block granularity and a fixed last-K
admission policy; production extraction/restoration and joint trie/eviction optimization remain open.

## Two hypotheses that survive provisionally

**N1 — role-grounded placement law.** Middle integration contribution, maximum recurrent refresh gap,
final readout distance, and all-required-source recall may predict which *from-scratch* recurrent/global
placements preserve language, retrieval, and composition at fixed state/compute. Unlike conversion gate
optimization, it must fit some layouts and prospectively predict unseen layouts, tasks, scales, and a
different recurrent/cache representation. It must beat uniform/random, layerwise sensitivity, greedy
replacement, and reproducible FlashMorph-style joint gates. Wrong rankings, sign-changing coefficients,
noncausal interventions, or reduction to count/final-layer presence falsify it.

**N2 — all-required-source recall law.** Multi-hop composition may depend on every route and payload
source surviving compression/selection, and this joint survival probability may predict failure beyond
captured dense mass, top-k overlap, single-source recall, reconstruction error, distance, and load. It
must generalize to unseen graphs/hops, real document sources, rates/budgets, and at least three scales,
with source-restoration interventions. If ordinary mass/reconstruction predicts equally well or the
relationship fails on real tasks, it is falsified.

## Decision

Neither hypothesis is novel yet. No mechanism, composition rule, generalizable law, or inseparable
systems method has passed the Paper 1 novelty gate. Full landscape review and prospectively powered
held-out causal evidence are both required. If neither survives, the work narrows to a replication/
systems study or is not presented as a novel architecture paper.

## Artifact

- [Machine-readable novelty landscape](../research/paper-1/novelty_landscape_v1.json)
