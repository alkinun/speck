# Ada-KV: adaptive head-wise KV eviction budgets

- **Paper:** [arXiv:2407.11550](https://arxiv.org/pdf/2407.11550)
- **Version reviewed:** v5, 16 October 2025
- **Code:** [FFY0/AdaKV](https://github.com/FFY0/AdaKV)
- **Primary topic:** allocate a fixed per-layer eviction budget non-uniformly across attention heads

## Objective and algorithm

Ada-KV studies post-hoc autoregressive KV eviction, not native sparse training. It defines eviction loss
as the L1 distance between dense and post-eviction multi-head attention outputs. The paper bounds this
loss by a constant involving the maximum transformed-value norm and the total attention probability
retained across heads. At a fixed allocation, retaining each head's top attention weights minimizes that
bound.

Adaptive allocation then concatenates attention weights across all heads in one layer, selects the
globally largest `B`, and gives each head a budget equal to how many of its entries survived that global
selection. Concentrated heads receive fewer slots and dispersed heads receive more. The method wraps
existing top-k eviction schemes such as SnapKV and Pyramid.

## GQA and safeguard semantics

Section 3.5 specifies arithmetic-mean attention weight within each GQA group as the physical-KV-head
selection score. This is principled for equal-size groups under the paper's shared maximum value/output
norm: a physical eviction indicator is shared by every query head in the group, so the relevant retained
mass is the sum of their attention weights; dividing every group by the same group size leaves the global
ranking unchanged. This extension is a Speck derivation from the paper's bound, not a theorem the paper
states explicitly. Unequal group sizes or query-head-specific norm weights do not inherit it unchanged.

The pinned code defaults to `mean` but also exposes `max`. Max aggregation is not rank-equivalent to
summed query-head retained mass and therefore lacks the same bound-optimality justification. It must be
treated as a separate empirical ablation, not an interchangeable implementation detail.

The paper's safeguard equation mixes `alpha * adaptive + (1-alpha) * uniform` and states a default
`alpha=0.2`. The pinned implementation instead computes approximately
`(1-floor_ratio) * adaptive + floor_ratio * uniform`, after truncating the uniform floor and then
rounding each head independently. It checks conservation before this transformation but not after it.
The coefficient direction and integer budget conservation are therefore unqualified; Speck must exclude
this safeguard until a separately frozen, conservation-preserving integerization rule is tested.

## Evidence boundary

The paper evaluates Llama-3.1 and Mistral instruction models on 13 RULER and 16 LongBench tasks across
fixed and ratio budgets. It separates question-aware eviction, where the query is visible during cache
selection, from harder question-agnostic eviction. Every method degrades materially in the latter, while
Ada allocation generally improves its wrapped baseline.

The included tasks span retrieval, variable tracking, word aggregation, single/multi-hop QA,
summarization, few-shot learning, and code. However, the allocation is head-wise only within each layer;
the authors explicitly leave cross-layer allocation and its theory to future work.

## What matters for Speck

Captured attention mass and attention-output error are mandatory baselines for any Speck selector or
compression diagnostic. Head-wise adaptive budget allocation is not novel, and all-required-source
recall must demonstrate predictive information beyond Ada-KV's bound, dense mass, and output error.

Ada-KV evicts exact KV after training. Speck's proposed diagnostic concerns native compressed/sparse
representations and multi-source causal composition, which is a plausible distinction only if it
generalizes and supports source-restoration interventions.

## Transfer cautions

- The theoretical bound uses a shared maximum transformed-value norm and may be loose for task-specific
  multi-source failures.
- Attention weights are observed under a particular question-aware or observation-window regime.
- Equal-size GQA makes mean and summed-mass rankings equivalent, but max aggregation does not share that
  guarantee; the paper/code safeguard semantics are also not identical as written.
- Eviction reduces stored KV; sparse attention may retain but selectively read entries, so system costs
  differ.
- Within-layer head allocation is not joint allocation across layer, sequence, recurrent, or compressed
  state.

## Bottom line

Ada-KV directly occupies attention-mass-based adaptive cache allocation. Speck N2 survives only if
constituent-complete source recall adds held-out causal predictive value beyond this strong baseline.
