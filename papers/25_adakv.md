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
- Eviction reduces stored KV; sparse attention may retain but selectively read entries, so system costs
  differ.
- Within-layer head allocation is not joint allocation across layer, sequence, recurrent, or compressed
  state.

## Bottom line

Ada-KV directly occupies attention-mass-based adaptive cache allocation. Speck N2 survives only if
constituent-complete source recall adds held-out causal predictive value beyond this strong baseline.
