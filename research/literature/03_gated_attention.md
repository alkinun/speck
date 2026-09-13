# Gated Attention for Large Language Models

- **Paper:** [arXiv:2505.06708](https://arxiv.org/pdf/2505.06708)
- **Version reviewed:** v1, 10 May 2025
- **Primary topic:** output gating for softmax attention

## Central claim

A query-dependent sigmoid gate placed after scaled dot-product attention (SDPA) and before the output
projection improves loss, downstream quality, training stability, and context extension. Among more
than 30 variants, the best design is head-specific and elementwise within each head.

The intervention is small but precisely located. Moving the gate to Q, K, V, or after the dense output;
sharing it across heads; using an input-independent gate; or replacing the sigmoid with SiLU all perform
worse in the reported study.

## Mechanism

For every query token and attention head, the ungated SDPA output `Y` is multiplied by
`sigmoid(X @ theta)`. The gate has the same channel dimensionality as the head output, so it can suppress
different value channels for different queries and heads before the heads are concatenated and projected.

The authors give two empirical explanations:

1. **Non-linearity.** Without a gate, the value projection and output projection form consecutive linear
   maps around the SDPA-weighted value path. Gating inserts a data-dependent non-linearity into that
   otherwise low-rank composition.
2. **Conditional sparsity.** Most learned gate values are small. This suppresses irrelevant SDPA output
   dimensions without forcing the softmax distribution itself to place probability mass on a sacrificial
   token.

In the 15B MoE analysis, the baseline directs an average `46.7%` of attention mass to the first token,
while the preferred gate reduces that to `4.8%`. This is why the paper describes the resulting models as
attention-sink-free. It does **not** mean attention scores become sparse in the strict indexed-computation
sense; the full SDPA is still computed.

## Experimental evidence

- The study covers 15B-total/2.54B-active MoE and 1.7B dense models, with runs from 400B to 3.5T
  tokens.
- On the 400B-token MoE screen, the preferred SDPA elementwise sigmoid gate reports PPL `5.761`,
  MMLU `60.82`, and GSM8K `55.27`, versus baseline PPL `6.026`, MMLU `58.79`, and GSM8K `52.92`.
- The 3.5T dense run shows substantially fewer loss spikes under the same recipe. Additional sweeps show
  that gated models tolerate higher learning rates and larger batches better, although the best recipe
  changes more than the gate alone.
- Head sharing and input-independent gates reduce sparsity and give back much of the gain. The preferred
  gate's mean score is `0.116`; the input-independent variant's is `0.335`.
- In context-extension experiments, the base 4K gated and ungated models are close. After YaRN extends
  them to 128K, the gated model degrades more slowly at 16K–128K. The paper links this to not relying on
  a first-token sink whose behavior changes when RoPE is rescaled.

## What matters for Speck

This is a low-complexity candidate for every global attention layer. It does not reduce asymptotic
attention FLOPs or KV bytes, but it may improve stability and length transfer for little parameter cost.
It also composes naturally with GQA, MLA, CLA, or sparse attention because it acts on the result of the
core attention operation.

The clean Speck test should hold the global operator, position encoding, layer placement, parameter
budget, and training order fixed, changing only:

- no gate;
- head-specific elementwise sigmoid gate;
- optionally a scalar-per-head gate to quantify whether channel granularity matters at small scale.

Record loss spikes, maximum safe learning rate, gate distributions, first-token attention mass, short
quality, and the full long-context curve. A better RULER point alone would not establish the proposed
mechanism.

## Limitations and cautions

- The explanation is empirical. The paper explicitly lacks a rigorous theory connecting removal of
  attention sinks to length generalization.
- “Sparsity” describes small activations after gating; it provides no direct sparse-kernel speedup.
- Some comparisons combine gating with a higher learning rate or larger batch, so use the fixed-recipe
  rows for causal attribution.
- The released evidence is dominated by one model family and tokenizer/data stack. Small dense models
  may learn different sink behavior.

## Bottom line

Apply the sigmoid gate after per-head SDPA and before output projection, then measure it. It is one of
the cheapest credible quality/stability interventions in this collection, but it is not an attention-cost
reduction by itself.
