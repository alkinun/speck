# Kimi Linear: An Expressive, Efficient Attention Architecture

- **Paper:** [arXiv:2510.26692](https://arxiv.org/pdf/2510.26692)
- **Version reviewed:** v2, 1 November 2025
- **Code:** [MoonshotAI/Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear) and
  [FLA KDA kernels](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda)
- **Primary topic:** channel-wise delta memory and 3:1 KDA/MLA hybrids

## Central claim

Kimi Linear reports a controlled large-scale result in which a hybrid of linear and full attention beats
a full-attention MLA model trained with the same recipe. Its new mixer, Kimi Delta Attention (KDA),
replaces Gated DeltaNet's one scalar decay per head and token with a separate decay per key channel.
Periodic MLA layers preserve content-addressable global retrieval.

The practical claim is narrower than “linear attention beats attention.” Kimi Linear is a **hybrid**:
three KDA layers followed by one global MLA layer, repeated uniformly. Its global layers remain
quadratic during prefill, and its KV cache grows with context in one quarter of the layers.

## Mechanism

KDA keeps a `d_k × d_v` state per head. Its transition first applies a diagonal channel-wise decay and
then a rank-one delta correction. This is a specialized diagonal-plus-low-rank transition, not an
arbitrary DPLR recurrence. The specialization lets the authors derive a chunkwise parallel algorithm
using dense matrix operations and a small triangular solve.

Compared with scalar decay, channel-wise decay can retain long-lived features while quickly forgetting
others inside the same head. KDA also uses normalized keys, scaled queries, a learned write strength,
short convolutions in the Q/K/V path, per-head output normalization, and a sigmoid output gate.

The global MLA layers use no explicit positional encoding in the best model. The paper interprets the
learned KDA transition as a data-dependent multiplicative position mechanism; KDA supplies order and
recency while NoPE MLA supplies global content lookup.

## Controlled ablations

The initial 16-layer scaling study keeps the stated FLOP budget and hyperparameters fixed:

| Variant | Train PPL | Validation PPL |
| --- | ---: | ---: |
| KDA:MLA `3:1` | 9.23 | 5.65 |
| full MLA `0:1` | 9.45 | 5.77 |
| `1:1` | 9.29 | 5.66 |
| `7:1` | 9.23 | 5.70 |
| `15:1` | 9.34 | 5.82 |
| `3:1`, no output gate | 9.25 | 5.67 |
| `3:1`, SiLU gate | 9.43 | 5.81 |
| `3:1`, no short convolution | 9.29 | 5.70 |

This makes three details hard to dismiss: some global attention is necessary, the `3:1` region is much
better than extremely sparse `15:1`, and sigmoid versus SiLU gating is a material confound.

## Large-model evidence

The matched main comparison uses 48B-total/3B-active MoE models trained on the same 1.4T tokens with
a 4,096-token base context and the same continuation recipe.

- At 128K, NoPE Kimi Linear reports `84.3` RULER, versus `81.3` for MLA and `80.5` for hybrid GDN-H.
- Across the paper's broader long-context mean, NoPE Kimi Linear scores `54.5`, versus `52.2` for MLA,
  `51.2` for GDN-H, and `51.8` for the RoPE Kimi variant. NoPE is therefore a substantive result, not
  merely a simplification.
- The paper reports similar KDA/GDN-H prefill latency, up to `2.9×` Kimi-over-MLA prefill speed at 1M,
  and `6.3×` decode throughput at 1M when lower cache use permits a larger batch. Batch-1 time per output
  token shows a smaller advantage, which is an important distinction.
- KDA wins the paper's small Palindrome, MQAR, and 64-stack studies more consistently than GDN.
- An additional 5.7T-token model reports `94.8` on RULER at 1M, but that comparison is not the clean
  1.4T architecture isolation.

## What matters for Speck

This is the closest external match to Speck's current direction: finite-state delta layers, periodic
global attention, base training at 4K, then context activation. It provides a strong experiment template,
not a result that can be copied without qualification.

The highest-value transfer sequence is:

1. GDN-SiLU versus GDN-sigmoid, holding scalar decay fixed;
2. GDN-sigmoid versus KDA-sigmoid, changing only decay granularity;
3. RoPE versus NoPE in global layers from the same parent;
4. only then sweep global-layer ratio and placement.

The local [Kimi transfer review](../findings/10_kimi_linear_transfer_review.md) and subsequent Speck
experiments already use this separation.

## Limitations and cautions

- No seed intervals are reported for the main architecture comparisons.
- The full context-extension data and schedule are not reproducible from the paper alone.
- The main model is an MoE with 3B active parameters; the optimal ratio may shift sharply at 150M or
  1.2B dense scale.
- MLA geometry and cache economics differ from Speck's GQA. “75% less KV cache” is an architectural
  ratio, not a portable byte count.
- The paper ablates global-layer count but not placement, and synthetic exact-memory tasks remain much
  easier than multi-document reasoning.

## Bottom line

Kimi Linear is the strongest positive reference here for a 3:1 finite-state/global hybrid. Its most useful
lesson is methodological: match recipes, separate gate/decay/position effects, test exact memory, and
measure systems behavior at the intended batch and length.
