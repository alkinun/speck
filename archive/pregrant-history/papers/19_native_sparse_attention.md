# Native Sparse Attention

- **Paper:** [arXiv:2502.11089](https://arxiv.org/pdf/2502.11089)
- **Version reviewed:** v2, 27 February 2025
- **Primary topic:** trainable, hardware-aligned sparse attention

## Central claim

Native Sparse Attention (NSA) processes every query through three learned branches: compressed global
block summaries, fine-grained tokens from dynamically selected blocks, and an uncompressed local sliding
window. The selection and kernels are designed together so sparsity reduces real training, prefill, and
decode cost rather than only theoretical FLOPs.

## Three branches

1. **Compression.** Overlapping sequential blocks are mapped to learned compressed K/V tokens with an
   MLP and intra-block position encoding. This gives every query a cheap coarse view of all history.
2. **Selection.** Attention scores already computed against compressed blocks induce block importance.
   Each query keeps the top blocks and attends to their original fine-grained K/V tokens. Selection is
   shared across query heads in a GQA group so K/V fetches are contiguous and reusable.
3. **Sliding window.** A dedicated local branch handles recent syntax and continuity, allowing the other
   branches to specialize in global scanning and retrieval. A learned gate combines branch outputs.

The reference configuration uses compression block length 32 with stride 16, selection blocks of 64,
top-16 selected blocks—including fixed initial/local blocks—and a 512-token window. The exact effective
sparsity varies with context length.

## Hardware design

Token-granular top-k produces scattered memory reads and poor tensor-core use. NSA instead loads
contiguous K/V blocks and groups queries that share their selected blocks. Separate Triton kernels cover
compression, selection, and window attention. The implementation handles forward and backward, making
the sparse operator part of pretraining rather than an inference-only eviction policy.

The authors report up to `9×` forward and `6×` backward attention-kernel speedup over full attention at
64K. Those are operator measurements; end-to-end model speedup is smaller because FFNs, routing, and
communication remain.

## Evidence

- The main controlled pretraining uses a 27B-total/3B-active, 30-layer GQA+MoE model trained on 260B
  tokens. Full attention and NSA share the rest of the architecture and recipe.
- NSA tracks full-attention loss closely and slightly improves the paper's aggregate short, long, and
  reasoning evaluations.
- The model retrieves a single needle across the 64K grid. Compression provides coarse global location;
  selection recovers original tokens from the chosen region.
- A supervised reasoning variant, NSA-R, exceeds the full-attention comparison on the reported AIME
  setup while using lower long-context attention cost.
- Alternative learned auxiliary selectors and parameter-free min/max heuristics have worse loss in the
  paper's 3B study, supporting the reuse of compression attention scores.

## What matters for Speck

NSA is a later-stage candidate for global layers when quadratic prefill, rather than KV capacity alone,
becomes dominant. Its most transferable idea is the three-way factorization: coarse global scan, exact
fine retrieval, and always-on local context.

Before adoption, Speck needs a production-shape kernel and a reference contract covering selected-block
indices, causality, backward gradients, grouped-query sharing, mixed precision, and cached decode. Compare
against both full attention and a much simpler MoBA/DSA selector.

## Limitations and cautions

- The method is architecture-and-kernel coupled; implementing only the math without the query grouping
  will not reproduce speed.
- Top-k routes gradient only through selected fine blocks; the compression branch and gating create the
  trainable global signal but do not make discrete selection fully differentiable.
- The 260B-token result is large but far below frontier horizons and has no multi-seed intervals.
- Strong single-needle retrieval and aggregate benchmarks do not establish worst-case multi-hop coverage.
- No mainstream consumer runtime offers this exact three-branch kernel path.

## Bottom line

NSA is the best reference for sparse attention designed natively for both learning and hardware. It is
more capable and more complex than Speck's first sparse experiment should be; use it as the target design
and validation checklist.
