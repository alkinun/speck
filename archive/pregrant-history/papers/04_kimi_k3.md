# Kimi K3: Open Frontier Intelligence

- **Paper:** [arXiv:2607.24653](https://arxiv.org/pdf/2607.24653)
- **Version reviewed:** v2, 7 August 2026
- **Primary topic:** frontier-scale KDA hybrid, depth mixing, and sparse MoE systems

## Central claim

Kimi K3 scales the Kimi Linear recipe to a 2.8T-total/104B-active, natively multimodal MoE with a
one-million-token context. Its architecture expands information flow along three axes: KDA plus gated
MLA over sequence, Attention Residuals over depth, and Stable LatentMoE over width.

This is valuable evidence that the components can coexist at frontier scale. It is not a clean
architecture ablation: K3 also changes data, optimizer behavior, multimodal training, post-training, and
infrastructure relative to earlier Kimi models.

## Architecture

### Sequence mixing

Each repeated block contains three KDA layers followed by one gated MLA layer. The released
configuration has 69 KDA and 24 MLA layers, including an additional MLA layer near the top. MLA uses
NoPE; the intervening KDA layers provide order and recency.

K3 modifies the earlier Kimi Linear KDA in two notable ways:

- it lower-bounds decay to prevent extremely small cumulative factors from creating numerical and
  chunkwise-kernel problems;
- it replaces the earlier low-rank output gate with a full-rank, channel-wise gate. MLA receives the same
  style of output gate.

### Depth mixing

Block Attention Residuals let each module use a learned pseudo-query to softmax-attend over the token
embedding, completed block summaries, and the current block's partial residual. K3 uses block size 12,
giving nine sources when the embedding is counted. This bounds cross-layer state and pipeline traffic.

### Width mixing

Stable LatentMoE exposes 896 routed experts and activates 16 per token, a routing sparsity of 56. Its
stability package is inseparable from the extreme expert count:

- RMSNorm before the latent up-projection controls scale;
- SiTU-GLU behaves like SwiGLU near zero but smoothly caps large activations;
- Quantile Balancing sets expert biases from a target-load quantile of router margins. A histogram and
  one all-reduce approximate the global quantile without gathering all token/expert scores.

The model also uses Per-Head Muon so orthogonalization does not couple attention heads with very
different gradient scales.

## Long-context training and systems

- Training length progresses from 8K to 64K during pretraining, then from 256K to 1M during cooldown.
- KDA Context Parallelism preserves the ordered product of token-dependent KDA transitions across
  ranks; states cannot simply be summed as in simpler linear attention.
- FlashKDA overlaps within-chunk work with cross-chunk state propagation. Separate decode kernels cover
  the one-token regime.
- The runtime jointly pages fixed KDA states and growing MLA caches. State-aware prefix caching must
  persist a KDA state at a reusable boundary, which creates much coarser natural block sizes than an
  MLA-only cache.
- Block AttnRes representations are sharded and checkpointed; the paper reports that this keeps the
  added backward state near the standard residual path.
- Million-token RL uses external cache retention and offloads KDA state together with corresponding MLA
  pages so hybrid prefix state remains consistent.

## Evidence and interpretation

The report attributes about `2.5×` scaling efficiency over Kimi K2 to the combined architecture, data,
and training recipe. It reports strong 1M-context agent and retrieval behavior, alongside broad
frontier-level evaluations. These results establish feasibility, not the isolated value of KDA, AttnRes,
Stable LatentMoE, or any specific ratio.

For Speck, the most important evidence is operational: a finite-state/global hybrid needs two cache
semantics, state-aware prefix reuse, context parallelism that respects recurrence order, and distinct
kernels for train/prefill/decode. Architecture and runtime must be designed together.

## What matters for Speck

- Preserve the 3:1 KDA/global layout as a reference arm, not as an assumed optimum.
- Consider full-rank sigmoid output gating only after isolating it from decay and NoPE changes.
- Treat Block AttnRes as a separate depth-axis experiment, starting around eight block summaries.
- If expert counts increase, evaluate normalized latent projections and bounded activations before
  inventing more complex load-balancing losses.
- Add prefix-cache tests that compare resumed hybrid state against uninterrupted execution bit-for-bit or
  within declared tolerances.

## Limitations and cautions

- The report is a whole-system technical report with no matched K2/K3 architectural isolation.
- Most pretraining data details and many evaluation harnesses are not reproducible from the report.
- Frontier MoE routing, context parallelism, and custom kernels are far beyond a consumer-runtime first
  implementation.
- A one-million-token window does not imply uniform quality across the full window; per-length and
  task-specific curves remain necessary.

## Bottom line

K3 shows what a mature KDA hybrid requires beyond a mixer equation: depth routing, extreme-width
stability, ordered context parallelism, hybrid cache management, and long-horizon post-training. For
Speck it is a destination architecture and systems checklist, not a first ablation.
