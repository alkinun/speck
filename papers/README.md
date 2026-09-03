# Papers on efficient, long-context language models

This directory is the literature layer for Speck's architecture research. It is deliberately separate
from [`findings/`](../findings/README.md): papers record what other teams report; findings record what
Speck has actually reproduced. A result in this directory is not a Speck result unless a finding links
to a checked experiment artifact.

## Reading conventions

- Every note was checked against the linked paper version, not only its abstract.
- Numbers are the authors' reported results unless a paragraph is explicitly labeled **Speck
  interpretation**.
- “Context” is split into trained, evaluated, and advertised/inference lengths whenever the paper
  makes that distinction.
- Speedups are meaningful only with the stated hardware, batch, precision, baseline, and sequence
  length. They should not be transplanted into Speck capacity plans as constants.
- Retrieval results from passkey or needle tests are treated as diagnostics, not evidence of robust
  long-document reasoning.
- Technical reports often bundle architecture, data, post-training, and systems changes. Their model
  comparisons do not isolate architecture unless the note identifies a controlled ablation.

## Suggested reading order

### Linear mixers and hybrid backbones

1. [Mamba-2 and Structured State Space Duality](06_mamba2_ssd.md) — the mathematical and kernel
   foundation for modern chunkwise linear mixers.
2. [Gated Delta Networks](01_gated_delta_networks.md) — combines global decay with targeted
   delta-rule correction.
3. [Kimi Linear](02_kimi_linear.md) — channel-wise KDA and the strongest controlled 3:1 hybrid
   comparison in this set.
4. [Gated Attention](03_gated_attention.md) — a low-cost output gate for exact-attention layers.
5. [Samba](07_samba.md) — the inexpensive Mamba + sliding-window-only hybrid.
6. [Nemotron-H](08_nemotron_h.md) and [Nemotron 3 Nano](09_nemotron_3_nano.md) — aggressive
   Mamba-2/attention ratios, FP8, and an honest extreme-length curve.
7. [LFM2](10_lfm2.md) — an edge-first counterpoint where gated short convolution, not an SSM, wins
   the hardware-in-the-loop search.
8. [Falcon-H1](11_falcon_h1.md) and [Hymba](12_hymba.md) — parallel hybrid-head alternatives.
9. [Zamba2](13_zamba2.md) — parameter-efficient reuse of shared attention blocks.
10. [MiniMax-01](14_minimax_01.md) and [MiniMax-M2](15_minimax_m2.md) — an unusually useful
    before/after pair: large-scale hybrid adoption followed by a full-attention retreat.

### Making the exact-attention path cheaper

11. [DeepSeek-V2 / MLA](16_deepseek_v2_mla.md) — compress the per-token KV representation.
12. [Cross-Layer Attention](17_cross_layer_attention.md) — share KV projections between adjacent
    layers.
13. [YOCO](18_yoco.md) — build one global KV cache and reuse it throughout a cross-decoder.
14. [Native Sparse Attention](19_native_sparse_attention.md) — jointly trained compression,
    selection, and local-window branches.
15. [MoBA](20_moba.md) — MoE-style routing over context blocks.
16. [DeepSeek-V3.2 / DSA](21_deepseek_v3_2.md) — retrofit token-level sparse attention into an MLA
    checkpoint.
17. [DeepSeek-V4](22_deepseek_v4.md) — combine sequence compression, sparsity, and dense attention
    at one-million-token scale.

### Scaling beyond the sequence mixer

18. [Attention Residuals](05_attention_residuals.md) — content-dependent information flow over
    model depth.
19. [Kimi K3](04_kimi_k3.md) — a frontier-scale integration of KDA, gated MLA, Block AttnRes, and
    an extremely sparse MoE.

## Decision map for Speck

| Question | Best starting papers | What must still be measured locally |
| --- | --- | --- |
| Which finite-state mixer? | GDN, Kimi Linear, Mamba-2 | gate isolation, MQAR/copy/state tracking, LM loss, kernels |
| How many global layers? | Kimi Linear, Nemotron-H, Mamba-2 | ratio and placement at Speck scale; multi-hop retrieval |
| Could local attention be enough? | Samba, LFM2, MiniMax-M2 | matched full/global controls beyond 32K |
| Layerwise or parallel hybrid? | Falcon-H1, Hymba, Kimi Linear | equal-parameter and equal-FLOP comparison |
| How should exact attention be compressed? | MLA, CLA, YOCO | cache bytes, decode latency, quality, runtime support |
| When should sparse attention enter? | NSA, MoBA, DSA, DeepSeek-V4 | differentiability, kernel maturity, prefill/decode parity |
| How should depth scale? | Attention Residuals, Falcon-H1 | deep/narrow sweep including latency and activation memory |
| What is the release gate? | MiniMax-M2, Nemotron 3 Nano | per-length RULER/NoLiMa/HELMET and multi-hop agent tests |

## Source set

The collection contains all 22 papers supplied in the research brief. This is slightly above the
requested approximate range because dropping two papers would break useful comparison pairs (especially
MiniMax-01/M2 and DeepSeek-V2/V3.2/V4).
