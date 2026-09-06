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
    at one-million-token scale. The [official-release source audit](46_deepseek_v4_official_release.md)
    pins the HCA/CSA/local equations, interleaved schedule, compressed-entry identities, causal state,
    inference-code limitations, and license boundary used by the sequence readiness successors.

### Scaling beyond the sequence mixer

18. [Attention Residuals](05_attention_residuals.md) — content-dependent information flow over
    model depth. The [official-source/code audit](45_attnres_official_source.md) pins the report's
    sublayer boundary rule and separates released K3 checkpoint inference from training initialization.
19. [Kimi K3](04_kimi_k3.md) — a frontier-scale integration of KDA, gated MLA, Block AttnRes, and
   an extremely sparse MoE. The [official-release source audit](43_kimi_k3_official_release.md) pins
   the implementation-complete equations, inference-code boundary, and license conditions used by the
   Stable LatentMoE readiness successor.

The [DeepSeekMoE official-source audit](44_deepseek_moe_official_source.md) separately pins the
conventional/fine-grained/shared-expert equations and distinguishes its single-device dropless training
path from the later inference-only expert-sharded V3 reference.

### Direct novelty-overlap audits

20. [FlashMorph](23_flashmorph.md) — jointly optimize interdependent full/linear layer placement during
    Transformer-to-hybrid conversion.
21. [Sparse Prefix Caching](24_sparse_prefix_caching.md) — exact distribution-aware recurrent-state
    checkpoint placement under a prefix-overlap law.
22. [Ada-KV](25_adakv.md) — attention-output-loss-guided head-wise KV eviction budgets.
23. [SqueezeAttention](26_squeezeattention.md) — before/after-attention layer importance and layer-wise
    allocation over sequence KV compressors.
24. [Budgeted Attention Allocation](27_budgeted_attention_allocation.md) — monotone externally
    requested head budgets from one checkpoint.
25. [Alternating Sparse Attention](28_alternating_sparse_attention.md) — redistribute local and
    compressed/selective latent branches across layers.
26. [HeadKV-R2](30_headkv.md) — profile retrieval-plus-reasoning heads and redistribute cache globally
    across layer/head cells.
27. [KV-compression attention dynamics](31_kv_compression_physics.md) — distinguish retention,
    reachability, and utilization under multi-hop cache eviction.
28. [BRIEF](32_brief.md) — decompose multi-hop questions, require one helpful proposition from each
    distinct source document, and train a query-aware evidence compressor on their concatenation.
29. [BRIEF-Pro](33_brief_pro.md) — scale oracle-plus-distractor evidence fusion beyond 10K words with
    user-controlled budgets and full compressor-plus-reader cost accounting.
30. [IterCOMP](34_itercomp.md) — distinguish complete/partial hop evidence, judge sufficiency, and
    iteratively retrieve the missing source during prompt compression.
31. [STEC](35_stec.md) — compress multi-trajectory supporting/conflicting evidence and reasoning paths
    into candidate-specific representations for final multi-hop answer selection.
32. [Rethinking hybrid-attention roles](36_rethinking_hybrid_attention.md) — show that middle full-
    attention layers carry retrieval while efficient mixers shape its learning trajectory across scale.
33. [Systematic hybrid linear attention](37_systematic_hybrid_linear.md) — train 72 models to separate
    mixer, uniform ratio, language quality, recall, and cache-efficiency effects.
34. [Distill-then-Replace](38_distill_then_replace.md) — greedily replace distilled attention layers
    with validation feedback and show static probe rankings miss placement interactions.
35. [Systematic hybrid-architecture design](39_hybrid_architectures_systematic.md) — compare
    from-scratch Transformer/Mamba ratios and early/middle/late depth placements at 350M and 1B.
36. [Massive activations in hybrid linear attention](40_massive_activations_hla.md) — trace
    architecture-aligned pre-attention spikes and plateaus, including controlled placement and gates.
37. [HALO and HypeNet](41_halo_hypenet.md) — select retained full-attention layers from recall/CSR
    sensitivity during efficient Transformer-to-hybrid distillation.
38. [KL-guided hybrid layer selection](42_kl_guided_layer_selection.md) — rank restore-one-layer
    hybrids by generic-text KL and test non-uniform placement across tasks, scales, and mixers.

### Supporting eviction baselines

39. [SnapKV](29_snapkv.md) — score an older prompt prefix from a trailing observation window, pool
    positions into local clusters, and retain a fixed prompt cache for generation.

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

The collection contains all 22 papers supplied in the research brief, nineteen later direct
novelty-overlap audits, and one supporting eviction baseline. The original set is slightly above the requested approximate range because
dropping two papers would break useful comparison pairs (especially MiniMax-01/M2 and
DeepSeek-V2/V3.2/V4).
