# Reference-paper depth audit

This audit uses the primary reports, not third-party summaries:

- [Kimi Linear, arXiv:2510.26692](https://arxiv.org/abs/2510.26692), v2 reviewed in the repository.
- [Kimi K3, arXiv:2607.24653](https://arxiv.org/abs/2607.24653), v2 reviewed in the repository.
- [DeepSeek-V4, arXiv:2606.19348](https://arxiv.org/abs/2606.19348), v1.

The goal is not to imitate their page count. It is to ensure every Speck claim has the mathematical,
empirical, systems, and failure-analysis support expected of a serious architecture report.

## Structural comparison

| Evidence layer | Kimi Linear | Kimi K3 | DeepSeek-V4 | Paper 1 requirement |
| --- | --- | --- | --- | --- |
| Central mechanism | KDA and its chunkwise algorithm | sequence/depth/width information flow | CSA/HCA hybrid attention | at least one Speck-specific novelty |
| Mathematical specification | recurrence, chunkwise derivation, DPLR relation, complexity | architecture definitions for KDA, gated MLA, AttnRes, LatentMoE | CSA/HCA/mHC/Muon equations and algorithms | equations, shapes, pseudocode, state transitions for every retained component |
| Mechanism tasks | palindrome, MQAR, 64-stack | mostly inherited architecture evidence plus broad internal analysis | limited component-level synthetic isolation in the final report | calibrated synthetic tasks plus causal interventions and counterexamples |
| Component ablations | gate, convolution, KDA:MLA ratio, NoPE/RoPE | component and stability descriptions, fewer clean whole-model isolations | strong design/accounting discussion, but many final gains are bundled | one-axis discovery, replication, pairwise interactions, `2^3` cube, final removals |
| Scaling law | five active-parameter points for KDA hybrid and MLA | explicit pretraining scaling-law section | two final scales and inherited development studies | five points per final architecture plus uncertainty and a held-out point |
| Matched pretraining | KDA/MLA/GDN-H at 48B-total/3B-active and 1.4T shared tokens | full 2.8T-total/104B-active system | 284B/13B and 1.6T/49B systems on 32T/33T tokens | dense and hybrid controls under identical tokenizer/data/tokens/optimizer/evaluation |
| Training recipe | optimizer, schedule, tokens, batch, context | data, scaling, recipe, long-context extension | data, model/training settings, length curriculum, stability interventions | full provenance, trajectories, instability events, time/energy to quality |
| Long-context evidence | RULER, MRCR, HELMET-ICL, LongBench, Frames, RepoQA, code suites | native extension and long-horizon agent evaluation | 1M training plus synthetic and corpus-level evaluations | internal retrieval/composition and pinned RULER/NoLiMa/HELMET per-length curves |
| Post-training transfer | matched SFT and math RL comparisons | extensive SFT/RL/distillation and deployment-aware training | specialist training, OPD, multiple reasoning modes, real-world tasks | optional, but identical candidate/control recipe whenever architecture transfer is claimed |
| Kernel/system evidence | KDA chunk/recurrent kernels and vLLM | KDA CP, MoE training, memory, serving, prefix cache, scheduling | communication overlap, kernels, CP, checkpointing, heterogeneous cache and disk cache | correctness plus training, prefill, decode, distributed, caching, and production-runtime evidence |
| Cost evidence | cache, latency, batch-1 and throughput distinctions | explicit cost-efficiency section | inference FLOP/cache estimates and serving design | analytic and realized cost on consumer and datacenter profiles |
| Limitations | mechanism and comparison boundaries | whole-system attribution limitations | explicitly acknowledges architectural complexity and incomplete principles | negative results, unresolved confidence intervals, unsupported runtimes, and exact non-claims |

## Lessons incorporated into Speck

### From Kimi Linear

1. A paper needs a precise mechanism, not only a model recipe.
2. Synthetic tasks should test the mechanism's alleged strength and expose where simpler recurrent
   alternatives fail.
3. Ratio, gate, local convolution, and positional treatment are separate axes.
4. Scaling efficiency requires multiple compute-optimal points, not a single large comparison.
5. A fair main result keeps architecture, active parameters, training tokens, data, and optimization
   aligned across attention baselines.
6. Short-context, long-context, and post-training learning can rank architectures differently.

### From Kimi K3

1. Sequence, depth, and width are a useful organization for information flow and sparsity.
2. Extreme MoE sparsity cannot be separated from normalization, bounded activation, balancing,
   communication, and memory management.
3. Recurrent/global hybrids require regime-specific kernels, state-aware context parallelism, and
   prefix-cache semantics.
4. A frontier model report covers data, pretraining, post-training, infrastructure, serving, broad
   evaluation, cost, and case studies—not architecture diagrams alone.
5. Whole-system success proves feasibility but is weaker evidence for individual component value;
   Speck therefore adds stricter removal and interaction requirements.

### From DeepSeek-V4

1. Long-context efficiency is a hierarchy: raw local tokens, selective higher-resolution summaries,
   and dense coarse summaries solve different information needs.
2. Compression rate, selection budget, local window, positional path, cache precision, and tail state
   must all be specified.
3. Analytic cache/FLOP reductions require a matching heterogeneous cache manager and sparse kernel.
4. Sparse attention should enter through staged dense warm-up and selector qualification, not from an
   untrained hard top-k assumption.
5. Data construction, model/training settings, stability failures, and mitigation belong in the paper.
6. Complex systems should openly state which mechanisms remain insufficiently understood.

## Standard Speck intentionally raises

Paper 1 will improve on the weakest common aspect of frontier technical reports: attribution. Each final
component must pass alone, in every pairwise combination, in the complete tri-axis presence/absence
cube, and in a final removal test. Scaling, kernel, and independent evaluation evidence must be linked
to the same versioned architecture and result lineage. If resource limits prevent that standard, the
paper narrows its claim rather than substituting an unmatched whole-system comparison.
