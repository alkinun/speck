# Alternating Sparse Attention with latent local/global branches

- **Paper:** [arXiv:2511.00819](https://arxiv.org/pdf/2511.00819)
- **Version reviewed:** v1, 2 November 2025
- **Primary topic:** alternate local and compressed/selective sparse-attention layers with latent caches

## Branch-role study

The paper first ablates Native Sparse Attention's window, compressed, and selective branches. Its
interpretation is that window attention drives language/local quality, selective attention drives
long-range retrieval, and putting every branch in every layer can create interference. Removing window
attention can improve retrieval even while harming language-oriented behavior; removing selection harms
retrieval, and finer compression alone does not replace it.

Alternating Sparse Attention (ASA) redistributes local window versus compressed/selective work across
different layers rather than using the same three-branch mixture everywhere. It combines MLA for local
window layers with a grouped latent formulation for compressed/selective layers. Consecutive queries can
reuse selected blocks to improve hardware utilization.

## Evidence boundary

The matched models have 340M and 1.3B parameters, train on 15B and 100B SlimPajama tokens, and use 8K
context. Compression/selection blocks are 16 tokens; the reported ASA setup reuses the first query's
selected blocks for four consecutive tokens and concentrates selection capacity into alternating layers.

The paper reports common-sense, single-needle variants, and LongBench. ASA often improves the aggregate,
but it is not uniformly best: some retrieval cells remain below GQA or NSA, especially harder word
needles. The published tables do not provide a multi-seed uncertainty design comparable to Speck's
promotion policy.

## What matters for Speck

The qualitative claim that local layers support language/local coherence while selective layers support
retrieval—and that alternating their placement can reduce interference—is prior art. A Speck
integration/refresh/readout law must therefore do more: preregister quantitative diagnostics, predict
unseen from-scratch placements/tasks/scales, beat ASA and joint-gate placement baselines, and survive
causal interventions.

ASA also reinforces the required separation of local, compressed, and selective branches. Its specific
MLA/GLA geometry, alternation, block 16, query grouping, results, and 50% cache statement do not transfer
to Speck.

## Transfer cautions

- Multiple changes—alternation, branch redistribution, MLA/GLA, and kernel grouping—enter the final ASA
  comparison.
- Eight-kilobyte training/evaluation does not establish 128K–1M behavior.
- Single-needle variants do not establish multi-source composition.
- Aggregate gains coexist with task/cell regressions.

## Bottom line

Alternating local and selective latent-attention layers are not novel. Speck must explain and predict
when placement works, rather than reproduce ASA's qualitative specialization story.
