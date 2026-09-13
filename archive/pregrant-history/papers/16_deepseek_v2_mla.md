# DeepSeek-V2 and Multi-Head Latent Attention

- **Paper:** [arXiv:2405.04434](https://arxiv.org/pdf/2405.04434)
- **Version reviewed:** v5, 19 June 2024
- **Code and weights:** [deepseek-ai/DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2)
- **Primary topic:** low-rank KV compression for exact global attention

## Central claim

Multi-Head Latent Attention (MLA) keeps global softmax attention but stores a compressed latent vector
per token and layer instead of independent full keys and values for every head. Learned up-projections
reconstruct the head-specific content used by attention. DeepSeek-V2 couples MLA with DeepSeekMoE in a
236B-total/21B-active model and reports a `93.3%` KV-cache reduction versus its earlier 67B dense model.

MLA is a cache representation change, not sparse attention. Its core attention is still global and
quadratic during training/prefill.

## Mechanism

For token hidden state `h_t`, a down-projection produces a joint KV latent `c_t^KV` of dimension `d_c`.
Separate up-projections turn that latent into the content keys and values for all heads. During decoding,
the runtime caches `c_t^KV`, not those expanded tensors. Query compression is also used to reduce
projection compute, though it does not change cache size.

RoPE creates a complication: a position-dependent rotation cannot generally be absorbed through the
low-rank up-projection. MLA therefore separates each key/query into:

- a content component produced through the compressed path;
- a smaller positional component with RoPE applied explicitly.

Only the joint content latent and shared positional key are cached. For DeepSeek-V2, the per-token,
per-layer cache contains `d_c + d_h^R` elements rather than `2 × n_heads × d_head`.

## DeepSeek-V2 configuration

- 60 layers, width 5,120.
- 128 attention heads of dimension 128.
- KV latent dimension 512, query latent dimension 1,536, decoupled RoPE dimension 64.
- 2 shared and 160 routed FFN experts; six routed experts active per token.
- Base training at 4K on 8.1T tokens; YaRN extension with 1,000 additional 32K steps for a 128K window.

The authors add RMSNorm around compressed bottlenecks because low-rank projection and fine-grained MoE
change activation scale.

## Evidence

- In a controlled appendix comparison, large MoE MLA uses 34.6K cached elements per token versus
  860.2K for MHA—about 4%—and slightly improves the four reported hard benchmarks. A smaller control
  uses 15.6K versus 110.6K, about 14%.
- Relative to DeepSeek 67B as a whole system, the paper reports `93.3%` less KV cache and up to `5.76×`
  maximum generation throughput. The throughput improvement includes the larger batches enabled by
  lower memory use.
- YaRN is applied only to the decoupled positional key path. Although extension training is at 32K, the
  model performs well on the paper's 128K single-needle test.
- The model supports 128K, but the published long-context evidence is far narrower than later
  RULER/HELMET suites.

## What matters for Speck

MLA is the strongest cache diet for the minority global layers in a KDA/GDN hybrid. Because those layers
dominate state at 128K–1M, compressing them can matter more than small improvements to the already fixed
recurrent state.

The first Speck comparison should hold global-layer count and placement fixed and compare GQA with MLA
on:

- exact cache bytes including positional components and quantization scales;
- prefill FLOPs and decode bandwidth at batch 1 and maximum resident batch;
- short loss, MQAR, and multi-hop long-context quality;
- a reference implementation versus the production kernel;
- NoPE MLA as a separate arm, because Kimi Linear avoids the decoupled-RoPE path entirely.

TransMLA-style conversion can initialize an MLA experiment from a GQA checkpoint, but conversion
quality is not evidence for from-scratch equivalence.

## Limitations and cautions

- MLA saves cache, not the `O(L^2)` score computation of a global layer.
- The headline 93.3% is versus DeepSeek 67B and includes different model geometry; use the controlled
  per-token element table for architecture reasoning.
- Expanded head-specific K/V may require specialized kernels to avoid giving back bandwidth and compute
  savings.
- The long-context extension is evaluated heavily with NIAH and does not isolate robust reasoning at
  128K.

## Bottom line

MLA is the leading way to keep exact global attention while making its cache much smaller. It is a
high-value second-stage optimization once Speck has established how many global layers it actually
needs and can afford the kernel work.
