# DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence

- **Paper:** [arXiv:2606.19348](https://arxiv.org/pdf/2606.19348)
- **Version reviewed:** v1, 26 April 2026
- **Primary topic:** compressed hybrid attention, million-token MoE, and systems co-design

## Central claim

DeepSeek-V4 combines two attention operators: Compressed Sparse Attention (CSA) for selective precise
access and Heavily Compressed Attention (HCA) for a dense coarse global view. Together with a small local
window, low-precision cache, mHC residuals, and Muon, the 1.6T-total/49B-active Pro model reports only
27% of V3.2's single-token FLOPs and 10% of its KV cache at 1M.

The 284B-total/13B-active Flash model reaches 10% of V3.2 FLOPs and 7% of its cache in the same estimate.
Both are trained natively through a 1M context stage rather than only converted at the end.

## Hybrid attention

### Compressed Sparse Attention

CSA compresses every `m` original token positions to one K/V entry, then uses a Lightning Indexer to
select the top-k compressed entries for each query. Its compressor uses two overlapping `m`-token views,
learned per-channel weights, and positional biases; adjacent compressed entries therefore share part of
their source span. The selected entry serves as both key and value in MQA.

### Heavily Compressed Attention

HCA compresses non-overlapping groups of `m'` tokens, where `m'` is much larger, then attends densely to
all resulting summaries. It has no sparse selector. The purpose is global coverage at very low resolution,
complementing CSA's selective higher-resolution path.

Both operators use low-rank query projection, per-head RMSNorm, shared-KV MQA, and a two-stage grouped
output projection. They add a 128-token uncompressed sliding window because causally compressed blocks
cannot expose tokens inside the query's unfinished block. Partial RoPE is applied to 64 dimensions of
queries and K/V, then inverse-position handling is applied to output dimensions because each compressed
entry also acts as a value. A learned sink logit lets a head allocate less than unit total mass to real
tokens.

## Model configurations

| Setting | V4-Flash | V4-Pro |
| --- | ---: | ---: |
| Layers / width | 43 / 4,096 | 61 / 7,168 |
| Total / active params | 284B / 13B | 1.6T / 49B |
| CSA compression `m` | 4 | 4 |
| CSA top-k | 512 | 1,024 |
| HCA compression `m'` | 128 | 128 |
| Query heads | 64 | 128 |
| Routed / active / shared experts | 256 / 6 / 1 | 384 / 6 / 1 |

Flash begins with two SWA layers; later layers alternate CSA and HCA. Pro begins with two HCA layers,
then alternates. Every Transformer block has an MoE FFN; early MoEs use hash routing.

## Precision and cache system

- RoPE cache dimensions stay BF16; other K/V dimensions use FP8, nearly halving storage versus BF16.
- The CSA indexer's attention computation uses FP4. Routed expert weights receive FP4 QAT after base
  training.
- The runtime separates fixed-size state for the SWA window and incomplete compression tails from paged
  compressed CSA/HCA entries. Cache block size respects the least common multiple of the two compression
  rates.
- On-disk prefix caching stores completed compressed entries. Unfinished tails and the final SWA region
  are recomputed on a hit.
- Relative to BF16 GQA8 with 128-dimensional heads, the paper estimates the V4 cache near 2% at 1M.

## Training and stability

Flash trains on 32T tokens and Pro on 33T. Both progress 4K → 16K → 64K → 1M. Flash uses dense
attention for its first 1T tokens and introduces sparse attention at 64K after an indexer warm-up; Pro has
a longer dense phase. This is native long-context training but still uses staged densification/sparsification.

Muon updates most matrices; AdamW remains for embeddings, output head, RMSNorm, and selected static
parameters. mHC expands the residual stream and constrains mixing matrices with Sinkhorn-Knopp
normalization for stable propagation.

Two pragmatic safeguards matter:

- **Anticipatory Routing** temporarily uses routing indices computed by an older model state when a loss
  spike is detected, decoupling router/backbone feedback at about 20% overhead while active.
- **SwiGLU clamping** constrains the linear branch to `[-10, 10]` and caps the gate branch at 10.

## Evidence

The report's base-model comparisons use a unified internal harness and show Flash outperforming V3.2 on
many tasks despite fewer active parameters, while Pro improves further. Post-trained models report strong
MRCR and CorpusQA behavior through 1M. These outcomes combine 32T+ data, model scale, architecture,
optimizer, and post-training; only the explicit FLOP/cache accounting transfers mechanically.

## What matters for Speck

For Speck, V4 suggests a long-term exact-attention direction: local raw tokens + dense aggressive
summaries + sparse less-compressed summaries. Build it only after testing simpler MLA, CLA, and
MoBA/DSA components independently. Compression quality, not just selector recall, becomes the central
failure mode.

## Limitations and cautions

- The authors explicitly describe the architecture as complex and plan to distill it to fewer essential
  pieces.
- Most component gains are not isolated at final scale, and no seed intervals are reported.
- Custom compressors, MQA kernels, mixed-precision cache, context parallelism, mHC, Muon, and MoE
  runtime make consumer implementation expensive.
- The causes of routing instability and why clamping works remain insufficiently understood.
- A 1M supported window still needs independent per-length, multi-hop, and real-document evaluation.

## Bottom line

DeepSeek-V4 is the current “compress, then sparsify” endpoint: keep a dense coarse global view, retrieve a
small precise subset, and preserve raw local context. It is a roadmap for mature long-context systems, not
a monolithic architecture Speck should copy before its simpler axes are settled.
