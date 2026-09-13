# You Only Cache Once: Decoder-Decoder Architectures for Language Models

- **Paper:** [arXiv:2405.05254](https://arxiv.org/pdf/2405.05254)
- **Version reviewed:** v2, 9 May 2024
- **Primary topic:** one global KV cache reused by many decoder layers

## Central claim

YOCO divides the network into a self-decoder and a cross-decoder. The self-decoder uses bounded-state
attention to contextualize the sequence and produces one global K/V cache. Every cross-decoder layer
reuses that cache with separate queries. This reduces global cache memory from proportional to
`layers × sequence` to approximately one sequence plus bounded self-decoder state.

Unlike CLA, which shares K/V within small adjacent groups, YOCO shares one representation across the
entire second half of the model.

## Architecture

For an `L`-block model, the first half is the self-decoder and the second half is the cross-decoder. The
self-decoder can use sliding-window attention or the paper's gated retention. Its final representation is
projected once into global K and V. Each later block applies causal cross-attention to those tensors,
followed by its own FFN.

The gated-retention variant adds data-dependent decay to RetNet. It performs better than the SWA
self-decoder in the paper's scaling curves, suggesting that global cache quality depends on how well the
first half contextualizes each token before the cache is frozen.

## Why prefill is cheaper

During prompt prefill, all prefix tokens must pass through the self-decoder to construct the shared cache.
For the cross-decoder, only the final prompt position is needed to predict the first generated token;
earlier prefix outputs from those layers are not K/V inputs to later cross layers. The runtime can therefore
skip most cross-decoder prefix computation. During autoregressive generation, new tokens update the
self-decoder state/cache and query the one shared global cache through all cross layers.

The paper gives cache complexity `O((N + L)D)` for YOCO versus `O(LND)` for a Transformer, where `N`
is sequence length, `L` layer count, and `D` hidden/cache width.

## Evidence

- YOCO-3B has 2.83B non-embedding parameters and is trained to 1.6T tokens under a StableLM-like
  recipe. Its aggregate zero-shot result is competitive with the 3B Transformer comparison.
- Scaling runs from 160M to 13B show similar loss trends to an optimized Llama-style Transformer;
  gated-retention YOCO is better than SWA YOCO in the reported fits.
- Context extension uses 64K, 256K, and 1M stages with 6B, 4B, and 1.5B tokens respectively. The 3B
  model reports near-perfect single-needle accuracy through 1M and competitive multi-needle results at
  128K.
- On the paper's hardware/profile, 512K Transformer prefill falls from about 180 seconds to under six;
  YOCO is `2.87×` faster even at 32K and `71.8×` at 1M.
- At 1M the reported end-to-end throughput is 43.1 versus 4.5 tokens/s (`9.6×`). The projected cache
  reduction grows with depth and reaches about `80×` for the paper's 65B configuration.

## What matters for Speck

YOCO is a more radical alternative to periodic global layers. It asks whether one carefully contextualized
global memory is enough for all later reasoning layers. For a small model, that could eliminate most
global cache duplication while retaining exact access to every token.

The high-value Speck experiment is a matched decoder-decoder arm with a KDA self-decoder and gated GQA
or MLA cross-decoder. Compare it with CLA2 and independent periodic attention on multi-hop tasks. If
successive reasoning layers need to construct new K/V representations of the context, YOCO's single cache
will be the bottleneck.

## Limitations and cautions

- Cache memory falls much more than attention compute during decode: every cross-decoder layer still
  attends over the full growing cache for each new token.
- Single-needle retrieval does not prove that one frozen global representation supports iterative
  multi-hop reasoning.
- The prefill speedup depends on a specialized execution graph that truly early-exits the cross-decoder.
- A 1M context was reached with explicit long-context continuation; it is not zero-shot architecture
  extrapolation.

## Bottom line

YOCO is the maximum-sharing cache design in this collection. It can make prefill and memory dramatically
cheaper, but its single global representation must be stress-tested on tasks where later layers need
different views of the same long context.
