# Hymba: A Hybrid-Head Architecture for Small Language Models

- **Paper:** [arXiv:2411.13676](https://arxiv.org/pdf/2411.13676)
- **Version reviewed:** v1, 20 November 2024
- **Primary topic:** small-model parallel heads, meta tokens, SWA, and KV sharing

## Central claim

Hymba combines attention and Mamba-style SSM heads in parallel inside every mixer. Attention provides
high-resolution “snapshot” memory; the SSM provides compressed, fading memory. The design then stacks
three cache reductions: most attention is sliding-window, adjacent layers share K/V, and fixed learnable
meta tokens absorb attention-sink behavior and initialize both memory paths.

This paper is unusually relevant at Speck's size because its main controlled studies are 300M–1B rather
than only frontier MoEs.

## Hybrid-head mixer

Both branches receive the same projected input. The attention and SSM outputs are separately normalized,
rescaled by learned per-channel vectors, averaged, and passed through an output projection. Separate
normalization is necessary because the raw SSM output is consistently larger.

Ablating individual branches shows task- and layer-dependent importance. In the reported HellaSwag
analysis, removing an SSM branch costs more on average than removing an attention branch, while some
inputs rely on the opposite path. This is evidence for complementary representations, not proof that
parallel fusion always dominates layerwise mixing.

## Cache design

Pure SWA in every layer loses more than 20 points on the paper's recall suite. Restoring global attention
only in the first, middle, and last layers recovers most recall while keeping other layers local. Adjacent
layers share K/V projections and cache entries, roughly halving the attention cache again.

The 300M/100B-token roadmap reports:

| Step | Commonsense | Recall | Throughput | FP16 cache at 8K |
| --- | ---: | ---: | ---: | ---: |
| sequential attention added to Mamba | 44.07 | 45.16 | 776 tok/s | 156.3 MB |
| parallel multi-head hybrid | 45.19 | 49.90 | 877 tok/s | 148.2 MB |
| mostly local attention | 44.56 | 48.79 | 2,400 tok/s | 41.2 MB |
| cross-layer KV sharing | 45.16 | 48.04 | 2,757 tok/s | 39.4 MB |
| add meta tokens | 45.59 | 51.79 | 2,696 tok/s | 40.0 MB |

## Meta tokens

A fixed set of learned embeddings is prepended to every prompt. They participate in attention and SSM
updates, and their K/V/state contribution can be precomputed for inference. The paper interprets them as:

- a learned target for probability mass that would otherwise accumulate on BOS;
- an “attend to nothing” option, generalizing the idea of an added zero softmax denominator;
- learned initialization for attention cache and recurrent state;
- reusable world-knowledge cues that activate differently for article, math, and code inputs.

Meta tokens reduce attention entropy and improve both recall and commonsense in the roadmap. The
interpretation is suggestive; the experiment does not prove that they store distinct semantic knowledge.

## Main evidence

- A matched 1B/100B-token comparison reports Hymba best on language-model loss and on the average of
  recall and commonsense tasks versus Mamba-2, Mamba-2+FFN, Llama-style attention, and Samba-style
  layerwise hybrid.
- The 1.5B model sees 1.5T pretraining tokens plus an 8K extension. Against Llama-3.2-3B, the paper
  reports `+1.32` average accuracy points, `11.67×` smaller cache, and `3.49×` throughput in its A100,
  8K, batch-128 measurement.
- A matched model trained at 1K and fine-tuned at 4K performs better than Mamba-2 and Llama-style
  controls on needle tests through 16K, including middle positions.

## What matters for Speck

Hymba supplies three separable experiments: parallel fusion, global/local attention placement, and
meta-token initialization. They should not be imported as one bundle.

Test meta tokens against the sigmoid SDPA output gate because both claim to solve forced attention or
sink behavior in different ways. Also compare shared-KV local/global attention with CLA2; the mechanisms
overlap but have different normalization and pipeline implications.

## Limitations and cautions

- Published throughput chooses the largest non-OOM batch per model, so memory savings and kernel speed
  are deliberately combined.
- Public baseline models have very different token budgets. The matched 100B-token study is more
  informative for architecture.
- Meta tokens consume context positions and introduce fixed state that must be represented in exports.
- The global layers still make cache and prefill grow with context, and 8K is far below Speck's target.

## Bottom line

Hymba is the strongest small-model case for parallel hybrid heads and demonstrates that SWA, cross-layer
KV sharing, and learned initialization can stack. Its components should be ablated independently against
KDA and gated attention.
