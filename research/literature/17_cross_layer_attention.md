# Reducing Transformer KV Cache Size with Cross-Layer Attention

- **Paper:** [arXiv:2405.12981](https://arxiv.org/pdf/2405.12981)
- **Version reviewed:** v1, 21 May 2024
- **Primary topic:** sharing K/V activations across adjacent layers

## Central claim

Cross-Layer Attention (CLA) lets several consecutive attention layers reuse one layer's K/V projections
and cached activations while retaining separate queries and attention computations. CLA2, sharing each
cache across a pair of adjacent layers, cuts cache size in half on top of MHA, GQA, or MQA and gives the
best accuracy/memory trade-off in the paper's 1B and 3B studies.

CLA is distinct from shared attention **weights**. The second layer does not recompute K/V with a shared
projection; it consumes the first layer's already computed K/V tensors.

## Mechanism and systems properties

Only cache-producing layers have K/V projections. Subsequent layers in a sharing group produce their own
queries but attend to the group's shared K/V. A sharing factor `s` reduces the number of unique layer
caches by approximately `s`.

CLA is orthogonal to within-layer head sharing. The paper's best low-cache designs combine MQA with
CLA2. Removing K/V projections slightly reduces parameters, training FLOPs, and training activation
memory, but it does **not** reduce core decode attention reads: the same shared cache is read again by
each consuming layer.

Pipeline parallelism needs care. Layers sharing a cache must be colocated or the K/V tensor must cross a
stage boundary, potentially trading memory savings for communication.

## Evidence

The 1B design space trains Llama-like models on SlimPajama and compares nearly two orders of magnitude
of cache budgets. The 3B models have width 3,072, 32 layers, length 2,048, and see 100B tokens.

- At 1B, MQA-CLA2 with head dimensions 64–128 improves validation perplexity by `0.21–0.48` versus
  non-CLA models at the same cache budget.
- After learning-rate tuning, a 128-dimensional MQA-CLA2 model is only `0.04` PPL worse than plain
  128-dimensional MQA while using half the cache, and is `0.31` PPL better than the 64-dimensional MQA
  model with the same cache size.
- At 3B, the 128-dimensional MQA-CLA2 model reports PPL `9.34` and matches or beats the corresponding
  same-head-size plain MQA despite half the cache. The 64-dimensional study shows a similar pattern.
- CLA3 and CLA4 still improve over simple low-dimensional MQA but are worse than CLA2 at a matched
  cache budget.
- Keeping the first/last cache independent or concentrating cache-producing layers at the front/back is
  worse than uniform adjacent-pair sharing.

## What matters for Speck

CLA2 is a low-complexity `2×` reduction on the cache of Speck's global attention layers and stacks with
GQA, output gating, and potentially MLA. It is especially attractive before implementing a much more
radical global operator.

The correct experiment keeps query heads, head dimension, global placement, and total model parameters
controlled. Because CLA removes projection parameters, reallocate the difference to the FFN for a
parameter-matched view and also report the unadjusted deployment view.

Measure both cache capacity and latency. CLA can increase effective batch capacity without accelerating
the inner attention operation, so batch-1 decode may barely change while batched throughput improves.

## Limitations and cautions

- Experiments train at length 2,048, not at long context. Cache economics extrapolate mechanically;
  quality under 128K dependencies does not.
- Sharing beyond two layers degrades more, and nonuniform layouts do not help in the tested setup.
- The paper does not provide end-to-end serving benchmarks for a production long-context stack.
- Interaction with MLA is not tested; sharing already compressed latents may expose a different capacity
  bottleneck.

## Bottom line

CLA2 is one of the safest cache interventions in this set: pair neighboring global layers around a shared
KV tensor, retain separate queries, and verify the small quality trade-off locally. Expect capacity gains,
not an automatic attention-kernel speedup.
