# KV-compression attention dynamics: retention, reachability, and rigidity

- **Paper:** [arXiv:2603.01426v1](https://arxiv.org/abs/2603.01426v1)
- **Version reviewed:** v1, 2 March 2026
- **Runtime named by paper:** [NVIDIA KVPress](https://github.com/NVIDIA/kvpress)
- **Primary topic:** interpret post-hoc KV eviction as perturbing token-level routing rather than only
  stored memory

## Framework

The paper distinguishes stored-token retention, accessibility through attention routes, and effective
utilization. It defines a layered token-attention graph from thresholded attention edges and a compressed
graph that removes edges through pruned KV positions. A token-route lottery ticket is defined as a
surviving subgraph with a path from the query to at least one answer-relevant token and sufficient model
behavior.

Its task-aware Global Eviction Ratio (GER) is the fraction of gold answer-token positions absent from
every head. A separate head-consensus statistic counts distinct per-head argmax tokens; low diversity is
used as a proxy for representational rigidity. The paper argues that extreme compression fails through
global erasure or survive-but-not-used rigidity.

The formal results have strong assumptions. The redundant-route bound assumes independent per-head
survival. The erasure proposition assumes the answer was not already copied into surviving residual or
KV representations. Token-route-lottery-ticket sufficiency is definitional; the experiments do not
identify a minimal sufficient subgraph or intervene on one route at a time.

## Evidence boundary

The synthetic suite covers direct attributes, manipulation, repeated/multiple entities, coreference,
long contexts, and generated multi-hop chains. Contexts span roughly 150–1,300 words; the Hops set has
100 generated passages and 1,600 queries. Uncompressed Hops F1 is already only 31.54 for Llama-3-8B and
27.13 for Qwen-2.5-7B, limiting attribution of every compressed failure to route removal.

Experiments use five instruction models from 3B to 14B, one RTX A6000, greedy decoding, Expected
Attention scores, and FINCH chunk or AdaKV head-wise pruning through KVPress. Compression ranges from
10% to 90%. Question-agnostic mode prunes context before the question; question-aware mode supplies the
list of candidate questions before pruning. The paper reports a high-compression performance cliff
aligned with GER and architecture-specific consensus patterns.

No dedicated artifact/code repository, immutable KVPress revision, model revision, dataset bundle, or
per-example analysis output is declared on the arXiv page. The authors explicitly leave real long-
document QA, RAG, code, tools, tighter operator guarantees, and architecture co-design to future work.

## What matters for Speck

The broad ideas that cache retention differs from accessibility, that multi-hop routes are fragile, that
all-head erasure predicts hallucination, and that routing diversity matters are already occupied. Speck
cannot present those as N2 novelty. GER, head consensus, and token-graph reachability are mandatory
diagnostic baselines if reproducibly materialized.

The remaining narrower distinction is conjunctive completeness. GER measures a fraction of answer tokens
globally erased, while its reachability event requires an intersection with at least one answer-relevant
position. N2 instead proposes separately frozen route and payload source sets and asks whether every
causally required set remains available. That can matter only if it predicts held-out failures beyond
GER/reachability/consensus and survives equal-budget restoration interventions on real documents and
multiple scales.

## Bottom line

Routing-aware KV-compression diagnostics and multi-hop fragility are not novel. All-required-source
conjunction remains only a plausibly narrower hypothesis; without incremental prediction and causal
restoration beyond this framework, N2 must be rejected.
