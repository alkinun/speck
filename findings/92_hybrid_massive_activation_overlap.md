# 92 — Hybrid massive-activation and placement overlap

## Direct evidence

Massive Activations in Hybrid Linear Attention LLMs defines pre-attention spikes (PAS) and inter-spike
plateaus (ISP). Across five recurrent mixers, six density endpoints/configurations, two controlled
scales, five domains, and public hybrids up to 397B parameters, massive activations align to the layers
immediately before full attention. Paired bootstrap intervals support increasing inter-spike retention
as attention becomes denser.

The source also trains a matched 24-layer GDN model from scratch with one full-attention layer at depth
4, 12, or 20. Middle and late attention beat early attention on most retrieval tasks. Yet all three
models have 99.53--100% PAS alignment and similar language/commonsense performance. Binary PAS
alignment therefore does not rank placement quality. Spike magnitude, retrieval quality, and
full/GDN-gate interventions likewise do not form a task-invariant ordering.

## Decision

PAS, ISP, sink-conditioned activation tracing, early/middle/late placement, and module-specific gate
interventions are mandatory N1 baselines, not Speck novelty. A prospective arbitrary-layout predictor
remains technically unperformed, but the saturated diagnostic supplies direct counterevidence to the
obvious formulation. N1 has a lower prior and independent review should explicitly consider retiring it.

No placement protocol or experiment is authorized. The frozen baseline remains independent.

## Artifacts

- [Paper note](../papers/40_massive_activations_hla.md)
- [Novelty landscape v7](../research/paper-1/novelty_landscape_v7.json)
