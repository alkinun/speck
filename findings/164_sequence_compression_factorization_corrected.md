# 164 — HCA, CSA, and local attention now have one coherent factorization

The coordinated v2 successor preserves the valid HCA/CSA/raw-local v1 quality, causal, state, selector,
and systems requirements while removing their incompatible architecture assumptions.

Five layer types are explicit: exact global; HCA core and CSA core as mechanism-only diagnostics; and
source-faithful HCA-local and CSA-local. Each compressed-attention layer has exactly one compressor.
HCA and CSA complement one another through a schedule over depth, never as parallel branches in the
same layer.

Cache identity is now representation-aware. HCA entries are non-overlapping with stride `m'`. CSA
entries are emitted every `m` tokens but, after the first, combine previous/current halves with `2m`
support. Raw exact entries, CSA entries, and HCA entries remain distinct when their source spans
overlap. Only an exact duplicate within the same representation family may be removed.

HCA compressor isolation now has four arms with common KV/output machinery: mean, static scalar
position, static channel-position, and the exact dynamic hidden-score-plus-position source form. CSA
then isolates non-overlap versus two-half overlap, evaluates an oracle over compressed-entry mass, and
only then compares mean-key and learned indexers.

The eight-stage experiment order selects parents first, then HCA compressor/rate, raw local recovery,
CSA overlap and oracle/indexer, HCA/CSA depth schedule, and finally ratio/placement confirmation. The
official code covers full prefill and one-token decode only; every arbitrary chunk, request, prefix,
eviction, resume, and gradient behavior remains a local future gate.

The three v1 files are preserved but lose joint-factorization authority. V2 is unregistered; no parent,
operator, rate, window, selector, schedule, implementation, training, or promotion is selected.

## Artifacts

- [Joint factorization v2](../research/paper-1/sequence_compression_factorization_v2.json)
- [Qualification](../results/Speck-Paper1/sequence-compression-factorization-v2-qualified.json)
