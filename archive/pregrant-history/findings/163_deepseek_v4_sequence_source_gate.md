# 163 — V4 source semantics require three sequence-readiness corrections

The initial official DeepSeek-V4-Flash release is pinned as one immutable nine-file source bundle,
including its 58-page report, config, inference model, TileLang kernel, and MIT license. Fresh network
validation reproduces every byte hash.

Three current assumptions are not source faithful:

1. HCA and CSA are separate interleaved layer types, not simultaneous branches. Flash has two initial
   pure-SWA layers followed by 21 CSA and 20 HCA layers; each CSA/HCA layer also has a raw local window.
2. CSA selects overlapping compressed KV entries, not raw blocks. After the first entry, each emitted
   representation has `2m` source-token support while entries are emitted every `m` tokens.
3. Raw local and compressed entries remain distinct KV identities in one attention softmax even when
   their underlying source spans overlap. Deduplicating by raw-token support deletes a real compressed
   representation.

HCA equations 20–23 also make the exact compressor explicit: separate KV and channel-wise score
projections, learned position/channel bias, per-channel position softmax, weighted pooling, and
post-pool RMSNorm. HCA v1's three simpler controls need an exact dynamic source arm before rate
selection. A no-local arm must be called `HCA core`, because the published operator includes local
attention.

The inference release precisely defines full-prefill and one-token decode tails. HCA retains one
non-overlapping projected KV/score tail; CSA retains prior/current halves for overlap. It does not
support arbitrary nonzero-start chunked prefill, and its nonpersistent buffers define no request,
prefix-hit, eviction, serialization, or resume identity. The top-level model is inference-only and has
no compressor/indexer training objectives or backward contract.

The audit authorizes coordinated HCA/CSA/raw-local v2 corrections only. It selects no published rate,
schedule, indexer, precision, position treatment, or local window, and authorizes no implementation,
training, or promotion.

## Artifacts

- [Official-release note](../papers/46_deepseek_v4_official_release.md)
- [Primary-source audit](../results/Speck-Paper1/deepseek-v4-sequence-primary-source-audit-v1.json)
