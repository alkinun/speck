# 78 — Adaptive cache salience-acquisition readiness

## Boundary

The allocation, equal-group GQA reduction, and conservation-safe apportionment references are now
qualified. They accept salience as input; none establishes how a real Speck checkpoint should produce
that salience or how unequal per-head cache lengths should execute. This gate keeps those questions
separate.

## Primary-source result

SnapKV scores an older prompt prefix from queries in a trailing observation window, aggregates those
query weights, pools over prefix positions, selects prefix identities per head, and concatenates their
original KV payloads with the complete observation window. Its settings are not universal: reported
windows span 16/32/64 and pooling kernels 5/7/13 across model/task suites. Its question-position test
still places the question inside the prompt; that is question-visible evidence, not reusable
question-agnostic prefix caching.

The frozen Speck definition computes each observation query's causal softmax over the actual visible
prompt, slices prefix columns without renormalizing, averages observation rows, applies the already-
qualified equal-group mean reduction, and only then pools positions. Pooling changes selection scores,
never the original KV payload or token identity. The complete observation window is a separately counted
physical-state cost.

## Local implementation blockers

Speck's native attention calls fused SDPA and returns projected outputs without attention weights. Its
Transformers wrapper explicitly rejects `output_attentions`. `AttentionState` is one fixed-capacity
rectangular tensor per layer/head with chronological ring eviction; it cannot represent arbitrary token
identities or unequal head lengths without padding away the claimed state benefit. No qualified
flattened/variable-length target kernel exists.

The next valid implementation is therefore an offline, read-only, checkpoint-specific attention probe.
It must reproduce production SDPA outputs on CPU and GPU while materializing only requested observation
rows, and it may not change `Attention.forward` or training. The three-pair Paper 1 result must first
select or reject the candidate parent checkpoint.

## Frozen successor sequence

After probe parity, acquisition separates question-visible full-prompt tails, genuinely reusable
context-only prefixes compressed before any question exists, and a future-query hindsight oracle with
no deployable authority. Subsequent stages isolate pooling, window/kernel geometry, and uniform versus
adaptive versus safeguarded allocation. Cache serialization, absolute positions, generated suffixes,
resume parity, overflow, and exact logical/allocated bytes are mandatory.

All-required-source recall must add held-out predictive value beyond retained attention mass and
attention-output error. A causal intervention restores one missing required source while evicting a
matched irrelevant span at the same physical budget; predicted completeness and task output must recover
together. Diagnostics cannot substitute for end-task quality, and a systems claim requires at least 10%
end-to-end improvement on named hardware after all state and probe costs are included.

## Decision

Attention instrumentation, visibility modes, pooling, geometry, variable-length state, kernels,
evaluation, and causal diagnostic value are all unqualified. No implementation, evaluation, training,
novelty change, architecture promotion, or paper-scale run is authorized by this gate.

## Artifacts

- [SnapKV primary-source note](../papers/29_snapkv.md)
- [Salience-acquisition readiness gate](../research/paper-1/adaptive_cache_salience_readiness_v1.json)
