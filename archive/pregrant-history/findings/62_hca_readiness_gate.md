# 62 — HCA readiness gate before implementation

## Question

Is “add HCA” specific enough to justify implementation or training after exact-cache representation is
selected?

## Finding

No. DeepSeek-V4 establishes that dense coarse summaries can complement selective higher-resolution
retrieval, but its rate 128, compressor, quality, cache ratios, and million-token system do not transfer
mechanically to Speck. Speck has not selected the parent backbone or exact-cache representation, and no
local compressor, causal state, or production-shaped kernel exists.

The gate now specifies the missing causal behavior. A summary covers one non-overlapping `m`-token
block and becomes visible only after its entire source block is available. Partial blocks must never
leak future tokens. Prefill and decode must agree on block identity and emission; prefix reuse,
eviction, serialization, and resume must retain or deterministically reconstruct the partial tail.
KDA remains the fixed local path. Adding raw SWA is a later intervention.

## Required compressor isolation

Rate cannot be studied while silently changing compressor capacity. A mechanism-scale precursor must
compare a parameterless block mean, learned scalar position weights shared across channels, and learned
per-channel position weights. It holds the parent, one rate, attention placement, data, and evaluation
fixed, then freezes one compressor on independent held-out sufficiency and quality evidence. It has no
promotion authority.

Only afterward does a conditional powers-of-two rate grid test 32, 64, 128, and 256 tokens per summary.
At 4K these retain 128, 64, 32, and 16 completed summaries; at 128K they retain 4,096, 2,048, 1,024,
and 512. This measures a Speck rate curve around the published 128 rather than importing it.

## Decision

HCA implementation and training remain blocked. Future accounting must include completed summaries,
partial-tail state, metadata, quantization scales, compressor compute, dense summary attention, and
dispatch. The prefill term remains `O(L²/m)`, not linear. A custom runtime must realize at least 20%
primary improvement and 25% state reduction while every quality constraint passes.

## Artifact

- [HCA readiness gate](../research/paper-1/hca_readiness_v1.json)
