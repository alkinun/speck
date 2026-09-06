# 115 — Crossed-factor finalist analysis v2 qualified

## Checked implementation

The collector and target lock now emit v2 records pinned to the corrected plan. The analyzer groups
three independent seed differences within each of the two fixed data orders, applies the df=2 bound
separately, and requires both strata plus every cell to pass. Every source repeats the same rule.

Pooled six-cell df=5 output and per-seed two-order means are labeled descriptive only. Secondary FLOP,
steady-time, and uncensored target summaries are stratified by order. Seven fixtures pass, including a
new adversarial case where a pooled zero mean cannot hide one -0.02 passing order and one +0.02 failing
order. All 84 configs remain byte-identical.

## Decision

V1 analysis and automation authority are false. V2 collector, lock, crossed-factor inference, and
stopping behavior qualify. Runtime preflight remains applicable because configs did not change.
Training stays blocked until automation v2 and a post-HELMET live qualification exist.

## Artifact

- [Finalist analysis qualification v2](../results/Speck-Paper1/finalist-analysis-qualified-v2.json)
