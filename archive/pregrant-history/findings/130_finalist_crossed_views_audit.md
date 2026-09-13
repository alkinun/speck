# 130 — Crossed-factor audit of every finalist analysis view

## Beyond endpoint loss

The frozen v2 analyzer was audited without finalist outputs. Endpoint loss and every source guardrail
use separate `df=2` bounds over three seeds within each of the two fixed data orders, plus hard cell
guards. Fixed analytic-FLOP loss, fixed steady-time loss, and uncensored time-to-quality also retain
the same two order strata; their pooled six-cell summaries are descriptive only. Only endpoint and
source guards feed the final language-screen decision.

## Adversarial coverage

A new fixture makes time-to-quality improve under order 0 and regress under order 1 while the naive
pooled mean remains positive. Both opposing strata remain visible. Another fixture censors exactly one
pair and verifies that all time-to-quality bounds—both stratum and pooled—are suppressed rather than
forming a complete-case estimate. Existing tests retain the one-order endpoint failure, all-pair
censoring, exact-cell, target-lock, and collector gates.

The original analysis test file is itself hash-pinned by the v2 qualification. A combined validation
caught an attempted edit before commit; it was restored byte-for-byte, and the new tests were moved to
a separate append-only module. The full paper validator then passed with the original implementation,
plan, CLI, and qualification-test hashes unchanged.

## Reporting-only residual

The endpoint pooled block carries an explicit `descriptive_only_naive_df_5` field. The three secondary
pooled blocks are named `pooled_descriptive` and are unambiguously governed by the pinned plan, but do
not repeat that authority field inline. This is a result-consumer hardening opportunity, not a numeric
or inferential flaw. Changing it now would invalidate the active analyzer hash, so the redundant labels
are deferred until after the sequence.

## Decision

All inferential and censoring computations qualify under v2. Pooled inference remains forbidden. No
interim result access, analysis-v3 successor, analyzer edit, or pass/fail change is authorized.

## Artifact

- [Crossed-views audit](../results/Speck-Paper1/finalist-crossed-views-audit-v1.json)
