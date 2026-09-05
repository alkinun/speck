# 79 — N2 direct-overlap audit and scope reduction

## New direct overlaps

Two full-text audits materially narrow the proposed all-required-source diagnostic.

HeadKV-R2 already constructs retrieval-reasoning examples, scores attention to the whole correct-answer
span, builds a static head-importance profile, and redistributes a global cache pool across layer/head
cells. It evaluates LongBench/LooGLE contextual QA and multi-needle bAbI Reasoning-in-a-Haystack. Thus,
reasoning-aware head importance, global head allocation, and multi-hop cache evaluation are prior art.

The March 2026 attention-dynamics paper already separates retention, accessibility, and utilization. It
defines answer-token Global Eviction Ratio, head consensus, compressed token-attention graphs, and
query-to-answer reachability; it studies question-aware and question-agnostic pruning, multi-hop chains,
five 3B–14B models, and compression through 90%. Thus, routing rather than storage, survive-but-not-used
failure, all-head answer erasure, multi-hop route fragility, and generic token-route reachability are
also prior art.

## Remaining distinction

The overlap is close but not identical. HeadKV-R2 deliberately scores the correct answer span rather
than every prerequisite reasoning statement. GER measures the fraction of answer tokens missing from
all heads, while the routing paper's reachability event requires a path to at least one answer-relevant
position. Neither audit labels multiple independently necessary route/payload groups, requires their
conjunction, tests incremental prediction beyond the combined baselines, or restores one missing group
while evicting a matched irrelevant group at constant physical budget.

N2 is therefore renamed the *conjunctive required-source survival law*. It may survive only if every
required group is annotated independently of model outputs; a frozen nested predictor adds calibrated
held-out value beyond HeadKV-R2 importance, GER, token-graph reachability, consensus, mass, output error,
and single-source recall; and fixed-budget single-group restoration recovers both the diagnostic and
task outcome across real documents, three scales, and a held-out compression family.

## Evidence cautions

HeadKV-R2 profiles each model with only two manually designed reasoning patterns expanded across length
and depth. Its bAbI answer is present in the inserted needle, and retrieval-only profiles remain strong.
The routing paper uses mostly synthetic contexts around 150–1,300 words; uncompressed Hops F1 is only
27–32; its route propositions assume independent head survival or no prior information leakage. It does
not identify minimal sufficient routes or report a one-source causal restoration intervention. These
limits preserve a testable distinction but do not prove novelty.

## Decision

No novelty gate passes. The broad N2 language is rejected and must not reappear in claims. The narrow
conjunctive hypothesis remains `plausibly_narrower`, pending immutable code/data audits, backward and
forward citation review, a closest-overlap table, independent expert review, and prospectively frozen
held-out causal evidence.

## Artifacts

- [HeadKV-R2 full-text note](../papers/30_headkv.md)
- [KV-compression routing full-text note](../papers/31_kv_compression_physics.md)
- [Versioned novelty audit](../research/paper-1/novelty_landscape_v2.json)
