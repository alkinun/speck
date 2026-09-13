# 46 — Post-contamination RULER v2 successor manifest

## Question

Can the RULER portion of the long-context evaluation be repaired after v1's critical overlap failure
without changing thresholds, inspecting model outputs, or quietly dropping source-document QA coverage?

## Alternatives

Regenerating `qa_1` and `qa_2` from another public natural-text corpus would still require evidence
that the new corpus is disjoint from the broad web-derived training mixture. It would also replace
official RULER case semantics with a local variant. There is no current evidence strong enough to make
either claim.

The selected response is narrower. V2 retains the eleven official synthetic tasks whose prompts,
answers, seeds, lengths, case identities, and upstream per-task scorers are already qualified. It
quarantines both source-document QA tasks in full. Natural source-document QA remains a required but
separate guardrail through HELMET's pinned RAG and long-QA categories, subject to their own rights,
data, and contamination qualification.

## Version boundary

Evaluation manifest v1 is preserved at repository revision
`8fbdefe9fefdc62b19c469b0cf2d79ed057caac3` with SHA-256
`b5439be0ffb37c8b8b46aec9791068b8e145dd5292419e90a335a447969b21ab`. It records the failed
answer-anchor gate. V2 explicitly supersedes that identity; it does not rewrite v1 or mark it passed.
No candidate or control output existed when either the quarantine or v2 scoring rule was frozen.

## V2 primary matrix

The primary tasks are eight NIAH variants plus variable tracking, common-word extraction, and
frequent-word extraction: `niah_single_1`, `niah_single_2`, `niah_single_3`, `niah_multikey_1`,
`niah_multikey_2`, `niah_multikey_3`, `niah_multivalue`, `niah_multiquery`, `vt`, `cwe`, and `fwe`.
At 100 cases across 4K, 8K, 16K, 32K, 64K, and 128K, this yields 6,600 frozen primary cases.

The primary score at each length is the unweighted mean of those eleven upstream exact-match task
accuracies. `qa_1` and `qa_2` contribute zero weight and may only be disclosed separately as
contaminated sensitivity analyses. This is a local v2 aggregate, not an official 13-task RULER score.

## Guardrail and decision

HELMET RAG and long-QA are the explicit source-document QA guardrail. Their currently blocked data
and rights state remains a failed/missing release gate; once available, their exact prompts must be
scanned against the applicable training inputs before supporting an absolute claim. A critical match
there will fail the guardrail rather than trigger another silent metric change.

RULER v2 is frozen, but candidate execution remains blocked on the other evaluation and candidate
export gates. The eleven retained tasks have no detected critical match in the sampled proxy windows;
they are not described as contamination-free, and v2 cannot support source-document reasoning by
itself.

## Artifacts

- [Active evaluation manifest v2](../research/architecture-promotion-v1/evaluation_manifest.json)
- [Failed v1 audit](../results/Speck-Paper1/contamination-ruler-v1.json)
- [Task-level disposition](../results/Speck-Paper1/contamination-ruler-v1-disposition.json)
- [RULER source/case contract](../research/architecture-promotion-v1/external/ruler_v1.json)
- [HELMET contract](../research/architecture-promotion-v1/external/helmet.json)
