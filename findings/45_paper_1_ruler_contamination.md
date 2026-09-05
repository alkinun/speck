# 45 — Paper 1 RULER contamination audit and v1 failure

## Question

Do exact RULER prompt or answer-adjacent token sequences occur in the packed data windows frozen for
the Paper 1 proxy, and can RULER v1 support an uncontaminated absolute capability claim?

## Frozen audit

The preregistered audit scanned exactly three 131,072,000-token packed windows: pair 0/seed 42 at
offset 0, pair 1/seed 43 at offset 536,870,912, and pair 2/seed 44 at offset 1,073,741,824. The
393,216,000-token scan preserved 4,096-token training-example boundaries and never invented adjacency
between examples or microbatches.

All 7,800 qualified RULER cases across 13 tasks and six lengths were tokenized with the packed-data
SentencePiece processor. The first case in every task/length cell also matched the pinned Transformers
tokenizer, for 78 parity checks. The scan deduplicated 1,300 full-prompt probes, 3,180,930
answer-anchored 32-token probes, and 15,780 fixed-position context probes while retaining all reference
counts.

## Result

No complete prompt matched, but the frozen zero-match answer-anchor gate failed: 28 unique
answer-anchored patterns occurred 29 times in the sampled training windows. Another 42 context
patterns matched and remain descriptive, as preregistered. The answer matches occurred in `dclm_edu`
(20), `pes2o` (5), `fineweb_edu` (2), `finemath_4plus` (1), and `wikimedia` (1); the context matches
occurred in `dclm_edu` (33), `fineweb_edu` (8), and `pes2o` (1).

The separate disposition pass reconstructs every reference behind each matched hash, including those
not embedded in the size-bounded raw audit. It localizes every critical match to RULER's `qa_1` and
`qa_2` source-document tasks. Twenty-eight cases are directly affected, spanning all 12 task/length
cells; both 600-case tasks are therefore quarantined in full. The other eleven tasks have no detected
critical match in these exact proxy windows, but that observation neither proves their data clean nor
makes the 13-task v1 manifest pass.

## Decision

RULER v1 fails its contamination gate. `qa_1` and `qa_2` are quarantined from uncontaminated and
absolute-capability claims. The zero-match threshold is unchanged, and candidate execution under v1
is not authorized. A new evaluation-manifest version must be frozen before candidate results. It must
either regenerate/replace the quarantined cells using inputs demonstrably disjoint from the training
data and rescan them, or remove them explicitly and add an independent source-document QA guardrail.

Matched candidate/control results on the quarantined tasks may only be disclosed as contaminated
sensitivity analysis. Shared training windows can help interpret a paired contrast, but cannot validate
an absolute score.

## Limits

This audit covers the exact 393,216,000 tokens consumed by the proxy design, not all past or future
training data. Exact overlap does not establish which upstream corpus supplied a packed example. Zero
detected overlap does not prove absence. NoLiMa and HELMET remain unscanned because their separately
recorded data/legal qualifications are incomplete.

## Artifacts

- [Frozen audit protocol](../research/paper-1/contamination_v1.json)
- [Failed raw audit](../results/Speck-Paper1/contamination-ruler-v1.json)
- [Frozen disposition protocol](../research/paper-1/contamination_disposition_v1.json)
- [Task-level disposition](../results/Speck-Paper1/contamination-ruler-v1-disposition.json)
- [Audit runner](../scripts/contamination_audit.py)
- [Disposition runner](../scripts/contamination_disposition.py)
