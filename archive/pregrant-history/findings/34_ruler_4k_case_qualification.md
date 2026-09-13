# 34 — RULERv1 4K deterministic case qualification

## Question

Can the pinned RULERv1/NeMo-Skills path materialize the complete 4K evaluation matrix from Speck's
retained source bundle with exact tokenizer accounting, deterministic reruns, and no network access?

## Frozen cell

The cell uses all 13 tasks in the active contract, 100 cases per task, seed 42, the pinned RULER data
generator at `c3f5e3b4`, the pinned NeMo-Skills conversion and scorer at `f4a3fd8`, and the attested
local Speck2 140M instruction export. NeMo-Skills reserves 50 tokens for its chat wrapper before
calling RULER. RULER then measures a five-token base-template placeholder. Qualification checks the
task length plus both values against 4,096; the largest accounted cases are exactly 4,096 tokens.

Runtime dependencies are locked, including the generator's undeclared SciPy dependency. The runner
preseeds all four source JSONs and both NLTK resources, enables the Hugging Face offline flags, and
injects an inherited `sitecustomize` guard that rejects IPv4/IPv6 DNS, connect, connect-ex, and send-to
operations. The guard first denies a deliberate connection self-test. Linux network namespaces are
not available in this environment, so the claim is specifically application-layer socket denial plus
zero observed generator attempts, not kernel namespace isolation.

## Upstream HotpotQA failure and repair

Untouched generation cannot finish the matrix. At `qa_2` example 7, the pinned loop tries 28 documents
and measures 6,318 tokens, tries 18 and measures 4,615, then subtracts its fixed increment and requests
eight documents even though the example has more required supporting documents. Python raises
`ValueError: Sample larger than population or is negative`. A bare `except` swallows it, the document
count can no longer decrease, and the process spins indefinitely. The current upstream `main` remains
the same pinned commit and contains the same loop.

The checked compatibility patch changes only this failure control flow. It catches `AssertionError`
rather than every exception, floors the next attempt at the number of required documents, and raises
if those required documents cannot fit. It changes `qa.py` from
`f00c0b59cf8698e90831bbf461c2c4f40f2a30f2469534c27455808c717281d0` to
`727c4c898daad0f73445f32c7a3381426d0da51df63a2ae9e0213f3faa558836`; the patch identity is
`5063fdb5f1737fdfd39f955e19a2cfa6418acd15cbcd24f513518295fa15f6a7`. It does not edit prompts,
answers, task source order, random seed, or scoring. With the repair, all 100 `qa_2` rows complete and
the maximum accounted length is 4,092.

## Result

Two complete generations produce byte-identical JSONL hashes for every task. The retained first pass
contains 1,300 cases and occupies 17MB including logs. The combined case-stream identity is
`d3b6e1da567cc6ad4853a545680de68dff5edf82a14ed9886c00c554ca731fba`. Every task has exactly 100
schema-valid rows; maximum accounted lengths range from 4,056 to 4,096. The generator recorded zero
network attempts in both passes.

The data are mixed-rights and stay under the local cache. Git retains the generator, compatibility
patch, source/code/tokenizer identities, per-task hashes, row counts, length extrema, and denial proof.

## Decision

The 4K case matrix qualifies. This is data-pipeline evidence, not a RULER capability result. No model
score is claimed, and RULER as a release gate remains blocked on the five longer case matrices plus
candidate-specific long-context exports and executions. Because every 4K reproducibility control
passes and retained size is modest, case generation may advance sequentially to 8K; each later length
must independently pass the same two-generation and network-denial gate.

## Artifacts

- [Case qualification report](../results/Speck-Architecture-Promotion-v1/ruler-cases-4096-qualified.json)
- [Compatibility patch](../research/architecture-promotion-v1/patches/ruler_qa_required_docs.patch)
- [Case runner](../scripts/ruler_case_prepare.py)
- [RULER contract](../research/architecture-promotion-v1/external/ruler_v1.json)
