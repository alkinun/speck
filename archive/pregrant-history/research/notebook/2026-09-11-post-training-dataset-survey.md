# 2026-09-11 — SpeckChat lineage and flagship instruction-data survey

## Context

The owner selected an Instruct flagship and requested deep dataset research, including the sources
behind both early SpeckChat collections. Reusing good source components is compatible with developing
a fresh flagship recipe; the old size-specific mixtures and 2K selection are not inherited authority.

## Work performed

Reconstructed the 300K and 500K mixtures from code and pinned release cards. Collected cards/metadata
for 32 repositories and bounded illustrative samples (158 viewer rows and 13 direct JSONL rows).
Preserved immutable revisions, header receipts, hashes, partial-view flags, and discovery failures.
Inspected schemas and selected contents, including concrete demonstration, test-oracle, preference,
language, and message-mask problems. This is source discovery, not a representative quality audit.

## Decisions

Recorded the owner's Base + Instruct decision in the pipeline draft. Recommend prioritizing selected
LMSYS/Magpie/Hermes/UltraChat components together with targeted SmolTalk, Nemotron, verified math/code,
and evidence-bearing grounded datasets. No source weights or corpus approvals were frozen.

The most consequential adapter requirement is per-message supervision for Nemotron v3: reconstruct
withheld seed prompts, omit separate reasoning content, and honor final-turn eligibility. Code data
needs verifier validation as well as answer checking. Preserve original family identities across
shared mixtures, and treat historical No Robots test exposure correctly.

## Evidence and links

- [Dataset survey](../flagship/POST_TRAINING_DATA_SURVEY.md)
- [Snapshot identities](../flagship/post_training_dataset_survey_v0.json)
- [Current Instruct pipeline](../flagship/POST_TRAINING.md)
- [SPE-176](https://linear.app/openspecklabs/issue/SPE-176)
- [SpeckChat1 builder](../../scripts/speckchat1_prepare.py)
- [SpeckChat2 builder](../../scripts/speckchat2_prepare.py)

## Open questions

- Which sources supply the most unique, verified useful tokens after global family deduplication?
- How much accepted English long-context data survives source, evidence, and length checks?
- Do original Hermes/UltraChat subsets outperform their prose-only derivatives for the flagship?
- Which sources' actual terms fit the intended public release and synthetic-data disclosure policy?

## Next actions

- Prepare a bounded, stratified qualification wave for the shortlisted components.
- Implement correct source adapters and per-message target eligibility before accepting Nemotron v3.
- Measure quality/length yield and overlap before freezing the initial SFT mixture.
- Keep the active training/budget contract unchanged until an executable successor is reviewed.
