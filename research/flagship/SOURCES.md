# Flagship source exploration and shortlist

Status: candidate registry v2 with bounded six-category technical qualification and project-owner
guarded-use approval, 2026-09-09. This
document records the broad source search before tokenizer and corpus freeze.
[`source_registry_v2.json`](source_registry_v2.json) pins the selected revisions and quotas and binds
[`release_and_data_use_policy_v1.json`](release_and_data_use_policy_v1.json). All 30 selected sources
have human guarded-use approval in [`source_rights_acceptance_v1.json`](source_rights_acceptance_v1.json).
This is source-use authority, not training authority: production operations, real firewall partitions,
tokenizer selection, packing, and launch receipt remain required.

Exploring many sources does not mean mixing all of them. The registry separates primary screens,
secondary diversity sources, tokenizer-only coverage, fallbacks, long-context sources, and datasets
held on terms or operational grounds. E1 compares a small number of controlled treatments; it does
not turn source discovery into an unbounded GPU search.

## 1. Main finding: The Stack v3 changes the code plan

The Stack v3 is an official 2026 Hugging Face Code release. Its current training card describes v3.1
as a 15.9 TB, roughly 3.6T-token source-code corpus spanning 173M repositories and 713 languages. It
contains file text inline, groups files by repository, applies language-agnostic near-deduplication,
quality heuristics and PII redaction, and reflects GitHub through 2025-08-07. V3.1 specifically fixes
an exact-duplicate leak in v3.0.

That makes `HuggingFaceCode/stack-v3-train` at commit
`8f3f25d86e44fd691428131efd17af75d4716499` the new primary raw-code candidate. We should not use it
unfiltered:

- admit `license_type=permissive` only; exclude `no_license`;
- exclude vendored files and forks, preserve repository and commit identity, and retain detected SPDX
  licenses for attribution;
- keep a declared language allocation rather than sampling 713 languages by raw frequency;
- serialize repositories and cleaned notebooks deterministically for base and long-context uses;
- scan again for secrets/PII and benchmark overlap because upstream detection is explicitly imperfect;
- maintain an opt-out/removal refresh procedure before final packing.

The 113.7 TB `stack-v3-full` corpus is research-only for us. It is unnecessary for grant 1 and would
create a storage and filtering program larger than the model program.

A real bounded profile now supports the operational choice. Twelve pinned shards (3.36 GB) exposed
252,860 repository rows and 2.84M files and produced 70.2 MB across eleven declared languages after
strict filtering. The adapter rejected 2.72M files by license type, 38,358 vendored files, four
ambiguous license references, and three high-confidence secret matches. All content receives a new
released-text SHA-256 because upstream `content_id` is preserved as provenance but does not
consistently equal plain SHA-1 of the released PII-redacted text. See
[finding 169](../../findings/169_stack_v3_bounded_qualification.md).

The security/language successor pins Gitleaks v8.30.1, excludes all 19 affected records from its
fully redacted report, applies a twelve-identifier conservative engineering license allowlist, and
English-filters extracted comments/docstrings plus Markdown. Its first 70 MB input failed the exact
tokenizer partition target and was preserved. A same-shard 85 MB successor changed no filters and
passed with 61.19 MB train and 6.42 MB evaluation after rejecting 1,318 license-policy records and
3,044 non-English-prose records. At this stage, training remained blocked on the later gates. See
[finding 170](../../findings/170_stack_v3_security_language_refinement.md).

The next frozen scan pins 2,278 HumanEval, MBPP, and BigCodeBench tasks. It removes 125 files
(1.79 MB) meeting the task-unique 13-gram or exact-field rule and discloses 66 additional 10-gram-only
sensitivities. No complete normalized benchmark field matched. The cleaned artifact still passes with
59.88 MB train and 5.94 MB evaluation. This closes only the bounded pinned-benchmark gate—not future
benchmark versions or the full Stack corpus. See
[finding 171](../../findings/171_stack_v3_code_decontamination.md).

## 2. Code treatments

E1S should compare exactly three treatments after CPU qualification:

1. **Stack v3.1 restricted:** permissive, non-vendor, non-fork repositories with a language and
   repository-quality allocation.
2. **Stack-Edu:** educational-score-filtered Stack v2 code, again restricted to acceptable file
   licenses. It is attractive for small-model quality but stores Software Heritage IDs rather than
   contents, so acquisition and missing-blob yield must be measured.
3. **Blend:** 60% Stack v3.1 restricted, 30% Stack-Edu, and 10% code prose/reference such as Python
   enhancement proposals. The exact percentages are a prior and freeze only after the sample audit.

`common-pile/stackv2_edu_filtered` is the operational/rights fallback and an independent validation
source. SmolLM's Python-Edu supplies narrow high-quality Python coverage for tokenizer training, not
the full code mixture. The Stack v2 dedup remains a fallback only. NVIDIA Nemotron-CC-Code is held:
it is gated by a data agreement and its downstream conditions require a separate acceptance review.

A bounded Stack-Edu SWH acquisition now passes for score-4+ Rust, Go, and SQL: 25.33 MB from 9,572
files and 7,394 repositories, with zero missing/failed blobs and exact SWH SHA-1 for every accepted
file. Metadata length differs for 634 valid blobs and is reported rather than substituted for the
cryptographic identity. Gitleaks finds three redacted findings across two records; removal,
contamination, and near-duplicate successors remain mandatory. See
[finding 172](../../findings/172_stack_edu_bounded_swh_sample.md).

The dedicated successor removes Stack-Edu's two Gitleaks-affected records. A Common Pile TypeScript
shard supplies only 11.78 MB under the same high-quality filters, so its tokenizer allocation drops
from 15% to 10% and Stack-Edu rises from 15% to 20%; the failed 20 MB target is preserved. A 12 MB
successor and zero-finding Gitleaks scan pass. Across the three bounded inputs, frozen
Stack-Edu→Stack-v3→Common-Pile precedence plus 128-permutation MinHash/LSH finds zero exact released-
text or verified ≥0.80 near-duplicate matches while retaining every quota. This is sample evidence,
not a full-corpus disjointness claim. See
[finding 173](../../findings/173_code_source_overlap_and_precedence.md).

The same pinned 2,278-task benchmark policy then removes 32 Stack-Edu files (241,650 bytes) and 14
Common Pile files (81,182 bytes). The v4 successors still exceed their 20M/2M and 10M/1M
train/evaluation quotas. They are strict subsets of the inputs that passed the bounded cross-source
comparison, so record removal cannot introduce a new duplicate. This closes the technical bounded
code-sample gates. The later guarded-use decision accepts the declared source terms and residual
metadata risk; production-scale global deduplication and acquisition cleanup/resume still block
training authority. See
[finding 174](../../findings/174_code_contamination_successors.md).

The tokenizer's provisional 100 MB code allocation is 55 MB Stack v3.1, 20 MB Stack-Edu, 10 MB
Common Pile Stack v2 educational code, 10 MB Python-Edu, and 5 MB Python language-design prose. This
is intentionally broader than the eventual stable mixture so the vocabulary is not overfit to one
code filter.

The two supplements now pass their bounded technical chain. Python-Edu contributes a cleaned 14.06
MB/1.29 MB train/evaluation partition after five Gitleaks-affected and 247 benchmark-critical files
are removed; its source metadata lacks file-level license identifiers, which the guarded-use record
accepts as a disclosed residual risk without corpus redistribution. The filtered public-domain PEP source contributes 9.96 MB/0.97 MB after nine
benchmark-critical documents are removed. A new five-source comparison covers 40,869 documents and
finds zero exact or verified ≥0.80 near-duplicate matches. The complete technical code slice is now
frozen at 55/20/10/10/5, with PEP→Python-Edu→Stack-Edu→Stack-v3→Common-Pile duplicate precedence.
See [finding 175](../../findings/175_code_tokenizer_supplements.md).

## 3. Web candidates

Primary E1W treatments remain well chosen but their versions change:

- **Ultra-FineWeb:** screen the newly released English L2/v1.4 selection and retain the integrated
  L1-HQ path as a reproducibility control. The project warns that upstream-source licenses still need
  inspection despite the project-level Apache-2.0 license.
- **FineWeb-Edu:** strong educational web candidate with both threshold and diversity trade-offs
  documented by its own ablations.
- **DCLM baseline:** broad English control with a different classifier/filtering philosophy.
- **Blend:** the three above, with FineWeb base providing a small diversity check rather than its own
  expensive arm.

Dolma v1.7 is valuable as a processing and mixture reference, but its 4.5 TB external-download layout
and older cutoff make it a poor primary acquisition target. Nemotron-CC-v2 is scientifically
interesting but held behind manual terms and possible upstream model-license conditions.

Provisional tokenizer web bytes: 35% Ultra-FineWeb v1.4, 30% FineWeb-Edu, 25% DCLM, and 10% FineWeb
base.

One revision-pinned shard per source now passes a bounded deterministic sample, stricter English
metadata plus independent-language checks, conservative PII/secret/adult-host/repetition filters,
Gitleaks exclusion, train/evaluation yield, and four-source exact/MinHash overlap analysis. The
31,092-document selection has zero exact or verified ≥0.80 near-duplicate cross-source matches and
retains every 35/30/25/10 quota. This does not establish upstream disjointness. See
[finding 176](../../findings/176_web_tokenizer_bounded_sources.md).

The pre-results flagship firewall freezes 20 immutable short-context, math, and code payloads with
63,652 unique tasks. Independent exact-field and task-unique 13-gram matching removes 92 records
(1.20 MB); a full successor rescan finds no remaining critical match. The 31,000 retained records
still exceed every source's train/evaluation byte quota, and strict subsetting preserves the prior
bounded zero-overlap result. Fifty-five 10-gram-only sensitivity records remain disclosed rather than
silently removed. Human guarded-use approval is complete; production global deduplication and
cleanup/resume still block use. See [finding 177](../../findings/177_web_evaluation_firewall.md).

## 4. Math candidates

The broad search found complementary, not interchangeable, sources:

- **FineMath-4+** is the high-quality incumbent; InfiWebMath-4+ adds a separately extracted web view.
- **MegaMath Web-Pro** is the main challenger. MegaMath also supplies math-related code and synthetic
  QA, which must remain separately labeled rather than being counted as organic web math.
- **OpenWebMath** is a stable, compact external baseline with strong LaTeX preservation.
- **Proof-Pile-2 Algebraic Stack** adds numerical, computer-algebra, and formal-math code, but its
  source-level license mapping needs additional work.
- **Nemotron-CC-Math 4+** reports strong upstream ablations, but is held because access and potential
  Phi-4-derived redistribution conditions are not yet accepted.

E1S compares FineMath-4+, MegaMath Web-Pro, and their blend. OpenWebMath, InfiWebMath, and Algebraic
Stack diversify the tokenizer sample and held-out audit; they do not each receive a GPU arm.

The bounded 30/15/25/15/10/5 tokenizer slice now passes immutable shard/schema, source-native
quality, English-prose, notation preservation, PII/security, six-source overlap, flagship benchmark
contamination, and partition-yield gates. One MegaMath Web-Pro secret-flagged record, three exact
cross-source duplicates, seven verified ≥0.80 near duplicates, and 980 benchmark-critical records are
removed. The 30,478-record successors rescan with zero critical matches and retain 113.32 MB train
plus 11.49 MB evaluation text. The guarded-use record accepts Common Crawl/page and code-license
metadata risk under attribution and no corpus redistribution; production global dedup and
cleanup/resume still block use. See [findings 179](../../findings/179_math_tokenizer_technical_qualification.md) and
[180](../../findings/180_math_rights_review_packet.md).

## 5. Synthetic and educational candidates

The stable synthetic screen remains Cosmopedia v2 versus Ultra-FineWeb-L3 Multi-Style versus a
blend. They have different failure modes: Cosmopedia is generated from web-seeded prompts with
Mixtral, while Ultra-FineWeb-L3 is derived through newer multi-style and QA transformations. The
Ultra-FineWeb-L3 QA and MegaMath QA subsets are stronger decay-phase candidates than stable-phase
defaults. Generator identity, seed-source rights, repetition, style entropy, answer leakage, and
model-phrase duplication must be audited separately from ordinary web filtering.

NVIDIA's large synthetic/DQA families remain research references until their gated agreement and
upstream model-license consequences are resolved.

The corrected bounded 35/35/15/15 slice now passes lineage-aware sampling, style/repetition/model-
phrase controls, PII/security, partition yield, four-source overlap, and the flagship firewall. Six
Gitleaks-affected and 353 benchmark-critical records are removed; zero exact or verified near
cross-source duplicate is found. The final 56,461 records retain 119.19 MB train and 11.91 MB
evaluation text and rescan with zero critical match. The first Cosmopedia output is forbidden because
its `seed_data` field was initially misread as seed text rather than a source label; the successor
correctly hashes the seed-bearing prompt. Missing generator/seed details are accepted as disclosed
limitations, and source/packed bytes will not be redistributed. See [findings 181](../../findings/181_synthetic_tokenizer_technical_qualification.md)
and [182](../../findings/182_synthetic_rights_review_packet.md).

## 6. Science, reference, books, and domain diversity

- **peS2o v3** replaces the older v2 integration target and remains the primary academic source.
- **FinePDFs-Edu English** is a promising complementary source for educational/legal/scientific PDF
  text, but OCR quality, boilerplate, source rights, and overlap with web/arXiv must pass before it
  enters training.
- **Common Pile filtered arXiv and PubMed** provide openly licensed diversity and independent source
  formatting. Proof-Pile arXiv is tokenizer-only because it overlaps these sources.
- **FineWiki English** is the primary reference candidate; the existing 2023 Wikimedia snapshot is
  the fallback. Attribution and share-alike handling remain explicit requirements.
- **Common Pile StackExchange, Gutenberg, LibreTexts, OER Commons, and Pressbooks** add technical
  discussion, books, and open educational material without pretending they are generic web.

Complete Gutenberg books, open textbooks, peS2o papers, and Stack v3 repositories are also candidates
for the separate long-document corpus. Base-mixture sampling and long-context document preservation
must use disjoint hashes.

The bounded 45/25/15/10/5 science slice now passes document identity, available license metadata,
English/science-content, OCR/boilerplate, PII/security, overlap, contamination, and partition-yield
gates. Gitleaks removes ten records. The first firewall correctly fails two quotas after contamination;
same-shard successors increase only the Common Pile arXiv and Proof-Pile margins and repeat every
downstream stage without weakening policy. The passing successor removes 137 benchmark-critical
records, rescans with zero critical match, and retains 4,262 documents with 117.94 MB train and 12.06
MB evaluation text. Guarded paper/PDF use is approved; production hardening remains blocked. See
[finding 184](../../findings/184_science_tokenizer_technical_qualification.md).

The bounded 35/20/15/15/5/10 reference slice now passes page/book/thread identity, English quality,
PII/security, rights-strength overlap precedence, benchmark contamination, and partition yield. Two
initial source-contract failures are preserved and corrected without weakening filters: Gutenberg's
language field mapping and OER Commons' inapplicable single-platform host cap. One secret-affected
and two verified near-duplicate records are removed, followed by 331 benchmark-critical records. The
24,275 final records rescan with zero critical match and retain 116.95 MB train plus 11.63 MB
evaluation text. The guarded-use record accepts attribution, share-alike, Gutenberg jurisdiction, and
item-license metadata risks without corpus redistribution. See [findings 185](../../findings/185_reference_tokenizer_technical_qualification.md)
and [186](../../findings/186_reference_rights_review_packet.md).

## 7. Qualification order before downloading at scale

1. Source cards and the guarded-use/attribution policy are frozen for all 30 selected sources.
2. **Bounded code pass:** all five code tokenizer inputs now pass identity, quality/language,
   security, benchmark-contamination, partition-yield, and bounded cross-source-overlap gates. Source
   use is approved; production operation remains separate.
3. **Bounded web pass:** four source samples pass identity, language/quality/local safety, security,
   yield, overlap, contamination, and guarded-use approval.
4. Run the real 20B operations rehearsal, materialize real firewall partitions, then materialize the
   660 MB tokenizer sample using the frozen quotas; failed
   sources are replaced within their category before any candidate tokenizer is trained.
5. Train and statically evaluate the three custom tokenizers plus Mistral 32K.
6. In parallel, build small source samples for E1 and estimate unique-token capacity, overlap,
   document lengths, and storage/download cost.
7. Only after the source screen shortlist is frozen, expand acquisition toward the 20B rehearsal.

This ordering lets us inspect dozens of datasets while paying storage and engineering cost for only
the sources that can plausibly enter the flagship.

Primary references: [The Stack v3](https://huggingface.co/datasets/HuggingFaceCode/stack-v3-train),
[Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu),
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
[Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb),
[FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath),
[MegaMath](https://huggingface.co/datasets/LLM360/MegaMath),
[Proof-Pile-2](https://huggingface.co/datasets/EleutherAI/proof-pile-2),
[peS2o](https://huggingface.co/datasets/allenai/peS2o),
[FinePDFs-Edu](https://huggingface.co/datasets/HuggingFaceFW/finepdfs-edu),
[FineWiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki), and
[the Common Pile](https://huggingface.co/common-pile).
