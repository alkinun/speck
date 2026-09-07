# Flagship source exploration and shortlist

Status: candidate registry v1, 2026-09-07. This document records the broad source search before
tokenizer and corpus freeze. [`source_registry.json`](source_registry.json) pins the exact repository
revisions inspected today and proposes tokenizer-sample byte quotas. It is not training authority:
every selected source still needs a source card, local file hashes, rights disposition, yield audit,
deduplication behavior, and contamination results.

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

The tokenizer's provisional 100 MB code allocation is 55 MB Stack v3.1, 15 MB Stack-Edu, 15 MB
Common Pile Stack v2 educational code, 10 MB Python-Edu, and 5 MB Python language-design prose. This
is intentionally broader than the eventual stable mixture so the vocabulary is not overfit to one
code filter.

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

## 5. Synthetic and educational candidates

The stable synthetic screen remains Cosmopedia v2 versus Ultra-FineWeb-L3 Multi-Style versus a
blend. They have different failure modes: Cosmopedia is generated from web-seeded prompts with
Mixtral, while Ultra-FineWeb-L3 is derived through newer multi-style and QA transformations. The
Ultra-FineWeb-L3 QA and MegaMath QA subsets are stronger decay-phase candidates than stable-phase
defaults. Generator identity, seed-source rights, repetition, style entropy, answer leakage, and
model-phrase duplication must be audited separately from ordinary web filtering.

NVIDIA's large synthetic/DQA families remain research references until their gated agreement and
upstream model-license consequences are resolved.

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

## 7. Qualification order before downloading at scale

1. Freeze source cards and acceptance/attribution policy for Stack v3, Stack-Edu, FineWiki, and every
   source with inherited terms.
2. Implement a repository-aware Stack v3 adapter and download only several pinned shards for schema,
   language, license, PII/secret, quality, and yield measurement.
3. Materialize the 660 MB tokenizer sample using the provisional quotas in the registry; failed
   sources are replaced within their category before any candidate tokenizer is trained.
4. Train and statically evaluate the three custom tokenizers plus Mistral 32K.
5. In parallel, build small source samples for E1 and estimate unique-token capacity, overlap,
   document lengths, and storage/download cost.
6. Only after the source screen shortlist is frozen, expand acquisition toward the 20B rehearsal.

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
