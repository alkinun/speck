# Corpus and downstream data review

2026-09-18. The first local audit is complete; main-corpus quality selection remains open.
The frozen 105M engineering pilot is unchanged. No GPU rental, training, corpus replacement,
or bulk candidate acquisition is needed for this review.

## Current data closeout

[data-readiness.json](data-readiness.json) consolidates the later retained-bank token/overlap work,
full assistant format census, sampled complete context lengths and bounded RL prompt/reference
inventory. It preserves the earlier receipts below. The current
[qualification packet](../main-data/QUALIFICATION.md#retained-inventory-closeout--2026-09-19) records
supply bounds and remaining decisions; [assistant data](../../docs/assistant.md) owns downstream
format/length details. No source is admitted by these CPU measurements, and no GPU work ran.

The [cohort similarity receipt](cohort-similarity.json) records an exhaustive full-text lexical
near-copy comparison across the fixed 218-record code review cohort: 23,653 pairs, zero matches
under the review thresholds, and no partition changes. This is additional review evidence only;
it does not close provenance, licensing, semantic deduplication or eligible-supply gates.

## Evidence and scope

[result.json](result.json) binds the full local report and review artifacts under
`/mnt/speck-data/speck/corpus-quality-20260918`. Raw corpus text stays outside Git.

The audit scans six complete retained document indices: the five indexed pilot sources plus
the separate natural UltraData-Math candidate. These contain **5,370,978 documents and
6,707,115,348 tokens**, including BOS/EOS. Text/index hashes are verified; this does not repeat
the token-shard verification or establish a jointly eligible training union.

Code scope is explicitly smaller: **49,686 benchmark-screened pilot candidate files**, across
11 languages, before final joint pilot exclusion. It is not a census of the 476.8M-token retained
code acquisition. Code token counts are not estimated from character lengths.

Sampling takes the lowest SHA-256 ranks of `seed:source:ordinal`, with seed 20260918, up to
16 documents per UTF-8 byte-length band (<=2,048; <=8,192; <=32,768; larger). Code additionally
stratifies by language. The resulting **924 documents** deliberately overrepresent rare lengths
and languages. Unweighted sample fractions are not population defect rates.

An initial **34-document qualitative review** inspects excerpts from one sampled document per
non-code length stratum and one per code language. Selection uses the lowest SHA-256 of
`review:sample_id`; excerpts and observations are retained. This is an assistant review of
extraction and apparent usefulness, not independent human annotation, complete-document
fact checking, or an estimate of corpus quality. Four additional FineWeb documents were
exploratory spot checks and are outside the 34-document review.

Recreate a packet with the retained plan (choose a new output directory):

```bash
uv run --no-sync python -m scripts.corpus_audit \
  /mnt/speck-data/speck/corpus-quality-20260918/plan.json \
  /mnt/speck-data/speck/corpus-quality-20260918/packet-repeat
```

## Findings

The sources contain useful material, with concrete opportunities to improve selection. Automated
boilerplate, repeated-line, HTML, and replacement-character flags are review hints only. In
particular, HTML can be intentional in code, and repeated mathematical structure can be useful.

| Source | Observed examples | Next check |
| --- | --- | --- |
| FineWeb-Edu | Coherent general/technical prose, plus newsletter calls to action and references to missing media | Check boilerplate removal while preserving broad coverage |
| FineMath 4+ | Worked explanations alongside repeated numeric conversions and a topic-directory page | Measure template families and missing solution/code content before changing selection |
| Cosmopedia v2 | Fluent instructional prose; a short chapter ends before covering its advertised topic | Check completeness and factual/solution correctness, not just fluency |
| peS2o | Specialist papers with figure/table dependencies and flattened mathematical notation | Inspect extraction and context boundaries; preserve useful scientific coverage |
| FineWiki | Broad reference coverage, including biographies, entertainment, and large tables | Measure table/reference overhead; do not equate nontechnical topics with low quality |
| Natural UltraData-Math | Worked math, forum duplication, missing graph references, and patent prose | Apply the same checks as FineMath; the newer source is not automatically better |
| Stack-Edu pilot candidates | Algorithms, tutorials, project modules, build scripts, and SQL dumps | Audit code roles and versions separately; single files may legitimately need repository context |

An exact additional FineMath host census covers all 753,071 documents / 1,124,167,472 tokens
and 55,122 hostnames. `nrich.maths.org` contributes 28,941,557 tokens (2.57%); the reviewed
sample from it is a topic directory, but that does **not** classify the whole host. The existence
of templated content justifies a family-level audit, not an arbitrary host blacklist.

A parse-only screen of all **58 sampled Python files** passes 54 under local Python 3.14.6.
All four reported failures involve Python 2-style `print` statements. This neither establishes
four broken programs nor validates the behavior of the 54 others. No corpus code was executed.

The complete indices also measure context exposure:

| Source | Share of tokens in documents longer than 4,096 tokens |
| --- | ---: |
| FineWeb-Edu | 23.00% |
| FineMath 4+ | 27.31% |
| Cosmopedia v2 | 0.00% |
| peS2o | 91.67% |
| FineWiki | 45.75% |
| Natural UltraData-Math candidate | 53.17% |

These are document-length statistics, not truncation or loss rates. Packing retains tokens,
but a training window cannot attend to an entire longer document. Do not drop all long documents
or increase model context based on this statistic alone.

## Candidate decisions

The subsequent [variant inspection](web-variants.json) pins the distinct default English,
English v1.4 and L1-derived English HQ directories at the same UltraFineWeb release. The
[qualification packet](../main-data/QUALIFICATION.md) keeps their source populations and schemas
separate for the next comparison; no corpus payload or quality result follows from that metadata.
The [subsequent natural-web inspection](NATURAL_WEB.md) adds six hash-verified HQ shards and
bounded default/control samples. HQ is now the priority for further qualification because its
records preserve page/WARC origins. High-score extraction failures keep quality gates open;
no source is admitted and no quality ranking is established.
The [inventory/DCLM follow-up](WEB_INVENTORY_DCLM.md) completes the HQ listing (6,000 shards,
477.97 GB compressed) and now records the twelve-shard acquisition/census, 192 stratified samples
and 24 reviewed texts/excerpts in [web-hq-stratified.json](web-hq-stratified.json). Extraction and
source-family work remains; no stricter cutoff or source allocation is selected. DCLM previews
use partial viewer indexes; integer and continuous educational cutoffs differ.
The [extraction follow-up](web-filter-validation.json) recovers all three matching archived
captures and compares frozen review flags on 16 fresh HQ and 16 fresh FineWeb-Edu documents.
Missing lists and mixed page boundaries are confirmed; false alarms and missed defects keep
the flags review-only. Web eligibility and usable-token counts remain open. See the
[current findings](WEB_INVENTORY_DCLM.md).

The [full retained-code census](code-supply.json) subsequently reopens all 1,999 v1/v2 archives:
714,369 files / 476,774,847 tokens, with exact aggregate replay. Role hints and repository co-presence
do not resolve missing commit fields; existing benchmark/content-family evidence holds 123 files.
The [bundle follow-up](code-bundles.json) recovers 19 linked files at four pinned revisions;
four content flags hold 15 files by family, and static review exposes test-oracle weaknesses.
The [expansion inventory/probe](code-expansion.json) subsequently verifies one new Python metadata
shard and retrieves sixteen blobs: thirteen length-matched files / 9,541 tokens, with four content
flags and no admission. The [origin/test review](code-application-origins.json) resolves all four
application revisions/notices, but the single direct test link is stale. The
[stratified preflight](code-yield-result.json) now recovers/screens 138 files across 11 languages
and 72 strata: 30 content flags, 31 sample family holds, no admission. Complete source-use/quality
gates on this frozen sample before estimating yield. The
[coding plan](../../docs/coding.md#retained-supply-census--2026-09-19) contains the language/role tables
and the gap to proposed exposure; no main-code supply is admitted.

The 105M-token engineering pilot intentionally uses the six retained sources listed in
[its frozen recipe](../pilot/README.md#recipe). This is not the final flagship mixture.
Natural Ultra-FineWeb, DCLM and DCLM-Edu were present in archived experiment configurations;
their absence from the current pilot does not record a quality rejection. Their qualification
for the current main-data plan remains open, and they should be explicit candidates here.

[web-candidate-versions.json](web-candidate-versions.json) records a September 19 check against
the Hugging Face dataset API: all four historical pins below still match their repository heads.
That verifies repository revision freshness, not the age of the underlying pages or superiority
over another source. The DCLM Parquet release has its own revision, distinct from the original
`dclm-baseline-1.0` repository; do not compare hashes across repositories.

| Source | Matching revision prefix | Main-data role and next check |
| --- | --- | --- |
| [Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb) | `02c85641e3d1` | HQ audit and three archived-capture joins complete; new flags stay review-only. Qualify source-aware repairs, source families and eligible token supply |
| [DCLM baseline Parquet](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0-parquet) | `817d6752765f` | Independent selection approach; URL/ID fields verified in partial-viewer preview. Source-use review and comparable source-file sample remain |
| [DCLM-Edu](https://huggingface.co/datasets/HuggingFaceTB/dclm-edu) | `dbad8ad71224` | Related filtered candidate; explicitly distinguish `edu_int_score >= 3` from continuous-score cutoff. Do not count parent/filtered overlap as extra supply |
| [Ultra-FineWeb-L3](https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L3) | `bc3b1ba986fc` | Synthetic candidate for the Cosmopedia comparison; continue the source/answer checks below |

The [September 19 paper review](../../docs/research.md#openbmb-web-data-review--2026-09-19)
prioritizes natural Ultra-FineWeb on matched published evidence: nine of nine English benchmarks
improve over FineWeb-Edu. The first bounded inspection above does not reproduce that training
comparison or establish the newer HQ route's quality. Continue against retained FineWeb-Edu,
retaining DCLM/DCLM-Edu as independent candidates, with declared language, source paths,
score filters and length/domain coverage. The archived Ultra-FineWeb cutoff of 0.8 differs
from the paper's 0.5; do not assume stricter filtering preserves its result or coverage.
Keep the synthetic L3-versus-Cosmopedia question separate. Published results justify this priority;
eligibility, supply and main weights remain open. No bulk acquisition or additional GPU
comparison is authorized by this shortlist; the already planned coding comparison stays separate.

Pinned metadata and dataset cards are retained locally; full corpora were not downloaded.
The follow-up below adds a small, revision-checked English L3 content inspection.
Published token totals use upstream tokenizers and do not count as our eligible supply.

| Candidate | Pinned revision | Decision |
| --- | --- | --- |
| [Ultra-FineWeb-L3](https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L3/tree/bc3b1ba986fcaef6871b9790a413b16267c2de0f) | `bc3b1ba986fcaef6871b9790a413b16267c2de0f` | First new-content candidate: English Q&A and multi-style subsets separately; review factual drift, completeness, and overlap before admission |
| [UltraData-Code](https://huggingface.co/datasets/openbmb/UltraData-Code/tree/85182d829f2ce7ea07cca72ebfc509deea1d9f5f) | `85182d829f2ce7ea07cca72ebfc509deea1d9f5f` | Investigate L2/L3 as a code upgrade; resolve source-repository license provenance first; generated test candidates are not execution verification |
| [UltraX-Preview](https://huggingface.co/datasets/openbmb/UltraX-Preview/tree/a88527587389fd4ab352e9ad1273f4c0a234d8df) | `a88527587389fd4ab352e9ad1273f4c0a234d8df` | Secondary candidate for editing/cleaning existing web; keep source-family identity and check overlap |

The code card requires complying with individual repository licenses; its Apache header alone
is insufficient provenance. The web cards also describe source material and redistribution
conditions. Candidate inspection is not training admission. Use the existing source-use review,
joint deduplication, benchmark exclusion, and tokenizer pipeline rather than a parallel importer.

## Next experiment, after data preparation

1. Expand the qualitative review into a labeled, stratified audit. Distinguish extraction,
   completeness, usefulness, correctness (verified/unverified), and template repetition. Calibrate
   any model-based scorer against reviewed examples; report source and length denominators.
2. The FineMath numeric-template census, retained-code census and initial expansion probe are
   complete, along with the four-module origin/test review. The stratified sample is now frozen
   and screened; complete separate natural-code and checked-exercise gates before estimating yield.
   Preserve the frozen pilot.
3. Extend the bounded English refined-web inspection into source/answer consistency checks and
   inspect source-eligible code candidates. Establish domain coverage and overlap with current
   stocks before assigning main-training weights.
4. Freeze a bounded baseline/candidate pretraining comparison from paired fresh initializations,
   with fixed architecture, schedule, total token exposure and development evaluations. Run it
   before choosing the main starting mixture. Predeclare the contrast; a combined recipe comparison
   cannot attribute improvement to an individual changed source. Keep final tests untouched and
   count all training, preparation and evaluation costs within the study reservation.

The [current study design](../../docs/coding.md#first-comparison-to-prepare) and
[numeric plan](../main-data/plan.json) own the 600-hour pretraining study cap within 900 data-research hours and the H100 cost scenarios.
They do not freeze a run length or guarantee that short-run learning will resolve the question.
Record the result, uncertainty and recipe decision before main pretraining; later continuation
comparisons answer a separate question.

Main training still requires sufficient eligible supply, justified repetition and weights,
quality evidence, and measured all-in runtime. The present audit completes an initial diagnostic;
it does not close those requirements.

## Follow-up decisions — 2026-09-18

[followup.json](followup.json) binds the new census, candidate view, inspection captures, and
validation. These decisions change preparation priorities; they do not change the frozen pilot.

**Keep FineMath; do not introduce blanket numeric deduplication.** A complete scan of its
753,071 documents / 1,124,167,472 tokens took 122.03 seconds locally. Grouping full text within
each hostname after numeric normalization found only **two repeated families: four documents,
1,967 tokens**. This strict test misses fuzzy templates and number words, and normalization can
merge distinct useful math problems. It does not establish a material benefit from a numeric
deduplication rule. Conversion-related text matching the diagnostic markers totals 4,085 documents
/ 5,193,431 tokens; that is a role hint, not a low-quality classification.

**Prepare a narrow directory exclusion for the next corpus candidate.** The exact host/path/text
predicate in `scripts.corpus_templates.role_hint` matches 8,563 topic-directory pages containing
18,775,504 tokens: **1.6702% of this FineMath stock**, or roughly 0.25% of total exposure if the
current 15% math share were retained. Twelve matched excerpts and six same-host unmatched excerpts
were inspected. All twelve matched excerpts are activity directories; five of the six unmatched
excerpts show additional directory formats, so this conservative v1 knowingly has incomplete recall.
These are qualitative checks, not formal precision/recall or downstream-quality estimates.

The materialized `finemath-templates/directory-candidate-v1/view.json` and hashed exclusion list
retain **744,508 documents / 1,105,391,968 tokens** by complementing excluded ordinals within the
exact original manifest. Original text and pilot artifacts remain intact. This is a reversible
candidate selection view, not a training manifest or a human-reviewed removal/rights ledger.
Joint exclusions, holdouts, and a future explicit packing plan remain necessary. Preserve useful
NRICH lesson pages; do not blacklist the host. Do not spend a separate two-arm GPU experiment
solely on this small cleanup; review its coverage and apply the same selected policy to both arms
of a larger comparison if it is adopted.

**Inspect English L3 Q&A first; no wholesale synthetic-data replacement.** We captured 96 rows
from the pinned Ultra-FineWeb-L3 revision, using four seeded 12-row windows per English format.
Every response supplied the expected `x-revision`; responses were complete and untruncated.
The ten retained response payloads total 334,432 bytes, excluding the initial four-row schema probe
and one repeated fetch caused by a helper-name collision; hashes remained identical after the fix.
Captures, request identities, and scripts stay outside Git. This clustered sample is not a
population quality estimate.

| English subset | Inspected rows | Mistral tokens, including BOS/EOS | Median tokens | Maximum tokens |
| --- | ---: | ---: | ---: | ---: |
| Q&A | 48 | 46,476 | 961 | 1,362 |
| Multi-style | 48 | 22,953 | 462 | 1,106 |

All 96 sampled records fit 4K; this does not prove corpus-wide fit. Eight head/tail excerpt
reviews include useful technical material and concrete defects. One generated Q&A equates
3.5 kilograms with approximately one tonne, changing the source's ambiguous quantity into an
inconsistent answer. One multi-style record contains a refusal/editorial assessment of the input
instead of a finished rewrite. Other examples retain missing-figure references or promotional
prose. These observations do not estimate relative quality versus Cosmopedia.

Prioritize the Q&A subset for further checks because the source passage is visible in the
inspected serialization, permitting source-to-answer consistency review. This is an inspectability
decision, not evidence that Q&A trains a better model. Released structured fields in the inspected
schema are `uid`, `content`, and `style`; there is no separate original URL/source-document ID.
Establish lineage/overlap as far as possible, reject editing artifacts and inconsistent answers,
and measure surviving tokens before proposing a substitution for any of the 10% synthetic share.
Main mixture weights and new-source admission remain undecided.

## Coding priority and bounded preview — 2026-09-18

The [coding plan](../../docs/coding.md) keeps checked exercises as a candidate for the first
pretraining data comparison. [code-preview.json](code-preview.json) binds a separate UltraData-Code inspection:
16 Python rows per tier, two schema probes, and 405,214 response bytes at a checked revision.
Raw records remain outside Git. Static tokenization and syntax checks identify preparation needs;
no sampled code was executed, no new source was admitted, and no GPU experiment was launched.
The plan separates publisher results, local observations, and proposed evaluation work.

The subsequent [isolated execution receipt](code-execution.json) tests those 16 L3 Python solutions
against their supplied tests: seven pass and nine fail. All seven passing originals reject a
grossly broken return-value mutation. Sandbox controls pass. This measures sample self-consistency,
not independent correctness, overall source quality, or training eligibility; all records remain
outside training. The [execution receipt](code-execution.json) retains the
limitations and two concrete failure reviews.

## Code lineage decision — 2026-09-19

[code-provenance.json](code-provenance.json) records a bounded offline audit of the same preview:
all 16 L3 records remain outside training because source origin, revision and license evidence
are unresolved. A cross-tier UUID join is not established by the published schema/card or these
samples. No-overlap in 17 retained rows per tier is not evidence against a global join. Two L3
raw fields are byte-identical despite different UUIDs. The original dataset revision is preserved;
no candidate code was executed and the frozen pilot/evaluation were not changed.

The next route was a 16-file practical Python cohort qualified from retained natural-code stock.
One concrete Stack-Edu candidate now has an exact byte match to an immutable upstream commit
and a retained license notice; its remaining eligibility, deduplication, exclusion and independent
test gates are explicit in the [provenance receipt](code-provenance.json).
The inspected 354-row natural-code unit itself has no populated commits; do not treat it as
already qualified. This follow-up does not admit sources or authorize a GPU comparison.

Reproduce with the retained external artifacts (no network or dataset imports needed):

```sh
python experiments/corpus-audit/audit_code_provenance.py \
  experiments/corpus-audit/code-provenance-inputs.json > /tmp/code-provenance.json
cmp /tmp/code-provenance.json experiments/corpus-audit/code-provenance.json
```

The replay verifies acquisition/response hashes, revisions, schema consistency, primary-source
snapshots, archived natural-code evidence and the upstream byte match. Manual card/license
interpretations are explicitly retained in the input receipt; replay does not automate legal
approval or discover a global join. [code-provenance-validation.json](code-provenance-validation.json)
records replay and fault-injection checks. Raw data and source notices remain outside Git under
`/mnt/speck-data/speck/openbmb-code-provenance-20260919/` and the original acquisition roots.

## Natural-code cohort — 2026-09-19

[natural-code-cohort.json](natural-code-cohort.json) records the completed lineage/initial-quality
audit of 16 practical Python files from that stock. All match immutable upstream bytes; 13 have
license-file evidence at the matching revision, 14 parse under Python 3.10, and one triggers the
existing conservative benchmark-exclusion filter. Ten files, totaling 8,702 Mistral tokens including
BOS/EOS, clear those preliminary checks. None is admitted to the new training intervention.
This purposeful cohort is not a source-wide quality or supply estimate.

The [cohort receipt](natural-code-cohort.json) records the
three missing-notice cases, two syntax outcomes, observed fork/rename families and next independent
tests. [Static notes](natural-code-review.json) cover eight files; no corpus code was executed.
All source bytes, complete license/NOTICE files, lookup responses and the original interrupted
tree-download receipt remain outside Git in `/mnt/speck-data/speck/natural-code-cohort-20260919/`.

Replay using retained evidence and the repository environment, without network access:

```sh
PYTHONPATH=. .venv/bin/python experiments/corpus-audit/audit_natural_code.py \
  experiments/corpus-audit/natural-code-inputs.json > /tmp/natural-code-cohort.json
cmp /tmp/natural-code-cohort.json experiments/corpus-audit/natural-code-cohort.json
.venv/bin/pytest -q tests/test_natural_code_audit.py
```

Replay verifies archived bytes, all captured responses, source/path/commit/tree/blob and notice
bindings, family identities, tokenizer and frozen benchmark inputs. It does not grant license
approval, prove correctness, freeze the expanded benchmark set, or execute dataset code. Syntax
runtime is recorded, so exact replay comparisons use the recorded Python environment.
[Validation](natural-code-validation.json) records an identical offline replay in 10.30 seconds
(12.08 child CPU seconds), seven focused tests, and rejection of altered payloads, revisions,
origins and tree bindings. Missing notices and truncated inventories remain held.

## Application origins and test linkage

The [four-module receipt](code-application-origins.json) closes the expansion probe's origin follow-up:
all four implementations match immutable revisions with complete trees and retained MIT notices.
One has a direct test link whose API is stale; three have no test-named paths. Static review and
29 verified response hashes establish provenance evidence, not independently checked examples.
No code ran and nothing is admitted. Raw files stay outside Git.

## Stratified code-yield preflight

[Protocol](code-yield-plan.json), [result](code-yield-result.json), and
[coding notes](../../docs/coding.md#stratified-retained-code-audit--2026-09-19) define the sample,
separate eligibility gates and next assessment. Replay into a fresh external directory:

```bash
PYTHONPATH=. uv run --no-sync python experiments/corpus-audit/audit_code_yield.py \
  experiments/corpus-audit/code-yield-plan.json /external/code-yield-replay
```

The command verifies input identities, original sampling denominators, selected source bytes and
tokens, then screens the sample. Output includes raw code and stays outside Git. It neither runs
corpus code nor admits training data. Keep incomplete gates unresolved; a clean screen is not yield.


The separate [Stack v3 preflight](stack-v3-broader.json) completes the
[frozen broader acquisition](stack-v3-sampling.json): sixteen groups, 29,347 repository rows and
a fixed 80-file review cohort. Four files trigger content flags and eight have known-family holds.
Its external manifest binds acquisition, selection and screen artifacts with exact offline replay.
The [common review](code-cohort-review.json) adds 24 full-text readings and 30 bounded origin
checks, with 28 Git-verified originals. Nine Stack v3 transformations are explained; one breaks
Python syntax and one original is unavailable. All original cohorts and weights remain intact.
The [Python follow-up](python-cohort-validation.json) adds twelve full readings, bringing common
coverage to 36 files and completing all 16 initially unheld Python records. It verifies eleven
additional requested originals (one was also checked previously), records two new GoLLIE family
holds and checks fourteen syntax controls on each of two Python versions. The diagnostic is not
a bulk eligibility filter.
The [family/provenance follow-up](code-family-provenance.json) replays known links with the existing
splitter and brought family-held records to 42, including a vendored Pylint file matching upstream
bytes. Arcade/stringutils installed identities are verified; the sampled stringutils file matches
its wheel but is absent from its same-version source archive. Full lineage and remaining notice/revision checks stay open.
The [Go/Rust review](go-rust-cohort-review.json) adds thirteen complete readings / 28,634 tokens,
covering all remaining unheld Go/Rust records. It brought common coverage to 49 files / 67,461 tokens;
at that stage, 129 unheld records remained unread and 42 family holds persisted. Static findings do not replace
provenance: no new origin checks or corpus execution occurred in this batch.
The [JS/TSX/Vue review](javascript-cohort-review.json) adds fourteen readings / 29,940 tokens and
one metadata-only LeetCode-family hold. It brought coverage to 63 files / 97,401 tokens, with 43 held
and 114 unheld records then unread. A Processing example and two dialect fixtures demonstrate
why observed language/context must remain separate from original labels and sampling weights.
The [TypeScript/C#/Kotlin review](typed-language-cohort-review.json) adds fourteen readings / 25,281
tokens, completing all unheld records with those labels. It brought coverage to 77 files / 122,682 tokens,
with 43 held and 100 unheld records then unread. Declarations, translated algorithms, tests and
teaching scaffolds require different context; this batch has no new or reused verified origins.
The [shell/build/config review](build-config-cohort-review.json) adds 24 readings / 36,832 tokens.
It brought coverage to 101 files / 159,514 tokens, with 43 held and 76 unheld records then unread.
Its twelve scripts, eight authored configs, two generated build files and two localization files show
why observed role must accompany language labels. It reuses one verified origin and one earlier
unavailable-original outcome, without new acquisition or corpus execution.
The [C/C++ review](c-cpp-cohort-review.json) adds 22 readings / 54,222 tokens and a metadata-only
hold for a record under `svcomp`, whose task text was not displayed or semantically reviewed.
It brought coverage to 123 files / 213,736 tokens, with 44 held and 53 unheld records then unread.
Test-harness limitations, template instantiation and SDK context remain distinct from validated
behavior or performance; this batch has no new or reused verified origins.
The [Java review](java-cohort-review.json) adds fifteen readings / 27,176 tokens, completing all
nineteen unheld Java records. It brought coverage to 138 files / 240,912 tokens, with 44 held and 38
unheld records then unread. Caller/test context and a restrictive notice needing applicability review
remain unresolved; one prior verified original is reused without a recovered notice or new lookup.
The [SQL-labelled review](sql-cohort-review.json) adds eleven readings / 37,430 tokens, completing
all unheld records with that label. It brought coverage to 149 files / 278,342 tokens, with 44 held and
27 unheld records then unread. Schemas, notebooks, dumps and test roles need distinct context; one
file contains CQL definitions. No new or reused verified origins, family holds or admissions result.
The [MATLAB/Objective-C/PHP review](application-cohort-review.json) adds six readings / 5,546 tokens,
completing all unheld records with those labels. It brought coverage to 155 files / 283,888 tokens, with
44 held and 21 unheld records then unread. Signal-validation assumptions, framework dependencies
and notice applicability remain unresolved; no new or reused verified origins or holds result.
The [Markdown review](markdown-cohort-review.json) adds eighteen readings / 58,611 tokens,
completing all 22 unheld Markdown records. It brought coverage to 173 files / 342,499 tokens, with
44 held and three unheld CSS/SCSS records then unread. Tutorials, reference/test excerpts and
personal/project prose retain complete document boundaries; no new origins, holds or admissions result.
The [coding guide](../../docs/coding.md#common-cohort-review) records findings and remaining gates.
No source ranking or eligible-yield estimate follows from these restricted batches.

The [stylesheet closeout](stylesheet-cohort-review.json) adds the final three readings / 70,475 tokens.
All 174 currently unheld records are now read. Cumulative coverage is 176 files / 412,974 tokens,
including two earlier reads now held; all 44 family holds remain intact. Page, template and component
roles, unresolved template attribution and a file notice referring to missing NOTICE context are
recorded. No browser, compiler, asset fetch or corpus code ran; no new or reused verified origins
are available for this batch. Disjoint per-cohort counts reconcile all 218 original records and
592,826 observed tokens. Reading completion does not establish source use, eligible supply or a
source ranking. The existing [qualification packet](../main-data/QUALIFICATION.md#next-bounded-data-packet)
owns the remaining provenance, family, intended-use and finite-inventory work before data-study selection.

The [notice follow-up](code-notice-provenance.json) checks all eight sampled files in four repositories
with recorded notice questions. Seven host origins are newly verified and one is reconfirmed; four
complete trees and four distinct ancestor notices are retained. Eclipse's NOTICE explains separate
code/non-code licenses, while Java restrictive wording and Ororus template attribution remain open.
It brought linked cohort-origin coverage to 18/104 unheld Stack-Edu and 26/70 unheld Stack v3 records. All reading
counts, holds and assessment fields remain unchanged. No source-use approval or admission follows.

The [pinned-origin follow-up](pinned-code-origins.json) checks all 44 remaining unheld Stack v3
records at pinned commits: 42 new verified origins and two source 404s. Four identities use smaller
path-specific metadata after recursive trees exceed the cap; ancestor-notice searches remain open
for those files. Fifteen distinct notices are recovered for 22 selected records without deciding
applicability. It brought linked origin coverage to 18/104 unheld Stack-Edu and 68/70 unheld Stack v3.
The derived metadata inventory separates held, verified, attempted-unresolved and unattempted records,
reconciling original counts/tokens. It adds no semantic readings, holds, eligible tokens or admissions.

The [retained-origin recovery](retained-code-origins.json) attempts all 85 previously unattempted
unheld Stack-Edu records through bounded path histories. Ten C++ origins and ten ancestor notices
are verified; 74 history requests and one tree check are blocked by GitHub quota exhaustion.
Current linked origin coverage is 28/104 unheld Stack-Edu and 68/70 unheld Stack v3. The successful
prefix is quota/order-dependent, not a source-quality estimate. All 218 assessments remain unchanged.
Resume blocked requests only after access changes; independent web/math qualification can proceed.
