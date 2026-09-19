# Pretraining corpus review

2026-09-18. The first local audit is complete; main-corpus quality selection remains open.
The frozen 105M engineering pilot is unchanged. No GPU rental, training, corpus replacement,
or bulk candidate acquisition is needed for this review.

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
| [Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb) | `02c85641e3d1` | Prioritize L1-derived English HQ for recoverable page/WARC origins; 12-shard stratified audit complete. Recover missing graphic/list content and page boundaries; qualify source families, extraction and eligible token supply |
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
2. The FineMath numeric-template census and first directory-exclusion candidate are complete
   (below). Audit code roles/versions in the full retained code acquisition next. Recount unique
   eligible supply after proposed changes. Preserve the frozen pilot.
3. Extend the bounded English refined-web inspection into source/answer consistency checks and
   inspect source-eligible code candidates. Establish domain coverage and overlap with current
   stocks before assigning main-training weights.
4. Freeze **one** two-arm data screening comparison: a common base checkpoint, fixed architecture,
   optimizer, global batch, LR schedule, token budget, and development evaluations. Change one
   source component at a time; preserve broad replay and keep final tests untouched. Count data
   preparation and evaluation costs as well as optimization. A single seed is a screening result,
   not a conclusive causal or paper-quality claim.

For budgeting only, **1B additional tokens per arm** at the measured baseline H100 rate of
13,614.8 tokens/s is **40.81 GPU-hours total optimization**, plus the shared base checkpoint,
validation, saves, preparation, and evaluations. Two full-cap development backend scenarios add
about 4.29 hours before grading. This is a provisional scale, not a frozen or launched experiment,
a GH200 forecast, or evidence that a 1B-token comparison will resolve the question. Decide the
affordable endpoint and primary metric before viewing the comparison results.

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

The [coding plan](../../docs/coding.md) prioritizes checked exercises for our first substantive
data comparison. [code-preview.json](code-preview.json) binds a separate UltraData-Code inspection:
16 Python rows per tier, two schema probes, and 405,214 response bytes at a checked revision.
Raw records remain outside Git. Static tokenization and syntax checks identify preparation needs;
no sampled code was executed, no new source was admitted, and no GPU experiment was launched.
The plan separates publisher results, local observations, and proposed evaluation work.

The subsequent [isolated execution receipt](code-execution.json) tests those 16 L3 Python solutions
against their supplied tests: seven pass and nine fail. All seven passing originals reject a
grossly broken return-value mutation. Sandbox controls pass. This measures sample self-consistency,
not independent correctness, overall source quality, or training eligibility; all records remain
outside training. The [coding notes](../../docs/coding.md#isolated-execution-follow-up) retain the
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
test gates are explicit in the [coding plan](../../docs/coding.md#source-lineage-follow-up--2026-09-19).
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

The [coding plan](../../docs/coding.md#natural-code-cohort-qualification--2026-09-19) records the
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
