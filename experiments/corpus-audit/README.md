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
