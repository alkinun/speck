# 47 — HELMET archive-external runtime and scorer boundary

## Question

Does the pinned 11GB HELMET archive contain everything needed to materialize the 105 configured
evaluation entries, and does the already qualified native model adapter imply an executable suite?

## Static inventory

No. An exact audit of the 14 active configs partitions the 105 entries into 55 archive-local entries
and 50 runtime-loaded entries. Recall (20), RAG (20), reranking (5), and citation generation (10) use
archive data, with citation prompt files supplied by the pinned code checkout. ICL (25), long QA (15),
and summarization (10) call Hugging Face datasets at runtime.

The upstream `requirements.txt` has no versions, and every remote `load_dataset` call omits a revision.
The audit therefore records current source heads as observations, not as identities bound by HELMET:

| Source | Observed revision | Config entries | Metadata/rights state |
| --- | --- | ---: | --- |
| NarrativeQA | `2e643e7` | 5 | Apache-2.0 metadata; underlying stories/scripts unresolved |
| InfiniteBench | `90f0394` | 15 | Apache-2.0 metadata; underlying long-book texts unresolved |
| Multi-LexSum | `055f5fa` | 5 | ODC-By data; summaries/metadata are CC-BY-NC-4.0 |
| TREC | `eb1e45c` | 10 | license metadata is unknown |
| Banking77 | `90d4e2e` | 5 | CC-BY-4.0 metadata; payload identity pending |
| CLINC150 | `155b9c7` | 5 | CC-BY-3.0 metadata; payload identity pending |
| NLU Evaluation Data | `0edc700` | 5 | CC-BY-4.0 metadata; payload identity pending |

Only metadata trees and small source/card blobs were fetched for this audit; no listed dataset payload
was acquired.

## Runtime incompatibility

Speck's qualified HELMET environment pins `datasets==5.0.1`. Its exact dataset-module resolver source
is hashed in the protocol and rejects Hub repositories that still expose Python loading scripts.
Multi-LexSum, TREC, Banking77, and NLU Evaluation Data have that layout at the observed revisions.
Consequently, the current pinned runtime cannot execute four of the seven archive-external source
families even if their payloads were otherwise authorized.

An offline successor must either use an independently pinned legacy loader to materialize immutable
snapshots, or convert the exact source revisions to data-only snapshots and prove example, split,
label, seed, and prompt parity. Silently following updated Hub exports is not allowed.

## Gated tokenizer and model judges

HELMET uses `meta-llama/Llama-2-7b-hf` three times for document-length filtering and truncation in long
QA and summarization. The repository rejects anonymous identity access and requires Llama 2 license
acceptance. No stored credential was used to bypass that decision, so tokenizer revision and bytes
remain unqualified.

The official aggregate is also not fully local. NarrativeQA uses `gpt-4-score`, and the pinned script
instantiates `gpt-4o-2024-05-13` at temperature 0.1 without a seed. The summarization aggregate uses a
related GPT-4 judge. Dated model naming does not establish stable API behavior, repeatability, cost,
or acceptable transmission of prompts and model outputs. The earlier `pytrec_eval` qualification
covers reranking only.

## RULER v2 guardrail impact

The v2 source-document guardrail splits cleanly at this boundary: its 20 RAG entries are archive-local,
while all 15 long-QA entries need NarrativeQA or InfiniteBench plus the gated Llama 2 tokenizer; one
of the three long-QA metrics additionally needs the unqualified proprietary judge. The guardrail
therefore remains blocked even after the main archive finishes downloading.

## Decision

The dependency inventory qualifies, but HELMET execution does not. Required next work is: finish and
safely inspect the archive; resolve component and underlying-work rights; pin and hash every runtime
payload; choose and parity-test a compatible offline loader; obtain an authorized tokenizer decision
or validate a semantics-preserving replacement; qualify or replace model judges; then materialize and
contamination-scan exact cases. Adapter qualification is not evidence for any of those steps.

## Artifacts

- [Runtime dependency protocol](../research/architecture-promotion-v1/helmet_runtime_dependencies_v1.json)
- [Runtime dependency audit](../results/Speck-Architecture-Promotion-v1/helmet-runtime-dependency-audit.json)
- [HELMET suite contract](../research/architecture-promotion-v1/external/helmet.json)
- [Archive metadata audit](../results/Speck-Architecture-Promotion-v1/helmet-data-metadata-qualification.json)
