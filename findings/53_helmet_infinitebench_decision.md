# 53 — HELMET InfiniteBench embedded-work and metric decision

## Question

Can HELMET's fifteen InfiniteBench cells be acquired and executed under the intended scope, and which
parts already have deterministic prompts and local scoring?

## Exact payload boundary

The pinned Hugging Face repository at `90f0394…` exposes a 12-split default dataset of approximately
2.50GB. HELMET needs only three objects:

| Task family | File | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| long-QA multiple choice | `longbook_choice_eng.jsonl` | 185,904,631 | `d2e64682…` |
| long-QA free response | `longbook_qa_eng.jsonl` | 298,297,185 | `7f53a3aa…` |
| summarization | `longbook_sum_eng.jsonl` | 78,601,868 | `4e4279cc…` |

The minimal future acquisition is therefore 562,803,684 bytes rather than the mutable default loader's
full repository. No payload file was downloaded.

## Provenance and rights

The Hugging Face card labels the dataset Apache-2.0, while the pinned
[upstream repository](https://github.com/OpenBMB/InfiniteBench) contains an MIT license for its
software and documentation. Neither statement, by itself, identifies rights for every embedded novel
or web-derived summary.

The [published paper](https://aclanthology.org/2024.acl-long.814/) says English novel tasks use novels
and gold summaries sourced from websites, including SparkNotes and CliffsNotes. Key-entity substitution
creates altered “fake novels,” but all three English tasks share those novels, and some brand-new or
little-known works are not entity-substituted. Transformation does not establish source-content
authority.

Current [SparkNotes terms](https://www.sparknotes.com/terms-of-use/) and
[CliffsNotes terms](https://www.cliffsnotes.com/terms-of-service) restrict their content to personal or
educational noncommercial use absent separate permission. Direct metadata capture from both sites
returned HTTP 403, so those indexed terms are supporting risk evidence rather than retained payload or
a substitute for work-level provenance.

## HELMET prompt and metric split

Unlike NarrativeQA and Multi-LexSum, InfiniteBench demonstration selection is consistently seeded.
The ten QA/multiple-choice cells use local `rougeL_f1` and exact-match metrics. This part of the prompt
and scorer design is technically preferable and is recorded as such.

All fifteen cells still use the gated Llama 2 tokenizer for length filtering and truncation, so exact
prompts do not yet exist. The five summarization cells introduce another change: upstream InfiniteBench
uses ROUGE-L-Sum, while HELMET replaces it with the unqualified proprietary `gpt-4-f1` judge.

## Decision

Metadata, minimal payload selection, seeded demonstration selection, and the local long-QA metrics
qualify. Embedded-work rights, intended commercial scope, payload acquisition, truncation identity,
case materialization, contamination, and summarization judging do not. No data or licensing contact
was attempted.

If rights or a pre-results replacement are resolved, the exact 562.8MB three-file subset—not the full
default repository—should be acquired. Long-QA QA/choice must retain local metrics; summarization needs
a separately justified judge decision.

This completes explicit dispositions for all seven archive-external HELMET source families: three have
qualified immutable paths, while TREC, Multi-LexSum, NarrativeQA, and InfiniteBench remain blocked.
This operational decision is not legal advice or a final ownership determination.

## Artifacts

- [Frozen decision protocol](../research/architecture-promotion-v1/helmet_infinitebench_decision_v1.json)
- [Blocked decision](../results/Speck-Architecture-Promotion-v1/helmet-infinitebench-decision.json)
- [Decision runner](../scripts/helmet_infinitebench_decision.py)
- [Complete runtime inventory](../results/Speck-Architecture-Promotion-v1/helmet-runtime-dependency-audit.json)
