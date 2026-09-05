# 43 — HELMET data metadata and storage plan

## Question

Which pinned HELMET payload is actually required, what evaluation matrix consumes it, what storage
boundary is safe, and what can be qualified before administrator access to the separate disk?

## Pinned archive

Dataset commit `dddb209d` contains only a citation-only README and two Git-LFS pointers. The pinned
HELMET code's download script references `data.tar.gz`; none of its 14 active category configs refers
to the later `data_v2.tar.gz` archive.

| Archive | Payload bytes | SHA-256 | Decision |
| --- | ---: | --- | --- |
| `data.tar.gz` | 11,271,916,108 | `9d693981aa3c065b8b2ff82ddf946141cdc4ece4524f18bff6f3fbd2a86982d9` | required |
| `data_v2.tar.gz` | 8,858,479,649 | `89e6f3a197c6079d6b3b9092d9cf4503e4989bdc0bd659741bbb4e43bfa7600e` | excluded |

The upstream shell script downloads mutable `main`. The qualified replacement URL addresses
`data.tar.gz` through the exact dataset revision. The metadata audit fetches zero new Git objects and
does not materialize either LFS payload.

## Configuration matrix

All seven categories have a short config covering 8K, 16K, 32K, and 64K plus a 128K config. Together
the 14 files declare 105 evaluation entries across recall, RAG, reranking, ICL, long QA, summarization,
and citation generation. Their exact hashes and per-entry lengths, generation limits, sample counts,
chat-template settings, dataset identifiers, and paths are retained in the qualification artifact.

The configs depend on RULER/JSON-KV, KILT/NQ/TriviaQA/HotpotQA/PopQA, MS MARCO, TREC, Banking77,
CLINC150, NLU evaluation data, NarrativeQA, InfiniteBench, Multi-LexSum, and ALCE/ASQA/QAMPARI.

## License boundary

The dataset repository contains no license file or dataset-card license declaration. The HELMET code's
MIT license cannot be inferred to cover these component datasets, and a citation list is not a grant
of rights. Component licenses therefore remain unqualified until the pinned archive is available for
file-level mapping and each upstream term is verified. No benchmark execution is authorized yet.

## Storage boundary

The compressed archive is 10.50GiB and upstream advertises roughly 34GB extracted. The frozen minimum
is 64GiB free before download so the archive, extracted tree, hashes, results, and failure-recovery
headroom coexist. The root filesystem fails this gate at roughly 20GB free.

`/dev/sda2` is a 6,001,156,685,824-byte ext4 partition and passes the capacity check, but it is
unmounted. Read-only mount and block-level inspection both require administrator authentication, so no
contents were inspected and no writes were attempted. An administrator must mount the intended volume
and grant a dedicated user-owned Speck directory before the capacity gate can be rerun.

## Decision

Dataset metadata, archive selection, config identity, and the storage plan qualify. Download,
extraction, component-license qualification, dataset-bound loader checks, and execution remain blocked.
Finding 47 subsequently establishes that 50 configured entries also require archive-external runtime
datasets, a gated truncation tokenizer, or model-judge qualification.

## Artifacts

- [Metadata qualification](../results/Speck-Architecture-Promotion-v1/helmet-data-metadata-qualification.json)
- [Audit runner](../scripts/helmet_data_metadata_audit.py)
- [HELMET contract](../research/architecture-promotion-v1/external/helmet.json)
