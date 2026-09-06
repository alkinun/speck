# 124 — HELMET archive-local rights and provenance audit

## Exact path-family accounting

The twice-reproduced archive inventory is partitioned without reading or extracting payload content:
15 RULER NIAH files, 5 JSON-KV files, 24 KILT-derived RAG files, 6 MS MARCO reranking files, and
2 ALCE citation files. These five groups account for all 52 config-declared local paths. The pinned
HELMET tree contains loaders and config generation, but no pipeline that reproduces these files from
exact source revisions, retrieval corpora, parameters, and seeds. The archive itself contains no
license, notice, README, citation, or provenance file.

## Rights boundary

Repository code licenses are not extended to separately sourced data. The RULER code is Apache-2.0,
but the exact generated payload identity is absent. KILT names a 2019-08-01 Wikipedia knowledge source
and several independently governed QA datasets, while HELMET supplies no field-level derivation or
attribution chain. Microsoft limits MS MARCO to noncommercial research, extends no IP license, warns
that it may not own the underlying documents, and makes use an acceptance event. ALCE combines ASQA,
QAMPARI, and retrieved passages; its MIT software license and QAMPARI's CC0 release do not establish
rights in every embedded annotation and passage.

Consequently, zero of five families is extraction-qualified. The archive remains unextracted and no
HELMET execution, evaluation-use, commercial-use, or redistribution authority follows. This is an
operational research gate, not legal advice or a claim that a particular use would be unlawful.

## Experimental immutability

This is append-only evidence. It does not alter the frozen v2 evaluation manifest, finalist program,
model configs, data windows, analysis, thresholds, or automation. A validator test briefly exposed
that mutating the active evaluation manifest would invalidate its preregistered launch pin; that
uncommitted mutation was fully removed, and the original manifest hash remains
`36e5fccdfd6d7b7df7f667b6cb6b0fca4657248a8929ac4a0452b3588207d8d6`.

## Required next gate

Obtain an exact author/rights-holder bill of materials and intended-use authority, or freeze a
pre-results successor that clean-room regenerates selected families from separately qualified inputs.
MS MARCO additionally needs recorded organizational scope acceptance; Wikipedia-derived fields need
complete attribution/share-alike accounting. Only then may a narrow extraction be authorized and each
file hashed before prompt and contamination qualification.

## Artifact

- [HELMET archive-local rights audit](../results/Speck-Architecture-Promotion-v1/helmet-archive-local-rights-audit.json)
