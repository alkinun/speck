# 184 — Five science sources pass bounded technical qualification

## Source and content qualification

The bounded science slice uses peS2o v3 full text, FinePDFs-Edu English, Common Pile arXiv and
PubMed, and Proof-Pile-2 arXiv at 45/25/15/10/5 MB. Accepted text is copied byte-for-byte and retains
paper/document identity, URL, date, source partition, and available license metadata. All sources
receive independent English, size, alphabetic, repeated-line, spaced-OCR, replacement-character,
boilerplate, PII, and secret checks.

FinePDF receives additional controls because its nominal English shard contains substantial language
switching and general educational PDFs rather than only science. Both full-document and page-average
language metadata must report English at confidence at least 0.8, py3langid must independently agree,
the document must be non-truncated and extracted with Docling, upstream duplicate counts are bounded,
and at least three frozen science vocabulary terms must occur. Of 7,862 candidates inspected, 1,202
fill the initial 30/3 MB margin; the largest rejection groups are non-science content, truncated PDFs,
RoLM OCR, and weak language confidence.

The peS2o sample comes from a v3 S2ORC full-text shard rather than the lexicographically first v3
shard, which contains only S2AG abstracts. Its engineering license filter accepts only the frozen
CC0/CCBY/CCBYSA/public-domain labels. Common Pile arXiv and PubMed likewise require their exact
per-record open-license strings. Proof-Pile exposes no per-paper license, so its technical sample is
not rights-qualified.

## Security, overlap, and fail-closed firewall

Gitleaks v8.30.1 scans 4,409 sampled documents. Twenty-five fully redacted findings map to ten
records—one peS2o and nine FinePDF records—which are removed. Frozen precedence favors Common Pile
PubMed, Common Pile arXiv, peS2o, FinePDF, then Proof-Pile based on available license/provenance
strength. It finds zero exact and zero verified ≥0.80 near cross-source duplicates among 4,399
security-clean documents.

The first frozen firewall fails closed after contamination removal: Common Pile arXiv retains
14,850,378 train bytes against 15 MB, while Proof-Pile retains 4,933,248/425,666 train/evaluation bytes
against 5/0.5 MB. The failure and its report remain preserved. No matching, quality, or quota rule is
changed. Same-shard successors increase only those two pre-firewall sample margins to 22/2.2 MB and
8/0.8 MB, then repeat security and overlap stages before a newly frozen firewall successor.

The successor firewall removes 137 records and 8,453,541 bytes against the same 20 payloads and
63,652 tasks. Eighteen removed records contain a complete benchmark field and 119 meet the primary
13-gram rule. They link to 115 unique tasks. Eighty-five sensitivity-only records linked to 64 tasks
remain disclosed. A complete rescan of every successor finds zero critical match.

The final bounded slice contains 4,262 documents, 117.94 MB train text, and 12.06 MB evaluation text;
all five source quotas pass. Strict record removal preserves the post-overlap zero-match property.

## Boundary

Human rights review remains mandatory: peS2o's pinned card does not document v3 license-selection
semantics, FinePDF exposes no per-document content license, Common Pile warns of inaccurate or
laundered metadata, and Proof-Pile lacks per-paper licenses. Production global deduplication and
cleanup/resume also remain open. No tokenizer sampling or training is authorized.

Artifacts:

- [Checked science summary](../results/data/science-tokenizer-sources-20260907.json)
- [Science rights packet](../results/data/science-rights-review-20260907.json)
- Failed runtime firewall: `/mnt/speck-data/speck/source-qualification/science-contamination-v1.building/report.json`
- Passing runtime firewall: `/mnt/speck-data/speck/source-qualification/science-contamination-v2/report.json`
