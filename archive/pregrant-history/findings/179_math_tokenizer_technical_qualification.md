# 179 — Six math sources pass bounded technical qualification

## Immutable source and content gates

The bounded math slice uses the lexicographically first declared shard for FineMath-4+,
InfiWebMath-4+, MegaMath Web-Pro, OpenWebMath, Proof-Pile-2 Algebraic Stack, and MegaMath Code. The
30/15/25/15/10/5 MB train allocation is sampled with 20% margins before destructive filters.
Accepted text is copied byte-for-byte: LaTeX and code notation are measured but never normalized or
rewritten. Declared English/quality metadata is enforced where present, independent py3langid checks
prose while exempting pure notation, and code comments/docstrings receive the code-language policy.

MegaMath Code's Parquet shard contains metadata rather than code contents. Its dedicated adapter
rejects no-license/non-permissive and unsupported-license rows, fetches deterministic candidates from
Software Heritage, verifies every blob against the recorded SHA-1, and preserves repository, path,
license, and blob lineage. It fetches 2,240 blobs to retain the initial 1,814-file bounded sample;
281 trustworthy blobs disagree with non-authoritative metadata length and are disclosed rather than
rejected. Algebraic Stack is likewise restricted to the conservative engineering allowlist.

## Security, overlap, and firewall result

Gitleaks v8.30.1 scans 31,469 initially sampled records with full redaction. Two findings map to one
MegaMath Web-Pro record, which is removed; the other five sources have no finding. Frozen precedence
then finds and removes three exact cross-source duplicates and seven verified 10-token-shingle near
duplicates at Jaccard at least 0.80. All ten come from OpenWebMath or InfiWebMath and every quota
survives.

The same pre-results flagship firewall covers 20 immutable payloads and 63,652 tasks. It removes 980
critical records (7,248,397 bytes): 57 have a complete eligible benchmark field, 930 meet the
task-unique 13-gram rule, and seven meet both. The removals link to 691 unique tasks and are reported
per source and benchmark. Another 703 sensitivity-only records linked to 415 unique tasks remain
retained and disclosed. A full rescan of all six successors finds zero critical record.

The final bounded slice contains 30,478 records, 113.32 MB of train-partition text and 11.49 MB of
evaluation-partition text. Every 30/15/25/15/10/5 source quota survives. Because contamination only
removes records, it preserves the post-overlap zero-match property.

## Boundary

This closes bounded identity, quality/language, notation preservation, PII/security, overlap,
contamination, and partition-yield gates. It does not approve Common Crawl/page rights, ODC-By
attribution, or original code licenses. Production global deduplication and cleanup/resume also
remain open. Do not authorize tokenizer sampling or training until those human and operational gates
pass.

Artifacts:

- [Checked math summary](../results/data/math-tokenizer-sources-20260907.json)
- [Frozen source registry](../research/flagship/source_registry.json)
- Runtime overlap result: `/mnt/speck-data/speck/source-qualification/math-cross-source-dedup-v1/report.json`
- Runtime contamination result: `/mnt/speck-data/speck/source-qualification/math-contamination-v1/report.json`
