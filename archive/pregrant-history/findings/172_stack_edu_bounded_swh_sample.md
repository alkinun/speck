# 172 — Stack-Edu SWH sampling works, with two secret-flagged records pending removal

## Bounded source

The Stack-Edu candidate stores scored metadata and Software Heritage blob IDs rather than file text.
We pinned its current revision and the complete Rust, Go, and SQL metadata shards: 5,556,954 rows and
roughly 544 MB of Parquet. The sampler admits score-4+, permissive, conservative-license, UTF-8/ASCII
records, fetches blobs from the dataset card's public SWH S3 location with 32 bounded workers, and
requires raw content SHA-1 to equal `blob_id`.

## Identity correction

The first run rejected 759 fetched blobs when either SHA-1 or metadata length differed. Inspection
showed the sampled SHA-1 matched exactly while `length_bytes` could differ by a few bytes. That
predecessor remains preserved. The successor makes cryptographic blob identity authoritative, reports
634 length-only discrepancies, and has zero hash mismatches, missing blobs, or exhausted fetch retries.

## Result and boundary

The successor inspected 421,748 metadata rows and fetched 11,008 blobs. After score, license,
encoding, English-comment, and repository-cap filtering, it retained 9,572 files from 7,394
repositories: 8.22 MB Rust, 9.00 MB Go, and 8.11 MB SQL. Every bounded quota passed.

Official Gitleaks v8.30.1 reports three fully redacted generic-key findings across two records. Those
records are not yet removed from this artifact, so dedicated-secret, contamination, near-duplicate,
manual-rights, and training-authority gates remain blocked.

Artifacts:

- [Checked summary](../results/data/stack-edu-bounded-sample-20260907.json)
- [Pinned sample plan](../research/flagship/stack_edu_sample_v1.json)
- Runtime result: `/mnt/speck-data/speck/source-qualification/stack-edu-v2/report.json`
