# 169 — Restricted Stack v3.1 passes bounded yield, not training authority

## Source and scope

The official `HuggingFaceCode/stack-v3-train` release changes the code-data plan. We pinned commit
`8f3f25d86e44fd691428131efd17af75d4716499`, the current v3.1 lineage after the dataset card's
reported exact-duplicate repair, and authorized only a bounded shard profile. No full-corpus download
or training use was authorized.

The first three-shard plan failed its declared language quotas after strict filtering and remains
preserved on the data volume. It also exposed an incorrect local assumption: upstream `content_id`
does not consistently equal a plain SHA-1 of the released UTF-8 text. The successor preserves the
upstream ID for provenance and computes an independent SHA-256 over released content for exact dedup.
It does not reinterpret the upstream identifier.

## Twelve-shard result

The successor pinned the twelve smallest current shards, 3,355,194,472 compressed bytes. It inspected
252,860 repository rows and 2,836,149 files. The restricted filter rejected 2,715,490 files by license
type, 38,358 vendored files, 181 fork repositories, four files containing ambiguous `LicenseRef-`
identifiers, and three files with high-confidence secret patterns. It accepted 48,777 files for the
declared languages.

The bounded artifact contains 70,227,292 content bytes from 16,972 files across 1,513 repositories
and eleven languages. Every language quota passed. File content, repository structure, and attribution
metadata are separate, hash-bound outputs. Upstream PII placeholders remain measurable; 351,798 files
across the full scanned set had an upstream content ID unequal to plain released-text SHA-1, confirming
that the two identities must not be conflated.

## Decision

Restricted Stack v3.1 remains the primary raw-code candidate and can supply its provisional tokenizer
quota. This is not training authority. Before E1S or final tokenizer sampling, freeze a manually
reviewed detected-license policy, analyze English prose in comments/docs/notebooks, qualify a dedicated
secret scanner, run cross-source near-duplicate and benchmark-contamination checks, and rehearse
acquisition cleanup/resume.

Artifacts:

- [Bounded qualification summary](../results/data/stack-v3.1-bounded-qualification-20260907.json)
- [Pinned plan](../research/flagship/stack_v3_qualification.json)
- Runtime result: `/mnt/speck-data/speck/source-qualification/stack-v3.1-v1/report.json`
