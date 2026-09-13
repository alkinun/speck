# 173 — Bounded code sources pass overlap analysis; Stack-Edu wins precedence

## Source completion

Stack-Edu's two Gitleaks-affected records were excluded, leaving 9,570 files with 22.47 MB in the
tokenizer training partition and 2.85 MB in evaluation. A pinned Common Pile Stack v2 educational
TypeScript shard could not fill the initial 20 MB target: its complete score-4+, conservative-license,
non-vendor, non-generated, English-filtered tail contained 11.78 MB. The failure remains preserved.

The Common Pile successor changes no filter, uses 12 MB, and retains 4,966 files from 3,768
repositories. Upstream blob IDs disagree with plain released-text SHA-1 for 154 records, so they stay
as provenance while released-content SHA-256 drives deduplication. Gitleaks finds zero records. The
tokenizer allocation moves from 15M/1.5M to 10M/1M; Stack-Edu receives the recovered 5M/0.5M.

## Exact and fuzzy comparison

The frozen blend precedence is Stack-Edu, then Stack v3, then Common Pile. Across 31,583 retained
documents, 30,444 with at least 50 lexical tokens receive 128-permutation MinHash signatures over
10-token shingles. LSH candidates at 0.80 must also pass exact Jaccard of the represented shingles.

The bounded samples contain zero exact released-text SHA-256 matches and zero verified near-duplicate
matches. Every source retains its tokenizer train/evaluation quota. This does not establish that the
full sources are disjoint: an exploratory upstream-ID comparison found shared blobs outside the exact
bounded high-score samples, and full-corpus global deduplication remains mandatory.

## Decision

Freeze provisional tokenizer weights at 55% restricted Stack v3, 20% Stack-Edu, 10% Common Pile
Stack v2 educational code, 10% Python-Edu, and 5% Python enhancement/design prose. For any blend,
specialist Stack-Edu records win duplicate precedence, then Stack v3, then Common Pile. Stack-Edu and
Common Pile still need the pinned benchmark-contamination successor; all sources still need final
rights/attribution authority before training.

Artifacts:

- [Checked overlap summary](../results/data/code-source-overlap-20260907.json)
- [Frozen duplicate plan](../research/flagship/code_cross_source_duplicates_v1.json)
- Runtime result: `/mnt/speck-data/speck/source-qualification/code-cross-source-dedup-v1/report.json`
