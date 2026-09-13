# 176 — Four web sources pass bounded sampling, security, and overlap gates

## Bounded source pass

One immutable shard each from Ultra-FineWeb English v1.4, FineWeb-Edu, DCLM baseline, and FineWeb
base was sampled in deterministic seeded row-group/content-hash order. All sources require declared
English with metadata confidence at least 0.8 plus independent py3langid confidence at least 0.8.
The local policy also rejects raw email/IP values outside declared upstream placeholders,
high-confidence secrets, explicit-adult host terms, excessive repeated lines, and host concentration.

The security-filtered partitions retain 42.03M/4.21M Ultra-FineWeb, 36.00M/3.60M FineWeb-Edu,
30.01M/3.07M DCLM, and 12.00M/1.21M FineWeb-base train/evaluation bytes. Gitleaks finds one DCLM
record and zero in the other sources; the affected DCLM record is removed. The sample stage also
rejects 1,531 low-confidence language-metadata records, 21 independently detected non-English
records, and 153 raw email/IP records across the four sources.

## Bounded overlap

Precedence is FineWeb-Edu, Ultra-FineWeb, DCLM, then FineWeb base. Across 31,092 retained documents,
31,089 receive 128-permutation MinHash signatures over 10-token shingles. The bounded selections
contain zero exact released-text matches and zero verified near duplicates at Jaccard 0.80.

This is not evidence that the upstream corpora are disjoint: all derive substantially from Common
Crawl, and the samples use different immutable shards and deterministic selections. Production
packing still requires global deduplication.

## Remaining boundary

The 35/30/25/10 tokenizer allocation remains provisionally feasible. It is not frozen for training:
the evaluation firewall must first pin general benchmark payloads and remove contamination, and a
human must accept each dataset license plus applicable Common Crawl/upstream terms. Cleanup/resume
qualification also remains open.

Artifacts:

- [Checked web summary](../results/data/web-tokenizer-sources-20260907.json)
- [Four-source overlap plan](../research/flagship/web_cross_source_duplicates_v1.json)
- Runtime overlap result: `/mnt/speck-data/speck/source-qualification/web-cross-source-dedup-v1/report.json`
