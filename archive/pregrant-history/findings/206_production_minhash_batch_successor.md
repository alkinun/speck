# 206 — Batched MinHash preserves exact dedup signatures and removes the resume bottleneck

The resumed 2B production calibration advanced only 30,000 records in approximately one hour once its
global SQLite index contained about 505,000 documents. Profiling isolated the dominant avoidable cost:
the frozen reference called `MinHash.update` once for every lexical shingle while the same datasketch
implementation exposes `update_batch`.

On 1,000 deterministic shingles, 128 permutations, seed 42, and 200 iterations, scalar and batched
paths produce byte-identical hashvalues. The local CPU microbenchmark measures 0.8902 seconds scalar
and 0.1009 seconds batched, an 8.82× signature-construction speedup. This is not an end-to-end dedup
speed claim: SQLite candidate lookup and exact Jaccard verification remain unchanged and may dominate
later.

The production-rehearsal successor temporarily installs the batched function only around the global
dedup call and restores the frozen implementation afterward. Exact hashvalues make the existing
checkpoint-57 SQLite state compatible; acquisition and the first 535,534 processed records are not
repeated. The calibration resumes as a persistent user service rather than a one-hour coding-harness
background job.

This is mechanical calibration qualification only. It does not pass global dedup, production
operations, data quality, or training authority before the remaining records and all downstream stages
complete.

Artifact: [batch qualification](../results/data/production-minhash-batch-qualification-20260910.json).
