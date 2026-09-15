# FineMath headroom and token cache complete

The [eleven-shard result](../../results/data/finemath-stock-headroom-20260914.json) completed on
2026-09-15 at 00:12 UTC. Its original launch-date filename is retained. The
[completion review](../../results/systems/finemath-headroom-verification-20260915.json) binds raw,
acquisition, exclusion and token-cache verification; the original eight-shard result and outage
attempts remain preserved.

The stock retains **753,071 documents / 3,674,756,372 UTF-8 text bytes / 1,124,167,472 Mistral tokens**
including BOS/EOS. It exceeds the 800M nominal requirement by 324,167,472 tokens and the **960M
preparation target by 164,167,472 tokens (17.10%)**. The target and filters were not lowered.
The old eight-shard retained text is a verified exact prefix. The combined stock replaces that
814,103,172-token measurement; the two banks must not be added.

All eleven complete raw files and acquisition units passed their hashes and configuration identities.
Acquisition retained 779,726 candidates. Final exclusion removed 14,238 exact reference matches,
1,047 verified-near reference matches and 11,370 verified-near candidate duplicates. Recomputed
analysis agrees with the published result, all reference outputs remain intact, exact/near controls
pass and final exact candidate/reference overlap is zero. The published index and removal hashes
also pass.

Natural postfilter domain proportions remain unchanged by a corpus cap: 56,661 known hosts, no
missing-host documents, byte HHI 0.006351. The largest host contributes about 4.38% of retained text
bytes. These are distribution diagnostics, not model-quality results.

The [token-cache receipt](../../results/data/finemath-token-stock-v2-20260914.json) reproduces exactly
753,071 documents and 1,124,167,472 tokens. The runtime plan binds the actual completed stock hash at
`/mnt/speck-data/speck/local-preparation-followups-v2-20260914/finemath_4plus-token-plan.json`.
The cache is under `/mnt/speck-data/speck/document-token-stock-v2/finemath_4plus`. All token shards
and the contiguous document index passed complete reopen. First/middle/last and documents crossing
physical shard boundaries were re-encoded and compared with the stored tokens during review.
This remains a reusable source cache, not joint background/treatment eligibility or a launch manifest.

The frozen successor exclusion invocation took **29,160.84 seconds (8.10 hours)** during concurrent
local preparation. SQLite commit accounted for **15,665.03 seconds (53.72%)** of that invocation;
this component is already included and must not be added again. The observed WAL peak was
**539,860,112 bytes (514.85 MiB)**: above the older 512 MiB observation but below this source plan's
2 GiB gate. Both capacity and storage gates pass. Cache construction measured 195.76 seconds;
surrounding verification adds cost.

These timings describe this larger shared-disk workload. They are not a controlled comparison with
the original resumed eight-shard run, a GH200 rate, or total historical all-attempt cost. The original
outage timing/WAL gaps remain disclosed. The commit cost identifies a concrete preparation bottleneck
for future execution qualification; successful headroom recovery does not establish a throughput gain.

The unattended sequence advanced to waiting for Cosmopedia after the cache passed. Its collected
systemd unit no longer supplies an exit status; the sequence explicitly records that limitation and
uses the bound completed publication and preparation gates. No model training or new source-use
approval occurred.
