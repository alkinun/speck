# Combined peS2o stock clears the preparation-headroom target

The [combined result](../../results/data/pes2o-stock-headroom-20260914.json), produced from clean
revision `bdba95b`, retains **113,232 documents**, **3,290,876,773 UTF-8 text bytes**, and
**820,097,493 frozen-Mistral tokens**, including BOS/EOS. Both complete pinned shards cover 221,898
physical rows. Capacity exceeds the **480M** target (400M nominal plus 20% preparation headroom).

The first acquisition unit is reused under the exact same configuration and payload hashes; the
second is newly filtered. The resulting 114,139 acquisition records are jointly excluded and
deduplicated. Removals comprise 873 exact reference matches, 15 exact candidate duplicates and
19 verified-near candidate duplicates. Exact/near controls pass; reference outputs remain unchanged;
final exact candidate/reference overlap is zero. Raw, plan, source-use, output, index and removal-ledger
identities were independently verified after publication.

The old retained science file is an exact **1,664,244,646-byte / 55,787-document prefix** of the new
retained file. Hashing that prefix reproduces the original output SHA-256
`07b737ad15297090448befb99708042dc319309899339317f239fa1d944f9b15`. Thus the earlier 403.56M-token
stock is demonstrably included. **Replace its capacity with 820.10M; do not add the two banks.**

## Operating observations

- Original-unit validation/copy: **1.88 seconds**.
- Acquisition loop including first-unit reopen and new second-unit filtering: **1,494.79 seconds**;
  this excludes the already-recorded first-shard filtering cost.
- Private complete-reference restoration: **114.74 seconds**.
- Combined exclusion invocation: **1,180.19 seconds**.
- Observed WAL peak: **426,840,272 bytes / 407.07 MiB**.

These are bounded local measurements; FineMath preparation overlapped part of the run. They are not
isolated throughput comparisons or a GH200-site forecast. The configured FULL-sync SQLite declaration
and unchanged source license/English/OCR criteria remain bound.

The [expanded token-cache plan](../flagship/pes2o_token_stock_v2.json) prepares a new immutable
document-indexed cache. Joint experiment-background eligibility and exact loader/launch manifests
remain pending, even though source-specific preparation headroom now passes.

The [expanded token cache](../../results/data/pes2o-token-stock-v2-20260914.json) is complete at the
same 820,097,493 tokens and 113,232 documents. Document probes, all shard/index hashes and a separate
complete reopen pass. Every earlier token shard, including the partial last shard, reproduces the
corresponding prefix of the expanded token stream. Cache construction/verification took 135.79 seconds
inside the packing function, excluding plan loading; this is not an isolated hardware benchmark.
