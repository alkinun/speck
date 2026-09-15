# FineMath incumbent stock

The [fixed plan](finemath_stock_preparation_v1.json) covers the first eight complete FineMath-4+
Parquet files at the approved revision, totaling 837,440 physical rows and 2,289,547,838 compressed
bytes. The [shard manifest](../../results/data/finemath-shard-manifest-20260914.json) records immutable
LFS hashes/sizes and revision-pinned remote footers. Acquisition must still verify every complete local
file SHA and physical row count; remote footer inspection alone is not raw-file qualification.

## Explicit preparation policy

The source reader requires released `language=en` and `int_score>=4`. Additional checks require finite
language score in [0.8, 1], qualified 256–1,000,000-byte size limits, valid available URLs and the
qualified PII/secret criteria. The inherited acquisition reader additionally bounds documents to
200–100,000 characters and applies benchmark, security, repeated-line and Gitleaks checks.

The same confident-English math-prose criterion used for natural Math L2 applies: at least 80
alphabetic prose characters after the existing math-aware extraction and English probability ≥0.8.
Insufficient prose is excluded for these preparation views. FineMath's extra released grade/language
metadata is retained; no corresponding labels are invented for UltraData-Math. Text is not rewritten.

**Corpus selection differs explicitly from the earlier bounded tokenizer sample.** Its 2MB-per-host
cap was a small-sample diversity rule. This pre-results preparation successor does not impose that
absolute cap on the larger training-source stock. It retains the natural postfilter host distribution
and reports post-exclusion concentration: known/missing-host coverage, largest hosts by UTF-8 bytes,
and known-host byte HHI. The report is diagnostic and does not secretly reweight the source.

Global exact/verified-near deduplication follows the full reference superset with the bound SQLite
policy. The fixed target is **960M tokens**, covering the proposed 800M incumbent requirement plus
20% preparation headroom. Any shortfall is recorded; no filter relaxation or automatic file expansion
occurs inside this execution. The new bank is not summed with overlapping older FineMath stock.

```bash
uv run --no-sync python -m scripts.prepare_finemath_stock \
  research/flagship/finemath_stock_preparation_v1.json \
  results/data/finemath-stock-preparation-20260914.json
```

Run from a clean revision. Acquisition is checkpointed per file; `--resume` requires the same revision
and plan. Joint experiment-background eligibility, document-span assembly, token order, exact quotas,
and launch manifests follow completed stock verification.

## Completed first stock

The [checked result](../../results/data/finemath-stock-preparation-20260914.json) retains 545,996
documents and 814,103,172 tokens. The 800M nominal requirement passes, while the 960M preparation
target is short by 145,896,828 tokens. Recovery and storage checks passed. This result is superseded by the completed eleven-shard
stock and token cache below. The command above identifies the completed run and must not be restarted into its existing
output. Additional supply requires a bound successor. See the [finding](../findings/2026-09-14-finemath-stock.md).

## Eleven-shard headroom successor

The [v2 plan](finemath_stock_preparation_v2.json) binds the completed result and the original plan,
keeps their policy identities, and adds complete files 8–10 at the same approved revision. Its
[shard manifest](../../results/data/finemath-shard-manifest-v2-20260914.json) contains 1,151,480
physical rows and 3,152,243,198 compressed bytes. The first eight entries are unchanged. The three
new files have revision-pinned LFS identities and remote footer checks; full local SHA/row checks
occur before their acquisition is accepted.

The 145.90M-token gap suggests roughly two additional shards at the observed mean yield. Three
were selected under preparation delegation to allow yield uncertainty, before any model outputs.
This does not predict a capacity pass or change the 960M target. Existing human source-use approval
continues to govern; preparation delegation is not model-launch authority.

The successor reuses all eight completed acquisition units only after configuration and payload
verification. It reruns full reference exclusion on the combined stream in a separate output tree,
including the maintained recovery index. No retained-token totals are added across the two banks.
The loader rejects changed original shard entries, policy drift, missing reuse declarations and
output nesting/overwrites. Acquisition configurations for all eight reused units match exactly.

```bash
uv run --no-sync python -m scripts.prepare_finemath_stock \
  research/flagship/finemath_stock_preparation_v2.json \
  results/data/finemath-stock-headroom-20260914.json
```

After successful publication, verify headroom and bind a separate token-cache plan to the actual
result hash. No speculative result identity or automatic training launch is allowed.

## Completed successor and token cache

The [v2 completion finding](../findings/2026-09-15-finemath-headroom.md) records **753,071 documents
and 1,124,167,472 tokens**. The 960M preparation target passes by 164,167,472 tokens. The earlier
eight-shard retained text is a verified exact prefix and is not additional supply. Full raw,
acquisition, exclusion/index/removal, reference and token-cache checks pass. The cache is available
at `/mnt/speck-data/speck/document-token-stock-v2/finemath_4plus`; its
[receipt](../../results/data/finemath-token-stock-v2-20260914.json) binds the actual runtime plan.
Both listed preparation commands now identify completed runs and must not be restarted into their
existing outputs. Joint experiment-view assembly remains the next scientific data qualification.
