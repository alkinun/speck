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
target is short by 145,896,828 tokens. Recovery and storage checks passed; token caching remains
pending. The command above identifies the completed run and must not be restarted into its existing
output. Additional supply requires a bound successor. See the [finding](../findings/2026-09-14-finemath-stock.md).
