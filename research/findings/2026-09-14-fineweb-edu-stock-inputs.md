# FineWeb-Edu stock inputs and raw acquisition

The [stock plan](../flagship/fineweb_edu_stock_preparation_v1.json) now binds all fourteen complete
`sample/10BT` files at approved revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`:
**9,672,101 physical rows / 28,518,193,415 compressed bytes**. The
[file manifest](../../results/data/fineweb-edu-shard-manifest-20260914.json) records immutable LFS
identities and remote Parquet footer hashes/schema/rows. Footer payloads and the discovery script
are retained in the runtime review directory. These observations still require full downloaded
payload verification and do not establish token capacity.

The original qualified file is the final `013_00000.parquet` shard. Its
[intake receipt](../../results/data/fineweb-edu-raw-cache-intake-20260914.json) verifies the
preserved original and new cache copy against the same SHA-256, size, schema and 182,101-row count.
At frozen implementation `67b913c`, its first 128 physical rows exercised the real stock reader
and py3langid-backed web-policy hook: 125 passed; three failed released language-confidence
criteria. This bounded check is before common security/Gitleaks/full exclusion and is not a
source-capacity measurement.

The [explicit corpus policy](../flagship/FINEWEB_EDU_STOCK.md) preserves qualified document
criteria and requires valid ID, integer grade and bounded language confidence. It deliberately
omits the tokenizer-sampler 2MB host cap, preserves natural postfilter host proportions and
requires concentration reporting. The target remains 4.4B nominal plus 20% measured Mistral
headroom. The original shard and any overlapping older bank are included, not additive supply.

The [raw-only launch](../../results/systems/fineweb-edu-raw-launch-20260914.json) freezes the
implementation, plan, service and log identities. It runs sequential complete-file downloads
with full hash/schema/row checks and durable per-file receipts, using Nice 19 and idle I/O
scheduling while FineMath, Cosmopedia and Stack-Edu processing continue. Failed attempts remain
recorded; resume verifies completed files and requires exact execution identity. Text processing,
reference exclusion, actual capacity, token caching and joint launch manifests remain separate.

Validation: **1,140 passed, 10 skipped, 125 deselected**, with formatting, lint, research catalog,
archive, source-pin and diff checks. The first formatting check found a leading blank line after
an unused-import fix; correcting it preceded the successful full run. Tests cover invalid metadata,
security/host guards, config-bound corpus policy, raw-file corruption/schema/count rejection and
interrupted acquisition replay. No tokenizer confirmations or model training were launched.
