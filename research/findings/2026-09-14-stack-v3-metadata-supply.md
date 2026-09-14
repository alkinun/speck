# Restricted Stack v3 local metadata supply

The [bound census](../flagship/stack_v3_metadata_census_v1.json) scanned the twelve complete
previously downloaded files at the approved revision. The [result](../../results/data/stack-v3-metadata-feasibility-20260914.json)
binds their hashes and the preserved analysis script. All 252,860 repository rows and 2,841,926
nested physical files were covered in 55.61 seconds including full input hashing and concurrent
local preparation. The earlier sampler counted files only after repository filtering; these
physical-file totals therefore have a different boundary.

This pass projects metadata columns without loading source text. It retains the recorded
permissive license allowlist, released vendor/fork and size/language rules, plus the common
explicit vendor-path exclusions. The tokenizer sampler's 1MB per-language/repository cap is
explicitly omitted as a full-stock rule. No language quotas or source-use decisions change.

| Language | Eligible declared MB (decimal) | E1S target M tokens with headroom |
| --- | ---: | ---: |
| Python | 30.895 | 90 |
| C++ | 23.832 | 54 |
| C | 9.439 | 18 |
| Java | 49.536 | 54 |
| JavaScript | 13.590 | 54 |
| TypeScript | 6.135 | 36 |
| Rust | 5.046 | 18 |
| Go | 8.333 | 18 |
| Shell | 2.360 | 3.6 |
| SQL | 1.278 | 3.6 |
| Markdown | 18.141 | 10.8 |

Declared bytes are **not tokens or usable capacity**. Content identity, English prose, security,
Gitleaks and full reference/candidate exclusion remain necessary. Eligible released content IDs
are distinct within each language across these files; this does not establish cross-language
uniqueness or ancestry independence. The upstream ID is not assumed to be plain SHA-1 of released
text.

The cached twelve-file view does not establish the required code supply. Before further large
transfers, inspect whether pinned metadata can target eligible repositories/files more efficiently,
or whether additional complete shards are required. Do not substitute these counts into paper
capacity tables, silently relax filters, or treat restricted Stack v3 as an already-ready larger
shared background. The Stack-Edu expanded census and matched-language feasibility decision remain
separate dependencies.
