# Large raw FineWiki records are highly sensitive to table whitespace

The [offline raw probe](../../results/data/finewiki-long-probe-20260916.json) scans the already pinned
first English FineWiki shard: 421,456 physical records, 2,053 above the earlier 100,000-character
stock limit, of which 25 exceed the probe's separate 2,000,000-character bound. It tokenizes only the
16 largest records within that bound using the frozen Mistral tokenizer. No new data was downloaded;
the complete raw SHA and source-use/preparation identities are bound in the result.

All 16 reach 64K+1 tokens before any new filtering, and nine reach 128K+1. Those raw lengths are
misleading as a measure of useful content. The [structural review](../../results/data/finewiki-long-structure-20260916.json)
reopens the exact records and checks their content hashes and original token counts. Whitespace makes
up 87.38–98.29% of their characters. For example, the 1,910,829-character record titled “Ibn Furak”
contains 1,826,972 whitespace characters and extensive padded table cells.

The diagnostic collapses horizontal whitespace runs to one space while preserving newlines and
non-whitespace characters, then counts tokens again. It neither edits the raw corpus nor selects a
production normalization. The result shows how strongly padding affects apparent length:

| Intact record length including lookahead | Original sample | After diagnostic whitespace collapse |
| --- | ---: | ---: |
| ≥32K+1 tokens | 16 | 10 |
| ≥64K+1 tokens | 16 | 4 |
| ≥128K+1 tokens | 9 | 0 |

These are physical-record counts, not independent-family counts. Two records carry the same
`enwiki/9962` article ID with different text hashes. The nine original 128K candidates represent eight
distinct article IDs; the four collapsed 64K candidates represent three. Neither article IDs nor this
small review establish complete edition/family independence.

This intentionally biased sample cannot estimate whole-source yield or establish that genuine long
FineWiki documents are absent. It does establish that increasing the character cap and counting raw
tokens would not suffice. Source-aware extraction/normalization, repetition/security checks, family
identity and actual post-exclusion supply remain prerequisites. No candidate was admitted to a
training stock; qualified supply remains unmeasured. The earlier short-bank filters and receipts are
unchanged. Four structural-check tests and three original-probe tests passed. Logs are retained under
`/mnt/speck-data/speck/finewiki-long-{probe,review}-20260916.log`.
