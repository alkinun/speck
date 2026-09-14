# Math L2 and FineWiki now have document-indexed token stock

Clean implementation `504f03e0982dcd3da7437a2a28e0a96a7ba3d89d` produced the checked
[Math L2](../../results/data/math-l2-token-stock-20260914.json) and
[FineWiki](../../results/data/finewiki-token-stock-20260914.json) token caches:

| Source | Documents | Tokens | uint16 token bytes |
| --- | ---: | ---: | ---: |
| Natural UltraData-Math L2-preview | 169,058 | 384,788,210 | 769,576,420 |
| English FineWiki | 387,313 | 582,070,378 | 1,164,140,756 |

The caches use the frozen Mistral artifact and include BOS/EOS for each unchanged document. Each
source's totals exactly reproduce its earlier independent count. Token shards are bounded at 100M
tokens. A JSONL index binds original content ID/hash and source ordinal to token start/length and
UTF-8 byte count; full original metadata remains available through the pinned source-text input.

Both sources passed first/middle/last span readback, including cross-shard reads, and complete
payload/index hash checks. A separate reopen invocation verified contiguous document coverage and
identical manifest hashes. The reported 68.21s Math and 94.52s FineWiki durations cover cache building
and verification inside the packing function, excluding plan-loading checks and inherited preparation.
The jobs overlapped on the same host, so these are not isolated throughput comparisons.

These are **token caches**, not finalized E1/E3 corpora. No mixture, training order or validation
partition is selected. Subsequent per-arm assembly must determine eligible whole-document spans
after joint background/treatment exclusion and bind exact quotas, order and loader lookahead. This
avoids retokenizing unchanged text while keeping membership decisions explicit.

See [TOKEN_STOCK.md](../flagship/TOKEN_STOCK.md) for commands, ownership/reopen behavior and the
retained-unpublished-build boundary.
