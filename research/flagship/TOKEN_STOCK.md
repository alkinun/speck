# Document-indexed token stock

The selected Mistral tokenizer can now convert qualified text stock into reusable token shards.
Each document retains an ordinal, content ID and text SHA-256, UTF-8 byte count, token start, and token
length. These refer to the hash-bound source JSONL containing its full provenance metadata.

The cache includes BOS/EOS per document and packs unchanged text in retained source order into
100M-token uint16 shards. Complete document/token totals must equal the earlier checked count under
the byte-identical selected tokenizer. First/middle/last document spans are read back across shard
boundaries and compared with encoded tokens; every published shard and index receives a SHA-256.
Reopen verifies payload hashes and complete contiguous document coverage.

This resolves tokenization work without choosing a training distribution. Experiment assembly can
later reference only eligible document ordinals/spans after joint-background and treatment exclusion,
then materialize the selected order, quotas and loader lookahead. The cache is not a training dataset
manifest: it specifies no mixture, train/validation split, seed order, or experiment-launch authority.

```bash
uv run --no-sync python -m scripts.tokenize_stock \
  research/flagship/math_l2_token_stock_v1.json results/data/math-l2-token-stock-20260914.json
uv run --no-sync python -m scripts.tokenize_stock \
  research/flagship/finewiki_token_stock_v1.json results/data/finewiki-token-stock-20260914.json
```

The implementation must be clean at execution. Complete caches are verified and reused. Publication
renames the staging directory only after validation. An interrupted unpublished cache remains in its
`.building` directory and is never deleted or reused implicitly; use an explicit new plan/destination
after investigating it. This stage does not claim power-loss qualification beyond the underlying
writer's existing behavior.

## Completed first caches

The [Math L2 result](../../results/data/math-l2-token-stock-20260914.json) and
[FineWiki result](../../results/data/finewiki-token-stock-20260914.json) reproduce 384,788,210 and
582,070,378 tokens respectively. Both passed complete published reopen verification and document-span
readback. The [finding](../findings/2026-09-14-document-token-stock.md) records counts and scope.

The [first peS2o cache](../../results/data/pes2o-token-stock-20260914.json) also passed document-span
readback and complete reopen at 403,558,463 tokens. Science headroom remains under expansion; this
cache is a retained subset, not extra unique supply to add to the forthcoming combined bank.
