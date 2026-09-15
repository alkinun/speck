> **Scope update, 2026-09-15:** the [long-context pivot](PIVOT.md) supersedes old experiment budgets, quotas and selection dependencies in this note. Retained mechanics and measurements remain evidence at their recorded identities. Use [LONG_CONTEXT_DATA.md](LONG_CONTEXT_DATA.md) and [FIRST_WAVE.md](FIRST_WAVE.md) for new preparation; do not automatically expand the retired wave.

# FineWiki reference stock

The first-wave proposal needs 400M reference-category tokens for E1W's common background. The older
checked bank contains 64.7M Mistral-reference FineWiki tokens. This preparation measures a larger
source-identical stock before final-tokenizer packing and joint experiment-view exclusion.

The [fixed plan](finewiki_stock_preparation_v1.json) covers all 421,456 physical rows of the first
English FineWiki shard. Its 2,510,037,970 bytes and SHA-256 are pinned to the approved dataset revision;
the existing local download is verified and reused. A Parquet footer check verifies complete-shard
row coverage before filtering.

The existing source acceptance and qualified reader are inherited from the checked calibration plan:
English `in_language`, `text` serialization, 200–100,000-character documents, preserved ID/title/URL,
and the established benchmark, security, repeated-line and Gitleaks policies. No new source-use decision
or text-field substitution is introduced by this stock plan.

The same reusable orchestration now serves natural Math L2 and FineWiki. Acquisition units are
independently resumable. Full reference-prefix restoration binds all six candidate slots explicitly;
only the reference-category candidate slot is populated here. The 288,872 firewall records retain
precedence. Their preservation, positive exact/near controls, and zero final exact overlap are checked.
The bound FULL-sync SQLite policy retains its 65,536-page checkpoint trigger and measured WAL envelope.

```bash
uv run --no-sync python -m scripts.prepare_reference_stock \
  research/flagship/finewiki_stock_preparation_v1.json \
  results/data/finewiki-stock-preparation-20260913.json
```

Use a clean checkout and new destinations. `--resume` requires the same revision and plan. A shortfall
is a recorded result, not an automatic enlargement of the source window. Mistral-reference counts
include BOS/EOS. This bank is not added to the older bank as distinct supply; overlap is expected.
Final-tokenizer counts, packing headroom, joint background deduplication, exact data orders, and model
launch manifests remain necessary before E1/E3 execution.

## Completed stock

The [result](../../results/data/finewiki-stock-preparation-20260914.json) retains **387,313 documents**
and **582,070,378 tokens**, covering the nominal 400M requirement. The counting tokenizer is byte-
identical to the [frozen Mistral base tokenizer](tokenizer_decision_v1.json). The run resumed after
acquisition; exclusion timing and WAL observations cover the resumed invocation only. Full details
and remaining eligibility/packing requirements are in the [finding](../findings/2026-09-14-finewiki-stock.md).
