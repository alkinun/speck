# Preparing the peS2o science background

The first-wave proposal requires 400M source-identical peS2o v3 tokens. PubMed counts are not peS2o
capacity. The [fixed preparation](pes2o_stock_preparation_v1.json) processes the complete 110,183-row
`train-0040-of-0136.zst` shard already used in the bounded science qualification. The pinned revision,
983,683,365-byte size and SHA-256 are checked; full physical row count is verified before acquisition.

## Source and text policy

The source inherits the existing human acceptance and the qualified science filter declaration.
Document licenses must be one of `CC0`, `CCBY`, `CCBYSA`, `pd`, or `public-domain`; missing and other
licenses are rejected. OA URL, license, paper ID, creation date, source partition, dataset revision,
file and physical row are preserved. This does not infer a missing title or parent-paper relationship.

The qualified science criteria require English probability at least 0.8 with pinned py3langid,
256–1,000,000 UTF-8 bytes, alphabetic ratio at least 0.2, repeated-line ratio at most 0.4, at most four
spaced-OCR sequences, replacement-character ratio at most 0.001 and boilerplate-line ratio at most 0.2.
Its placeholder-aware PII and high-confidence-secret checks also apply. There is no science-term
minimum or host quota in this peS2o declaration.

The inherited acquisition path additionally applies its 200–100,000-character reader envelope and
benchmark/security checks, followed by Gitleaks. These are sequential filters; rejection counts are
not independent prevalence estimates. Exact/near candidate deduplication is performed downstream,
after giving the complete firewall-reference superset precedence. Text is retained without rewriting.

## Execution and follow-up

```bash
uv run --no-sync python -m scripts.prepare_science_stock \
  research/flagship/pes2o_stock_preparation_v1.json \
  results/data/pes2o-stock-preparation-20260914.json
```

Execute from a clean revision; `--resume` requires that same revision and plan. Zstandard input is
streamed, with physical rows preserved across filtering and resume. The stock counter is bound to
the frozen Mistral base tokenizer, including BOS/EOS. Record a shortfall without silently expanding
this shard window. Joint background/treatment eligibility, packing headroom and launch manifests
remain separate requirements even when nominal capacity passes.

## First measured stock

The [result](../../results/data/pes2o-stock-preparation-20260914.json) retains 55,787 papers and
403,558,463 selected-tokenizer tokens. Nominal capacity passes, but only 0.89% headroom remains.
Prepare additional pinned science supply before claiming joint-view/packing readiness. The
[finding](../findings/2026-09-14-pes2o-stock.md) records all counts and operating boundaries.

## Headroom successor

[V2](pes2o_stock_preparation_v2.json) adds the complete pinned shard 41, whose
[intake](../../results/data/pes2o-next-shard-intake-20260914.json) verifies 111,715 physical rows and
the released LFS SHA-256. The target is 480M tokens: the nominal 400M plus 20% preparation headroom.

The completed first acquisition unit is copied only after its exact configuration and payload hashes
match the new plan. Only its config, manifest, retained records and security report are copied; old
scratch and interrupted outputs are not imported. The second shard is filtered normally. Both units
are then concatenated in fixed shard order and jointly deduplicated with the full reference prefix.
The combined bank replaces the first measurement, rather than adding overlapping totals.

```bash
uv run --no-sync python -m scripts.prepare_science_stock \
  research/flagship/pes2o_stock_preparation_v2.json \
  results/data/pes2o-stock-headroom-20260914.json
```

Reuse verification/copy time is reported separately from the acquisition loop; the original first-
shard filtering cost is not charged again as new work. Exact policies, source approval, and tokenizer
remain bound. Completed first-shard token stock is retained while larger joint eligibility is measured.
