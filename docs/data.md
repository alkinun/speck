# Data

The goal and next corpus are defined in [PLAN.md](../PLAN.md). Prepare one pilot with broad text,
math, and code; choose weights from actual eligible supply. Old source quotas are historical.

## Pipeline

Pin source revisions and permitted use → acquire/filter → deduplicate and exclude evaluation
material → partition → tokenize/pack → verify → train. Source-separated shards allow later mixture
changes without retokenizing. Recheck tokenizer fingerprints and hashes whenever reusing stock.

| Responsibility | Code / command |
| --- | --- |
| Source readers, configuration, packing, resume | `speck/data/{acquisition,configuration,packing,dataset}.py` |
| Global disk-backed deduplication | `scripts.production_data_preprocess` (`--batched-minhash` is optional) |
| Secret filtering, near duplicates, contamination | `scripts.text_gitleaks_filter`, `text_near_duplicates`, `text_contamination` |
| Source-use review | `scripts.source_rights_review` |
| Checked retained-stock tokenization | `scripts.tokenize_stock` |
| Distributed loading | `speck/data/loader.py` |

For a complete experiment directory containing `tokenizer.json` and `data.json`:

```bash
uv run --no-sync python -m scripts.tokenizer_prepare PATH_TO_EXPERIMENT
uv run --no-sync python -m scripts.data_prepare PATH_TO_EXPERIMENT
```

These can download substantial data. Use `make smoke` for the offline fixture. Each preparation
command accepts `--help`; the old finite source-acquisition jobs run only from the
[historical checkout](../archive/README.md), where their original plans and code are preserved.

## Retained material

The runtime store is `/mnt/speck-data/speck` on the maintainer's machine; portable code defaults to
`~/.cache/speck`, overridden by `speck_base_dir`. Existing corpora and caches were not moved or deleted.
The frozen tokenizer is under `tokenizer-final-mistral-v1` on that volume. Its model SHA-256 is
`dadfd56d766715c61d2ef780a525ab43b8e6da4de6865bda3d95fdef5e134055`.

FineWeb's cache manifest is under `document-token-stock-e1s-v1/fineweb_edu`; its receipt reports
2.307B tokens and a passing reopen. The 476.775M-token Stack-Edu acquisition still needs full
exclusion. Neither those receipts nor the completed specialist stocks establish a joint eligible
pilot corpus on their own. Reopen manifests, preserve source rights, count overlapping banks once,
and verify cross-source/validation separation before reuse. The complete receipts are in the Git snapshot.

The finite [first pilot](../experiments/pilot/README.md) has now been selected, jointly excluded,
packed, and reopened from those retained inputs. It stores 105,652,323 training tokens including
loader reserve and whole-document overshoot, plus 799,536 validation tokens. The scheduled training
exposure is 104,857,600 tokens. Its packed manifest is at
`/mnt/speck-data/speck/data/flagship-pilot-105m/manifest.json`; selection, exclusion, and packing
records are under `/mnt/speck-data/speck/flagship-preparation-20260917/pilot`.
The [compact preparation receipt](../experiments/pilot/preparation.json) binds those manifests and
records complete one- and four-rank CPU loader scans, exact fresh-process replay, and source/language
exposure. Benchmark matching removed 3,838 candidate documents before joint exclusion; those are
conservative matches, not proven leakage. Joint exclusion subsequently removed 19 code documents.
These finite pilot counts do not establish eligible supply for the main training horizon.

## Artifact discipline

Keep source revisions, filters, counts, hashes, tokenizer identity, data order, and output locations
in each run's manifests. Keep runtime data/checkpoints/logs outside Git, and back up irreplaceable
checkpoints before dependent work. Preserve failed attempts. Corpus text and packed shards are not
redistributed as model-release artifacts. Exact/near-duplicate and benchmark exclusion remain
requirements even though their old administrative workflow has been retired.
