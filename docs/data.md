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

## Main-corpus headroom

The [initial content audit](../experiments/corpus-audit/README.md) now adds a reproducible
924-document review packet, 34 qualitative excerpt reviews, a full FineMath host census, and
document-length measurements. It identifies concrete selection questions; it does not certify
corpus quality. Main training needs a labeled quality/coverage audit and one costed data comparison
alongside the supply and integrity checks below. The frozen engineering pilot remains unchanged.
The [follow-up decisions](../experiments/corpus-audit/README.md#follow-up-decisions--2026-09-18)
keep FineMath, prepare a reversible 8,563-document topic-directory exclusion candidate, and
prioritize consistency checks on English L3 Q&A before any synthetic-source replacement.
No blanket numeric-deduplication rule or new main mixture has been adopted.
The [web candidate shortlist](../experiments/corpus-audit/README.md#candidate-decisions) explicitly
prioritizes natural Ultra-FineWeb qualification following the
[matched paper results](research.md#openbmb-web-data-review--2026-09-19), with FineWeb-Edu as
the control, DCLM baseline/DCLM-Edu as independent candidates, and synthetic L3 assessed separately.
Their historical pins match current repository heads as checked September 19. None is in the
frozen pilot; the main mixture still needs a comparable content/coverage audit and source eligibility.

The [coding plan](coding.md) now prioritizes a checked-exercise comparison. A pinned, bounded
UltraData-Code preview establishes schema and serialization questions, not training eligibility
or correctness. Preserve practical and multilingual code coverage while preparing that candidate.

The [September 18 reopen](../experiments/pilot/supply.json) verifies all shard hashes and every document-index span for the five
retained token stocks used by the pilot. It counts the larger peS2o v2 bank once; its v1 predecessor
is not additional supply. Stack-Edu below is the archived acquisition receipt, before full exclusion.

| Source | Retained tokens | Pilot share |
| --- | ---: | ---: |
| FineWeb-Edu | 2,306,703,052 | 50% |
| Stack-Edu | 476,774,847 | 15% |
| FineMath 4+ | 1,124,167,472 | 15% |
| Cosmopedia v2 | 1,489,288,743 | 10% |
| peS2o v3 | 820,097,493 | 5% |
| FineWiki English | 582,070,378 | 5% |

These are 6,799,101,985 source tokens before joint eligibility, not a 6.8B training manifest.
At the frozen pilot weights, total code supply gives a 3.18B-token single-pass mixture upper bound;
the 5% Go share within code lowers it to **2,755,093,600 total mixture tokens**. Validation reserves,
benchmark exclusions, and joint duplicates reduce that ceiling. FineWeb's corresponding ceiling
is 4.61B. A long run needs additional approved source acquisition, explicitly permitted repetition,
or a separately frozen mixture; existing specialist stocks cannot fill arbitrary missing code/web
quotas. Natural UltraData-Math remains separate and is not silently added to this mixture.

At uint16 storage, each billion packed tokens requires about 2 GB for token IDs, before indexes,
validation, preparation intermediates, and duplicate databases. Keep raw acquisition, exclusion
outputs, packed data, and recovery checkpoints separately budgeted. The rental needs only the
236 MiB pilot directory and small assistant pack; it does not need the full retained stock.
After the pilot, choose the main horizon from measured all-in throughput and eligible supply,
then run the same source-rights, joint-exclusion, partition, pack, and full-loader checks at that scale.

## Artifact discipline

Keep source revisions, filters, counts, hashes, tokenizer identity, data order, and output locations
in each run's manifests. Keep runtime data/checkpoints/logs outside Git, and back up irreplaceable
checkpoints before dependent work. Preserve failed attempts. Corpus text and packed shards are not
redistributed as model-release artifacts. Exact/near-duplicate and benchmark exclusion remain
requirements even though their old administrative workflow has been retired.
