# Data

The shared data pipeline, its rules and its commands. The [program design](program.md) defines the
experiments. Per-source gates are in
[source-readiness.json](../experiments/main-data/source-readiness.json), and the mixture is in
[plan.json](../experiments/main-data/plan.json) (`parent.starting_mixture`).

## Pipeline

Every record at every stage follows one auditable path:

1. identify source, licence or use condition, family, acquisition receipt and transformations;
2. extract with a source-specific parser, keeping the original identity;
3. remove exact and near duplicates, keeping a family graph for held-out splits;
4. apply quality, language, safety, benchmark-contamination and coverage filters;
5. attach lineage, quality fields, derivation cost and accepted-token accounting;
6. partition by source family before tokenization and packing;
7. verify manifests, token counts, packing, sampling weights, checksums and resumability;
8. train only from the hash-bound manifest.

The [data-design contract](../experiments/main-data/data-design-contract.json) lists the manifest
fields each stage records. No gate is relaxed for small runs.

**Derived text.** The parent's starting mixture is natural and source-traceable. Derived text enters
pretraining only as a declared arm (P6 on the ladder, D1 on the parent), identified by lineage,
generator, verification and cost. It never fills a gap in natural supply. AI-assisted filtering may
recommend decisions, but the original source is kept and the model, prompt and decision recorded.

**Counting.** Raw stock, accepted unique tokens, exposure and rejected material are separate counts.
Accepted tokens are counted only after family, overlap and extraction review, and overlapping banks
count once. Every repeated pass is declared and counted as exposure.

| Responsibility | Code |
| --- | --- |
| Source readers, configuration, packing, resume | `speck/data/{acquisition,configuration,packing,dataset}.py` |
| Global disk-backed deduplication | `scripts.production_data_preprocess` (`--index-directory` puts the index on flash) |
| Benchmark exclusion | `speck/evaluation/protocol.py` (`BenchmarkExclusion`), `scripts.livecodebench_exclusion` |
| Cross-source family graph and partitions | `scripts.joint_family_graph` |
| Ladder corpora from family buckets | `scripts.ladder_prepare` |
| Per-document token index | `speck/data/document_index.py` |
| Source-use review | `scripts.source_rights_review` (validates a human decision; never makes one) |
| Distributed loading | `speck/data/loader.py` |

## Acquiring candidate stock

Two resumable runs grow the selected-web and code stock. Each step writes a receipt, reruns resume
from the last completed unit, and nothing is admitted. `D` is the runtime store
(`/mnt/speck-data/speck`). Put `TMPDIR` and the SQLite index on local flash, and run one heavy writer
per spinning disk.

**Ultra-FineWeb HQ crawl.** This reuses the 2026-09-22 HQ preprocess plan and its firewall and
deduplication policy.

```bash
python -m scripts.ultrafineweb acquire experiments/corpus-audit/ultrafineweb-hq-listing.json CRAWL $D/hq-CRAWL
python -m scripts.ultrafineweb convert $D/hq-CRAWL $D/joint-family-partition-20260922/hq-preprocess-plan.json $D/hq-CRAWL-preprocess
python -m scripts.production_data_preprocess $D/hq-CRAWL-preprocess/preprocess-plan.json --index-directory FLASH_DIR
python -m scripts.ultrafineweb census $D/hq-CRAWL-preprocess/excluded $D/hq-CRAWL-census $D/hq-CRAWL-census.json
```

The supply gap uses the census receipt's `distinct_from_fineweb_edu_tokens`.

**Stack-Edu code.** Fetch each language of a tier, summarize the completed tranches, then convert
the retained stock and every tranche into one preprocessor input.

```bash
python -m scripts.stack_edu acquire $D/stack-edu-census-20260923/census-listing.json experiments/corpus-audit/stack-edu-metadata-census.json LANGUAGE TIER $D/stack-edu-content-v1
python -m scripts.stack_edu summarize $D/stack-edu-content-v1 experiments/corpus-audit/stack-edu-acquisition.json
python -m scripts.stack_edu convert $D/data-qualification-20260919/code-supply/acquisition.json $D/stack-edu-content-v1 $D/joint-family-partition-20260922/hq-preprocess-plan.json $D/stack-edu-code-preprocess
python -m scripts.production_data_preprocess $D/stack-edu-code-preprocess/preprocess-plan.json --index-directory FLASH_DIR
```

`summarize` skips a tranche until its `tranche.json` exists. Tier 3 is 6,045 units of 4,096 rows
across 15 languages, about a minute per unit. `scripts.stack_edu_census` rebuilds the metadata
census.

**Storage.** Each billion packed uint16 tokens takes about 2 GB before indexes and intermediates.
Bulk corpora suit a large disk. The deduplication index, small unit files and scanner scratch are
random synced writes and belong on flash. Fsync a unit's records before the manifest that marks it
complete. The frozen tokenizer is `tokenizer-final-mistral-v1` (model SHA-256
`dadfd56d766715c61d2ef780a525ab43b8e6da4de6865bda3d95fdef5e134055`).

## Source quality

Code and reasoning quality are a pretraining requirement, not something left to SFT. Prioritize
natural code with tests, documentation, API use and project context; correct worked math and
scientific explanation from foundations to hard problems; and coherent repository bundles for
context extension. Keep everyday and nontechnical prose as well. An educational score, long
reasoning or a passing generated test is not quality by itself.

Rules carried from the [audits](../experiments/corpus-audit/README.md):

- Score labels are not interchangeable cutoffs. Name the exact field and operator; FineMath 4+ or
  InfiWebMath 3+ labels are not continuous-score thresholds.
- Extraction and arithmetic flags are review hints until validated, not filters or correctness
  certificates.
- An exact normalized benchmark match is a floor for contamination, not a clearance.
- Keep source-family identity across original pages, rewrites, Q&A and derivatives; related forks,
  patches and exercises share one family.
- Natural code needs immutable identity, applicable notices, intact content, family and benchmark
  exclusions and joint deduplication. It does not need invented tests. Verified exercises also need
  clear specifications, linked tests and independent oracles, and are tracked separately. Notice
  text governs, not a scanner or publisher label.

The P5 code contrast (Stack-Edu against Stack v3) keeps total code share, language coverage, other
banks and serialization fixed.
