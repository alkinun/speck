# Data Preparation

Speck prepares remote text corpora into deterministic, sharded token streams. The pipeline resolves
repositories, discovers input files, streams and filters documents, performs global exact
deduplication, tokenizes text, and writes packed `uint16` training and validation shards.

## Prepare a Corpus

Run commands from the repository root. Download and verify the experiment tokenizer first:

```bash
uv run --extra cpu python -m scripts.tokenizer_prepare experiments/Speck1-140M
```

Then prepare the configured data:

```bash
uv run --extra cpu python -m scripts.data_prepare experiments/Speck1-140M
```

The Speck1.5 corpus is isolated from the base corpus:

```bash
uv run --extra cpu python -m scripts.data_prepare experiments/Speck1.5-140M
```

Pass `--restart` only when intentionally discarding an incomplete staged build. A completed output
directory is never overwritten.

## Output Paths

Runtime artifacts use `~/.cache/speck` by default. Set `speck_base_dir` to move the cache root before
running a command.

Packed data resolves in this order:

1. The explicit `data.output_dir` value.
2. `~/.cache/speck/data/<data.output_name>` when `output_name` is configured.
3. The legacy `~/.cache/speck/data/packed` path.

Preparation builds in a sibling `.building` directory and atomically promotes it only after the
corpus is complete.

## Base Pretraining Mixture

`experiments/Speck1-140M` requests 5,000,000,000 training tokens across three phases:

| Phase end | Ultra-FineWeb | DCLM | Cosmopedia v2 | FineMath-4+ | Ultra-FineWeb-L3 Multi-Style |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3,500,000,000 | 45% | 35% | 12% | 8% | 0% |
| 4,500,000,000 | 30% | 25% | 15% | 12% | 18% |
| 5,000,000,000 | 20% | 15% | 20% | 15% | 30% |

The phase durations and integer weights derive these source targets:

| Source | Tokens |
| --- | ---: |
| Ultra-FineWeb | 1.975B |
| DCLM | 1.55B |
| Cosmopedia v2 | 670M |
| FineMath-4+ | 475M |
| Ultra-FineWeb-L3 Multi-Style | 330M |

Preparation adds a 262,144-token per-source loader reserve for the configured maximum 65,536-token
distributed microbatch. Actual packed output can exceed 5B only by those reserves and one final
full-document overshoot per source.

The base recipe resolves floating source revisions when a build starts and records the resolved
commits in the staged state and final manifest. A completed build is internally reproducible, but a
new clean build can resolve newer source commits.

## Speck1.5 Curriculum

`experiments/Speck1.5-140M` pins every source revision and uses a foundation phase through 3.5B
tokens, a capability ramp through 4.5B, and a final 500M-token capability anneal:

| Phase end | FineWeb-Edu | DCLM-Edu | Ultra-FineWeb | DCLM | FineMath-4+ | UltraData-Math L3 Textbook-Exercise | UltraData-Math L3 Multi-Style | English Wikipedia | peS2o | Ultra-FineWeb-L3 Multi-Style | Cosmopedia v2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3.5B | 32% | 20% | 13% | 6% | 7% | 2% | 1% | 4% | 4% | 5% | 6% |
| 4.5B | 22% | 16% | 9% | 4% | 12% | 4% | 3% | 5% | 7% | 8% | 10% |
| 5.0B | 10% | 8% | 5% | 2% | 18% | 6% | 5% | 5% | 8% | 16% | 17% |

The phase durations produce these aggregate targets:

| Source | Tokens | Share |
| --- | ---: | ---: |
| FineWeb-Edu | 1.390B | 27.8% |
| DCLM-Edu | 900M | 18.0% |
| Ultra-FineWeb | 570M | 11.4% |
| DCLM | 260M | 5.2% |
| FineMath-4+ | 455M | 9.1% |
| UltraData-Math L3 Textbook-Exercise | 140M | 2.8% |
| UltraData-Math L3 Multi-Style | 90M | 1.8% |
| English Wikipedia | 215M | 4.3% |
| peS2o | 250M | 5.0% |
| Ultra-FineWeb-L3 Multi-Style | 335M | 6.7% |
| Cosmopedia v2 | 395M | 7.9% |

Category totals are 62.4% natural web, 13.7% math, 9.3% knowledge/science, and 14.6% general
synthetic. Specialist sources appear first so their documents win global cross-source
deduplication.

Source-specific constraints include:

- DCLM-Edu retains English rows with strict raw `edu_score > 3.5`.
- Ultra-FineWeb retains rows scored at least 0.8.
- Mixed-language UltraData-Math sources retain rows identified as English by pinned
  `py3langid==0.3.0`.
- peS2o uses only V2 `s2orc` full-text training shards and excludes `s2ag` title-and-abstract
  records.

Dataset repositories, revisions, paths, filters, and source order are recorded in
`experiments/Speck1.5-140M/data.json`.

## Determinism and Deduplication

Input discovery uses each Hugging Face repository tree rather than datasets-server previews. Files
are deterministically shuffled per source. The pipeline downloads and reads one remote Parquet or
gzip-compressed JSONL file at a time, removes it after processing, and writes source-local train and
validation shards under `sources/<source-id>/`.

Validation reserves 5M tokens per source. Validation loading schedules source streams equally;
training loading follows the configured phased weights.

Global exact deduplication normalizes text with Unicode NFKC, lowercasing, and whitespace collapse,
then records a 128-bit BLAKE2 hash. A hash collision is treated as a duplicate. Fuzzy and LSH
deduplication are intentionally excluded. The expected roughly 6M hashes are journaled at 16 bytes
each. Tokenizer calls are bounded to 1,024 documents and 2,000,000 aggregate input characters.

## Disk and Resume Behavior

Preparation performs a live disk-space preflight before creating staged data. For the 5B-token
recipe, the current estimate includes about 10.05GB of packed data, a 20GiB temporary raw-shard
allowance, and at least 5GiB of deduplication/index headroom, or about 36.9GB total. The preflight
reports required and free bytes and credits reusable staged bytes when resuming.

After each completed remote source file, preparation closes and checkpoints packed shards,
source-local index bytes, and the deduplication journal with checksums. A retry validates those
boundaries, removes only partial work from the interrupted file, and resumes at the next file.

The final manifest records source identities, resolved revisions, quotas, actual token counts,
filters, tokenizer identity, phase schedule, shard checksums, and preparation statistics. Training
and architecture search validate this manifest before consuming the corpus.
