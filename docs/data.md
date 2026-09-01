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

Prepare the isolated 20B-token Speck2 corpus in the same way:

```bash
uv run --extra cpu python -m scripts.data_prepare experiments/Speck2-140M
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

## Speck2 Curriculum

`experiments/Speck2-140M` preserves Speck1's broad-web foundation while adding a fixed English
Wikipedia allocation and using a gentler capability anneal. It requests 20B tokens across three
phases:

| Phase end | Ultra-FineWeb HQ | DCLM | Cosmopedia v2 | FineMath-4+ | Ultra-FineWeb-L3 Multi-Style | English Wikipedia |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14B | 43% | 31% | 8% | 8% | 5% | 5% |
| 18B | 39% | 27% | 10% | 11% | 8% | 5% |
| 20B | 31% | 19% | 14% | 12% | 19% | 5% |

The phase durations produce these exact aggregate targets:

| Source | Tokens | Share |
| --- | ---: | ---: |
| Ultra-FineWeb HQ | 8.2B | 41% |
| DCLM | 5.8B | 29% |
| Cosmopedia v2 | 1.8B | 9% |
| FineMath-4+ | 1.8B | 9% |
| Ultra-FineWeb-L3 Multi-Style | 1.4B | 7% |
| English Wikipedia | 1.0B | 5% |

Broad web declines gradually from 74% to 66% to 50%. General synthetic data rises from 13% to
18% to 33%, FineMath rises from 8% to 12%, and Wikipedia remains fixed at 5%. The aggregate mix is
70% broad web, 16% general synthetic, 9% math, and 5% reference knowledge.

Speck2 uses the newer `data/ultrafineweb_l1_en_hq` Ultra-FineWeb tree. It retains ordinary DCLM
rather than DCLM-Edu, uses only the English Multi-Style portion of Ultra-FineWeb-L3, and includes no
dedicated code, academic-paper, multilingual, instruction, or synthetic-math source. Every source
revision is pinned in `experiments/Speck2-140M/data.json`.

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

For Speck2, the same estimator requires 81,569,555,072 bytes, about 76GiB: approximately 40.1GB of
packed data, a 20GiB temporary raw-shard allowance, and 20.0GB of deduplication/index headroom.

After each completed remote source file, preparation closes and checkpoints packed shards,
source-local index bytes, and the deduplication journal with checksums. A retry validates those
boundaries, removes only partial work from the interrupted file, and resumes at the next file.

The final manifest records source identities, resolved revisions, quotas, actual token counts,
filters, tokenizer identity, phase schedule, shard checksums, and preparation statistics. Training
validates this manifest before consuming the corpus.
