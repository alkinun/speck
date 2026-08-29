# Speck

Speck is a compact research harness for experimenting with small causal language model architectures, training them, benchmarking optimization performance, and running checkpoint inference.

## Model

Models are groups of residual blocks. Each block contains ordered stages, and a stage can run one or more branches in parallel. Supported architecture components include:

- Global or sliding grouped-query attention.
- Gated causal convolution.
- SwiGLU feed-forward layers.
- Repeated blocks with optional weight sharing.
- Heterogeneous block widths and attention head dimensions.

## Setup

Speck requires Python 3.10 or later and uses [uv](https://docs.astral.sh/uv/). Create either a GPU or CPU environment:

```bash
uv sync --extra gpu
# or
uv sync --extra cpu
```

Activation is optional when using `uv run`. For an interactive shell, use the command appropriate to your shell:

```bash
# POSIX shells
source .venv/bin/activate

# fish
source .venv/bin/activate.fish
```

## Experiments

Checked-in experiment directories identify an architecture and its data, tokenizer, and training configuration:

- `experiments/Speck1-140M` is the 140,652,288-parameter production and architecture-search configuration.
- `experiments/Speck1.5-140M` keeps that architecture, tokenizer, and 5B-token optimization recipe while using an isolated three-phase corpus curriculum.
- `experiments/Speck1-140M-Instruct` reuses that base architecture and tokenizer to post-train `Speck1-140M-Instruct` on SpeckChat1.
- `experiments/Speck1.1-140M-Instruct` reuses that base architecture and tokenizer to post-train `Speck1.1-140M-Instruct` on SpeckChat2 for one epoch.
- `experiments/Speck1.1-140M-Instruct-2ep` retains the corresponding two-epoch training run.

Model names follow `Speck<generation>-<size>`, with an optional decimal generation for intermediate families.

A search-capable experiment directory contains five JSON files:

```text
model.json      Architecture and dimensions.
tokenizer.json  Tokenizer artifact and local directory.
data.json       Sources, phased mixture, filters, dedup, shards, and packed output.
train.json      Optimization, batching, logging, and checkpoints.
search.json     Search space, training rungs, scoring, and profiling contract.
```

Artifacts use `~/.cache/speck` by default. Checkpoints are written to `~/.cache/speck/checkpoints/<train.run>`. Packed data uses an explicit `data.output_dir`, `~/.cache/speck/data/<data.output_name>` when a name is configured, or the legacy `~/.cache/speck/data/packed` default. Set `speck_base_dir` to move the cache root.

Despite its historical name, `train.json`'s `min_lr` is a multiplier of the peak `lr`, not an absolute learning rate. For example, `0.1` ends the schedule at 10% of the peak rate.

## Tokenizer

Download and verify the tokenizer configured by an experiment:

```bash
python -m scripts.tokenizer_prepare experiments/Speck1-140M
```

## Data

Resolve, stream, filter, deduplicate, tokenize, and pack the configured sources:

```bash
python -m scripts.data_prepare experiments/Speck1-140M
```

Prepare the isolated Speck1.5 corpus under `~/.cache/speck/data/Speck1.5-140M`:

```bash
python -m scripts.data_prepare experiments/Speck1.5-140M
```

The base pretraining experiment requests 5,000,000,000 training tokens. Its phase schedule is:

| Phase end | ultra_fineweb | dclm | cosmopedia_v2 | finemath_4plus | ultrafineweb_l3 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3,500,000,000 | 45% | 35% | 12% | 8% | 0% |
| 4,500,000,000 | 30% | 25% | 15% | 12% | 18% |
| 5,000,000,000 | 20% | 15% | 20% | 15% | 30% |

The phase durations and integer weights derive source targets of 1.975B, 1.55B, 670M, 475M, and 330M tokens respectively. Preparation adds a derived 262,144-token per-source loader reserve for the configured maximum 65,536-token distributed microbatch, then reports each requested target, reserve, and actual full-document result. Actual packed training data can exceed 5B only by these configured reserves and one final full-document overshoot per source.

The Speck1.5 corpus uses a foundation phase through 3.5B tokens, a capability ramp through
4.5B, and a final 500M-token capability anneal:

| Phase end | FineWeb-Edu | DCLM-Edu | Ultra-FineWeb | DCLM | FineMath | Textbook math | Multi-style math | Wikimedia | peS2o | UFW-L3 | Cosmopedia |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3.5B | 32% | 20% | 13% | 6% | 7% | 2% | 1% | 4% | 4% | 5% | 6% |
| 4.5B | 22% | 16% | 9% | 4% | 12% | 4% | 3% | 5% | 7% | 8% | 10% |
| 5.0B | 10% | 8% | 5% | 2% | 18% | 6% | 5% | 5% | 8% | 16% | 17% |

The phase durations derive these exact aggregate targets:

| Source | Tokens | Share |
| --- | ---: | ---: |
| FineWeb-Edu | 1.390B | 27.8% |
| DCLM-Edu | 900M | 18.0% |
| Ultra-FineWeb | 570M | 11.4% |
| DCLM | 260M | 5.2% |
| FineMath-4+ | 455M | 9.1% |
| UltraData-Math L3 Textbook-Exercise | 140M | 2.8% |
| UltraData-Math L3 Multi-Style | 90M | 1.8% |
| Wikimedia | 215M | 4.3% |
| peS2o | 250M | 5.0% |
| Ultra-FineWeb-L3 Multi-Style | 335M | 6.7% |
| Cosmopedia v2 | 395M | 7.9% |

The category totals are 62.4% natural web, 13.7% math, 9.3% knowledge/science, and 14.6%
general synthetic. Specialist sources appear first so their documents win global cross-source deduplication.
DCLM-Edu retains English rows with strict raw `edu_score > 3.5`; Ultra-FineWeb retains rows
scored at least 0.8; the two mixed-language UltraData-Math configurations retain rows identified
as English by pinned `py3langid==0.3.0`. peS2o uses only the V2 `s2orc` full-text training shards,
excluding its title-and-abstract `s2ag` records. All repository revisions and dataset paths are pinned
in `data.json`.

The completed stationary corpus, preparation log, and checkpoints are archived with `-old` suffixes.
The replacement corpus and checkpoint run both use the clean `Speck1.5-140M` artifact name in
their respective data and checkpoint directories.

Repository revisions are resolved once and pinned, and input discovery uses the Hugging Face repository tree rather than datasets-server previews. Files are deterministically shuffled per source. Preparation downloads and reads only one remote Parquet or gzip-JSONL file at a time, removes it immediately, and writes train and validation shards under `sources/<source-id>/`. Validation reserves 5M tokens per source and the loader schedules those streams equally.

Exact global deduplication normalizes text with Unicode NFKC, lowercasing, and whitespace collapse before recording a 128-bit BLAKE2 hash. The expected roughly 6M hashes remain practical in memory and are journaled compactly at 16 bytes each. A collision is treated as a duplicate; fuzzy and LSH deduplication are intentionally excluded. Tokenizer calls are bounded to 1,024 documents and 2,000,000 aggregate input characters.

Preparation performs a live disk-space preflight before creating staged data. The current estimate includes about 10.05GB of packed uint16 data, a 20GiB temporary raw-shard allowance, and at least 5GiB of dedup/index headroom, for about 36.9GB total required capacity. The command reports required and currently free bytes and credits reusable staged bytes on resume.

Preparation builds under the sibling `.building` directory and atomically publishes the final directory. Every completed remote source file closes and checkpoints packed shards, source-local index bytes, and the dedup journal with checksums. A retry validates those boundaries, removes only partial work from the interrupted file, and resumes at the next file. Pass `--restart` to discard all staged state. A completed output directory is never overwritten.

## Training

Authenticate with Weights & Biases, then start a single-GPU run:

```bash
wandb login
python -m scripts.base_train experiments/Speck1-140M
```

Use `experiments/Speck1.5-140M` to train against its isolated packed corpus with the same command structure.

Weights & Biases logging is enabled unless `train.run` is `dummy`. Checkpoints remain local; Speck does not upload training checkpoints to Hugging Face.

Launch distributed data-parallel training with `torchrun`:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  experiments/Speck1-140M
```

The configured 65,536-token optimizer batch is divisible by `device_batch_size * sequence_length * world_size` for world sizes 1, 2, 4, and 8. Since 5B is not batch-aligned, training performs 76,294 optimizer steps and consumes 5,000,003,584 tokens. Mixture phases are selected from each global microbatch's starting token position, so a microbatch that begins before a phase boundary remains in that phase even if it straddles the boundary.

Existing checkpoints are never resumed implicitly. A run fails rather than overwrite them unless an exact checkpoint step is supplied:

```bash
python -m scripts.base_train experiments/Speck1-140M --resume <checkpoint-step>
```

Resume validates the architecture, packed-data manifest, optimizer settings, batch geometry, training horizon, world size, and that the next-batch loader offset exactly equals completed optimizer-step tokens. It restores the optimizer, data position, elapsed time, and W&B run identity.

## Instruction Tuning

### SpeckChat1

Prepare the pinned `specklabs/SpeckChat1` dataset with the Speck chat template and assistant-only loss mask:

```bash
python -m scripts.sft_prepare experiments/Speck1-140M-Instruct
```

The current SpeckChat1 post-training configuration uses `<|system|>`, `<|user|>`, and `<|assistant|>` as token IDs 32000-32002, preserves the pretrained BOS/EOS tokens, and holds out 1,000 conversations for validation. Conversations are isolated in 256-, 512-, 1,024-, or 2,048-token buckets. The per-device batches are 32, 16, 8, and 4 respectively, so every microbatch has the same 8,192-token compute budget without unnecessary 2,048-token padding. Start one epoch of full-model instruction tuning from the pinned `specklabs/Speck1-140M` release:

```bash
python -m scripts.sft_train experiments/Speck1-140M-Instruct
```

Use `torchrun` as with base training for multiple GPUs. SFT checkpoints and a Hugging Face-compatible tokenizer are written under `~/.cache/speck/checkpoints/Speck1-140M-Instruct`. Resume only from an explicit SFT step with `--resume <checkpoint-step>`.

Generate from the instruction-tuned checkpoint by selecting its directory. The prompt is automatically rendered as a user message when the checkpoint metadata identifies SFT:

```bash
python -m scripts.infer "Explain why the sky is blue." \
  --checkpoint-dir ~/.cache/speck/checkpoints/Speck1-140M-Instruct
```

### SpeckChat2

Rebuild and publish the 500,000-row `specklabs/SpeckChat2` train split with pinned source revisions, source-specific quality filters, exact prompt deduplication, and Speck-tokenizer length checks:

```bash
uv run scripts/speckchat2_prepare.py
```

The mixture contains 200K LMSYS DeepSeek conversations, 130K Magpie Llama 3.1 multi-turn conversations, 85K Hermes, 65K UltraChat, 10K Magpie Reasoning, 8K No Robots, and 2K Everyday Conversations. It uses only source training splits and intentionally publishes no validation or test split. Use `--output-dir <path> --no-push` to build a local dataset instead.

The `experiments/Speck1.1-140M-Instruct` configuration pins the published 500,000-row SpeckChat2 dataset and the original `Speck1-140M` base weights. Prepare its isolated assistant-masked data, holding out 1,000 conversations for validation:

```bash
uv run --extra gpu python -m scripts.sft_prepare experiments/Speck1.1-140M-Instruct
```

Run one epoch of full-model post-training:

```bash
uv run --extra gpu python -m scripts.sft_train experiments/Speck1.1-140M-Instruct
```

Prepared data is written under `~/.cache/speck/data/SpeckChat2-v3`, and checkpoints are written under `~/.cache/speck/checkpoints/Speck1.1-140M-Instruct`.

Run the retained two-epoch variant against the same prepared data:

```bash
uv run --extra gpu python -m scripts.sft_train experiments/Speck1.1-140M-Instruct-2ep
```

Its checkpoints are written under `~/.cache/speck/checkpoints/Speck1.1-140M-Instruct-2ep`.

## Inference

Generate from the latest checkpoint, or select one with `--step`:

```bash
python -m scripts.infer "The meaning of life is" \
  --experiment experiments/Speck1-140M
```

Useful controls include `--max-tokens`, `--temperature`, `--top-k`, `--device`, and `--checkpoint-dir`.

### Transformers releases

Export, validate, and publish the canonical one-epoch instruction checkpoint as a BF16
Transformers repository:

```bash
uv run --extra cpu python -m scripts.model_publish --expected-epochs 1
```

Published likelihood evaluation supports binary right-padded batches when `use_cache=False`.
Left padding, mask gaps, and cached padded inference remain unsupported. Apply the same tracked
compatibility code to the existing base-model repository without changing its weights:

```bash
uv run --extra cpu --with transformers==5.1.0 python -m scripts.model_code_publish
```

The code-only publisher verifies the immutable source, model-weight LFS checksum, Auto class
loading, parameter count, padded-batch logit parity, uploaded code hashes, and unchanged remote
weights. Use `--no-upload` to run every local validation without creating a Hub commit.

### GGUF

Build BF16, Q4_K_M, Q5_K_M, and Q8_0 GGUF variants of the published instruction model,
smoke-test each file with a pinned llama.cpp checkout, and publish them to the Hugging Face Hub:

```bash
uv run --extra cpu python -m scripts.gguf_publish
```

Generated weights and the llama.cpp checkout are kept under `~/.cache/speck`, not in this
repository. Use `--no-upload` for a local build, repeat `--quantization <type>` to select a
different set, or pass `--llama-cpp <path>` to use an existing checkout. The publisher resolves
the source model to an immutable revision and uploads only after every requested file passes a
llama.cpp load and inference smoke test. Work is capped at four concurrent jobs by default; use
`--resume` after an interruption to validate and reuse completed files.

## Benchmarking

Run every benchmark in the pinned Open SLM Leaderboard configuration against the immutable
`specklabs/Speck1-140M` release:

```bash
uv run --extra gpu --group open-slm python -m scripts.open_slm_eval all
```

The configuration in `experiments/Speck1-140M/open_slm.json` pins the leaderboard, model,
lm-eval harness, standard-task datasets, and both official ArithMark repositories and file
checksums. Results default to `~/.cache/speck/evaluations/open-slm/Speck1-140M`. Use the
individual `lm-eval`, `arithmark-2`, `arithmark-3`, and `summary` stages to resume a run, or
pass `--limit 2` to `lm-eval` for a smoke test. ArithMark 2.0's verified official runner
right-pads without disabling the model cache; the wrapper leaves its scoring code unchanged
and sets `model.config.use_cache=False` immediately after model loading.

Pinned zero-shot results are recorded under `results/<model>/open_slm.json`:

| Model | HellaSwag | ARC-Easy | ARC-Challenge | PIQA | ArithMark-3 | Int Index | ArithMark-2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Speck1-140M | 35.03 | 46.68 | 25.94 | 63.87 | 36.60 | 18.15 | 31.52 |
| Speck1-140M-Instruct | 35.22 | 45.66 | 25.85 | 63.60 | 36.10 | 17.75 | 33.64 |
| Speck1.1-140M-Instruct | 35.64 | 46.93 | 26.02 | 64.15 | 33.70 | 17.90 | 32.44 |

Evaluate both public instruct releases with the same raw-continuation tasks and no chat
template:

```bash
uv run --extra gpu --group open-slm python -m scripts.open_slm_eval all \
  --config experiments/Speck1-140M-Instruct/open_slm.json
uv run --extra gpu --group open-slm python -m scripts.open_slm_eval all \
  --config experiments/Speck1.1-140M-Instruct/open_slm.json
```

The model-specific configs inherit the benchmark and dataset pins from the base evaluation
config. Their outputs default to separate directories named after each Hub repository.

Measure compiled optimization steps with synthetic input:

```bash
python -m scripts.benchmark experiments/Speck1-140M \
  --mode compute \
  --output benchmark.json
```

Use `--mode end-to-end --data-dir ~/.cache/speck/data/packed` to include packed-data loading. Warmup is reported separately. `--peak-tflops` reports model FLOPs utilization, and `--no-compile` measures eager execution.

Run the gated BananaMind Base Bench 1.1 continuation benchmark with its pinned official runner:

```bash
python -m scripts.bananamind_bench \
  --model experiments/Speck1-140M \
  --speck-checkpoint-step 76294 \
  --device cuda \
  --dtype bfloat16 \
  --batch-size 32
```

Accept the dataset gate and authenticate with Hugging Face first. The wrapper verifies the runner and data checksums, pins checkpoint and tokenizer hashes in the report, and rejects resume when the checkpoint or numerical configuration changes. It delegates scoring to the official runner and requires `transformers` in the execution environment.

Compare normalized prompt-prefill and cached-decoding inference speed with pinned model revisions:

```bash
python -m scripts.inference_benchmark --model speck --device cpu
python -m scripts.inference_benchmark --model speck --device cuda
```

Select `speck`, `supra`, `gptx`, `banana`, or `smol`. CPU defaults to FP32 batch 1; CUDA defaults to BF16 batches 1 and 32. The benchmark excludes tokenization, uses eager SDPA, returns only the final-position logit, and records raw synchronized timings. External models require `transformers`.

## Tests

```bash
uv run --extra cpu --group dev pytest -q
```

## Architecture Search

Search uses the baseline experiment and stores a resumable study under `~/.cache/speck/search/<name>`:

```bash
uv run --extra gpu python -m scripts.search run experiments/Speck1-140M \
  --name evolution-01 \
  --hours 3

uv run --extra gpu python -m scripts.search status evolution-01
uv run --extra gpu python -m scripts.search finalize evolution-01
```

Search requires a prepared packed dataset and a compatible CUDA runtime. See
[Architecture search](docs/search.md) for prerequisites, lifecycle, resume semantics, runtime
contracts, output files, and finalization requirements.
