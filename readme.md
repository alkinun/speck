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

- `experiments/speck00-200m` is the 182,206,848-parameter baseline and architecture-search configuration.
- `experiments/speck00-50m` is the 48,769,856-parameter efficiency finalist.
- `experiments/speck00-160m` is the 156,984,832-parameter balanced finalist.

A search-capable experiment directory contains five JSON files:

```text
model.json      Architecture and dimensions.
tokenizer.json  Tokenizer artifact and local directory.
data.json       Source, filters, token budgets, splits, and packed output.
train.json      Optimization, batching, logging, and checkpoints.
search.json     Search space, training rungs, scoring, and profiling contract.
```

Finalist directories intentionally omit `search.json`; they are complete configurations for ordinary preparation, training, benchmarking, and inference.

Artifacts use `~/.cache/speck` by default. Checkpoints are written to `~/.cache/speck/checkpoints/<train.run>`. The current packed-data implementation writes to `~/.cache/speck/ultra_fineweb/packed` when `data.output_dir` is `null`; set `data.output_dir` to `~/.cache/speck/packed` if a top-level packed-data directory is required. Set `speck_base_dir` to move the cache root.

Despite its historical name, `train.json`'s `min_lr` is a multiplier of the peak `lr`, not an absolute learning rate. For example, `0.1` ends the schedule at 10% of the peak rate.

## Tokenizer

Download and verify the tokenizer configured by an experiment:

```bash
python -m scripts.tokenizer_prepare experiments/speck00-50m
```

## Data

Download, filter, tokenize, and pack the configured dataset:

```bash
python -m scripts.data_prepare experiments/speck00-50m
```

Preparation writes train and validation shards plus a manifest. If an incomplete `.building` directory exists, pass `--restart` to replace that partial build. A completed output directory is not overwritten.

## Training

Authenticate with Weights & Biases, then start a single-GPU run:

```bash
wandb login
python -m scripts.base_train experiments/speck00-50m
```

Weights & Biases logging is enabled unless `train.run` is `dummy`. Checkpoints remain local; Speck does not upload training checkpoints to Hugging Face.

Launch distributed data-parallel training with `torchrun`:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  experiments/speck00-50m
```

`train.batch_tokens` must be divisible by `device_batch_size * sequence_length * world_size`. Adjust the training configuration for the available devices before launching.

Existing checkpoints are never resumed implicitly. A run fails rather than overwrite them unless an exact checkpoint step is supplied:

```bash
python -m scripts.base_train experiments/speck00-50m --resume <checkpoint-step>
```

Resume validates the architecture, packed-data manifest, optimizer settings, batch geometry, training horizon, and world size. It restores the optimizer, data position, elapsed time, and W&B run identity.

## Inference

Generate from the latest checkpoint, or select one with `--step`:

```bash
python -m scripts.infer "The meaning of life is" \
  --experiment experiments/speck00-50m
```

Useful controls include `--max-tokens`, `--temperature`, `--top-k`, `--device`, and `--checkpoint-dir`.

## Benchmarking

Measure compiled optimization steps with synthetic input:

```bash
python -m scripts.benchmark experiments/speck00-50m \
  --mode compute \
  --output benchmark.json
```

Use `--mode end-to-end --data-dir ~/.cache/speck/ultra_fineweb/packed` to include packed-data loading. Warmup is reported separately. `--peak-tflops` reports model FLOPs utilization, and `--no-compile` measures eager execution.

## Tests

```bash
uv run --extra cpu --group dev pytest -q
```

## Architecture Search

Search uses the baseline experiment and stores a resumable study under `~/.cache/speck/search/<name>`:

```bash
python -m scripts.search run experiments/speck00-200m \
  --name evolution-01 \
  --hours 3

python -m scripts.search status evolution-01
python -m scripts.search finalize evolution-01
```

See [Architecture search](docs/search.md) for prerequisites, runtime contracts, promotion behavior, output files, and finalization.
