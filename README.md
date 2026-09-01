# Speck

Speck is a compact research harness for designing, training, evaluating, and exporting small causal
language models. It supports heterogeneous residual blocks, phased data mixtures, resumable
training, full-model instruction tuning, architecture search, and reproducible evaluation tools.

## Capabilities

- Global or sliding grouped-query attention.
- Gated causal convolution.
- SwiGLU feed-forward layers.
- Repeated blocks with optional weight sharing.
- Heterogeneous block widths and attention head dimensions.
- Deterministic corpus preparation with filtering, exact deduplication, and packed shards.
- Single-GPU and distributed data-parallel training with explicit checkpoint resume.
- Local checkpoint inference, Transformers and GGUF export, and pinned benchmark wrappers.

Models are groups of residual blocks. Each block contains ordered stages, and a stage can execute
one or more branches in parallel.

## Requirements

Speck requires Python 3.10 or later and uses [uv](https://docs.astral.sh/uv/). Run commands from the
repository root.

Install a CPU environment for data preparation, tests, and CPU tooling:

```bash
uv sync --extra cpu
```

Install the CUDA 12.8 PyTorch environment for training and GPU evaluation:

```bash
uv sync --extra gpu
```

Activation is not required when commands use `uv run`. For an interactive shell, activate
`.venv/bin/activate` or the equivalent script for your shell.

## Core Workflow

Prepare the tokenizer and the configured 5B-token corpus:

```bash
uv run --extra cpu python -m scripts.tokenizer_prepare experiments/Speck1-140M
uv run --extra cpu python -m scripts.data_prepare experiments/Speck1-140M
```

Data preparation is a long-running network- and disk-intensive job. The current 5B-token recipe
preflights about 36.9GB of free space and can resume at validated source-file boundaries. Read the
[data preparation guide](docs/data.md) before starting it.

Train on one GPU:

```bash
wandb login
uv run --extra gpu python -m scripts.base_train experiments/Speck1-140M
```

Generate from the latest completed checkpoint:

```bash
uv run --extra gpu python -m scripts.infer "The meaning of life is" \
  --experiment experiments/Speck1-140M
```

See [Training and inference](docs/training.md) for distributed launch, checkpoint contracts,
instruction tuning, and generation options.

## Experiments

Checked-in experiment directories bind an architecture to its tokenizer, data, and training
configuration:

| Experiment | Purpose |
| --- | --- |
| `experiments/Speck1-140M` | 140,652,288-parameter base model, production recipe, and architecture-search baseline. |
| `experiments/Speck1.5-140M` | Same architecture and 5B-token optimization recipe with an isolated, pinned three-phase corpus curriculum. |
| `experiments/Speck2-140M` | Same architecture with a pinned 20B-token quality curriculum and scaled training schedule. |
| `experiments/Speck1-140M-Instruct` | One-epoch SpeckChat1 supervised fine-tuning of `Speck1-140M`. |
| `experiments/Speck1.1-140M-Instruct` | One-epoch SpeckChat2 supervised fine-tuning of the original base weights. |
| `experiments/Speck1.1-140M-Instruct-2ep` | Retained two-epoch SpeckChat2 variant. |
| `experiments/Speck1.5-140M-Instruct` | One-epoch SpeckChat2 supervised fine-tuning of the pinned Speck1.5 base. |
| `experiments/Speck2-140M-Instruct` | One-epoch SpeckChat2 supervised fine-tuning of the pinned Speck2 base. |

Model names follow `Speck<generation>-<size>`. Instruction-tuned variants append `-Instruct`, and
retained recipe variants may append an explicit suffix such as `-2ep`. Decimal generations identify
intermediate families.

A base experiment can contain:

```text
model.json      Architecture and dimensions.
tokenizer.json  Tokenizer source, revision, artifact filename, and prepared local directory.
data.json       Sources, phases, filters, deduplication, shards, and packed output.
train.json      Optimization, batching, logging, and checkpoints.
search.json     Search space, training rungs, scoring, and profiling contract.
open_slm.json   Pinned model-quality evaluation contract.
```

Instruction experiments replace `data.json` and `train.json` with `sft.json` when they consume a
prepared conversation dataset.

## Guides

| Guide | Contents |
| --- | --- |
| [Data preparation](docs/data.md) | Corpus mixtures, paths, filtering, deduplication, disk planning, and resume behavior. |
| [Training and inference](docs/training.md) | Base training, DDP, checkpoint resume, SFT, and local generation. |
| [Evaluation and benchmarking](docs/evaluation.md) | Open SLM, BananaMind, optimization, inference performance, and checked results. |
| [Architecture search](docs/search.md) | Prerequisites, lifecycle, promotion, reproducibility contracts, artifacts, and finalization. |
| [Releasing models](docs/releasing.md) | Maintainer-only Transformers, code-only, and GGUF publication workflows. |
| [Contributing](CONTRIBUTING.md) | Development setup, formatting, linting, tests, and change guidelines. |

## Storage

Runtime artifacts use `~/.cache/speck` by default:

```text
~/.cache/speck/
  checkpoints/   Base and SFT checkpoints.
  data/          Packed corpora and SFT datasets.
  evaluations/   Open SLM evaluation working directories and reports.
  benchmarks/    BananaMind and ad hoc benchmark reports.
  gguf/          Generated GGUF artifacts and conversion state.
  model-cards/   Generated model-card staging directories.
  releases/      Local Transformers exports.
  search/        Architecture-search studies.
  tokenizer/     Downloaded tokenizer artifacts.
  tools/         Pinned tool checkouts such as llama.cpp.
```

Set `speck_base_dir` before running a command to move the cache root. Explicit output paths in an
experiment or CLI take precedence.

## Development

Install the development group and run the local quality gate:

```bash
uv sync --extra cpu --group dev
uv run --extra cpu --group dev ruff format --check .
uv run --extra cpu --group dev ruff check .
uv run --extra cpu --group dev pytest -q
```

The test suite is CPU-only by default. CUDA behavior and remote publishing require targeted manual
or integration validation in addition to these checks.

## License

Speck is available under the [MIT License](LICENSE).
