# Training and inference

The [program design](program.md#experiments) defines the ladder, the parent, its decay and
mid-training branches and the SFT probe. This guide covers the implemented training paths.

`make setup` installs the CPU environment; CUDA uses `uv sync --extra gpu --extra linear`.

## Offline baseline

```bash
make smoke
uv run --no-sync python -m scripts.smoke --output-dir /tmp/speck-smoke-run   # keep the artifacts
```

Tiny local data, a hybrid base run with exact interrupted/resumed parameter equality, held-out
loss, a mid-training branch onto masked chat rows, and SFT with exact resume. CPU only, no downloads.

## Base training

An experiment directory supplies `model.json`, `tokenizer.json`, `data.json` and `train.json`
(configurations may `extends` another directory's):

```bash
uv run --no-sync python -m scripts.tokenizer_prepare PATH_TO_EXPERIMENT
uv run --no-sync python -m scripts.data_prepare PATH_TO_EXPERIMENT
uv run --no-sync python -m scripts.base_train PATH_TO_EXPERIMENT
uv run --no-sync torchrun --standalone --nproc-per-node=4 -m scripts.base_train PATH_TO_EXPERIMENT
```

Use `--device cpu --no-compile` for small CPU runs; run name `dummy` disables W&B. Resume with
`--resume STEP`; resume checks the model, data cursor, optimizer, tokenizer and schedule, and a
changed recipe needs a new run, not an edited resume. These commands enforce no budget; a real run
needs its own frozen recipe and cost ceiling. [Slurm](slurm.md) covers scheduler requeue.

Ladder runs are fresh runs. Arms within a comparison share initial model tensors; verify their
hashes before training.

**Immutable on resume:** microbatch, activation checkpointing and `deterministic`. They are frozen
per rung during [GH200 qualification](compute-qualification.md). `deterministic: true` enables
deterministic PyTorch algorithms and a reproducible cuBLAS workspace. Training and the benchmark
compile with `COMPILE_OPTIONS` from `speck/operations/runtime.py`. Keep `TORCHINDUCTOR_CACHE_DIR`
and its `triton/` subdirectory across requeues; runtime setup pins `TRITON_CACHE_DIR` under it,
because different cached FLA kernel choices change numerics.

Checkpoints include every rank's Python, NumPy, CPU and CUDA RNG state. `scripts.training_replay`
exercises the production trainer for four steps with a restart after two and compares every tensor
and the loader and RNG state; `--phase sft` does the same for a finite SFT recipe.

## Branches: decay and mid-training

Decay and mid-training runs are branches of preserved parent checkpoints:
`--branch-from DIRECTORY --branch-step STEP` with a branch kind.

- `--branch-kind same` keeps the parent's model and data manifest and inherits optimizer and data
  state.
- `--branch-kind data` allows a new manifest under `training_phase: data_continuation`, with the
  same architecture, sequence length, optimizer, schedule and world size. It resets only the data
  cursor and requires `--branch-schedule inherit`.
- `--branch-kind context` allows a new manifest and context length under
  `training_phase: context_extension`, resets the data cursor and keeps optimizer state. It is the
  only path that changes sequence length.

One packed manifest can declare token-endpoint mixture phases for a planned curriculum. A source can
be row-packed with `packing: {row_tokens, open_rows}`: whole records are placed best-fit into rows of
exactly the sequence length, never truncated, with a parallel loss mask. `record_format: messages`
supervises only assistant content. 16K and 32K memory and throughput still need GH200
qualification.

Before any branch, freeze the parent identity, objective, data, schedule, optimizer and cost.

## SFT probe

SFT needs its own `sft.json`, prepared assistant-masked data and an explicit parent:

```bash
uv run --no-sync python -m scripts.sft_prepare PATH_TO_SFT_EXPERIMENT
uv run --no-sync python -m scripts.sft_train PATH_TO_SFT_EXPERIMENT
uv run --no-sync python -m scripts.infer "Explain this result:" \
  --experiment PATH_TO_EXPERIMENT --checkpoint-dir CHECKPOINT_DIRECTORY --max-tokens 128
```

Bind a native base checkpoint as the parent with
`speck.export.pretrained.native_pretrained_source(directory, step)` and use the result as
`sft.json`'s `pretrained` setting. Changed parent bytes fail before loading. Local data uses
`dataset.format: "messages_v1"` or `prompt_completion_v1` with pinned Parquet files; overlength
conversations are rejected, never truncated. Hub datasets must set `long_sequences: "reject"`.

Chat format v2 supervises only `weight: 1` assistant turns and their EOS; `weight: 0` turns are
context. [Assistant data](assistant.md) defines the tool serialization and the stock the probe draws
from. Before freezing the probe dataset, audit the stock and gate outcomes:

```bash
uv run --no-sync python -m scripts.sft_audit /external/cache/generator-train-*.arrow \
  --tokenizer /external/tokenizer/tokenizer.model --lengths 4096 8192 16384 \
  --output /external/reports/sft-audit.json
uv run --no-sync python -m scripts.sft_verify /external/cache/sft-train.parquet \
  --output /external/reports/sft-outcomes.json
```

`sft_audit` counts every row and tokenizes a deterministic sample per subset, reporting complete-
conversation fit without truncation; pass conversation shards only, not `cache-*.arrow` shuffle
indices. `sft_verify` checks rows that declare `exact_text`, `code` or `tool` verification; the rest
stay `unverified`.
