# Training and inference

The reusable trainer lives in `speck.training.base`; command scripts remain thin entry points.
Install the CPU environment with `make setup`, or the CUDA/FLA environment with
`uv sync --extra gpu --extra linear`.

## CPU smoke workflow

```bash
uv run --no-sync python -m scripts.smoke
```

This generates a local tokenizer and tiny corpus, trains a small recurrent/attention model,
interrupts and resumes a second run, checks exact final-parameter equality, and evaluates held-out
loss. It needs no downloaded data, W&B login, or GPU. Use `--output-dir NEW_DIRECTORY` to retain
configs, shards, checkpoints, logs, and the evaluation report.

## Base training

Prepare the [data and tokenizer](data.md), then train a complete experiment:

```bash
uv run --no-sync python -m scripts.base_train experiments/Speck1-140M
```

The installed `speck train` command is equivalent. Use `--device cpu --no-compile` for appropriate
small CPU experiments. Training normally logs to W&B; the `dummy` run name uses the local null logger.
The flagship's actual launch experiments are tracked separately from this older release example.

For a four-GPU job outside Slurm:

```bash
uv run --no-sync torchrun --standalone --nproc-per-node=4 \
  -m scripts.base_train PATH_TO_EXPERIMENT
```

Allocated Slurm runs use the [Slurm wrapper and wave manifest](slurm.md).

## Resume and continuation

```bash
uv run --no-sync python -m scripts.base_train PATH_TO_EXPERIMENT --resume STEP
```

Resume is explicit. Completed checkpoints bind model configuration, optimizer state, data position,
tokenizer, schedules, and timing. Partial/requeue checkpoints remain distinct from completed release
artifacts. Publication recovery preserves a previous complete checkpoint if replacement fails.

Branching uses `--branch-from DIRECTORY --branch-step STEP`. A changed context/data stage uses an
explicit context branch and new schedule, prepared through `scripts.context_stage_prepare`.
Inspect `--help` for the full boundary; do not change a resume configuration to disguise a new run.
The [context guide](long_context.md) explains length and retention checks.

## Supervised fine-tuning

```bash
uv run --no-sync python -m scripts.sft_prepare experiments/Speck1-140M-Instruct
uv run --no-sync python -m scripts.sft_train experiments/Speck1-140M-Instruct
```

SFT uses assistant-masked data and a pinned parent checkpoint. `speck sft` is the installed entry
point. The flagship's proposed data and stages are in the [post-training protocol](../research/flagship/POST_TRAINING.md).

## Inference

```bash
uv run --no-sync python -m scripts.infer "The meaning of life is" \
  --experiment experiments/Speck1-140M --max-tokens 64
```

Use `--checkpoint-dir`, `--step`, `--device`, `--temperature`, and `--top-k` to select the checkpoint
and generation settings. `speck infer` is equivalent. Shared cached generation is implemented in
`speck.model.generation`.

The [archived guide](../archive/pregrant-history/docs/training.md) retains detailed older release,
SpeckChat, and continuation recipes at their original revision.
