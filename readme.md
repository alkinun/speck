# speck

speck is a compact research harness for training hybrid decoder language models.

## model

The model is defined as groups of residual blocks. Blocks contain ordered stages, and each stage can run one or more branches in parallel:

- global or sliding grouped-query attention
- gated causal convolution
- SwiGLU
- immediate repetition with optional weight sharing
- heterogeneous block widths and attention head dimensions

`experiments/speck00-200m/model.json` defines the checked-in 182,206,848-parameter model.

## setup

```bash
uv sync --extra gpu
source .venv/bin/activate
```

Use `--extra cpu` instead of `--extra gpu` for a CPU environment.

## tokenizer

```bash
python -m scripts.tokenizer_prepare experiments/speck00-200m
```

## data

```bash
python -m scripts.data_prepare experiments/speck00-200m
```

## training

```bash
wandb login
python -m scripts.base_train experiments/speck00-200m
```

Experiment configuration is split into four explicit JSON files:

```text
data.json       source, filters, splits, token budgets, and packed output
tokenizer.json  tokenizer artifact and local directory
model.json      block architecture and dimensions
train.json      optimization, batching, logging, and checkpoints
```

Artifacts default to `~/.cache/speck`. Set `speck_base_dir` to use another location.

For distributed training, copy the experiment and adjust `device_batch_size` and `batch_tokens` before launching:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- experiments/speck00-200m
```

Resume a checkpoint with:

```bash
python -m scripts.base_train experiments/speck00-200m --resume <checkpoint-step>
```

## inference

```bash
python -m scripts.infer "the meaning of life is" --experiment experiments/speck00-200m
```

## benchmarking

```bash
python -m scripts.benchmark experiments/speck00-200m \
  --mode compute \
  --output benchmark.json
```

Use `--mode end-to-end` with `--data-dir` to include packed-data loading. Warmup is reported separately; `--peak-tflops` adds model FLOPs utilization and `--no-compile` measures eager execution.

## checks

```bash
uv run --extra cpu --group dev pytest -q
```

Architecture search is intentionally absent while a smaller evolutionary design is specified.
