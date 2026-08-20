# speck

speck ships one model: `speck00-200m`, a 199,511,808-parameter decoder language model.

## model

- 12 blocks, width 1024, and swiglu width 4096
- mlp in every block and grouped-query attention in alternating blocks
- 16 query heads and 4 kv heads with head dimension 64
- qk-norm, rope, rmsnorm, and tied input/output embeddings
- 32k mistral tokenizer

| component | parameters |
| --- | ---: |
| tied token embedding | 32,768,000 |
| 12 swiglu mlps | 150,994,944 |
| 6 grouped-query attention layers | 15,728,640 |
| block rmsnorms | 18,432 |
| qk-norms | 768 |
| final rmsnorm | 1,024 |
| total | 199,511,808 |

## layout

```text
speck/    model, tokenizer, data, checkpoints, and runtime helpers
scripts/  tokenizer setup, data preparation, training, and inference
experiments/  self-contained experiment configurations
tests/    focused unit tests
```

artifacts are stored in `~/.cache/speck`. set `speck_base_dir` to use another location.

## setup

```bash
uv sync --extra gpu
source .venv/bin/activate
```

use `--extra cpu` instead of `--extra gpu` for a cpu environment.

## tokenizer

download and verify the tokenizer selected by an experiment:

```bash
python -m scripts.tokenizer_prepare experiments/speck00-200m
```

## data

stream ultra-fineweb and create local packed token shards:

```bash
python -m scripts.data_prepare experiments/speck00-200m
```

## training

```bash
wandb login
hf auth login

python -m scripts.base_train experiments/speck00-200m
```

the experiment directory is the unit of configuration:

```text
data.json       source, filters, splits, token budgets, and packed output
tokenizer.json  tokenizer artifact and local directory
model.json      architecture and dimensions
train.json      optimization, batching, logging, and checkpoints
```

copy the directory to start another experiment. json keeps each run explicit and diffable; `null` output directories use `~/.cache/speck`. the checked-in experiment uploads checkpoints to `specklabs/speck00-200m`; set `hf_repo` to an empty string to keep them local.

for distributed training:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- experiments/speck00-200m
```

## resume

```bash
python -m scripts.base_train experiments/speck00-200m --resume=30518
```

## inference

```bash
python -m scripts.infer "the meaning of life is" --experiment experiments/speck00-200m
```

## benchmarking

measure the production optimization step with synthetic tokens:

```bash
python -m scripts.benchmark experiments/speck00-200m \
  --mode compute \
  --output benchmarks/baseline-compute.json
```

include the packed data path in the measurement:

```bash
python -m scripts.benchmark experiments/speck00-200m \
  --mode end-to-end \
  --data-dir ~/.cache/speck/benchmark-200m \
  --output benchmarks/baseline-end-to-end.json
```

warmup is reported separately. use `--peak-tflops` to include model flops utilization and `--no-compile` to measure eager execution.

compare short real-data training runs:

```bash
python -m scripts.quality_benchmark --label baseline
```

## checks

```bash
uv run --extra cpu --group dev python -m pytest -q
uvx pyright
```

use `--device`, `--resume`, and `--no-compile` for runtime-only training overrides. everything that defines an experiment lives in its config directory.
