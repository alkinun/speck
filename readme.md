# speck

## layout

```text
speck/    model, tokenizer, data, checkpoints, and runtime helpers
scripts/  tokenizer setup, data preparation, training, and inference
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

download and verify the pinned mistral tokenizer:

```bash
python -m scripts.tokenizer_prepare
```

## data

stream ultra-fineweb and create local packed token shards:

```bash
python -m scripts.data_prepare \
  --train-tokens=10000524288 \
  --validation-tokens=20000000 \
  --min-score=0.8 \
  --seed=42
```

## training

```bash
wandb login
hf auth login

python -m scripts.base_train \
  --run=speck-50m-10b \
  --device-batch-size=16 \
  --sequence-length=2048 \
  --train-tokens=10000000000 \
  --batch-tokens=524288 \
  --hf-repo=owner/repo
```

omit `--hf-repo` to keep checkpoints local. use `--hf-upload-optimizer` to include optimizer state in hf commits.

for distributed training:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --run=speck-50m-10b \
  --device-batch-size=4 \
  --hf-repo=owner/repo
```

## resume

```bash
python -m scripts.base_train \
  --run=speck-50m-10b \
  --device-batch-size=16 \
  --hf-repo=owner/repo \
  --resume=1907
```

## inference

```bash
python -m scripts.infer "the meaning of life is"
```

## checks

```bash
uv run --extra cpu --group dev python -m pytest -q
uvx pyright
```

use `python -m scripts.base_train --help` and `python -m scripts.data_prepare --help` for all options.
