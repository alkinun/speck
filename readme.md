# speck

speck ships one main model configuration: `speck00-200m`, a 182,206,848-parameter decoder language model selected by architecture search.

## model

- 11 blocks with swiglu width 4096
- hidden width 1024 except for width 960 at block 7
- grouped-query attention at blocks 3, 5, 7, and 9
- kv head counts 4, 4, 1, and 4 respectively, with head dimension 64
- qk-norm, rope, rmsnorm, and tied input/output embeddings
- 32k mistral tokenizer

block indices are zero-based. `experiments/speck00-200m/model.json` is the exact architecture definition.

## architecture selection

the completed `speck00-search-v2` study evaluated 128 architectures and 216 trials without failures. these are the most useful final-rung tradeoffs measured on an rtx 3090:

| id | role | validation nll | parameters | 4-bit size | kv bytes/token | prefill 2048 | decode 2048 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 21 | best quality | 5.7359 | 199.5m | 98.1 mib | 6144 | 20.29 ms | 2.835 ms |
| 62 | quality/size | 5.7443 | 184.8m | 90.9 mib | 6144 | 18.53 ms | 2.578 ms |
| **86** | **selected main model** | **5.7589** | **182.2m** | **89.6 mib** | **3328** | **17.70 ms** | **2.473 ms** |
| 104 | fastest compact | 5.8606 | 157.0m | 77.2 mib | 5120 | 15.53 ms | 2.201 ms |
| 126 | lowest balanced regret | 5.8637 | 159.4m | 78.4 mib | 3840 | 16.25 ms | 2.324 ms |

architecture 86 is selected because its quality estimate overlaps the top quality tier while materially reducing parameter count, latency, peak memory, and kv cache use. full training remains the final validation of the selection.

## layout

```text
speck/    model, tokenizer, data, checkpoints, runtime, and search helpers
scripts/  tokenizer setup, data preparation, training, inference, and search
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
search.json     architecture space, fidelity rungs, objectives, and evolution
```

copy the directory to start another experiment. json keeps each run explicit and diffable; `null` output directories use `~/.cache/speck`. the checked-in experiment uploads checkpoints to `specklabs/speck00-200m`; set `hf_repo` to an empty string to keep them local.

for distributed training, copy the experiment and set `device_batch_size` and
`batch_tokens` for the target world size before launching:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- experiments/speck00-200m
```

## resume

```bash
python -m scripts.base_train experiments/speck00-200m --resume <checkpoint-step>
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
  --output benchmarks/selected-compute.json
```

include the packed data path in the measurement:

```bash
python -m scripts.benchmark experiments/speck00-200m \
  --mode end-to-end \
  --data-dir ~/.cache/speck/benchmark-200m \
  --output benchmarks/selected-end-to-end.json
```

warmup is reported separately. use `--peak-tflops` to include model flops utilization and `--no-compile` to measure eager execution.

compare short real-data training runs:

```bash
python -m scripts.quality_benchmark experiments/speck00-200m \
  --label architecture-86 \
  --output benchmarks/architecture-86-quality.json
```

the quality benchmark inherits the experiment's packed data path, physical batch size, optimizer batch, and recurring evaluation budget unless they are overridden.

## architecture search

version three search, its reliability contract, its differences from version two, and its operator workflow are documented in `docs/search-v3.md`. version two studies retain their original schema and semantics; version three studies use a separate configuration and study root.

run or resume the v3 calibration workflow, including its live dashboard, from bash, fish, or another terminal with:

```bash
./scripts/run_search_v3.sh --study speck00-v3-search
```

interrupting the launcher is safe; running the same command resumes committed study state. use `./scripts/run_search_v3.sh --help` for dashboard, experiment, configuration, and scheduling-rate options.

start a new study from the selected architecture:

```bash
python -m scripts.architecture_search run experiments/speck00-200m \
  --study speck00-search-next \
  --device cuda
```

inspect the study:

```bash
python -m scripts.architecture_search status speck00-search-v2
python -m scripts.architecture_search frontier speck00-search-v2
python -m scripts.search_dashboard speck00-search-v2
```

the same dashboard supports version three objective sets, token horizons, runs, actions, checkpoints, profiles, and posterior anchors:

```bash
PYTHONPATH=. uv run --extra gpu python -m scripts.search_dashboard calibration-v3 \
  --host 127.0.0.1 --port 8000
```

version one and two state is stored under `~/.cache/speck/search/<study>`; version three state is stored under `~/.cache/speck/search-v3/<study>`. changing comparison-sensitive search settings requires a new study name.

## checks

```bash
uv run --extra cpu --group dev python -m pytest -q
uvx pyright
```

use `--device`, `--resume`, and `--no-compile` for runtime-only training overrides. everything that defines an experiment lives in its config directory.
