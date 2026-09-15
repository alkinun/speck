# Speck

Speck is a research and training toolkit for small, efficient language models with recurrent
processing and periodic global attention. It includes deterministic data preparation, native and
distributed training, checkpoint recovery, long-context evaluation, and model export.

The current research program targets **efficient long-context intelligence**: a general-purpose
**1.2B KDA/GQA hybrid** within **5,000 GH200 GPU-hours**, with document/history reasoning, retained
general capability, and measured quality-cost comparisons. Start with the [research overview](research/README.md)
for the selected model, paper, budget and execution state.

## Setup

Python 3.10+ and [uv](https://docs.astral.sh/uv/) are required. Run commands from the checkout:

```bash
make setup
make quality
```

For CUDA training, use `uv sync --extra gpu --extra linear` (CUDA 12.8 PyTorch and optional FLA
kernels). The GH200/arm64 environment requires its own recorded hardware qualification.

## Use the toolkit

| Task | Guide |
| --- | --- |
| Prepare data and tokenizers | [Data](docs/data.md) |
| Train, resume, fine-tune, or generate | [Training](docs/training.md) |
| Evaluate quality, context, and performance | [Evaluation](docs/evaluation.md) |
| Operate distributed jobs | [Slurm](docs/slurm.md) |
| Export and publish checkpoints | [Releases](docs/releasing.md) |
| Develop the software | [Contributing](CONTRIBUTING.md) |

Command entry points use `python -m scripts.<command> --help`. The
[CPU smoke workflow](docs/training.md#cpu-smoke-workflow) exercises data preparation, training,
checkpoint resume, and evaluation without downloading a corpus.

Installation also provides `speck train`, `speck sft`, `speck infer`, `speck evaluate`,
`speck benchmark`, and `speck export`; each accepts `--help`.

Runtime datasets, checkpoints, exports, and logs live under `~/.cache/speck` by default.
Set `speck_base_dir` to select a different artifact volume.

## Repository map

```text
speck/         Reusable model, training, data, evaluation, and operations code
scripts/       Command entry points and experiment preparation tools
tests/         Software tests, evidence checks, and explicit integration checks
experiments/   Maintained examples and current runnable experiments
results/       Current result records
docs/          Operational guides
research/      Current program, conclusions, literature, and working context
paper/         Claim registry and manuscript assets
archive/       Preserved pre-grant research, results, and original-path inventory
```

## Research history

Completed experiments, negative results, source surveys, and predecessor plans remain in the
[physical archive](archive/README.md). Its manifest covers every file in the original checkout;
historical code is recoverable at the recorded Git revision.

```bash
python -m scripts.archive check
python -m scripts.archive restore /path/to/frozen-checkout
```

## License and citation

Source code is [MIT licensed](LICENSE). The flagship weight-release policy is Apache-2.0;
training corpus text and packed shards are not redistributed. See the
[release policy](research/flagship/release_and_data_use_policy_v1.json) and [citation](CITATION.cff).
