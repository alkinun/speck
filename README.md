# Speck

Speck is a small-language-model training project. We want a broadly useful assistant with particular
strength in **math, coding, tool use, and reliable instruction following**.

Start with [the plan](PLAN.md). It contains the current direction, status, and next experiment.
[Research notes](docs/research.md) explain what we take from Qwen, MiniCPM, and DeepSeek.

## Get a working baseline

Python 3.10+ and uv are required. From this checkout:

```bash
make setup
make smoke
make quality
```

The smoke workflow builds tiny local data, trains a hybrid base model, initializes an assistant from
its native checkpoint, and verifies exact resume for both stages. It also evaluates held-out base
loss. It uses CPU only and downloads no corpus.

The companion [technical report outline](docs/report.md) defines the evidence to collect for release.
The first GPU configuration is [experiments/qualification](experiments/qualification/README.md).
It checks the existing 1.2B KDA/GQA model at 4K before any corpus-training commitment.
GH200 fit, throughput, and cluster operation still need measurement.

## Working with the project

| Task | Guide |
| --- | --- |
| Prepare and reuse data | [Data](docs/data.md) |
| Train, resume, fine-tune, generate | [Training](docs/training.md) |
| Measure capability and cost | [Evaluation](docs/evaluation.md) |
| Run scheduler-managed training | [Slurm](docs/slurm.md) |
| Export a checkpoint | [Releasing](docs/releasing.md) |
| Make a change | [Contributing](CONTRIBUTING.md) |

Installed commands: `speck train`, `speck sft`, `speck infer`, `speck evaluate`, `speck benchmark`,
and `speck export`. Each accepts `--help`; preparation commands use `python -m scripts.<command>`.
Provide explicit experiment paths. Runtime data/checkpoints/logs live outside Git, under
`~/.cache/speck` by default; set `speck_base_dir` for another volume.

```text
PLAN.md        One current direction and next step
speck/         Model, data, training, evaluation, export, and runtime code
scripts/       Maintained command entry points
tests/         Behavioral and integration checks
experiments/   Concrete runnable configurations
docs/          Short operational guides and research notes
archive/       Pointer to the complete historical Git snapshot
```

Earlier plans, papers, results, and retired tools are recoverable through the
[history guide](archive/README.md). They do not govern current experiments.

Source code is [MIT licensed](LICENSE). Planned model weights use Apache-2.0. Corpus text and packed
training shards are not release artifacts. See the [citation](CITATION.cff).
