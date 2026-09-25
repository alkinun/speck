# Speck

SpeckLabs scales in steps, each much larger than the last. This first release is the **data step**:
its product is **measured, transferable findings about the pretraining and mid-training data
pipeline**, published as openly as the licences allow, so later and much larger releases can start
from them. A model ladder (50m, 130m and 410m) carries most experiments, and one **1.2B parent**
trained from scratch carries decay and mid-training; a fixed SFT recipe probes every branch. The
released base and light always-thinking assistant for **coding, math and tools** are the best
branches of those experiments. The KDA/NoPE-GQA architecture is fixed by declaration, so the release
makes no architecture claim, and post-training research belongs to later releases.

[PLAN.md](PLAN.md) gives status and the next work. The [program design](docs/program.md) owns the
goal, method, experiments and compute budget, and the [paper outline](docs/paper.md) what the report
must show. [plan.json](experiments/main-data/plan.json) owns the numbers.

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

The ladder configurations are in [experiments/ladder](experiments/ladder/README.md); GH200 access
follows the [qualification runbook](docs/compute-qualification.md).

## Working with the project

| Task | Guide |
| --- | --- |
| Prepare and reuse data | [Data](docs/data.md) |
| Train, resume, fine-tune, generate | [Training](docs/training.md) |
| Measure capability and cost | [Evaluation](docs/evaluation.md) |
| Qualify GH200 access | [Qualification](docs/compute-qualification.md) |
| Run scheduler-managed training | [Slurm](docs/slurm.md) |
| Export a checkpoint | [Releasing](docs/releasing.md) |
| Make a change | [Contributing](CONTRIBUTING.md) |

Every command is `python -m scripts.<command>` (for example `scripts.base_train`,
`scripts.sft_train`, `scripts.infer`) and accepts `--help`.
Provide explicit experiment paths. Runtime data/checkpoints/logs live outside Git, under
`~/.cache/speck` by default; set `speck_base_dir` for another volume.

```text
PLAN.md        One current direction and next step
speck/         Model, data, training, evaluation, export, and runtime code
scripts/       Maintained command entry points
tests/         Behavioral and integration checks
experiments/   Runnable configurations, design records and result receipts
docs/          Program design, paper outline and operational guides
archive/       Pointer to the complete historical Git snapshot
```

Earlier plans, papers, results, and retired tools are recoverable through the
[history guide](archive/README.md). They do not govern current experiments.

Source code is [MIT licensed](LICENSE); [Releasing](docs/releasing.md) covers weights and data. See
the [citation](CITATION.cff).
