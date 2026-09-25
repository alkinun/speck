# Speck

SpeckLabs scales in steps, each much larger than the last. This first release is the **data step**:
its product is **measured, transferable findings about the pretraining and mid-training data
pipeline**, published as openly as the licences allow, so later and much larger releases can start
from them. A model ladder (50m, 130m and 410m parameters) carries most experiments, and one
**1.2B parent** model, trained from scratch, carries the decay and mid-training experiments as
branches. A fixed SFT recipe probes every branch. The released base model and light reasoning
assistant for **coding, math and tools** are the best of those branches. The architecture, a hybrid
of Kimi Delta Attention (KDA) and global attention layers, is [held fixed](docs/program.md#model).

- [PLAN.md](PLAN.md): status and the next work
- [Program design](docs/program.md): goal, method, experiments and compute budget
- [Paper outline](docs/paper.md): what the report must show
- [plan.json](experiments/main-data/plan.json): the numbers

## Get a working baseline

Python 3.10+ and uv are required. From this checkout:

```bash
make setup
make smoke
make quality
```

The smoke workflow builds tiny local data, trains a hybrid base model, branches it onto masked chat
rows, trains an assistant from it, and verifies exact resume at each stage. It also evaluates
held-out base loss. It uses CPU only and downloads no corpus.

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
Provide explicit experiment paths. Data, checkpoints and logs live outside Git in the data store,
`~/.cache/speck` by default; set `speck_base_dir` to use another location.

```text
PLAN.md        One current direction and next step
speck/         Model, data, training, evaluation, export, and runtime code
scripts/       Maintained command entry points
tests/         Behavioral and integration checks
experiments/   Runnable configurations, design records and result receipts
docs/          Program design, paper outline and operational guides
```

Source code is [MIT licensed](LICENSE) and released weights are Apache-2.0; see
[Releasing](docs/releasing.md#licences). See the [citation](CITATION.cff).

## History

Retired code and records remain in Git history and do not govern current work. The tag
`pre-cleanup-2026-09-25` marks the last large removal and `pre-simplification-2026-09-17` the earlier
140M releases. Read an old file with `git show TAG:PATH`.
