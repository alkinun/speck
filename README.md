# Speck

SpeckLabs scales in steps, each much larger than the last. This first release is the **data step**:
its product is **measured, transferable findings about the pretraining and mid-training data
pipeline**, published as openly as the licences allow, so later and much larger releases can start
from them. A model ladder (50m, 130m and 410m) carries most experiments, and one **1.2B parent**
trained from scratch carries decay and mid-training; a fixed SFT recipe probes every branch. The
released base and light always-thinking assistant for **coding, math and tools** are the best
branches of those experiments. The KDA/NoPE-GQA architecture is fixed by declaration, so the release
makes no architecture claim, and post-training research belongs to later releases.

[PLAN.md](PLAN.md) gives status and the next work, the [program design](docs/program.md) the goal,
method, experiments and compute budget, the [paper outline](docs/paper.md) what the report must
show, and [plan.json](experiments/main-data/plan.json) the numbers.

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
Provide explicit experiment paths. Runtime data/checkpoints/logs live outside Git, under
`~/.cache/speck` by default; set `speck_base_dir` for another volume.

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

Retired plans, results, tools and records live only in Git history and do not govern current work.
Two tags mark the large removals: `pre-simplification-2026-09-17` (the earlier 140M releases and
research matrices) and `pre-cleanup-2026-09-25` (post-training code, the RL pilot, the R0
diagnostic, one-off corpus audits and research notes). Older `precleanup-*` and `archive/*` tags
keep earlier branch work. Read a file with
`git show TAG:PATH`, or check out a tag in a separate worktree with its own lockfile to rerun old
workflows.
