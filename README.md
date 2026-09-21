# Speck

SpeckLabs' first model and paper focus on **data and training across pretraining, mid-training,
and post-training**. We are training a **1.2B all-active model from scratch**, with a base release
and an always-thinking assistant for **agentic coding, coding, math and tools**. The deliverable is
an open data pipeline covering all six training stages; the model proves the pipeline runs end to
end. The KDA/GQA reference is the **fixed substrate for this release, frozen by declaration rather
than selected over a control**: the bounded architecture/efficiency comparison was deferred on
2026-09-21 to a later allocation, so no architecture superiority or parity claim is available here.
The paper measures training and inference cost as reference-only evidence alongside the data
findings; architecture research, including MoE and attention residuals, belongs to later releases
with larger allocations.

Start with the [program overview](docs/program.md) for the training lifecycle, data, compute and
release, and the [model notes](docs/model.md) for the reference backbone and the deferred
architecture study. [PLAN.md](PLAN.md) gives current status and the next work;
the [main data plan](experiments/main-data/README.md) records numeric targets and feasibility.
The H100 pilot and backups are complete; GH200/four-worker qualification and flagship training are ahead.
[Research notes](docs/research.md) distinguish publisher findings from our own evidence. The
[readiness summary](PLAN.md#readiness-for-experiment-design) identifies what we can design now and
what still needs qualification before experiments launch.

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
The ordered access procedure is the [GH200 qualification runbook](docs/compute-qualification.md).

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
experiments/   Runnable configurations, clearly labeled preparation plans, and result receipts
docs/          Program outline, operational guides, and research notes
archive/       Pointer to the complete historical Git snapshot
```

Earlier plans, papers, results, and retired tools are recoverable through the
[history guide](archive/README.md). They do not govern current experiments.

Source code is [MIT licensed](LICENSE). Planned model weights use Apache-2.0. Corpus text and packed
training shards are not release artifacts. See the [citation](CITATION.cff).
