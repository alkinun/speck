# Experiments and evidence

The first release is the data step: a model ladder and one 1.2B parent measure the data of
pretraining and mid-training, with a fixed SFT probe scoring each branch. The table distinguishes
runnable work and completed evidence from preparation plans.

| Directory | Role | Current status |
| --- | --- | --- |
| [ladder](ladder/README.md) | 50m/130m/410m rung configurations generated from one shape rule | 50m learning-rate sweep running |
| [qualification](qualification/README.md) | Exact selected 1.2B model and finite hardware checks | Single-H100 rehearsal/timing complete; GH200 and four-worker qualification ahead |
| [pilot](pilot/README.md) | Frozen engineering run of the 1.2B reference | Training, export, 2,619 development scores and backups complete; weak base capability |
| [rl-pilot](rl-pilot/README.md) | RL from a math SFT of the pilot base, on the RTX 3090 | Engineering pilot kept for a later step; not part of this release |
| [corpus-audit](corpus-audit/README.md) | Source-quality, provenance and eligibility evidence | Bounded audits complete; larger qualified supply and independent code checks still needed |
| [main-data](main-data/README.md) | Parent mixture, supply and numeric plan | Preparation plan, not a runnable training launch or admitted corpus |

`make smoke` creates a tiny offline base-to-assistant experiment in a temporary directory.
Use its `--output-dir` option to inspect configs, shards, checkpoints and reports. It does not
qualify full-size CUDA, distributed runtime or model quality.

Follow [PLAN.md](../PLAN.md) for work order and the [program design](../docs/program.md) for the
goal, experiments and budget. Actual launches bind exact configs, clean code, input manifests and
cost limits. Earlier 140M releases and research matrices remain in [history](../archive/README.md).
