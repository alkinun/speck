# Experiments and evidence

The first release is the data step: a model ladder and one 1.2B parent measure the data pipeline of
each training stage on one fixed architecture family. The table distinguishes runnable work and
completed evidence from preparation plans. Architecture studies belong to later programs.

| Directory | Role | Current status |
| --- | --- | --- |
| [ladder](ladder/shapes.py) | 50m/130m/410m rung configurations generated from one shape rule | Configurations written; no rung trained |
| [qualification](qualification/README.md) | Exact selected 1.2B model and finite hardware checks | Single-H100 rehearsal/timing complete; GH200 and four-worker qualification ahead |
| [pilot](pilot/README.md) | Frozen 104,857,600-token engineering run | Training, export, 2,619 development scores and backups complete; weak base capability |
| [rl-pilot](rl-pilot/README.md) | Stage-5 RL from a math SFT of the pilot base, on the RTX 3090 | Engineering pilot; no grant hours, no data admission |
| [corpus-audit](corpus-audit/README.md) | Source-quality, provenance and eligibility evidence | Bounded audits complete; larger qualified supply and independent code checks still needed |
| [main-data](main-data/README.md) | Parent mixture, supply and numeric plan | Preparation plan, not a runnable training launch or admitted corpus |

`make smoke` creates a tiny offline base-to-assistant experiment in a temporary directory.
Use its `--output-dir` option to inspect configs, shards, checkpoints and reports. It does not
qualify full-size CUDA, distributed runtime or model quality.

Follow [PLAN.md](../PLAN.md) for work order and the [program design](../docs/program.md) for the
goal, experiments and budget. Actual launches bind exact configs, clean code, input manifests and
cost limits. Earlier 140M releases and research matrices remain in [history](../archive/README.md).
