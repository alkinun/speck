# Experiments and evidence

The first release studies data and pretraining, mid-training and post-training on one fixed model.
The table distinguishes runnable work and completed evidence from preparation plans. Broader
architecture studies belong to later programs; no architecture search is active here.

| Directory | Role | Current status |
| --- | --- | --- |
| [qualification](qualification/README.md) | Exact selected 1.2B model and finite hardware checks | Single-H100 rehearsal/timing complete; GH200 and four-worker qualification ahead |
| [pilot](pilot/README.md) | Frozen 104,857,600-token engineering run | Training, export, 2,619 development scores and backups complete; weak base capability |
| [corpus-audit](corpus-audit/README.md) | Source-quality, provenance and eligibility evidence | Bounded audits complete; larger qualified supply and independent code checks still needed |
| [main-data](main-data/README.md) | Working mixture, scale and compute targets | Preparation plan, not a runnable training launch or admitted corpus |

`make smoke` creates a tiny offline base-to-assistant experiment in a temporary directory.
Use its `--output-dir` option to inspect configs, shards, checkpoints and reports. It does not
qualify full-size CUDA, distributed runtime or model quality.

Follow [PLAN.md](../PLAN.md) for work order and the [program overview](../docs/program.md) for the
connected design. Actual launches bind exact configs, clean code, input manifests and cost limits.
Earlier 140M releases and research matrices remain in [history](../archive/README.md).
