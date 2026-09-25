# Slurm operations

Scheduled training runs in immutable one-node waves of one-GPU arrays or four-GPU jobs, with
checkpoint-boundary stops, requeue and GPU-hour accounting. The site account, partition and requeue
behavior are confirmed by the scheduler canary in [GH200 qualification](compute-qualification.md).

```bash
uv run --no-sync python -m scripts.slurm_ops validate WAVE.json
uv run --no-sync python -m scripts.slurm_ops preflight WAVE.json
uv run --no-sync python -m scripts.slurm_ops render WAVE.json --account ACCOUNT --partition PARTITION
uv run --no-sync python -m scripts.slurm_ops submit WAVE.json --account ACCOUNT --partition PARTITION
uv run --no-sync python -m scripts.slurm_ops collect SUBMISSION.json
uv run --no-sync python -m scripts.slurm_ops summary
```

A wave (`speck_slurm_wave` v1, validated in [speck/operations/slurm.py](../speck/operations/slurm.py))
binds the clean Git revision, the hash of [plan.json](../experiments/main-data/plan.json), experiment
and data hashes, commands, resources and retry limits. Every job names the `budget_line` it charges,
a key of `compute.budget_gpu_hours` such as `pretraining_ladder`; only `reserve` jobs may use the
reserve. `make plan-check` checks the module's budget constants against plan.json.

Training jobs run `torchrun -m scripts.slurm_base_train ... --slurm-requeue-resume`. The timeout
signal requests a checkpoint at the next optimizer boundary, and a requeued job resumes from the
last complete checkpoint. Each wave gets its own `TORCHINDUCTOR_CACHE_DIR`, so requeued attempts
reuse one compilation. SFT jobs are not requeued. Job dependencies are limited to collection and
evaluation jobs.

`submit` records the wave's maximum GPU-hour commitment before calling `sbatch`, and each returned
job ID. `collect` records scheduler observations, `summary` reports cost, and `retry` resubmits a
job only after a mechanical failure. Reserve jobs are never submitted automatically. All attempts,
including failures, are charged to one ledger.
