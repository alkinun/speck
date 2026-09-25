# Slurm operations

The runtime supports frozen one-node waves, one-GPU arrays or four-GPU jobs, checkpoint-boundary
stops, requeue, and accounting. Site account, partition, storage, and scheduler behavior still need
qualification. The bounded [hardware check](../experiments/qualification/README.md) is a separate
supervised command, not a qualified Slurm/requeue launcher.

```bash
uv run --no-sync python -m scripts.slurm_ops --help
uv run --no-sync python -m scripts.slurm_ops validate /shared/wave.json
uv run --no-sync python -m scripts.slurm_ops preflight /shared/wave.json
uv run --no-sync python -m scripts.slurm_ops render /shared/wave.json \
  --account CONFIRMED_ACCOUNT --partition CONFIRMED_PARTITION
```

A wave binds the exact clean Git revision, the hash of
[plan.json](../experiments/main-data/plan.json), experiment and data hashes, commands, resources
and retry limits. Build it only after the recipe and cost are measured. Every job names the
`budget_line` it charges (a key of `compute.budget_gpu_hours`, such as `pretraining_ladder`); only
`reserve` jobs may use the reserve. `speck_slurm_wave` v1 is validated in
[speck/operations/slurm.py](../speck/operations/slurm.py), whose budget constants `make plan-check`
checks against plan.json.

Training jobs use `torchrun -m scripts.slurm_base_train ... --slurm-requeue-resume`. Signals request
an optimizer-boundary checkpoint; requeue resumes the last complete checkpoint explicitly.
The rendered script exports a `TORCHINDUCTOR_CACHE_DIR` under `RUNTIME_ROOT/inductor/DIGEST`,
scoped to the wave so every requeued attempt reuses one compilation. The selected training
recipe compiles with max-autotune; without that cache each preemption pays full autotune again. SFT
requeue is not supported. Dependencies are restricted to collection/evaluation jobs, so training
promotion requires an inspected new wave.

After reviewing the rendered script, `submit` records the maximum commitment before `sbatch` and
records each returned job ID. `collect` reads scheduler observations and `summary` reports cost.
`retry` permits unchanged mechanical failures only. Reserve execution remains separate and is never
automatically submitted. Use the same ledger for the allocation, including failed attempts and
all allocated GPUs. Partial submissions require reconciliation before another attempt.

Older wave schemas are in [Git history](../README.md#history). New waves must
bind current code and actual inputs; a restored old wave does not become a current launch manifest.
