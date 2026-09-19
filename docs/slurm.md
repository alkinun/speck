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

A wave binds the exact clean Git revision, experiment/data hashes, commands, resources, and retry
limits. Build it only after pilot settings and cost are chosen. `speck_slurm_wave` v1 is validated in
[speck/operations/slurm.py](../speck/operations/slurm.py); tests contain minimal complete fixtures.
The bound execution budget uses 5,000 total hours: 4,600 scheduled and 400 protected for
evaluation/recovery, matching the [current plan](../experiments/main-data/plan.json). Phase names come from that
plan; a phase's `conditional` field marks reserve, and zero-hour phases cannot launch compute.
Historical P1–P8 labels are not required. Old 4,111/889 wave plans must be replayed with their
original checkout; they cannot authorize current launches. Planning reservations still need a
frozen execution manifest and per-stage cost checks before submission.

Training jobs use `torchrun -m scripts.slurm_base_train ... --slurm-requeue-resume`. Signals request
an optimizer-boundary checkpoint; requeue resumes the last complete checkpoint explicitly. SFT
requeue is not supported. Dependencies are restricted to collection/evaluation jobs, so training
promotion requires an inspected new wave.

After reviewing the rendered script, `submit` records the maximum commitment before `sbatch` and
records each returned job ID. `collect` reads scheduler observations and `summary` reports cost.
`retry` permits unchanged mechanical failures only. Reserve execution remains separate and is never
automatically submitted. Use the same ledger for the allocation, including failed attempts and
all allocated GPUs. Partial submissions require reconciliation before another attempt.

Full command/schema history remains in the [frozen checkout](../archive/README.md). New waves must
bind current code and actual inputs; a restored old wave does not become a current launch manifest.
