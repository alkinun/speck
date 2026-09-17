# First hardware qualification

This checks the existing 1.2B KDA/GQA model at 4K. [model.json](model.json) holds its exact geometry;
[plan.json](plan.json) holds finite execution settings. There is no catalog or predecessor-plan chain.
The model has 32,003 embedding rows; synthetic inputs use the original 32,000-token vocabulary.
The checked plan enables deterministic PyTorch algorithms, including deterministic attention
backward, and a reproducible cuBLAS workspace. A local CUDA diagnostic found that default attention
backward can vary across otherwise identical runs. Preserve this setting across the initial and
restarted workers; the strict restart tolerance has not been relaxed. Record its throughput cost.

Bind the configuration and current source hashes without allocating weights or launching workers:

```bash
uv run --no-sync python -m scripts.r0_execute experiments/qualification/plan.json \
  --workers 1 --allocated-gpus 4
```

On the actual allocated node, after checking environment, storage, and prior costs:

```bash
uv run --no-sync python -m scripts.r0_execute experiments/qualification/plan.json \
  --workers 1 --allocated-gpus 4 --run \
  --ledger /shared/speck/qualification-ledger --prior-r0-gpu-hours ACCOUNTED_HOURS
```

Declare every allocated GPU, including idle ones. `ACCOUNTED_HOURS` is a known nonnegative number.
Inspect the one-worker result before using `--workers 4`. The command requires CUDA for execution.
It reserves both worker generations against the 70-hour ceiling, enforces timeouts and the 128GiB
free-space floor, preserves failures, and refuses unresolved prior attempts.

Initial workers optimize synthetic shifted-token inputs, save model/optimizer/cursor/RNG state and
an uninterrupted reference, then exit. Fresh workers reload and compare the next step. Source hashes,
input hashes, memory, step timings, checkpoint costs, and failures remain in the attempt directory.
The learning rate and batch are diagnostic settings. The probe does not establish model quality,
sustained corpus throughput, hard-crash recovery, cached-generation parity, or Slurm/requeue behavior.
Those need separate checks before the real pilot. No 32K/128K sweep is part of this first step.
