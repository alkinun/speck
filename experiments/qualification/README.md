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
an uninterrupted reference, then exit. Fresh workers first initialize the backend with the declared
number of synthetic warmup steps, discard their optimizer state, then restore the saved model,
optimizer, cursor, and RNG before comparing the next step. This keeps first-use backend initialization
from consuming the restored Python RNG. Warmup costs are included in the attempt accounting. Source hashes,
input hashes, memory, step timings, checkpoint costs, and failures remain in the attempt directory.
The learning rate and batch are diagnostic settings. The probe does not establish model quality,
sustained corpus throughput, hard-crash recovery, cached-generation parity, or Slurm/requeue behavior.
Those need separate checks before the real pilot. No 32K/128K sweep is part of this first step.

## Local preparation result

The [retained RTX 3090 result](local-result.json) records a successful full-size, one-worker
optimization and fresh-process model/optimizer/RNG comparison at the original tolerance. It used
`PYTORCH_ALLOC_CONF=expandable_segments:True`, deterministic algorithms, and no compilation.
The earlier OOM, nondeterministic-gradient failure, and cold-backend RNG failure are preserved.
Synthetic step throughput is about 1,937 tokens/s on that machine; it is not corpus throughput or
a GH200 projection. The four supervised attempts consumed 0.585 local allocated GPU-hours;
standalone kernel/tests are outside that ledger. Repeat qualification on the actual allocation.
