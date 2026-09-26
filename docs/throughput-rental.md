# H100 throughput rental

A rented 80 GiB H100 SXM, run before grant access, that measures what the plan now assumes: the
1.2B parent's and every ladder rung's throughput and utilization, and the `liger_aligned` loss
against `liger`. Its packet is
[`throughput-h100.json`](../experiments/qualification/throughput-h100.json). It spends no grant
GPU-hours and is charged to the external rental ledger, as the [H100 pilot](../experiments/pilot/h100-run.json)
was. It ran on 2026-09-26; the result is
[`throughput-h100/sweep.json`](../experiments/qualification/throughput-h100/sweep.json).

It does not replace the [GH200 qualification](compute-qualification.md). `device_batch_size`,
activation checkpointing and determinism are immutable on resume and are frozen on the grant
hardware, which has 96 GiB and more bandwidth. For this bandwidth-bound workload the GH200 rates
should be at or above the H100's, so these measurements are a conservative anchor for the budget.

## What the operator provides

1. One 80 GiB H100 SXM instance with about 200 GB of persistent storage. Record the instance ID,
   hourly rate, storage charges and intended deletion time.
2. SSH access.
3. A rental spend ceiling, so the stop conditions have a number to compare to.

The operator controls creation and deletion. Stopping a benchmark does not stop billing.

## Before renting

Prepare everything locally so no rented time is spent on setup. On a clean commit, with `TMPDIR` on
real disk:

```bash
make quality
make plan-check
uv run --no-sync python -m scripts.gh200_check bundle --output BUNDLE_DIR \
  --data PILOT_PACK_DIR --tokenizer TOKENIZER_DIR --assistant ASSISTANT_REHEARSAL_DIR
```

## On the instance

Transfer the bundle and its receipt and verify the SHA-256. Clone, install and bind as the
[one-worker check](compute-qualification.md#one-worker-check) does; the lock carries the x86_64
CUDA wheels. Then, from `code/`:

```bash
export TORCHINDUCTOR_CACHE_DIR=$PWD/../inductor-cache
export TRITON_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR/triton
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p ../results/throughput-h100
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv
```

Confirm the card is an 80 GiB H100 SXM before spending anything. Run the one-worker check first,
so the single-GPU path is known to work on this host:

```bash
uv run --no-sync python -m scripts.gh200_check run BUNDLE_DIR --allow-other-gpu \
  --output ../results/one-worker --seconds 5400
```

Print the sweep from the packet rather than retyping it, and run it in the packet's order:

```bash
uv run --no-sync python experiments/qualification/check_throughput_packet.py \
  experiments/qualification/throughput-h100.json --print-commands
```

- The first three parent runs isolate the pilot recipe, the checkpointing-off gain and the compile
  gain.
- `h100-mb4` and its four repeats set the noise band. Summarize them before reading any delta:

  ```bash
  uv run --no-sync python -m scripts.throughput_summary \
    ../results/throughput-h100/h100-mb4.json ../results/throughput-h100/h100-mb4-r*.json \
    --output ../results/throughput-h100/noise-summary.json
  ```

- `h100-mb4-aligned-r*` and each rung's `-aligned` run are the loss A/B.
- The three `best` runs take the microbatch and accumulation the microbatch runs selected. Substitute
  them in the copied commands, preserving 131,072 tokens per update. Keep the committed packet
  unchanged; editing it would mark later receipts dirty.

Record SM clock, temperature and power around every run.

## Stop conditions

Stop and diagnose rather than explore:

- Clock drift beyond 5% or unstable step times.
- Out-of-memory at a microbatch the card was expected to hold. Record the failing configuration and
  continue with the next rung.
- Rental spend reaching the declared ceiling. Preserve every receipt and stop rather than extending.

## What to bring back

Copy `../results/`, including the profile trace, and verify it by SHA-256 before deleting the
instance. Leave out the one-worker check's checkpoints and exports (about 83 GB; its receipt already
holds their parity comparison) and bring back a manifest of their names and sizes. Then record in the
repository:

- The parent's tokens/s and MFU at each microbatch, and its speedup over the pilot recipe on the same
  host.
- Tokens/s for each rung at its fastest microbatch, and the plan's projections recomputed from them.
- The `liger_aligned` delta against the noise band, at the parent and each rung.
- Memory headroom, warmup cost and graph-break diagnostics.

Verify the instance is actually deleted. Provider billing state is separate from local backups.
