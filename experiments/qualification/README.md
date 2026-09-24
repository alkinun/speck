# First hardware qualification

The exact 1.2B KDA/GQA reference at 4K is in [model.json](model.json); [plan.json](plan.json) holds
the finite settings of the synthetic R0 recovery diagnostic, which keeps its own 70-GPU-hour
ceiling inside the program's runtime-qualification line. The
[GH200 access qualification runbook](../../docs/compute-qualification.md) owns the full access
sequence. The model has 32,003 embedding rows; synthetic inputs use the original 32,000-token
vocabulary. Deterministic algorithms, including attention backward, and a reproducible cuBLAS
workspace stay enabled across initial and restarted workers; the restart tolerance is not relaxed.

Bind the configuration and source hashes without launching workers, then run on the allocated node:

```bash
uv run --no-sync python -m scripts.r0_execute experiments/qualification/plan.json \
  --workers 1 --allocated-gpus 1
uv run --no-sync python -m scripts.r0_execute experiments/qualification/plan.json \
  --workers 1 --allocated-gpus 1 --run \
  --ledger /shared/speck/qualification-ledger --prior-r0-gpu-hours ACCOUNTED_HOURS
```

Declare every allocated GPU. Inspect the one-worker result before `--workers 4`. The command
reserves both worker generations against the ceiling, enforces timeouts and a 128GiB free-space
floor, preserves failures and refuses unresolved prior attempts. Fresh workers warm the backend,
restore model, optimizer, cursor and RNG, and compare the next step. The probe does not establish
model quality, sustained throughput, hard-crash recovery or scheduler behavior.

## What is qualified

| Check | Result | Receipt |
| --- | --- | --- |
| RTX 3090, one worker | Full-size optimization and fresh-process restart at the original tolerance; about 1,937 synthetic tokens/s | [local-result.json](local-result.json) |
| One H100 | Real-data base and assistant recovery, numerical/kernel checks, native generation, CPU export parity | [h100-result.json](h100-result.json) |
| One H100 timing | Components below; evaluator defects fixed and re-exported | [timing-result.json](timing-result.json) |

| Measurement on one H100 SXM | Result |
| --- | --- |
| Base, batch 131,072, microbatch 1, 4K | 13,615 tokens/s; 28 steady steps; 19.3 GiB peak allocated |
| Same token batch, microbatch 4 | 16,398 tokens/s; six steady steps; not restart-qualified |
| Full 786,432-token validation | 15.9 seconds initially, 14.2 seconds warm |
| Durable model + optimizer checkpoint | 10.7–11.0 seconds; 9.16 GiB |
| Fresh-process restart overhead | 16.9 seconds beyond optimizer/validation/save work |
| SFT at fixed 4K | 13,493 padded positions/s; about 3,110 supervised tokens/s at this sample's density |
| Native cached decode, 1K prefix + 256 tokens | 55 tokens/s at batch 1; 428 aggregate tokens/s at batch 8 |

[timing.json](timing.json), [microbatch-timing.json](microbatch-timing.json),
[sft-timing.json](sft-timing.json) and [evaluation-timing.json](evaluation-timing.json) hold the
probe configurations, measured with `scripts.training_timing`, which observed the real trainer and
is now kept only in Git history. The SFT rate depends on its 23% supervised density, so
final SFT cost needs actual corpus lengths.

## What is open

ARM64 GH200 execution, four-worker communication and restart, compiled DDP, sustained throughput
with checkpoint and validation overhead, and Slurm requeue all remain to be qualified on the
allocation, as do per-rung rates for the [ladder](../../docs/program.md#the-ladder). Historical
projections in these receipts do not describe the current [budget](../../docs/program.md#compute)
or [supply](../../PLAN.md#supply).
