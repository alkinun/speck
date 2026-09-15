# Bounded R0 executor

The [execution preparation plan](r0_execution_preparation_v2.json) binds the six
[checked shapes](../../results/systems/r0-shape-preparation-20260915.json). It prepares a finite
synthetic optimization/checkpoint diagnostic. It does not grant scientific model-launch authority
or certify full R0 readiness. Actual allocation/site access remains pending.

The [fresh-process local qualification](../../results/systems/r0-fresh-process-local-qualification-20260915.json)
at `2bd1676` passes 30 focused tests and binds 12 requests: hybrid/dense at 4K/32K/128K, each with
one/four workers. Tiny CPU dense and KDA models pass persisted checkpoint/RNG recovery in fresh
processes; two Gloo ranks also pass restart and final-weight agreement. CPU fixtures leave GPU fit
and four-GPU pass fields null. Full quality: 1,274 passed, 10 skipped, 125 deselected, plus format,
lint, catalog and archive checks. The evidence-bound v1 implementation is preserved in the
[successor snapshot](../history/2026-09-15-r0-fresh-process/manifest.json); its
[original qualification](../../results/systems/r0-executor-local-qualification-20260915.json) remains
valid at its recorded revision. Neither local receipt qualifies actual GPU execution.

## Prepare and inspect

This command validates identities and settings and prints a request fingerprint without execution:

```bash
uv run --no-sync python -m scripts.r0_execute \
  research/flagship/r0_execution_preparation_v2.json \
  --case hybrid-4096 --workers 1 --allocated-gpus 4
```

Inputs are the embedded checked model configuration, effective vocabulary 32,003 and synthetic
input vocabulary 32,000. Each CPU-generated microbatch contains sequence-length + 1 tokens;
input/target spans share the shifted positions. Seeds depend on rank and microbatch ordinal,
independently of model RNG. Record payload hashes, including both replay copies. Synthetic input
repetition does not select a production exposure policy or open any sealed data.

Default settings are batch one, accumulation one, two warmup steps, five measured steps, constant
diagnostic LR 1e-4, Muon plus AdamW, clipping 1.0, decay 0.1, activation checkpointing and Liger loss.
There is one additional uninterrupted step, followed by its replay in newly started workers. These
are engineering settings, not chosen R1/flagship optimizer settings. The outer model and Muon are eager by default; KDA still
uses FLA/Triton on CUDA. Global attention uses recorded SDPA automatic dispatch, not a claim that a
particular attention kernel was observed. The plan permits no automatic backend or shape fallback.
Changed settings require a preserved bound successor before comparing results.

The executor follows existing training precision: FP32 parameter/optimizer storage and BF16 CUDA
activations, with recurrent state rules inherited from the model. The shape report's analytic BF16
parameter-byte field is not an observation of this executor's parameter storage. Actual parameter
dtypes and allocator peaks are recorded; GPU memory fit remains unmeasured locally.

## At an allocated site

Review the node/dependencies, storage and budget accounting first. Start with the smallest context
and inspect its result before another case. Do not launch the six-case matrix automatically. The
shared ledger must cover all attempts in this R0 allocation. An example invocation, once site details
and the external-prior cost have been established, is:

```bash
uv run --no-sync python -m scripts.r0_execute \
  research/flagship/r0_execution_preparation_v2.json \
  --case hybrid-4096 --workers 1 --allocated-gpus 4 --run \
  --ledger /site/shared/speck/r0-ledger --prior-r0-gpu-hours ACCOUNTED_HOURS
```

`ACCOUNTED_HOURS` must be a known nonnegative number; no implicit zero is supplied. Declare all
allocated GPUs even when fewer workers are active. Public execution requires CUDA. The supervisor
starts each rank in its own tracked process group, with a unique file rendezvous for NCCL. Rank
identity and source/request hashes are rechecked at startup. Site scheduler/cgroup integration is
still needed; this launcher is not a qualified Slurm/requeue wrapper.

A v2 attempt has two worker generations, each with a 900-second wall deadline and ten-second
termination grace. It reserves both phases before launching: 2.0222 GPU-hours with four GPUs allocated.
An unsuccessful initial phase prevents the restart phase and retains the complete reservation.
The ledger serializes attempts and retains reservations for successful and unsuccessful work; it never silently refunds the difference between
a reservation and observed time. It refuses a reservation beyond the 70-hour envelope, changed prior
accounting, changed result receipts and unresolved prior attempts. The default free-disk floor is
128 GiB for per-rank model/optimizer checkpoints; it is a guard, not proof that a shared filesystem
will remain available. Keep checkpoints and logs outside the repository.

## Results, failure and recovery

Every attempt gets a new directory with its request, process IDs, shared log, durable rank progress,
rank results, per-rank checkpoint payloads and supervisor result. V2 stores these under separate
`initial/` and `restart/` directories; nested rank paths resolve within their named phase. The initial
workers must finish successfully and exit before the next generation starts. OOM, unsupported backend,
numerical/parity failures and other execution errors are recorded separately. Missing ranks or a nonzero exit
cannot produce a successful aggregate result. Timeouts/interrupts stop every tracked rank group;
partial rank reports remain available. A hard-killed supervisor leaves its reservation unresolved.
Check actual processes, scheduler state and process start identities before recovery; PIDs can be
reused. Preserve the original attempt and reconcile its cost before a new directory is launched.

Measured synthetic step throughput includes CPU input generation/transfer and optimization. It
excludes progress publication and checkpoint/replay work; those costs remain in worker/supervisor wall
time and separate checkpoint timings. The denominator uses the slowest rank's measured-step sum;
global processed tokens are counted once. Warmup includes cold kernel startup; report warmup separately
from measured steps. This short diagnostic cannot establish sustained production throughput.

Allocator peaks cover construction through replay, including checkpoint reload, and exclude host
memory and non-PyTorch CUDA allocations. Observed GPU-hours charge all declared allocated GPUs from
supervisor launch through termination. Scheduler startup, idle and cleanup outside that window and
other R0 work are not measured here: scheduler-total hours remain null until reconciled. Do not use
another ledger or a smaller declared allocation to bypass those costs.

Checkpoint checks compare next-step loss, model tensors and optimizer state against uninterrupted
execution at rtol 1e-5 / atol 1e-6. V2 persists a hash-bound baseline, data cursor, RNG payload and
uninterrupted reference, then loads them in new worker processes. It checks continuation of Torch CPU,
Torch CUDA when applicable, Python and NumPy RNG streams. A dedicated probe exercises those streams
even when the model has no stochastic layers. CPU qualification does not exercise CUDA RNG.

The original v1 protocol records same-process parity; v2 records process-restart parity and leaves
the same-process field null. A completed producer checkpoint and clean process exit are prerequisites
here. This is not a hard-crash, production-loader restart or scheduler-requeue qualification. Preserve
failures rather than loosen tolerances after seeing results.

## Remaining R0 qualification

Actual GH200/arm64 dependencies, NCCL and exact-shape execution; independent numerical-reference and
cached-generation parity; CUDA checkpoint/RNG restart, hard interruption and scheduler/requeue; production-loader
and sustained end-to-end throughput; and complete scheduler/all-attempt cost reconciliation remain
open. A bounded synthetic pass establishes none of the paper's quality or useful-context claims.
