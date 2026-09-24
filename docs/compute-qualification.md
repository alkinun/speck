# GH200 access qualification

This is the runbook for the runtime-qualification line of the 5,000-GPU-hour allocation (see
[the budget](program.md#compute)). It qualifies the hardware, software stack, distributed runtime
and recovery path, and measures throughput for the parent and every ladder rung. It trains no model
and selects no data.

The numeric authority is [`experiments/main-data/plan.json`](../experiments/main-data/plan.json).
The synthetic R0 diagnostic is bound by [`experiments/qualification/plan.json`](../experiments/qualification/plan.json),
whose GPU-hour ceiling limits R0 attempts inside the runtime-qualification line. The real-data
one-worker check and distributed canaries use their own frozen settings. Every phase is charged
once to the allocation ledger.

## Entry gate

Before reserving a GPU, complete the following on the current clean commit:

```bash
export TMPDIR=/path/to/a/large/disk   # the suite needs more scratch than a small tmpfs holds
make quality
make plan-check
uv run --no-sync python -m scripts.r0_execute experiments/qualification/plan.json \
  --workers 1 --allocated-gpus 1
```

The distributed and SQLite tests write large temporary files. On a host whose `/tmp` is a small
tmpfs they fail with `OSError: [Errno 122] Disk quota exceeded`, which looks like a code failure
and is not one; point `TMPDIR` at real disk first so the gate reports the truth.

The last command only binds a request. It must report `request_bound_no_execution`; it does not
allocate hardware. Build a fresh private bundle only after the working tree is clean:

```bash
uv run --no-sync python -m scripts.gh200_check bundle \
  --output /external/gh200-transfer \
  --data /mnt/speck-data/speck/data/flagship-pilot-105m \
  --tokenizer /mnt/speck-data/speck/tokenizer-final-mistral-v1 \
  --assistant /mnt/speck-data/speck/gh200-readiness-20260918/assistant-rehearsal-2
```

The bundle receipt, source commit, tokenizer, pilot data and assistant rehearsal are the exact
inputs for the real-data check. Do not copy a workstation virtual environment or credentials.
Transfer the archive and receipt, verify the archive SHA-256 after transfer, and extract them on
persistent storage with at least 128 GiB free for the runner and approximately 200 GB reserved for
the complete qualification workspace.

## Ordered access sequence

Run phases in this order. A failed phase stops the sequence; preserve its output and record the
failure before attempting any retry. Every phase records the actual device, source commit, input
hashes, wall time, allocated GPU count and allocated GPU-hours.

| Phase | Allocation | Command or action | Pass condition | Stop condition |
| --- | ---: | --- | --- | --- |
| Site preflight | 0 training hours | Record instance ID, allocation, architecture, GPU names, driver/CUDA, Python/uv, storage, network and scheduler. Verify the bundled commit and install the locked ARM64 GPU environment. | ARM64 GH200, expected dependencies, persistent output path, and free-space floor are confirmed. | Wrong hardware, missing dependency, insufficient storage, or unverified bundle. |
| R0 single-worker diagnostic | 1 declared GPU | Run `scripts.r0_execute` with `--workers 1 --allocated-gpus 1` and the shared R0 ledger. | Synthetic KDA forward/backward, checkpoint, fresh-process restart, loader/RNG and numerical checks pass. | Any worker failure, tensor/state mismatch, timeout, or unresolved prior ledger attempt. |
| Real-data single-worker check | 1 declared GPU | Bind the bundle, then run `scripts.gh200_check run` for the bounded 90-minute sequence. | Pilot loader scan/replay, kernels, 4K base restart, CUDA generation, base export parity, SFT restart and SFT export parity all pass. | Any phase failure. Do not proceed to distributed tests until the failed artifact is diagnosed. |
| R0 four-worker diagnostic | 4 declared GPUs | Repeat the R0 request with `--workers 4 --allocated-gpus 4`, using a new request identity and the same R0 ledger. | All four ranks pass the synthetic collective, checkpoint and restart checks with the declared tolerance. | Collective timeout, rank mismatch, communication failure, or R0 ceiling failure. |
| Four-worker training replay | 4 declared GPUs | Run the finite training replay with `scripts.training_replay`, `--workers 4 --allocated-gpus 4`, a four-step horizon and restart at step two. Run it **twice**: once eager, then once with `--compile`. | Real pilot data, optimizer/model parity, exact loader/RNG recovery and aggregate throughput are recorded in both modes. | Any rank failure, changed update geometry, or distributed parity failure. A compiled-only failure blocks the compiled recipe, not the allocation: fall back to eager and record the cost. |
| Throughput confirmation | 1 declared GPU | Run the ten bounded configurations in [`experiments/qualification/throughput-gh200.json`](../experiments/qualification/throughput-gh200.json) with `scripts.benchmark`, reusing one persistent Inductor cache. The packet lists 1.2B runs only; per-rung runs for each [ladder rung](program.md#the-ladder) are still to be added ([PLAN work order](../PLAN.md#work-order)). | Microbatch, activation checkpointing and determinism are selected from measurement, and tokens per second per allocated GPU is recorded for the parent and every rung. | Clock drift beyond 5%, unstable step times, or a graph-break count that differs from the Ampere result without explanation. |
| Scheduler canary | Site allocation | Bind one finite Slurm wave with current code and real input hashes. Exercise timeout-boundary checkpoint/requeue once. | The scheduler returns the job identity, the trainer checkpoints at the requested signal, resumes the last complete checkpoint, and accounting reports all allocated GPUs. | Missing account/partition, unsupported signal/requeue behavior, duplicate submission, or accounting mismatch. |
| Qualification closeout | 0 training hours | Download receipts, logs, manifests, failed artifacts and scheduler accounting. Reconcile the allocation ledger. | A signed local closeout identifies the usable worker count, measured throughput, recovery status, costs and remaining limits. | Missing receipt, unaccounted GPU-hours, or any unresolved phase result. |

The one-worker real-data command is intentionally separate from the synthetic R0 command. The
former checks the actual loader, export and assistant path; the latter isolates model/runtime and
distributed behavior with a small deterministic workload. Neither result establishes model quality,
long-context capability, sustained training throughput, or data eligibility.

For the real-data phase, extract the bundle on persistent storage, select the branch its manifest
records, and verify the checkout commit against `bundle.json` before binding:

```bash
cd /workspace/gh200-transfer
speck_bundle_branch=$(python3 -c 'import json; print(json.load(open("bundle.json"))["branch"])')
git clone --branch "$speck_bundle_branch" code.bundle code
cd code
# uv 0.12.1 (the preparation version), installed for ARM64, and Python 3.10.
uv sync --locked --python 3.10 --extra gpu --extra linear --group dev --group transformers
uv run --no-sync python -m scripts.gh200_check bind /workspace/gh200-transfer
uv run --no-sync python -m scripts.gh200_check run /workspace/gh200-transfer \
  --output /workspace/gh200-results/one-worker \
  --seconds 5400
```

Run the throughput packet from this `code/` checkout after binding. It uses `../relocated-base`
and writes to `../results/throughput-gh200`, keeping relocated inputs and outputs outside the
clean source tree. Substitute selected geometry in the printed commands, not in the tracked packet.

The bundle carries committed source and history, the frozen tokenizer, the verified 105M-token
pilot pack and a finite assistant rehearsal. It deliberately contains no virtual environments,
credentials, model downloads or local checkpoints. Do not publish the corpus payload, and do not
copy this workstation's x86 environments onto an ARM64 host. The lock pins ARM64 CUDA 12.8 PyTorch
2.9.1 and Triton 3.5.1 wheels; that they install and run on the rented image still needs
confirmation. Use a persistent terminal, retain the console log, and watch this first qualification.

`run` verifies the clean source commit and every input hash, records runtime/GPU identities, and
executes five steps under one shared 90-minute deadline:

1. Reopen the pilot pack, scan 64 loader batches, and replay saved batches in a fresh process.
2. Check KDA numerical behavior, gradients, recurrence/cache and timing at bounded lengths.
3. Train four full-size 4K base steps on real pilot data, restart from step two in a new process,
   and compare model/optimizer tensors plus exact loader/RNG state. Its diagnostic batch is one
   sequence; this is not the pilot optimization schedule.
4. Exercise native cached generation and export/tokenizer/Transformers parity for that checkpoint.
5. Initialize SFT from the identified base, train the finite assistant rehearsal in four optimizer
   steps, repeat the restart check, then check generation and export again.

Any failing phase stops the sequence and retains its logs, checkpoints and failed result. The
deadline covers supervised child execution with cleanup grace; hashing, comparison, installation and
transfer also consume rental time. It is not a provider billing cap or an instance-deletion
mechanism, and the reported GPU-hours cover this attempt only.

The command requires exactly one visible GPU. If the provider cannot expose one GPU separately,
pass `--allow-other-gpu` only for an explicitly recorded engineering exception and charge every
allocated idle GPU in the allocation ledger. That exception does not qualify single-GPU cost. An
H100 SXM/NVL/PCIe or H200 can run the same single-GPU sequence with that flag, using the host
architecture's dependencies; that qualifies neither ARM64, Grace memory/interconnect behavior nor
GH200 performance.

Rebuild the bundle from current committed code. The historical transfer archive recorded in the
[H100 timing receipt](../experiments/qualification/timing-result.json) predates the export and
evaluator fixes, so it is evidence, not a current release artifact. The completed
[pilot](../experiments/pilot/README.md) stays frozen, and no current bundle exists merely because
the build command is documented here.

For the distributed training replay, bind the current pilot configuration and run a finite
four-step check with a restart at step two:

```bash
uv run --no-sync python -m scripts.training_replay /workspace/gh200-transfer/relocated-base \
  --phase base --device cuda --workers 4 --allocated-gpus 4 \
  --steps 4 --checkpoint-step 2 --seconds 1800 \
  --output /workspace/gh200-results/four-worker-replay

uv run --no-sync python -m scripts.training_replay /workspace/gh200-transfer/relocated-base \
  --phase base --device cuda --workers 4 --allocated-gpus 4 --compile \
  --steps 4 --checkpoint-step 2 --seconds 3600 \
  --output /workspace/gh200-results/four-worker-replay-compiled
```

This is a runtime and recovery measurement. It is not a training continuation, a throughput
guarantee, or permission to extend the parent horizon. The second invocation is the only place the
selected compiled recipe meets four-worker DDP before the ladder and parent runs; allow the longer deadline for
max-autotune warmup and export a persistent `TORCHINDUCTOR_CACHE_DIR` first. Preserve its `triton/`
subdirectory too (or an explicitly configured `TRITON_CACHE_DIR`); it holds FLA kernel choices.

## Accounting and receipts

Keep two explicitly linked records:

1. the R0 ledger, which enforces the R0 ceiling of the synthetic diagnostic;
2. the program ledger's runtime-qualification line, which covers all site setup,
   R0 attempts, real-data checks, distributed canaries, scheduler work and allocated idle time.

An R0 attempt must be launched with an explicit prior value, for example:

```bash
uv run --no-sync python -m scripts.r0_execute experiments/qualification/plan.json \
  --workers 1 --allocated-gpus 1 --run \
  --ledger /shared/speck/qualification-ledger/r0 \
  --prior-r0-gpu-hours 0
```

The prior value is the already-accounted R0 allocation, not a guessed provider bill. The program
ledger separately records transfer, installation, storage, network, scheduler, idle and failed-work
costs. Never refund reserved headroom or charge one attempt to two ledgers.

Each phase must retain its JSON result, console log, command line, request/config hashes, source
commit, device identity, allocated GPU count, wall time and allocated GPU-hours. Preserve failed
attempts. A training process stopping does not stop provider billing or delete an instance.

## Qualification boundary

The throughput phase exists because the frozen pilot recipe is inefficient. The RTX 3090 proxy
result and its limits are in [performance](performance.md#evidence-and-limits); it is not the
parent's figure. This phase benchmarks `experiments/pilot`, the 1.2B reference, so it measures the
real one. Quote no parent speedup before it returns.

If the optional [H100 throughput rental](throughput-rental.md) has run first, this phase is a confirmation rather than a discovery: the expected ranking is known, and a
result that contradicts it is a reason to stop and diagnose rather than to explore inside a
line that cannot be re-spent. That rental does not shorten this phase or substitute for it.

Microbatch, activation checkpointing and determinism are immutable on resume, so they must be
frozen here, before any ladder or parent run starts. The measured rate replaces
`h100_full_trainer_tokens_per_second_1_2b` as the anchor for the parent horizon, and the per-rung
rates convert the ladder line into run counts. Apply `compute.measured_anchor.overhead_derate`
first, because this sweep runs `--mode compute` and the anchor it replaces is a full-trainer rate
that includes startup, validation and saves. What a faster or slower rate changes is predeclared in
`compute.rules`.

One further gap this phase cannot close: it is single-GPU, so it does not qualify the compiled path
under DistributedDataParallel. `base.py` compiles the DDP-wrapped module and six graph breaks
remain, which is where DDPOptimizer bucketing can interact badly. Run the four-worker training
replay with `--compile` before any compiled multi-worker wave.

A successful closeout permits costing the ladder, parent and branch lines from measured runtime
inputs. It does not admit data, prove 16K/32K usefulness, or authorize Slurm training waves; those
need their own frozen records and receipts.

The closeout must state separately:

- one-worker ARM64/software status;
- four-worker communication and restart status, eager and compiled separately;
- the measured parent speedup over the frozen pilot recipe, not inherited from the proxy sweep;
- tokens/s per allocated GPU for each ladder rung and the resulting run counts;
- scheduler/requeue status;
- sustained tokens/s with checkpoint and validation overhead;
- memory and storage observations;
- inference/export parity status;
- all-in allocated GPU-hours and non-GPU costs;
- unresolved limitations and the exact next qualification action.
