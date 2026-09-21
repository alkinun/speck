# GH200 access qualification

This is the authoritative runbook for the first **120 GPU-hours** of the confirmed
5,000-GPU-hour allocation. It qualifies the hardware, software stack, distributed runtime and
recovery path. It does not train the flagship model, select a data source or authorize production.

The numeric authority is [`experiments/main-data/plan.json`](../experiments/main-data/plan.json).
The bounded synthetic diagnostic is [`experiments/qualification/plan.json`](../experiments/qualification/plan.json),
which retains its own 70-GPU-hour R0 ceiling. That ceiling is an internal limit for R0 attempts;
R0 work still consumes the program's 120-hour runtime-qualification reservation and must be recorded
once in the allocation ledger. That reservation grew from 100 hours when the architecture study was
deferred: 20 of its released hours fund reference-only training and inference efficiency profiling,
which that study used to own and the report still needs.

The real-data one-worker rehearsal and distributed canaries use their own frozen settings and the
same allocation ledger.

## Entry gate

Before reserving a GPU, complete the following on the current clean commit:

```bash
make quality
make plan-check
uv run --no-sync python -m scripts.r0_execute experiments/qualification/plan.json \
  --workers 1 --allocated-gpus 1
```

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
| R0 four-worker diagnostic | 4 declared GPUs | Repeat the R0 request with `--workers 4 --allocated-gpus 4`, using a new request identity and the same R0 ledger. | All four ranks pass the synthetic collective, checkpoint and restart checks with the declared tolerance. | Collective timeout, rank mismatch, communication failure, or budget reservation failure. |
| Four-worker production replay | 4 declared GPUs | Run the finite production-trainer replay with `scripts.training_replay`, `--workers 4 --allocated-gpus 4`, a four-step horizon and restart at step two. Run it **twice**: once eager, then once with `--compile`. | Real pilot data, optimizer/model parity, exact loader/RNG recovery and aggregate throughput are recorded in both modes. | Any rank failure, changed update geometry, or distributed parity failure. A compiled-only failure blocks the compiled recipe, not the allocation: fall back to eager and record the cost. |
| Throughput confirmation | 1 declared GPU | Run the ten bounded configurations in [`experiments/qualification/throughput-gh200.json`](../experiments/qualification/throughput-gh200.json) with `scripts.benchmark`, reusing one persistent Inductor cache. | Microbatch, activation checkpointing and determinism are selected from measurement, and tokens per second per allocated GPU is recorded. | Clock drift beyond 5%, unstable step times, or a graph-break count that differs from the Ampere result without explanation. |
| Scheduler canary | Site allocation | Bind one finite Slurm wave with current code and real input hashes. Exercise timeout-boundary checkpoint/requeue once. | The scheduler returns the job identity, the trainer checkpoints at the requested signal, resumes the last complete checkpoint, and accounting reports all allocated GPUs. | Missing account/partition, unsupported signal/requeue behavior, duplicate submission, or accounting mismatch. |
| Qualification closeout | 0 training hours | Download receipts, logs, manifests, failed artifacts and scheduler accounting. Reconcile the allocation ledger. | A signed local closeout identifies the usable worker count, measured throughput, recovery status, costs and remaining limits. | Missing receipt, unaccounted GPU-hours, or any unresolved phase result. |

The one-worker real-data command is intentionally separate from the synthetic R0 command. The
former checks the actual loader, export and assistant path; the latter isolates model/runtime and
distributed behavior with a small deterministic workload. Neither result establishes model quality,
long-context capability, sustained production throughput, or data eligibility.

For the real-data phase, use the bundled checkout and run:

```bash
uv sync --locked --python 3.10 --extra gpu --extra linear --group dev --group transformers
uv run --no-sync python -m scripts.gh200_check bind /workspace/gh200-transfer
uv run --no-sync python -m scripts.gh200_check run /workspace/gh200-transfer \
  --output /workspace/gh200-results/one-worker \
  --seconds 5400
```

The command requires exactly one visible GPU. If the provider cannot expose one GPU separately,
pass `--allow-other-gpu` only for an explicitly recorded engineering exception and charge every
allocated idle GPU in the allocation ledger. That exception does not qualify single-GPU cost.

For the distributed production replay, bind the current pilot configuration and run a finite
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

This is a runtime and recovery measurement. It is not a production continuation, a throughput
guarantee, or permission to extend the base horizon. The second invocation is the only place the
selected compiled recipe meets four-worker DDP before production; allow the longer deadline for
max-autotune warmup and export a persistent `TORCHINDUCTOR_CACHE_DIR` first.

## Accounting and receipts

Keep two explicitly linked records:

1. the R0 reservation ledger, whose 70-GPU-hour ceiling protects the synthetic diagnostic;
2. the program qualification ledger, whose 120-GPU-hour reservation covers all site setup,
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

The throughput phase exists because the frozen pilot recipe is inefficient. A bounded RTX 3090 pass
measured 2.084x against it **on a 318M proxy** and selected a configuration; see
[training](training.md#throughput-settings). That number is not the flagship's: the only 1.2B point
in the sweep is the eager checkpointed baseline at 34.6% utilization, against the proxy baseline's
25.0%, so the flagship's headroom to the same ceiling is 1.51x. This phase benchmarks
`experiments/pilot`, the 1.2B reference, so it measures the real figure. Quote no flagship speedup
before it returns.

Microbatch, activation checkpointing and determinism are immutable on resume, so they must be
frozen here, before any production stage starts. The measured rate replaces
`h100_full_trainer_tokens_per_second` as the anchor for every horizon in the numeric plan, which is
why this phase precedes costing — but apply `compute.throughput_reanchoring_rule.overhead_derate`
first, because this sweep runs `--mode compute` and the anchor it replaces is a full-trainer rate
that includes startup, validation and saves. The surplus and shortfall directions are predeclared
in the same rule.

One further gap this phase cannot close: it is single-GPU, so it does not qualify the compiled path
under DistributedDataParallel. `base.py` compiles the DDP-wrapped module and six graph breaks
remain, which is where DDPOptimizer bucketing can interact badly. Run the four-worker production
replay with `--compile` before any compiled production wave.

A successful closeout permits us to cost the 600-hour staged mid-training production reservation
and the 1,800-hour 4K base reservation using measured runtime inputs. It does not admit data, prove
16K/32K usefulness, qualify compiled distributed execution, or authorize Slurm production waves.
Those decisions require their own frozen packets and receipts.

The closeout must state separately:

- one-worker ARM64/software status;
- four-worker communication and restart status, eager and compiled separately;
- the measured flagship speedup over the frozen pilot recipe, stated as a flagship number and not
  inherited from the proxy sweep;
- scheduler/requeue status;
- sustained tokens/s with checkpoint and validation overhead;
- memory and storage observations;
- inference/export parity status;
- all-in allocated GPU-hours and non-GPU costs;
- unresolved limitations and the exact next qualification action.
