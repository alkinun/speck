# GH200 qualification

The runbook for the runtime-qualification line of the [budget](program.md#compute). It qualifies
the hardware, software stack, distributed runtime and recovery path, and measures throughput for the
parent and every ladder rung. It trains no model and selects no data. Every phase is charged once to
the allocation ledger.

## Entry gate

On a clean commit, with `TMPDIR` on real disk (the distributed and SQLite tests fill a small tmpfs
and report it as a test failure):

```bash
make quality
make plan-check
```

Build a private bundle from the clean tree:

```bash
uv run --no-sync python -m scripts.gh200_check bundle --output BUNDLE_DIR \
  --data PILOT_PACK_DIR --tokenizer TOKENIZER_DIR --assistant ASSISTANT_REHEARSAL_DIR
```

The bundle carries committed source and history, the frozen tokenizer, the 105M-token pilot pack
and a small assistant rehearsal. It carries no virtual environment, credentials or checkpoints.
Transfer the archive and its receipt, verify the SHA-256 after transfer, and extract on persistent
storage with about 200 GB free.

## Sequence

Run the phases in order. A failed phase stops the sequence; keep its output and record the failure
before any retry. Every phase records device, source commit, input hashes, wall time and allocated
GPU-hours.

| Phase | GPUs | Action | Pass condition |
| --- | ---: | --- | --- |
| Site preflight | 0 | Record instance, architecture, driver/CUDA, storage and scheduler; verify the bundle; install the ARM64 environment | ARM64 GH200, dependencies installed, storage confirmed |
| One-worker check | 1 | `scripts.gh200_check run` | Loader replay, kernels, base restart, generation, export parity, SFT restart and export all pass |
| Four-worker replay | 4 | `scripts.training_replay`, eager then `--compile` | Model, optimizer, loader and RNG parity after restart in both modes |
| Throughput | 1 | The runs in [throughput-gh200.json](../experiments/qualification/throughput-gh200.json), plus per-rung runs | Microbatch, activation checkpointing and determinism selected; tokens/s recorded for the parent and every rung |
| Scheduler canary | site | One finite [Slurm](slurm.md) wave with a timeout-boundary checkpoint and requeue | Checkpoint at the signal, resume, accounting matches |
| Closeout | 0 | Download receipts, logs and accounting; reconcile the ledger | Every GPU-hour accounted for |

### One-worker check

```bash
cd /workspace/gh200-transfer
speck_bundle_branch=$(python3 -c 'import json; print(json.load(open("bundle.json"))["branch"])')
git clone --branch "$speck_bundle_branch" code.bundle code
cd code
uv sync --locked --python 3.10 --extra gpu --extra linear --group dev --group transformers
uv run --no-sync python -m scripts.gh200_check bind /workspace/gh200-transfer
uv run --no-sync python -m scripts.gh200_check run /workspace/gh200-transfer \
  --output /workspace/gh200-results/one-worker --seconds 5400
```

`run` verifies the clean commit and every input hash, then runs five steps under one 90-minute
deadline: reopen the pilot pack and replay loader batches in a fresh process; check KDA numerics;
train four full-size 4K base steps and restart from step two, comparing every tensor and the loader
and RNG state; check cached generation and export parity; initialize SFT from that checkpoint and
repeat the restart and export checks. It requires exactly one visible GPU. The lock pins ARM64
CUDA 12.8 PyTorch 2.9.1 and Triton 3.5.1 wheels, which still need to be confirmed on the site image.

### Four-worker replay

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

The compiled run is the only place the compiled recipe meets four-worker DDP before real runs.
Export a persistent `TORCHINDUCTOR_CACHE_DIR` first and keep its `triton/` subdirectory, which holds
FLA kernel choices. A compiled-only failure blocks the compiled recipe, not the allocation: fall
back to eager and record the cost.

### Throughput

Print the packet's commands rather than retyping them, and run them from `code/` after binding:

```bash
uv run --no-sync python experiments/qualification/check_throughput_packet.py \
  experiments/qualification/throughput-gh200.json --print-commands
```

Run the baseline five times first; a difference smaller than that spread is not a result.
`scripts.throughput_summary` checks the repeats and writes the noise summary. Substitute selected
microbatch values in the printed commands, not in the tracked packet. Stop on clock drift beyond
5%, unstable step times, or an unexpected out-of-memory.

Microbatch, activation checkpointing and determinism are immutable on resume, so they are frozen
here, before any ladder or parent run. The measured rate replaces
`h100_full_trainer_tokens_per_second_1_2b` in [plan.json](../experiments/main-data/plan.json) after
applying `compute.measured_anchor.overhead_derate`, since this sweep excludes startup, validation
and saves. Per-rung rates convert the ladder line into run counts. `compute.rules` predeclares what
a faster or slower rate changes.

## Throughput evidence so far

- The [RTX 3090 sweep](../experiments/qualification/throughput-3090/sweep.json) measured 2.084x on
  a 318M proxy (checkpointing off, compile with max-autotune, larger microbatch). Its endpoint also
  changed tokens per update, and the optimized 1.2B model did not fit the card, so the parent's
  speedup is unmeasured.
- The [H100 pilot](../experiments/pilot/h100-run.json) ran eager and checkpointed at about 10%
  estimated MFU. That is an upper bound on cost, not an estimate.
- Compiled restart parity holds for [base](../experiments/qualification/compiled-recovery-descent-3090.json)
  and [SFT](../experiments/qualification/compiled-sft-recovery-3090.json) once Inductor's
  coordinate-descent tuning is excluded. `COMPILE_OPTIONS` in `speck/operations/runtime.py` excludes
  it; it re-timed kernels per process and broke bitwise restart.

MFU = tokens/s × model FLOPs/token ÷ dense BF16 peak. Keep compute, loader-inclusive
(`--mode end-to-end`) and full-trainer rates separate. Only a sustained trainer run at the real
checkpoint cadence measures the full-trainer overhead.

## Closeout

State separately: one-worker ARM64 status; four-worker restart status, eager and compiled; the
parent's measured speedup over the pilot recipe; tokens/s per GPU for each rung and the resulting
run counts; scheduler and requeue status; memory and storage observations; export parity; all-in
GPU-hours and non-GPU costs; open limitations.

Lessons from the H100 pilot:

- A trainer stopping does not stop billing or delete an instance.
- Verify backups by hash before deleting remote checkpoints.
- Never reuse or reset a budget ledger to unblock an attempt.
- Some hosts deny the user namespaces the code grader needs. `--defer-code-grading` keeps
  generation on the GPU host and `scripts.code_grade` finishes grading locally.
