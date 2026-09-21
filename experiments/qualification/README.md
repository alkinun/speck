# First hardware qualification

The complete access sequence is in the [GH200 access qualification runbook](../../docs/compute-qualification.md).
This file owns the bounded synthetic R0 diagnostic and its 70-GPU-hour internal ceiling; the
program reserves 120 GPU-hours for all hardware/runtime qualification, including real-data,
distributed and scheduler checks.

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
  --workers 1 --allocated-gpus 1
```

On the actual allocated node, after checking environment, storage, and prior costs:

```bash
uv run --no-sync python -m scripts.r0_execute experiments/qualification/plan.json \
  --workers 1 --allocated-gpus 1 --run \
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
Those need separate checks before a new corpus run; the later H100 pilot is already complete. No 32K/128K sweep is part of this first step.

## Local preparation result

The [retained RTX 3090 result](local-result.json) records a successful full-size, one-worker
optimization and fresh-process model/optimizer/RNG comparison at the original tolerance. It used
`PYTORCH_ALLOC_CONF=expandable_segments:True`, deterministic algorithms, and no compilation.
The earlier OOM, nondeterministic-gradient failure, and cold-backend RNG failure are preserved.
Synthetic step throughput is about 1,937 tokens/s on that machine; it is not corpus throughput or
a GH200 projection. The four supervised attempts consumed 0.585 local allocated GPU-hours;
standalone kernel/tests are outside that ledger. Repeat qualification on the actual allocation.

## H100 rental result

The [single-H100 receipt](h100-result.json) records passing production-data base and assistant
fresh-process recovery, numerical/kernel checks, native CUDA generation, and CPU export parity.
The 131,072-token batch also completed a four-step timing probe with 32 accumulated microbatches.
Its approximately 13.7K tokens/s is a short one-worker measurement, not sustained throughput or a
GH200/distributed projection. The receipt preserves failures, corrected verification, source commits,
all-in cost boundaries, local evidence hashes, and the latest portable bundle. Use the
[rental runbook](../../docs/gh200.md) for execution; actual allocation qualification remains open.

## Measured planning inputs

The [longer H100 receipt](timing-result.json) supersedes the short probe for detailed component
timing. The later [completed pilot](../pilot/h100-run.json) supplies actual full-trainer and
development evaluation totals; the estimates below remain historical planning evidence.
[timing.json](timing.json) keeps the pilot's 800-step learning-rate schedule but stops after 32 steps,
resumes to 48, and increases validation/save frequency for measurement. The
[microbatch comparison](microbatch-timing.json), [SFT probe](sft-timing.json), and
[evaluation sample](evaluation-timing.json) run sequentially under the same 40-minute execution
deadline. Full resolved configurations, commands, supervisors, raw timings and failures are in the
receipt's external evidence archive. `scripts.training_timing` observes the production trainer;
it does not replace a deadline supervisor or cumulative GPU-hour ledger.

| Measurement on one H100 SXM | Result |
| --- | --- |
| Base, batch 131,072, microbatch 1, 4K | 13,615 tokens/s; 28 steady steps; 19.3 GiB peak allocated |
| Same token batch, microbatch 4 | 16,398 tokens/s; six steady steps; 19.3 GiB peak allocated |
| Full 786,432-token validation | 15.9 seconds initially, 14.2 seconds warm |
| Durable model + optimizer checkpoint | 10.7–11.0 seconds; 9.16 GiB |
| Fresh-process restart overhead | 16.9 seconds beyond optimizer/validation/save work; state restore itself 4.0 seconds |
| SFT at fixed 4K | 13,493 padded positions/s; about 3,110 supervised tokens/s at this sample's density |
| Native cached decode, 1K prefix + 256 tokens | 55 tokens/s at batch 1; 428 aggregate tokens/s at batch 8 |

The base run completed 6,291,456 tokens; held-out loss fell from 10.78 to 7.11. This is a short
engineering learning check, not the completed pilot or evidence of useful capabilities. The
microbatch-four comparison is not yet restart-qualified. The SFT probe repeats 64 conversations
four times deliberately: 1,048,576 computed positions contain 241,652 supervised tokens (23%).
Its roughly 5.4 optimizer minutes per million supervised tokens is specific to that padding/masking
density. Final SFT cost requires actual corpus counts, length buckets, epochs and checkpoint cadence.

For the unchanged microbatch-one pilot, `800 × 9.627 seconds`, nine validation passes, eight saves
and measured process/startup overhead give **2.20 GPU-hours**, about **$7.69** at the screenshot's
$3.49/GPU-hour. Eight retained checkpoints need **73.25 GiB** before data, environments and exports.
An interruption can lose up to about **16 minutes** between 100-step checkpoints, plus restart work.
These are projections using warm caches; add cold setup, transfers, failures and idle time.

The corrected pinned HFLM backend completed 70 development requests: two each for GSM8K, IFEval
and HumanEval+, and 32 each for ARC-Challenge and HellaSwag. Full-cap samples took about 22.2 seconds
for 1,024-token GSM8K responses, 20.9 seconds for 1,024-token code responses, and 10.5 seconds for
512-token IFEval responses. Four-choice likelihood requests averaged 0.165 and 0.159 seconds/task.
The full-budget scenario is **2.15 hours for development / 9.01 hours for final per checkpoint**.
Observed early stopping instead gives 2.02 / 8.49 hours. Neither is a measured full-suite runtime or
a guaranteed bound: samples are small and source-order biased; model output lengths and final
prompt lengths can differ. Grading, setup and transfer are additional. The code grader alone allows
up to 8.25 / 32.75 minutes of execution time for development/final, excluding sandbox overhead.
All development prompts/options fit 4K including a BOS and the output budget; no final task contents
were selected. No capability scores or generated-code execution were performed on this root-run pod.

This run found evaluator failures that tiny generation checks missed: likelihood scoring inherited
last-token-only cache defaults, EOS decoded to an empty stop string, and base generation could emit
unassigned assistant rows. Fixed exports pass parity and the full timing sample. Old evidence is
preserved; re-export checkpoints before using the corrected evaluator. The new transfer archive pins
the fixes at `533383e`. Portable checks pass **700 tests**. Supervised phases, including failures,
used **0.350 allocated GPU-hours**; the observed setup/fix/idle window used **0.662 hours**. This is
not a provider bill and excludes earlier/later rental time and non-GPU charges.

The timing study’s historical 3,991-unassigned-hour scenario implied 195.6B compute-only tokens
under ideal per-GPU scaling and zero additional overhead/post-training cost. Its then-current
mixture stock was capped near 2.76B before exclusions. These historical figures do not describe
the revised allocation or supply; use the [current plan](../../PLAN.md#compute) for both.

The current program reserves 120 hours for all hardware/runtime qualification. This frozen
rehearsal retains its original 70-hour ceiling; it is a bounded subset, not the entire new phase.
New probes need their own identified settings and must fit the remaining program reservation.
