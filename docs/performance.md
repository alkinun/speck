# Performance and optimization plan

This is the working performance document for the fixed 1.2B KDA/GQA reference model. It keeps
measured facts, hypotheses, and decisions separate so an H100 or GH200 run can change a number
without silently changing the training plan. The runnable packets are
[`throughput-h100.json`](../experiments/qualification/throughput-h100.json) and
[`throughput-gh200.json`](../experiments/qualification/throughput-gh200.json).

## What is known before the next rental

The only complete flagship throughput evidence is the eager, activation-checkpointed 1.2B point on
an RTX 3090: 3,274 tokens/s at 34.6% model FLOPs utilization. The H100 engineering pilot reached
13,595 steady optimizer tokens/s at 9.8% utilization, but it used a different measurement path:
eager execution, checkpointing, and microbatch one. These rates are useful anchors, not a direct
speedup comparison.

The 3090 optimization sweep used a 318M geometry-matched proxy. Its best deterministic point was
17,906 tokens/s and 52.2% proxy utilization, or 2.084x over its checkpointed eager baseline. That
does not transfer as a flagship speedup. The 1.2B baseline starts at 34.6%, so reaching the proxy
ceiling would be about 1.51x from the flagship baseline.

The proxy profile explains where measurement effort should go:

| Profile category | Share of device time | Interpretation |
| --- | ---: | --- |
| GEMM | 55.5% | The larger flagship should benefit from better Hopper matrix occupancy. |
| KDA mixer | 21.3% | The pinned FLA kernels are a real ceiling and need a Hopper profile before custom work is funded. |
| Launch stalls | about 11% | Six graph breaks and eager KDA islands limit graph capture; CUDA graphs were 0.7% slower on the proxy. |
| Copy/pointwise | 15.6% combined | Weight casts, normalization, gates, and layout traffic are the first fusion targets after the H100 trace. |

End-to-end loading was 0.6% faster than synthetic compute on the proxy, so the loader is not the
first optimization target. The current H100 packet is therefore a compute ladder plus one explicit
end-to-end check, followed by a profile of the selected geometry.

## Measurement definitions

The benchmark reports useful optimizer-step throughput. `--mode compute` uses device-resident
synthetic batches and excludes startup, validation, checkpoint saves, and data I/O. `--mode
end-to-end` reads the packed manifest. The production planning rate is the compute rate multiplied
by the measured all-in overhead derate; the current pilot derate is 0.9459 and must be remeasured at
the destination save and validation cadence.

The reported model FLOPs estimate is `6 * linear_estimate + attention_or_recurrent_estimate` per
token. It includes the vocabulary projection and excludes optimizer work and activation
recomputation. This is the same convention used by the existing receipts and is suitable for
within-model comparisons. It is not a hardware counter and should not be compared with a vendor
TFLOPS figure without naming the convention. Checkpointing adds recompute work to wall time, so
checkpointed and non-checkpointed MFU values must carry their setting in every table.

Every benchmark receipt now includes per-step standard deviation and coefficient of variation,
step-rate p10/p50/p90, and the FLOP-accounting convention. Use the median for ranking, the aggregate
rate for horizon arithmetic, and the between-step spread as a noise diagnostic. A single short run
is a smoke check; a decision needs repeated baseline runs and a stable window.

## H100 rental protocol

Before the metered ladder:

1. Build from a clean commit, record the full commit hash, install versions, H100 SXM model and
   memory, and pin `TORCHINDUCTOR_CACHE_DIR` on persistent storage.
2. Run the frozen pilot baseline five times. Keep all receipts and use their median and spread as
   the noise band. Do not call a later delta real when it is inside that band.
   Summarize the repeated receipts with the repository helper:
   `uv run --no-sync python -m scripts.throughput_summary results/throughput-h100/h100-pilot-baseline-r*.json
   --output results/throughput-h100/baseline-summary.json`. It rejects mixed geometry or runtime
   settings before calculating the band.
3. Use at least ten warmup steps and a 30-step measured window for each selected configuration.
   The packet is already bound to this window; do not shorten it after the baseline noise band is
   known.
4. Run the ladder in this order: checkpointed eager baseline, checkpointing-off eager,
   checkpointing-off compiled, constant-global-batch microbatch 2/4/8, selected end-to-end, and
   selected profile. Keep deterministic execution for the decision path; run one non-deterministic
   comparison only to quantify the available ceiling.
5. Capture `nvidia-smi` clocks, temperature, power, peak allocated/reserved memory, graph-break
   count, and the profile trace. Reject or repeat a result if clocks move by more than 5%, the
   measured window is unstable, or the process recompiles during the measured steps.

The H100 result narrows uncertainty and validates the flagship path. It does not freeze the
production microbatch or re-anchor the allocation; those decisions still require the GH200 and
four-worker qualification because device memory, ARM64 kernels, collectives, and scheduler
recovery differ.

## Optimization order

The order below follows expected return, confidence, and risk.

1. **Freeze geometry first.** Choose the largest microbatch that fits with checkpointing off at the
   production sequence length and global batch. This reduces launch and reduction overhead without
   changing the token schedule. Record memory headroom, not only whether the run fits.
2. **Keep compiled Liger as the default candidate.** The proxy measured a large compile gain with
   checkpointing off, while compilation was effectively neutral with checkpointing on. Liger avoids
   materializing the full vocabulary logits in float32. Recheck both facts on Hopper.
3. **Profile the flagship KDA path.** If KDA plus launch stalls remain above roughly 30% of device
   time, prototype a pinned `torch.library` custom op boundary or an FLA upgrade in an isolated
   branch. Require forward, backward, recurrent-state, deterministic, and resume parity before
   counting any speedup.
4. **Reduce cast and layout traffic.** The model intentionally keeps FP32 parameters and performs
   BF16 activations, so the trace should identify repeated weight casts, RMSNorm casts, transpose,
   and copy kernels. Test one change at a time: autocast-compatible linear weights, fused norm/gate
   paths, or a contiguous KDA projection layout. Keep a loss and gradient parity receipt for each.
5. **Measure optimizer cost separately.** Batched Muon already groups compatible matrices and
   compiles its step. Report optimizer wall time and peak memory independently before changing it;
   the proxy showed optimizer work binding memory mainly with checkpointing on, while the
   checkpointing-off path was backward-bound. An optimizer rewrite is justified only if the Hopper
   trace shows a material step-time share or it unlocks a larger microbatch.
6. **Only then test lower-level variants.** Compare FLA chunk sizes, deterministic kernels, CUDA
   graphs, and alternative attention implementations under the same seed, geometry, and measured
   window. Stop after three consecutive candidates improve aggregate throughput by less than 2% or
   violate numerical/restart parity.

## Decisions after measurement

Record these values in the receipt before changing any plan:

- flagship tokens/s for eager checkpointed, eager no-checkpoint, and compiled no-checkpoint paths;
- speedup ratios with the same geometry and the frozen baseline as denominator;
- aggregate and median tokens/s, MFU under the declared FLOP convention, memory peaks, and noise;
- compile warmup time and graph-break count;
- compute-to-end-to-end derate at the production cadence;
- profile category shares and the largest non-useful kernel;
- deterministic cost and whether it is smaller than the microbatch gain.

Apply the predeclared allocation rule afterward: a throughput surplus returns to data research and
supply-bound stages; a shortfall reduces the base horizon while protecting downstream and evaluation
reservations. Do not use a proxy result to extend the horizon or claim architecture superiority.
