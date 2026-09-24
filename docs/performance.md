# Performance and optimization plan

The fixed 1.2B KDA/GQA reference is the [program's parent](program.md#model); its costs are
revised only from a measured speedup. [GH200 qualification](compute-qualification.md) owns grant
hardware, per-rung rates and distributed checks; the optional [H100 rental](throughput-rental.md)
owns its own procedure; the JSON packets own run settings. This document owns the throughput
evidence, measurement definitions and optimization priorities.

## Evidence and limits

The [3090 sweep](../experiments/qualification/throughput-3090/sweep.json) measured 2.084x on a
318M proxy, from 8,592 to 17,906 tokens/s. Its endpoint also changed tokens per update from 32,768
to 36,864, so the headline includes a change in optimizer amortization as well as implementation
and microbatch changes. The only 1.2B point in that sweep was eager and checkpointed: 3,274 tokens/s.
The optimized 1.2B model did not fit the 24 GiB card. The proxy's 52.2% estimated MFU establishes
neither a 1.2B ceiling nor a transferable speedup.

The [H100 pilot](../experiments/pilot/h100-run.json) measured 13,595 steady optimizer tokens/s and
12,859 full-trainer tokens/s. MFU uses the exact model estimate, 7.513 GFLOP/token, while
[planning](program.md#the-ladder) uses 6 × parameters; with the packet's 989.5 TFLOP/s denominator
these rates correspond to about 10.3% and 9.8% estimated MFU respectively. A lower
MFU on a different GPU does not identify the bottleneck: launch latency, memory traffic, kernel
selection, optimizer amortization and different timing boundaries must be measured separately.

Historical profiles suggest investigating GEMMs, KDA, casts and launches. Their category shares
are historical diagnostics: the old summarizer could include CPU custom-autograd events alongside
device kernels. The current `kernel_summary` filters to CUDA events before classifying names. Its
denominator sums kernel durations, not wall time, and launch markers remain separate. Inspect the
trace for overlap, idle time and classification errors before attributing a bottleneck. Historical
receipts remain unchanged.

The [local compiled recovery screen](../experiments/qualification/compiled-recovery-3090.json)
failed on the 318M proxy: both full-trainer processes completed, but 286 of 321 saved model
tensors exceeded the existing restart tolerance. Loader/RNG state and counters matched exactly.
One cause is a cache change between eager FLA warmup and lazy Inductor initialization: identical
KDA inputs selected different cached kernels. Runtime setup fixes the Triton cache before
either path starts, but the [follow-up screen](../experiments/qualification/compiled-recovery-cache-3090.json)
still failed. The [isolation receipt](../experiments/qualification/compiled-recovery-descent-3090.json)
locates the remaining cause in Inductor's coordinate-descent tuning, which re-times reduction
configurations in every process. With fixed weights and batch, each process was bitwise
repeatable, but two fresh processes disagreed. Eager, plain compile and `max_autotune` alone agreed
bitwise across processes, even from independent cold Inductor caches. Coordinate descent alone
did not. Every compiled training path uses the shared `COMPILE_OPTIONS`, which exclude it. The
training replay then passed at the unchanged tolerance, including optimizer state. Paired proxy
repeats measured the change at about 0.5% slower. The sweep's 2.084x includes coordinate descent,
so it slightly overstates the current recipe. A [compiled SFT replay](../experiments/qualification/compiled-sft-recovery-3090.json)
on the 1.2B reference, with checkpointing on, also passed. Each result is one Ampere worker; Hopper
and four-worker compiled DDP remain unqualified.

## Measurement definitions

| Metric | Numerator and timing boundary | Use |
| --- | --- | --- |
| Compute throughput | Tokens / timed synthetic optimization steps; includes backward, clipping and optimizer | Implementation comparisons |
| Loader-inclusive throughput (`--mode end-to-end`) | Tokens / timed optimization steps with packed loading | Check input delivery at the selected geometry |
| Full-trainer throughput | Tokens / trainer wall time including startup, validation and saves | Cost the training stage at its actual cadence |
| Allocated throughput | Useful tokens / all allocated GPU-seconds, including failures and idle time | Allocation planning after distributed qualification |

The loader-inclusive benchmark **does not measure the full-trainer derate**. The pilot's
full-trainer/steady ratio, 0.9459, is an interim assumption, not a conservative bound. Faster steps
can increase the fraction spent saving and validating. Replace it using a sustained trainer run
with the selected recipe, real checkpoint cadence and an explicit startup/restart boundary. Single-GPU
rates also need measured four-worker scaling before they can cost the allocation.

Estimated MFU = tokens/s × model FLOPs/token / declared dense BF16 peak FLOPs/s. The current model
estimate counts six times linear work plus attention/recurrent work; it includes the vocabulary
projection and excludes optimizer FLOPs, recomputation and many elementwise operations. It is not
a hardware counter. Keep its convention and precision denominator beside every result. At fixed
model and context, higher tokens/s raises estimated MFU; tokens/model-FLOP is fixed by the estimate.
Improving capability per FLOP additionally requires matched learning curves, not a speed probe.

Receipt version 2 adds timing variation and rate quantiles. Compare aggregate tokens/s across
repeated runs; keep within-run variation separate from between-run spread. Neither a low step-time
CV nor a five-run range is a confidence interval or a correctness check. Warmup includes useful
training work and compilation; it is not pure compile time.

## Optimization order

1. **Measure geometry at fixed tokens per update.** Test checkpointing off and the microbatch
   ladder, retaining memory headroom and the fastest stable configuration. Larger batches are not
   automatically faster. Microbatch changes alter loader scheduling and cannot be silently resumed.
2. **Compare compilation at the same geometry.** Keep Liger and deterministic execution as the
   candidates selected by the proxy. Measure both compiled forward/backward and optimizer cost;
   the current compile toggle changes both. The benchmark calls the same loss path as the
   real trainer; preflight checks every run.
3. **Follow the 1.2B trace.** Investigate KDA graph boundaries if host gaps dominate; investigate
   normalization, gates, weight casts and layout copies if those dominate. Batched Muon already
   groups matrices and compiles its update; optimize it only when its measured time or memory
   cost warrants it. Removing unused recurrent final-state outputs is another bounded candidate.
4. **Qualify one change at a time.** Compare outputs and gradients against the reference, then
   deterministic and fresh-process restart behavior. Cached-state changes also need recurrent
   parity. Test the winning candidate on packed data and the full trainer before promotion.

Keep model, context, tokens per update, precision, source revision and dependencies explicit.
Use repeated baseline/candidate measurements; retain inconclusive and failed attempts. Stop a
local search after three gains below the larger of 2% or the observed noise range, then return to
profiling rather than accumulating unmeasured options.

## Result handoff

Record one row per configuration with receipt path, model/context, microbatch/accumulation,
checkpointing, compile settings, median across run rates and range, speedup denominator, estimated
MFU/peak convention, memory peaks and warmup time. Keep loader-inclusive and full-trainer rates
separate. Attach the selected trace and parity/restart results. Mark unmeasured cells as unmeasured.

Apply the [numeric plan](../experiments/main-data/plan.json)'s `compute.rules` only once GH200
costs and distributed overhead are measured. The optional H100 rental supplies implementation
evidence; it does not lengthen the parent run or establish architecture superiority.
