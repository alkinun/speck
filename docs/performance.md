# Performance and optimization plan

The fixed 1.2B KDA/GQA reference needs a measured flagship speedup before we revise its costs.
[The rental runbook](throughput-rental.md) owns the H100 procedure; [GH200 qualification](compute-qualification.md)
owns production hardware and distributed checks. The JSON packets own run settings. This document
owns measurement definitions and the optimization priorities, not another copy of those procedures.

## Evidence and limits

The [3090 sweep](../experiments/qualification/throughput-3090/sweep.json) measured 2.084x on a
318M proxy, from 8,592 to 17,906 tokens/s. Its endpoint also changed tokens per update from 32,768
to 36,864, so the headline includes a change in optimizer amortization as well as implementation
and microbatch changes. The only 1.2B point in that sweep was eager and checkpointed: 3,274 tokens/s.
The optimized flagship did not fit the 24 GiB card. The proxy's 52.2% estimated MFU establishes
neither a flagship ceiling nor a transferable speedup.

The [H100 pilot](../experiments/pilot/h100-run.json) measured 13,595 steady optimizer tokens/s and
12,859 full-trainer tokens/s. With the current 7.513 GFLOP/token estimate and the packet's 989.5
TFLOP/s denominator, these correspond to about 10.3% and 9.8% estimated MFU respectively. A lower
MFU on a different GPU does not identify the bottleneck: launch latency, memory traffic, kernel
selection, optimizer amortization and different timing boundaries must be measured separately.

Historical profiles suggest investigating GEMMs, KDA, casts and launches. Their category shares
are historical diagnostics: the old summarizer could include CPU custom-autograd events alongside
device kernels. The current `kernel_summary` filters to CUDA events before classifying names. Its
denominator sums kernel durations, not wall time, and launch markers remain separate. Inspect the
trace for overlap, idle time and classification errors before attributing a bottleneck. Historical
receipts remain unchanged.

The [local compiled recovery screen](../experiments/qualification/compiled-recovery-3090.json)
failed on the 318M proxy: both production-trainer processes completed, but 286 of 321 saved model
tensors exceeded the existing restart tolerance. Loader/RNG state and counters matched exactly.
The cause is unresolved. Compiled typed-output execution succeeds, but that does not qualify
recovery. Keep compiled throughput runs exploratory; isolate model and optimizer compilation from
the same checkpoint before promoting the recipe. This is an Ampere finding, not a Hopper result.

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
with the selected recipe, production cadence and an explicit startup/restart boundary. Single-GPU
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
   the current compile toggle changes both. Both rental packets require `--training-output` to
   match the production trainer's typed loss diagnostics; preflight checks every run.
3. **Follow the flagship trace.** Investigate KDA graph boundaries if host gaps dominate; investigate
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

Use the [numeric plan](../experiments/main-data/plan.json)'s existing surplus/shortfall rules only
once GH200 costs and distributed overhead are measured. The H100 rental supplies implementation
evidence; it does not increase the 80B working horizon or establish architecture superiority.
