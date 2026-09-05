# 29 — Paper 1 baseline hardware preflight failure

## Question

Do the matched dense-global and five-cache KDA/GQA baseline arms pass the full-size compiled training,
memory, cached-decode, and Transformers-export path on the named RTX 3090 before training begins?

## Contract

Both random-weight 153.96M-parameter arms used their materialized batch-4, 4,096-token configuration,
Torch cross entropy, compiled model, compiled Muon update, and the frozen CUDA 12.8 environment. Each
arm had to start at or below 50C. The preflight makes no quality or throughput claim: it executes one
full-size microbatch and optimizer update, short native cached decoding, and a temporary Transformers
export whose files are hashed before removal.

The first attempt stopped before the KDA arm when the GPU reached 52C. The runner was then changed to
wait for the existing limit rather than relax it. A later KDA export check exposed the same decode
problem recorded by the final instrumented run. Failed attempts were treated as diagnostic evidence,
not silently converted into passes.

## Compiled training result

Both arms completed a finite forward, backward, clipped-gradient, and Muon optimizer step at the exact
shape. Both fit the 16GiB peak-allocation envelope.

| Arm | Random loss | Gradient norm | Peak allocated | Envelope |
| --- | ---: | ---: | ---: | --- |
| Dense global | 10.40982 | 0.79477 | 10.23GiB | pass |
| Five-cache KDA/GQA | 10.41088 | 2.10649 | 14.14GiB | pass |

The one-shot compile-and-step durations are intentionally not interpreted as throughput. In
particular, KDA's lower analytic FLOP count did not imply lower peak training memory in this path.

## Decode and export result

The temporary CPU Transformers exports loaded all weights, reproduced native full logits, completed
two greedy generation tokens, and matched their CPU incremental reference exactly for both arms.
Those checks verify export structure, not the target CUDA decode path.

The native CUDA comparison used eight tokens and 256,000 logits per arm. It compared one full call
with eight cached one-token calls under the existing `rtol=0.02`, `atol=0.02` criterion.

| Arm | Max abs error | RMS error | Relative RMS | Failing logits | Token argmax agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense global | 0.02344 | 0.004752 | 0.01811 | 1 / 256,000 | 1.00 |
| Five-cache KDA/GQA | 0.05469 | 0.010826 | 0.03861 | 8,180 / 256,000 | 0.75 |

Dense attention misses the elementwise criterion by one tail logit without changing an argmax. KDA
shows a materially broader discrepancy and changes the selected token at two of eight positions. The
isolated KDA kernel test previously measured small chunk-versus-recurrent operator error, but this
full-depth result demonstrates that operator qualification did not bound composed-model decode drift.

## Decision

**Preflight failed.** The tolerance is not widened after observing the result, and baseline training
does not launch. The next valid experiment is a layerwise CUDA decode diagnostic that separates dense
attention numerical order from KDA chunk/fused-recurrent accumulation, convolution state, recurrent
state, and depth amplification. It must test multiple seeds and lengths and connect hidden-state error
to token decisions before any correction or revised numerical contract is proposed.

No architecture quality, efficiency, or promotion claim follows from this preflight.

## Artifact

- [Machine-readable preflight](../results/Speck-Paper1/baseline-preflight.json)
