# 30 — Full-depth CUDA decode failure classification

## Question

Is the Paper 1 baseline preflight failure a cache-state correctness bug, an FLA KDA kernel defect, a
causal-convolution history defect, or a broader full-versus-incremental BF16 execution effect—and does
it alter trained greedy generation?

## Frozen random-weight matrix

The matrix was frozen after disclosing the seed-42 length-8 pilot. It uses exact Paper 1 baseline
architectures, seeds 42/43/44, prefix-matched lengths 8/64/512, and the existing `rtol=0.02`,
`atol=0.02` threshold. It contains:

- native CUDA dense and KDA paths in all 18 architecture/seed/length cells;
- KDA with the auditable Torch recurrence in nine cells;
- Torch KDA with every delayed convolution tap removed in nine cells; and
- length-8 CPU sentinels for both architectures.

Every linear projection, normalization, branch output, and residual stage is captured at the final
prefix position. All-position logits retain RMS, scale, elementwise tolerance, and token-decision
metrics.

## Device and state classification

The CPU sentinels pass:

| Arm | Relative RMS | Max absolute error | Argmax agreement |
| --- | ---: | ---: | ---: |
| Dense global | 1.41e-6 | 2.15e-6 | 1.00 |
| Five-cache KDA/GQA | 4.32e-6 | 6.21e-6 | 1.00 |

This rejects a device-independent cache/state-semantics bug under the tested prefix. On CUDA, both
architectures fail every random-weight cell, so the old elementwise full-model threshold is not a
KDA-specific correctness test.

| Arm | Tokens | Median relative RMS | Range | Median argmax agreement |
| --- | ---: | ---: | ---: | ---: |
| Dense global | 8 | 0.02016 | 0.02003–0.02023 | 0.8750 |
| Dense global | 64 | 0.02031 | 0.01942–0.02035 | 0.9531 |
| Dense global | 512 | 0.01749 | 0.01684–0.01810 | 0.9355 |
| Five-cache KDA/GQA | 8 | 0.04511 | 0.04421–0.04559 | 0.8750 |
| Five-cache KDA/GQA | 64 | 0.06270 | 0.06040–0.06289 | 0.8906 |
| Five-cache KDA/GQA | 512 | 0.04858 | 0.04746–0.04901 | 0.8848 |

KDA's median relative RMS is `2.24×`, `3.09×`, and `2.78×` dense at 8, 64, and 512 tokens. Its
median fraction of logits outside tolerance is 6.18%, 16.56%, and 8.01%; dense is 0.0063%, 0.0070%,
and 0.00067%.

## Layerwise mechanism

At lengths 8 and 64, the shared input adapter already differs by roughly 0.25–0.30% relative RMS.
This is before any attention or recurrence and is consistent with shape-dependent BF16 GEMM
accumulation for full-sequence versus one-token calls. Dense amplifies this seed about `6.8×–8.0×` by
the logits. KDA amplifies it about `15.5×–23.5×`.

At length 512 the final-token adapter often matches exactly, but earlier-token K/V or KDA projection
differences enter the attention/recurrent state. The first hard failures then appear in the early
layer-1/2 normalization path. Thus final-position adapter equality does not imply equal prefix state.

The first elementwise failures usually occur in an early FFN or next-block normalization, not inside
the first projection itself. Small projection/branch differences are repeatedly renormalized and
amplified through FFNs, residual depth, and global/recurrent integration.

## Backend and convolution ablations

Replacing FLA chunk/fused-recurrent KDA with the auditable Torch recurrence retains 83–102% of native
KDA relative RMS and preserves token disagreements in the declared falsification rule for all nine
cells. FLA is therefore not the sole cause, though it contributes at some lengths.

Removing every delayed causal-convolution tap retains 83–97% of Torch-KDA relative RMS and also
preserves the declared failure signal in all nine cells. Convolution history is not the sole cause.

## Trained checkpoint sentinels

Four immutable 131M-token checkpoints use the first packed validation prefix: one historical dense
seed and KDA seeds 42/43/44. Each tests teacher-forced prefixes and 32-token common-history and
free-running greedy continuations.

| Arm | Seed coverage | Prompt | Relative RMS | Teacher-forced argmax | Divergent 32-token runs |
| --- | --- | ---: | ---: | ---: | ---: |
| Dense global | 42 | 8 | 0.00604 | 1.0000 | 0/1 |
| Dense global | 42 | 64 | 0.00544 | 0.9844 | 0/1 |
| Dense global | 42 | 512 | 0.00485 | 0.9961 | 0/1 |
| Five-cache KDA/GQA | 42/43/44 | 8 | 0.00595 median | 1.0000 median | 2/3 |
| Five-cache KDA/GQA | 42/43/44 | 64 | 0.00481 median | 1.0000 median | 0/3 |
| Five-cache KDA/GQA | 42/43/44 | 512 | 0.00451 median | 0.9961 median | 1/3 |

Training reduces relative RMS substantially and makes teacher-forced token decisions much more stable.
The strict elementwise threshold nevertheless fails for every trained cell, including dense, with max
absolute differences from 0.10 to 0.25. It is therefore not aligned with full-model BF16 behavioral
equivalence.

The trained result is not harmless everywhere. KDA seed 43 diverges in both common-history and
free-running generation at step 6 from the 8-token prompt and step 12 from the 512-token prompt. Seed
44 diverges at step 24 from the 8-token prompt. KDA matches completely in the other six cells; the
dense sentinel matches in all three. One validation prefix is not a population estimate, but these
counterexamples prevent simply waiving the preflight.

## Decision

The failure is classified as **full-model CUDA/BF16 execution sensitivity with architecture-dependent
depth amplification**. It is not a general cache-state bug, not solely FLA recurrence, and not solely
convolution history.

The preflight remains failed and baseline training remains blocked. The observed full-model
elementwise threshold must not be silently relaxed. A new versioned cache-equivalence contract needs
representative prompts and must separate:

- strict operator/state parity;
- distribution error and top-k overlap on a common history;
- decision disagreement conditioned on the full-path logit margin;
- free-running divergence horizon; and
- candidate non-inferiority relative to the conventional dense CUDA control.

The random-weight matrix has diagnostic authority and the trained runs are operational sentinels.
Neither replaces matched Paper 1 baseline evidence or promotes an architecture.

## Artifacts

- [Random-weight diagnostic contract](../research/paper-1/cuda_decode_diagnostic.json)
- [Random-weight diagnostic result](../results/Speck-Paper1/cuda-decode-diagnostic.json)
- [Trained sentinel contract](../research/paper-1/cuda_decode_trained_sentinel.json)
- [Trained sentinel result](../results/Speck-Paper1/cuda-decode-trained-sentinel.json)
