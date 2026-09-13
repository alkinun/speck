# 11 — KDA implementation and kernel qualification

## Question

Can Speck represent Kimi Delta Attention as a first-class, auditable operation and use the pinned
FLA implementation accurately at the production 150M head geometry?

## Scientific isolation

This implementation isolates KDA's defining intervention: channel-wise recurrent decay. It keeps
Speck's current full-rank output-gate projection instead of simultaneously adopting Kimi's
low-rank output gate. The activation is sigmoid, matching Kimi. A future GDN-sigmoid versus
KDA-sigmoid experiment will therefore change decay granularity and its required projection, but
not output-gate rank or activation.

KDA is a distinct `kimi_delta_attention` architecture kind rather than a mode hidden inside GDN.
Existing `gated_deltanet` configs, parameter names, state dictionaries, and default SiLU output
gates are unchanged.

## Implementation

The operation contains:

- the same fused Q/K/V/output-gate projection geometry as Speck GDN;
- kernel-size-4 depthwise causal convolution over Q/K/V;
- grouped value heads;
- a rank-`value_head_dim` projection producing one decay per value head and key channel;
- one beta update rate per value head;
- per-head RMSNorm, sigmoid output gate, and output projection;
- a fixed float32 recurrent state of shape
  `batch × value_heads × key_head_dim × value_head_dim`;
- chunkwise FLA execution for multi-token input and fused recurrent FLA execution for one-token
  decode.

The CUDA call supplies already parameterized log decay and already-sigmoided beta. It deliberately
does not use FLA's `**kwargs` for undocumented in-kernel activation flags. Both called function
signatures are checked at qualification time, and the run refuses any
`flash-linear-attention` version other than the project pin, `0.5.0`.

The Torch reference normalizes Q/K, applies the `1/sqrt(head_dim)` query scale, expands shared key
heads across grouped value heads, applies per-channel decay, and then performs the delta-rule error
correction. A unit test proves that repeating a scalar decay over every channel reduces to the
existing grouped GDN recurrence.

## Static verification

Before the kernel result was frozen:

- architecture grammar and validation passed;
- channel-constant KDA-to-GDN reduction passed;
- full-forward versus cached token decode passed on the Torch reference;
- recurrent state was confirmed independent of context length and separately attributed as
  `kimi_delta_attention` in memory reports;
- CPU backward gradients were finite;
- the paper's chunk-FLOPs formula was tied to a hand-calculated test case;
- the generated Transformers wrapper loaded KDA weights and matched native logits;
- the entire repository suite passed: `294 passed`;
- Ruff and `git diff --check` passed.

Implementation commit: `f544196ca2377af9f8bfb632a28df8171590baad`.

## CUDA qualification

The immutable qualification was run from the clean implementation commit with:

- GPU: NVIDIA GeForce RTX 3090, compute capability 8.6;
- Torch: `2.9.1+cu128`;
- `flash-linear-attention`: `0.5.0`;
- batch: 1;
- key heads: 3;
- value heads: 6;
- head dimension: 64;
- dtype: bf16 Q/K/V/beta, float32 log decay and recurrent state;
- seed: 42;
- 5 warmups and 20 timed repetitions.

### Chunkwise forward

| Length | Output max abs error | Final-state max abs error | Repeat error | Median raw-op time | Raw tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.0009766 | 0.0035033 | 0 | 0.414 ms | 154,415 |
| 512 | 0.0009766 | 0.0051058 | 0 | 0.417 ms | 1,227,684 |
| 4,096 | 0.0009766 | 0.0042887 | 0 | 0.506 ms | 8,093,950 |

These are warmed raw recurrence-kernel measurements, not full-layer or language-model throughput.
The apparent tok/s increase reflects amortized launch overhead and parallel chunk work.

### Backward

| Input | Gradient max abs error |
| --- | ---: |
| Q | 7.45e-9 |
| K | 2.38e-7 |
| V | 9.54e-7 |
| log decay | 1.13e-6 |
| beta | 7.63e-6 |

All gradients were finite. The declared acceptance tolerance was `0.05`; the maximum observed
error was `7.63e-6`.

### Recurrent decode

Sixty-four one-token fused-recurrent steps were compared with one chunkwise length-64 call:

- output max absolute error: `0.0004883`;
- final-state max absolute error: `0.0035726`;
- declared tolerance: `0.02`.

## Result

Qualification passed with no failures. KDA is available, deterministic under this test, consistent
with the auditable recurrence, differentiable, and compatible with Speck's grouped 3/6-head
geometry. This removes kernel feasibility as a blocker to the synthetic and language-model
experiments.

It does not yet prove:

- full 150M compiled-step throughput or peak memory;
- numerical behavior after thousands of optimization steps;
- performance at the paper's head dimension 128;
- quality improvements over GDN;
- parity for variable-length packed sequences or context parallelism.

Subsequent result: KDA passed calibrated 32-query MQAR in all three seeds and robustly beat
sigmoid-gated GDN, while SiLU-gated GDN remained competitive. See
[13 — Synthetic MQAR](13_synthetic_mqar.md).

Machine-readable artifact:
[KDA kernel qualification](../results/KimiLinearTransfer/kda_kernel_qualification.json).
